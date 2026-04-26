"""Funções de nó do LangGraph para o agente RAG.

Cada nó é uma função independente com a assinatura ``(state: AgentState) -> dict``.
Os nós retornam apenas os campos de estado que atualizam — o LangGraph os mescla.

Todas as chamadas de LLM passam por ``get_llm()`` da fábrica existente.
Todas as strings de prompt são importadas de ``prompts.py``.
"""
from __future__ import annotations

import concurrent.futures
import re
from typing import TypeVar

import tiktoken
from pydantic import BaseModel, ValidationError

from src.agent.prompts import (
    FAITHFULNESS_CHECK_PROMPT,
    FAITHFULNESS_CORRECTION_PROMPT,
    GENERATOR_PROMPT,
    MULTIHOP_SUBQUERY_PROMPT,
    QUERY_ANALYZER_PROMPT,
)
from src.agent.query_expansion import QueryExpander
from src.agent.state import AgentState
from src.ai.llm.factory import get_llm
from src.core.config import Constants
from src.core.models import (
    FaithfulnessResult,
    MultiHopSubQuery,
    QueryAnalysis,
    RetrievalResult,
)
from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.utils.logger import LoggingService

logger = LoggingService.setup_logger(__name__)

_tokenizer: tiktoken.Encoding | None = None


def _get_tokenizer() -> tiktoken.Encoding:
    """Carrega o encoder tiktoken de forma preguiçosa (singleton)."""
    global _tokenizer  # noqa: PLW0603
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer


def _count_tokens(text: str) -> int:
    """Conta tokens usando o encoding tiktoken cl100k_base."""
    return len(_get_tokenizer().encode(text))


def _get_llm():
    """Retorna a instância do LLM Anthropic configurada."""
    return get_llm("anthropic", Constants.CLAUDE_MODEL)


_M = TypeVar("_M", bound=BaseModel)


def _parse_model(raw: str, model: type[_M], fallback: _M | None = None) -> _M | None:
    """Remove blocos de markdown, analisa o JSON e valida contra um modelo Pydantic.

    Args:
        raw:      String de resposta bruta do LLM (pode conter blocos de código markdown).
        model:    Classe de modelo Pydantic para validação.
        fallback: Valor a retornar em caso de falha de análise/validação.
                  Se ``None``, retorna ``None`` em caso de falha.

    Returns:
        Uma instância de modelo validada, ou ``fallback`` / ``None`` em caso de falha.
    """
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
    try:
        return model.model_validate_json(cleaned)
    except (ValidationError, ValueError):
        logger.debug("_parse_model: falha ao validar %s. Raw: %s", model.__name__, raw)
        return fallback


# ── Node 1: query_analyzer ─────────────────────────────────────────────────


def query_analyzer(state: AgentState) -> dict:
    """Classifica a query como simples, comparativa ou multi-hop.

    Chama o Claude com ``QUERY_ANALYZER_PROMPT`` e analisa a resposta
    em um modelo Pydantic ``QueryAnalysis``.

    Atualiza:
        query_type
    """
    query = state["original_query"]
    logger.info("query_analyzer: analisando '%s'", query[:80])

    llm = _get_llm()
    prompt = QUERY_ANALYZER_PROMPT.format(query=query)

    raw = llm.generate(
        prompt=prompt,
        temperature=0.0,
        max_tokens=256,
    )

    result = _parse_model(raw, QueryAnalysis, fallback=QueryAnalysis())

    if result is None or result.query_type == "simple" and not raw.strip():
        logger.warning(
            "query_analyzer: falha ao parsear resposta — usando 'simple'. Raw: %s",
            raw[:200],
        )
    else:
        logger.info(
            "query_analyzer: tipo='%s' | razão='%s'",
            result.query_type,
            result.reasoning[:120],
        )

    return {"query_type": (result or QueryAnalysis()).query_type}


# ── Node 2: query_expander ─────────────────────────────────────────────────


def query_expander(state: AgentState) -> dict:
    """Gera documento HyDE e reformulações de query.

    Ignorado (retorna apenas a query original) se ``HYDE_ENABLED=false``.

    Atualiza:
        expanded_queries, hyde_document
    """
    query = state["original_query"]
    logger.info("query_expander: expandindo query")

    # Para rodadas multi-hop > 1, gera uma sub-query em vez de expansão completa
    retrieval_round = state.get("retrieval_round", 0)
    if retrieval_round >= 1 and state.get("query_type") == "multi_hop":
        logger.info(
            "query_expander: rodada multi-hop %d — gerando sub-query",
            retrieval_round + 1,
        )
        sub_query = _generate_multihop_subquery(state)
        return {
            "expanded_queries": [sub_query],
            "hyde_document": None,
            "retrieval_round": retrieval_round + 1,
        }

    llm = _get_llm()
    expander = QueryExpander(llm)
    hyde_doc, reformulations = expander.expand(query)

    # Monta a lista completa: documento HyDE + reformulações + query original
    expanded: list[str] = []
    if hyde_doc:
        expanded.append(hyde_doc)
    expanded.extend(reformulations)
    if query not in expanded:
        expanded.append(query)

    logger.info(
        "query_expander: %d queries expandidas (HyDE=%s)",
        len(expanded),
        "sim" if hyde_doc else "não",
    )

    return {
        "expanded_queries": expanded,
        "hyde_document": hyde_doc if hyde_doc else None,
        "retrieval_round": 1,
    }


def _generate_multihop_subquery(state: AgentState) -> str:
    """Gera a próxima sub-query para recuperação multi-hop."""
    llm = _get_llm()

    chunks = state.get("retrieved_chunks", [])
    retrieved_summary = "\n".join(
        f"- {c.chunk_id}: {c.text[:150]}..." for c in chunks[:5]
    )

    prompt = MULTIHOP_SUBQUERY_PROMPT.format(
        query=state["original_query"],
        retrieved_summary=retrieved_summary or "(nenhuma informação recuperada ainda)",
        current_answer=state.get("answer", "(nenhuma resposta gerada ainda)"),
    )

    raw = llm.generate(
        prompt=prompt,
        temperature=0.0,
        max_tokens=512,
    )

    result = _parse_model(raw, MultiHopSubQuery)
    if result is not None:
        logger.info(
            "Multi-hop sub-query: '%s' | razão: '%s'",
            result.sub_query[:80],
            result.reasoning[:100],
        )
        return result.sub_query

    logger.warning("Falha ao gerar sub-query multi-hop — reutilizando query original")
    return state["original_query"]


# ── Node 3: retriever ──────────────────────────────────────────────────────


def retriever(state: AgentState) -> dict:
    """Executa o RetrievalPipeline para cada query expandida e agrega os resultados.

    Para rodadas multi-hop > 1: anexa novos chunks aos existentes e
    remove duplicatas globalmente.

    Atualiza:
        retrieved_chunks, retrieval_round
    """

    expanded_queries = state.get("expanded_queries", [state["original_query"]])
    retrieval_round = state.get("retrieval_round", 1)
    existing_chunks = list(state.get("retrieved_chunks", []))

    logger.info(
        "retriever: round %d — %d queries a executar",
        retrieval_round,
        len(expanded_queries),
    )

    pipeline = RetrievalPipeline()

    # Coleta resultados de todas as queries expandidas simultaneamente
    all_results: list[RetrievalResult] = []

    def _fetch(idx: int, q: str) -> list[RetrievalResult]:
        res = pipeline.run(q)
        logger.debug(
            "retriever: query %d/%d retornou %d resultado(s)",
            idx + 1,
            len(expanded_queries),
            len(res),
        )
        return res

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(_fetch, i, q)
            for i, q in enumerate(expanded_queries)
        ]
        for future in concurrent.futures.as_completed(futures):
            all_results.extend(future.result())

    # Desduplicação: mantém o maior rerank_score por chunk_id
    best_by_id: dict[str, RetrievalResult] = {}
    for chunk in existing_chunks + all_results:
        cid = chunk.chunk_id
        score = chunk.rerank_score or chunk.rrf_score or chunk.score
        existing_score = (
            best_by_id[cid].rerank_score
            or best_by_id[cid].rrf_score
            or best_by_id[cid].score
        ) if cid in best_by_id else -float("inf")

        if score > existing_score:
            best_by_id[cid] = chunk

    deduped = list(best_by_id.values())

    logger.info(
        "retriever: %d chunks únicos (de %d total bruto)",
        len(deduped),
        len(existing_chunks) + len(all_results),
    )

    return {
        "retrieved_chunks": deduped,
        "retrieval_round": retrieval_round,
    }


# ── Node 4: reranker ──────────────────────────────────────────────────────


def reranker(state: AgentState) -> dict:
    """Reavalia a pontuação de todos os chunks recuperados contra a query original.

    Usa o ``CrossEncoderReranker`` para garantir que o ranking final seja relativo
    à intenção real do usuário (não às queries expandidas).

    Atualiza:
        retrieved_chunks (reordenados e limitados)
    """
    from src.retrieval.reranker import CrossEncoderReranker

    chunks = state.get("retrieved_chunks", [])
    original_query = state["original_query"]

    if not chunks:
        logger.warning("reranker: nenhum chunk para reranquear")
        return {"retrieved_chunks": []}

    logger.info(
        "reranker: reranqueando %d chunks contra query original",
        len(chunks),
    )

    reranker_instance = CrossEncoderReranker(top_k=Constants.RERANKER_TOP_K)
    reranked = reranker_instance.rerank(original_query, chunks)

    # Log top-3
    for i, chunk in enumerate(reranked[:3]):
        logger.debug(
            "reranker: top-%d — chunk_id='%s' score=%.4f",
            i + 1,
            chunk.chunk_id,
            chunk.rerank_score or 0.0,
        )

    return {"retrieved_chunks": reranked}


# ── Node 5: context_assembler ─────────────────────────────────────────────


def context_assembler(state: AgentState) -> dict:
    """Monta a string final de contexto a partir dos chunks reranqueados.

    Formata cada chunk com cabeçalhos de metadados, respeita o limite de tokens
    usando tiktoken e constrói a lista de fontes.

    Atualiza:
        assembled_context, context_token_count, sources
    """
    chunks = state.get("retrieved_chunks", [])
    max_tokens = Constants.CONTEXT_MAX_TOKENS

    logger.info("context_assembler: montando contexto de %d chunks", len(chunks))

    context_parts: list[str] = []
    sources: list[dict] = []
    total_tokens = 0
    seen_texts: set[str] = set()

    for chunk in chunks:
        # Pula duplicatas exatas
        text_hash = chunk.text.strip()
        if text_hash in seen_texts:
            continue
        seen_texts.add(text_hash)

        # Extração de metadados
        meta = chunk.metadata or {}
        doc_name = meta.get("titulo") or meta.get("source_file", "Documento")
        section = meta.get("material", "")
        page = meta.get("page", "?")

        # Formatação do chunk
        formatted = (
            f"[Documento: {doc_name} | Seção: {section} | Página: {page}]\n"
            f"{chunk.text}\n"
            f"---"
        )

        chunk_tokens = _count_tokens(formatted)
        if total_tokens + chunk_tokens > max_tokens:
            logger.debug(
                "context_assembler: budget atingido em %d tokens "
                "(chunk pularia para %d)",
                total_tokens,
                total_tokens + chunk_tokens,
            )
            break

        context_parts.append(formatted)
        total_tokens += chunk_tokens

        sources.append({
            "chunk_id": chunk.chunk_id,
            "doc_name": doc_name,
            "section": section,
            "page": page,
            "rerank_score": chunk.rerank_score,
        })

    assembled = "\n\n".join(context_parts)

    logger.info(
        "context_assembler: %d chunks incluídos | %d tokens",
        len(context_parts),
        total_tokens,
    )

    return {
        "assembled_context": assembled,
        "context_token_count": total_tokens,
        "sources": sources,
    }


# ── Node 6: generator ─────────────────────────────────────────────────────


def generator(state: AgentState) -> dict:
    """Chama o Claude para gerar uma resposta fundamentada com citações inline.

    Atualiza:
        answer
    """
    context = state.get("assembled_context", "")
    query = state["original_query"]

    logger.info("generator: gerando resposta")

    if not context:
        logger.warning("generator: contexto vazio — resposta será limitada")

    llm = _get_llm()
    prompt = GENERATOR_PROMPT.format(context=context, query=query)

    answer = llm.generate(
        prompt=prompt,
        temperature=0.0,
        max_tokens=Constants.LLM_MAX_TOKENS,
    )

    logger.info("generator: resposta gerada (%d chars)", len(answer))

    return {"answer": answer.strip()}


# ── Node 7: faithfulness_check ─────────────────────────────────────────────


def faithfulness_check(state: AgentState) -> dict:
    """Avalia se a resposta está fundamentada no contexto.

    Se não estiver fundamentada e restarem tentativas, reescreve a resposta usando
    ``FAITHFULNESS_CORRECTION_PROMPT``.

    Atualiza:
        faithfulness_score, faithfulness_reasoning, is_grounded,
        final_answer, faithfulness_retries, answer (se corrigida)
    """
    answer = state.get("answer", "")
    context = state.get("assembled_context", "")
    query = state["original_query"]
    retries = state.get("faithfulness_retries", 0)

    logger.info(
        "faithfulness_check: avaliando resposta (tentativa %d/%d)",
        retries + 1,
        Constants.MAX_FAITHFULNESS_RETRIES + 1,
    )

    llm = _get_llm()

    # ── Avaliar fidelidade ─────────────────────────────────────────
    eval_prompt = FAITHFULNESS_CHECK_PROMPT.format(
        query=query, context=context, answer=answer
    )

    raw = llm.generate(
        prompt=eval_prompt,
        temperature=0.0,
        max_tokens=1024,
    )

    # Fallback: assume como fundamentada para que o grafo não trave em saídas ruins do LLM
    fallback = FaithfulnessResult(is_grounded=True, reasoning="Falha ao parsear avaliação")
    eval_result = _parse_model(raw, FaithfulnessResult, fallback=fallback)

    if eval_result is fallback:
        logger.warning(
            "faithfulness_check: resposta inválida do LLM — assumindo grounded. Raw: %s",
            raw[:200],
        )

    logger.info(
        "faithfulness_check: grounded=%s score=%.2f",
        eval_result.is_grounded,
        eval_result.score,
    )

    # ── Se fundamentada ou sem tentativas restantes → finalizar ────
    if eval_result.is_grounded or retries >= Constants.MAX_FAITHFULNESS_RETRIES:
        if not eval_result.is_grounded:
            logger.warning(
                "faithfulness_check: não grounded mas retries esgotadas (%d/%d)",
                retries,
                Constants.MAX_FAITHFULNESS_RETRIES,
            )
        return {
            "faithfulness_score": eval_result.score,
            "faithfulness_reasoning": eval_result.reasoning,
            "is_grounded": eval_result.is_grounded,
            "final_answer": answer,
            "faithfulness_retries": retries,
        }

    # ── Não fundamentada + tentativas restantes → corrigir ─────────
    logger.warning(
        "faithfulness_check: %d afirmações não suportadas — corrigindo",
        len(eval_result.unsupported_claims),
    )

    correction_prompt = FAITHFULNESS_CORRECTION_PROMPT.format(
        context=context,
        answer=answer,
        faithfulness_evaluation=eval_result.reasoning,
        unsupported_claims="\n".join(f"- {c}" for c in eval_result.unsupported_claims),
    )

    corrected = llm.generate(
        prompt=correction_prompt,
        temperature=0.0,
        max_tokens=Constants.LLM_MAX_TOKENS,
    )

    logger.info(
        "faithfulness_check: resposta corrigida (%d → %d chars)",
        len(answer),
        len(corrected),
    )

    return {
        "faithfulness_score": eval_result.score,
        "faithfulness_reasoning": eval_result.reasoning,
        "is_grounded": False,
        "answer": corrected.strip(),
        "faithfulness_retries": retries + 1,
    }

