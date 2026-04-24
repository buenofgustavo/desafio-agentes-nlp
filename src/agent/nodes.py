"""LangGraph node functions for the RAG agent.

Each node is a standalone function with signature ``(state: AgentState) -> dict``.
Nodes return only the state fields they update — LangGraph merges them.

All LLM calls go through ``get_llm()`` from the existing factory.
All prompt strings are imported from ``prompts.py``.
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

# ── Shared helpers ─────────────────────────────────────────────────────────

_tokenizer: tiktoken.Encoding | None = None


def _get_tokenizer() -> tiktoken.Encoding:
    """Lazy-load the tiktoken encoder (singleton)."""
    global _tokenizer  # noqa: PLW0603
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer


def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken cl100k_base encoding."""
    return len(_get_tokenizer().encode(text))


def _get_llm():
    """Return the configured Anthropic LLM instance."""
    return get_llm("anthropic", Constants.CLAUDE_MODEL)


_M = TypeVar("_M", bound=BaseModel)


def _parse_model(raw: str, model: type[_M], fallback: _M | None = None) -> _M | None:
    """Strip markdown fences, parse JSON, and validate against a Pydantic model.

    Args:
        raw:      Raw LLM response string (may contain markdown code fences).
        model:    Pydantic model class to validate against.
        fallback: Value to return on parse/validation failure.
                  If ``None``, returns ``None`` on failure.

    Returns:
        A validated model instance, or ``fallback`` / ``None`` on failure.
    """
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
    try:
        return model.model_validate_json(cleaned)
    except (ValidationError, ValueError):
        logger.debug("_parse_model: falha ao validar %s. Raw: %s", model.__name__, raw)
        return fallback


# ── Node 1: query_analyzer ─────────────────────────────────────────────────


def query_analyzer(state: AgentState) -> dict:
    """Classify the query as simple, comparative, or multi_hop.

    Calls Claude with ``QUERY_ANALYZER_PROMPT`` and parses the response
    into a ``QueryAnalysis`` Pydantic model.

    Updates:
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
    """Generate HyDE document and query reformulations.

    Skipped (returns original query only) if ``HYDE_ENABLED=false``.

    Updates:
        expanded_queries, hyde_document
    """
    query = state["original_query"]
    logger.info("query_expander: expandindo query")

    # For multi-hop rounds > 1, generate a sub-query instead
    retrieval_round = state.get("retrieval_round", 0)
    if retrieval_round >= 1 and state.get("query_type") == "multi_hop":
        logger.info(
            "query_expander: multi-hop round %d — gerando sub-query",
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

    # Build the full list: HyDE doc + reformulations + original query
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
    """Generate the next sub-query for multi-hop retrieval."""
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
    """Run RetrievalPipeline for each expanded query and aggregate results.

    For multi-hop rounds > 1: appends new chunks to existing ones and
    de-duplicates globally.

    Updates:
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

    # Collect results from all expanded queries concurrently
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

    # De-duplicate: keep highest rerank_score per chunk_id
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
    """Re-score all retrieved chunks against the original query.

    Uses ``CrossEncoderReranker`` to ensure the final ranking is relative
    to the user's actual intent (not expanded queries).

    Updates:
        retrieved_chunks (reordered and trimmed)
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
    """Assemble the final context string from reranked chunks.

    Formats each chunk with metadata headers, enforces the token budget
    using tiktoken, and builds the sources list.

    Updates:
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
        # Skip exact duplicates
        text_hash = chunk.text.strip()
        if text_hash in seen_texts:
            continue
        seen_texts.add(text_hash)

        # Extract metadata
        meta = chunk.metadata or {}
        doc_name = meta.get("titulo") or meta.get("source_file", "Documento")
        section = meta.get("material", "")
        page = meta.get("page", "?")

        # Format chunk
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
    """Call Claude to generate a grounded answer with inline citations.

    Updates:
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
    """Evaluate whether the answer is grounded in the context.

    If not grounded and retries remain, rewrites the answer using
    ``FAITHFULNESS_CORRECTION_PROMPT``.

    Updates:
        faithfulness_score, faithfulness_reasoning, is_grounded,
        final_answer, faithfulness_retries, answer (if corrected)
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

    # ── Evaluate faithfulness ──────────────────────────────────────
    eval_prompt = FAITHFULNESS_CHECK_PROMPT.format(
        query=query, context=context, answer=answer
    )

    raw = llm.generate(
        prompt=eval_prompt,
        temperature=0.0,
        max_tokens=1024,
    )

    # Fallback: assume grounded so the graph doesn't stall on bad LLM output
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

    # ── If grounded or out of retries → finalize ───────────────────
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

    # ── Not grounded + retries remaining → correct ─────────────────
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

