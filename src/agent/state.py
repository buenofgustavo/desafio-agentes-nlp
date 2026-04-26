"""Definição do AgentState para o LangGraph.

Define o TypedDict de estado compartilhado que flui através de todos os nós do grafo,
além de uma função fábrica para criar o estado inicial a partir de uma query do usuário.
"""
from __future__ import annotations

from typing import Literal, TypedDict

from src.core.models import RetrievalResult


class AgentState(TypedDict, total=False):
    """Estado compartilhado para o grafo do agente RAG.

    Todos os campos são opcionais (``total=False``) para que os nós precisem apenas
    retornar os campos que atualizam — o LangGraph os mescla automaticamente.
    """

    # ── Entrada ────────────────────────────────────────────────────
    original_query: str

    # ── Análise de Query ───────────────────────────────────────────
    query_type: Literal["simple", "comparative", "multi_hop"]

    # ── Expansão de Query ──────────────────────────────────────────
    expanded_queries: list[str]
    hyde_document: str | None

    # ── Recuperação (Retrieval) ────────────────────────────────────
    retrieved_chunks: list[RetrievalResult]
    retrieval_round: int

    # ── Contexto ───────────────────────────────────────────────────
    assembled_context: str
    context_token_count: int

    # ── Geração ────────────────────────────────────────────────────
    answer: str
    sources: list[dict]

    # ── Fidelidade (Faithfulness) ──────────────────────────────────
    faithfulness_score: float | None
    faithfulness_reasoning: str | None
    faithfulness_retries: int

    # ── Saída ──────────────────────────────────────────────────────
    final_answer: str
    is_grounded: bool


def initial_state(query: str) -> AgentState:
    """Cria um AgentState inicial com padrões sensatos.

    Args:
        query: A pergunta original do usuário.

    Returns:
        Um ``AgentState`` totalmente inicializado, pronto para invocação do grafo.
    """
    return AgentState(
        original_query=query,
        query_type="simple",
        expanded_queries=[],
        hyde_document=None,
        retrieved_chunks=[],
        retrieval_round=0,
        assembled_context="",
        context_token_count=0,
        answer="",
        sources=[],
        faithfulness_score=None,
        faithfulness_reasoning=None,
        faithfulness_retries=0,
        final_answer="",
        is_grounded=False,
    )
