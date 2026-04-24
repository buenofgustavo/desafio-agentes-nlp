"""LangGraph AgentState definition.

Defines the shared state TypedDict that flows through all graph nodes,
plus a factory function to create an initial state from a user query.
"""
from __future__ import annotations

from typing import Literal, TypedDict

from src.core.models import RetrievalResult


class AgentState(TypedDict, total=False):
    """Shared state for the RAG agent graph.

    All fields are optional (``total=False``) so that nodes only need to
    return the fields they update — LangGraph merges them automatically.

    The state must remain JSON-serializable for LangSmith tracing.
    ``RetrievalResult`` objects are Pydantic models and serialize via
    ``.model_dump()``.
    """

    # ── Input ──────────────────────────────────────────────────────
    original_query: str

    # ── Query analysis ─────────────────────────────────────────────
    query_type: Literal["simple", "comparative", "multi_hop"]

    # ── Query expansion ────────────────────────────────────────────
    expanded_queries: list[str]
    hyde_document: str | None

    # ── Retrieval ──────────────────────────────────────────────────
    retrieved_chunks: list[RetrievalResult]
    retrieval_round: int

    # ── Context ────────────────────────────────────────────────────
    assembled_context: str
    context_token_count: int

    # ── Generation ─────────────────────────────────────────────────
    answer: str
    sources: list[dict]

    # ── Faithfulness ───────────────────────────────────────────────
    faithfulness_score: float | None
    faithfulness_reasoning: str | None
    faithfulness_retries: int

    # ── Output ─────────────────────────────────────────────────────
    final_answer: str
    is_grounded: bool


def initial_state(query: str) -> AgentState:
    """Create an initial AgentState with sensible defaults.

    Args:
        query: The user's original question.

    Returns:
        A fully initialized ``AgentState`` ready for graph invocation.
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
