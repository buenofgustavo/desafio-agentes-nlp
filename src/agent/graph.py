"""LangGraph state machine for the RAG agent.

Defines and compiles the graph connecting the 7 nodes with conditional
edges for faithfulness retries and multi-hop loops.

Usage::

    from src.agent.graph import agent_graph

    result = agent_graph.invoke({"original_query": "..."})
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.agent.nodes import (
    context_assembler,
    faithfulness_check,
    generator,
    query_analyzer,
    query_expander,
    reranker,
    retriever,
)
from src.agent.state import AgentState
from src.core.config import Constants
from src.utils.logger import LoggingService

logger = LoggingService.setup_logger(__name__)


def route_after_faithfulness(state: AgentState) -> str:
    """Routing logic after faithfulness check.

    Decision tree:
        1. If grounded → END
        2. If not grounded and faithfulness retries remain → retry
           (go back to faithfulness_check, which will correct the answer)
        3. If multi_hop and retrieval rounds remain → go back to
           query_expander for another retrieval round
        4. Otherwise → END (with warning logged)

    Args:
        state: Current agent state after faithfulness_check.

    Returns:
        One of ``"end"``, ``"retry_faithfulness"``, or ``"retry_multihop"``.
    """
    is_grounded = state.get("is_grounded", False)
    retries = state.get("faithfulness_retries", 0)
    query_type = state.get("query_type", "simple")
    retrieval_round = state.get("retrieval_round", 1)

    # 1. Grounded → done
    if is_grounded:
        logger.info("route: resposta grounded — finalizando")
        return "end"

    # 2. Not grounded + faithfulness retries remaining
    if retries < Constants.MAX_FAITHFULNESS_RETRIES:
        logger.info(
            "route: não grounded — retry faithfulness (%d/%d)",
            retries + 1,
            Constants.MAX_FAITHFULNESS_RETRIES,
        )
        return "retry_faithfulness"

    # 3. Multi-hop + retrieval rounds remaining
    if query_type == "multi_hop" and retrieval_round < Constants.MULTIHOP_MAX_ROUNDS:
        logger.info(
            "route: multi-hop — nova rodada de retrieval (%d/%d)",
            retrieval_round + 1,
            Constants.MULTIHOP_MAX_ROUNDS,
        )
        return "retry_multihop"

    # 4. Exhausted all options
    logger.warning(
        "route: não grounded e sem retries/rounds restantes — "
        "finalizando com melhor resposta disponível"
    )
    return "end"


def build_graph() -> StateGraph:
    """Build and compile the LangGraph agent graph.

    Returns:
        A compiled graph ready for invocation via ``.invoke()``.
    """
    graph = StateGraph(AgentState)

    # ── Register nodes ─────────────────────────────────────────────
    graph.add_node("query_analyzer", query_analyzer)
    graph.add_node("query_expander", query_expander)
    graph.add_node("retriever", retriever)
    graph.add_node("reranker", reranker)
    graph.add_node("context_assembler", context_assembler)
    graph.add_node("generator", generator)
    graph.add_node("faithfulness_check", faithfulness_check)

    # ── Entry point ────────────────────────────────────────────────
    graph.set_entry_point("query_analyzer")

    # ── Linear edges ───────────────────────────────────────────────
    graph.add_edge("query_analyzer", "query_expander")
    graph.add_edge("query_expander", "retriever")
    graph.add_edge("retriever", "reranker")
    graph.add_edge("reranker", "context_assembler")
    graph.add_edge("context_assembler", "generator")
    graph.add_edge("generator", "faithfulness_check")

    # ── Conditional edge: faithfulness → END or loop ───────────────
    graph.add_conditional_edges(
        "faithfulness_check",
        route_after_faithfulness,
        {
            "end": END,
            "retry_faithfulness": "faithfulness_check",
            "retry_multihop": "query_expander",
        },
    )

    logger.info("Agent graph compilado com sucesso")
    return graph.compile()


# ── Module-level singleton ─────────────────────────────────────────────────
agent_graph = build_graph()
