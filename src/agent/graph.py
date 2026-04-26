"""Máquina de estado LangGraph para o agente RAG.

Define e compila o grafo conectando os 7 nós com arestas condicionais
para tentativas de fidelidade e loops multi-hop.

Uso::

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
    """Lógica de roteamento após a verificação de fidelidade.

    Árvore de decisão:
        1. Se fundamentada (grounded) → END
        2. Se não fundamentada e restarem tentativas de fidelidade → retry
           (volta para o faithfulness_check, que corrigirá a resposta)
        3. Se for multi-hop e restarem rodadas de recuperação → volta para
           o query_expander para outra rodada de busca
        4. Caso contrário → END (com log de aviso)

    Args:
        state: Estado atual do agente após faithfulness_check.

    Returns:
        Um entre ``"end"``, ``"retry_faithfulness"`` ou ``"retry_multihop"``.
    """
    is_grounded = state.get("is_grounded", False)
    retries = state.get("faithfulness_retries", 0)
    query_type = state.get("query_type", "simple")
    retrieval_round = state.get("retrieval_round", 1)

    # 1. Fundamentada → finalizado
    if is_grounded:
        logger.info("route: resposta fundamentada — finalizando")
        return "end"

    # 2. Não fundamentada + tentativas de fidelidade restantes
    if retries < Constants.MAX_FAITHFULNESS_RETRIES:
        logger.info(
            "route: não fundamentada — retry faithfulness (%d/%d)",
            retries + 1,
            Constants.MAX_FAITHFULNESS_RETRIES,
        )
        return "retry_faithfulness"

    # 3. Multi-hop + rodadas de recuperação restantes
    if query_type == "multi_hop" and retrieval_round < Constants.MULTIHOP_MAX_ROUNDS:
        logger.info(
            "route: multi-hop — nova rodada de busca (%d/%d)",
            retrieval_round + 1,
            Constants.MULTIHOP_MAX_ROUNDS,
        )
        return "retry_multihop"

    # 4. Opções esgotadas
    logger.warning(
        "route: não fundamentada e sem tentativas restantes — "
        "finalizando com a melhor resposta disponível"
    )
    return "end"


def build_graph() -> StateGraph:
    """Constrói e compila o grafo do agente LangGraph.

    Returns:
        Um grafo compilado pronto para invocação via ``.invoke()``.
    """
    graph = StateGraph(AgentState)

    # ── Registro dos nós ───────────────────────────────────────────
    graph.add_node("query_analyzer", query_analyzer)
    graph.add_node("query_expander", query_expander)
    graph.add_node("retriever", retriever)
    graph.add_node("reranker", reranker)
    graph.add_node("context_assembler", context_assembler)
    graph.add_node("generator", generator)
    graph.add_node("faithfulness_check", faithfulness_check)

    # ── Ponto de entrada ───────────────────────────────────────────
    graph.set_entry_point("query_analyzer")

    # ── Arestas lineares ───────────────────────────────────────────
    graph.add_edge("query_analyzer", "query_expander")
    graph.add_edge("query_expander", "retriever")
    graph.add_edge("retriever", "reranker")
    graph.add_edge("reranker", "context_assembler")
    graph.add_edge("context_assembler", "generator")
    graph.add_edge("generator", "faithfulness_check")

    # ── Aresta condicional: fidelidade → END ou loop ───────────────
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


# ── Singleton em nível de módulo ───────────────────────────────────────────
agent_graph = build_graph()
