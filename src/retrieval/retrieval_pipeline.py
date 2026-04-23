"""Unified retrieval pipeline.

This is the **single import point** for LangGraph agent.
It wires together BM25, dense, hybrid, and cross-encoder reranking from
``Constants`` and degrades gracefully if the BM25 index is absent.

Usage::

    from src.retrieval.retrieval_pipeline import RetrievalPipeline

    pipe = RetrievalPipeline()
    results = pipe.run("Qual é o prazo mínimo para revisão tarifária?")
"""
from __future__ import annotations

from src.core.config import Constants
from src.core.models import RetrievalResult
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.semantic_search import SemanticSearch
from src.utils.logger import LoggingService

logger = LoggingService.setup_logger(__name__)


class RetrievalPipeline:
    """End-to-end retrieval pipeline: hybrid retrieval → cross-encoder reranking.

    Initialization loads all components from config values in ``Constants``.
    If the BM25 index file is absent, a ``WARNING`` is logged and the
    pipeline continues in dense-only mode — it **never raises** during init.

    This class is the only import Phase 4 needs from the retrieval layer.
    """

    def __init__(self) -> None:
        """Initialize all retrieval components from config. Never raises."""
        logger.info("Inicializando RetrievalPipeline…")

        # ── Dense retriever (always available) ──────────────────────────
        self._dense = SemanticSearch(top_k=Constants.DENSE_TOP_K)

        # ── BM25 retriever (may be unavailable) ─────────────────────────
        self._bm25 = BM25Retriever()
        if not self._bm25.is_built:
            logger.warning(
                "RetrievalPipeline: índice BM25 ausente — "
                "pipeline operará em modo dense-only. "
                "Execute 'python -m src.retrieval.bm25_retriever --rebuild' "
                "para habilitar busca híbrida."
            )

        # ── Hybrid combiner ──────────────────────────────────────────────
        self._hybrid = HybridRetriever(
            bm25_retriever=self._bm25,
            dense_retriever=self._dense,
            bm25_top_k=Constants.BM25_TOP_K,
            dense_top_k=Constants.DENSE_TOP_K,
            rrf_k=Constants.RRF_K,
            final_top_k=Constants.HYBRID_FINAL_TOP_K,
        )

        # ── Cross-encoder reranker ───────────────────────────────────────
        self._reranker = CrossEncoderReranker(top_k=Constants.RERANKER_TOP_K)

        logger.info("RetrievalPipeline pronto.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, query: str) -> list[RetrievalResult]:
        """Execute the full retrieval pipeline for a query.

        Steps:
            1. Hybrid retrieval (BM25 + dense + RRF fusion).
            2. Cross-encoder reranking of the fused candidates.

        Args:
            query: The user search query (plain Portuguese text).

        Returns:
            Top ``RERANKER_TOP_K`` ``RetrievalResult`` objects with all score
            fields (``score``, ``rrf_score``, ``rerank_score``) populated,
            sorted by ``rerank_score`` descending.
        """
        logger.info(f"RetrievalPipeline.run: '{query[:80]}'")

        candidates = self._hybrid.retrieve(query)
        if not candidates:
            logger.warning(
                "RetrievalPipeline: busca híbrida não retornou candidatos."
            )
            return []

        reranked = self._reranker.rerank(query, candidates)
        logger.info(f"RetrievalPipeline: {len(reranked)} resultado(s) finais")
        return reranked

    # ------------------------------------------------------------------
    # Accessors for evaluation script
    # ------------------------------------------------------------------

    @property
    def dense(self) -> SemanticSearch:
        """Expose the dense retriever for the baseline evaluation."""
        return self._dense
