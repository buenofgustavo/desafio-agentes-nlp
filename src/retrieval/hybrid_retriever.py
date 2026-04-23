"""Hybrid BM25 + Dense retriever with Reciprocal Rank Fusion (RRF).

Both retrievers are run **concurrently** via a ``ThreadPoolExecutor`` and
their ranked result lists are fused using the standard RRF formula::

    RRF_score(d) = Σ  1 / (k + rank_i)

where *k = 60* is the standard constant from Cormack et al. (2009).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Optional

from src.core.config import Constants
from src.core.models import RetrievalResult
from src.utils.logger import LoggingService

if TYPE_CHECKING:
    from src.retrieval.bm25_retriever import BM25Retriever
    from src.retrieval.semantic_search import SemanticSearch

logger = LoggingService.setup_logger(__name__)


# ---------------------------------------------------------------------------
# RRF helper
# ---------------------------------------------------------------------------


def _compute_rrf(
    results_lists: list[list[RetrievalResult]],
    k: int = 60,
) -> dict[str, float]:
    """Compute Reciprocal Rank Fusion scores across multiple ranked lists.

    A chunk appearing in multiple lists accumulates contributions from each
    ranked position it occupies.

    Args:
        results_lists: List of ranked ``RetrievalResult`` lists from
            different retrievers (order matters — rank 1 is index 0).
        k: RRF constant. Standard value is 60.

    Returns:
        Dict mapping ``chunk_id`` → aggregated RRF score.
    """
    rrf: dict[str, float] = {}
    for ranked_list in results_lists:
        for rank, result in enumerate(ranked_list, start=1):
            cid = result.chunk_id
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (k + rank)
    return rrf


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------


class HybridRetriever:
    """Hybrid retriever combining BM25 and dense search via RRF fusion.

    Both retrievers are injected via the constructor (dependency injection)
    and run in parallel threads. If BM25 is unavailable the retriever
    gracefully falls back to dense-only results.
    """

    def __init__(
        self,
        bm25_retriever: "BM25Retriever",
        dense_retriever: "SemanticSearch",
        bm25_top_k: Optional[int] = None,
        dense_top_k: Optional[int] = None,
        rrf_k: Optional[int] = None,
        final_top_k: Optional[int] = None,
    ) -> None:
        """Initialize the hybrid retriever.

        Args:
            bm25_retriever: A ``BM25Retriever`` instance (may be unbuilt).
            dense_retriever: A ``SemanticSearch`` instance.
            bm25_top_k: Candidates fetched from BM25. Defaults to
                ``Constants.BM25_TOP_K``.
            dense_top_k: Candidates fetched from dense search. Defaults to
                ``Constants.DENSE_TOP_K``.
            rrf_k: RRF *k* constant. Defaults to ``Constants.RRF_K``.
            final_top_k: Results returned after fusion. Defaults to
                ``Constants.HYBRID_FINAL_TOP_K``.
        """
        self._bm25 = bm25_retriever
        self._dense = dense_retriever
        self._bm25_top_k: int = (
            bm25_top_k if bm25_top_k is not None else Constants.BM25_TOP_K
        )
        self._dense_top_k: int = (
            dense_top_k if dense_top_k is not None else Constants.DENSE_TOP_K
        )
        self._rrf_k: int = rrf_k if rrf_k is not None else Constants.RRF_K
        self._final_top_k: int = (
            final_top_k if final_top_k is not None else Constants.HYBRID_FINAL_TOP_K
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> list[RetrievalResult]:
        """Run parallel retrieval and fuse results via RRF.

        Returns top ``final_top_k`` results with ``source="hybrid"`` and
        ``rrf_score`` populated, sorted by ``rrf_score`` descending.

        Falls back to dense-only (with a WARNING) when BM25 is unavailable.

        Args:
            query: The user search query.

        Returns:
            List of fused ``RetrievalResult`` objects.
        """
        if not self._bm25.is_built:
            logger.warning(
                "HybridRetriever: índice BM25 indisponível — "
                "operando em modo dense-only."
            )
            dense_results = self._dense.search(query, top_k=self._dense_top_k)
            return self._label_as_hybrid(dense_results)

        # ── Parallel retrieval ──────────────────────────────────────────
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_bm25 = pool.submit(self._bm25.search, query, self._bm25_top_k)
            fut_dense = pool.submit(self._dense.search, query, self._dense_top_k)
            bm25_results: list[RetrievalResult] = fut_bm25.result()
            dense_results: list[RetrievalResult] = fut_dense.result()

        # ── Build merged chunk lookup ───────────────────────────────────
        all_by_id: dict[str, RetrievalResult] = {}
        bm25_ids: set[str] = set()
        dense_ids: set[str] = set()

        for r in bm25_results:
            all_by_id[r.chunk_id] = r
            bm25_ids.add(r.chunk_id)

        for r in dense_results:
            if r.chunk_id not in all_by_id:
                all_by_id[r.chunk_id] = r
            dense_ids.add(r.chunk_id)

        # ── Debug: origin breakdown ─────────────────────────────────────
        both = bm25_ids & dense_ids
        logger.debug(
            f"HybridRetriever origin breakdown — "
            f"bm25={len(bm25_ids)}, dense={len(dense_ids)}, "
            f"ambos={len(both)}, só_bm25={len(bm25_ids - dense_ids)}, "
            f"só_dense={len(dense_ids - bm25_ids)}"
        )

        # ── RRF fusion ──────────────────────────────────────────────────
        rrf_scores = _compute_rrf(
            [bm25_results, dense_results], k=self._rrf_k
        )

        sorted_ids = sorted(
            rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True
        )

        fused: list[RetrievalResult] = []
        for cid in sorted_ids[: self._final_top_k]:
            base = all_by_id[cid]
            rs = rrf_scores[cid]
            fused.append(
                RetrievalResult(
                    chunk_id=cid,
                    text=base.text,
                    metadata=base.metadata,
                    score=rs,
                    source="hybrid",
                    rrf_score=rs,
                )
            )

        logger.info(
            f"HybridRetriever: {len(fused)} resultado(s) após fusão RRF "
            f"(bm25={len(bm25_ids)}, dense={len(dense_ids)})"
        )
        return fused

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _label_as_hybrid(
        self, results: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        """Re-label dense-only results as hybrid (fallback path).

        Assigns a synthetic ``rrf_score`` using rank position so downstream
        code always has a numeric value to sort on.
        """
        out: list[RetrievalResult] = []
        for rank, r in enumerate(results[: self._final_top_k], start=1):
            synthetic_rrf = 1.0 / (self._rrf_k + rank)
            out.append(
                RetrievalResult(
                    chunk_id=r.chunk_id,
                    text=r.text,
                    metadata=r.metadata,
                    score=r.score,
                    source="hybrid",
                    rrf_score=synthetic_rrf,
                )
            )
        return out
