"""Recuperador híbrido BM25 + Denso com Reciprocal Rank Fusion (RRF).

Ambos os recuperadores são executados **simultaneamente** via um ``ThreadPoolExecutor`` e
suas listas de resultados ranqueadas são fundidas usando a fórmula padrão RRF::

    RRF_score(d) = Σ  1 / (k + rank_i)

onde *k = 60* é a constante padrão de Cormack et al. (2009).
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


def _compute_rrf(
    results_lists: list[list[RetrievalResult]],
    k: int = 60,
) -> dict[str, float]:
    """Calcula os scores de Reciprocal Rank Fusion em múltiplas listas ranqueadas.

    Um chunk que aparece em múltiplas listas acumula contribuições de cada
    posição de ranking que ocupa.

    Args:
        results_lists: Lista de listas ranqueadas ``RetrievalResult`` de
            diferentes recuperadores (a ordem importa — rank 1 é o índice 0).
        k: Constante RRF. O valor padrão é 60.

    Returns:
        Dicionário mapeando ``chunk_id`` → score RRF agregado.
    """
    rrf: dict[str, float] = {}
    for ranked_list in results_lists:
        for rank, result in enumerate(ranked_list, start=1):
            cid = result.chunk_id
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (k + rank)
    return rrf


class HybridRetriever:
    """Recuperador híbrido que combina BM25 e busca densa via fusão RRF.

    Ambos os recuperadores são injetados via construtor e executados em threads
    paralelas. Se o BM25 estiver indisponível, o recuperador reverte graciosamente
    para resultados apenas densos.
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
        """Inicializa o recuperador híbrido.

        Args:
            bm25_retriever: Uma instância de ``BM25Retriever``.
            dense_retriever: Uma instância de ``SemanticSearch``.
            bm25_top_k: Candidatos buscados no BM25. Padrão:
                ``Constants.BM25_TOP_K``.
            dense_top_k: Candidatos buscados na busca densa. Padrão:
                ``Constants.DENSE_TOP_K``.
            rrf_k: Constante RRF *k*. Padrão: ``Constants.RRF_K``.
            final_top_k: Resultados retornados após a fusão. Padrão:
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
    # API Pública
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> list[RetrievalResult]:
        """Executa recuperação paralela e funde os resultados via RRF.

        Args:
            query: A query de busca do usuário.

        Returns:
            Lista de objetos ``RetrievalResult`` fundidos.
        """
        if not self._bm25.is_built:
            logger.warning(
                "HybridRetriever: índice BM25 indisponível — "
                "operando em modo apenas denso (dense-only)."
            )
            dense_results = self._dense.search(query, top_k=self._dense_top_k)
            return self._label_as_hybrid(dense_results)

        # ── Recuperação paralela ────────────────────────────────────────
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_bm25 = pool.submit(self._bm25.search, query, self._bm25_top_k)
            fut_dense = pool.submit(self._dense.search, query, self._dense_top_k)
            bm25_results: list[RetrievalResult] = fut_bm25.result()
            dense_results: list[RetrievalResult] = fut_dense.result()

        # ── Construção do lookup de chunks mesclados ────────────────────
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

        # ── Debug: detalhamento da origem ──────────────────────────────
        both = bm25_ids & dense_ids
        logger.debug(
            f"HybridRetriever detalhamento — "
            f"bm25={len(bm25_ids)}, dense={len(dense_ids)}, "
            f"ambos={len(both)}, só_bm25={len(bm25_ids - dense_ids)}, "
            f"só_dense={len(dense_ids - bm25_ids)}"
        )

        # ── Fusão RRF ───────────────────────────────────────────────────
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
    # Auxiliares privados
    # ------------------------------------------------------------------

    def _label_as_hybrid(
        self, results: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        """Rotula resultados apenas densos como híbridos (caminho de fallback).

        Atribui um ``rrf_score`` sintético usando a posição no ranking para que
        o código posterior sempre tenha um valor numérico para ordenar.
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
