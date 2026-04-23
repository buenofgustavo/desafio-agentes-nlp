"""Cross-encoder reranker using ms-marco-multilingual-rerank-mmarco-v2.

The model is loaded **once at instantiation** (not per query) and scoring
is done in a **single batched inference call** over all candidates, keeping
GPU/CPU utilisation high and avoiding Python-level for-loops over the model.

Thread safety is ensured by an instance-level lock around ``model.predict()``.
"""
from __future__ import annotations

import threading
from typing import Optional

import numpy as np
from sentence_transformers import CrossEncoder

from src.core.config import Constants
from src.core.models import RetrievalResult
from src.utils.logger import LoggingService

logger = LoggingService.setup_logger(__name__)


class CrossEncoderReranker:
    """Cross-encoder reranker for retrieval candidate lists.

    Loads the model once and uses batched inference for fast scoring.
    Safe to call from multiple threads (lock protects ``predict()``).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> None:
        """Load the cross-encoder model.

        Auto-detects CUDA vs CPU for device placement.

        Args:
            model_name: HuggingFace model identifier. Defaults to
                ``Constants.RERANKER_MODEL``.
            top_k: Number of candidates to return after scoring. Defaults to
                ``Constants.RERANKER_TOP_K``.
        """
        self._model_name: str = model_name or Constants.RERANKER_MODEL
        self._top_k: int = top_k if top_k is not None else Constants.RERANKER_TOP_K
        self._lock = threading.Lock()

        logger.info(f"Carregando cross-encoder: '{self._model_name}'…")
        self._model = CrossEncoder(self._model_name)
        logger.info("Cross-encoder carregado com sucesso.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Score and rerank candidates in a single batched inference call.

        All (query, passage) pairs are scored together so hardware stays
        as busy as possible. Results are sorted by ``rerank_score`` descending
        and the top ``top_k`` are returned.

        Args:
            query: The original search query.
            candidates: Candidate results to rerank (typically from
                ``HybridRetriever``).

        Returns:
            Top ``top_k`` ``RetrievalResult`` objects with ``rerank_score``
            populated, sorted descending. Returns ``[]`` for empty input.
        """
        if not candidates:
            return []

        pairs = [(query, c.text) for c in candidates]

        with self._lock:
            raw_scores: np.ndarray = self._model.predict(
                pairs,
                batch_size=32,
                show_progress_bar=False,
            )

        score_list: list[float] = raw_scores.tolist()

        scored = sorted(
            zip(candidates, score_list),
            key=lambda pair: pair[1],
            reverse=True,
        )

        reranked: list[RetrievalResult] = []
        for candidate, score in scored[: self._top_k]:
            reranked.append(
                RetrievalResult(
                    chunk_id=candidate.chunk_id,
                    text=candidate.text,
                    metadata=candidate.metadata,
                    score=candidate.score,
                    source=candidate.source,
                    rrf_score=candidate.rrf_score,
                    rerank_score=float(score),
                )
            )

        logger.debug(
            f"CrossEncoderReranker: {len(reranked)} resultado(s) "
            f"reordenados de {len(candidates)} candidatos"
        )
        return reranked
