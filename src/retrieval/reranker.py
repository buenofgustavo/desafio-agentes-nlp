"""Reranker Cross-encoder usando cross-encoder/mmarco-mMiniLMv2-L12-H384-v1.

O modelo é carregado **uma vez na instanciação** e a pontuação é feita em uma
**única chamada de inferência em lote** sobre todos os candidatos.

A thread safety é garantida por um lock em nível de instância ao redor de ``model.predict()``.
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
    """Reranker Cross-encoder para listas de candidatos de recuperação.

    Carrega o modelo uma vez e usa inferência em lote para pontuação rápida.
    Seguro para ser chamado de múltiplas threads.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> None:
        """Carrega o modelo cross-encoder.

        Args:
            model_name: Identificador do modelo no HuggingFace. Padrão:
                ``Constants.RERANKER_MODEL``.
            top_k: Número de candidatos a retornar após a pontuação. Padrão:
                ``Constants.RERANKER_TOP_K``.
        """
        self._model_name: str = model_name or Constants.RERANKER_MODEL
        self._top_k: int = top_k if top_k is not None else Constants.RERANKER_TOP_K
        self._lock = threading.Lock()

        logger.info(f"Carregando cross-encoder: '{self._model_name}'…")
        self._model = CrossEncoder(self._model_name)
        logger.info("Cross-encoder carregado com sucesso.")

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Pontua e reordena candidatos em uma única chamada de inferência em lote.

        Args:
            query: A query de busca original.
            candidates: Resultados candidatos para reordenar (tipicamente do
                ``HybridRetriever``).

        Returns:
            Top ``top_k`` objetos ``RetrievalResult`` com ``rerank_score``
            preenchido, ordenados decrescentemente.
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
