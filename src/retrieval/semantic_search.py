"""Dense semantic search using multilingual-e5-large embeddings and Qdrant.

This module wraps the existing Qdrant search infrastructure established in
``rag_pipeline.py`` and exposes a clean, typed interface compatible with the
Phase 2 hybrid pipeline.
"""
from __future__ import annotations

from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from src.ai.embeddings.embedder import embed_query, get_embedding_model
from src.core.config import Constants
from src.core.models import RetrievalResult
from src.indexing.storage.vector_store import get_qdrant_client
from src.utils.logger import LoggingService

logger = LoggingService.setup_logger(__name__)


class SemanticSearch:
    """Dense retriever backed by Qdrant + multilingual-e5-large embeddings.

    Mirrors the retrieval logic in ``RAGService.retrieve()`` from
    ``rag_pipeline.py``, but returns ``RetrievalResult`` objects so it can
    be composed with ``HybridRetriever`` and ``CrossEncoderReranker``.

    The embedding model is loaded eagerly at construction time so the first
    query does not incur the large model-loading overhead.
    """

    def __init__(
        self,
        qdrant_client: Optional[QdrantClient] = None,
        collection: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> None:
        """Initialize the dense retriever.

        Args:
            qdrant_client: Optional pre-built ``QdrantClient``. If *None*,
                one is created from ``QDRANT_URL`` in config.
            collection: Qdrant collection name. Defaults to
                ``Constants.QDRANT_COLLECTION``.
            top_k: Default number of results to return. Defaults to
                ``Constants.DENSE_TOP_K``.
        """
        self._client: QdrantClient = qdrant_client or get_qdrant_client()
        self._collection: str = collection or Constants.QDRANT_COLLECTION
        self._top_k: int = top_k if top_k is not None else Constants.DENSE_TOP_K

        # Eagerly load the embedding model — prevents cold-start latency on
        # the first query call.
        get_embedding_model()

        logger.info(
            f"SemanticSearch inicializado (coleção='{self._collection}', top_k={self._top_k})"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: float = 0.0,
        apenas_vigentes: bool = False,
    ) -> list[RetrievalResult]:
        """Run dense vector search against Qdrant.

        Follows the same pattern as ``RAGService.retrieve()`` in
        ``rag_pipeline.py``: embed the query with the ``query:`` prefix
        required by multilingual-e5, then search with cosine similarity.

        Args:
            query: The search query string (plain text, no prefix needed).
            top_k: Max results to return. Overrides the instance default
                when provided.
            score_threshold: Minimum cosine similarity score (0–1).
                Results below this threshold are discarded by Qdrant.
            apenas_vigentes: If *True*, filters to documents whose ``situacao``
                field equals ``'NÃO CONSTA REVOGAÇÃO EXPRESSA'``.

        Returns:
            Ranked list of ``RetrievalResult`` ordered by cosine score
            descending, with ``source="dense"``.
        """
        k: int = top_k if top_k is not None else self._top_k

        vector: list[float] = embed_query(query)

        query_filter: Optional[Filter] = None
        if apenas_vigentes:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="situacao",
                        match=MatchValue(value="NÃO CONSTA REVOGAÇÃO EXPRESSA"),
                    )
                ]
            )

        try:
            hits = self._client.search(
                collection_name=self._collection,
                query_vector=vector,
                limit=k,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=True,
            )
        except Exception as exc:
            logger.error(f"SemanticSearch: erro ao consultar Qdrant — {exc}")
            return []

        results: list[RetrievalResult] = []
        for hit in hits:
            payload: dict = hit.payload or {}
            # Everything except 'text' becomes metadata (source_file, page,
            # titulo, assunto, situacao, publicacao, ementa, etc.)
            metadata: dict = {k: v for k, v in payload.items() if k != "text"}
            results.append(
                RetrievalResult(
                    chunk_id=str(hit.id),
                    text=payload.get("text", ""),
                    metadata=metadata,
                    score=float(hit.score),
                    source="dense",
                )
            )

        logger.debug(
            f"SemanticSearch: {len(results)} resultado(s) para '{query[:60]}'"
        )
        return results
