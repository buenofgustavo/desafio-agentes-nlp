"""Busca semântica densa usando embeddings all-MiniLM-L6-v2 e Qdrant.

Este módulo expõe uma interface limpa e tipada para busca vetorial,
compatível com o pipeline híbrido da Fase 2.
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
    """Recuperador denso baseado em Qdrant + embeddings all-MiniLM-L6-v2.

    Retorna objetos ``RetrievalResult`` para que possa ser composto com
    ``HybridRetriever`` e ``CrossEncoderReranker``.

    O modelo de embedding é carregado no momento da construção para evitar
    latência na primeira consulta.
    """

    def __init__(
        self,
        qdrant_client: Optional[QdrantClient] = None,
        collection: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> None:
        """Inicializa o recuperador denso.

        Args:
            qdrant_client: ``QdrantClient`` pré-construído opcional. Se *None*,
                um será criado a partir de ``QDRANT_URL`` na configuração.
            collection: Nome da coleção no Qdrant. Padrão:
                ``Constants.QDRANT_COLLECTION``.
            top_k: Número padrão de resultados a retornar. Padrão:
                ``Constants.DENSE_TOP_K``.
        """
        self._client: QdrantClient = qdrant_client or get_qdrant_client()
        self._collection: str = collection or Constants.QDRANT_COLLECTION
        self._top_k: int = top_k if top_k is not None else Constants.DENSE_TOP_K

        # Carrega o modelo de embedding antecipadamente — evita latência de cold-start
        get_embedding_model()

        logger.info(
            f"SemanticSearch inicializado (coleção='{self._collection}', top_k={self._top_k})"
        )

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: float = 0.0,
        apenas_vigentes: bool = False,
    ) -> list[RetrievalResult]:
        """Executa busca vetorial densa no Qdrant.

        Args:
            query: A string de busca (texto simples).
            top_k: Máximo de resultados a retornar. Sobrescreve o padrão da instância.
            score_threshold: Score mínimo de similaridade de cosseno (0–1).
            apenas_vigentes: Se *True*, filtra documentos cujo campo ``situacao``
                seja ``'NÃO CONSTA REVOGAÇÃO EXPRESSA'``.

        Returns:
            Lista ranqueada de ``RetrievalResult`` ordenada por score de cosseno
            decrescente, com ``source="dense"``.
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
            response = self._client.query_points(
                collection_name=self._collection,
                query=vector,
                limit=k,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=True,
            )
            hits = response.points
        except Exception as exc:
            logger.error(f"SemanticSearch: erro ao consultar Qdrant — {exc}")
            return []

        results: list[RetrievalResult] = []
        for hit in hits:
            payload: dict = hit.payload or {}
            # Tudo exceto 'text' torna-se metadados
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
