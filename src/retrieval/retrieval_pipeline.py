"""Pipeline de recuperação unificado.

Este é o **ponto de importação único** para o agente LangGraph.
Conecta o BM25, busca densa, híbrida e reordenação (reranking) por cross-encoder.

Uso::

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
    """Pipeline de recuperação ponta a ponta: busca híbrida → reordenação por cross-encoder.

    A inicialização carrega todos os componentes a partir dos valores de configuração.
    """

    def __init__(self) -> None:
        """Inicializa todos os componentes de recuperação a partir da configuração em paralelo."""
        import concurrent.futures
        logger.info("Inicializando RetrievalPipeline em paralelo…")

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # ── Dispara o carregamento dos componentes em paralelo ────────
            f_dense = executor.submit(SemanticSearch, top_k=Constants.DENSE_TOP_K)
            f_bm25 = executor.submit(BM25Retriever)
            f_reranker = executor.submit(CrossEncoderReranker, top_k=Constants.RERANKER_TOP_K)

            # ── Aguarda a conclusão e atribui à instância ────────────────
            self._dense = f_dense.result()
            self._bm25 = f_bm25.result()
            self._reranker = f_reranker.result()

        # ── Combinador híbrido (depende dos anteriores, mas é rápido) ─────
        if not self._bm25.is_built:
            logger.warning(
                "RetrievalPipeline: índice BM25 ausente — "
                "pipeline operará em modo apenas denso. "
                "Execute 'python -m src.retrieval.bm25_retriever --rebuild' "
                "para habilitar busca híbrida."
            )

        self._hybrid = HybridRetriever(
            bm25_retriever=self._bm25,
            dense_retriever=self._dense,
            bm25_top_k=Constants.BM25_TOP_K,
            dense_top_k=Constants.DENSE_TOP_K,
            rrf_k=Constants.RRF_K,
            final_top_k=Constants.HYBRID_FINAL_TOP_K,
        )

        logger.info("RetrievalPipeline pronto.")

    def run(self, query: str) -> list[RetrievalResult]:
        """Executa o pipeline de recuperação completo para uma query.

        Passos:
            1. Recuperação híbrida (BM25 + denso + fusão RRF).
            2. Reordenação por cross-encoder dos candidatos fundidos.

        Args:
            query: A query de busca do usuário (texto simples em português).

        Returns:
            Lista de objetos ``RetrievalResult`` ordenados por ``rerank_score`` decrescente.
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

    @property
    def dense(self) -> SemanticSearch:
        """Expõe o recuperador denso para a avaliação baseline."""
        return self._dense
