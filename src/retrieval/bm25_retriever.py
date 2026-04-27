"""Recuperador esparso BM25 sobre o corpus de chunks indexados no Qdrant.

O índice é construído fazendo o scroll de TODOS os pontos do Qdrant (garantindo que o BM25
e a busca densa operem sobre dados idênticos), tokenizado com pré-processamento
específico para português e persistido no disco para carregamento rápido.

Uso via CLI::

    python -m src.retrieval.bm25_retriever --rebuild
    python -m src.retrieval.bm25_retriever --rebuild --index-path /caminho/personalizado
"""
from __future__ import annotations

import argparse
import pickle
from typing import Any, Iterable, Union
from pathlib import Path
from typing import Optional

import nltk
import bm25s

from src.core.config import Constants
from src.core.models import RetrievalResult
from src.indexing.storage.vector_store import get_qdrant_client
from src.utils.logger import LoggingService

logger = LoggingService.setup_logger(__name__)

_PT_STOPWORDS: Optional[set[str]] = None


def _get_stopwords() -> set[str]:
    """Carrega as stopwords do NLTK para português de forma preguiçosa, baixando-as se necessário."""
    global _PT_STOPWORDS
    if _PT_STOPWORDS is None:
        try:
            from nltk.corpus import stopwords as _sw

            _PT_STOPWORDS = set(_sw.words("portuguese"))
        except LookupError:
            logger.warning("NLTK stopwords não encontrados — baixando agora…")
            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords as _sw

            _PT_STOPWORDS = set(_sw.words("portuguese"))
        logger.debug(f"Stopwords PT carregadas: {len(_PT_STOPWORDS)} termos")
    return _PT_STOPWORDS

def _tokenize(text_or_texts: Union[str, Iterable[str]]) -> Any:
    """Tokeniza com pré-processamento específico para português usando bm25s.

    Args:
        text_or_texts: Uma única string (para a query) ou uma lista de strings (para o corpus).

    Returns:
        O objeto tokenizado esperado por bm25s.BM25.
    """
    stopwords_pt = _get_stopwords()
    
    # bm25s.tokenize natively handles lowercasing, splitting, and stopword removal
    return bm25s.tokenize(text_or_texts, stopwords=stopwords_pt)


class BM25Retriever:
    """Recuperador esparso BM25 baseado em um índice persistido sobre chunks do Qdrant.

    Fluxo de construção:
        1. Scroll de todos os pontos do Qdrant para obter textos e IDs dos chunks.
        2. Tokenização.
        3. Ajuste do ``bm25s.BM25`` no corpus tokenizado.
        4. Persistência do índice no ``index_path`` e do armazenamento de chunks em ``chunks.pkl``.

    Em carregamentos subsequentes, apenas o passo 4 (desserialização) é necessário.
    """

    def __init__(self, index_path: Optional[str] = None) -> None:
        """Carrega o índice do disco se ele existir; caso contrário, avisa e permanece não construído.

        Args:
            index_path: Caminho para o diretório persistido. Padrão:
                ``Constants.BM25_INDEX_PATH``.
        """
        self._index_path = (
            Path(index_path) if index_path else Constants.BM25_INDEX_PATH
        )
        
        if self._index_path.suffix == '.pkl':
            self._index_path = self._index_path.with_suffix('')
        
        self._bm25: Optional[bm25s.BM25] = None
        # Cada entrada: {"chunk_id": str, "text": str, "metadata": dict}
        self._chunks: list[dict] = []
        self._is_built: bool = False

        if self._index_path.exists():
            try:
                self.load()
            except Exception as exc:
                logger.warning(
                    f"BM25Retriever: falha ao carregar índice existente ({exc}). "
                    "Execute '--rebuild' para reconstruir."
                )
        else:
            logger.warning(
                f"BM25Retriever: índice não encontrado em '{self._index_path}'. "
                "Execute 'python -m src.retrieval.bm25_retriever --rebuild'."
            )


    def build(self, chunks: Optional[list[dict]] = None) -> None:
        """Constrói o índice BM25 e o persiste no disco.

        Se *chunks* for ``None``, todos os pontos do Qdrant são coletados automaticamente via scroll.

        Args:
            chunks: Lista opcional de dicionários pré-construída, cada um com as chaves
                ``chunk_id``, ``text`` e ``metadata``. Útil para testes unitários.
        """
        if chunks is None:
            chunks = self._scroll_qdrant_chunks()

        if not chunks:
            logger.error("BM25Retriever.build: nenhum chunk encontrado — abortando.")
            return

        logger.info(f"Construindo índice BM25 com {len(chunks):,} chunks…")
        self._chunks = chunks
        
        texts = [c["text"] for c in self._chunks]
        
        logger.info("Tokenizando textos com bm25s...")
        stopwords_pt = _get_stopwords()
        corpus_tokens = _tokenize(texts)

        logger.info("Criando o índice de matrizes esparsas...")
        self._bm25 = bm25s.BM25()
        self._bm25.index(corpus_tokens)
        self._is_built = True

        logger.info("Salvando modelo bm25s e metadados no disco...")
        self._bm25.save(str(self._index_path))

        # Salva nossos chunks brutos lado a lado para podermos mapear índices de volta aos IDs do Qdrant
        with open(self._index_path / "chunks.pkl", "wb") as fh:
            pickle.dump(self._chunks, fh, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"Índice BM25 salvo no diretório '{self._index_path}'")

    def load(self) -> None:
        """Carrega o índice BM25 persistido do disco.

        Raises:
            FileNotFoundError: Se o ``index_path`` não existir.
        """
        if not self._index_path.exists():
            raise FileNotFoundError(
                f"BM25Retriever: arquivo de índice não encontrado: {self._index_path}"
            )

        logger.info(f"Carregando índice BM25 de '{self._index_path}'…")
        
        self._bm25 = bm25s.BM25.load(str(self._index_path), load_corpus=False)
        
        chunks_path = self._index_path / "chunks.pkl"
        with open(chunks_path, "rb") as fh:
            self._chunks = pickle.load(fh)
            
        self._is_built = True
        logger.info(f"Índice BM25 carregado: {len(self._chunks):,} chunks")

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[RetrievalResult]:
        """Busca no índice BM25 e retorna os resultados ranqueados.

        Retorna uma lista vazia com log de AVISO se o índice não estiver construído.

        Args:
            query: Query de busca em texto simples.
            top_k: Número de resultados. Padrão: ``Constants.BM25_TOP_K``.

        Returns:
            Lista ranqueada de ``RetrievalResult`` com ``source="bm25"``,
            ordenada por score BM25 decrescente. Resultados com score zero são omitidos.
        """
        if not self._is_built:
            logger.warning(
                "BM25Retriever.search: índice não construído — retornando lista vazia."
            )
            return []

        k: int = top_k if top_k is not None else Constants.BM25_TOP_K
        tokens = _tokenize(query)

        if not tokens:
            logger.warning(
                "BM25Retriever.search: query sem tokens válidos após pré-processamento."
            )
            return []

        raw_scores = self._bm25.get_scores(tokens)

        # Ranqueia os índices por score decrescente
        top_indices = sorted(
            range(len(raw_scores)), key=lambda i: raw_scores[i], reverse=True
        )[:k]

        results: list[RetrievalResult] = []
        for idx in top_indices:
            score = float(raw_scores[idx])
            if score <= 0.0:
                break  # Scores BM25Okapi são não-negativos; 0 significa sem sobreposição de termos
            chunk = self._chunks[idx]
            results.append(
                RetrievalResult(
                    chunk_id=chunk["chunk_id"],
                    text=chunk["text"],
                    metadata=chunk["metadata"],
                    score=score,
                    source="bm25",
                )
            )

        logger.debug(
            f"BM25: {len(results)} resultado(s) para '{query[:60]}'"
        )
        return results

    @property
    def is_built(self) -> bool:
        """``True`` se o índice foi construído ou carregado com sucesso."""
        return self._is_built


    def _scroll_qdrant_chunks(self) -> list[dict]:
        """Faz scroll de todos os pontos do Qdrant e retorna como uma lista de dicionários de chunks.

        Returns:
            Lista de dicionários com chaves ``chunk_id``, ``text`` e ``metadata``.
            Pontos com payloads de texto vazios são ignorados silenciosamente.
        """
        client = get_qdrant_client(timeout=600.0)
        collection = Constants.QDRANT_COLLECTION

        logger.info(
            f"Fazendo scroll de todos os pontos da coleção '{collection}'…"
        )

        chunks: list[dict] = []
        offset = None
        batch_size = 100

        while True:
            batch, next_offset = client.scroll(
                collection_name=collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in batch:
                payload: dict = point.payload or {}
                text: str = payload.get("text", "")
                if not text.strip():
                    continue
                metadata = {k: v for k, v in payload.items() if k != "text"}
                chunks.append(
                    {
                        "chunk_id": str(point.id),
                        "text": text,
                        "metadata": metadata,
                    }
                )

            logger.debug(f"  … {len(chunks):,} chunks coletados até agora")

            if next_offset is None:
                break
            offset = next_offset

        logger.info(f"Scroll concluído: {len(chunks):,} chunks no total")
        return chunks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gerencia o índice BM25 do pipeline RAG"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Reconstrói o índice BM25 a partir do Qdrant",
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default=None,
        help="Caminho alternativo para o diretório de índice",
    )
    args = parser.parse_args()

    retriever = BM25Retriever(index_path=args.index_path)

    if args.rebuild:
        logger.info("Iniciando rebuild do índice BM25…")
        retriever.build()
        logger.info("Rebuild concluído com sucesso.")
    elif not retriever.is_built:
        logger.error(
            "Índice BM25 não encontrado. Use '--rebuild' para construí-lo."
        )
