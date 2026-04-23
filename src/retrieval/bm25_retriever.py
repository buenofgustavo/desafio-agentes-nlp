"""BM25 sparse retriever over the Qdrant-indexed chunk corpus.

The index is built by scrolling ALL points from Qdrant (ensuring BM25 and
dense retrieval operate over identical data), tokenized with Portuguese-aware
preprocessing, and persisted to disk as a pickle file for fast reloads.

CLI usage::

    python -m src.retrieval.bm25_retriever --rebuild
    python -m src.retrieval.bm25_retriever --rebuild --index-path /custom/path.pkl
"""
from __future__ import annotations

import argparse
import pickle
import re
import string
from pathlib import Path
from typing import Optional

import nltk
from rank_bm25 import BM25Okapi

from src.core.config import Constants
from src.core.models import RetrievalResult
from src.indexing.storage.vector_store import get_qdrant_client
from src.utils.logger import LoggingService

logger = LoggingService.setup_logger(__name__)

# ---------------------------------------------------------------------------
# Portuguese tokenizer
# ---------------------------------------------------------------------------

_PT_STOPWORDS: Optional[set[str]] = None


def _get_stopwords() -> set[str]:
    """Lazy-load Portuguese NLTK stopwords, downloading them if necessary."""
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


_PUNCT_RE = re.compile(r"[" + re.escape(string.punctuation) + r"\d]+")


def _tokenize(text: str) -> list[str]:
    """Tokenize with Portuguese-aware preprocessing.

    Steps:
        1. Lowercase.
        2. Remove punctuation and digits.
        3. Split on whitespace.
        4. Remove Portuguese stopwords and single-character tokens.

    Args:
        text: Raw input text.

    Returns:
        List of cleaned tokens suitable for BM25 indexing.
    """
    text = _PUNCT_RE.sub(" ", text.lower())
    stopwords = _get_stopwords()
    return [tok for tok in text.split() if tok not in stopwords and len(tok) > 1]


# ---------------------------------------------------------------------------
# BM25Retriever
# ---------------------------------------------------------------------------


class BM25Retriever:
    """BM25 sparse retriever backed by a persisted index over Qdrant chunks.

    Build workflow:
        1. Scroll all points from Qdrant to get chunk texts and IDs.
        2. Tokenize with ``_tokenize()``.
        3. Fit ``BM25Okapi`` on the tokenized corpus.
        4. Persist the index and chunk store to ``index_path`` via pickle.

    On subsequent loads, only step 4 (deserialization) is needed.
    """

    def __init__(self, index_path: Optional[str] = None) -> None:
        """Load the index from disk if it exists; otherwise warn and stay unbuilt.

        Args:
            index_path: Path to the persisted pickle file. Defaults to
                ``Constants.BM25_INDEX_PATH``.
        """
        self._index_path: Path = (
            Path(index_path) if index_path else Constants.BM25_INDEX_PATH
        )
        self._bm25: Optional[BM25Okapi] = None
        # Each entry: {"chunk_id": str, "text": str, "metadata": dict}
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, chunks: Optional[list[dict]] = None) -> None:
        """Build the BM25 index and persist it to disk.

        If *chunks* is ``None``, all Qdrant points are scrolled automatically.

        Args:
            chunks: Optional pre-built list of dicts, each with keys
                ``chunk_id``, ``text``, and ``metadata``. Useful for
                unit-testing without a live Qdrant instance.
        """
        if chunks is None:
            chunks = self._scroll_qdrant_chunks()

        if not chunks:
            logger.error("BM25Retriever.build: nenhum chunk encontrado — abortando.")
            return

        logger.info(f"Construindo índice BM25 com {len(chunks):,} chunks…")
        self._chunks = chunks

        tokenized = [_tokenize(c["text"]) for c in chunks]
        self._bm25 = BM25Okapi(tokenized)
        self._is_built = True

        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._index_path, "wb") as fh:
            pickle.dump({"bm25": self._bm25, "chunks": self._chunks}, fh)

        logger.info(f"Índice BM25 salvo em '{self._index_path}' ({len(chunks):,} chunks)")

    def load(self) -> None:
        """Load the persisted BM25 index from disk.

        Raises:
            FileNotFoundError: If ``index_path`` does not exist.
        """
        if not self._index_path.exists():
            raise FileNotFoundError(
                f"BM25Retriever: arquivo de índice não encontrado: {self._index_path}"
            )

        logger.info(f"Carregando índice BM25 de '{self._index_path}'…")
        with open(self._index_path, "rb") as fh:
            data = pickle.load(fh)

        self._bm25 = data["bm25"]
        self._chunks = data["chunks"]
        self._is_built = True
        logger.info(f"Índice BM25 carregado: {len(self._chunks):,} chunks")

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[RetrievalResult]:
        """Search the BM25 index and return ranked results.

        Returns an empty list with a WARNING log if the index is not built.

        Args:
            query: Plain-text search query.
            top_k: Number of results. Defaults to ``Constants.BM25_TOP_K``.

        Returns:
            Ranked list of ``RetrievalResult`` with ``source="bm25"``,
            sorted by BM25 score descending. Zero-score results are omitted.
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

        # Rank indices by score descending
        top_indices = sorted(
            range(len(raw_scores)), key=lambda i: raw_scores[i], reverse=True
        )[:k]

        results: list[RetrievalResult] = []
        for idx in top_indices:
            score = float(raw_scores[idx])
            if score <= 0.0:
                break  # BM25Okapi scores are non-negative; 0 means no term overlap
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
        """``True`` if the index has been successfully built or loaded."""
        return self._is_built

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _scroll_qdrant_chunks(self) -> list[dict]:
        """Scroll all points from Qdrant and return as a list of chunk dicts.

        Returns:
            List of dicts with keys ``chunk_id``, ``text``, and ``metadata``.
            Points with empty ``text`` payloads are silently skipped.
        """
        client = get_qdrant_client()
        collection = Constants.QDRANT_COLLECTION

        logger.info(
            f"Fazendo scroll de todos os pontos da coleção '{collection}'…"
        )

        chunks: list[dict] = []
        offset = None
        batch_size = 256

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


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

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
        help="Caminho alternativo para o arquivo de índice (.pkl)",
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
