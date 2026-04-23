"""Responsável pela geração dos Embeddings"""
import os
from typing import List

from sentence_transformers import SentenceTransformer

from src.utils.logger import LoggingService

logger = LoggingService.setup_logger(__name__)

_model = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        name = os.getenv('EMBEDDING_MODEL', 'intfloat/multilingual-e5-large')
        logger.info(f'Carregando modelo: {name} (~1.1 GB, pode demorar)')
        _model = SentenceTransformer(name)
    return _model

def embed_chunks(texts: List[str]) -> List[List[float]]:
    """
    Prefixo 'passage:' obrigatório para o e5 - indexação de documentos.
    Processa em batches.
    """
    model = get_model()
    prefixed = [f'passage: {t}' for t in texts]
    return model.encode(
        prefixed,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).tolist()

def embed_query(query: str) -> List[float]:
    """
    Prefixo 'query:' obrigatório para o e5 - busca semântica.
    """
    model = get_model()
    return model.encode(
        f'query: {query}',
        normalize_embeddings=True,
    ).tolist()