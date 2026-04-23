"""Responsável pela geração dos Embeddings"""
import os
from typing import List

from sentence_transformers import SentenceTransformer

from src.utils.logger import LoggingService

logger = LoggingService.setup_logger(__name__)

# Maximize CPU parallelism for PyTorch operations as early as possible.
try:
    import torch
    _cpu_count = os.cpu_count() or 1
    torch.set_num_threads(_cpu_count)
    torch.set_num_interop_threads(max(1, _cpu_count // 2))
except Exception:
    pass

_model = None

def _default_device() -> str:
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        return 'cpu'


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        name = os.getenv('EMBEDDING_MODEL', 'intfloat/multilingual-e5-large')
        device = os.getenv('EMBEDDING_DEVICE', _default_device())
        logger.info(f'Carregando modelo: {name} no dispositivo {device} (~1.1 GB, pode demorar)')
        _model = SentenceTransformer(name, device=device)
    return _model


def embed_chunks(texts: List[str]) -> List[List[float]]:
    """
    Prefixo 'passage:' obrigatório para o e5 - indexação de documentos.

    Otimizações aplicadas:
    - batch_size tunado via env EMBEDDING_BATCH_SIZE (padrão 64 para CPU).
    - num_workers removido: não suportado no sentence-transformers >= 5.x.
    - Threads PyTorch configuradas no import para maximizar paralelismo CPU.
    """
    model = get_embedding_model()
    # 64 is a good default for CPU with multilingual-e5-large;
    # raise via EMBEDDING_BATCH_SIZE if you have more RAM.
    batch_size = int(os.getenv('EMBEDDING_BATCH_SIZE', '64'))
    prefixed = [f'passage: {t}' for t in texts]
    return model.encode(
        prefixed,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,  # necessário para similaridade coseno
    ).tolist()

def embed_query(query: str) -> List[float]:
    """
    Prefixo 'query:' obrigatório para o e5 - busca semântica.
    """
    model = get_embedding_model()
    return model.encode(
        f'query: {query}',
        normalize_embeddings=True,
    ).tolist()