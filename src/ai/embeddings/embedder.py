"""Responsável pela geração dos Embeddings"""
import os
from typing import List

from sentence_transformers import SentenceTransformer

from src.utils.logger import LoggingService

logger = LoggingService.setup_logger(__name__)

# Maximiza o paralelismo da CPU para operações PyTorch o mais cedo possível.
try:
    import torch
    _cpu_count = os.cpu_count() or 1
    torch.set_num_threads(_cpu_count)
    torch.set_num_interop_threads(max(1, _cpu_count // 2))
except Exception:
    pass

import threading

_model = None
_lock = threading.Lock()

def _default_device() -> str:
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        return 'cpu'


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
                device = os.getenv('EMBEDDING_DEVICE', _default_device())
                logger.info(f'Carregando modelo: {name} no dispositivo {device} (pode demorar)')
                _model = SentenceTransformer(name, device=device)
    return _model


def embed_chunks(texts: List[str]) -> List[List[float]]:
    """
    Prefixo 'passage:' obrigatório para o e5 - indexação de documentos.

    Otimizações aplicadas:
    - batch_size ajustado via env EMBEDDING_BATCH_SIZE (padrão 64 para CPU).
    - num_workers removido: não suportado no sentence-transformers >= 5.x.
    - Threads PyTorch configuradas no import para maximizar paralelismo CPU.
    """
    model = get_embedding_model()
    # 64 é um bom padrão para CPU com all-MiniLM-L6-v2;
    # aumente via EMBEDDING_BATCH_SIZE se tiver mais RAM.
    batch_size = int(os.getenv('EMBEDDING_BATCH_SIZE', '64'))
    prefixed = [f'passage: {t}' for t in texts]
    return model.encode(
        prefixed,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,  # necessário para similaridade cosseno
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