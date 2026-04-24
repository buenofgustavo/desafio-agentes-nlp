"""Responsável pela geração dos Embeddings"""
import os
import json
from typing import Dict, List

from sentence_transformers import SentenceTransformer

from src.core.config import Constants
from src.core.models import ChildChunk
from src.utils.logger import LoggingService
import torch

logger = LoggingService.setup_logger(__name__)

torch.set_num_threads(os.cpu_count())

_model = None
def get_embedding_model() -> str:
    global _model
    if _model is None:
        _model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return _model


def embed_chunks(chunks: List[ChildChunk]) -> None:
    model = get_embedding_model()

    batch_size = int(os.getenv('EMBEDDING_BATCH_SIZE', '512'))

    path = Constants.EMBEDDINGS_DIR / "embeddings.jsonl"
    Constants.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            texts = [c.text_to_embed for c in batch]

            vectors = model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )

            for chunk, vector in zip(batch, vectors):
                row = {**chunk.__dict__, "embedding": vector.tolist()}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

            print(f"Processado: {i + len(batch)}/{len(chunks)}")

def embed_query(query: str) -> List[float]:
    model = get_embedding_model()
    return model.encode(
        f'query: {query}',
        normalize_embeddings=True,
    ).tolist()

def save_embeddings(rows: List[Dict]):
    Constants.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Salvando {len(rows)} embeddings em {Constants.EMBEDDINGS_DIR}...")

    with open(Constants.EMBEDDINGS_DIR, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)