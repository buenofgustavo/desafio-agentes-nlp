"""Coleção criada no Qdrant"""
from typing import Dict, List, Tuple
import time
import uuid
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.utils.logger import LoggingService

logger = LoggingService.setup_logger(__name__)

load_dotenv()

def get_qdrant_client() -> QdrantClient:
    logger.info('Carregando QdrantClient')
    return QdrantClient(url=os.getenv('QDRANT_URL', 'http://localhost:6333'))


def create_collection(client: QdrantClient, name: str, vector_size: int = 384) -> None:
    
    existing = [c.name for c in client.get_collections().collections]
    
    if name in existing:
        logger.info(f'Coleção "{name}" já existe - pulando criação.')
        return
    logger.info(f'Criando coleção "{name}" com vector_size={vector_size}...')    
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    logger.info(f'Coleção "{name}" criada com sucesso.')


def upload_embeddings(client: QdrantClient, collection_name: str, iterator, batch_size=1000) -> int:
    batch = []
    total = 0
    start_time = time.time()

    for i, row in enumerate(iterator):
        batch.append(
            PointStruct(
                id=i,
                vector=row["embedding"],
                payload={
                    "source_file": row["source_file"],
                    "page": row["page"],
                    "parent_index": row["parent_index"],
                    "child_index": row["child_index"],
                    "titulo": row["titulo"],
                    "esfera": row["esfera"],
                    "situacao": row["situacao"],
                    "assinatura": row["assinatura"],
                    "publicacao": row["publicacao"],
                    "assunto": row["assunto"],
                    "ementa": row["ementa"],
                    "text_to_embed": row["text_to_embed"],
                    "parent_text": row["parent_text"],
                    "text": row["text"]
                },
            )
        )

        if len(batch) >= batch_size:
            client.upsert(collection_name, points=batch)
            total += len(batch)
            print(f"{total} inseridos")

            total_expected = 4500000
            elapsed = time.time() - start_time
            rate = total / elapsed if elapsed > 0 else 0
            remaining = total_expected - total
            eta = remaining / rate if rate > 0 else 0 

            print(f"{total}/{total_expected} | ({(total/total_expected)*100:.2f}%) | {rate:.0f} itens/s | ETA: {eta/60:.1f} min")

            batch = []

    if batch:
        client.upsert(collection_name, points=batch)
        total += len(batch)

    return total