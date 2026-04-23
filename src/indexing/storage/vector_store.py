"""Coleção criada no Qdrant"""
import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.utils.logger import LoggingService

logger = LoggingService.setup_logger(__name__)

load_dotenv()

def get_qdrant_client() -> QdrantClient:
    logger.info('Carregando QdrantClient')
    return QdrantClient(url=os.getenv('QDRANT_URL', 'http://localhost:6333'))

def create_collection(name: str, vector_size: int = 1024) -> None:
    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    
    if name in existing:
        logger.info(f'Coleção "{name}" já existe - pulando criação.')
        return
        
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    logger.info(f'Coleção "{name}" criada com sucesso.')