"""Responsável pela busca semântica dos documentos"""
import os
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from src.ai.embeddings.embedder import embed_query
from src.utils.logger import LoggingService

logger = LoggingService.setup_logger(__name__)

load_dotenv()

@dataclass
class SearchResult:
    text: str
    score: float
    source_file: str
    page: int
    titulo: str
    assunto: str
    situacao: str
    data_publicacao: str

def search(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.5,
    apenas_vigentes: bool = False,
) -> List[SearchResult]:
    client = QdrantClient(url=os.getenv('QDRANT_URL', 'http://localhost:6333'))
    collection = os.getenv('QDRANT_COLLECTION', 'setor_eletrico')
    
    vector = embed_query(query)
    
    query_filter = None
    if apenas_vigentes:
        query_filter = Filter(must=[FieldCondition(
            key='situacao',
            match=MatchValue(value='NÃO CONSTA REVOGAÇÃO EXPRESSA'),
        )])
        
    hits = client.search(
        collection_name=collection,
        query_vector=vector,
        limit=top_k,
        score_threshold=score_threshold,
        query_filter=query_filter,
        with_payload=True,
    )
    
    return [SearchResult(
        text=h.payload.get('text', ''),
        score=h.score,
        source_file=h.payload.get('source_file', ''),
        page=h.payload.get('page', 0),
        titulo=h.payload.get('titulo', ''),
        assunto=h.payload.get('assunto', ''),
        situacao=h.payload.get('situacao', ''),
        data_publicacao=h.payload.get('data_publicacao', ''),
    ) for h in hits]