from fastapi import FastAPI

from src.core.config import Constants
from src.ai.embeddings.embedder import get_embedding_model
from src.indexing.storage.vector_store import get_qdrant_client
from src.ai.embeddings.embedder import embed_query
from src.utils.logger import LoggingService
from src.ai.llm.factory import get_llm

from typing import Dict, List, Tuple
from dataclasses import dataclass
import os
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv
from qdrant_client.models import Filter, FieldCondition, MatchValue

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


class RAGService:
    def __init__(self, app: FastAPI):
        self.llm = app.state.llm
        self.embedding_model = app.state.embedder
        self.qdrant_client = app.state.qdrant
        
    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.5, apenas_vigentes: bool = False) -> List[SearchResult]:

        collection = os.getenv('QDRANT_COLLECTION', 'setor_eletrico')
        
        vector = embed_query(query)
        
        query_filter = None
        if apenas_vigentes:
            query_filter = Filter(must=[FieldCondition(
                key='situacao',
                match=MatchValue(value='NÃO CONSTA REVOGAÇÃO EXPRESSA'),
            )])
            
        hits = self.qdrant_client.search(
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
    

    def answer(self, question: str, top_k: int = 3) -> Tuple[str, List[Dict]]:
        if self.llm is None:
            raise RuntimeError("LLM não configurada. Crie um arquivo .env.")

        contexts = self.retrieve(question=question, top_k=top_k)
        context_text = "\n\n".join(
            [f"[Fonte {i + 1}] {ctx.text}" for i, ctx in enumerate(contexts)]
        )

        prompt = (
            "Responda usando somente o contexto recuperado.\n"
            "Se a resposta não estiver no contexto, diga explicitamente que não encontrou.\n\n"
            f"Contexto:\n{context_text}\n\nPergunta: {question}"
        )

        response = self.llm.generate(
            prompt=prompt,
            system_prompt="Você é um assistente objetivo e fiel ao contexto fornecido.",
            temperature=0.1,
        )

        return response, contexts
    