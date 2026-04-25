from fastapi import FastAPI

from src.core.config import Constants
from src.ai.embeddings.embedder import get_embedding_model
from src.indexing.storage.vector_store import get_qdrant_client
from src.ai.embeddings.embedder import embed_query
from src.utils.logger import LoggingService
from src.ai.llm.factory import get_llm
from src.retrieval.reranker import CrossEncoderReranker
from src.core.models import RetrievalResult

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
    ementa: str


class RAGService:
    def __init__(self, app: FastAPI):
        self.llm = app.state.llm
        self.embedding_model = app.state.embedder
        self.qdrant_client = app.state.qdrant
        self.reranker = CrossEncoderReranker()
        
    def retrieve(self, query: str, top_k: int = 10, score_threshold: float = 0.3, apenas_vigentes: bool = False) -> List[SearchResult]:

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
            ementa=h.payload.get('ementa', ''),
        ) for h in hits]
    

    def answer(self, question: str, top_k: int = 5) -> Tuple[str, Dict]:
        if self.llm is None:
            raise RuntimeError("LLM não configurada. Crie um arquivo .env.")

        # Retrieval
        contexts = self.retrieve(
            query=question,
            top_k=10,
            apenas_vigentes=True
        )

        # Mapear índices para preservar metadata
        context_map = {i: ctx for i, ctx in enumerate(contexts)}

        # Converte para RetrievalResult (formato do reranker)
        candidates = [
            RetrievalResult(
                chunk_id=i,
                text=ctx.text,
                metadata={},
                score=ctx.score,
                source=ctx.source_file,
                rrf_score=0.0
            )
            for i, ctx in enumerate(contexts)
        ]

        # Aplica o reranker (cross-encoder)
        reranked = self.reranker.rerank(question, candidates)

        # Reordena mantendo o metadata original
        contexts = [
            context_map[r.chunk_id]
            for r in reranked
        ]

        # Filtro de qualidade
        contexts = contexts[:5]
        if not contexts:
            contexts = self.retrieve(
            query=question,
            top_k=3,
            score_threshold=0.0,
            apenas_vigentes=True
        )

        # Construção do contexto enriquecido
        context_text = "\n\n".join([
            f"""[Fonte {i + 1}]
    Título: {ctx.titulo}
    Assunto: {ctx.assunto}
    Data de publicação: {ctx.data_publicacao}
    Página: {ctx.page}

    Conteúdo:
    {ctx.ementa + "\n\n" if ctx.ementa else ""}
    {ctx.text[:1200]}
    """
            for i, ctx in enumerate(contexts)
        ])

        # Prompt com estrutura e citações
        prompt = f"""
    Sua tarefa é responder perguntas com base EXCLUSIVAMENTE no contexto fornecido.

    Regras:
    - Use apenas as informações do contexto.
    - Não invente ou complemente com conhecimento externo.
    - Se a resposta não estiver no contexto, diga exatamente: "Não encontrei essa informação nos documentos."
    - Seja claro, objetivo e técnico quando necessário.

    - Para cada informação relevante na resposta, cite explicitamente a fonte no formato:
    (Fonte X)

    - As citações devem aparecer no final de cada frase ou bullet point.

    - Estruture a resposta da seguinte forma:

    Resposta:
    <resposta clara e objetiva com citações (Fonte X) ao longo do texto>

    Fontes utilizadas:
    - Liste apenas as fontes que foram realmente citadas

    - Sempre que possível, organize a resposta em bullet points.

    Contexto:
    {context_text}

    Pergunta:
    {question}

    Resposta:
    """

        # Chamada do LLM com o prompt estruturado
        response = self.llm.generate(
            prompt=prompt,
            system_prompt="""
    Você é um especialista no setor elétrico brasileiro.

    Responda apenas com base no contexto fornecido.
    Seja preciso, técnico quando necessário, e não invente informações.
    """,
            temperature=0.1,
        )

        # Construção de explicabilidade
        evidences = [
            {
                "fonte": f"Fonte {i+1}",
                "titulo": ctx.titulo,
                "pagina": ctx.page,
                "score": round(ctx.score, 3),
                "trecho": ctx.text[:300]
            }
            for i, ctx in enumerate(contexts)
        ]

        return response, {
            "contexts": contexts,
            "evidences": evidences
    }