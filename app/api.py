"""API REST FastAPI para o agente RAG.

Expõe dois endpoints:
    GET  /health   — status do sistema, incluindo conectividade com Qdrant
    POST /query    — executa o agente LangGraph completo e retorna uma resposta estruturada
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.agent.graph import agent_graph
from src.agent.state import initial_state
from src.core.config import Constants
from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.utils.logger import LoggingService

logger = LoggingService.setup_logger(__name__)


# ── Modelos Pydantic ───────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=5000)


class QueryResponse(BaseModel):
    question: str
    answer: str
    query_type: str
    faithfulness_score: float | None
    is_grounded: bool
    retrieval_round: int
    sources: list[dict]
    latency_seconds: float


# ── Lifespan: inicializa singletons uma única vez ──────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa componentes pesados no startup; limpa no shutdown."""
    logger.info("API: inicializando componentes (lifespan startup)…")

    # RetrievalPipeline — carrega modelo de embedding + BM25 + cross-encoder
    app.state.retrieval_pipeline = RetrievalPipeline()

    # ThreadPoolExecutor — usado para rodar o agente síncrono fora do event loop
    app.state.executor = ThreadPoolExecutor(max_workers=4)

    # agent_graph é um singleton em nível de módulo importado de graph.py
    app.state.agent_graph = agent_graph

    logger.info("API: todos os componentes prontos.")
    yield

    # Shutdown
    app.state.executor.shutdown(wait=False)
    logger.info("API: shutdown concluído.")


# ── Aplicação FastAPI ──────────────────────────────────────────────────────

app = FastAPI(
    title="RAG — Setor Elétrico Brasileiro",
    description="API de perguntas e respostas sobre documentos do setor elétrico brasileiro.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Middleware: cabeçalho X-Request-ID ─────────────────────────────────────


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Anexa um X-Request-ID único a cada resposta e loga a requisição."""
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    response = await call_next(request)
    latency = time.perf_counter() - start
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "%s %s → %d | %.3fs | id=%s",
        request.method,
        request.url.path,
        response.status_code,
        latency,
        request_id,
    )
    return response


# ── Auxiliares ─────────────────────────────────────────────────────────────


def _check_qdrant() -> str:
    """Retorna 'connected' ou 'error: <detail>' testando a API REST do Qdrant."""
    try:
        with httpx.Client(timeout=3.0) as client:
            r = client.get(f"{Constants.QDRANT_URL}/collections")
            r.raise_for_status()
        return "connected"
    except Exception as exc:
        logger.warning("Qdrant health check falhou: %s", exc)
        return f"error: {exc}"


def _run_agent(question: str) -> dict[str, Any]:
    """Executa o agente LangGraph de forma síncrona e retorna um dicionário de resultados.

    Esta função é chamada dentro de um ThreadPoolExecutor.
    """
    state = initial_state(question)
    result = agent_graph.invoke(state)

    return {
        "answer": result.get("final_answer") or result.get("answer", ""),
        "query_type": result.get("query_type", "simple"),
        "faithfulness_score": result.get("faithfulness_score"),
        "is_grounded": result.get("is_grounded", False),
        "retrieval_round": result.get("retrieval_round", 1),
        "sources": result.get("sources", []),
    }


# ── Endpoints ─────────────────────────────────────────────────────────────


@app.get("/health", summary="Check de saúde do sistema")
async def health() -> dict:
    """Retorna o status do sistema."""
    qdrant_status = _check_qdrant()
    bm25_loaded = (
        hasattr(app.state, "retrieval_pipeline")
        and app.state.retrieval_pipeline._bm25.is_built
    )
    return {
        "status": "ok",
        "qdrant": qdrant_status,
        "bm25_index": "carregado" if bm25_loaded else "não carregado (modo dense-only)",
        "model": Constants.CLAUDE_MODEL,
    }


@app.post("/query", response_model=QueryResponse, summary="Executar agente RAG")
async def query(req: QueryRequest) -> QueryResponse:
    """Executa o agente LangGraph completo para a pergunta fornecida.

    O agente roda em um pool de threads para evitar o bloqueio do event loop assíncrono.
    """
    logger.info("POST /query: question='%s'", req.question[:80])
    start = time.perf_counter()

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            app.state.executor,
            _run_agent,
            req.question,
        )
    except Exception as exc:  
        logger.error("Falha na execução do agente: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Agent execution failed", "detail": str(exc)},
        )

    latency = round(time.perf_counter() - start, 3)
    logger.info("POST /query: concluído em %.3fs", latency)

    return QueryResponse(
        question=req.question,
        answer=result["answer"],
        query_type=result["query_type"],
        faithfulness_score=result["faithfulness_score"],
        is_grounded=result["is_grounded"],
        retrieval_round=result["retrieval_round"],
        sources=result.get("sources", []),
        latency_seconds=latency,
    )



