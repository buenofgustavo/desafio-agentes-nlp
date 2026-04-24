"""FastAPI REST API for the RAG agent.

Exposes three endpoints:
    GET  /health   — system status including Qdrant connectivity
    POST /query    — runs the full LangGraph agent and returns a structured response
    GET  /metrics  — returns the latest evaluation report if available

Architecture notes:
    - RetrievalPipeline and agent_graph are initialized ONCE at startup via
      FastAPI lifespan events and stored in ``app.state``.
    - The /query handler runs the synchronous agent in a ThreadPoolExecutor
      to avoid blocking the async event loop.
    - Every response carries a ``X-Request-ID`` header for traceability.
    - No authentication — this is a demo.
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

# ── Pydantic models ────────────────────────────────────────────────────────


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


# ── Lifespan: initialize singletons once ──────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize heavy components once at startup; clean up at shutdown."""
    logger.info("API: inicializando componentes (lifespan startup)…")

    # RetrievalPipeline — loads embedding model + BM25 + cross-encoder
    app.state.retrieval_pipeline = RetrievalPipeline()

    # ThreadPoolExecutor — used to run the synchronous agent off the event loop
    app.state.executor = ThreadPoolExecutor(max_workers=4)

    # agent_graph is a module-level singleton imported from graph.py
    app.state.agent_graph = agent_graph

    logger.info("API: todos os componentes prontos.")
    yield

    # Shutdown
    app.state.executor.shutdown(wait=False)
    logger.info("API: shutdown concluído.")


# ── FastAPI application ────────────────────────────────────────────────────

app = FastAPI(
    title="RAG — Setor Elétrico Brasileiro",
    description="API de perguntas e respostas sobre documentos do setor elétrico brasileiro.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Middleware: X-Request-ID header ───────────────────────────────────────


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Attach a unique X-Request-ID to every response and log the request."""
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


# ── Helpers ───────────────────────────────────────────────────────────────


def _check_qdrant() -> str:
    """Return 'connected' or 'error: <detail>' by probing the Qdrant REST API."""
    try:
        with httpx.Client(timeout=3.0) as client:
            r = client.get(f"{Constants.QDRANT_URL}/collections")
            r.raise_for_status()
        return "connected"
    except Exception as exc:  # noqa: BLE001
        logger.warning("Qdrant health check falhou: %s", exc)
        return f"error: {exc}"


def _run_agent(question: str) -> dict[str, Any]:
    """Run the LangGraph agent synchronously and return a result dict.

    This function is called inside a ThreadPoolExecutor so it must not
    await anything.
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


@app.get("/health", summary="System health check")
async def health() -> dict:
    """Return system status.  Always returns 200 — Qdrant status is informative."""
    qdrant_status = _check_qdrant()
    bm25_loaded = (
        hasattr(app.state, "retrieval_pipeline")
        and app.state.retrieval_pipeline._bm25.is_built
    )
    return {
        "status": "ok",
        "qdrant": qdrant_status,
        "bm25_index": "loaded" if bm25_loaded else "not loaded (dense-only mode)",
        "model": Constants.CLAUDE_MODEL,
    }


@app.post("/query", response_model=QueryResponse, summary="Run RAG agent")
async def query(req: QueryRequest) -> QueryResponse:
    """Execute the full LangGraph agent for the given question.

    The agent runs in a thread pool to avoid blocking the async event loop.
    Returns HTTP 500 on agent failure.
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
    except Exception as exc:  # noqa: BLE001
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


@app.get("/metrics", summary="Latest evaluation metrics")
async def metrics() -> dict:
    """Return the latest evaluation report from disk if it exists."""
    report_path = Constants.EVALUATION_REPORT_PATH
    if report_path.exists():
        try:
            with report_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Falha ao ler relatório de avaliação: %s", exc)
            return {"status": "error reading report", "detail": str(exc)}
    return {"status": "evaluation not run"}
