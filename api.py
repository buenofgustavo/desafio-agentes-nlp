from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from src.ai.embeddings.embedder import get_embedding_model
from src.ai.llm.factory import get_llm
from src.indexing.storage.vector_store import get_qdrant_client
from src.retrieval.rag_pipeline import RAGService

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 🚀 startup
    print("--------------Iniciando API...--------------")

    # warmup (opcional)
    app.state.llm = get_llm("ollama", "llama3")
    app.state.qdrant = get_qdrant_client()
    app.state.embedder = get_embedding_model()
    app.state.rag = RAGService(app)
    yield

    # 🧹 shutdown
    print("--------------Encerrando API...--------------")

app = FastAPI(title="RAG API", lifespan=lifespan)

class ChatRequest(BaseModel):
    pergunta: str = Field(..., min_length=2)
    top_k: int = Field(default=3, ge=1, le=10)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    try:
        resposta, contextos = app.state.rag.answer(question=req.pergunta, top_k=req.top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Falha no pipeline RAG: {exc}"
        ) from exc

    return {
        "pergunta": req.pergunta,
        "resposta": resposta,
        "fontes": [
            {
                "titulo": c.titulo,
                "arquivo": c.source_file,
                "pagina": c.page,
                "score": c.score
            }
            for c in contextos
        ],
    }
