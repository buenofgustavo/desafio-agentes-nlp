import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

class Constants:
	"""Classe para armazenar as constantes do projeto, como diretórios e comandos."""
	BASE_DIR = Path(__file__).resolve().parent.parent.parent
	DATA_DIR = BASE_DIR / "data"
	RAW_DATA_DIR = DATA_DIR / "raw"
	LOGS_DIR = BASE_DIR / "logs"

	JSON_DIR = RAW_DATA_DIR / "json"
	DOCUMENTS_DIR = RAW_DATA_DIR / "documents"
	DOCLING_MARKDOWN_DIR = RAW_DATA_DIR / "docling_markdown"
	PROCESSED_DATA_DIR = DATA_DIR / "processed"

	OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
	OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
	ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

	# ─ Retrieval ──────────────────────────────────────────────────
	QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
	QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "setor_eletrico")
	EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
	RERANKER_MODEL: str = os.getenv(
		"RERANKER_MODEL",
		"cross-encoder/ms-marco-multilingual-rerank-mmarco-v2",
	)
	BM25_INDEX_PATH: Path = BASE_DIR / os.getenv(
		"BM25_INDEX_PATH", "data/processed/bm25_index.pkl"
	)
	BM25_TOP_K: int = int(os.getenv("BM25_TOP_K", "20"))
	DENSE_TOP_K: int = int(os.getenv("DENSE_TOP_K", "20"))
	RRF_K: int = int(os.getenv("RRF_K", "60"))
	HYBRID_FINAL_TOP_K: int = int(os.getenv("HYBRID_FINAL_TOP_K", "10"))
	RERANKER_TOP_K: int = int(os.getenv("RERANKER_TOP_K", "5"))