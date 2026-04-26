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

	# Busca e Recuperação (Retrieval)
	QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
	QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "setor_eletrico")
	EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
	RERANKER_MODEL: str = os.getenv(
		"RERANKER_MODEL",
		"cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
	)
	BM25_INDEX_PATH: Path = BASE_DIR / os.getenv(
		"BM25_INDEX_PATH", "data/retrieval/bm25_index"
	)
	BM25_TOP_K: int = int(os.getenv("BM25_TOP_K", "20"))
	DENSE_TOP_K: int = int(os.getenv("DENSE_TOP_K", "20"))
	RRF_K: int = int(os.getenv("RRF_K", "60"))
	HYBRID_FINAL_TOP_K: int = int(os.getenv("HYBRID_FINAL_TOP_K", "10"))
	RERANKER_TOP_K: int = int(os.getenv("RERANKER_TOP_K", "5"))

	# Agente / LLM
	CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5")
	LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))
	LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))

	# Comportamento do Agente
	MAX_FAITHFULNESS_RETRIES: int = int(os.getenv("MAX_FAITHFULNESS_RETRIES", "2"))
	CONTEXT_MAX_TOKENS: int = int(os.getenv("CONTEXT_MAX_TOKENS", "6000"))
	MULTIHOP_MAX_ROUNDS: int = int(os.getenv("MULTIHOP_MAX_ROUNDS", "3"))
	HYDE_ENABLED: bool = os.getenv("HYDE_ENABLED", "true").lower() == "true"
	QUERY_REFORMULATIONS: int = int(os.getenv("QUERY_REFORMULATIONS", "2"))
	
	# API
	API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
	API_PORT: int = int(os.getenv("API_PORT", "8000"))
	EVALUATION_REPORT_PATH: Path = DATA_DIR / os.getenv(
		"EVALUATION_REPORT_PATH", "evaluation/final_report.json"
	)
