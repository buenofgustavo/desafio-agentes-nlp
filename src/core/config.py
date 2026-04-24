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
	EMBEDDINGS_DIR = DATA_DIR / "embeddings"
	GCP_BUCKET_PROCESSED_JSON_PATH = "gs://aneel-raw-data/processed-json/"

	OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
	OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
	ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")