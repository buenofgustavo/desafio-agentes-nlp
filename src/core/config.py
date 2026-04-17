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
