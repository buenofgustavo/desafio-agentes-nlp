import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

class Constants:
	"""Classe para armazenar as constantes do projeto, como diretórios e comandos."""
	BASE_DIR = Path(__file__).resolve().parent.parent
	DATA_DIR = BASE_DIR / "data"
	RAW_DATA_DIR = DATA_DIR / "raw"
	
	JSON_DIR = RAW_DATA_DIR / "json"
	PDFS_DIR = RAW_DATA_DIR / "pdfs"

	PROCESSED_DATA_DIR = DATA_DIR / "processed"

	DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", str(BASE_DIR / "data" / "documents")))