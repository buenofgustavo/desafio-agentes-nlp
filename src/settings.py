import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", str(BASE_DIR / "data" / "documents")))
TESSERACT_CMD = os.getenv(
	"TESSERACT_CMD",
	r"C:\Program Files\Tesseract-OCR\tesseract.exe",
)