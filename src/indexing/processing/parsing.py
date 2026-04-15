"""Responsável pela extração de texto de documentos"""

from pdf2image import convert_from_path
import pytesseract
from pathlib import Path
from typing import List, Dict
import glob
import os
import re
from src.core.config import TESSERACT_CMD
from src.utils.logger import LoggingService

logger = LoggingService.setup_logger(__name__)

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

def load_documents(directory: Path) -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []
    supported = {".pdf"}

    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in supported:
            continue
        text = extract_document_text(path)
        if not text:
            continue
        docs.append({"source": str(path), "text": normalize_whitespace(text)})

    return docs


def extract_document_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8").strip()
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    return ""


def extract_text_from_pdf(path: Path) -> str:
    pages = convert_from_path(path, 500)
    text = ""
    for imgBlob in pages:
        text += pytesseract.image_to_string(imgBlob,lang='eng')
        ## passar texto e pdf para llm
    return text

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

