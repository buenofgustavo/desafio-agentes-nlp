"""Extração de texto de arquivos PDF utilizando múltiplas estratégias (PyMuPDF, pdfplumber, OCR)."""

import fitz  # type: ignore # PyMuPDF
import pdfplumber
import pytesseract  # type: ignore
from pdf2image import convert_from_path
from pathlib import Path

from src.utils.logger import LoggingService
from src.utils.file_utils import FileManager

logger = LoggingService.setup_logger(__name__)


class PdfExtractor:
    """Classe responsável por extrair texto de arquivos PDF.
    
    Estratégia:
    1. PyMuPDF (fitz) como extrator principal
    2. pdfplumber como fallback (melhor para PDFs com tabelas complexas)
    3. OCR (Tesseract) como último recurso para PDFs escaneados
    """

    @staticmethod
    def _extract_with_pymupdf(pdf_path: Path) -> str:
        """Extrai texto usando PyMuPDF (fitz)."""
        try:
            doc = fitz.open(pdf_path)
            pages_text = []
            for page in doc:
                text = page.get_text("text")
                if text and text.strip():
                    pages_text.append(text)
            doc.close()

            if pages_text:
                full_text = "\n".join(pages_text)
                logger.debug(f"Texto extraído com sucesso via PyMuPDF ({len(pages_text)} páginas)")
                return full_text
        except fitz.FileDataError as e:
            logger.error(f"Arquivo não é um PDF válido '{pdf_path.name}': {e}")
            raise
        except Exception as e:
            logger.warning(f"Erro ao extrair com PyMuPDF '{pdf_path.name}': {e}")

        return ""

    @staticmethod
    def _extract_tables_with_pdfplumber(pdf_path: Path) -> str:
        """Extrai tabelas estruturadas usando pdfplumber e formata como texto pipe-delimited."""
        try:
            tables_text = []
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            for row in table:
                                if row:
                                    cells = [str(cell).strip() if cell else "" for cell in row]
                                    cells = [c for c in cells if c]
                                    if cells:
                                        tables_text.append(" | ".join(cells))

            if tables_text:
                logger.debug(f"Tabelas extraídas com pdfplumber ({len(tables_text)} linhas)")
                return "\n".join(tables_text)
        except Exception as e:
            logger.error(f"Erro ao extrair tabelas com pdfplumber '{pdf_path.name}': {e}")

        return ""

    @staticmethod
    def _extract_with_pdfplumber_text(pdf_path: Path) -> str:
        """Extrai texto puro usando pdfplumber como fallback."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = "\n".join(
                    page.extract_text() for page in pdf.pages if page.extract_text()
                )
                if text.strip():
                    logger.debug("Texto extraído com sucesso via pdfplumber (fallback texto)")
                    return text
        except Exception as e:
            logger.error(f"Erro ao extrair texto com pdfplumber '{pdf_path.name}': {e}")

        return ""

    @staticmethod
    def _extract_with_ocr(pdf_path: Path) -> str:
        """Extrai texto via OCR (Tesseract)."""
        try:
            images = convert_from_path(pdf_path)
            ocr_text = []

            for image in images:
                image_text = pytesseract.image_to_string(image, lang='por')
                if image_text and image_text.strip():
                    ocr_text.append(image_text)

            full_ocr_text = "\n".join(ocr_text).strip()
            if full_ocr_text:
                logger.debug(f"Texto extraído com sucesso via OCR ({len(ocr_text)} páginas)")
                return full_ocr_text

            logger.warning(f"Nenhum texto extraído mesmo com OCR!")
            return ""

        except Exception as e:
            logger.error(f"Erro na extração via OCR: {e}")
            return ""

    @staticmethod
    def _verify_pdf_is_corrupted(pdf_path: Path) -> bool:
        """Verifica se o arquivo PDF está corrompido."""
        try:
            doc = fitz.open(pdf_path)
            is_valid = doc.is_pdf and doc.page_count > 0
            doc.close()
            
            if not is_valid:
                logger.warning(f"PDF '{pdf_path.name}' parece estar vazio ou sem páginas extraíveis. Abortando extração.")
                return True
            
            return False
        except Exception as e:
            logger.warning(f"Falha ao validar estruturalmente o PDF '{pdf_path.name}' cruzado com PyMuPDF. Abortando extração. Erro: {e}")
            return True

    @classmethod
    def extract_text_from_pdf(cls, pdf_path: Path) -> str:
        """Extrai texto de um arquivo PDF usando estratégia em cascata.
        
        Retorna o texto extraído ou uma string vazia em caso de falha.
        """
        if not FileManager.verify_file_exists(pdf_path):
            logger.warning(f"Arquivo PDF não encontrado: {pdf_path}")
            return ""

        if not FileManager.verify_file_has_content(pdf_path):
            logger.warning(f"Arquivo vazio (0 bytes) ignorado: {pdf_path.name}")
            return ""

        logger.debug(f"Começando extração de arquivo PDF: {pdf_path.name}")

        if cls._verify_pdf_is_corrupted(pdf_path):
            return ""

        is_pdf_broken = False

        # 1. PyMuPDF como extrator principal
        try:
            text = cls._extract_with_pymupdf(pdf_path)
            if text:
                tables_text = cls._extract_tables_with_pdfplumber(pdf_path)
                if tables_text:
                    text += "\n\n--- TABELAS EXTRAÍDAS ---\n" + tables_text
                return text
        except Exception:
            is_pdf_broken = True

        if is_pdf_broken:
            logger.warning(f"PDF '{pdf_path.name}' sofreu falha grave de parsing. Abortando fallbacks.")
            return ""

        # 2. pdfplumber como fallback para texto
        text = cls._extract_with_pdfplumber_text(pdf_path)
        if text:
            return text

        # 3. OCR como último recurso
        return cls._extract_with_ocr(pdf_path)