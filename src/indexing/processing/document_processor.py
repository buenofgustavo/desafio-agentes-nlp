"""Coordenação do processamento de documentos originais: extração de texto via PDF/Docx/etc e persistência como JSON."""

from pathlib import Path
from typing import List, Dict, Any

from src.core.models import AneelRecord, ProcessedDocument, PdfDocument
from src.core.config import Constants
from src.indexing.processing.base_processor import BaseProcessor
from src.indexing.processing.extractor.pdf_extractor import PdfExtractor
from src.indexing.processing.extractor.docx_extractor import DocxExtractor
from src.indexing.processing.extractor.spreadsheet_extractor import SpreadsheetExtractor
from src.indexing.processing.extractor.html_extractor import HtmlExtractor
from src.indexing.processing.extractor.zip_extractor import ZipExtractor
from src.indexing.processing.extractor.rar_extractor import RarExtractor
from src.utils.file_utils import FileManager
from src.utils.logger import LoggingService

logger = LoggingService.setup_logger(__name__)

class DocumentProcessor(BaseProcessor):
    """Processa documentos baixados: extrai texto usando ferramentas específicas e salva como JSON com metadados."""

    @classmethod
    def _extract_text_from_file(cls, file_path: Path, file_ext: str) -> str:
        """Extrai texto bruto de um arquivo baseado na sua extensão."""
        match file_ext:
            case ".pdf":
                return PdfExtractor.extract_text_from_pdf(file_path)
            case ".docx":
                return DocxExtractor.extract_text_from_docx(file_path)
            case ".xlsx" | ".xlsm":
                return SpreadsheetExtractor.extract_text_from_spreadsheet(file_path, file_extension=file_ext)
            case ".htm":
                return HtmlExtractor.extract_text_from_html(file_path)
            case _:
                logger.warning(f"Extensão não suportada para extração: {file_ext} ({file_path.name})")
                return ""

    @classmethod
    def _extract_text_from_archive(cls, archive_path: Path, file_ext: str) -> str:
        """Extrai um ZIP/RAR para pasta temporária e concatena o texto de todos os arquivos internos."""
        extract_dir = archive_path.parent / f"_extracted_{archive_path.stem}"
        extract_dir.mkdir(parents=True, exist_ok=True)

        try:
            if file_ext == ".zip":
                ZipExtractor.extract_zip_file(archive_path, extract_dir)
            elif file_ext == ".rar":
                RarExtractor.extract_rar_file(archive_path, extract_dir)

            text_parts = []
            for file_path in sorted(extract_dir.rglob("*")):
                if not FileManager.verify_file_exists(file_path):
                    continue

                ext = cls._resolve_extension(file_path)
                if ext not in cls.SUPPORTED_TEXT_EXTENSIONS:
                    continue

                file_text = cls._extract_text_from_file(file_path, ext)
                if file_text and file_text.strip():
                    text_parts.append(f"--- Texto extraído de {file_path.name} ---\n{file_text}")

            return "\n\n".join(text_parts)

        except Exception as e:
            logger.error(f"Erro ao processar arquivo compactado '{archive_path.name}': {e}")
            return ""
        finally:
            FileManager.remove_path(extract_dir)

    @classmethod
    def _process_single_document_task(cls, pdf: PdfDocument, record: AneelRecord, input_dir: Path, output_dir: Path) -> Dict[str, Any]:
        """Processa um único documento original e os extrai usando o DocumentProcessor."""

        result = {"status": "failed", "pdf_arquivo": pdf.arquivo, "error": None}

        try:
            output_path = cls._output_json_path(output_dir, pdf.arquivo)

            if output_path.exists():
                result["status"] = "skipped_existing"
                return result

            file_path = input_dir / pdf.arquivo

            if not FileManager.verify_file_exists(file_path):
                result["status"] = "skipped_not_found"
                return result

            if not FileManager.verify_file_has_content(file_path):
                result["status"] = "failed"
                result["error"] = "Arquivo vazio (0 bytes)"
                return result

            document_text = cls._process_single_file(file_path)

            if not document_text or not document_text.strip():
                result["status"] = "failed"
                result["error"] = "Nenhum texto extraído"
            else:
                doc = ProcessedDocument.from_extraction(pdf, record, document_text=document_text)
                cls._save_document(doc, output_path)
                result["status"] = "success"
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)

        return result

    @classmethod
    def process_all_documents(
        cls,
        records: List[AneelRecord],
        input_dir: Path = Constants.DOCUMENTS_DIR,
        output_dir: Path = Constants.PROCESSED_DATA_DIR,
    ) -> None:
        """Processa todos os documentos originais."""
        super().process_all_documents(records, input_dir, output_dir)