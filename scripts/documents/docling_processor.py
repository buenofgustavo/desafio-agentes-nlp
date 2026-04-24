"""Coordenação do processamento usando Docling: lê de .md e salva JSON."""

from pathlib import Path
from typing import List, Dict, Any

from src.core.models import AneelRecord, ProcessedDocument, PdfDocument
from src.core.config import Constants
from scripts.documents.base_processor import BaseProcessor
from src.utils.logger import LoggingService
from src.utils.file_utils import FileManager

logger = LoggingService.setup_logger(__name__)

class DoclingProcessor(BaseProcessor):
    """Lê os arquivos Markdown gerados pelo Docling e salva no formato JSON padronizado da pipeline."""

    @classmethod
    def _extract_text_from_file(cls, file_path: Path, file_ext: str) -> str:
        """Lê o arquivo markdown correspondente a um documento simples."""
        md_file = file_path.parent / f"{file_path.stem}.md"
        if FileManager.verify_file_exists(md_file):
            with open(md_file, "r", encoding="utf-8") as f:
                return f.read()
        return ""

    @classmethod
    def _extract_text_from_archive(cls, archive_path: Path, file_ext: str) -> str:
        """Busca a pasta criada correspondente para pegar todos os arquivos Markdown lá dentro."""
        archive_md_dir = archive_path.parent / archive_path.stem
        if not FileManager.verify_folder_exists(archive_md_dir):
            return ""
        
        text_parts = []
        for md_file in sorted(archive_md_dir.rglob("*.md")):
            with open(md_file, "r", encoding="utf-8") as f:
                file_text = f.read()
                if file_text and file_text.strip():
                    text_parts.append(f"--- Texto extraído de {md_file.name} ---\n{file_text}")
        
        return "\n\n".join(text_parts)

    @classmethod
    def _process_single_document_task(cls, pdf: PdfDocument, record: AneelRecord, input_dir: Path, output_dir: Path) -> Dict[str, Any]:
        """Processa um único documento pegando o Markdown correspondente gerado no modo Docling."""

        result = {"status": "failed", "pdf_arquivo": pdf.arquivo, "error": None}

        try:
            output_path = cls._output_json_path(output_dir, pdf.arquivo)

            if output_path.exists():
                result["status"] = "skipped_existing"
                return result

            file_path = input_dir / pdf.arquivo
            document_text = cls._process_single_file(file_path)

            if not document_text or not document_text.strip():
                # Falha por não achar os markdowns referenciados ou estarem vazios
                result["status"] = "failed"
                result["error"] = "Nenhum texto encontrado nos Markdowns"
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
        input_dir: Path = Constants.DOCLING_MARKDOWN_DIR,
        output_dir: Path = Constants.PROCESSED_DATA_DIR,
    ) -> None:
        """Processa documentos pegando de markdowns com base nos recordes."""
        super().process_all_documents(records, input_dir, output_dir)
