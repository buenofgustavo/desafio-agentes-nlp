"""Coordenação do processamento de documentos: extração de texto e persistência como JSON."""

import json
import tempfile
from pathlib import Path
from typing import List, Optional, Set

from src.core.models import AneelRecord, PdfDocument, ProcessedDocument
from src.core.config import Constants
from src.indexing.processing.extractor.pdf_extractor import PdfExtractor
from src.indexing.processing.extractor.docx_extractor import DocxExtractor
from src.indexing.processing.extractor.spreadsheet_extractor import SpreadsheetExtractor
from src.indexing.processing.extractor.html_extractor import HtmlExtractor
from src.indexing.processing.extractor.zip_extractor import ZipExtractor
from src.indexing.processing.extractor.rar_extractor import RarExtractor
from src.utils.file_utils import FileManager
from src.utils.logger import LoggingService

logger = LoggingService.setup_logger(__name__)


class DocumentProcessor:
    """Processa documentos baixados: extrai texto e salva como JSON com metadados."""

    SUPPORTED_TEXT_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xlsm", ".htm"}
    ARCHIVE_EXTENSIONS = {".zip", ".rar"}
    ALL_SUPPORTED_EXTENSIONS = SUPPORTED_TEXT_EXTENSIONS | ARCHIVE_EXTENSIONS

    # ── Extração de texto ──────────────────────────────────────────

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
            for inner_file in sorted(extract_dir.rglob("*")):
                if not inner_file.is_file():
                    continue

                inner_ext = cls._resolve_extension(inner_file)
                if inner_ext not in cls.SUPPORTED_TEXT_EXTENSIONS:
                    continue

                inner_text = cls._extract_text_from_file(inner_file, inner_ext)
                if inner_text and inner_text.strip():
                    text_parts.append(f"--- Texto extraído de {inner_file.name} ---\n{inner_text}")

            return "\n\n".join(text_parts)

        except Exception as e:
            logger.error(f"Erro ao processar arquivo compactado '{archive_path.name}': {e}")
            return ""
        finally:
            # Limpa a pasta de extração temporária
            FileManager.remove_path(extract_dir)

    # ── Resolução de extensão ──────────────────────────────────────

    @classmethod
    def _resolve_extension(cls, file_path: Path) -> str:
        """Determina a extensão do arquivo, usando heurística se necessário."""
        ext = FileManager.get_file_extension(file_path)

        if ext and ext in cls.ALL_SUPPORTED_EXTENSIONS:
            return ext

        # Tenta adivinhar a extensão pelo conteúdo do arquivo
        guessed = FileManager.guess_file_extension(file_path)
        if guessed and guessed in cls.ALL_SUPPORTED_EXTENSIONS:
            logger.debug(f"Extensão do arquivo '{file_path.name}' detectada como '{guessed}' via análise de conteúdo.")
            return guessed

        return ext or ""

    # ── Persistência ───────────────────────────────────────────────

    @staticmethod
    def _output_json_path(output_dir: Path, filename: str) -> Path:
        """Gera o caminho do JSON de saída a partir do nome do arquivo original."""
        stem = Path(filename).stem
        return output_dir / f"{stem}.json"

    @staticmethod
    def _save_document(doc: ProcessedDocument, output_path: Path) -> None:
        """Serializa e salva o ProcessedDocument como JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(doc.to_dict(), f, ensure_ascii=False, indent=2)

    # ── Processamento individual ───────────────────────────────────

    @classmethod
    def _process_single_file(cls, file_path: Path) -> str:
        """Extrai texto de um único arquivo (texto direto ou arquivo compactado).
        
        Retorna o texto extraído ou string vazia em caso de falha.
        """
        ext = cls._resolve_extension(file_path)

        if not ext:
            logger.warning(f"Não foi possível determinar a extensão de: {file_path.name}")
            return ""

        if ext in cls.ARCHIVE_EXTENSIONS:
            return cls._extract_text_from_archive(file_path, ext)
        elif ext in cls.SUPPORTED_TEXT_EXTENSIONS:
            return cls._extract_text_from_file(file_path, ext)
        else:
            logger.warning(f"Extensão '{ext}' não suportada: {file_path.name}")
            return ""

    # ── Processamento em lote ──────────────────────────────────────

    @classmethod
    def process_all(
        cls,
        records: List[AneelRecord],
        documents_dir: Path = Constants.DOCUMENTS_DIR,
        output_dir: Path = Constants.PROCESSED_DATA_DIR,
    ) -> None:
        """Processa todos os documentos em dois passes:
        
        1. Arquivos vinculados a registros (com metadados completos)
        2. Arquivos órfãos no diretório (sem metadados)
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        stats = {
            "success": 0,
            "skipped_existing": 0,
            "skipped_not_found": 0,
            "failed": 0,
            "orphans_processed": 0,
        }

        # ── Pass 1: Arquivos vinculados a registros ────────────────
        processed_filenames: Set[str] = set()

        total_pdfs = sum(len(record.pdfs) for record in records)
        logger.info(f"Pass 1: Processando {total_pdfs} documentos vinculados a {len(records)} registros.")

        processed_count = 0
        for record in records:
            for pdf in record.pdfs:
                processed_count += 1
                processed_filenames.add(pdf.arquivo)

                output_path = cls._output_json_path(output_dir, pdf.arquivo)

                if output_path.exists():
                    stats["skipped_existing"] += 1
                    continue

                file_path = documents_dir / pdf.arquivo

                if not FileManager.verify_file_exists(file_path):
                    logger.warning(f"Arquivo não encontrado no disco: {pdf.arquivo}")
                    stats["skipped_not_found"] += 1
                    continue

                if not FileManager.verify_file_has_content(file_path):
                    logger.warning(f"Arquivo vazio (0 bytes) ignorado: {pdf.arquivo}")
                    doc = ProcessedDocument.from_extraction(pdf, record, erro="Arquivo vazio (0 bytes)")
                    cls._save_document(doc, output_path)
                    stats["failed"] += 1
                    continue

                try:
                    text = cls._process_single_file(file_path)

                    if not text or not text.strip():
                        doc = ProcessedDocument.from_extraction(pdf, record, erro="Nenhum texto extraído")
                        cls._save_document(doc, output_path)
                        stats["failed"] += 1
                    else:
                        doc = ProcessedDocument.from_extraction(pdf, record, text=text)
                        cls._save_document(doc, output_path)
                        stats["success"] += 1

                except Exception as e:
                    logger.error(f"Erro inesperado ao processar '{pdf.arquivo}': {e}")
                    doc = ProcessedDocument.from_extraction(pdf, record, erro=str(e))
                    cls._save_document(doc, output_path)
                    stats["failed"] += 1

                if processed_count % 500 == 0:
                    logger.info(
                        f"Progresso Pass 1: {processed_count}/{total_pdfs} "
                        f"({100 * processed_count / total_pdfs:.1f}%) | "
                        f"Sucesso={stats['success']}, Falhas={stats['failed']}, "
                        f"Ignorados={stats['skipped_existing']}"
                    )

        logger.info(
            f"Pass 1 concluído. Sucesso={stats['success']}, "
            f"Falhas={stats['failed']}, Já existiam={stats['skipped_existing']}, "
            f"Não encontrados={stats['skipped_not_found']}"
        )

        # ── Pass 2: Arquivos órfãos ────────────────────────────────
        logger.info("Pass 2: Verificando arquivos órfãos no diretório de documentos.")

        orphan_count = 0
        for file_path in sorted(documents_dir.iterdir()):
            if not file_path.is_file():
                continue

            if file_path.name in processed_filenames:
                continue

            output_path = cls._output_json_path(output_dir, file_path.name)
            if output_path.exists():
                stats["skipped_existing"] += 1
                continue

            if not FileManager.verify_file_has_content(file_path):
                continue

            try:
                text = cls._process_single_file(file_path)

                if not text or not text.strip():
                    doc = ProcessedDocument(
                        arquivo_origem=file_path.name,
                        texto="",
                        titulo=None, autor=None, material=None,
                        esfera=None, situacao=None, assinatura=None,
                        publicacao=None, assunto=None, ementa=None,
                        sucesso=False,
                        erro_mensagem="Nenhum texto extraído (arquivo órfão)",
                    )
                else:
                    doc = ProcessedDocument(
                        arquivo_origem=file_path.name,
                        texto=text,
                        titulo=None, autor=None, material=None,
                        esfera=None, situacao=None, assinatura=None,
                        publicacao=None, assunto=None, ementa=None,
                    )
                    stats["orphans_processed"] += 1

                cls._save_document(doc, output_path)
                orphan_count += 1

            except Exception as e:
                logger.error(f"Erro ao processar arquivo órfão '{file_path.name}': {e}")
                stats["failed"] += 1

        logger.info(
            f"Pass 2 concluído. Órfãos processados={stats['orphans_processed']}, "
            f"total de órfãos encontrados={orphan_count}"
        )

        # ── Resumo final ───────────────────────────────────────────
        logger.info(
            f"Processamento finalizado. "
            f"Sucesso={stats['success']}, "
            f"Órfãos={stats['orphans_processed']}, "
            f"Falhas={stats['failed']}, "
            f"Já existiam={stats['skipped_existing']}, "
            f"Não encontrados={stats['skipped_not_found']}"
        )
