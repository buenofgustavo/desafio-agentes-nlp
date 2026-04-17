"""Coordenação do processamento de documentos: extração de texto e persistência como JSON."""

import os
import json
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
import multiprocessing as mp
import dataclasses

from src.core.models import AneelRecord, ProcessedDocument, PdfDocument
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
    def _resolve_extension(cls, file_path: Path) -> str:
        """Determina a extensão do arquivo, usando heurística se necessário."""
        ext = FileManager.get_file_extension(file_path)

        if ext and ext in cls.ALL_SUPPORTED_EXTENSIONS:
            return ext

        guessed = FileManager.guess_file_extension(file_path)
        if guessed and guessed in cls.ALL_SUPPORTED_EXTENSIONS:
            logger.debug(f"Extensão do arquivo '{file_path.name}' detectada como '{guessed}' via análise de conteúdo.")
            return guessed

        return ext or ""

    @classmethod
    def _output_json_path(cls, output_dir: Path, filename: str) -> Path:
        """Gera o caminho do JSON de saída a partir do nome do arquivo original."""
        stem = Path(filename).stem.strip()
        ext = cls._resolve_extension(Path(filename))
        return output_dir / f"{stem}_{ext.replace('.', '')}.json"

    @staticmethod
    def _save_document(doc: ProcessedDocument, output_path: Path) -> None:
        """Serializa e salva o ProcessedDocument como JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(doc.to_dict(), f, ensure_ascii=False, indent=2)

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
        
    @classmethod
    def _show_progress(cls, processed_count: int, total_documents: int, stats: Dict[str, int]) -> None:
        """Exibe o progresso do processamento no console."""
        logger.info(
            f"\n--------------------------------\n"
            f"Progresso: {processed_count}/{total_documents}\n"
            f"({100 * processed_count / total_documents:.1f}%)\n"
            f"Sucesso={stats['success']}, Falhas={stats['failed']}\n"
            f"Já existiam={stats['skipped_existing']}, Não encontrados={stats['skipped_not_found']}\n"
            f"--------------------------------"
        )

    @classmethod
    def _process_single_document_task(cls, pdf: PdfDocument, record: AneelRecord, documents_dir: Path, output_dir: Path) -> Dict[str, Any]:
        """Processa um único documento para ser executado no ProcessPoolExecutor."""

        result = {"status": "failed", "pdf_arquivo": pdf.arquivo, "error": None}

        try:
            output_path = cls._output_json_path(output_dir, pdf.arquivo)

            if output_path.exists():
                result["status"] = "skipped_existing"
                return result

            file_path = documents_dir / pdf.arquivo

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
        documents_dir: Path = Constants.DOCUMENTS_DIR,
        output_dir: Path = Constants.PROCESSED_DATA_DIR,
    ) -> None:
        """Processa todos os documentos salvos nos registros usando ProcessPoolExecutor."""

        output_dir.mkdir(parents=True, exist_ok=True)

        stats = {
            "success": 0,
            "skipped_existing": 0,
            "skipped_not_found": 0,
            "failed": 0,
        }

        total_documents = sum(len(record.pdfs) for record in records)
        max_workers = 2
        logger.info(f"Processando {total_documents} documentos usando {max_workers} processos locais (spawn context).")

        def task_generator():
            for record in records:
                # Remove as referências iterativas de pdfs para não sobrecarregar o IPC pickling.
                slim_record = dataclasses.replace(record, pdfs=[])
                for pdf in record.pdfs:
                    yield (pdf, slim_record, documents_dir, output_dir)

        task_iter = task_generator()
        processed_count = 0
        
        # O uso de 'spawn' previne a duplicação de todo o uso de RAM (Copy-On-Write no Linux) 
        # que estava ocorrendo durante a execução do fork no multiprocessamento.
        ctx = mp.get_context("spawn")
        
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            with tqdm(total=total_documents, desc="Extraindo Documentos", unit="docs") as pbar:
                active_futures = {}
                
                # Preenche a fila inicial de processamento
                for _ in range(max_workers * 2):
                    try:
                        params = next(task_iter)
                        future = executor.submit(cls._process_single_document_task, *params)
                        active_futures[future] = params[0].arquivo
                    except StopIteration:
                        break

                # Bounding dinâmico para garantir paralelismo sem encher a memória do Master Queue
                while active_futures:
                    done, _ = wait(active_futures.keys(), return_when=FIRST_COMPLETED)
                    
                    for future in done:
                        pdf_arquivo = active_futures.pop(future)
                        processed_count += 1
                        try:
                            result = future.result()
                            status = result.get("status", "failed")
                            
                            if status not in stats:
                                stats[status] = 0
                            stats[status] += 1
                            
                            if status == "failed" and result.get("error"):
                                logger.warning(f"Falha ao processar '{pdf_arquivo}': {result['error']}")

                            if processed_count % 200 == 0:
                                cls._show_progress(processed_count, total_documents, stats)
                                
                        except Exception as e:
                            logger.error(f"Erro inesperado no processo paralelo para '{pdf_arquivo}': {e}")
                            stats["failed"] += 1
                        finally:
                            pbar.update(1)
                            
                        # Submete a próxima tarefa assim que uma for concluída
                        try:
                            params = next(task_iter)
                            new_future = executor.submit(cls._process_single_document_task, *params)
                            active_futures[new_future] = params[0].arquivo
                        except StopIteration:
                            pass

        logger.info("Processamento concluído...")
        cls._show_progress(processed_count, total_documents, stats)