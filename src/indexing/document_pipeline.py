"""Responsável por orquestrar o processo de ingestão dos documentos, desde a extração do texto até a geração dos chunks e embeddings."""
import time
from typing import List
from src.indexing.ingestion.json_loader import JsonLoader
from src.indexing.ingestion.document_downloader import DocumentDownloader
from src.core.config import Constants
from src.core.models import AneelRecord
from src.utils.logger import LoggingService
from src.indexing.processing.document_processor import DocumentProcessor

logger = LoggingService.setup_logger(__name__)

class DocumentPipeline:
    """Pipeline de ingestão de documentos legislativos."""

    @staticmethod
    def run_parsing(records: List[AneelRecord]) -> None:
        """Executa o processo de parsing dos documentos."""
        logger.info(f"Iniciando processo de parsing de {len(records)} documentos.")
        start_time = time.time()
        
        DocumentProcessor.process_all(records)
        
        duration = time.time() - start_time
        logger.info(f"Processo de parsing concluído em {duration:.2f} segundos.")

    @staticmethod
    def execute() -> None:
        """Executa o pipeline completo de ingestão."""
        logger.info("Iniciando pipeline de ingestão.")

        start_time = time.time()
        all_aneel_records = JsonLoader.load_json_folder_data(Constants.JSON_DIR)
        duration = time.time() - start_time
        logger.info(f"Extração de dados JSON concluída em {duration:.2f} segundos. Total de registros: {len(all_aneel_records)}")
        
        start_time = time.time()
        with DocumentDownloader() as document_downloader:
            document_downloader.download_documents(records=all_aneel_records)
        duration = time.time() - start_time
        logger.info(f"Download de documentos concluído em {duration/60:.2f} minutos.")
        
        DocumentPipeline.run_parsing(all_aneel_records)

if __name__ == "__main__":
    DocumentPipeline.execute()