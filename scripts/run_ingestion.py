"""Responsável por orquestrar o processo de ingestão dos documentos, desde o download dos documentos até o parsing desses documentos."""
import time
from typing import List
from src.indexing.ingestion.json_loader import JsonLoader
from src.indexing.ingestion.document_downloader import DocumentDownloader
from src.core.config import Constants
from src.core.models import AneelRecord
from src.utils.logger import LoggingService
from src.indexing.processing.document_processor import DocumentProcessor
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = LoggingService.setup_logger(__name__)

class DocumentPipeline:
    """Pipeline de ingestão de documentos legislativos."""
    @staticmethod
    def load_json_data() -> List[AneelRecord]:
        """Carrega os dados JSON dos registros."""
        logger.info(f"Carregando dados JSON do diretório: {Constants.JSON_DIR}")
        start_time = time.time()
        records = JsonLoader.load_json_folder_data(Constants.JSON_DIR)
        duration = time.time() - start_time
        logger.info(f"Extração de dados JSON concluída em {duration:.2f} segundos. Total de registros: {len(records)}")
        return records
    
    @staticmethod
    def run_downloading(records: List[AneelRecord]) -> None:
        """Executa o processo de download dos documentos."""
        logger.info(f"Iniciando processo de download de documentos.")
        start_time = time.time()
        
        with DocumentDownloader() as document_downloader:
            document_downloader.download_documents(records=records)
        
        duration = time.time() - start_time
        logger.info(f"Processo de download concluído em {duration/60:.2f} minutos.")

    @staticmethod
    def run_parsing(records: List[AneelRecord]) -> None:
        """Executa o processo de parsing dos documentos."""
        logger.info(f"Iniciando processo de parsing de {len(records)} registros.")
        start_time = time.time()
        
        DocumentProcessor.process_all_documents(records)
        
        duration = time.time() - start_time
        logger.info(f"Processo de parsing concluído em {duration:.2f} segundos.")

    @staticmethod
    def execute() -> None:
        """Executa o pipeline completo de ingestão."""
        logger.info("Iniciando pipeline de ingestão.")

        # 1. JSON load
        all_aneel_records = DocumentPipeline.load_json_data()
        
        # 2. Download dos documentos
        DocumentPipeline.run_downloading(all_aneel_records)
        
        # 3. Parsing dos documentos
        DocumentPipeline.run_parsing(all_aneel_records)

if __name__ == "__main__":
    DocumentPipeline.execute()