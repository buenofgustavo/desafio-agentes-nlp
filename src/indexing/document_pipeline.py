"""Responsável por orquestrar o processo de ingestão dos documentos, desde a extração do texto até a geração dos chunks e embeddings."""
import time
from src.indexing.ingestion.json_loader import JsonLoader
from src.indexing.ingestion.document_downloader import DocumentDownloader
from src.core.config import Constants
from src.utils.logger import LoggingService

logger = LoggingService.setup_logger(__name__)

class DocumentPipeline:
    """Pipeline de ingestão de documentos legislativos."""
    
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

if __name__ == "__main__":
    DocumentPipeline.execute()