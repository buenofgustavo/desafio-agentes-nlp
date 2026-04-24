"""Responsável por orquestrar o processo de ingestão dos documentos, desde a extração do texto até a geração dos chunks e embeddings."""
import time
from src.core.config import Constants
from src.utils.logger import LoggingService
from src.utils.processed_store import load_all_processed
from src.indexing.parser.parser import parser
from src.indexing.chunker.chunker import DocumentChunker
from src.ai.embeddings.embedder import embed_chunks

logger = LoggingService.setup_logger(__name__)

class DocumentPipeline:

    @staticmethod
    def execute() -> None:
        """Executa o pipeline completo de ingestão."""
        logger.info("Iniciando pipeline de ingestão.")

        # parser()

        t0 = time.time()
        documents = load_all_processed()
        t1 = time.time()
        logger.info(f"Documentos carregados: {len(documents)}")
        logger.info(f"[load_documents] {t1 - t0:.2f}s")

        if not documents:
            logger.warning("Nenhum documento encontrado.")
            return

        t2 = time.time()
        chunks = DocumentChunker().run(documents)
        t3 = time.time()

        logger.info(f"Total de chunks gerados: {len(chunks)}")
        logger.info(f"[chunking] {t3 - t2:.2f}s")
        print(chunks[:1])
        t4 = time.time()
        embeddings = embed_chunks(chunks)
        t5 = time.time()
        logger.info(f"[embeddings] {t5 - t4:.2f}s")
        print(embeddings[:1])
        
        logger.info(f"[total_pipeline] {t3 - t0:.2f}s")


if __name__ == "__main__":
    DocumentPipeline.execute()