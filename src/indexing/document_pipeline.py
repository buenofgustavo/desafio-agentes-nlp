"""Responsável por orquestrar o processo de ingestão dos documentos, desde a extração do texto até a geração dos chunks e embeddings."""
import json
import time
import os
from src.core.config import Constants
from src.indexing.storage.vector_store import create_collection, get_qdrant_client, upload_embeddings
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

        # t0 = time.time()
        # documents = load_all_processed()
        # t1 = time.time()
        # logger.info(f"Documentos carregados: {len(documents)}")
        # logger.info(f"[load_documents] {t1 - t0:.2f}s")

        # if not documents:
        #     logger.warning("Nenhum documento encontrado.")
        #     return

        # t2 = time.time()
        # chunks = DocumentChunker().run(documents)
        # t3 = time.time()
        # logger.info(f"Total de chunks gerados: {len(chunks)}")
        # logger.info(f"[chunking] {t3 - t2:.2f}s")
        # print(chunks[:1])

        # t4 = time.time()
        # chunks_filtrados = chunks[326656:]
        # embeddings = embed_chunks(chunks_filtrados)
        # t5 = time.time()
        # logger.info(f"[embeddings] {t5 - t4:.2f}s")

        # t6 = time.time()
        # collection_name = "desafio-agentes-nlp"
        # vector_size = get_vector_size(Constants.EMBEDDINGS_DIR)
        # print(f"Vector size: {vector_size}")
        client = get_qdrant_client()
        # create_collection(client, collection_name, vector_size=vector_size)

        # iterator = iter_embeddings(Constants.EMBEDDINGS_DIR)
        # total_inserted = upload_embeddings(client, collection_name, iterator)
        # logger.info(f"Total de embeddings inseridos: {total_inserted}")
        # t7 = time.time()
        info = client.get_collection("desafio-agentes-nlp")
        print(info.points_count)
        print(info.config)
        # logger.info(f"[total_pipeline] {t6 - t7:.2f}s")

def iter_embeddings(path):
    for file_name in os.listdir(path):
        if file_name.endswith(".jsonl"):
            with open(os.path.join(path, file_name), "r", encoding="utf-8") as f:
                for line in f:
                    yield json.loads(line)

def get_vector_size(path):
    for file_name in os.listdir(path):
        if file_name.endswith(".jsonl"):
            file_path = os.path.join(path, file_name)

            with open(file_path, "r", encoding="utf-8") as f:
                first_line = f.readline()
                if first_line:
                    data = json.loads(first_line)
                    return len(data["embedding"])

    raise ValueError("Nenhum embedding encontrado")

if __name__ == "__main__":
    DocumentPipeline.execute()