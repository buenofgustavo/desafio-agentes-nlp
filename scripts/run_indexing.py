"""Workflow para indexação dos documentos."""
import os
import sys
import uuid
import hashlib
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from typing import List
from dataclasses import asdict

# Configura path para importar módulos src
sys.path.append(str(Path(__file__).parent.parent))

from qdrant_client.models import PointStruct
from src.indexing.storage.processed_store import load_all_processed
from src.indexing.processing.chunker.chunker import DocumentChunker
from src.ai.embeddings.embedder import embed_chunks
from src.indexing.storage.vector_store import get_qdrant_client, create_collection
from src.utils.logger import LoggingService
from src.core.models import ChildChunk

logger = LoggingService.setup_logger(__name__)
load_dotenv()

COLLECTION = os.getenv('QDRANT_COLLECTION', 'setor_eletrico')
BATCH_SIZE = 64

def _generate_deterministic_id(chunk: ChildChunk) -> str:
    """Generate a deterministic UUID based on chunk location to prevent duplicate insertions."""
    unique_str = f"{chunk.source_file}_{chunk.page}_{chunk.parent_index}_{chunk.child_index}"
    return str(uuid.UUID(hashlib.md5(unique_str.encode("utf-8")).hexdigest()))

def run_indexing():
    logger.info('Criando coleção no Qdrant...')
    create_collection(COLLECTION)
    client = get_qdrant_client()

    logger.info('Carregando processed JSONs...')
    documents = load_all_processed()
    logger.info(f'{len(documents)} documentos carregados')
    
    chunker = DocumentChunker()
    
    chunk_buffer: List[ChildChunk] = []
    total_processed_chunks = 0
    
    def _flush_buffer():
        nonlocal total_processed_chunks
        if not chunk_buffer:
            return
            
        num_chunks = len(chunk_buffer)
        logger.info(f"Sincronizando buffer com {num_chunks} chunks no Qdrant...")
            
        for i in range(0, len(chunk_buffer), BATCH_SIZE):
            batch = chunk_buffer[i : i + BATCH_SIZE]
            texts = [c.text_to_embed for c in batch]
            
            # Embedding e upsert
            vectors = embed_chunks(texts)
            
            points = []
            for chunk, vec in zip(batch, vectors):
                payload = asdict(chunk)
                points.append(
                    PointStruct(
                        id=_generate_deterministic_id(chunk),
                        vector=vec,
                        payload=payload
                    )
                )
            try:
                client.upsert(collection_name=COLLECTION, points=points)
                total_processed_chunks += len(points)
            except Exception as e:
                logger.error(f"Erro ao inserir lote no Qdrant: {e}")
                
        chunk_buffer.clear()
        logger.info(f"Buffer sincronizado. Total indexado: {total_processed_chunks} chunks.")

    logger.info('Iniciando processamento (Chunking + Indexação)...')
    for doc in tqdm(documents, desc='Processando documentos'):
        doc_dict = {
            "pages": [{"page": 1, "text": doc.texto_documento}]
        }
        
        meta_dict = {
            "source_file": doc.arquivo_origem,
            "titulo": doc.titulo,
            "autor": doc.autor,
            "material": doc.material,
            "esfera": doc.esfera,
            "situacao": doc.situacao,
            "assinatura": doc.assinatura,
            "publicacao": doc.publicacao,
            "assunto": doc.assunto,
            "ementa": doc.ementa
        }
        
        try:
            doc_chunks = chunker.chunk_document(doc_dict, meta=meta_dict, use_context=False)
            chunk_buffer.extend(doc_chunks)
            
            if len(chunk_buffer) >= BATCH_SIZE * 50:
                _flush_buffer()
                
        except Exception as e:
            logger.error(f"Erro ao processar documento {doc.arquivo_origem}: {e}")

    if chunk_buffer:
        _flush_buffer()

    total_in_db = client.count(collection_name=COLLECTION).count
    logger.info(f'Indexação concluída!')
    logger.info(f'Total de chunks processados nesta sessão: {total_processed_chunks}')
    logger.info(f'Total de vetores na coleção {COLLECTION}: {total_in_db}')

if __name__ == '__main__':
    run_indexing()
