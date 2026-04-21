"""Workflow para indexação dos documentos."""
import os
import sys
import uuid
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# Configura path para importar módulos src
sys.path.append(str(Path(__file__).parent.parent))

from qdrant_client.models import PointStruct
from src.indexing.storage.processed_store import load_all_processed
from src.indexing.processing.chunker.chunker import DocumentChunker
from src.ai.embeddings.embedder import embed_chunks
from src.indexing.storage.vector_store import get_client, create_collection
from src.utils.logger import LoggingService

logger = LoggingService.setup_logger(__name__)
load_dotenv()

COLLECTION = os.getenv('QDRANT_COLLECTION', 'setor_eletrico')
BATCH_SIZE = 64

def run_indexing():
    logger.info('Criando coleção no Qdrant...')
    create_collection(COLLECTION)
    client = get_client()

    logger.info('Carregando processed JSONs...')
    documents = load_all_processed()
    logger.info(f'{len(documents)} documentos carregados')
    
    all_chunks = []
    logger.info('Gerando chunks...')
    for doc in tqdm(documents, desc='Chunking'):
        # Adapta ProcessedDocument para o que o chunk_document espera sem quebrar
        # já que ProcessedDocument mapeia texto_documento direto
        doc_dict = doc.to_dict()
        doc_dict['source_file'] = doc.arquivo_origem
        doc_dict['pages'] = [{'page': 1, 'text': doc.texto_documento}]
        
        meta = {
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
        
        # Chama a função de chunking com use_context=False
        try:
            doc_chunks = DocumentChunker.chunk_document(doc_dict, meta=meta)
            all_chunks.extend(doc_chunks)
        except TypeError:
            # Caso a API de chunk_document não receba use_context:
            doc_chunks = DocumentChunker.chunk_document(doc_dict)
            all_chunks.extend(doc_chunks)

    logger.info(f'Total de chunks gerados: {len(all_chunks)}')

    logger.info(f'Indexando chunks no Qdrant em lotes de {BATCH_SIZE}...')
    for i in tqdm(range(0, len(all_chunks), BATCH_SIZE), desc='Indexando'):
        batch = all_chunks[i : i + BATCH_SIZE]
        
        # Pode ter a propriedade `text_to_embed` ou apenas `text` dependendo da implementação do chunker
        texts = [getattr(c, 'text_to_embed', c.text) for c in batch]
        
        vectors = embed_chunks(texts)
        
        points = []
        for chunk, vec in zip(batch, vectors):
            payload = {
                'text': getattr(chunk, 'text_to_embed', chunk.text),
                'source_file': getattr(chunk, 'source_file', doc.arquivo_origem),
                'page': getattr(chunk, 'page', 1),
                'titulo': chunk.titulo,
                'assunto': chunk.assunto,
                'situacao': getattr(chunk, 'situacao', doc.situacao),
                'data_publicacao': getattr(chunk, 'publicacao', chunk.titulo) # Mapeamento tolerante
            }
            
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload=payload
                )
            )
            
        client.upsert(collection_name=COLLECTION, points=points)

    total = client.count(collection_name=COLLECTION).count
    logger.info(f'Indexação concluída! Vetores na coleção {COLLECTION}: {total}')

if __name__ == '__main__':
    run_indexing()
