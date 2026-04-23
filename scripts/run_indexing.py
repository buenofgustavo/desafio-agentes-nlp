"""Workflow para indexação dos documentos."""
import os
import sys
import uuid
import hashlib
import time
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
# Keep in sync with EMBEDDING_BATCH_SIZE in embedder.py (default 64).
BATCH_SIZE = int(os.getenv('EMBEDDING_BATCH_SIZE', '64'))
# Flush every 2 000 chunks: starts embedding sooner and caps peak RAM usage.


def _generate_deterministic_id(chunk: ChildChunk) -> str:
    """Generate a deterministic UUID based on chunk location to prevent duplicate insertions."""
    unique_str = f"{chunk.source_file}_{chunk.page}_{chunk.parent_index}_{chunk.child_index}"
    return str(uuid.UUID(hashlib.md5(unique_str.encode("utf-8")).hexdigest()))


def run_indexing():
    start_time = time.time()
    
    logger.info('='*80)
    logger.info('INICIANDO PIPELINE DE CHUNKING + INDEXAÇÃO')
    logger.info('='*80)
    
    # Step 1: Collection setup
    logger.info('📦 [1/4] Criando/Verificando coleção no Qdrant...')
    create_collection(COLLECTION)
    client = get_qdrant_client()
    logger.info(f'✓ Coleção "{COLLECTION}" pronta')

    # Step 2: Load documents and identify already-indexed ones
    load_start = time.time()
    logger.info('📄 [2/4] Carregando documentos processados...')
    documents = load_all_processed()
    load_time = time.time() - load_start
    logger.info(f'✓ {len(documents)} documentos carregados em {load_time:.2f}s')
    
    # Get already indexed source files
    logger.info('🔍 Identificando documentos já indexados...')
    indexed_files = set()
    try:
        # Scroll through collection to get all source_file values
        indexed_point_count = 0
        scroll_result = client.scroll(
            collection_name=COLLECTION,
            limit=1000,
            with_payload=True,
            with_vectors=False
        )
        
        while scroll_result[0]:
            for point in scroll_result[0]:
                if point.payload and 'source_file' in point.payload:
                    indexed_files.add(point.payload['source_file'])
                    indexed_point_count += 1
            
            if scroll_result[1] is None:
                break
            
            scroll_result = client.scroll(
                collection_name=COLLECTION,
                limit=1000,
                offset=scroll_result[1],
                with_payload=True,
                with_vectors=False
            )
        
        logger.info(f'✓ {len(indexed_files)} arquivos únicos já indexados ({indexed_point_count} pontos na coleção)')
    except Exception as e:
        logger.warning(f'⚠️  Não foi possível recuperar arquivos já indexados: {e}. Processando todos.')
    
    # Filter documents to process only new ones
    docs_to_process = [doc for doc in documents if doc.arquivo_origem not in indexed_files]
    docs_skipped = len(documents) - len(docs_to_process)
    
    if docs_skipped > 0:
        logger.info(f'⏭️  {docs_skipped} documentos já indexados serão pulados')
        logger.info(f'📝 {len(docs_to_process)} documentos novos serão processados')
    
    # Initialize processing
    chunker = DocumentChunker()
    
    chunk_buffer: List[ChildChunk] = []
    total_processed_chunks = 0
    total_batches_flushed = 0
    documents_with_errors = 0
    stats_by_doc = []
    documents_skipped = docs_skipped
    
    def _flush_buffer():
        nonlocal total_processed_chunks, total_batches_flushed
        if not chunk_buffer:
            logger.debug('Buffer vazio, pulando flush')
            return
            
        num_chunks = len(chunk_buffer)
        logger.info(f'🔄 Sincronizando buffer: {num_chunks} chunks em lotes de {BATCH_SIZE}...')
        
        batch_count = 0
        embedding_start = time.time()
        
        for i in range(0, len(chunk_buffer), BATCH_SIZE):
            batch = chunk_buffer[i : i + BATCH_SIZE]
            batch_count += 1
            texts = [c.text_to_embed for c in batch]
            
            # Embedding
            logger.debug(f'  Batch {batch_count}: Gerando embeddings para {len(texts)} textos...')
            embed_start = time.time()
            vectors = embed_chunks(texts)
            embed_time = time.time() - embed_start
            logger.debug(f'  Batch {batch_count}: ✓ Embeddings gerados em {embed_time:.2f}s')
            
            # Prepare points for upsert
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
            
            # Upsert to Qdrant
            try:
                upsert_start = time.time()
                client.upsert(collection_name=COLLECTION, points=points)
                upsert_time = time.time() - upsert_start
                total_processed_chunks += len(points)
                logger.debug(f'  Batch {batch_count}: ✓ {len(points)} pontos inseridos em {upsert_time:.2f}s')
            except Exception as e:
                logger.error(f'  Batch {batch_count}: ✗ Erro ao inserir lote: {type(e).__name__}: {e}')
                
        embedding_total_time = time.time() - embedding_start
        total_batches_flushed += 1
        chunk_buffer.clear()
        logger.info(f'✓ Buffer sincronizado ({batch_count} lotes, {embedding_total_time:.2f}s). Total indexado: {total_processed_chunks} chunks')

    logger.info('🔪 [3/4] Iniciando processamento (Chunking + Indexação)...')
    logger.info('-'*80)
    
    chunk_start_time = time.time()
    for doc_idx, doc in enumerate(tqdm(docs_to_process, desc='Docs'), start=1):
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
            chunk_time = time.time()
            doc_chunks = chunker.chunk_document(doc_dict, meta=meta_dict, use_context=False)
            chunk_duration = time.time() - chunk_time
            
            chunk_buffer.extend(doc_chunks)
            stats_by_doc.append({
                'arquivo': doc.arquivo_origem,
                'chunks': len(doc_chunks),
                'tempo': chunk_duration,
                'tamanho_texto': len(doc.texto_documento)
            })
            
            logger.debug(f'  [{doc_idx}] {doc.arquivo_origem}: {len(doc_chunks)} chunks em {chunk_duration:.2f}s')
            
            if len(chunk_buffer) >= BATCH_SIZE * 50:
                logger.info(f'  ⚠️  Buffer atingiu {len(chunk_buffer)} chunks, sincronizando...')
                _flush_buffer()
                
        except Exception as e:
            documents_with_errors += 1
            logger.error(f'  [{doc_idx}] ✗ Erro ao processar {doc.arquivo_origem}: {type(e).__name__}: {e}')

    chunk_phase_time = time.time() - chunk_start_time
    logger.info('-'*80)
    logger.info(f'✓ Fase de chunking concluída em {chunk_phase_time:.2f}s')
    
    # Final flush
    if chunk_buffer:
        logger.info('🔄 Sincronizando buffer final...')
        _flush_buffer()

    # Summary
    logger.info('')
    logger.info('='*80)
    logger.info('📊 RESUMO DA INDEXAÇÃO')
    logger.info('='*80)
    
    total_in_db = client.count(collection_name=COLLECTION).count
    total_time = time.time() - start_time
    
    logger.info(f'✓ Total de documentos processados: {len(docs_to_process) - documents_with_errors}/{len(docs_to_process)}')
    logger.info(f'⏭️  Documentos pulados (já indexados): {documents_skipped}')
    logger.info(f'✗ Documentos com erro: {documents_with_errors}')
    logger.info(f'✓ Total de chunks indexados nesta sessão: {total_processed_chunks}')
    logger.info(f'✓ Total de vetores na coleção "{COLLECTION}": {total_in_db}')
    logger.info(f'✓ Flushes do buffer: {total_batches_flushed}')
    logger.info(f'⏱️  Tempo total: {total_time:.2f}s ({total_time/60:.2f}m)')
    
    if stats_by_doc:
        avg_chunks = sum(s['chunks'] for s in stats_by_doc) / len(stats_by_doc)
        avg_time = sum(s['tempo'] for s in stats_by_doc) / len(stats_by_doc)
        max_chunks = max(s['chunks'] for s in stats_by_doc)
        min_chunks = min(s['chunks'] for s in stats_by_doc)
        
        logger.info(f'📈 Estatísticas por documento:')
        logger.info(f'   • Chunks: média={avg_chunks:.1f}, min={min_chunks}, max={max_chunks}')
        logger.info(f'   • Tempo: {avg_time:.3f}s por documento em média')
    
    logger.info('='*80)

if __name__ == '__main__':
    run_indexing()
