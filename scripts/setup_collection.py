"""Script de configuração para criar a coleção no Qdrant."""
import os
import sys
from pathlib import Path

# Configura path para importar módulos src
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from src.indexing.storage.vector_store import create_collection, get_qdrant_client
from src.utils.logger import LoggingService

logger = LoggingService.setup_logger(__name__)
load_dotenv()

def main():
    """Cria a coleção do Qdrant com base nas variáveis de ambiente."""
    collection_name = os.getenv('QDRANT_COLLECTION', 'setor_eletrico')
    vector_size = int(os.getenv('QDRANT_VECTOR_SIZE', '384'))
    
    logger.info('='*80)
    logger.info('CONFIGURAÇÃO DE COLEÇÃO QDRANT')
    logger.info('='*80)
    logger.info(f'Nome da coleção: {collection_name}')
    logger.info(f'Tamanho do vetor: {vector_size}')
    logger.info('')
    
    try:
        # Verifica conexão
        logger.info('🔗 Verificando conexão com Qdrant...')
        get_qdrant_client()
        logger.info('✓ Conectado ao Qdrant')
        logger.info('')
        
        # Cria coleção
        logger.info(f'📦 Criando coleção "{collection_name}"...')
        create_collection(name=collection_name, vector_size=vector_size)
        logger.info(f'✓ Coleção "{collection_name}" pronta')
        logger.info('')
        
    except Exception as e:
        logger.error(f'✗ Erro ao configurar coleção: {type(e).__name__}: {e}')
        sys.exit(1)

if __name__ == '__main__':
    main()
