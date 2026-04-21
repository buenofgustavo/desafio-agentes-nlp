"""Script para validar os resultados da busca semântica."""
import sys
from pathlib import Path

# Configura path para importar módulos src
sys.path.append(str(Path(__file__).parent.parent))

from src.retrieval.semantic_search import search
from src.utils.logger import LoggingService

logger = LoggingService.setup_logger(__name__)

def run_validation():
    queries = [
        'Quais são os critérios para outorga de Pequena Central Hidrelétrica?',
        'Como é calculada a redução das tarifas TUSD e TUST?',
        'Quais são as penalidades por descumprimento de auto de infração?',
        'Requisitos para registro de inventário hidráulico',
    ]

    for q in queries:
        logger.info(f'\nQuery: {q}')
        try:
            results = search(q, top_k=3, apenas_vigentes=True)
            for i, r in enumerate(results, 1):
                logger.info(f' [{i}] score={r.score:.3f} | {r.titulo} | pág.{r.page}')
                logger.info(f' {r.text[:180]}...')
        except Exception as e:
            logger.error(f'Falha ao buscar por {q}: {e}')

if __name__ == '__main__':
    run_validation()
