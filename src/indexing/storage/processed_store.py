"""Geralmente carrega os JSONs processados do diretório e mapeia para a estrutura ProcessedDocument."""
import json
from pathlib import Path
from typing import List

from src.core.config import Constants
from src.core.models import ProcessedDocument
from src.utils.logger import LoggingService

logger = LoggingService.setup_logger(__name__)

def load_all_processed(processed_dir: str | Path = Constants.PROCESSED_DATA_DIR) -> List[ProcessedDocument]:
    """Lê todos os JSONs processados e retorna uma lista de dicionários mapeados para ProcessedDocument."""
    
    docs = []
    processed_path = Path(processed_dir)
    
    if not processed_path.exists():
        logger.warning(f"Diretório de processados não existe: {processed_dir}")
        return docs
        
    json_files = list(processed_path.glob("*.json"))
    logger.info(f"Carregando {len(json_files)} documentos processados de {processed_dir}")
    
    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            doc = ProcessedDocument(
                arquivo_origem=data.get('arquivo_origem', path.stem),
                texto_documento=data.get('texto_documento', ''),
                titulo=data.get('titulo'),
                autor=data.get('autor'),
                material=data.get('material'),
                esfera=data.get('esfera'),
                situacao=data.get('situacao'),
                assinatura=data.get('assinatura'),
                publicacao=data.get('publicacao'),
                assunto=data.get('assunto'),
                ementa=data.get('ementa'),
            )
            docs.append(doc)
        except Exception as e:
            logger.error(f"Erro ao ler JSON {path.name}: {e}")
            
    return docs
