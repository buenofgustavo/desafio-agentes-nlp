"""Script para baixar e extrair o dataset do desafio."""
import os
import requests
import zipfile
from pathlib import Path
from src.core.config import Constants
from src.utils.logger import LoggingService

logger = LoggingService.setup_logger(__name__)

DATASET_URL = "https://github.com/buenofgustavo/desafio-agentes-nlp/releases/download/dataset-desafio/dados_grupo_estudos.zip"
DOWNLOAD_FOLDER = Constants.JSON_DIR
DOWNLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
ZIP_PATH = DOWNLOAD_FOLDER / "dados_grupo_estudos.zip"

def download_dataset():
    """Baixa o arquivo zip do dataset."""
    try:
        logger.info("Baixando Dataset...")
        
        response = requests.get(DATASET_URL)
        response.raise_for_status()
        
        with open(ZIP_PATH, "wb") as f:
            f.write(response.content)

        logger.info("Download completo!")
        
    except Exception as e:
        logger.error(f"Erro no download do arquivo zip: {e}")
        
def extract_zip():
    """Extrai o conteúdo do arquivo zip."""
    try:
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(DOWNLOAD_FOLDER)
            
        logger.info("Extração completa do arquivo zip")
    except Exception as e:
        logger.error(f"Erro ao extrair arquivo zip: {e}")
        
def main():
    """Executa o download e a extração, removendo o zip ao final."""
    download_dataset()
    extract_zip()
    
    if ZIP_PATH.exists():
        os.remove(ZIP_PATH)
    
if __name__ == "__main__":
    main()