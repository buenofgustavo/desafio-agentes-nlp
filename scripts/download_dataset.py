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
    try:
        logger.info("Downloading Dataset...")
        
        response = requests.get(DATASET_URL)
        response.raise_for_status()
        
        with open(ZIP_PATH, "wb") as f:
            f.write(response.content)

        logger.info("Dowload completo!")
        
    except Exception as e:
        logger.error(f"Erro no download do arquivo zip: {e}")
        
def extract_zip():
    try:
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(DOWNLOAD_FOLDER)
            
        logger.info("Extração completa do arquivo zip")
    except Exception as e:
        logger.error(f"Erro ao extrair arquivo zip: {e}")
        
def main():
    download_dataset()
    extract_zip()
    
    os.remove(ZIP_PATH)
    
if __name__ == "__main__":
    main()