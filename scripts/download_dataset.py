import os
import requests
import zipfile
from pathlib import Path

DATASET_URL = "https://github.com/buenofgustavo/desafio-agentes-nlp/releases/download/dataset-desafio/dados_grupo_estudos.zip"
DATA_PATH = Path("data")
DOWNLOAD_FOLDER = DATA_PATH / "json"
DOWNLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
ZIP_PATH = DOWNLOAD_FOLDER / "dados_grupo_estudos.zip"


def download_dataset():
    try:
        print("Downloading Dataset...")
        
        response = requests.get(DATASET_URL)
        response.raise_for_status()
        
        with open(ZIP_PATH, "wb") as f:
            f.write(response.content)

        print("Dowload completo!")
        
    except Exception as e:
        print(f"Erro no download do arquivo zip: {e}")
        
def extract_zip():
    try:
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(DOWNLOAD_FOLDER)
            
        print("Extração completa do arquivo zip")
    except Exception as e:
        print(f"Erro ao extrair arquivo zip: {e}")
        
def main():
    download_dataset()
    extract_zip()
    
    os.remove(ZIP_PATH)
    
if __name__ == "__main__":
    main()