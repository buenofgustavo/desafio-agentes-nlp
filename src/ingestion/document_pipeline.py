"""Responsável por orquestrar o processo de ingestão dos documentos, desde a extração do texto até a geração dos chunks e embeddings."""
from src.ingestion.json_loader import JsonLoader
from src.settings import Constants

def main() -> None:
    all_aneel_records = JsonLoader.load_json_folder_data(Constants.JSON_DIR)

if __name__ == "__main__":
    main()