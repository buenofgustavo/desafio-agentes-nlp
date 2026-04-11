"""Responsável por orquestrar o processo de ingestão dos documentos, desde a extração do texto até a geração dos chunks e embeddings."""
from src.indexing.ingestion.json_loader import JsonLoader
from src.indexing.ingestion.pdf_loader import PdfDownloader
from src.core.config import Constants

def main() -> None:
    all_aneel_records = JsonLoader.load_json_folder_data(Constants.JSON_DIR)
    PdfDownloader.download_pdfs(all_aneel_records)

if __name__ == "__main__":
    main()