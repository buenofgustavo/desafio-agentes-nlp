"""Responsável por orquestrar o processo de ingestão dos documentos, desde a extração do texto até a geração dos chunks e embeddings."""


from src.settings import DOCUMENTS_DIR
from src.ingestion.parsing import load_documents

def main() -> None:
    docs = load_documents(DOCUMENTS_DIR)
    if not docs:
        raise RuntimeError(
            f"Nenhum documento encontrado em {DOCUMENTS_DIR}. "
            "Adicione arquivos na pasta de documentos."
        )
    print(docs)

if __name__ == "__main__":
    main()