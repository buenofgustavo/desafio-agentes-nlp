from src.core.config import DOCUMENTS_DIR
from src.indexing.processing.parsing import load_documents

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