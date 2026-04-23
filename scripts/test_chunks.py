"""Script para testar o chunking de documentos processados."""
import sys
import argparse
from pathlib import Path

# Configura path para importar módulos src
sys.path.append(str(Path(__file__).parent.parent))

from src.indexing.storage.processed_store import load_all_processed
from src.indexing.processing.chunker.chunker import DocumentChunker
from src.utils.logger import LoggingService

logger = LoggingService.setup_logger(__name__)

def test_chunks(limit=1, file_name=None):
    logger.info("Carregando documentos processados...")
    documents = load_all_processed()
    
    if not documents:
        logger.error("Nenhum documento processado encontrado.")
        return

    if file_name:
        selected_docs = [d for d in documents if d.arquivo_origem == file_name]
        if not selected_docs:
            logger.warning(f"Documento '{file_name}' não encontrado. Usando o primeiro disponível.")
            selected_docs = documents[:1]
    else:
        selected_docs = documents[:limit]

    chunker = DocumentChunker()

    for doc in selected_docs:
        logger.info(f"\n--- Testando Chunking para: {doc.arquivo_origem} ---")
        
        doc_dict = {
            "pages": [{"page": 1, "text": doc.texto_documento}]
        }
        
        meta_dict = {
            "source_file": doc.arquivo_origem,
            "titulo": doc.titulo,
            "autor": doc.autor,
            "material": doc.material,
            "esfera": doc.esfera,
            "situacao": doc.situacao,
            "assinatura": doc.assinatura,
            "publicacao": doc.publicacao,
            "assunto": doc.assunto,
            "ementa": doc.ementa
        }

        try:
            chunks = chunker.chunk_document(doc_dict, meta=meta_dict, use_context=False)
            logger.info(f"Total de chunks gerados: {len(chunks)}")
            
            if not chunks:
                logger.warning("Nenhum chunk gerado para este documento.")
                continue

            # Mostrar Resumo
            parents = sorted(list(set([c.parent_index for c in chunks])))
            logger.info(f"Estratégia: Parent-Child")
            logger.info(f"Número de Parent chunks: {len(parents)}")
            
            # Mostrar os primeiros 3 chunks
            logger.info("\nDetalhamento dos primeiros chunks:")
            for i, chunk in enumerate(chunks[:3]):
                print(f"\nCHUNK #{i+1} (Page {chunk.page}, Parent {chunk.parent_index}, Child {chunk.child_index})")
                print(f"Tamanho: {len(chunk.text)} caracteres")
                print("-" * 20)
                print(f"{chunk.text}")
                print("-" * 20)
            
            if len(chunks) > 3:
                print(f"\n... e mais {len(chunks)-3} chunks.")

        except Exception as e:
            logger.error(f"Erro ao processar {doc.arquivo_origem}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testa o chunking de documentos.")
    parser.add_argument("--limit", type=int, default=1, help="Número de documentos para testar.")
    parser.add_argument("--file", type=str, help="Nome do arquivo específico para testar.")
    
    args = parser.parse_args()
    test_chunks(limit=args.limit, file_name=args.file)
