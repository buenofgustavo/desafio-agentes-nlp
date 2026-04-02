INDEXAÇÃO (feita uma vez, offline):
Documentos → Parsing → Chunks → Embeddings → VectorDB

Documentos:
Identificar o documento e tipo

Parsing:
Obter os dados dos arquivos
Usar pytesseract + llm para PDFs

CONSULTA (a cada pergunta do usuário, online):
Query → Embed → VectorDB → Top-K Chunks → LLM → Resposta