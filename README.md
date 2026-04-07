INDEXAÇÃO (feita uma vez, offline):
Documentos → Parsing → Chunks → Embeddings → VectorDB

CONSULTA (a cada pergunta do usuário, online):
Query → Embed → VectorDB → Top-K Chunks → LLM → Resposta
