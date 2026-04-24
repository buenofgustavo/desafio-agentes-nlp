#!/bin/bash
set -e

# Assegurar que a variável QDRANT_URL possui um valor default caso não esteja definida
QDRANT_URL=${QDRANT_URL:-"http://qdrant:6333"}

echo "Aguardando Qdrant em $QDRANT_URL..."

# Usar curl para checar o status do Qdrant repetidamente
until curl -s "$QDRANT_URL/collections" > /dev/null; do
  echo "Aguardando Qdrant iniciar..."
  sleep 2
done

echo "Qdrant está online!"

# Verifica se o arquivo do índice já existe
if [ ! -f "data/retrieval/bm25_index.pkl" ]; then
    echo "Índice BM25 não encontrado. Iniciando rebuild automático..."
    python -m src.retrieval.bm25_retriever --rebuild
else
    echo "Índice BM25 já existe. Pulando rebuild."
fi

# Inicia o comando principal do container passado no CMD do Dockerfile
echo "Executando o comando principal: $@"
exec "$@"
