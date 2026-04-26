# Variáveis
PYTHON = python3
PIP = pip
DOCKER = docker-compose
GCP_BUCKET_DOCUMENTS_PATH = gs://aneel-raw-data/aneel-documents/
GCP_BUCKET_PROCESSED_JSON_PATH = gs://aneel-raw-data/processed-json/
GCP_BUCKET_QDRANT_SNAPSHOT_PATH = gs://aneel-raw-data/qdrant-snapshot/

help:
	@echo "Comandos disponíveis no projeto RAG Setor Elétrico:"
	@echo "  make install             - Instala as dependências do projeto"
	@echo "  make up                  - Sobe a infraestrutura via Docker"
	@echo "  make down                - Para e remove os containers do Docker"
	@echo "  make dataset             - Baixa e extrai o dataset necessário para o projeto"
	@echo "  make ingestion           - Executa o pipeline de ingestão dos documentos"
	@echo "  make indexing            - Gera embeddings e indexa no Qdrant"
	@echo ""
	@echo "  make build-bm25          - Constrói e persiste o índice BM25 (requer Qdrant ativo)"
	@echo ""
	@echo "  make run-agent           - Executa o agente RAG em modo interativo"
	@echo ""
	@echo "  make sync-data           - Baixa os Documentos da Aneel do bucket GCP"
	@echo "  make upload-data         - Envia os Documentos para o bucket GCP"
	@echo "  make sync-processed-json - Baixa os JSONs processados do bucket GCP"
	@echo "  make upload-processed-json - Envia os JSONs processados para o bucket GCP"
	

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt

up:
	$(DOCKER) up -d

down:
	$(DOCKER) down

dataset:
	$(PYTHON) -m scripts.download_dataset

ingestion:
	$(PYTHON) -m scripts.run_ingestion

indexing:
	$(PYTHON) -m scripts.run_indexing

build-bm25:
	@echo "Construindo índice BM25 a partir do Qdrant..."
	$(PYTHON) -m src.retrieval.bm25_retriever --rebuild

run-agent:
	$(PYTHON) scripts/run_agent.py

# ===================== Qdrant snapshot =====================================
sync-qdrant-snapshot:
	@echo "Sincronizando snapshot do Qdrant com o GCP..."

	mkdir -p qdrant_setup

	time gcloud storage cp $(GCP_BUCKET_QDRANT_SNAPSHOT_PATH)desafio-agentes-nlp.snapshot qdrant_setup/

# (Privado, só para admins)
upload-qdrant-snapshot:
	@echo "Enviando snapshot do Qdrant para o GCP..."
	time gcloud storage cp qdrant_setup/desafio-agentes-nlp.snapshot $(GCP_BUCKET_QDRANT_SNAPSHOT_PATH)
# ===========================================================================

# ================== DADOS BRUTOS (.pdf, .htm, .xlsm, etc) ==================
sync-data:
	@echo "Iniciando o download paralelo do Storage..."
	
	mkdir -p data/raw/documents
	
	time gcloud storage rsync $(GCP_BUCKET_DOCUMENTS_PATH) data/raw/documents/ --recursive

# (Privado, só para admins)
upload-data:
	@echo "Enviando novos Documentos do disco para o Storage..."
	time gcloud storage rsync data/raw/documents/ $(GCP_BUCKET_DOCUMENTS_PATH) --recursive
# ============================================================================

# ===================== DADOS PROCESSADOS (.json) ============================
sync-processed-json:
	@echo "Iniciando o download paralelo do Storage (Processed JSON)..."
	
	mkdir -p data/processed
	
	time gcloud storage rsync $(GCP_BUCKET_PROCESSED_JSON_PATH) data/processed/ --recursive

# (Privado, só para admins)
upload-processed-json:
	@echo "Enviando novos arquivos JSON do disco para o Storage..."
	time gcloud storage rsync data/processed/ $(GCP_BUCKET_PROCESSED_JSON_PATH) --recursive
# ============================================================================

