# Variáveis
PYTHON = python3
PIP = pip
DOCKER = docker-compose
GCP_BUCKET_DOCUMENTS_PATH = gs://aneel-raw-data/aneel-documents/
GCP_BUCKET_PROCESSED_JSON_PATH = gs://aneel-raw-data/processed-json/
GCP_BUCKET_DOCLING_MARKDOWN_PATH = gs://aneel-raw-data/docling-markdowns/

# Alvo padrão quando você digita apenas 'make'
help:
	@echo "Comandos disponíveis no projeto RAG Setor Elétrico:"
	@echo "  make install  - Instala as dependências do projeto"
	@echo "  make up       - Sobe o banco de dados (Qdrant) via Docker"
	@echo "  make down     - Para e remove os containers do Docker"
	@echo "  make dataset  - Baixa e extrai o dataset necessário para o projeto"
	@echo "  make ingestion - Executa o pipeline de ingestão dos documentos"
	@echo "  make sync-data  - Baixa os Documentos da Aneel que foram enviados pro bucket GCP (Acesso Público)"
	@echo "  make upload-data - Envia os Documentos do disco para o bucket GCP (Acesso Privado, apenas para admins)"

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
	$(PYTHON) -m src.indexing.document_pipeline


# ================== DADOS BRUTOS (.pdf, .htm, .xlsm, etc) ==================
sync-data:
	@echo "Iniciando o download paralelo do Storage..."
	
	mkdir -p data/raw/documents
	
	time gcloud storage rsync $(GCP_BUCKET_DOCUMENTS_PATH) data/raw/documents/ --recursive

# Upload TO the cloud (Private, only for admins)
upload-data:
	@echo "Enviando novos Documentos do disco para o Storage..."
	time gcloud storage rsync data/raw/documents/ $(GCP_BUCKET_DOCUMENTS_PATH) --recursive
# ============================================================================

# ===================== DADOS PROCESSADOS (.json) ============================
sync-processed-json:
	@echo "Iniciando o download paralelo do Storage (Processed JSON)..."
	
	mkdir -p data/processed
	
	time gcloud storage rsync $(GCP_BUCKET_PROCESSED_JSON_PATH) data/processed/ --recursive

upload-processed-json:
	@echo "Enviando novos arquivos JSON do disco para o Storage..."
	time gcloud storage rsync data/processed/ $(GCP_BUCKET_PROCESSED_JSON_PATH) --recursive
# ============================================================================

# ========================== DOCLING MARKDOWN ================================
sync-docling-markdown:
	@echo "Iniciando o download paralelo do Storage (Docling Markdown)..."
	
	mkdir -p data/raw/docling_markdown
	
	time gcloud storage rsync $(GCP_BUCKET_DOCLING_MARKDOWN_PATH) data/raw/docling_markdown/ --recursive
# ============================================================================