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
	@echo "  make install             - Instala as dependências do projeto"
	@echo "  make up                  - Sobe o banco de dados (Qdrant) via Docker"
	@echo "  make down                - Para e remove os containers do Docker"
	@echo "  make dataset             - Baixa e extrai o dataset necessário para o projeto"
	@echo "  make ingestion           - Executa o pipeline de ingestião dos documentos"
	@echo "  make indexing            - Gera embeddings e indexa no Qdrant"
	@echo ""
	@echo "  make build-bm25          - Constrói e persiste o índice BM25 (requer Qdrant ativo)"
	@echo "  make generate-benchmark  - Anota golden_chunk_ids no benchmark via busca densa"
	@echo "  make eval-retrieval      - Avalia baseline vs hybrid+reranker e gera relatório"
	@echo ""
	@echo "  make sync-data           - Baixa os Documentos da Aneel do bucket GCP"
	@echo "  make upload-data         - Envia os Documentos para o bucket GCP"
	@echo "  make sync-processed-json - Baixa os JSONs processados do bucket GCP"
	@echo "  make sync-docling-markdown - Baixa os Markdowns Docling do bucket GCP"

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

indexing:
	$(PYTHON) -m scripts.run_indexing

# ── Fase 2: Recuperação Híbrida ───────────────────────────────────────────

build-bm25:
	@echo "Construindo índice BM25 a partir do Qdrant..."
	$(PYTHON) -m src.retrieval.bm25_retriever --rebuild

generate-benchmark:
	@echo "Anotando benchmark com golden_chunk_ids via busca densa..."
	$(PYTHON) scripts/generate_benchmark.py

eval-retrieval:
	$(PYTHON) scripts/run_retrieval_eval.py \
		--benchmark data/processed/benchmark.json \
		--output data/processed/retrieval_report.json

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