# Variáveis
PYTHON = python3
PIP = pip
DOCKER = docker-compose

# Alvo padrão quando você digita apenas 'make'
help:
	@echo "Comandos disponíveis no projeto RAG Setor Elétrico:"
	@echo "  make install  - Instala as dependências do projeto"
	@echo "  make up       - Sobe o banco de dados (Qdrant) via Docker"
	@echo "  make down     - Para e remove os containers do Docker"
	@echo "  make dataset  - Baixa e extrai o dataset necessário para o projeto"

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
