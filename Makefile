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
	@echo "  make ingest   - Roda a pipeline de ingestão de documentos (PDF -> Qdrant)"
	@echo "  make run      - Inicia a API FastAPI em modo de desenvolvimento"
	@echo "  make test     - Roda os testes unitários com Pytest"
	@echo "  make eval     - Executa o benchmark de avaliação do RAGAS"
	@echo "  make clean    - Limpa arquivos de cache do Python e de testes"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt

up:
	$(DOCKER) up -d

down:
	$(DOCKER) down
