# Desafio Agentes NLP - RAG Setor Elétrico

Este projeto implementa uma pipeline avançada de Retrieval-Augmented Generation (RAG) voltada para o setor elétrico. Ele utiliza agentes de NLP, FastAPI, LangGraph, Streamlit, e o banco vetorial Qdrant para criar um sistema de respostas a perguntas eficiente e especializado.

## 📋 Pré-requisitos

Certifique-se de ter as seguintes ferramentas instaladas:
- [Python 3.10+](https://www.python.org/downloads/)
- [Docker](https://docs.docker.com/get-docker/) e [Docker Compose](https://docs.docker.com/compose/install/)
- `make` (Opcional, mas recomendado. Geralmente pré-instalado em Linux/macOS. No Windows, pode ser usado via WSL, Git Bash ou instalado via choco/scoop)

## 🚀 Como Configurar e Executar

Siga os passos abaixo para configurar o ambiente, subir o banco de dados e rodar a aplicação:

### 1. Configurar o Ambiente Virtual (venv)

Recomendamos fortemente isolar as dependências em um ambiente virtual.

**No Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**No Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Instalar as Dependências

Com o ambiente virtual ativado (`(venv)`), instale os pacotes necessários:

```bash
# Via Make
make install
```
```bash
# Via pip
pip install -r requirements.txt
```

**Dependências de NLP (Modelos e Stopwords):**
Após instalar os pacotes, você precisará baixar os pacotes necessários para o NLTK, usado no processamento de linguagem natural:

```bash
python -m nltk.downloader stopwords punkt
```

### 3. Configurar Variáveis de Ambiente

Baseie-se no arquivo de exemplo para criar o seu arquivo de ambiente e adicione as suas chaves e tokens (como a API Key da Anthropic, por exemplo):

```bash
cp .env.example .env
```

### 4. Baixar o Dataset

Para baixar os dados necessários para o projeto e extraí-los na pasta correta (`data/raw`), execute:

```bash
# Via Make
make dataset
```
```bash
# Chamada direta via Python
python3 -m scripts.download_dataset
```

### 5. Subir os Containers (Banco Vetorial Qdrant)

O sistema de busca semântica depende do Qdrant. Para subir os containers na sua máquina de forma simples, execute:

```bash
# Via Make
make up
```
```bash
# Via Docker Compose
docker-compose up -d
```

#### 📊 Acessando o Dashboard do Qdrant
Uma vez que o container estiver rodando, o Qdrant fornece uma interface de usuário excelente (Web UI) onde você pode visualizar suas coleções de vetores e realizar consultas direto pelo navegador:
**Acesse:** 👉 [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

### 6. Pipeline de Indexação e Recuperação Híbrida

O projeto incorpora uma busca híbrida unindo BM25 (esparsa) e Embeddings Densos (Qdrant), combinados com reordenação avançada (Reranking cross-encoder).

**a) Realizar o Chunking e a Indexação Densa**
Aplica o chunking nos documentos processados e salva os embeddings vetoriais no banco Qdrant.
```bash
make indexing
```

**b) Construir o Índice Esparso (BM25)**
Percorre os chunks salvos no Qdrant e constrói um índice local para a busca por palavras-chave. *(Requer o Qdrant rodando com a indexação anterior finalizada)*.
```bash
make build-bm25
```

**c) Avaliar o Sistema de Recuperação**
Para testar a performance (Context Precision e Context Recall) do sistema híbrido em comparação com uma busca estritamente densa:
```bash
# Anotar o benchmark com chunks gerados pelo Qdrant
make generate-benchmark

# Executar a avaliação completa da recuperação e salvar o relatório JSON
make eval-retrieval
```

> 💡 **Dica:** Para verificar todos os comandos disponíveis e o que cada um faz, basta rodar `make` ou `make help` no seu terminal.

## 📁 Estrutura de Diretórios Básica

- `data/` - Dados, arquivos brutos ou processados.
- `docs/` - Documentações adicionais.
- `notebooks/` - Jupyter Notebooks para exploração, rascunhos e avaliações.
- `src/` - O código-fonte principal que orquestra a aplicação.
- `Makefile` - Arquivo contendo atalhos (targets) para comandos chave.

## 👨‍💻 Equipe técnica

- **Igor Reis Braziel** - [braziel@discente.ufg.br](mailto:braziel@discente.ufg.br)
