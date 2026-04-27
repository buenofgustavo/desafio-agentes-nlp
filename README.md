# RAG — Setor Elétrico Brasileiro

Sistema de **Retrieval-Augmented Generation (RAG)** aplicado a documentos técnicos e regulatórios do setor elétrico brasileiro (ANEEL). O sistema combina busca híbrida BM25 + embeddings densos com um agente LangGraph que realiza expansão de queries (HyDE + reformulações), múltiplos rounds de recuperação, geração de resposta com Claude e verificação automática de fidelidade. A interface é exposta via FastAPI e Streamlit.

---

## 🏗️ Arquitetura

```
Documentos PDF, XLSX, etc.
      │
      ▼
┌───────────────────────────────────────────────────────┐
│  Parsing & Indexação                                  │
│  PyMuPDF e Outros → Chunks → Embeddings → Qdrant      │
│  BM25Retriever (índice esparso local)                 │
└───────────────────────────────────────────────────────┘
      │
      ▼
┌───────────────────────────────────────────────────────┐
│  Recuperação Híbrida                                  │
│  BM25 + Dense → RRF Fusion → Reranker                 │
│  (CrossEncoderReranker)                               │
└───────────────────────────────────────────────────────┘
      │
      ▼
┌───────────────────────────────────────────────────────┐
│  Agente LangGraph                                     │
│  query_analyzer → query_expander →                    │
│  retriever → reranker →                               │
│  context_assembler → generator →                      │
│  faithfulness_check (self-correction loop)            │
└───────────────────────────────────────────────────────┘
      │
      ├─── FastAPI  (app/api.py)   → http://localhost:8000
      └─── Streamlit (app/ui.py)  → http://localhost:8501
```

Para detalhes de design, consulte a documentação no código-fonte.

---

## ⚙️ Pré-requisitos
Para executar este projeto, sua máquina precisa ter instalado:
* **Docker**
* **Docker Compose**
* **Google Cloud SDK (gcloud CLI)** (para sincronização de dados/snapshots)
* Uma chave de API válida da **Anthropic (Claude)**.

---

## 🚀 Guia Rápido de Execução (Passo a Passo)

### Passo 0: Clonar o Repositório
Abra o terminal e execute os comandos abaixo para clonar o projeto e entrar na pasta:
```bash
git clone https://github.com/buenofgustavo/desafio-agentes-nlp.git
cd desafio-agentes-nlp
```

### Passo 1: Configurar a Chave de API
Na raiz desta pasta, existe um arquivo chamado `.env.example`.
1. Renomeie este arquivo para `.env`.
2. Abra o arquivo e insira a sua chave da Anthropic:
```env
ANTHROPIC_API_KEY=sk-ant-sua-chave-aqui...
```

### Passo 2: Baixar o Snapshot do Banco de Dados
Para evitar o re-processamento de milhares de documentos, baixe o snapshot do Qdrant diretamente do nosso bucket no GCP:
```bash
mkdir -p qdrant_setup

gcloud storage cp gs://aneel-raw-data/qdrant-snapshot/desafio-agentes-nlp.snapshot qdrant_setup/
```

### Passo 3: Iniciar a Infraestrutura
Abra o terminal na raiz do projeto e execute o comando abaixo para baixar as imagens otimizadas e iniciar todos os serviços:
```bash
docker-compose up -d
```

### Passo 4: Restaurar o Banco de Dados Vetorial (Qdrant)
Com os containers rodando e o snapshot presente na pasta `qdrant_setup/`, execute o comando abaixo para carregar os dados:
```bash
curl -X PUT 'http://localhost:6333/collections/setor_eletrico/snapshots/recover' \
-H 'Content-Type: application/json' \
-d '{"location": "file:///qdrant/snapshots/desafio-agentes-nlp.snapshot"}'
```

### Passo 5 (Opcional): Construir o Índice BM25
O sistema utiliza busca híbrida. Enquanto os embeddings densos estão no Qdrant, o índice BM25 é local e precisa ser gerado a partir dos dados restaurados:
```bash
docker exec -it rag_api_setor_eletrico python -m src.retrieval.bm25_retriever --rebuild
```

### Passo 6: Acessar a Aplicação
Com o banco populado, o índice BM25 pronto e a API rodando, acesse a interface do usuário pelo seu navegador:
👉 **[http://localhost:8501](http://localhost:8501)**

---

## 🗄️ Gerenciamento de Dados e Scripts

Todos os dados processados, documentos brutos e snapshots do Qdrant estão armazenados em nosso bucket no **GCP (Google Cloud Platform)**. Isso garante:
- **Replicação Rápida:** Sincronização eficiente dos dados para facilitar o setup inicial.
- **Sempre Atualizado:** A nuvem atua como fonte de verdade dos documentos.

Para interagir com o GCP e gerenciar os dados, verifique o `Makefile` na raiz do projeto. Ele contém comandos essenciais (como `make sync-data`, `make sync-processed-json` e `make sync-qdrant-snapshot`).

Além disso, a pasta `scripts/` contém os arquivos executáveis do pipeline. O usuário pode rodar individualmente qualquer etapa (como ingestão, indexação ou inicialização da coleção) de forma manual, se desejar.

---

## 📁 Estrutura do Projeto

```
desafio-agentes-nlp/
├── app/
│   ├── api.py              # FastAPI — endpoints de inferência e healthcheck
│   └── ui.py               # Interface Streamlit para interação com o usuário
├── data/
│   ├── raw/                # Documentos brutos (PDF, XLSX, etc.)
│   └── processed/          # Dados processados e chunks em formato JSON
├── qdrant_setup/           # Snapshots e arquivos de configuração do Qdrant
├── scripts/
│   ├── download_dataset.py # Download do dataset (JSONs)
│   ├── run_indexing.py     # Pipeline de indexação e criação de embeddings
│   ├── run_ingestion.py    # Processamento e ingestão de documentos
│   ├── run_agent.py        # Execução do agente via linha de comando
│   └── setup_collection.py # Script de inicialização da coleção no Qdrant
├── src/
│   ├── agent/              # Lógica do agente LangGraph
│   │   ├── graph.py        # Definição e compilação do grafo de estados
│   │   ├── nodes.py        # Implementação das funções de cada nó
│   │   ├── state.py        # Definição do esquema de estado do agente
│   │   └── query_expansion.py # Técnicas de expansão de consulta (HyDE)
│   ├── ai/
│   │   ├── embeddings/     # Geração de vetores (all-MiniLM-L6-v2)
│   │   └── llm/            # Clientes para Anthropic, OpenAI e Ollama
│   ├── core/
│   │   ├── config.py       # Gerenciamento de variáveis de ambiente
│   │   └── models.py       # Modelos Pydantic para validação de dados
│   ├── indexing/           # Módulos de ingestão, processamento e storage
│   │   ├── ingestion/      # Download e carregamento de documentos
│   │   ├── processing/     # Extração de texto e chunking
│   │   └── storage/        # Interface com o banco vetorial Qdrant
│   ├── retrieval/          # Estratégias de busca (BM25, Semântica, Híbrida)
│   └── utils/
│       ├── logger.py       # Serviço centralizado de logs
│       └── file_utils.py   # Manipulação de arquivos e diretórios
├── Dockerfile.api          # Definição do container para a API
├── Dockerfile.ui           # Definição do container para a UI
├── docker-compose.yml      # Orquestração de containers e rede
├── Makefile                # Atalhos para comandos comuns (build, run)
├── requirements.txt        # Dependências Python do projeto
└── .env.example            # Exemplo de configuração de ambiente
```

---

## ⚙️ Principais Decisões de Design

| Decisão | Motivo |
|---------|--------|
| **Busca híbrida BM25 + densa com RRF** | Combina recall de palavras-chave (BM25) com semântica (embeddings); RRF é robusto a variações de escala |
| **Cross-encoder para reranking** | Maior precisão que bi-encoders para ordenação final; custo aceitável pós-filtro |
| **LangGraph como orquestrador do agente** | Estado imutável auditável, roteamento condicional limpo, suporte nativo a multi-hop e retry loops |
| **HyDE + reformulações de query** | Melhora recall para queries ambíguas; HyDE contorna o vocabulary mismatch em busca densa |
| **FastAPI + Streamlit desacoplados** | UI stateless e substituível; API testável independentemente via `curl`/`httpx` |

---

## 🕵️‍♂️ Extração de Dados

Durante o download dos documentos da ANEEL, a biblioteca padrão `requests` era frequentemente bloqueada. A solução para realizar o bypass no Cloudflare foi utilizar a biblioteca `curl_cffi` com *sessions*:

- **Impersonação de Navegador:** O `curl_cffi` usa a tecnologia `curl-impersonate` para imitar perfeitamente o comportamento de rede de navegadores reais. Ele copia exatamente as extensões TLS, a ordem de cifras e pacotes HTTP/2 do Chrome/Edge/Safari, passando despercebido pelo WAF (Web Application Firewall).
- **O Poder das Sessions:** Utilizar uma sessão (em vez de requisições isoladas) foi crucial. A sessão gerencia os cookies automaticamente (como o desafio invisível `__cf_clearance` do Cloudflare) para as requisições subsequentes e faz reaproveitamento de conexão (*Connection Pooling* via *Keep-Alive*). Isso não apenas acelera drasticamente a extração, mas também torna o padrão de tráfego muito mais orgânico e confiável para o servidor.

---

## 👨‍💻 Equipe técnica

- **Igor Reis Braziel** — [braziel@discente.ufg.br](mailto:braziel@discente.ufg.br)
- **Gustavo Bueno Ferreira** — [gustavobueno2@discente.ufg.br](mailto:gustavobueno2@discente.ufg.br)
- **Breno Machado Barros** — [breno_machado@discente.ufg.br](mailto:breno_machado@discente.ufg.br)
