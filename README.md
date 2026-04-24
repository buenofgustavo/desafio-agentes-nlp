# RAG — Setor Elétrico Brasileiro

Sistema de **Retrieval-Augmented Generation (RAG)** aplicado a documentos técnicos e regulatórios do setor elétrico brasileiro (ANEEL). O sistema combina busca híbrida BM25 + embeddings densos com um agente LangGraph que realiza expansão de queries (HyDE + reformulações), múltiplos rounds de recuperação, geração de resposta com Claude e verificação automática de fidelidade. A interface é exposta via FastAPI e Streamlit.

---

## 🏗️ Arquitetura

```
Documentos PDF
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

Para detalhes de design, consulte a documentação em `docs/`.

---

## 📋 Pré-requisitos

| Requisito | Versão mínima |
|-----------|--------------|
| Python | 3.11+ |
| Docker + Docker Compose | qualquer versão recente |
| Anthropic API Key | — |
| RAM | ~8 GB |
| Disco | ~10 GB |

---

## 🚀 Setup

### 1. Clonar o repositório

```bash
git clone <repo>
cd desafio-agentes-nlp
```

### 2. Configurar variáveis de ambiente

```bash
cp .env.example .env
# Edite .env e adicione sua ANTHROPIC_API_KEY
```



### 3. Subir a infraestrutura (Qdrant)

```bash
make infra   # sobe apenas o Qdrant
```

### 4. Baixar o dataset

```bash
make dataset
```

### 5. Indexar documentos

```bash
# Parsing, chunking e indexação vetorial no Qdrant
python scripts/run_indexing.py --input data/raw/

# Construir índice BM25 (requer Qdrant ativo e indexação concluída)
python -m src.retrieval.bm25_retriever --rebuild
```

---

## ▶️ Rodando o Demo

```bash
make demo   # inicia Qdrant + FastAPI (docker-compose up -d)
make ui     # inicia o Streamlit (abre http://localhost:8501)
```

Acesse:
- **API**: http://localhost:8000/docs
- **UI**: http://localhost:8501
- **Qdrant Dashboard**: http://localhost:6333/dashboard

### Verificar saúde da API

```bash
curl http://localhost:8000/health
```

### Fazer uma consulta via API

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Qual é o objetivo do PRODIST?"}'
```

### Parar os serviços

```bash
make stop
```

---

## 📁 Estrutura do Projeto

```
desafio-agentes-nlp/
├── app/
│   ├── api.py              # FastAPI — endpoints /health /query /metrics
│   └── ui.py               # Streamlit UI
├── data/
│   ├── raw/                # Documentos brutos e markdowns Docling
│   └── processed/          # Dados processados e relatórios
├── docs/                   # Documentação técnica adicional
├── scripts/
│   ├── run_indexing.py     # Pipeline de indexação (Fase 1)
│   ├── run_retrieval_eval.py # Avaliação de recuperação (Fase 2)
│   └── generate_benchmark.py # Anotação do benchmark
├── src/
│   ├── agent/              # Agente LangGraph (Fase 4)
│   │   ├── graph.py        # Grafo compilado + singleton agent_graph
│   │   ├── nodes.py        # 7 nós do agente
│   │   ├── state.py        # AgentState TypedDict
│   │   ├── prompts.py      # Templates de prompts
│   │   └── query_expansion.py # HyDE + reformulações
│   ├── ai/
│   │   ├── embeddings/     # Embedder (multilingual-e5-large)
│   │   └── llm/            # Anthropic / OpenAI / Ollama wrappers
│   ├── core/
│   │   ├── config.py       # Constantes e variáveis de ambiente
│   │   └── models.py       # Modelos Pydantic/dataclass
│   ├── indexing/           # Parsers, chunkers, vector store
│   ├── retrieval/          # BM25, dense, hybrid, reranker, pipeline
│   └── utils/
│       ├── logger.py       # LoggingService (usar exclusivamente)
│       └── file_utils.py
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
└── .env.example
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

## 👨‍💻 Equipe técnica

- **Igor Reis Braziel** — [braziel@discente.ufg.br](mailto:braziel@discente.ufg.br)
- **Gustavo Bueno Ferreira** — [gustavobueno2@discente.ufg.br](mailto:gustavobueno2@discente.ufg.br)
