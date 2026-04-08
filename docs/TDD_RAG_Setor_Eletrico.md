**TECHNICAL DESIGN DOCUMENT**

**Sistema RAG para o Setor Elétrico Brasileiro**

Pipeline de Recuperação de Informação com Agentes de IA

<table>
<colgroup>
<col style="width: 23%" />
<col style="width: 2%" />
<col style="width: 23%" />
<col style="width: 2%" />
<col style="width: 23%" />
<col style="width: 2%" />
<col style="width: 23%" />
</colgroup>
<tbody>
<tr class="odd">
<td><p><strong>Versão</strong></p>
<p><strong>1.0</strong></p></td>
<td></td>
<td><p><strong>Status</strong></p>
<p><strong>Em Desenvolvimento</strong></p></td>
<td></td>
<td><p><strong>Domínio</strong></p>
<p><strong>Setor Elétrico BR</strong></p></td>
<td></td>
<td><p><strong>Tipo</strong></p>
<p><strong>RAG + Agentes</strong></p></td>
</tr>
</tbody>
</table>

**1. Visão Geral do Projeto**

**1.1 Contexto**

Este documento descreve a arquitetura técnica e o plano de implementação de um sistema agêntico de Recuperação Aumentada por Geração (RAG) aplicado a documentos reais do setor elétrico brasileiro. O projeto é desenvolvido no contexto de um grupo de estudos em IA, com foco na construção de uma pipeline robusta de recuperação de informação.

**1.2 Objetivo**

Construir um sistema capaz de responder perguntas técnicas com base em um corpus de documentos do setor elétrico, avaliado por um benchmark de perguntas anotadas por especialista.

**1.3 Critérios de Sucesso**

- Alta acurácia nas respostas ao benchmark do especialista

- Pipeline de retrieval que supere busca semântica simples (baseline)

- Respostas fundamentadas (faithfulness) --- sem alucinações

- Sistema modular e reproduzível por qualquer membro do time

**1.4 Fora do Escopo**

- Interface de usuário elaborada (apenas UI mínima para demo)

- Deploy em produção ou escalabilidade

- Fine-tuning de modelos

**2. Arquitetura do Sistema**

**2.1 Visão de Alto Nível**

O sistema é composto por dois subsistemas principais: a Pipeline de Indexação (offline) e a Pipeline de Consulta (online), coordenados por uma camada agêntica.

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<tbody>
<tr class="odd">
<td><p><strong>PIPELINE DE INDEXAÇÃO (offline)</strong></p>
<p>Documentos PDF/Word → Parser (Docling/PyMuPDF) → Chunker → Embedder → Vector Store (Qdrant)</p>
<p>+ BM25 Index</p>
<p><strong>PIPELINE DE CONSULTA (online)</strong></p>
<p>Pergunta → Query Expansion → Retriever Híbrido → Reranker → LLM (Claude) → Resposta</p>
<p>(BM25 + Dense) (Cross-Encoder)</p></td>
</tr>
</tbody>
</table>

**2.2 Componentes Principais**

| **Camada**   | **Tecnologia Escolhida** | **Justificativa**                                                    |
|--------------|--------------------------|----------------------------------------------------------------------|
| Parsing      | Docling + PyMuPDF        | Docling para layouts complexos/tabelas; PyMuPDF como fallback rápido |
| Chunking     | Semântico + Hierárquico  | Preserva contexto de seções longas típicas de normas técnicas        |
| Embeddings   | multilingual-e5-large    | Gratuito, multilíngue, alta performance em português                 |
| Vector Store | Qdrant (Docker)          | Open-source, suporte a filtros de metadados, escalável               |
| Retrieval    | Híbrido BM25 + Dense     | BM25 captura termos técnicos exatos; dense captura semântica         |
| Reranker     | cross-encoder/ms-marco   | Maior ganho de qualidade com menor esforço de implementação          |
| LLM          | Claude Sonnet via API    | Contexto longo, ótimo raciocínio em domínios técnicos                |
| Orquestração | LangGraph + LlamaIndex   | LangGraph para controle do fluxo; LlamaIndex para abstrações RAG     |
| Backend      | FastAPI                  | Leve, assíncrono, fácil de expor endpoints de consulta               |
| Avaliação    | RAGAS + LangSmith        | Métricas padrão de RAG + tracing para debug do pipeline              |

**3. Pipeline Detalhada**

**3.1 Ingestão e Parsing de Documentos**

**Estratégia de parsing adaptativa:** o sistema detecta o tipo de documento e escolhe o parser adequado automaticamente.

- Docling: documentos com tabelas complexas, layouts multi-coluna, figuras

- PyMuPDF: PDFs simples, extração rápida de texto corrido

- Tesseract OCR: documentos escaneados ou com imagens de texto

**Metadados extraídos por chunk:** nome do documento, número de página, seção, tipo de conteúdo (tabela/texto/figura). Esses metadados são usados para filtros no retrieval.

**3.2 Estratégia de Chunking**

Dois níveis de granularidade são mantidos em paralelo no vector store:

- Parent chunks (800-1000 tokens): preservam contexto amplo de seções inteiras

- Child chunks (150-200 tokens): granularidade fina para retrieval preciso

O retrieval busca pelos child chunks, mas retorna o parent chunk completo ao LLM --- técnica conhecida como Parent-Document Retrieval. Isso garante precisão no match e contexto suficiente na geração.

**3.3 Retrieval Híbrido**

O sistema combina dois sinais de relevância com fusão por Reciprocal Rank Fusion (RRF):

- BM25 (lexical): captura termos técnicos exatos como siglas, normas (ANEEL, ONS, PRODIST), códigos de equipamentos

- Dense retrieval (semântico): captura intenção e paráfrases mesmo sem match exato de termos

**Top-K de cada retriever:** 20 resultados → fusão RRF → top 10 → reranker → top 5 enviados ao LLM.

**3.4 Reranking**

**Modelo:** cross-encoder/ms-marco-multilingual-rerank-mmarco-v2 (HuggingFace, gratuito). O reranker avalia cada par (pergunta, chunk) de forma conjunta, produzindo um score de relevância muito mais preciso que embeddings independentes.

**3.5 Query Expansion**

Antes do retrieval, o agente executa uma etapa de expansão da query:

- HyDE (Hypothetical Document Embeddings): gera uma resposta hipotética e usa seu embedding para busca

- Reformulação: o LLM gera 2-3 variações da pergunta original para cobertura maior

**4. Fluxo Agêntico (LangGraph)**

O agente é implementado como um grafo de estados com os seguintes nós:

| **\#** | **Nó**             | **Responsabilidade**                                                         |
|--------|--------------------|------------------------------------------------------------------------------|
| 1      | query_analyzer     | Classifica a pergunta: factual simples, comparativa, ou que requer multi-hop |
| 2      | query_expander     | Executa HyDE e reformulação da query                                         |
| 3      | retriever          | Retrieval híbrido BM25 + dense, fusão RRF                                    |
| 4      | reranker           | Reordena chunks por relevância com cross-encoder                             |
| 5      | context_assembler  | Monta o contexto final, remove duplicatas, aplica limite de tokens           |
| 6      | generator          | Chama o LLM com o contexto e gera a resposta fundamentada                    |
| 7      | faithfulness_check | Verifica se a resposta está ancorada nos documentos (loop de auto-correção)  |

Para perguntas classificadas como multi-hop, o agente executa múltiplas rodadas de retrieval antes de gerar a resposta final.

**5. Estrutura do Projeto**

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<tbody>
<tr class="odd">
<td><p>rag-setor-eletrico/</p>
<p>├── data/</p>
<p>│ ├── raw/ # documentos originais (PDFs)</p>
<p>│ └── processed/ # chunks + metadados em JSON</p>
<p>├── src/</p>
<p>│ ├── ingestion/</p>
<p>│ │ ├── parser.py # Docling / PyMuPDF / OCR</p>
<p>│ │ └── chunker.py # chunking semântico + hierárquico</p>
<p>│ ├── indexing/</p>
<p>│ │ ├── embedder.py # multilingual-e5-large</p>
<p>│ │ ├── bm25.py # índice BM25</p>
<p>│ │ └── vector_store.py # Qdrant client</p>
<p>│ ├── retrieval/</p>
<p>│ │ ├── hybrid.py # fusão RRF</p>
<p>│ │ ├── reranker.py # cross-encoder</p>
<p>│ │ └── query_expansion.py</p>
<p>│ ├── agent/</p>
<p>│ │ ├── graph.py # LangGraph state machine</p>
<p>│ │ └── nodes.py # cada nó do agente</p>
<p>│ └── api/</p>
<p>│ └── main.py # FastAPI endpoints</p>
<p>├── evaluation/</p>
<p>│ ├── benchmark.py # runner do benchmark</p>
<p>│ └── metrics.py # RAGAS metrics</p>
<p>├── docker-compose.yml # Qdrant + API</p>
<p>├── notebooks/ # experimentos e análises</p>
<p>└── README.md</p></td>
</tr>
</tbody>
</table>

**6. Roadmap de Implementação**

| **Fase** | **Nome**          | **Entregas**                                                                                                | **Duração** |
|----------|-------------------|-------------------------------------------------------------------------------------------------------------|-------------|
| Fase 1   | Fundação          | Setup do repo, Docker (Qdrant), parser básico (PyMuPDF), chunking fixo, embeddings, busca semântica simples | 2-3 dias    |
| Fase 2   | Retrieval Robusto | BM25 index, retrieval híbrido com RRF, reranker cross-encoder, avaliação do retrieval isolado               | 3-4 dias    |
| Fase 3   | Parsing Avançado  | Docling para tabelas complexas, chunking hierárquico (parent-child), metadados ricos por chunk              | 2-3 dias    |
| Fase 4   | Agente LangGraph  | Query expansion (HyDE), grafo de estados, faithfulness check, multi-hop para queries complexas              | 3-4 dias    |
| Fase 5   | Avaliação Final   | RAGAS completo, runner do benchmark, análise de erros, ajustes finos de parâmetros (K, threshold)           | 2-3 dias    |
| Fase 6   | Demo              | FastAPI endpoint, UI mínima (Streamlit), README com instruções de execução                                  | 1-2 dias    |

**Estratégia recomendada:** implementar o baseline completo (Fase 1) antes de qualquer otimização. Só adicionar complexidade onde os dados do benchmark mostrarem deficiência real.

**7. Estratégia de Avaliação**

**7.1 Métricas do Benchmark**

| **Métrica**        | **O que mede**                         | **Ferramenta**      |
|--------------------|----------------------------------------|---------------------|
| Context Recall     | O retriever buscou os chunks certos?   | RAGAS               |
| Context Precision  | Os chunks retornados são relevantes?   | RAGAS               |
| Faithfulness       | A resposta está fundamentada nos docs? | RAGAS               |
| Answer Relevancy   | A resposta endereça a pergunta?        | RAGAS               |
| Answer Correctness | Match com gabarito do especialista     | Custom (similarity) |

**7.2 Processo de Avaliação Iterativa**

- Fase 1: avaliar retrieval isolado (antes de rodar o LLM) --- mede Context Recall/Precision

- Fase 2: avaliar o sistema end-to-end com todas as métricas RAGAS

- Fase 3: análise qualitativa das perguntas erradas --- identificar padrões de falha

- Fase 4: ajuste fino de hiperparâmetros (K, pesos BM25 vs dense, threshold de reranker)

**7.3 Baseline para Comparação**

Toda otimização deve ser comparada contra o baseline de busca semântica pura (dense-only, sem reranker) para justificar a complexidade adicionada.

**8. Decisões Técnicas e Trade-offs**

| **Decisão**  | **Escolha Feita**                       | **Alternativa Descartada**                                       |
|--------------|-----------------------------------------|------------------------------------------------------------------|
| Embeddings   | multilingual-e5-large (local, gratuito) | text-embedding-3-small (OpenAI, pago, ligeiramente melhor em EN) |
| Vector Store | Qdrant (Docker local)                   | Pinecone (managed, sem custo fixo, mas latência de rede)         |
| Chunking     | Hierárquico parent-child                | Chunking fixo por tokens (mais simples, perda de contexto)       |
| Reranker     | Cross-encoder local (HuggingFace)       | Cohere Rerank (melhor, mas pago por request)                     |
| Orquestração | LangGraph                               | LangChain LCEL puro (menos controle de fluxo condicional)        |
| LLM          | Claude Sonnet                           | GPT-4o (similar performance, custo comparável)                   |

**9. Configuração do Ambiente**

**9.1 Pré-requisitos**

- Python 3.11+

- Docker e Docker Compose

- Chave de API da Anthropic (Claude)

- \~8 GB de RAM (para o modelo de embeddings local)

- \~10 GB de disco (Qdrant + modelos HuggingFace)

**9.2 Variáveis de Ambiente**

| **Variável**      | **Descrição**                                                        |
|-------------------|----------------------------------------------------------------------|
| ANTHROPIC_API_KEY | Chave de API para o Claude                                           |
| QDRANT_URL        | URL do Qdrant (default: http://localhost:6333)                       |
| EMBEDDING_MODEL   | Nome do modelo HuggingFace (default: intfloat/multilingual-e5-large) |
| RERANKER_MODEL    | Modelo de reranking (default: cross-encoder/ms-marco-multilingual)   |
| LANGCHAIN_API_KEY | Chave do LangSmith para tracing (opcional)                           |

**9.3 Comandos de Inicialização**

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<tbody>
<tr class="odd">
<td><p># 1. Subir Qdrant</p>
<p>docker-compose up -d</p>
<p># 2. Instalar dependências Python</p>
<p>pip install -r requirements.txt</p>
<p># 3. Indexar documentos</p>
<p>python -m src.ingestion.pipeline --input data/raw/</p>
<p># 4. Iniciar API</p>
<p>uvicorn src.api.main:app --reload</p>
<p># 5. Rodar benchmark de avaliação</p>
<p>python evaluation/benchmark.py --questions data/benchmark.json</p></td>
</tr>
</tbody>
</table>

Grupo de Estudos em IA • Technical Design Document • v1.0
