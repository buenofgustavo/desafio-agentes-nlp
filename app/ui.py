"""Interface Streamlit para a demonstração RAG.

Toda a busca de dados é feita via HTTP para a API FastAPI em http://localhost:8000.
Este arquivo NÃO deve importar de src/agent/ ou de qualquer módulo pesado do backend diretamente.

Uso:
    streamlit run app/ui.py
"""
from __future__ import annotations

import httpx
import streamlit as st

# ── Configuração ──────────────────────────────────────────────────────────
import os

API_BASE = os.getenv("API_URL", "http://api:8000")
QUERY_TIMEOUT = 180  # segundos

st.set_page_config(
    page_title="RAG — Setor Elétrico Brasileiro",
    page_icon="⚡",
    layout="wide",
)

# ── Cabeçalho ───────────────────────────────────────────────────────────────

st.title("⚡ RAG — Setor Elétrico Brasileiro")
st.caption(
    "Sistema de perguntas e respostas sobre documentos regulatórios do setor elétrico "
    "brasileiro (ANEEL) com agente LangGraph e recuperação híbrida."
)
st.divider()

# ── Barra lateral ──────────────────────────────────────────────────────────

with st.sidebar:
    st.caption("Powered by LangGraph + Claude + Qdrant")

# ── Indicador de saúde do sistema ──────────────────────────────────────────

try:
    health = httpx.get(f"{API_BASE}/health", timeout=5).json()
    qdrant_ok = health.get("qdrant") == "connected"
    col1, col2 = st.columns(2)
    col1.metric("API", "🟢 Online")
    col2.metric("Qdrant", "🟢 Conectado" if qdrant_ok else "🔴 Desconectado")
except Exception:
    st.error("⚠️ API offline.")

st.divider()

# ── Entrada de pergunta ───────────────────────────────────────────────────

question = st.text_area(
    label="Sua pergunta",
    height=120,
    key="question_input",
)

submit = st.button(
    "⚡ Perguntar",
    type="primary",
    disabled=not question.strip(),
    use_container_width=True,
)

# ── Execução da consulta e resposta ────────────────────────────────────────

if submit and question.strip():
    with st.spinner("Consultando o agente RAG…"):
        try:
            response = httpx.post(
                f"{API_BASE}/query",
                json={"question": question.strip()},
                timeout=QUERY_TIMEOUT,
            )

            if response.status_code == 200:
                data = response.json()

                # ── Resposta ──────────────────────────────────────────────
                st.subheader("💬 Resposta")
                st.markdown(data["answer"])

                # ── Métricas ──────────────────────────────────────────────
                m1, m2, m3 = st.columns(3)
                m1.metric(
                    "Tipo de consulta",
                    data.get("query_type", "—").replace("_", " ").title(),
                )
                faithfulness = data.get("faithfulness_score")
                m2.metric(
                    "Score de fidelidade",
                    f"{faithfulness:.0%}" if faithfulness is not None else "N/A",
                )
                m3.metric(
                    "Latência",
                    f"{data.get('latency_seconds', 0):.1f}s",
                )

                # ── Badge de fundamentação ────────────────────────────────
                is_grounded = data.get("is_grounded", False)
                if is_grounded:
                    st.success("✅ Resposta fundamentada nos documentos")
                else:
                    st.warning("⚠️ Resposta pode conter informações não verificadas")

                # ── Fontes ────────────────────────────────────────────────
                sources = data.get("sources", [])
                if sources:
                    with st.expander(f"📚 Fontes ({len(sources)} chunks)"):
                        for i, src in enumerate(sources, 1):
                            doc = src.get("doc_name") or src.get("document", "—")
                            section = src.get("section", "—")
                            page = src.get("page", "—")
                            score = src.get("rerank_score")
                            score_str = f" | score: {score:.4f}" if score else ""
                            st.markdown(
                                f"**{i}.** `{doc}` — seção `{section}`, "
                                f"pág. `{page}`{score_str}"
                            )

            elif response.status_code == 422:
                detail = response.json().get("detail", "Entrada inválida")
                st.error(f"Erro de validação: {detail}")
            else:
                detail = response.json().get("detail", response.text)
                st.error(f"Erro ao consultar o agente: {detail}")

        except httpx.TimeoutException:
            st.error(
                "Erro ao consultar o agente: timeout após "
                f"{QUERY_TIMEOUT}s. O agente ainda está inicializando?"
            )
        except Exception as exc:
            st.error(f"Erro ao consultar o agente: {exc}")
