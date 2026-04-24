"""Streamlit UI for the RAG demo.

All data fetching is done via HTTP to the FastAPI API at http://localhost:8000.
This file must NOT import from src/agent/ or any heavy backend module directly.

Usage:
    streamlit run app/ui.py
"""
from __future__ import annotations

import httpx
import streamlit as st

# ── Configuration ──────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000"
QUERY_TIMEOUT = 60  # seconds

st.set_page_config(
    page_title="RAG — Setor Elétrico Brasileiro",
    page_icon="⚡",
    layout="wide",
)

# ── Header ─────────────────────────────────────────────────────────────────

st.title("⚡ RAG — Setor Elétrico Brasileiro")
st.caption(
    "Sistema de perguntas e respostas sobre documentos regulatórios do setor elétrico "
    "brasileiro (ANEEL, ONS, PRODIST) com agente LangGraph e recuperação híbrida."
)
st.divider()

# ── Sidebar: metrics ───────────────────────────────────────────────────────

with st.sidebar:
    st.header("📊 Métricas de Avaliação")
    try:
        resp = httpx.get(f"{API_BASE}/metrics", timeout=5)
        data = resp.json()
        if data.get("status") in ("evaluation not run", "error reading report"):
            st.info("Avaliação não executada")
        else:
            # Display numeric values as progress bars
            displayed = False
            for key, value in data.items():
                if isinstance(value, (int, float)) and 0.0 <= value <= 1.0:
                    st.metric(label=key, value=f"{value:.2%}")
                    st.progress(float(value))
                    displayed = True
                elif isinstance(value, (int, float)):
                    st.metric(label=key, value=value)
                    displayed = True
            if not displayed:
                st.json(data)
    except Exception:
        st.warning("API indisponível — métricas não carregadas")

    st.divider()
    st.caption("Powered by LangGraph + Claude + Qdrant")

# ── Health check indicator ─────────────────────────────────────────────────

try:
    health = httpx.get(f"{API_BASE}/health", timeout=5).json()
    qdrant_ok = health.get("qdrant") == "connected"
    col1, col2 = st.columns(2)
    col1.metric("API", "🟢 Online")
    col2.metric("Qdrant", "🟢 Conectado" if qdrant_ok else "🔴 Desconectado")
except Exception:
    st.error("⚠️ API offline — certifique-se de que `make demo` está rodando.")

st.divider()

# ── Query input ────────────────────────────────────────────────────────────

question = st.text_area(
    label="Sua pergunta",
    placeholder="Ex: Qual é o prazo mínimo para revisão tarifária no PRODIST?",
    height=120,
    key="question_input",
)



submit = st.button(
    "⚡ Perguntar",
    type="primary",
    disabled=not question.strip(),
    use_container_width=True,
)

# ── Query execution & response ─────────────────────────────────────────────

if submit and question.strip():
    with st.spinner("Consultando o agente RAG… (pode levar até 60s)"):
        try:
            response = httpx.post(
                f"{API_BASE}/query",
                json={"question": question.strip()},
                timeout=QUERY_TIMEOUT,
            )

            if response.status_code == 200:
                data = response.json()

                # ── Answer ────────────────────────────────────────────────
                st.subheader("💬 Resposta")
                st.info(data["answer"])

                # ── Metrics row ───────────────────────────────────────────
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

                # ── Groundedness badge ────────────────────────────────────
                is_grounded = data.get("is_grounded", False)
                if is_grounded:
                    st.success("✅ Resposta fundamentada nos documentos")
                else:
                    st.warning("⚠️ Resposta pode conter informações não verificadas")

                # ── Sources ───────────────────────────────────────────────
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
