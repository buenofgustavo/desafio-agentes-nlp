"""CLI entrypoint para executar o agente RAG.

Uso:
    # Modo interativo
    python scripts/run_agent.py

    # Consulta única
    python scripts/run_agent.py --query "Qual é a tensão nominal da rede?"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.graph import agent_graph  
from src.agent.state import initial_state  
from src.utils.logger import LoggingService  

logger = LoggingService.setup_logger(__name__)


# ── Auxiliares ─────────────────────────────────────────────────────────────


def _run_single_query(query: str) -> dict:
    """Executa o agente em uma única consulta e retorna o estado final completo."""
    state = initial_state(query)
    result = agent_graph.invoke(state)
    return result


def _format_response(result: dict) -> str:
    """Formata uma resposta estruturada para saída no terminal."""
    grounded = result.get("is_grounded", False)
    score = result.get("faithfulness_score")
    score_str = f"{score:.2f}" if score is not None else "N/A"
    check_mark = "✓" if grounded else "✗"

    sources = result.get("sources", [])
    sources_lines = []
    for i, src in enumerate(sources, 1):
        doc = src.get("doc_name", "?")
        page = src.get("page", "?")
        sources_lines.append(f"  [{i}] {doc}, p. {page}")

    return (
        f"\n{'='*60}\n"
        f"Tipo de consulta : {result.get('query_type', '?')}\n"
        f"Chunks usados    : {len(sources)}\n"
        f"Fidelidade       : {score_str} {check_mark}\n"
        f"Tentativas       : {result.get('faithfulness_retries', 0)}\n"
        f"Rodada recup.    : {result.get('retrieval_round', 1)}\n"
        f"{'='*60}\n\n"
        f"Resposta:\n{result.get('final_answer') or result.get('answer', '(sem resposta)')}\n\n"
        f"Fontes:\n"
        + ("\n".join(sources_lines) if sources_lines else "  (nenhuma fonte)")
        + f"\n{'='*60}\n"
    )


# ── Modos de Execução ──────────────────────────────────────────────────────


def interactive_mode() -> None:
    """Executa o agente em modo interativo (REPL)."""
    print("\n🤖 Agente RAG — Setor Elétrico Brasileiro")
    print("   Digite 'sair' ou 'exit' para encerrar.\n")

    while True:
        try:
            query = input("Pergunta: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando.")
            break

        if not query or query.lower() in ("sair", "exit", "quit"):
            print("Encerrando.")
            break

        t0 = time.time()
        try:
            result = _run_single_query(query)
            elapsed = time.time() - t0
            print(_format_response(result))
            print(f"  ⏱  {elapsed:.1f}s\n")
        except Exception:
            logger.error("Erro ao processar query", exc_info=True)
            print("  ❌ Erro ao processar. Veja o log para detalhes.\n")


def single_query_mode(query: str) -> None:
    """Executa o agente em uma consulta única e imprime a saída estruturada."""
    logger.info("Modo de consulta única: '%s'", query[:80])
    t0 = time.time()
    result = _run_single_query(query)
    elapsed = time.time() - t0
    print(_format_response(result))
    print(f"  ⏱  {elapsed:.1f}s")


def batch_mode(batch_path: str, output_path: str) -> None:
    """Executa o agente em todas as perguntas de um arquivo JSON de benchmark."""
    batch_file = Path(batch_path)
    if not batch_file.exists():
        print(f"❌ Arquivo não encontrado: {batch_path}")
        sys.exit(1)

    with open(batch_file, encoding="utf-8") as f:
        data = json.load(f)

    # Normaliza formato de entrada
    questions: list[dict] = []
    if isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, str):
                questions.append({"question_id": f"q{i+1:03d}", "question": item})
            elif isinstance(item, dict):
                q = item.get("question") or item.get("pergunta", "")
                qid = item.get("question_id") or item.get("id", f"q{i+1:03d}")
                questions.append({"question_id": qid, "question": q})

    logger.info("Modo lote: %d perguntas carregadas de %s", len(questions), batch_path)
    print(f"\n📋 Processando {len(questions)} perguntas...\n")

    answers: list[dict] = []
    t0_total = time.time()

    for i, q in enumerate(questions, 1):
        question = q["question"]
        qid = q["question_id"]
        print(f"  [{i}/{len(questions)}] {qid}: {question[:60]}...")

        t0 = time.time()
        try:
            result = _run_single_query(question)
            elapsed = time.time() - t0

            answer_record = {
                "question_id": qid,
                "question": question,
                "query_type": result.get("query_type", "simple"),
                "final_answer": result.get("final_answer")
                or result.get("answer", ""),
                "sources": result.get("sources", []),
                "faithfulness_score": result.get("faithfulness_score"),
                "faithfulness_retries": result.get("faithfulness_retries", 0),
                "is_grounded": result.get("is_grounded", False),
                "retrieval_round": result.get("retrieval_round", 1),
                "elapsed_seconds": round(elapsed, 1),
            }
            answers.append(answer_record)

            grounded_mark = "✓" if answer_record["is_grounded"] else "✗"
            score = answer_record["faithfulness_score"]
            score_str = f"{score:.2f}" if score is not None else "N/A"
            print(f"    → {answer_record['query_type']} | "
                  f"fidel={score_str} {grounded_mark} | "
                  f"{elapsed:.1f}s")
        except Exception:
            logger.error("Erro na pergunta %s", qid, exc_info=True)
            answers.append({
                "question_id": qid,
                "question": question,
                "error": "Falha ao processar",
            })
            print("    → ❌ Erro")

    # Salva resultados
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)

    elapsed_total = time.time() - t0_total
    grounded_count = sum(1 for a in answers if a.get("is_grounded"))
    avg_score = [
        a["faithfulness_score"]
        for a in answers
        if a.get("faithfulness_score") is not None
    ]

    print(f"\n{'='*60}")
    print(f"📊 Resumo do Lote")
    print(f"   Total     : {len(answers)}")
    print(f"   Fundam.   : {grounded_count}/{len(answers)}")
    if avg_score:
        print(f"   Avg faith : {sum(avg_score)/len(avg_score):.2f}")
    print(f"   Tempo     : {elapsed_total:.1f}s")
    print(f"   Salvo em  : {out_file}")
    print(f"{'='*60}\n")


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    """Faz o parse dos argumentos e despacha para o modo apropriado."""
    parser = argparse.ArgumentParser(
        description="Agente RAG — Setor Elétrico Brasileiro",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Executa uma consulta única e imprime o resultado.",
    )
    parser.add_argument(
        "--batch", "-b",
        type=str,
        help="Caminho para um arquivo JSON de benchmark com uma lista de perguntas.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/processed/agent_answers.json",
        help="Caminho de saída para os resultados em lote (padrão: data/processed/agent_answers.json).",
    )

    args = parser.parse_args()

    if args.query:
        single_query_mode(args.query)
    elif args.batch:
        batch_mode(args.batch, args.output)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
