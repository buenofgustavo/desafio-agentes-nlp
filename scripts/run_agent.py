"""CLI entrypoint for running the RAG agent.

Usage:
    # Interactive mode
    python scripts/run_agent.py

    # Single query
    python scripts/run_agent.py --query "Qual é a tensão nominal da rede?"

    # Batch mode (JSON file with list of questions)
    python scripts/run_agent.py --batch data/processed/benchmark.json \\
        --output results/answers.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.graph import agent_graph  # noqa: E402
from src.agent.state import initial_state  # noqa: E402
from src.utils.logger import LoggingService  # noqa: E402

logger = LoggingService.setup_logger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────


def _run_single_query(query: str) -> dict:
    """Run the agent on a single query and return the full final state."""
    state = initial_state(query)
    result = agent_graph.invoke(state)
    return result


def _format_response(result: dict) -> str:
    """Format a structured response for terminal output."""
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
        f"Query type    : {result.get('query_type', '?')}\n"
        f"Chunks used   : {len(sources)}\n"
        f"Faithfulness  : {score_str} {check_mark}\n"
        f"Retries       : {result.get('faithfulness_retries', 0)}\n"
        f"Retrieval rnd : {result.get('retrieval_round', 1)}\n"
        f"{'='*60}\n\n"
        f"Answer:\n{result.get('final_answer') or result.get('answer', '(sem resposta)')}\n\n"
        f"Sources:\n"
        + ("\n".join(sources_lines) if sources_lines else "  (nenhuma fonte)")
        + f"\n{'='*60}\n"
    )


# ── Modes ──────────────────────────────────────────────────────────────────


def interactive_mode() -> None:
    """Run the agent in interactive REPL mode."""
    print("\n🤖 RAG Agent — Setor Elétrico Brasileiro")
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
    """Run the agent on a single query and print structured output."""
    logger.info("Single query mode: '%s'", query[:80])
    t0 = time.time()
    result = _run_single_query(query)
    elapsed = time.time() - t0
    print(_format_response(result))
    print(f"  ⏱  {elapsed:.1f}s")


def batch_mode(batch_path: str, output_path: str) -> None:
    """Run the agent on all questions from a JSON benchmark file.

    Expected input format::

        [
            {"question": "...", "question_id": "q001"},
            ...
        ]

    Or flat list of strings::

        ["question 1", "question 2", ...]
    """
    batch_file = Path(batch_path)
    if not batch_file.exists():
        print(f"❌ Arquivo não encontrado: {batch_path}")
        sys.exit(1)

    with open(batch_file, encoding="utf-8") as f:
        data = json.load(f)

    # Normalize input format
    questions: list[dict] = []
    if isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, str):
                questions.append({"question_id": f"q{i+1:03d}", "question": item})
            elif isinstance(item, dict):
                q = item.get("question") or item.get("pergunta", "")
                qid = item.get("question_id") or item.get("id", f"q{i+1:03d}")
                questions.append({"question_id": qid, "question": q})

    logger.info("Batch mode: %d perguntas carregadas de %s", len(questions), batch_path)
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
                  f"faith={score_str} {grounded_mark} | "
                  f"{elapsed:.1f}s")
        except Exception:
            logger.error("Erro na pergunta %s", qid, exc_info=True)
            answers.append({
                "question_id": qid,
                "question": question,
                "error": "Falha ao processar",
            })
            print("    → ❌ Erro")

    # Save results
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
    print(f"📊 Resumo do Batch")
    print(f"   Total     : {len(answers)}")
    print(f"   Grounded  : {grounded_count}/{len(answers)}")
    if avg_score:
        print(f"   Avg faith : {sum(avg_score)/len(avg_score):.2f}")
    print(f"   Tempo     : {elapsed_total:.1f}s")
    print(f"   Salvo em  : {out_file}")
    print(f"{'='*60}\n")


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    """Parse arguments and dispatch to the appropriate mode."""
    parser = argparse.ArgumentParser(
        description="RAG Agent — Setor Elétrico Brasileiro",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Run a single query and print the result.",
    )
    parser.add_argument(
        "--batch", "-b",
        type=str,
        help="Path to a JSON benchmark file with a list of questions.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/processed/agent_answers.json",
        help="Output path for batch results (default: data/processed/agent_answers.json).",
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
