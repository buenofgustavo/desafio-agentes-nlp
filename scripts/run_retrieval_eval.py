#!/usr/bin/env python3
"""Retrieval evaluation script — Phase 2.

Evaluates retrieval quality in isolation (before any LLM is invoked) by
comparing two pipelines on a benchmark of question / golden-chunk pairs:

  Baseline   — ``SemanticSearch`` dense-only, top-5
  Hybrid     — ``RetrievalPipeline`` (BM25 + dense + RRF + cross-encoder), top-5

Metrics per question:
  - context_recall    — fraction of golden IDs found in retrieved set
  - context_precision — fraction of retrieved IDs that are in golden set

Usage::

    python scripts/run_retrieval_eval.py \\
        --benchmark data/processed/benchmark.json \\
        --output    data/processed/retrieval_report.json

    # or via Make:
    make eval-retrieval

Prerequisites:
    1. Qdrant running  (``make up``)
    2. Index populated (``make indexing``)
    3. BM25 index built (``make build-bm25``)
    4. Benchmark annotated (``make generate-benchmark``)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Ensure src/ is importable when called as a top-level script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import Constants
from src.retrieval.eval_metrics import context_precision, context_recall
from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.retrieval.semantic_search import SemanticSearch
from src.utils.logger import LoggingService

logger = LoggingService.setup_logger(__name__)

# Number of final results requested from each pipeline
TOP_K_EVAL = Constants.RERANKER_TOP_K  # 5 by default


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_benchmark(path: Path) -> list[dict]:
    """Load and validate benchmark.json.

    Args:
        path: Absolute or relative path to the benchmark file.

    Returns:
        List of question dicts.

    Raises:
        SystemExit: With code 1 if the file is missing or malformed.
    """
    if not path.exists():
        print(
            f"\n[ERRO CRÍTICO] Arquivo de benchmark não encontrado: {path}\n"
            "Execute primeiro:\n"
            "  python scripts/generate_benchmark.py\n"
            "ou crie o arquivo manualmente em data/processed/benchmark.json",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        print(
            f"[ERRO CRÍTICO] benchmark.json está inválido: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not isinstance(data, list) or not data:
        print(
            "[ERRO CRÍTICO] benchmark.json deve ser uma lista não vazia.",
            file=sys.stderr,
        )
        sys.exit(1)

    return data


def _warn_empty_golden(questions: list[dict]) -> int:
    """Log a warning and return count for questions with no golden IDs."""
    empty = [q for q in questions if not q.get("golden_chunk_ids")]
    if empty:
        logger.warning(
            f"{len(empty)} questão(ões) sem golden_chunk_ids — serão ignoradas nas métricas. "
            "Execute 'make generate-benchmark' para anotar automaticamente."
        )
    return len(empty)


def _print_summary_table(
    num_q: int,
    num_skipped: int,
    baseline: dict[str, float],
    hybrid: dict[str, float],
) -> None:
    """Print a formatted summary table to stdout."""
    sep = "─" * 60
    print(f"\n{'═' * 60}")
    print("  AVALIAÇÃO DE RECUPERAÇÃO — Fase 2".center(60))
    print(f"{'═' * 60}")
    print(f"  Questões avaliadas : {num_q - num_skipped:>4}")
    if num_skipped:
        print(f"  Questões ignoradas : {num_skipped:>4}  (sem golden IDs)")
    print(sep)
    print(f"  {'Métrica':<32} {'Baseline':>10}  {'Hybrid+Re':>10}")
    print(sep)
    print(
        f"  {'Context Recall':<32} "
        f"{baseline['avg_context_recall']:>10.4f}  "
        f"{hybrid['avg_context_recall']:>10.4f}"
    )
    print(
        f"  {'Context Precision':<32} "
        f"{baseline['avg_context_precision']:>10.4f}  "
        f"{hybrid['avg_context_precision']:>10.4f}"
    )
    print(sep)

    # Delta
    dr = hybrid["avg_context_recall"] - baseline["avg_context_recall"]
    dp = hybrid["avg_context_precision"] - baseline["avg_context_precision"]
    dr_str = f"{dr:+.4f}"
    dp_str = f"{dp:+.4f}"
    print(f"  {'Δ Recall (hybrid − baseline)':<32} {dr_str:>22}")
    print(f"  {'Δ Precision (hybrid − baseline)':<32} {dp_str:>22}")
    print(f"{'═' * 60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Avalia a qualidade de recuperação: baseline dense vs. hybrid+reranker"
    )
    parser.add_argument(
        "--benchmark",
        default="data/processed/benchmark.json",
        help="Caminho do arquivo benchmark.json",
    )
    parser.add_argument(
        "--output",
        default="data/processed/retrieval_report.json",
        help="Caminho de saída do relatório JSON",
    )
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark)
    output_path = Path(args.output)

    # ── Load benchmark ───────────────────────────────────────────────────
    questions = _load_benchmark(benchmark_path)
    num_empty = _warn_empty_golden(questions)

    evaluable = [q for q in questions if q.get("golden_chunk_ids")]
    if not evaluable:
        print(
            "\n[ERRO CRÍTICO] Nenhuma questão possui golden_chunk_ids.\n"
            "Execute primeiro: python scripts/generate_benchmark.py",
            file=sys.stderr,
        )
        return 1

    logger.info(
        f"Iniciando avaliação: {len(evaluable)} questões avaliáveis "
        f"(de {len(questions)} no benchmark)"
    )

    # ── Initialize pipelines ─────────────────────────────────────────────
    logger.info("Inicializando Baseline (SemanticSearch dense-only)…")
    baseline_retriever = SemanticSearch(top_k=TOP_K_EVAL)

    logger.info("Inicializando pipeline Hybrid+Reranker (RetrievalPipeline)…")
    pipeline = RetrievalPipeline()

    # ── Evaluate ─────────────────────────────────────────────────────────
    per_question: list[dict] = []
    baseline_recalls: list[float] = []
    baseline_precisions: list[float] = []
    hybrid_recalls: list[float] = []
    hybrid_precisions: list[float] = []

    for idx, item in enumerate(evaluable, start=1):
        qid = item.get("question_id", f"q{idx:03d}")
        question: str = item["question"]
        golden: list[str] = item["golden_chunk_ids"]

        logger.info(f"  [{idx}/{len(evaluable)}] {qid}: «{question[:60]}…»")

        # Baseline: dense-only top-K
        try:
            baseline_results = baseline_retriever.search(
                question, top_k=TOP_K_EVAL, score_threshold=0.0
            )
            baseline_ids = [r.chunk_id for r in baseline_results]
        except Exception as exc:
            logger.error(f"  [{qid}] Baseline falhou: {exc}")
            baseline_ids = []

        # Hybrid + reranker
        try:
            hybrid_results = pipeline.run(question)
            hybrid_ids = [r.chunk_id for r in hybrid_results]
        except Exception as exc:
            logger.error(f"  [{qid}] Hybrid falhou: {exc}")
            hybrid_ids = []

        # Metrics
        b_recall = context_recall(baseline_ids, golden)
        b_prec = context_precision(baseline_ids, golden)
        h_recall = context_recall(hybrid_ids, golden)
        h_prec = context_precision(hybrid_ids, golden)

        baseline_recalls.append(b_recall)
        baseline_precisions.append(b_prec)
        hybrid_recalls.append(h_recall)
        hybrid_precisions.append(h_prec)

        per_question.append(
            {
                "question_id": qid,
                "question": question,
                "baseline": {
                    "context_recall": round(b_recall, 4),
                    "context_precision": round(b_prec, 4),
                    "retrieved_ids": baseline_ids,
                },
                "hybrid": {
                    "context_recall": round(h_recall, 4),
                    "context_precision": round(h_prec, 4),
                    "retrieved_ids": hybrid_ids,
                },
                "golden_chunk_ids": golden,
            }
        )

        logger.info(
            f"  [{qid}] baseline recall={b_recall:.3f} prec={b_prec:.3f} | "
            f"hybrid recall={h_recall:.3f} prec={h_prec:.3f}"
        )

    # ── Aggregate ─────────────────────────────────────────────────────────
    def _avg(lst: list[float]) -> float:
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    baseline_agg = {
        "avg_context_recall": _avg(baseline_recalls),
        "avg_context_precision": _avg(baseline_precisions),
    }
    hybrid_agg = {
        "avg_context_recall": _avg(hybrid_recalls),
        "avg_context_precision": _avg(hybrid_precisions),
    }

    # ── Build report ──────────────────────────────────────────────────────
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "num_questions": len(evaluable),
        "num_skipped": num_empty,
        "top_k": TOP_K_EVAL,
        "baseline_dense_only": baseline_agg,
        "hybrid_with_reranker": hybrid_agg,
        "per_question": per_question,
    }

    # ── Write output ──────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    logger.info(f"Relatório salvo em '{output_path}'")

    # ── Print summary table ───────────────────────────────────────────────
    _print_summary_table(
        num_q=len(questions),
        num_skipped=num_empty,
        baseline=baseline_agg,
        hybrid=hybrid_agg,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
