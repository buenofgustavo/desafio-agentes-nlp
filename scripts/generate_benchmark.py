#!/usr/bin/env python3
"""Populate benchmark.json with golden chunk IDs from Qdrant.

This script auto-labels the benchmark by querying Qdrant with the dense
retriever (top-10).  The returned chunk IDs become the ``golden_chunk_ids``
for evaluation.  This is a *self-supervised* annotation strategy — it is
intentionally biased toward the dense retriever but provides an objective
starting point for comparing different pipelines.

Run ONCE before ``make eval-retrieval``::

    python scripts/generate_benchmark.py
    python scripts/generate_benchmark.py --benchmark data/processed/benchmark.json \\
                                          --top-k 10

Requirements:
    - Qdrant must be running (``make up``).
    - The index must be populated (``make indexing``).
    - The embedding model must be available.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure src/ is importable when called as a top-level script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.semantic_search import SemanticSearch
from src.utils.logger import LoggingService

logger = LoggingService.setup_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Popula golden_chunk_ids no benchmark via busca densa"
    )
    parser.add_argument(
        "--benchmark",
        default="data/processed/benchmark.json",
        help="Caminho do arquivo benchmark.json (default: data/processed/benchmark.json)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Número de chunks recuperados por questão (default: 10)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Sobrescreve golden_chunk_ids existentes (default: pula questões já anotadas)",
    )
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark)
    if not benchmark_path.exists():
        print(
            f"[ERRO] Benchmark não encontrado: {benchmark_path}\n"
            "Certifique-se de que o arquivo existe antes de executar este script.",
            file=sys.stderr,
        )
        return 1

    with open(benchmark_path, "r", encoding="utf-8") as fh:
        questions: list[dict] = json.load(fh)

    logger.info(
        f"Carregando SemanticSearch para anotar {len(questions)} questão(ões)…"
    )
    retriever = SemanticSearch(top_k=args.top_k)

    updated = 0
    for item in questions:
        qid = item.get("question_id", "?")
        existing = item.get("golden_chunk_ids", [])

        if existing and not args.overwrite:
            logger.info(f"  [{qid}] já anotado ({len(existing)} IDs) — pulado")
            continue

        query: str = item["question"]
        results = retriever.search(query, top_k=args.top_k, score_threshold=0.0)
        ids = [r.chunk_id for r in results]
        item["golden_chunk_ids"] = ids
        updated += 1
        logger.info(f"  [{qid}] anotado com {len(ids)} chunk(s)")

    with open(benchmark_path, "w", encoding="utf-8") as fh:
        json.dump(questions, fh, ensure_ascii=False, indent=2)

    logger.info(
        f"Benchmark salvo em '{benchmark_path}' "
        f"({updated} questão(ões) atualizada(s))"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
