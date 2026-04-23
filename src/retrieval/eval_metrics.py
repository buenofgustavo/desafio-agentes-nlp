"""Retrieval quality metrics for evaluation.

Both metrics operate on lists of chunk IDs (strings) and return floats in
the range [0, 1]. They are intentionally kept stateless and dependency-free
so they can be unit-tested without any infrastructure.
"""
from __future__ import annotations


def context_recall(
    retrieved_ids: list[str],
    golden_ids: list[str],
) -> float:
    """Fraction of golden chunk IDs present in the retrieved set.

    Measures **coverage**: how many of the expected relevant chunks were
    actually retrieved, regardless of their rank.

    Args:
        retrieved_ids: Chunk IDs returned by the retriever (any order).
        golden_ids: Ground-truth relevant chunk IDs.

    Returns:
        Float in ``[0, 1]``. Returns ``0.0`` when *golden_ids* is empty.

    Example::

        >>> context_recall(["a", "b", "c"], ["a", "c", "d"])
        0.6666666666666666
    """
    if not golden_ids:
        return 0.0
    retrieved_set = set(retrieved_ids)
    hits = sum(1 for gid in golden_ids if gid in retrieved_set)
    return hits / len(golden_ids)


def context_precision(
    retrieved_ids: list[str],
    golden_ids: list[str],
) -> float:
    """Fraction of retrieved chunk IDs that are in the golden set.

    Measures **accuracy**: how many of the returned chunks are actually
    relevant. Penalises noisy or off-topic retrievals.

    Args:
        retrieved_ids: Chunk IDs returned by the retriever.
        golden_ids: Ground-truth relevant chunk IDs.

    Returns:
        Float in ``[0, 1]``. Returns ``0.0`` when *retrieved_ids* is empty.

    Example::

        >>> context_precision(["a", "b", "c"], ["a", "c", "d"])
        0.6666666666666666
    """
    if not retrieved_ids:
        return 0.0
    golden_set = set(golden_ids)
    hits = sum(1 for rid in retrieved_ids if rid in golden_set)
    return hits / len(retrieved_ids)
