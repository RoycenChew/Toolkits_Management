"""Ranking metrics over a binary relevance vector.

Small enough to own, and worth owning: these are the numbers every retrieval
decision is argued with, so you should be able to read their definitions in
twenty seconds rather than trust a dependency.

Every function takes `relevance` — a list of 0/1 in rank order, best first — so
they compose with any retriever without knowing anything about it.

The subtlety that matters is recall's denominator. `total_relevant` is the number
of relevant items *in the corpus*, which a golden set usually does not know. When
it is unknown, "recall" computed against the retrieved set alone is really
precision wearing a different name, and reporting it as recall overstates the
system. This module makes `total_relevant` explicit and optional, and names the
fallback honestly: `hit_rate`.
"""
from __future__ import annotations

import math
from collections.abc import Sequence


def _truncate(relevance: Sequence[int], k: int | None) -> list[int]:
    if k is None:
        return list(relevance)
    if k < 1:
        raise ValueError("k must be at least 1")
    return list(relevance[:k])


def hit_rate(relevance: Sequence[int], k: int | None = None) -> float:
    """1.0 if any relevant item appears in the top k, else 0.0.

    The most honest metric when the golden set lists 'a passage that answers
    this' rather than 'every passage that could'. Averaged over cases it reads as
    'how often did we surface something useful at all'.
    """
    return 1.0 if any(_truncate(relevance, k)) else 0.0


def precision_at_k(relevance: Sequence[int], k: int) -> float:
    window = _truncate(relevance, k)
    if not window:
        return 0.0
    return sum(window) / len(window)


def recall_at_k(
    relevance: Sequence[int], k: int, total_relevant: int | None = None
) -> float:
    """Fraction of all relevant items retrieved in the top k.

    Falls back to `hit_rate` when `total_relevant` is unknown, because a recall
    figure with a made-up denominator is worse than no figure at all.
    """
    if total_relevant is None:
        return hit_rate(relevance, k)
    if total_relevant <= 0:
        return 0.0
    return min(1.0, sum(_truncate(relevance, k)) / total_relevant)


def reciprocal_rank(relevance: Sequence[int], k: int | None = None) -> float:
    """1 / rank of the first relevant item. 0.0 if none.

    Averaged across cases this is MRR. It is the metric that tracks what a user
    actually experiences, because it cares enormously about position one.
    """
    for index, value in enumerate(_truncate(relevance, k), start=1):
        if value:
            return 1.0 / index
    return 0.0


def dcg_at_k(relevance: Sequence[int], k: int) -> float:
    return sum(
        value / math.log2(index + 1)
        for index, value in enumerate(_truncate(relevance, k), start=1)
    )


def ndcg_at_k(
    relevance: Sequence[int], k: int, total_relevant: int | None = None
) -> float:
    """DCG normalised by the best achievable ordering.

    The ideal ranking puts every relevant item first. When `total_relevant` is
    known it is used, so a system that retrieved two of five relevant items is
    not credited as perfect for ordering those two well.
    """
    actual = dcg_at_k(relevance, k)
    ideal_count = total_relevant if total_relevant is not None else sum(relevance)
    ideal = dcg_at_k([1] * min(ideal_count, k), k)
    if ideal <= 0:
        return 0.0
    return actual / ideal


def average_precision(
    relevance: Sequence[int], total_relevant: int | None = None
) -> float:
    """Mean of precision@i at every rank i holding a relevant item."""
    hits = 0
    total = 0.0
    for index, value in enumerate(relevance, start=1):
        if value:
            hits += 1
            total += hits / index
    denominator = total_relevant if total_relevant is not None else hits
    if not denominator:
        return 0.0
    return total / denominator


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


__all__ = [
    "average_precision",
    "dcg_at_k",
    "hit_rate",
    "mean",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
