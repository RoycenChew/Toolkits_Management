"""Hybrid retrieval fusion with an optional reranking cascade.

Extracted pattern, not extracted code: Reciprocal Rank Fusion (Cormack et al.),
the relative-score / distribution-based fusion variants used by Qdrant and
Weaviate, and the cheap-recall -> expensive-rerank cascade that ColBERT/PLAID
and SPLADE pipelines rely on. Reimplemented from the algorithms, so the result
carries no upstream licence or dependency.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .models import (
    FusedItem,
    FusionConfig,
    FusionMethod,
    FusionRequest,
    FusionResult,
    RankedList,
    Reranker,
)


def _minmax(scores: Sequence[float]) -> list[float]:
    """Scale to [0, 1]. A flat list maps to all-1.0 rather than dividing by zero."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    span = hi - lo
    if span <= 0:
        return [1.0] * len(scores)
    return [(s - lo) / span for s in scores]


def _zscore(scores: Sequence[float]) -> list[float]:
    """Standardise to mean 0 / sd 1. Zero variance maps to all-0.0."""
    n = len(scores)
    if n == 0:
        return []
    mean = sum(scores) / n
    var = sum((s - mean) ** 2 for s in scores) / n
    sd = math.sqrt(var)
    if sd <= 0:
        return [0.0] * n
    return [(s - mean) / sd for s in scores]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return num / (na * nb)


class HybridRankerComponent:
    """Merge several ranked lists, then optionally rerank and diversify.

    The point of rank-based fusion (RRF) is that it needs no score calibration:
    a BM25 score of 14.2 and a cosine similarity of 0.81 are not comparable, but
    their ranks are. The score-based methods exist for the cases where you do
    trust the scale within each individual list.
    """

    def execute(
        self, input_data: FusionRequest, reranker: Reranker | None = None
    ) -> FusionResult:
        cfg = input_data.config
        lists = [rl for rl in input_data.ranked_lists if rl.items and rl.weight > 0]
        if not lists:
            return FusionResult(
                items=[], method=cfg.method, candidates_considered=0, reranked=False
            )

        fused = self._fuse(lists, cfg)

        reranked = False
        if reranker is not None and cfg.rerank_budget > 0:
            head = fused[: cfg.rerank_budget]
            tail = fused[cfg.rerank_budget :]
            fused = self._rerank(input_data.query, head, reranker, cfg) + tail
            reranked = True

        if cfg.mmr_lambda < 1.0 and input_data.vectors:
            fused = self._mmr(fused, input_data.vectors, cfg)

        return FusionResult(
            items=fused[: cfg.top_k],
            method=cfg.method,
            candidates_considered=len({i.id for rl in lists for i in rl.items}),
            reranked=reranked,
        )

    # --- fusion ----------------------------------------------------------

    def _fuse(self, lists: Sequence[RankedList], cfg: FusionConfig) -> list[FusedItem]:
        totals: dict[str, float] = defaultdict(float)
        contributions: dict[str, dict[str, float]] = defaultdict(dict)
        ranks: dict[str, dict[str, int]] = defaultdict(dict)
        payloads: dict[str, dict[str, Any]] = {}

        for rl in lists:
            per_item = self._contributions(rl, cfg)
            for rank, (item, value) in enumerate(zip(rl.items, per_item), start=1):
                weighted = value * rl.weight
                totals[item.id] += weighted
                contributions[item.id][rl.source] = weighted
                ranks[item.id][rl.source] = rank
                # First list to supply a key wins; later lists only fill gaps.
                merged = payloads.setdefault(item.id, {})
                for key, val in item.payload.items():
                    merged.setdefault(key, val)

        fused = [
            FusedItem(
                id=item_id,
                score=score,
                payload=payloads.get(item_id, {}),
                contributions=dict(contributions[item_id]),
                ranks=dict(ranks[item_id]),
            )
            for item_id, score in totals.items()
        ]
        # Tie-break on id so ordering is deterministic across runs.
        fused.sort(key=lambda f: (-f.score, f.id))
        return fused

    def _contributions(self, rl: RankedList, cfg: FusionConfig) -> list[float]:
        if cfg.method is FusionMethod.RRF:
            return [1.0 / (cfg.rrf_k + rank) for rank in range(1, len(rl.items) + 1)]
        raw = [i.score for i in rl.items]
        if cfg.method is FusionMethod.RELATIVE_SCORE:
            return _minmax(raw)
        return _zscore(raw)

    # --- cascade ---------------------------------------------------------

    def _rerank(
        self,
        query: str,
        head: Sequence[FusedItem],
        reranker: Reranker,
        cfg: FusionConfig,
    ) -> list[FusedItem]:
        if not head:
            return []
        scores = list(reranker.score(query, head))
        if len(scores) != len(head):
            raise ValueError(
                "reranker returned %d scores for %d items" % (len(scores), len(head))
            )
        # Blend on a common [0,1] scale so rerank_weight means what it says.
        rr = _minmax(scores)
        retrieval = _minmax([f.score for f in head])
        w = cfg.rerank_weight
        out = [
            FusedItem(
                id=f.id,
                score=w * r + (1.0 - w) * base,
                payload=f.payload,
                contributions=dict(f.contributions, _rerank=r),
                ranks=f.ranks,
            )
            for f, r, base in zip(head, rr, retrieval)
        ]
        out.sort(key=lambda f: (-f.score, f.id))
        return out

    # --- diversification -------------------------------------------------

    def _mmr(
        self,
        items: Sequence[FusedItem],
        vectors: Mapping[str, Sequence[float]],
        cfg: FusionConfig,
    ) -> list[FusedItem]:
        """Maximal Marginal Relevance: greedily take the item that is relevant
        but least similar to what is already selected. Items with no vector are
        appended at the end rather than silently dropped.
        """
        pool = [f for f in items if f.id in vectors]
        missing = [f for f in items if f.id not in vectors]
        if not pool:
            return list(items)

        rel = dict(zip([f.id for f in pool], _minmax([f.score for f in pool])))
        lam = cfg.mmr_lambda
        selected: list[FusedItem] = []
        remaining = list(pool)

        while remaining and len(selected) < cfg.top_k:
            best = None
            best_val = -math.inf
            for cand in remaining:
                penalty = max(
                    [_cosine(vectors[cand.id], vectors[s.id]) for s in selected] or [0.0]
                )
                val = lam * rel[cand.id] - (1.0 - lam) * penalty
                if val > best_val:
                    best = cand
                    best_val = val
            if best is None:
                break
            selected.append(best)
            remaining.remove(best)

        return selected + remaining + missing


__all__ = ["HybridRankerComponent"]
