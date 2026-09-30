"""Data contracts for the hybrid fusion + cascade reranking component."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class FusionMethod(str, Enum):
    """How multiple ranked lists are merged into one."""

    RRF = "rrf"
    """Reciprocal Rank Fusion: rank-based, ignores raw scores entirely."""
    RELATIVE_SCORE = "relative_score"
    """Min-max normalise each list's scores, then take a weighted sum."""
    DISTRIBUTION = "distribution"
    """Z-score normalise each list's scores, then take a weighted sum."""


@dataclass(frozen=True)
class RankedItem:
    """One retrieved item. `score` is whatever the retriever produced."""

    id: str
    score: float = 0.0
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class RankedList:
    """A single retriever's output. Items must be ordered best-first."""

    source: str
    items: Sequence[RankedItem]
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError("weight must be non-negative")


@dataclass(frozen=True)
class FusedItem:
    """A merged result, carrying per-source attribution for debuggability."""

    id: str
    score: float
    payload: Mapping[str, Any]
    contributions: Mapping[str, float]
    """source name -> that source's contribution to `score`."""
    ranks: Mapping[str, int]
    """source name -> 1-based rank in that source's list."""


@dataclass
class FusionConfig:
    method: FusionMethod = FusionMethod.RRF
    rrf_k: int = 60
    """RRF smoothing constant. 60 is the value from the original TREC work."""
    rerank_budget: int = 0
    """How many fused candidates to pass to the reranker. 0 disables reranking."""
    rerank_weight: float = 1.0
    """1.0 = trust the reranker fully; <1.0 blends it with the fused score."""
    top_k: int = 10
    mmr_lambda: float = 1.0
    """1.0 = pure relevance. <1.0 trades relevance for diversity (needs vectors)."""

    def __post_init__(self) -> None:
        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if not 0.0 <= self.mmr_lambda <= 1.0:
            raise ValueError("mmr_lambda must be in [0, 1]")
        if not 0.0 <= self.rerank_weight <= 1.0:
            raise ValueError("rerank_weight must be in [0, 1]")


@dataclass
class FusionRequest:
    query: str
    ranked_lists: Sequence[RankedList]
    config: FusionConfig = field(default_factory=FusionConfig)
    vectors: Mapping[str, Sequence[float]] | None = None
    """Optional id -> embedding, required only when mmr_lambda < 1."""


@dataclass
class FusionResult:
    items: Sequence[FusedItem]
    method: FusionMethod
    candidates_considered: int
    reranked: bool


class Reranker(Protocol):
    """Anything that can score (query, document) pairs more accurately than retrieval.

    A cross-encoder, a late-interaction MaxSim scorer, or an LLM judge all fit.
    Must return one score per item, in the same order as `items`.
    """

    def score(self, query: str, items: Sequence[FusedItem]) -> Sequence[float]: ...
