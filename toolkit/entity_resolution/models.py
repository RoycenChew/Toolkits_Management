"""Data contracts for the entity resolution component."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

# Aliased because two dataclasses here have an attribute literally named `field`
# (the record column being compared). The annotation shadows the import for any
# reader, and for a type checker, even though it happens to work at runtime.
from dataclasses import field as dataclass_field
from typing import Any

Record = Mapping[str, Any]
"""One row to be matched. Values are compared as strings unless a custom
comparator is supplied for that field."""


@dataclass(frozen=True)
class Predicate:
    """A cheap, deterministic blocking key generator.

    A predicate maps one record to zero or more block keys. Two records are
    considered a candidate pair if they share at least one key under at least
    one selected predicate. Predicates trade precision for speed: they exist
    only to avoid the O(n^2) full comparison.
    """

    name: str
    field: str
    fn: Callable[[str], Sequence[str]]

    def keys(self, record: Record) -> list[str]:
        raw = record.get(self.field)
        if raw is None:
            return []
        text = str(raw).strip().lower()
        if not text:
            return []
        return [self.name + ":" + self.field + ":" + k for k in self.fn(text) if k]


@dataclass(frozen=True)
class ComparisonLevel:
    """One discrete outcome of comparing a field, in the Fellegi-Sunter sense.

    `threshold` is the minimum similarity (0..1) required to reach this level.
    Levels for a field are evaluated best-first, so order them descending.
    """

    label: str
    threshold: float


@dataclass(frozen=True)
class FieldComparison:
    """How a single field contributes evidence to the match decision."""

    field: str
    levels: Sequence[ComparisonLevel] = dataclass_field(
        default_factory=lambda: (
            ComparisonLevel("exact", 1.0),
            ComparisonLevel("close", 0.88),
            ComparisonLevel("similar", 0.70),
        )
    )
    comparator: Callable[[str, str], float] | None = None
    """Returns similarity in [0, 1]. Defaults to normalised affine-gap distance."""


@dataclass(frozen=True)
class CandidatePair:
    left: str
    right: str
    shared_keys: int
    """How many blocking keys the two records had in common. A weak prior."""


@dataclass(frozen=True)
class ScoredPair:
    left: str
    right: str
    match_probability: float
    match_weight: float
    """log2 Bayes factor. Positive favours match, negative favours non-match.
    This is the interpretable number: +4 means 16x more likely to be a match."""
    pattern: Mapping[str, str]
    """field -> the comparison level reached. The explanation of the score."""


@dataclass(frozen=True)
class EntityCluster:
    cluster_id: int
    record_ids: Sequence[str]
    cohesion: float
    """Mean match probability of the within-cluster pairs that were scored."""


@dataclass
class TrainedModel:
    """Learned Fellegi-Sunter parameters. Serialisable and reusable."""

    lambda_prior: float
    m_probabilities: Mapping[str, Mapping[str, float]]
    """field -> level label -> P(level | records match)."""
    u_probabilities: Mapping[str, Mapping[str, float]]
    """field -> level label -> P(level | records do not match)."""
    iterations: int
    converged: bool


@dataclass
class ResolutionConfig:
    comparisons: Sequence[FieldComparison]
    predicates: Sequence[Predicate] | None = None
    """None selects the built-in predicate library."""
    max_block_size: int = 200
    """Blocks larger than this are dropped: they are almost always a degenerate
    key (an empty string, a default value) and would dominate the pair budget."""
    max_predicates: int = 6
    """How many predicates the greedy cover is allowed to select."""
    em_iterations: int = 30
    em_tolerance: float = 1e-5
    match_threshold: float = 0.9
    """Minimum match probability for a pair to join a cluster."""
    cluster_link: str = "average"
    """'average' for average-linkage agglomeration, 'connected' for transitive
    closure. Connected components are faster but chain unrelated records."""


@dataclass
class ResolutionRequest:
    records: Mapping[str, Record]
    config: ResolutionConfig
    labelled_pairs: Sequence[tuple[str, str, bool]] = ()
    """Optional (left, right, is_match) triples. Used to train the blocking
    predicate cover. Matching itself stays unsupervised."""


@dataclass
class ResolutionResult:
    clusters: Sequence[EntityCluster]
    scored_pairs: Sequence[ScoredPair]
    model: TrainedModel
    selected_predicates: Sequence[str]
    pairs_compared: int
    pairs_avoided: int
    """How many comparisons blocking saved versus the full n(n-1)/2 cross join."""
