"""Data contracts for the evaluation harness.

The single most important decision in this module is how ground truth is
expressed. The obvious choice — list the `chunk_id`s that should be retrieved —
is wrong, because chunk ids change whenever the chunking configuration changes,
and re-tuning chunking is exactly the experiment you most want to evaluate. A
golden set keyed on chunk ids invalidates itself the moment it becomes useful.

So relevance is expressed in terms that survive re-chunking: a snippet of text
that must appear in a relevant passage, or the document and page it must come
from. Both are stable under any chunking strategy, and both are things a human
can author while reading the source document.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EvalCase:
    """One question with its expected outcome.

    A case is relevant-matched if *any* of `expected_snippets` appears in a
    retrieved chunk, or the chunk's doc/page matches. Snippets are the easiest to
    author and the most durable; pages are useful when the answer is a table or a
    figure whose text you would rather not transcribe.
    """

    case_id: str
    query: str
    expected_snippets: Sequence[str] = field(default_factory=list)
    expected_doc_ids: Sequence[str] = field(default_factory=list)
    expected_pages: Sequence[int] = field(default_factory=list)
    expected_answer_contains: Sequence[str] = field(default_factory=list)
    """Substrings the generated answer must contain. Crude, but it catches the
    regressions that matter and needs no judge model."""
    unanswerable: bool = False
    """The corpus genuinely does not cover this. A correct system refuses.
    Without these cases you cannot measure the relevance gate at all, and a
    system that answers everything will score well on a set of answerable
    questions alone."""
    tags: Sequence[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.query or not self.query.strip():
            raise ValueError("case " + self.case_id + " has an empty query")
        if not self.unanswerable and not (
            self.expected_snippets or self.expected_doc_ids or self.expected_pages
        ):
            raise ValueError(
                "case "
                + self.case_id
                + " is answerable but declares no expectation; give it a snippet,"
                " a doc id or a page, or mark it unanswerable"
            )


@dataclass
class EvalDataset:
    name: str
    cases: Sequence[EvalCase]

    @staticmethod
    def from_jsonl(path: str, name: str | None = None) -> EvalDataset:
        cases: list[EvalCase] = []
        with open(path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text or text.startswith("#"):
                    continue
                try:
                    raw = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "invalid JSON on line " + str(line_number) + " of " + path
                    ) from exc
                raw.setdefault("case_id", "case-" + str(line_number))
                cases.append(EvalCase(**raw))
        if not cases:
            raise ValueError("no cases found in " + path)
        return EvalDataset(name=name or path, cases=cases)

    def to_jsonl(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            for case in self.cases:
                handle.write(json.dumps(asdict(case), ensure_ascii=False) + "\n")


@dataclass
class EvalConfig:
    k_values: Sequence[int] = (1, 3, 5, 10)
    """Cut-offs to report. Reporting several is deliberate: a change can improve
    recall@10 while hurting recall@1, and only one of those reaches the user."""
    top_k: int = 5
    """Chunks the pipeline puts in the prompt for each case."""
    candidates_per_retriever: int = 20
    evaluate_answers: bool = True
    """Off when no LLM is configured, or when you only care about retrieval."""

    relevance_gate: bool = True
    min_dense_similarity: float | None = None
    """Retrieval knobs surfaced here so a sweep can vary them between runs and the
    diff records what changed. Without these on the eval config, comparing gate
    settings means reaching around the harness to patch the pipeline."""

    lexical_weight: float = 1.0
    dense_weight: float = 1.0
    rrf_k: int = 60
    rerank_budget: int = 0
    """Fusion knobs, for the same reason.

    Without them the harness could not answer the one question `hybrid_ranker`
    exists to settle - does fusing two retrievers beat either alone - because
    there was no way to turn a retriever off between runs. Setting
    `lexical_weight=0.0` or `dense_weight=0.0` gives the single-retriever
    baselines, so an ablation is three eval runs and a `diff_reports` call
    instead of a bespoke script.
    """


@dataclass
class CaseResult:
    case_id: str
    query: str
    retrieved_ids: Sequence[str]
    relevance: Sequence[int]
    """Binary judgement per retrieved chunk, in rank order."""
    unanswerable: bool = False
    refused: bool = False
    grounded: bool = True
    answer: str = ""
    answer_hits: Sequence[str] = field(default_factory=list)
    answer_misses: Sequence[str] = field(default_factory=list)
    unverified_markers: Sequence[int] = field(default_factory=list)
    ground_truth_missing: bool = False
    """No indexed chunk satisfies this case's expectations.

    The case is unscoreable rather than failed: nothing could have retrieved it.
    Counted as a `dataset_error` and excluded from the retrieval metrics, so an
    extraction fault or a mis-transcribed snippet is not reported as a ranking
    problem.
    """
    cited_relevant: bool = False
    """Did the model cite at least one chunk that was actually relevant? Higher
    bar than 'a relevant chunk was retrieved', and closer to what a reader sees."""
    seconds: float = 0.0

    @property
    def correct(self) -> bool:
        """The single per-case verdict used for the regression diff."""
        if self.unanswerable:
            return self.refused
        if self.refused:
            return False
        if self.answer_misses:
            return False
        return any(self.relevance)


@dataclass
class EvalReport:
    dataset: str
    metrics: Mapping[str, float]
    cases: Sequence[CaseResult]
    config: Mapping[str, Any] = field(default_factory=dict)
    """A snapshot of what produced these numbers. Without it a diff can tell you
    something changed but never what."""
    created_at: str = ""

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "dataset": self.dataset,
                    "metrics": dict(self.metrics),
                    "config": dict(self.config),
                    "created_at": self.created_at,
                    "cases": [asdict(c) for c in self.cases],
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

    @staticmethod
    def from_json(path: str) -> EvalReport:
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
        return EvalReport(
            dataset=raw["dataset"],
            metrics=raw["metrics"],
            cases=[CaseResult(**c) for c in raw.get("cases", [])],
            config=raw.get("config", {}),
            created_at=raw.get("created_at", ""),
        )


@dataclass
class MetricDelta:
    name: str
    before: float
    after: float

    @property
    def change(self) -> float:
        return self.after - self.before


@dataclass
class RegressionDiff:
    """What changed between two runs, at both the aggregate and the case level.

    Aggregate numbers say whether a change helped on average. The per-case lists
    say which questions it broke, which is the part that tells you why.
    """

    deltas: Sequence[MetricDelta]
    fixed: Sequence[str]
    broken: Sequence[str]
    unchanged: int
    before_config: Mapping[str, Any] = field(default_factory=dict)
    after_config: Mapping[str, Any] = field(default_factory=dict)

    @property
    def net(self) -> int:
        return len(self.fixed) - len(self.broken)

    def render(self) -> str:
        lines = ["metric                     before    after     change"]
        for delta in self.deltas:
            lines.append(
                delta.name.ljust(26)
                + format(delta.before, ".3f").rjust(7)
                + format(delta.after, ".3f").rjust(10)
                + format(delta.change, "+.3f").rjust(11)
            )
        lines.append("")
        lines.append(
            "cases fixed: "
            + str(len(self.fixed))
            + "  broken: "
            + str(len(self.broken))
            + "  unchanged: "
            + str(self.unchanged)
        )
        if self.broken:
            lines.append("broken: " + ", ".join(self.broken[:10]))
        if self.fixed:
            lines.append("fixed:  " + ", ".join(self.fixed[:10]))
        changed = {
            key: (self.before_config.get(key), self.after_config.get(key))
            for key in set(self.before_config) | set(self.after_config)
            if self.before_config.get(key) != self.after_config.get(key)
        }
        if changed:
            lines.append("")
            lines.append("config changes:")
            for key, (before, after) in sorted(changed.items()):
                lines.append("  " + key + ": " + repr(before) + " -> " + repr(after))
        return "\n".join(lines)
