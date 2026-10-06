"""Data contracts for the extraction harness.

The decision that shapes this module is how ground truth is expressed: a
**mapping from path to expected value**, where a path is the same string
`extraction` already puts on every `FieldResult` — `total`,
`line_items[3].amount`.

The alternative is an expected JSON object per case. It reads better and is
worse to work with: a case written that way must restate every field to assert
one of them, so a golden set grows faster than the attention available to keep
it right, and a schema change invalidates every case rather than the cases that
mention the changed field. Paths let a case assert exactly what was checked by
hand, and `line_items` is the one key whose value is a list of rows, because a
table is scored by matching rather than by equality.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .metrics import LineItemComparison, LineItemScore, WrongCell


@dataclass
class ExtractionCase:
    """One document with the values a human verified on it.

    `expected` maps a path to the value that path should hold. A path whose
    expected value is a list of mappings is a table, scored with precision and
    recall instead of equality.
    """

    case_id: str
    document: str = ""
    """How the runner's `load_source` hook should find this document — a path,
    a URI, an id. Unused when `source_text` is set."""
    source_text: str = ""
    """The document inline. Keeps a golden set self-contained for the cases
    where the text is short enough, which is what lets the harness be tested
    without a corpus on disk."""
    expected: Mapping[str, Any] = field(default_factory=dict)
    required_paths: Sequence[str] = field(default_factory=list)
    """Paths that must be right for an accepted result not to be a silent
    error. Empty means every expected path is required: a case that marks
    nothing required would make the silent error rate structurally zero, which
    is a worse default than treating everything as mattering."""
    tags: Sequence[str] = field(default_factory=list)
    notes: str = ""
    """Why this case is in the set. Written for the person who finds it failing
    in eight months."""

    def __post_init__(self) -> None:
        if not self.case_id or not self.case_id.strip():
            raise ValueError("every case needs a case_id")
        if not self.expected:
            raise ValueError(
                "case "
                + self.case_id
                + " expects nothing, so it would score 1.0 on everything and"
                " measure nothing; give it at least one expected path"
            )

    def required(self) -> list[str]:
        return list(self.required_paths) if self.required_paths else list(self.expected)


@dataclass
class ExtractionGolden:
    """A named set of cases, one per document."""

    name: str
    cases: Sequence[ExtractionCase]

    @staticmethod
    def from_jsonl(path: str, name: str | None = None) -> ExtractionGolden:
        cases: list[ExtractionCase] = []
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
                cases.append(ExtractionCase(**raw))
        if not cases:
            raise ValueError("no cases found in " + path)
        return ExtractionGolden(name=name or path, cases=cases)

    def to_jsonl(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            for case in self.cases:
                handle.write(
                    json.dumps(
                        {
                            "case_id": case.case_id,
                            "document": case.document,
                            "source_text": case.source_text,
                            "expected": dict(case.expected),
                            "required_paths": list(case.required_paths),
                            "tags": list(case.tags),
                            "notes": case.notes,
                        },
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n"
                )


@dataclass
class ExtractionEvalConfig:
    line_items_path: str = "line_items"
    """Which expected key holds the table. Only used to name it in the report;
    any path whose expected value is a list of mappings is scored as one."""
    description_keys: Sequence[str] = ("description", "desc", "item", "name", "label")
    amount_keys: Sequence[str] = ("amount", "total", "line_total", "value", "price")
    """Row identity, surfaced here so a schema that names its columns
    differently needs no new code."""
    record_raw_responses: bool = False
    """Off by default: a report with every raw completion in it is large, and
    the completions are the one part that cannot be diffed usefully."""


@dataclass
class FieldOutcome:
    """One path, what was expected, what came back, and whether it grounded."""

    path: str
    expected: Any
    actual: Any
    correct: bool
    grounded: bool = False
    match: str = ""
    """The `MatchClass` value from extraction, as a string so the report is
    plain JSON. Distinguishes "wrong and fabricated" from "wrong and quoted",
    which need different fixes."""
    required: bool = False


@dataclass
class ExtractionCaseResult:
    case_id: str
    outcomes: Sequence[FieldOutcome]
    schema_valid: bool = True
    accepted: bool = True
    line_items: LineItemScore | None = None
    line_item_cells: LineItemComparison | None = None
    """Per-cell detail behind the row score.

    `line_items` says whether the rows were found; this says whether each of
    their cells was right, per column. A report keeping only the first cannot
    answer "which column is the model weakest on", which is the question that
    says what to change in the prompt."""
    attempts: int = 1
    truncated: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0
    issues: Sequence[str] = field(default_factory=list)
    raw_responses: Sequence[str] = field(default_factory=list)

    @property
    def field_accuracy(self) -> float:
        from .metrics import field_accuracy as _accuracy

        return _accuracy(self.outcomes)

    @property
    def required_wrong(self) -> list[str]:
        return [o.path for o in self.outcomes if o.required and not o.correct]

    @property
    def silent_error(self) -> bool:
        """Accepted, and wrong where it mattered.

        The one number this unit exists for. A rejected wrong result cost a
        retry; this one cost whatever acting on it costs, and nothing anywhere
        said so.
        """
        return self.accepted and bool(self.required_wrong)

    @property
    def correct(self) -> bool:
        """The per-case verdict the regression diff uses."""
        return self.schema_valid and not self.required_wrong


@dataclass
class ExtractionEvalReport:
    dataset: str
    metrics: Mapping[str, float]
    cases: Sequence[ExtractionCaseResult]
    config: Mapping[str, Any] = field(default_factory=dict)
    """A snapshot of what produced these numbers. Without it a diff can tell you
    something changed but never what."""
    created_at: str = ""

    def case_map(self) -> dict[str, ExtractionCaseResult]:
        return {case.case_id: case for case in self.cases}

    @property
    def silent_errors(self) -> list[str]:
        return [case.case_id for case in self.cases if case.silent_error]

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            # default=str because expected and actual values include Decimal,
            # which json cannot serialise. A harness whose report cannot be
            # written is a harness with no history.
            json.dump(
                {
                    "dataset": self.dataset,
                    "metrics": dict(self.metrics),
                    "config": dict(self.config),
                    "created_at": self.created_at,
                    "cases": [_case_to_dict(case) for case in self.cases],
                },
                handle,
                ensure_ascii=False,
                indent=2,
                default=str,
            )

    @staticmethod
    def from_json(path: str) -> ExtractionEvalReport:
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
        return ExtractionEvalReport(
            dataset=raw["dataset"],
            metrics=raw.get("metrics", {}),
            cases=[_case_from_dict(c) for c in raw.get("cases", [])],
            config=raw.get("config", {}),
            created_at=raw.get("created_at", ""),
        )


def _case_to_dict(case: ExtractionCaseResult) -> dict[str, Any]:
    rows = case.line_items
    cells = case.line_item_cells
    return {
        "case_id": case.case_id,
        "schema_valid": case.schema_valid,
        "accepted": case.accepted,
        "attempts": case.attempts,
        "truncated": case.truncated,
        "input_tokens": case.input_tokens,
        "output_tokens": case.output_tokens,
        "cost_usd": case.cost_usd,
        "seconds": case.seconds,
        "issues": list(case.issues),
        "raw_responses": list(case.raw_responses),
        "line_items": (
            None
            if rows is None
            else {
                "expected_rows": rows.expected_rows,
                "extracted_rows": rows.extracted_rows,
                "matched": rows.matched,
            }
        ),
        "line_item_cells": (
            None
            if cells is None
            else {
                "expected_rows": cells.expected_rows,
                "extracted_rows": cells.extracted_rows,
                "matched_rows": cells.matched_rows,
                "cells_compared": dict(cells.cells_compared),
                "cells_correct": dict(cells.cells_correct),
                "wrong_cells": [
                    {"row": c.row, "column": c.column,
                     "expected": c.expected, "actual": c.actual}
                    for c in cells.wrong_cells
                ],
            }
        ),
        "outcomes": [
            {
                "path": o.path,
                "expected": o.expected,
                "actual": o.actual,
                "correct": o.correct,
                "grounded": o.grounded,
                "match": o.match,
                "required": o.required,
            }
            for o in case.outcomes
        ],
    }


def _case_from_dict(raw: Mapping[str, Any]) -> ExtractionCaseResult:
    rows = raw.get("line_items")
    cells = raw.get("line_item_cells")
    return ExtractionCaseResult(
        case_id=raw["case_id"],
        outcomes=[FieldOutcome(**o) for o in raw.get("outcomes", [])],
        schema_valid=bool(raw.get("schema_valid", True)),
        accepted=bool(raw.get("accepted", True)),
        line_items=None if rows is None else LineItemScore(**rows),
        line_item_cells=None if cells is None else LineItemComparison(
            expected_rows=int(cells["expected_rows"]),
            extracted_rows=int(cells["extracted_rows"]),
            matched_rows=int(cells["matched_rows"]),
            wrong_cells=[WrongCell(**c) for c in cells.get("wrong_cells", [])],
            cells_compared=dict(cells.get("cells_compared", {})),
            cells_correct=dict(cells.get("cells_correct", {})),
        ),
        attempts=int(raw.get("attempts", 1)),
        truncated=bool(raw.get("truncated", False)),
        input_tokens=int(raw.get("input_tokens", 0)),
        output_tokens=int(raw.get("output_tokens", 0)),
        cost_usd=float(raw.get("cost_usd", 0.0)),
        seconds=float(raw.get("seconds", 0.0)),
        issues=list(raw.get("issues", [])),
        raw_responses=list(raw.get("raw_responses", [])),
    )


@dataclass(frozen=True)
class ExtractionMetricDelta:
    name: str
    before: float
    after: float

    @property
    def change(self) -> float:
        return self.after - self.before


@dataclass
class ExtractionRegressionDiff:
    """What changed between two runs, at both the aggregate and the case level.

    Aggregate numbers say whether a change helped on average. The per-case
    lists say which documents it broke, which is the part that tells you why -
    and a change that lifts field accuracy while breaking two cases is a change
    worth looking at rather than shipping.
    """

    deltas: Sequence[ExtractionMetricDelta]
    fixed: Sequence[str]
    broken: Sequence[str]
    unchanged: int
    before_config: Mapping[str, Any] = field(default_factory=dict)
    after_config: Mapping[str, Any] = field(default_factory=dict)

    @property
    def net(self) -> int:
        return len(self.fixed) - len(self.broken)

    def render(self) -> str:
        lines = ["metric                           before    after     change"]
        for delta in self.deltas:
            lines.append(
                delta.name.ljust(32)
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


__all__ = [
    "ExtractionCase",
    "ExtractionCaseResult",
    "ExtractionEvalConfig",
    "ExtractionEvalReport",
    "ExtractionGolden",
    "ExtractionMetricDelta",
    "ExtractionRegressionDiff",
    "FieldOutcome",
]
