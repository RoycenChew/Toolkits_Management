"""Run a golden set through an extractor and report what it cost to be wrong.

The acceptance predicate is the part worth arguing about. Every other metric
here is a property of the extraction; `silent_error_rate` is a property of the
extraction **and the gate in front of it**, and that gate is a product
decision. A stricter gate turns silent errors into loud ones at the cost of
rejecting more work, and the only way to see that trade is to be able to vary
it between runs — so the predicate is injected and recorded in the report
config, where the diff will show it changed.

The default is `result.valid`: the schema was satisfied and no issue remains.
That is the weakest honest gate, and it is the one most pipelines ship with.
"""
from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from ..core.models import Usage
from .metrics import (
    compare_line_items,
    field_accuracy,
    grounding_rate,
    line_item_scores,
    mean,
    values_equal,
)
from .models import (
    ExtractionCase,
    ExtractionCaseResult,
    ExtractionEvalConfig,
    ExtractionEvalReport,
    ExtractionGolden,
    ExtractionMetricDelta,
    ExtractionRegressionDiff,
    FieldOutcome,
)

_INDEX = re.compile(r"^([^\[\]]*)\[(\d+)\]$")


def value_at(data: Any, path: str) -> Any:
    """Resolve `line_items[3].amount` against a plain object.

    Returns None for anything the path does not reach, which is the same answer
    as "the model returned null there" on purpose: from the point of view of a
    reader of the output, a field that is absent and a field that is null are
    the same disappointment.
    """
    current: Any = data
    for raw in path.split("."):
        if current is None:
            return None
        match = _INDEX.match(raw)
        key, index = (match.group(1), int(match.group(2))) if match else (raw, None)
        if key:
            if not isinstance(current, Mapping) or key not in current:
                return None
            current = current[key]
        if index is not None:
            if not isinstance(current, Sequence) or isinstance(current, (str, bytes)):
                return None
            if index >= len(current):
                return None
            current = current[index]
    return current


class ExtractionEvalRunner:
    """Golden set in, report out. Knows nothing about where documents live.

    `load_source` is how a case reaches a real document, and it is a hook
    rather than a built-in loader because resolving `document` is the one part
    of this that cannot be stdlib-only: a PDF needs an adapter. A case with
    `source_text` needs no hook at all, which is what keeps this unit testable
    without a corpus.
    """

    def __init__(
        self,
        component: Any,
        schema: Any,
        extraction_config: Any = None,
        load_source: Callable[[ExtractionCase], Any] | None = None,
        accept: Callable[[Any], bool] | None = None,
        instructions: str = "",
    ) -> None:
        if component is None:
            raise ValueError("ExtractionEvalRunner needs an extraction component")
        if schema is None:
            raise ValueError("ExtractionEvalRunner needs a schema")
        self._component = component
        self._schema = schema
        self._extraction_config = extraction_config
        self._load_source = load_source
        self._accept = accept or (lambda result: bool(result.valid))
        self._instructions = instructions

    def execute(
        self,
        golden: ExtractionGolden,
        config: ExtractionEvalConfig | None = None,
    ) -> ExtractionEvalReport:
        settings = config or ExtractionEvalConfig()
        results = [self._run_case(case, settings) for case in golden.cases]
        return ExtractionEvalReport(
            dataset=golden.name,
            metrics=self._aggregate(results),
            cases=results,
            config=self._config_snapshot(settings),
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    # --- one case ---------------------------------------------------------

    def _source(self, case: ExtractionCase) -> Any:
        if case.source_text:
            return case.source_text
        if self._load_source is None:
            raise ValueError(
                "case "
                + case.case_id
                + " has no source_text, so the runner needs a load_source hook to"
                " reach " + (case.document or "its document")
            )
        return self._load_source(case)

    def _run_case(
        self, case: ExtractionCase, settings: ExtractionEvalConfig
    ) -> ExtractionCaseResult:
        from ..extraction import ExtractionRequest

        source = self._source(case)
        request = (
            ExtractionRequest(
                schema=self._schema,
                source=source,
                config=self._extraction_config,
                instructions=self._instructions,
            )
            if self._extraction_config is not None
            else ExtractionRequest(
                schema=self._schema, source=source, instructions=self._instructions
            )
        )

        started = time.monotonic()
        result = self._component.execute(request)
        seconds = time.monotonic() - started

        leaves = result.leaf_map() if hasattr(result, "leaf_map") else {}
        required = set(case.required())
        outcomes: list[FieldOutcome] = []
        rows = None
        cells = None

        for path, expected in case.expected.items():
            if _is_table(expected):
                actual_rows = value_at(result.data, path)
                present = actual_rows if isinstance(actual_rows, Sequence) else []
                rows = line_item_scores(
                    expected, present,
                    settings.description_keys, settings.amount_keys,
                )
                cells = compare_line_items(
                    expected,
                    [r for r in present if isinstance(r, Mapping)],
                    description_keys=settings.description_keys,
                )
                continue
            leaf = leaves.get(path)
            actual = leaf.value if leaf is not None else value_at(result.data, path)
            outcomes.append(
                FieldOutcome(
                    path=path,
                    expected=expected,
                    actual=actual,
                    correct=values_equal(expected, actual),
                    grounded=bool(leaf.grounded) if leaf is not None else False,
                    match=(
                        getattr(leaf.match, "value", str(leaf.match))
                        if leaf is not None
                        else ""
                    ),
                    required=path in required,
                )
            )

        usage = getattr(result, "usage", None) or Usage()
        return ExtractionCaseResult(
            case_id=case.case_id,
            outcomes=outcomes,
            schema_valid=bool(result.valid),
            accepted=bool(self._accept(result)),
            line_items=rows,
            line_item_cells=cells,
            attempts=int(getattr(result, "attempts", 1)),
            truncated=bool(getattr(result, "truncated", False)),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=usage.cost_usd,
            seconds=seconds,
            issues=[issue.render() for issue in result.issues],
            raw_responses=(
                list(getattr(result, "raw_responses", []))
                if settings.record_raw_responses
                else []
            ),
        )

    # --- aggregation ------------------------------------------------------

    def _aggregate(self, results: Sequence[ExtractionCaseResult]) -> dict[str, float]:
        accepted = [r for r in results if r.accepted]
        silent = [r for r in results if r.silent_error]
        with_rows = [r.line_items for r in results if r.line_items is not None]

        matched = sum(r.matched for r in with_rows)
        extracted = sum(r.extracted_rows for r in with_rows)
        expected = sum(r.expected_rows for r in with_rows)
        # Micro-averaged: every row weighs the same, so a 40-row invoice counts
        # for more than a 2-row one. Macro-averaging over cases would let one
        # tiny document outvote a large table, which is backwards for a metric
        # about rows.
        precision = (matched / extracted) if extracted else 0.0
        recall = (matched / expected) if expected else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0

        all_outcomes = [o for r in results for o in r.outcomes]
        return {
            "cases": float(len(results)),
            "field_accuracy": field_accuracy(all_outcomes),
            "grounding_rate": grounding_rate(all_outcomes),
            "schema_validity_rate": mean([1.0 if r.schema_valid else 0.0 for r in results]),
            "acceptance_rate": mean([1.0 if r.accepted else 0.0 for r in results]),
            # Two denominators, named, because the ambiguity is otherwise
            # silent: over all cases the number is comparable between runs, and
            # over accepted ones it is "of what you shipped, how much was
            # wrong". A stricter gate improves the first and can worsen the
            # second, and that is exactly the trade worth seeing.
            "silent_error_rate": (len(silent) / len(results)) if results else 0.0,
            "silent_error_rate_of_accepted": (
                (len(silent) / len(accepted)) if accepted else 0.0
            ),
            "line_item_precision": precision,
            "line_item_recall": recall,
            "line_item_f1": f1,
            "mean_attempts": mean([float(r.attempts) for r in results]),
            "truncated_cases": float(sum(1 for r in results if r.truncated)),
            "total_input_tokens": float(sum(r.input_tokens for r in results)),
            "total_output_tokens": float(sum(r.output_tokens for r in results)),
            "total_cost_usd": float(sum(r.cost_usd for r in results)),
            "mean_seconds": mean([r.seconds for r in results]),
        }

    def _config_snapshot(self, settings: ExtractionEvalConfig) -> dict[str, Any]:
        extraction = self._extraction_config
        snapshot: dict[str, Any] = {
            "schema": getattr(self._schema, "name", ""),
            "line_items_path": settings.line_items_path,
            "accept": getattr(self._accept, "__name__", "custom"),
            "instructions": self._instructions,
        }
        for name in (
            "max_repairs",
            "temperature",
            "require_grounding",
            "context_char_limit",
            "date_order",
            "coerce",
        ):
            if extraction is not None and hasattr(extraction, name):
                snapshot[name] = getattr(extraction, name)
        model = getattr(getattr(self._component, "_llm", None), "model_version", None)
        if model:
            snapshot["model_version"] = model
        return snapshot


def _is_table(expected: Any) -> bool:
    if isinstance(expected, (Mapping, str, bytes)):
        return False
    if not isinstance(expected, Sequence):
        return False
    return all(isinstance(row, Mapping) for row in expected)


def diff_extraction_reports(
    before: ExtractionEvalReport, after: ExtractionEvalReport
) -> ExtractionRegressionDiff:
    """Compare two runs at the aggregate and per-case level.

    Only metrics present in both are compared, so adding a metric does not
    fabricate a delta against a run that never measured it.
    """
    shared = [key for key in after.metrics if key in before.metrics]
    deltas = [
        ExtractionMetricDelta(
            name=key,
            before=float(before.metrics[key]),
            after=float(after.metrics[key]),
        )
        for key in sorted(shared)
    ]

    previous = before.case_map()
    fixed: list[str] = []
    broken: list[str] = []
    unchanged = 0
    for case in after.cases:
        earlier = previous.get(case.case_id)
        if earlier is None:
            continue
        if case.correct and not earlier.correct:
            fixed.append(case.case_id)
        elif earlier.correct and not case.correct:
            broken.append(case.case_id)
        else:
            unchanged += 1

    return ExtractionRegressionDiff(
        deltas=deltas,
        fixed=sorted(fixed),
        broken=sorted(broken),
        unchanged=unchanged,
        before_config=dict(before.config),
        after_config=dict(after.config),
    )


__all__ = ["ExtractionEvalRunner", "diff_extraction_reports", "value_at"]
