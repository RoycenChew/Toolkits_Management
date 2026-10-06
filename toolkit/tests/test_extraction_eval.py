"""TK-7: measuring an extractor, including the error it does not report.

`evaluation` answers "did retrieval improve". Nothing answered "did extraction
improve", and the two cannot share a harness: a retrieval case is a question
with relevant passages, an extraction case is a document with expected values
per path, and the metric that matters most here has no equivalent there.

That metric is the **silent error rate**: results the pipeline accepted that
have a wrong required field. Field accuracy counts every mistake equally, which
flatters a system that fails loudly. What a reader of the output actually
suffers is the subset that passed validation, got no warning, and was wrong —
and a change can improve field accuracy while making that subset larger.

Run standalone: python -m pytest toolkit/tests/test_extraction_eval.py
"""
from __future__ import annotations

import json
import os
import sys
from decimal import Decimal

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from toolkit.adapters import ScriptedLLM  # noqa: E402
from toolkit.core import Block, Document, Provenance  # noqa: E402
from toolkit.extraction import (  # noqa: E402
    ExtractionComponent,
    ExtractionConfig,
    ExtractionSchema,
    FieldSpec,
    FieldType,
)
from toolkit.extraction_eval import (  # noqa: E402
    ExtractionCase,
    ExtractionEvalConfig,
    ExtractionEvalReport,
    ExtractionEvalRunner,
    ExtractionGolden,
    compare_line_items,
    diff_extraction_reports,
    field_accuracy,
    line_item_scores,
    mean,
    values_equal,
)

_DOC_ONE = (
    "ACME Industrial Supplies\n"
    "Invoice No: INV-88213\n"
    "Date: 14 March 2024\n"
    "Hex bolt stainless 2 10.00 20.00\n"
    "Washer flat 3 5.00 15.00\n"
    "Total Due: USD 35.00\n"
)
_DOC_TWO = (
    "Borneo Industrial Supply\n"
    "Invoice No: INV-90001\n"
    "Date: 02 April 2024\n"
    "Cable tie black 4 2.50 10.00\n"
    "Total Due: MYR 10.00\n"
)


def _schema() -> ExtractionSchema:
    return ExtractionSchema(
        "Invoice",
        [
            FieldSpec("invoice_number", FieldType.STRING, "The invoice reference"),
            FieldSpec("issued", FieldType.DATE, "Date on the invoice"),
            FieldSpec("total", FieldType.DECIMAL, "Total due"),
            FieldSpec(
                "line_items",
                FieldType.ARRAY,
                "Table rows",
                required=False,
                fields=[
                    FieldSpec("description", FieldType.STRING, "Item"),
                    FieldSpec("quantity", FieldType.INTEGER, "Units"),
                    FieldSpec("amount", FieldType.DECIMAL, "Line amount"),
                ],
            ),
        ],
    )


def _golden() -> ExtractionGolden:
    return ExtractionGolden(
        "invoices",
        [
            ExtractionCase(
                case_id="inv-1",
                source_text=_DOC_ONE,
                expected={
                    "invoice_number": "INV-88213",
                    "issued": "2024-03-14",
                    "total": "35.00",
                    "line_items": [
                        {"description": "Hex bolt stainless", "amount": "20.00"},
                        {"description": "Washer flat", "amount": "15.00"},
                    ],
                },
                required_paths=["invoice_number", "total"],
            ),
            ExtractionCase(
                case_id="inv-2",
                source_text=_DOC_TWO,
                expected={
                    "invoice_number": "INV-90001",
                    "issued": "2024-04-02",
                    "total": "10.00",
                    "line_items": [
                        {"description": "Cable tie black", "amount": "10.00"}
                    ],
                },
                required_paths=["invoice_number", "total"],
            ),
        ],
    )


def _perfect_responses() -> list[str]:
    return [
        json.dumps(
            {
                "invoice_number": "INV-88213",
                "issued": "14 March 2024",
                "total": "USD 35.00",
                "line_items": [
                    {"description": "Hex bolt stainless", "quantity": 2, "amount": "20.00"},
                    {"description": "Washer flat", "quantity": 3, "amount": "15.00"},
                ],
            }
        ),
        json.dumps(
            {
                "invoice_number": "INV-90001",
                "issued": "02 April 2024",
                "total": "MYR 10.00",
                "line_items": [
                    {"description": "Cable tie black", "quantity": 4, "amount": "10.00"}
                ],
            }
        ),
    ]


def _run(responses: list[str], **config) -> ExtractionEvalReport:
    component = ExtractionComponent(ScriptedLLM(responses=responses))
    runner = ExtractionEvalRunner(
        component,
        _schema(),
        extraction_config=ExtractionConfig(max_repairs=0),
    )
    return runner.execute(_golden(), ExtractionEvalConfig(**config))


# --------------------------------------------------------------------------
# The comparison rules
# --------------------------------------------------------------------------


def test_normalised_exact_match_is_what_equality_means_here() -> None:
    """A model that returns the right fact in a different shape is right. One
    that returns a different fact is not, however similar it looks."""
    assert values_equal("35.00", Decimal("35.00"))
    assert values_equal("35.00", 35)
    assert values_equal("35.0", Decimal("35"))
    assert values_equal("ACME Corp", "acme corp")
    assert values_equal("ACME Corp,", "ACME Corp")
    assert values_equal("INV-88213", " inv-88213 ")
    assert values_equal(None, None)

    assert not values_equal("35.00", "350.00")
    assert not values_equal("35.00", None)
    assert not values_equal("ACME Corp", "ACME Corporation")
    assert not values_equal(True, "yes but no")


def test_field_accuracy_is_the_share_of_paths_that_were_right() -> None:
    assert field_accuracy([]) == 0.0
    assert mean([]) == 0.0
    assert mean([1.0, 0.0]) == 0.5


def test_rows_are_matched_on_the_description_alone() -> None:
    """Changed deliberately, and the reason is worth keeping.

    This used to match on `(description, amount)` together, on the argument
    that either alone is ambiguous - two rows of a real invoice do share an
    amount. That argument is sound for one combined precision/recall number and
    wrong for anything finer: a row with a wrong amount came back *unmatched*,
    so the report said a row was missing when in truth one of its four cells
    was wrong. It could not see a wrong `quantity` at all, because the key
    ignored it.

    So the questions are split. `line_item_scores` answers "is the row
    present"; `compare_line_items` answers "and is each cell right", per
    column. Both are needed, and a model that returns every row full of rubbish
    scores 1.0 here and badly there.
    """
    expected = [
        {"description": "Hex bolt", "amount": "20.00"},
        {"description": "Washer flat", "amount": "15.00"},
    ]
    exact = line_item_scores(expected, list(expected))
    assert exact.matched == 2 and exact.precision == 1.0 and exact.f1 == 1.0

    # One row right, one hallucinated, one missed.
    partial = line_item_scores(
        expected,
        [
            {"description": "Hex bolt", "amount": "20.00"},
            {"description": "Grommet", "amount": "99.00"},
        ],
    )
    assert partial.matched == 1
    assert partial.precision == 0.5
    assert partial.recall == 0.5
    assert partial.f1 == 0.5

    # Same description, wrong amount: the row IS present now. That is the
    # change - its wrongness is a cell-level fact, reported by
    # compare_line_items, not a missing row.
    wrong_amount = line_item_scores(
        expected, [{"description": "Hex bolt", "amount": "21.00"}]
    )
    assert wrong_amount.matched == 1
    assert compare_line_items(
        expected[:1], [{"description": "Hex bolt", "amount": "21.00"}]
    ).any_cell_wrong is True

    # A duplicated correct row is one match and one false positive, not two
    # matches: multiset matching, or precision is unbounded above.
    duplicated = line_item_scores(
        expected,
        [
            {"description": "Hex bolt", "amount": "20.00"},
            {"description": "Hex bolt", "amount": "20.00"},
        ],
    )
    assert duplicated.matched == 1 and duplicated.precision == 0.5


def test_an_empty_table_scores_zero_rather_than_dividing_by_zero() -> None:
    assert line_item_scores([], []).f1 == 0.0
    assert line_item_scores([{"description": "a", "amount": "1"}], []).recall == 0.0
    assert line_item_scores([], [{"description": "a", "amount": "1"}]).precision == 0.0


# --------------------------------------------------------------------------
# A clean run
# --------------------------------------------------------------------------


def test_a_perfect_extractor_scores_one_on_everything() -> None:
    report = _run(_perfect_responses())

    assert report.metrics["cases"] == 2
    assert report.metrics["field_accuracy"] == 1.0
    assert report.metrics["schema_validity_rate"] == 1.0
    assert report.metrics["line_item_f1"] == 1.0
    assert report.metrics["silent_error_rate"] == 0.0
    assert report.metrics["grounding_rate"] > 0.0
    assert all(case.correct for case in report.cases)


def test_tokens_and_cost_come_from_usage() -> None:
    """Accuracy without cost is half a comparison: the cheapest way to improve
    a number is usually to spend more."""
    report = _run(_perfect_responses())
    assert report.metrics["total_input_tokens"] > 0
    assert report.metrics["total_output_tokens"] > 0
    assert report.metrics["total_cost_usd"] == 0.0  # ScriptedLLM is free
    assert report.metrics["mean_attempts"] == 1.0


def test_grounding_rate_counts_leaves_that_were_found_on_the_page() -> None:
    report = _run(_perfect_responses())
    case = report.cases[0]
    grounded = [o for o in case.outcomes if o.grounded]
    assert grounded, "a verbatim invoice number must ground"
    assert 0.0 <= report.metrics["grounding_rate"] <= 1.0


# --------------------------------------------------------------------------
# The metric this unit exists for
# --------------------------------------------------------------------------


def test_a_wrong_required_field_in_an_accepted_result_is_a_silent_error() -> None:
    """The failure mode that matters. The schema is satisfied, nothing is
    flagged, the pipeline ships it, and the invoice number is wrong."""
    responses = _perfect_responses()
    responses[0] = responses[0].replace("INV-88213", "INV-88214")
    report = _run(responses)

    assert report.metrics["schema_validity_rate"] == 1.0, "it validated fine"
    assert report.metrics["acceptance_rate"] == 1.0, "and was accepted"
    assert report.metrics["silent_error_rate"] == 0.5
    assert report.metrics["silent_error_rate_of_accepted"] == 0.5

    bad = report.case_map()["inv-1"]
    assert bad.silent_error is True
    assert bad.correct is False
    assert [o.path for o in bad.outcomes if not o.correct] == ["invoice_number"]


def test_a_rejected_wrong_result_is_a_loud_error_not_a_silent_one() -> None:
    """A result the gate caught cost a retry, not a wrong payment. Counting it
    with the silent ones is what makes the number useless."""
    responses = _perfect_responses()
    responses[0] = responses[0].replace('"total": "USD 35.00"', '"total": "ask Bob"')
    report = _run(responses)

    bad = report.case_map()["inv-1"]
    assert bad.schema_valid is False
    assert bad.accepted is False
    assert bad.silent_error is False
    assert report.metrics["silent_error_rate"] == 0.0
    assert report.metrics["schema_validity_rate"] == 0.5


def test_the_acceptance_predicate_is_the_callers() -> None:
    """What counts as shippable is a product decision, not a metric. A stricter
    gate turns silent errors into loud ones, and the harness has to be able to
    show that."""
    responses = _perfect_responses()
    responses[0] = responses[0].replace("INV-88213", "INV-88214")

    component = ExtractionComponent(ScriptedLLM(responses=responses))
    runner = ExtractionEvalRunner(
        component,
        _schema(),
        extraction_config=ExtractionConfig(max_repairs=0),
        # Refuse anything with an ungrounded leaf. The fabricated invoice
        # number is not on the page, so this gate catches it.
        accept=lambda result: result.valid and not result.ungrounded_paths,
    )
    report = runner.execute(_golden())

    assert report.metrics["silent_error_rate"] == 0.0
    assert report.metrics["acceptance_rate"] == 0.5
    assert report.case_map()["inv-1"].accepted is False


# --------------------------------------------------------------------------
# Report and diff
# --------------------------------------------------------------------------


def test_a_report_round_trips_through_json(tmp_path) -> None:
    """Decimal values are in here, and `json.dump` cannot serialise one. A
    harness whose report cannot be written is a harness with no history."""
    report = _run(_perfect_responses())
    path = str(tmp_path / "report.json")
    report.to_json(path)

    reloaded = ExtractionEvalReport.from_json(path)
    assert reloaded.dataset == report.dataset
    assert reloaded.metrics == report.metrics
    assert [c.case_id for c in reloaded.cases] == [c.case_id for c in report.cases]
    assert reloaded.config == report.config


def test_the_diff_names_the_cases_a_change_broke() -> None:
    before = _run(_perfect_responses())
    responses = _perfect_responses()
    responses[1] = responses[1].replace("INV-90001", "INV-90002")
    after = _run(responses)

    diff = diff_extraction_reports(before, after)
    assert list(diff.broken) == ["inv-2"]
    assert list(diff.fixed) == []
    assert diff.unchanged == 1
    assert diff.net == -1

    by_name = {d.name: d for d in diff.deltas}
    assert by_name["field_accuracy"].change < 0
    assert by_name["silent_error_rate"].change > 0
    rendered = diff.render()
    assert "inv-2" in rendered and "silent_error_rate" in rendered


def test_the_diff_reports_a_fix_as_well_as_a_break() -> None:
    responses = _perfect_responses()
    responses[0] = responses[0].replace("INV-88213", "INV-88214")
    before = _run(responses)
    after = _run(_perfect_responses())

    diff = diff_extraction_reports(before, after)
    assert list(diff.fixed) == ["inv-1"]
    assert list(diff.broken) == []
    assert diff.net == 1


def test_only_metrics_present_in_both_runs_are_compared() -> None:
    """Adding a metric must not fabricate a delta against a run that never
    measured it."""
    before = _run(_perfect_responses())
    trimmed = ExtractionEvalReport(
        dataset=before.dataset,
        metrics={"field_accuracy": 0.5},
        cases=before.cases,
        config=before.config,
    )
    diff = diff_extraction_reports(trimmed, before)
    assert [d.name for d in diff.deltas] == ["field_accuracy"]


# --------------------------------------------------------------------------
# The golden set
# --------------------------------------------------------------------------


def test_a_golden_set_round_trips_through_jsonl(tmp_path) -> None:
    path = str(tmp_path / "golden.jsonl")
    _golden().to_jsonl(path)
    loaded = ExtractionGolden.from_jsonl(path, name="invoices")

    assert [c.case_id for c in loaded.cases] == ["inv-1", "inv-2"]
    assert loaded.cases[0].expected["invoice_number"] == "INV-88213"
    assert loaded.cases[0].required_paths == ["invoice_number", "total"]


def test_jsonl_skips_blanks_and_comments_and_names_a_bad_line(tmp_path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(
        '# a comment\n\n{"case_id": "a", "expected": {"total": "1.00"}}\n',
        encoding="utf-8",
    )
    assert len(ExtractionGolden.from_jsonl(str(path)).cases) == 1

    broken = tmp_path / "broken.jsonl"
    broken.write_text(
        '{"case_id": "a", "expected": {"total": "1.00"}}\n{not json}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as caught:
        ExtractionGolden.from_jsonl(str(broken))
    assert "line 2" in str(caught.value)


def test_a_case_with_no_expectation_is_refused() -> None:
    """A case that expects nothing scores 1.0 on everything and measures
    nothing, which is worse than having no case at all."""
    with pytest.raises(ValueError):
        ExtractionCase(case_id="empty", source_text="x", expected={})


def test_a_case_needs_a_source_the_runner_can_reach() -> None:
    golden = ExtractionGolden(
        "no-source",
        [ExtractionCase("a", document="invoices/a.pdf", expected={"total": "1.00"})],
    )
    runner = ExtractionEvalRunner(
        ExtractionComponent(ScriptedLLM(responses=['{"total": "1.00"}'])), _schema()
    )
    with pytest.raises(ValueError) as caught:
        runner.execute(golden)
    assert "load_source" in str(caught.value)


def test_a_load_source_hook_reaches_real_documents() -> None:
    golden = ExtractionGolden(
        "hooked",
        [
            ExtractionCase(
                "a", document="doc-one", expected={"invoice_number": "INV-88213"}
            )
        ],
    )
    seen: list[str] = []

    def load_source(case: ExtractionCase) -> str:
        seen.append(case.document)
        return _DOC_ONE

    runner = ExtractionEvalRunner(
        ExtractionComponent(ScriptedLLM(responses=['{"invoice_number": "INV-88213"}'])),
        ExtractionSchema(
            "Invoice", [FieldSpec("invoice_number", FieldType.STRING, "Reference")]
        ),
        load_source=load_source,
    )
    report = runner.execute(golden)
    assert seen == ["doc-one"]
    assert report.metrics["field_accuracy"] == 1.0


def test_a_truncated_case_is_counted_and_not_hidden() -> None:
    """A document the model only partly saw is not a fair measurement of the
    model, and a report that does not say so invites the wrong conclusion.

    Needs a real multi-block `Document`: a plain string is a single segment, and
    the context builder always sends the first segment whole, so nothing is ever
    dropped from one.
    """
    blocks = [
        Block(
            text="Line " + str(n) + ": Hex bolt stainless, amount 50.00",
            provenance=Provenance(page=1 + n // 40),
        )
        for n in range(300)
    ]
    golden = ExtractionGolden("long", [ExtractionCase("a", expected={"total": "1.00"})])
    component = ExtractionComponent(ScriptedLLM(responses=['{"total": "1.00"}']))
    runner = ExtractionEvalRunner(
        component,
        ExtractionSchema("Invoice", [FieldSpec("total", FieldType.DECIMAL, "Due")]),
        extraction_config=ExtractionConfig(max_repairs=0, context_char_limit=200),
        load_source=lambda case: Document(doc_id="long", blocks=blocks, page_count=8),
    )
    report = runner.execute(golden)
    assert report.metrics["truncated_cases"] == 1
    assert report.cases[0].truncated is True


def test_a_missing_path_is_wrong_rather_than_absent() -> None:
    """A field the model omitted entirely is not a gap in the measurement, it
    is a wrong answer. Skipping it would let a model score well by answering
    less."""
    golden = ExtractionGolden(
        "partial",
        [
            ExtractionCase(
                "a",
                source_text=_DOC_ONE,
                expected={"invoice_number": "INV-88213", "total": "35.00"},
                required_paths=["total"],
            )
        ],
    )
    component = ExtractionComponent(
        ScriptedLLM(responses=['{"invoice_number": "INV-88213"}'])
    )
    runner = ExtractionEvalRunner(
        component, _schema(), extraction_config=ExtractionConfig(max_repairs=0)
    )
    report = runner.execute(golden)

    outcome = {o.path: o for o in report.cases[0].outcomes}["total"]
    assert outcome.correct is False
    assert outcome.actual is None
    assert report.metrics["field_accuracy"] == 0.5
