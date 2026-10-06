"""TK-4: word-level grounding, Decimal money, and honest truncation.

Every test here names a defect measured on a synthetic two-page invoice run
through `PdfPlumberSource` and `ExtractionComponent`. Before this change:

* a quantity of `7` that is nowhere on the page was reported `grounded=True`,
  because grounding was substring matching and `7` sits inside `INV-2026-0417`
  and inside `7.25`;
* array fields were never grounded at all, so 30 line items carried no evidence;
* money came back as `float`, so sums drifted;
* validation returned only the first issue inside an array, so a repair round
  fixed one row per attempt;
* the context was silently cut at `context_char_limit`, losing the totals of a
  long invoice with no signal;
* `03/10/2026` was refused as ambiguous with no way to say "day first".

Run standalone: python -m pytest toolkit/tests/test_extraction_grounding.py
"""
from __future__ import annotations

import json
import os
import sys
from decimal import Decimal

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _path in (_ROOT, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from _invoice_pdf import invoice_pdf, pdfplumber_available  # noqa: E402

from toolkit.adapters import PdfPlumberSource, ScriptedLLM  # noqa: E402
from toolkit.core import (  # noqa: E402
    BBox,
    Block,
    BlockType,
    Document,
    Provenance,
    Word,
    WordSource,
)
from toolkit.extraction import (  # noqa: E402
    Evidence,
    ExtractionComponent,
    ExtractionConfig,
    ExtractionRequest,
    ExtractionSchema,
    FieldSpec,
    FieldType,
    MatchClass,
)

needs_pdf = pytest.mark.skipif(
    not pdfplumber_available(), reason="pdfplumber not installed"
)


def _schema() -> ExtractionSchema:
    return ExtractionSchema(
        name="Invoice",
        fields=[
            FieldSpec("invoice_number", FieldType.STRING, "The invoice reference"),
            FieldSpec("issued", FieldType.DATE, "Date on the invoice"),
            FieldSpec("total", FieldType.DECIMAL, "Total due, excluding currency"),
            FieldSpec(
                "line_items",
                FieldType.ARRAY,
                "One entry per row of the table",
                fields=[
                    FieldSpec("description", FieldType.STRING, "Item description"),
                    FieldSpec("quantity", FieldType.INTEGER, "Units ordered"),
                    FieldSpec("unit_price", FieldType.DECIMAL, "Price per unit"),
                    FieldSpec("amount", FieldType.DECIMAL, "Quantity times unit price"),
                ],
            ),
        ],
    )


def _money(value: Decimal) -> str:
    return f"{value:,.2f}"


def _payload(fixture, *, total=None, rows=None) -> str:
    """A perfect extraction of the fixture, with optional per-row sabotage."""
    items = []
    for index, line in enumerate(fixture.lines):
        row = {
            "description": line["description"],
            "quantity": line["quantity"],
            "unit_price": _money(line["unit_price"]),
            "amount": _money(line["amount"]),
        }
        row.update((rows or {}).get(index, {}))
        items.append(row)
    return json.dumps(
        {
            "invoice_number": fixture.invoice_number,
            "issued": fixture.date_text,
            "total": _money(fixture.total) if total is None else total,
            "line_items": items,
        }
    )


def _run(fixture, payload: str, **config):
    doc = PdfPlumberSource().load(fixture.path)
    component = ExtractionComponent(ScriptedLLM(responses=[payload]))
    return component.execute(
        ExtractionRequest(
            schema=_schema(),
            source=doc,
            config=ExtractionConfig(max_repairs=0, date_order="DMY", **config),
        )
    )


# --------------------------------------------------------------------------
# Token boundaries: the false positive that started this
# --------------------------------------------------------------------------


@needs_pdf
def test_a_quantity_that_is_not_on_the_page_is_not_found() -> None:
    """Measured defect: `quantity: 7` was grounded by substring match inside
    `INV-2026-0417`, and `7` also sits inside the unit price `7.25`."""
    fixture = invoice_pdf()
    result = _run(fixture, _payload(fixture, rows={3: {"quantity": 7}}))

    leaves = {leaf.path: leaf for leaf in result.leaves}
    fabricated = leaves["line_items[3].quantity"]
    assert fabricated.value == 7
    assert fabricated.match is MatchClass.NOT_FOUND
    assert fabricated.grounded is False
    assert list(fabricated.evidence) == []


@needs_pdf
def test_the_real_quantity_is_exact_with_its_own_cell_box() -> None:
    """The same cell, unsabotaged: grounded on that row's own quantity word,
    not on another row's identical digit."""
    fixture = invoice_pdf()
    result = _run(fixture, _payload(fixture))

    leaves = {leaf.path: leaf for leaf in result.leaves}
    quantity = leaves["line_items[3].quantity"]
    description = leaves["line_items[3].description"]

    assert quantity.value == fixture.lines[3]["quantity"]
    assert quantity.match is MatchClass.EXACT
    assert quantity.grounded is True

    evidence = quantity.evidence[0]
    assert isinstance(evidence, Evidence)
    assert list(evidence.words) == [str(fixture.lines[3]["quantity"])]
    assert evidence.page == 1

    # The cell's own box: on the description's line, and to the right of it.
    anchor = description.evidence[0].bbox
    assert evidence.bbox is not None and anchor is not None
    assert evidence.bbox.y0 < anchor.y1 and anchor.y0 < evidence.bbox.y1
    assert evidence.bbox.width < anchor.width
    assert evidence.bbox.x0 > anchor.x1


# --------------------------------------------------------------------------
# Decimal money
# --------------------------------------------------------------------------


@needs_pdf
@pytest.mark.parametrize("returned", ["MYR 6,511.05", "6511.05", 6511.05])
def test_a_decimal_field_is_a_decimal_grounded_normalized_on_page_two(returned) -> None:
    """Measured defect: money came back as a float, so sums drifted. The page
    writes 'MYR 6,511.05'; no spelling of the value is byte-identical to that
    token run, so the match is `normalized`, not `exact`."""
    fixture = invoice_pdf()
    result = _run(fixture, _payload(fixture, total=returned))

    total = result.field_map()["total"]
    assert total.value == Decimal("6511.05")
    assert isinstance(total.value, Decimal)
    assert total.match is MatchClass.NORMALIZED
    assert total.grounded is True
    assert total.evidence[0].page == 2
    assert total.provenance is not None and total.provenance.page == 2
    assert result.valid


@needs_pdf
def test_decimal_sums_do_not_drift() -> None:
    """The point of Decimal: the extracted amounts add up to the subtotal the
    document states, exactly, which floats do not guarantee."""
    fixture = invoice_pdf()
    result = _run(fixture, _payload(fixture))
    amounts = [row["amount"] for row in result.data["line_items"]]
    assert all(isinstance(a, Decimal) for a in amounts)
    assert sum(amounts, Decimal("0")) == fixture.subtotal


# --------------------------------------------------------------------------
# Line-item grounding
# --------------------------------------------------------------------------


@needs_pdf
def test_every_line_item_leaf_is_grounded() -> None:
    """Measured defect: array fields were never grounded; their FieldResult had
    no evidence at all."""
    fixture = invoice_pdf()
    result = _run(fixture, _payload(fixture))

    rows = [leaf for leaf in result.leaves if leaf.path.startswith("line_items[")]
    assert len(rows) == 4 * len(fixture.lines) == 120
    assert result.ungrounded_paths == []
    assert all(leaf.evidence for leaf in rows)
    # Each row's four cells sit on one line of one page.
    for index in range(len(fixture.lines)):
        prefix = "line_items[" + str(index) + "]."
        pages = {leaf.evidence[0].page for leaf in rows if leaf.path.startswith(prefix)}
        assert len(pages) == 1


@needs_pdf
def test_one_altered_amount_is_the_only_ungrounded_leaf() -> None:
    fixture = invoice_pdf()
    result = _run(fixture, _payload(fixture, rows={17: {"amount": "999.99"}}))

    assert result.ungrounded_paths == ["line_items[17].amount"]
    altered = {leaf.path: leaf for leaf in result.leaves}["line_items[17].amount"]
    assert altered.value == Decimal("999.99")
    assert altered.match is MatchClass.NOT_FOUND


# --------------------------------------------------------------------------
# Every array issue, not the first
# --------------------------------------------------------------------------


@needs_pdf
def test_two_bad_rows_produce_two_issues_in_one_attempt() -> None:
    """Measured defect: `_validate_value` returned only the first issue inside
    an array, so a repair round could only ever fix one row."""
    fixture = invoice_pdf()
    result = _run(
        fixture,
        _payload(fixture, rows={4: {"quantity": "many"}, 19: {"quantity": "a few"}}),
    )

    paths = [issue.path for issue in result.issues]
    assert "line_items[4].quantity" in paths
    assert "line_items[19].quantity" in paths
    assert result.attempts == 1


# --------------------------------------------------------------------------
# Truncation is reported
# --------------------------------------------------------------------------


def test_a_document_longer_than_the_limit_reports_truncation() -> None:
    """Measured defect: `_context` stopped at `context_char_limit` silently, so
    a long invoice lost its totals with no signal. Not a repair issue: a repair
    cannot put back text the model never saw."""
    blocks = [
        Block(
            text="Line " + str(n) + ": Hex bolt stainless, quantity 2, amount 50.00",
            type=BlockType.PARAGRAPH,
            provenance=Provenance(page=1 + n // 40, bbox=BBox(0, n, 400, n + 10)),
        )
        for n in range(600)
    ]
    document = Document(doc_id="long", blocks=blocks, page_count=15)
    schema = ExtractionSchema(
        "Invoice", [FieldSpec("invoice_number", FieldType.STRING, "Reference")]
    )
    llm = ScriptedLLM(responses=['{"invoice_number": "Line 0"}'])

    result = ExtractionComponent(llm).execute(
        ExtractionRequest(
            schema=schema,
            source=document,
            config=ExtractionConfig(max_repairs=0, context_char_limit=1000),
        )
    )

    assert result.truncated is True
    assert result.warnings, "truncation must be announced"
    message = " ".join(result.warnings)
    assert "truncat" in message.lower()
    assert any(char.isdigit() for char in message), "say how much was sent"
    assert not [i for i in result.issues if "truncat" in i.message.lower()]


def test_a_document_inside_the_limit_is_not_reported_as_truncated() -> None:
    schema = ExtractionSchema(
        "Invoice", [FieldSpec("invoice_number", FieldType.STRING, "Reference")]
    )
    result = ExtractionComponent(
        ScriptedLLM(responses=['{"invoice_number": "INV-1"}'])
    ).execute(
        ExtractionRequest(
            schema=schema,
            source="Invoice No: INV-1",
            config=ExtractionConfig(max_repairs=0),
        )
    )
    assert result.truncated is False and list(result.warnings) == []


# --------------------------------------------------------------------------
# Date order
# --------------------------------------------------------------------------


@needs_pdf
def test_a_day_first_hint_resolves_an_ambiguous_date_and_grounds_it() -> None:
    """`03/10/2026` is 3 October in a Malaysian invoice and 10 March in a US
    one. Refusing it is right without a hint; with one it must both coerce and
    ground against the page's own spelling."""
    fixture = invoice_pdf()
    result = _run(fixture, _payload(fixture))

    issued = result.field_map()["issued"]
    assert issued.value == "2026-10-03"
    assert issued.grounded is True
    assert issued.match is MatchClass.NORMALIZED
    assert list(issued.evidence[0].words) == ["03/10/2026"]


@needs_pdf
def test_a_month_first_hint_reads_the_same_digits_the_other_way() -> None:
    fixture = invoice_pdf()
    doc = PdfPlumberSource().load(fixture.path)
    result = ExtractionComponent(ScriptedLLM(responses=[_payload(fixture)])).execute(
        ExtractionRequest(
            schema=_schema(),
            source=doc,
            config=ExtractionConfig(max_repairs=0, date_order="MDY"),
        )
    )
    assert result.field_map()["issued"].value == "2026-03-10"


def test_without_a_hint_an_ambiguous_date_stays_a_validation_issue() -> None:
    schema = ExtractionSchema("Invoice", [FieldSpec("issued", FieldType.DATE, "Date")])
    result = ExtractionComponent(
        ScriptedLLM(responses=['{"issued": "03/10/2026"}'])
    ).execute(
        ExtractionRequest(
            schema=schema,
            source="Date: 03/10/2026",
            config=ExtractionConfig(max_repairs=0),
        )
    )
    assert [i.path for i in result.issues] == ["issued"]


# --------------------------------------------------------------------------
# The no-words fallback
# --------------------------------------------------------------------------


def test_a_document_with_no_words_grounds_through_block_text() -> None:
    """A plain-text source emits no words. Grounding must still work, and must
    still respect word boundaries: `7` is in neither `INV-2026-0417` nor
    `7.25`."""
    document = Document(
        doc_id="no-words",
        blocks=[
            Block(
                text="Invoice No: INV-2026-0417 Unit Price 7.25 Qty 3",
                provenance=Provenance(page=1, bbox=BBox(0, 0, 400, 20)),
            )
        ],
    )
    assert list(document.words) == []
    schema = ExtractionSchema(
        "Invoice",
        [
            FieldSpec("invoice_number", FieldType.STRING, "Reference"),
            FieldSpec("quantity", FieldType.INTEGER, "Units"),
        ],
    )
    payload = '{"invoice_number": "INV-2026-0417", "quantity": 7}'
    result = ExtractionComponent(ScriptedLLM(responses=[payload])).execute(
        ExtractionRequest(
            schema=schema, source=document, config=ExtractionConfig(max_repairs=0)
        )
    )

    fields = result.field_map()
    assert fields["invoice_number"].grounded is True
    assert fields["invoice_number"].match is MatchClass.EXACT
    assert fields["invoice_number"].evidence[0].page == 1
    assert fields["quantity"].grounded is False
    assert fields["quantity"].match is MatchClass.NOT_FOUND


def test_block_text_fallback_still_finds_a_whole_numeric_token() -> None:
    document = Document(
        doc_id="no-words",
        blocks=[Block(text="Qty 3 Unit Price 7.25", provenance=Provenance(page=1))],
    )
    schema = ExtractionSchema("Q", [FieldSpec("quantity", FieldType.INTEGER, "Units")])
    result = ExtractionComponent(ScriptedLLM(responses=['{"quantity": 3}'])).execute(
        ExtractionRequest(
            schema=schema, source=document, config=ExtractionConfig(max_repairs=0)
        )
    )
    quantity = result.field_map()["quantity"]
    assert quantity.grounded is True and quantity.evidence[0].page == 1


# --------------------------------------------------------------------------
# Match classes
# --------------------------------------------------------------------------


def test_booleans_and_containers_are_not_checked_rather_than_not_found() -> None:
    """'true' appears in prose constantly; a boolean that grounds is noise. The
    distinction matters because `not_found` is a fabrication signal and
    `not_checked` is not."""
    schema = ExtractionSchema(
        "Doc",
        [
            FieldSpec("paid", FieldType.BOOLEAN, "Settled?"),
            FieldSpec("tags", FieldType.ARRAY, "Labels", item_type=FieldType.STRING),
        ],
    )
    result = ExtractionComponent(
        ScriptedLLM(responses=['{"paid": true, "tags": ["urgent"]}'])
    ).execute(
        ExtractionRequest(
            schema=schema,
            source="Status: unpaid. Flagged urgent.",
            config=ExtractionConfig(max_repairs=0),
        )
    )
    fields = result.field_map()
    assert fields["paid"].match is MatchClass.NOT_CHECKED
    assert fields["tags"].match is MatchClass.NOT_CHECKED
    # The array's *elements* are leaves, and those are checked.
    leaves = {leaf.path: leaf for leaf in result.leaves}
    assert leaves["tags[0]"].grounded is True


def test_a_low_confidence_ocr_word_one_edit_away_is_fuzzy_and_not_grounded() -> None:
    """A likely OCR misread is worth reporting as such, but it is not evidence:
    `fuzzy_ocr` is deliberately `grounded=False`."""
    document = Document(
        doc_id="scan",
        blocks=[Block(text="Invoice No: lNV-2026-0417", provenance=Provenance(page=1))],
        words=[
            Word("Invoice", 1, BBox(0, 0, 30, 10), 0.98, WordSource.OCR),
            Word("No:", 1, BBox(31, 0, 45, 10), 0.97, WordSource.OCR),
            Word("lNV-2026-0417", 1, BBox(46, 0, 110, 10), 0.41, WordSource.OCR),
        ],
    )
    schema = ExtractionSchema(
        "Doc", [FieldSpec("invoice_number", FieldType.STRING, "Reference")]
    )
    result = ExtractionComponent(
        ScriptedLLM(responses=['{"invoice_number": "INV-2026-0417"}'])
    ).execute(
        ExtractionRequest(
            schema=schema, source=document, config=ExtractionConfig(max_repairs=0)
        )
    )
    field = result.field_map()["invoice_number"]
    assert field.match is MatchClass.FUZZY_OCR
    assert field.grounded is False
    assert list(field.evidence[0].words) == ["lNV-2026-0417"]
    assert field.evidence[0].source is WordSource.OCR
    assert field.evidence[0].min_confidence == pytest.approx(0.41)


def test_evidence_merges_the_boxes_of_a_multi_word_match() -> None:
    document = Document(
        doc_id="d",
        blocks=[
            Block(text="Bill to: Northwind Trading", provenance=Provenance(page=1))
        ],
        words=[
            Word("Bill", 1, BBox(0, 0, 20, 10)),
            Word("to:", 1, BBox(22, 0, 34, 10)),
            Word("Northwind", 1, BBox(40, 0, 90, 10)),
            Word("Trading", 1, BBox(92, 0, 130, 10)),
        ],
    )
    schema = ExtractionSchema("Doc", [FieldSpec("customer", FieldType.STRING, "Buyer")])
    result = ExtractionComponent(
        ScriptedLLM(responses=['{"customer": "Northwind Trading"}'])
    ).execute(
        ExtractionRequest(
            schema=schema, source=document, config=ExtractionConfig(max_repairs=0)
        )
    )
    field = result.field_map()["customer"]
    assert field.match is MatchClass.EXACT
    evidence = field.evidence[0]
    assert list(evidence.words) == ["Northwind", "Trading"]
    assert evidence.bbox == BBox(40, 0, 130, 10)
    assert evidence.source is WordSource.TEXT_LAYER
    assert evidence.min_confidence is None


def test_a_case_and_punctuation_difference_is_normalized_not_exact() -> None:
    document = Document(
        doc_id="d",
        blocks=[Block(text="Vendor: ACME CORP", provenance=Provenance(page=1))],
        words=[
            Word("Vendor:", 1, BBox(0, 0, 30, 10)),
            Word("ACME", 1, BBox(32, 0, 60, 10)),
            Word("CORP", 1, BBox(62, 0, 90, 10)),
        ],
    )
    schema = ExtractionSchema("Doc", [FieldSpec("vendor", FieldType.STRING, "Seller")])
    result = ExtractionComponent(
        ScriptedLLM(responses=['{"vendor": "Acme Corp"}'])
    ).execute(
        ExtractionRequest(
            schema=schema, source=document, config=ExtractionConfig(max_repairs=0)
        )
    )
    field = result.field_map()["vendor"]
    assert field.grounded is True
    assert field.match is MatchClass.NORMALIZED
    assert list(field.evidence[0].words) == ["ACME", "CORP"]


def test_a_decimal_field_rejects_nonsense_and_keeps_the_path() -> None:
    schema = ExtractionSchema("Doc", [FieldSpec("total", FieldType.DECIMAL, "Due")])
    result = ExtractionComponent(
        ScriptedLLM(responses=['{"total": "about ten"}'])
    ).execute(
        ExtractionRequest(
            schema=schema, source="Total: 10.00", config=ExtractionConfig(max_repairs=0)
        )
    )
    assert [i.path for i in result.issues] == ["total"]


def test_a_decimal_field_honours_minimum_and_maximum() -> None:
    schema = ExtractionSchema(
        "Doc", [FieldSpec("total", FieldType.DECIMAL, "Due", minimum=0)]
    )
    result = ExtractionComponent(
        ScriptedLLM(responses=['{"total": "($310.00)"}'])
    ).execute(
        ExtractionRequest(
            schema=schema,
            source="Credit: (310.00)",
            config=ExtractionConfig(max_repairs=0),
        )
    )
    assert result.data["total"] == Decimal("-310.00")
    assert [i.path for i in result.issues] == ["total"]


@needs_pdf
def test_a_repair_prompt_survives_decimal_values() -> None:
    """`json.dumps` cannot serialise a Decimal; the repair prompt builder is the
    place that would find that out at runtime, on attempt two."""
    fixture = invoice_pdf()
    doc = PdfPlumberSource().load(fixture.path)
    bad = _payload(fixture, rows={0: {"quantity": "many"}})
    llm = ScriptedLLM(responses=[bad, _payload(fixture)])
    result = ExtractionComponent(llm).execute(
        ExtractionRequest(
            schema=_schema(),
            source=doc,
            config=ExtractionConfig(max_repairs=1, date_order="DMY"),
        )
    )
    assert result.attempts == 2 and result.valid
    assert any("line_items[0].quantity" in m.content for m in llm.calls[1])
