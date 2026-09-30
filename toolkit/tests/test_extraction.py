"""Tests for schema-driven extraction, the repair loop, and document splitting.

The repair tests are the important ones: they assert that the *second* prompt
names the specific failure, because a repair loop that re-asks blindly is not a
repair loop.

Run standalone: python toolkit/tests/test_extraction.py
"""
from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from toolkit.adapters import ScriptedLLM  # noqa: E402
from toolkit.core import (  # noqa: E402
    BBox,
    Block,
    BlockType,
    Chunk,
    Document,
    Provenance,
    ValidationFailed,
)
from toolkit.extraction import (  # noqa: E402
    DocumentSplitterComponent,
    ExtractionComponent,
    ExtractionConfig,
    ExtractionRequest,
    ExtractionSchema,
    FieldSpec,
    FieldType,
    extract_json_object,
)

_INVOICE_TEXT = (
    "ACME Industrial Supplies\n"
    "Invoice number INV-88213\n"
    "Issued 14 March 2024\n"
    "Bill to: Northwind Trading\n"
    "Total due: $1,240.50\n"
    "Status: unpaid\n"
)


def _invoice_schema() -> ExtractionSchema:
    return ExtractionSchema(
        name="Invoice",
        fields=[
            FieldSpec("invoice_number", FieldType.STRING, "The invoice reference"),
            FieldSpec("issued", FieldType.DATE, "Date the invoice was issued"),
            FieldSpec("total", FieldType.NUMBER, "Total amount due", minimum=0),
            FieldSpec(
                "status",
                FieldType.STRING,
                "Payment status",
                enum=["paid", "unpaid", "overdue"],
            ),
            FieldSpec("notes", FieldType.STRING, "Any notes", required=False),
        ],
    )


def _document() -> Document:
    return Document(
        doc_id="doc:inv",
        page_count=1,
        source_uri="/tmp/invoice.pdf",
        blocks=[
            Block(line, BlockType.PARAGRAPH, Provenance(1, BBox(50, 40 + i * 20, 400, 55 + i * 20)))
            for i, line in enumerate(_INVOICE_TEXT.strip().splitlines())
        ],
    )


# --------------------------------------------------------------------------
# JSON extraction from messy model output
# --------------------------------------------------------------------------


def test_json_is_recovered_from_fences_prose_and_trailing_text():
    expected = {"a": 1}
    assert extract_json_object('{"a": 1}') == expected
    assert extract_json_object('```json\n{"a": 1}\n```') == expected
    assert extract_json_object('```\n{"a": 1}\n```') == expected
    assert extract_json_object('Here is the result:\n{"a": 1}\nHope that helps!') == expected
    assert extract_json_object('{"a": 1} and some commentary') == expected


def test_brace_matching_respects_strings_and_escapes():
    payload = '{"note": "a } brace and a \\" quote", "n": 2}'
    assert extract_json_object("prefix " + payload + " suffix") == {
        "note": 'a } brace and a " quote',
        "n": 2,
    }
    assert extract_json_object('{"outer": {"inner": 1}}') == {"outer": {"inner": 1}}


def test_unparseable_response_raises():
    for bad in ("", "   ", "no json here at all", "{unclosed"):
        try:
            extract_json_object(bad)
        except ValueError:
            continue
        raise AssertionError("expected ValueError for " + repr(bad))


# --------------------------------------------------------------------------
# Happy path, coercion, provenance
# --------------------------------------------------------------------------


def test_extracts_validates_and_grounds_each_field():
    llm = ScriptedLLM(
        responses=[
            json.dumps(
                {
                    "invoice_number": "INV-88213",
                    "issued": "2024-03-14",
                    "total": 1240.50,
                    "status": "unpaid",
                    "notes": None,
                }
            )
        ]
    )
    result = ExtractionComponent(llm).execute(
        ExtractionRequest(_invoice_schema(), _document())
    )
    assert result.valid, [i.render() for i in result.issues]
    assert result.attempts == 1
    assert result.data["invoice_number"] == "INV-88213"
    assert result.data["total"] == 1240.50

    fields = result.field_map()
    # The value appears verbatim in the source, so it carries that page's box.
    assert fields["invoice_number"].grounded
    assert fields["invoice_number"].provenance is not None
    assert fields["invoice_number"].provenance.page == 1
    assert fields["invoice_number"].provenance.bbox is not None
    # An optional field left null is not a hallucination and not grounded either.
    assert fields["notes"].value is None


def test_string_numbers_and_dates_are_coerced_locally():
    """Models return '$1,240.50' and '14 March 2024'. Fixing that locally is
    cheaper than spending a repair round on formatting."""
    llm = ScriptedLLM(
        responses=[
            json.dumps(
                {
                    "invoice_number": "INV-88213",
                    "issued": "14 March 2024",
                    "total": "$1,240.50",
                    "status": "unpaid",
                }
            )
        ]
    )
    result = ExtractionComponent(llm).execute(
        ExtractionRequest(_invoice_schema(), _document())
    )
    assert result.valid, [i.render() for i in result.issues]
    assert result.data["issued"] == "2024-03-14"
    assert result.data["total"] == 1240.5
    assert len(llm.calls) == 1, "coercion must not cost a repair round"


def test_ambiguous_dates_are_refused_rather_than_guessed():
    """01/02/2024 is 1 February or 2 January depending on the document's origin.
    Guessing silently corrupts data, so it becomes a repair."""
    llm = ScriptedLLM(
        responses=[
            json.dumps({"invoice_number": "X", "issued": "01/02/2024",
                        "total": 1, "status": "paid"}),
            json.dumps({"invoice_number": "X", "issued": "2024-02-01",
                        "total": 1, "status": "paid"}),
        ]
    )
    result = ExtractionComponent(llm).execute(
        ExtractionRequest(_invoice_schema(), _document())
    )
    assert result.attempts == 2
    assert result.data["issued"] == "2024-02-01"


def test_integer_and_boolean_handling():
    schema = ExtractionSchema(
        "Flags",
        [
            FieldSpec("count", FieldType.INTEGER, "How many"),
            FieldSpec("active", FieldType.BOOLEAN, "Is it active"),
        ],
    )
    llm = ScriptedLLM(responses=[json.dumps({"count": "42", "active": "yes"})])
    result = ExtractionComponent(llm).execute(
        ExtractionRequest(schema, "there are 42 items and it is active")
    )
    assert result.valid, [i.render() for i in result.issues]
    assert result.data["count"] == 42 and result.data["count"].__class__ is int
    assert result.data["active"] is True

    # A non-whole number for an integer field is an error, not a silent truncation.
    strict = ScriptedLLM(responses=[json.dumps({"count": 4.5, "active": True})] * 3)
    failed = ExtractionComponent(strict).execute(
        ExtractionRequest(schema, "text", ExtractionConfig(max_repairs=0))
    )
    assert not failed.valid
    assert "whole number" in failed.issues[0].message


# --------------------------------------------------------------------------
# The repair loop - the reason this component exists
# --------------------------------------------------------------------------


def test_repair_prompt_names_the_specific_failure():
    """A loop that re-asks blindly is not a repair loop. The second prompt must
    carry the field, the error and the value the model produced."""
    llm = ScriptedLLM(
        responses=[
            json.dumps({"invoice_number": "INV-88213", "issued": "2024-03-14",
                        "total": 1240.5, "status": "pending"}),
            json.dumps({"invoice_number": "INV-88213", "issued": "2024-03-14",
                        "total": 1240.5, "status": "unpaid"}),
        ]
    )
    result = ExtractionComponent(llm).execute(
        ExtractionRequest(_invoice_schema(), _document())
    )
    assert result.valid
    assert result.attempts == 2

    repair_prompt = llm.calls[1][-1].content
    assert "status" in repair_prompt
    assert "one of: paid, unpaid, overdue" in repair_prompt
    assert "pending" in repair_prompt, "the rejected value must be quoted back"
    assert "Payment status" in repair_prompt, "the field description helps it correct"
    assert "Leave every other field exactly as it was" in repair_prompt
    assert "invoice_number" not in repair_prompt, (
        "fields that passed must not be re-litigated"
    )


def test_repairs_are_bounded_and_issues_survive():
    llm = ScriptedLLM(
        responses=[json.dumps({"invoice_number": "X", "issued": "2024-01-01",
                               "total": -5, "status": "bad"})] * 6
    )
    result = ExtractionComponent(llm).execute(
        ExtractionRequest(_invoice_schema(), _document(),
                          ExtractionConfig(max_repairs=2))
    )
    assert result.attempts == 3, "one initial attempt plus two repairs"
    assert len(llm.calls) == 3
    assert not result.valid
    paths = {i.path for i in result.issues}
    assert "status" in paths and "total" in paths


def test_invalid_json_triggers_a_targeted_retry():
    llm = ScriptedLLM(
        responses=[
            "I could not find the data.",
            json.dumps({"invoice_number": "INV-88213", "issued": "2024-03-14",
                        "total": 1240.5, "status": "unpaid"}),
        ]
    )
    result = ExtractionComponent(llm).execute(
        ExtractionRequest(_invoice_schema(), _document())
    )
    assert result.valid
    assert "not valid JSON" in llm.calls[1][-1].content


def test_missing_required_field_is_reported_distinctly_from_null():
    llm = ScriptedLLM(
        responses=[json.dumps({"issued": "2024-03-14", "total": 1,
                               "status": "paid", "invoice_number": None})] * 3
    )
    result = ExtractionComponent(llm).execute(
        ExtractionRequest(_invoice_schema(), _document(),
                          ExtractionConfig(max_repairs=0))
    )
    issue = [i for i in result.issues if i.path == "invoice_number"][0]
    assert "null or empty" in issue.message


def test_strict_mode_raises():
    llm = ScriptedLLM(responses=[json.dumps({"invoice_number": "X"})] * 3)
    try:
        ExtractionComponent(llm).execute(
            ExtractionRequest(_invoice_schema(), _document(),
                              ExtractionConfig(max_repairs=0, strict=True))
        )
    except ValidationFailed as exc:
        assert "issued" in str(exc)
    else:
        raise AssertionError("expected ValidationFailed")


# --------------------------------------------------------------------------
# Grounding
# --------------------------------------------------------------------------


def test_ungrounded_values_are_flagged():
    llm = ScriptedLLM(
        responses=[
            json.dumps({"invoice_number": "INV-99999", "issued": "2024-03-14",
                        "total": 1240.5, "status": "unpaid"})
        ]
    )
    result = ExtractionComponent(llm).execute(
        ExtractionRequest(_invoice_schema(), _document())
    )
    assert result.valid, "grounding is advisory unless require_grounding is set"
    names = {f.name for f in result.ungrounded}
    assert "invoice_number" in names, "a fabricated reference must be flagged"
    assert "total" not in names, "1240.5 appears in the source as 1,240.50"


def test_require_grounding_turns_fabrication_into_a_repair():
    llm = ScriptedLLM(
        responses=[
            json.dumps({"invoice_number": "INV-99999", "issued": "2024-03-14",
                        "total": 1240.5, "status": "unpaid"}),
            json.dumps({"invoice_number": "INV-88213", "issued": "2024-03-14",
                        "total": 1240.5, "status": "unpaid"}),
        ]
    )
    result = ExtractionComponent(llm).execute(
        ExtractionRequest(_invoice_schema(), _document(),
                          ExtractionConfig(require_grounding=True))
    )
    assert result.valid, [i.render() for i in result.issues]
    assert result.attempts == 2
    assert result.data["invoice_number"] == "INV-88213"
    repair = llm.calls[1][-1].content
    assert "does not appear in the document" in repair
    assert "invoice_number" in repair
    # The ISO date never appears verbatim in "Issued 14 March 2024", so requiring
    # grounding for it would fire on a correct extraction.
    assert "issued" not in repair


def test_normalised_field_types_are_exempt_from_the_grounding_requirement():
    """A signal that flags correct work is worse than no signal, because people
    learn to ignore it."""
    schema = ExtractionSchema(
        "Dates",
        [
            FieldSpec("issued", FieldType.DATE, "Issue date"),
            FieldSpec("settled", FieldType.BOOLEAN, "Has it been settled"),
        ],
    )
    llm = ScriptedLLM(responses=[json.dumps({"issued": "2024-03-14", "settled": False})])
    result = ExtractionComponent(llm).execute(
        ExtractionRequest(schema, _document(), ExtractionConfig(require_grounding=True))
    )
    assert result.valid, [i.render() for i in result.issues]
    assert len(llm.calls) == 1, "no repair round should have been spent"


# --------------------------------------------------------------------------
# Nested structures and sources
# --------------------------------------------------------------------------


def test_nested_objects_and_arrays_validate_with_precise_paths():
    schema = ExtractionSchema(
        "Order",
        [
            FieldSpec(
                "line_items",
                FieldType.ARRAY,
                "Every line on the order",
                fields=[
                    FieldSpec("sku", FieldType.STRING, "Item code"),
                    FieldSpec("qty", FieldType.INTEGER, "Quantity", minimum=1),
                ],
            ),
            FieldSpec(
                "buyer",
                FieldType.OBJECT,
                "Who ordered",
                fields=[FieldSpec("name", FieldType.STRING, "Legal name")],
            ),
            FieldSpec("tags", FieldType.ARRAY, "Labels", item_type=FieldType.STRING,
                      required=False),
        ],
    )
    good = ScriptedLLM(
        responses=[
            json.dumps(
                {
                    "line_items": [{"sku": "A1", "qty": 2}, {"sku": "B2", "qty": 1}],
                    "buyer": {"name": "Northwind"},
                    "tags": ["urgent", "export"],
                }
            )
        ]
    )
    result = ExtractionComponent(good).execute(ExtractionRequest(schema, "order text"))
    assert result.valid, [i.render() for i in result.issues]
    assert len(result.data["line_items"]) == 2
    assert result.data["buyer"]["name"] == "Northwind"

    bad = ScriptedLLM(
        responses=[
            json.dumps(
                {
                    "line_items": [{"sku": "A1", "qty": 2}, {"sku": "B2", "qty": 0}],
                    "buyer": {"name": "Northwind"},
                }
            )
        ]
        * 3
    )
    failed = ExtractionComponent(bad).execute(
        ExtractionRequest(schema, "order text", ExtractionConfig(max_repairs=0))
    )
    assert not failed.valid
    assert failed.issues[0].path == "line_items[1].qty", failed.issues[0].render()


def test_accepts_document_chunks_or_plain_text():
    schema = ExtractionSchema("S", [FieldSpec("who", FieldType.STRING, "the name")])
    payload = json.dumps({"who": "Northwind"})

    for source in (
        "Bill to: Northwind Trading",
        _document(),
        [
            Chunk("c0", "Bill to: Northwind Trading", "doc:1", 0,
                  [Provenance(3, BBox(0, 0, 10, 10))])
        ],
    ):
        result = ExtractionComponent(ScriptedLLM(responses=[payload])).execute(
            ExtractionRequest(schema, source)
        )
        assert result.valid
        assert result.data["who"] == "Northwind"

    chunk_result = ExtractionComponent(ScriptedLLM(responses=[payload])).execute(
        ExtractionRequest(
            schema,
            [Chunk("c0", "Bill to: Northwind Trading", "doc:1", 0,
                   [Provenance(3, BBox(0, 0, 10, 10))])],
        )
    )
    assert chunk_result.field_map()["who"].provenance.page == 3

    try:
        ExtractionComponent(ScriptedLLM(responses=[payload])).execute(
            ExtractionRequest(schema, 12345)
        )
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError for an unsupported source")


def test_schema_validation_rejects_malformed_specs():
    for build in (
        lambda: ExtractionSchema("S", []),
        lambda: ExtractionSchema("S", [FieldSpec("a"), FieldSpec("a")]),
        lambda: FieldSpec("", FieldType.STRING),
        lambda: FieldSpec("x", FieldType.ARRAY),
        lambda: FieldSpec("x", FieldType.OBJECT),
    ):
        try:
            build()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_from_pydantic_when_available():
    try:
        from pydantic import BaseModel, Field
    except ImportError:
        return  # optional dependency; the stdlib path is tested everywhere else

    class Buyer(BaseModel):
        name: str = Field(description="Legal name")

    class Order(BaseModel):
        reference: str = Field(description="Order reference")
        total: float = Field(description="Total due", ge=0)
        buyer: Buyer
        note: str | None = Field(default=None, description="Optional note")

    schema = ExtractionSchema.from_pydantic(Order)
    specs = schema.field_map()
    assert specs["reference"].type is FieldType.STRING
    assert specs["reference"].required
    assert specs["total"].type is FieldType.NUMBER
    assert specs["total"].minimum == 0
    assert specs["buyer"].type is FieldType.OBJECT
    assert [f.name for f in specs["buyer"].fields] == ["name"]
    assert not specs["note"].required, "Optional[...] must become optional"

    try:
        ExtractionSchema.from_pydantic(object())
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError for a non-Pydantic argument")


# --------------------------------------------------------------------------
# Document splitting
# --------------------------------------------------------------------------


def _multi_invoice_document() -> Document:
    """Three invoices in one file: pages 1-2, 3-4, 5. Each restarts numbering."""
    blocks: list[Block] = []
    page = 1
    for invoice, length in enumerate([2, 2, 1], start=1):
        for offset in range(length):
            blocks.append(
                Block("Invoice", BlockType.HEADING,
                      Provenance(page, BBox(50, 40, 300, 60)), level=1)
                if offset == 0
                else Block("Continued line items for invoice " + str(invoice),
                           BlockType.PARAGRAPH, Provenance(page, BBox(50, 100, 400, 120)))
            )
            blocks.append(
                Block("Body text for invoice " + str(invoice) + " page " + str(offset + 1),
                      BlockType.PARAGRAPH, Provenance(page, BBox(50, 140, 400, 160)))
            )
            blocks.append(
                Block("Page " + str(offset + 1) + " of " + str(length),
                      BlockType.PAGE_FOOTER, Provenance(page, BBox(250, 760, 350, 775)))
            )
            page += 1
    return Document(doc_id="doc:multi", page_count=page - 1, blocks=blocks)


def test_splitter_finds_boundaries_at_page_number_restarts():
    result = DocumentSplitterComponent().execute(_multi_invoice_document())
    assert result.page_count == 5
    ranges = [(s.start_page, s.end_page) for s in result.segments]
    assert ranges == [(1, 2), (3, 4), (5, 5)], ranges
    assert result.segments[1].reason == "page numbering restarted"
    assert result.segments[0].reason == "start of file"


def test_splitter_leaves_a_single_document_alone():
    document = Document(
        doc_id="doc:one",
        page_count=3,
        blocks=[
            b
            for page in (1, 2, 3)
            for b in (
                Block("Body on page " + str(page), BlockType.PARAGRAPH,
                      Provenance(page, BBox(0, 100, 100, 120))),
                Block("Page " + str(page) + " of 3", BlockType.PAGE_FOOTER,
                      Provenance(page, BBox(0, 760, 100, 775))),
            )
        ],
    )
    result = DocumentSplitterComponent().execute(document)
    assert [(s.start_page, s.end_page) for s in result.segments] == [(1, 3)]


def test_page_references_in_body_text_are_not_boundaries():
    """'see page 1 of the appendix' in prose is a reference, not a page number."""
    document = Document(
        doc_id="doc:ref",
        page_count=2,
        blocks=[
            Block("Body text", BlockType.PARAGRAPH, Provenance(1, BBox(0, 100, 100, 120))),
            Block("Page 1 of 2", BlockType.PAGE_FOOTER, Provenance(1, BBox(0, 760, 100, 775))),
            Block("As noted on page 1 of 4, the limit applies.", BlockType.PARAGRAPH,
                  Provenance(2, BBox(0, 100, 200, 120))),
            Block("Page 2 of 2", BlockType.PAGE_FOOTER, Provenance(2, BBox(0, 760, 100, 775))),
        ],
    )
    result = DocumentSplitterComponent().execute(document)
    assert [(s.start_page, s.end_page) for s in result.segments] == [(1, 2)]


def test_splitter_labels_segments_when_given_a_model():
    llm = ScriptedLLM(responses=["invoice", "invoice", "credit note"])
    result = DocumentSplitterComponent(llm).execute(
        _multi_invoice_document(), label_segments=True
    )
    assert [s.label for s in result.segments] == ["invoice", "invoice", "credit note"]
    assert len(llm.calls) == 3


def test_splitter_handles_empty_and_single_page_documents():
    empty = DocumentSplitterComponent().execute(Document(doc_id="d", blocks=[]))
    assert empty.segments == [] and empty.page_count == 0

    single = DocumentSplitterComponent().execute(
        Document(
            doc_id="d",
            blocks=[Block("only", BlockType.PARAGRAPH, Provenance(1, BBox(0, 0, 1, 1)))],
        )
    )
    assert [(s.start_page, s.end_page) for s in single.segments] == [(1, 1)]


def _main() -> int:
    functions = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    failures = 0
    lines = []
    for name, fn in functions:
        try:
            fn()
            lines.append("PASS " + name)
        except Exception as exc:  # noqa: BLE001 - runner
            failures += 1
            lines.append("FAIL " + name + ": " + repr(exc))
    lines.append("")
    lines.append(str(len(functions) - failures) + "/" + str(len(functions)) + " passed")
    sys.stdout.write("\n".join(lines) + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
