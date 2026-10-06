"""The word-level evidence layer (TK-1) and the OCR source (TK-2).

Measured before this change: a 40-row line-item table reached downstream code as
one paragraph per page, each with a single box around the whole table. A value
extracted from row 23 could be pointed at only as "somewhere in this table".
Words now survive parsing, in the same reading order as the blocks, so evidence
can be a single cell.
"""
from __future__ import annotations

import importlib
import os
import shutil
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

from _invoice_pdf import invoice_pdf, pdfplumber_available  # noqa: E402

from toolkit.adapters import (  # noqa: E402
    PdfPlumberSource,
    PlainTextSource,
    ScriptedLLM,
    TesseractSource,
)
from toolkit.core import BBox, Document, MissingDependency, Word, WordSource  # noqa: E402
from toolkit.extraction import (  # noqa: E402
    ExtractionComponent,
    ExtractionConfig,
    ExtractionRequest,
    ExtractionSchema,
    FieldSpec,
    FieldType,
    MatchClass,
)

needs_pdf = pytest.mark.skipif(not pdfplumber_available(), reason="pdfplumber not installed")


def _importable(name: str) -> bool:
    try:
        importlib.import_module(name)
    except Exception:  # noqa: BLE001 - absent or broken both mean "not usable"
        return False
    return True


def _tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


needs_render = pytest.mark.skipif(
    not (_importable("pypdfium2") and _importable("PIL")),
    reason="the ocr extra (pypdfium2, pillow) is not installed",
)
needs_tesseract = pytest.mark.skipif(
    not _tesseract_available(), reason="the tesseract binary is not on PATH"
)


# --------------------------------------------------------------------------
# TK-1: the contract
# --------------------------------------------------------------------------


def test_documents_built_without_words_still_work() -> None:
    """Additive: every existing constructor call keeps working."""
    doc = Document(doc_id="d")
    assert doc.words == [] and doc.page_sizes == {} and doc.words_on(1) == []


def test_word_rejects_impossible_values() -> None:
    box = BBox(0, 0, 1, 1)
    with pytest.raises(ValueError):
        Word("x", page=0, bbox=box)
    with pytest.raises(ValueError):
        Word("x", page=1, bbox=box, confidence=95.0)  # a 0-100 score, not normalised


def test_a_source_with_no_geometry_emits_no_words(tmp_path) -> None:
    path = tmp_path / "note.md"
    path.write_text("# Invoice\nTotal 10.00\n", encoding="utf-8")
    doc = PlainTextSource().load(str(path))
    assert doc.blocks and doc.words == []


# --------------------------------------------------------------------------
# TK-1: pdfplumber keeps its words
# --------------------------------------------------------------------------


@needs_pdf
def test_pdf_words_survive_parsing_with_their_own_boxes() -> None:
    fixture = invoice_pdf()
    doc = PdfPlumberSource().load(fixture.path)

    assert len(doc.words) > 200
    assert doc.page_sizes == {1: (612.0, 792.0), 2: (612.0, 792.0)}
    assert all(w.source is WordSource.TEXT_LAYER and w.confidence is None for w in doc.words)
    for word in doc.words:
        width, height = doc.page_sizes[word.page]
        assert 0 <= word.bbox.x0 < word.bbox.x1 <= width
        assert 0 <= word.bbox.y0 < word.bbox.y1 <= height


@needs_pdf
def test_one_table_cell_has_its_own_box_not_the_tables() -> None:
    """The measured defect: the cell's box used to be the whole table's box."""
    fixture = invoice_pdf()
    doc = PdfPlumberSource().load(fixture.path)
    amount = "{:,.2f}".format(fixture.lines[22]["amount"])  # first row on page 2

    cells = [w for w in doc.words if w.text == amount]
    assert len(cells) == 1 and cells[0].page == 2

    table_block = next(b for b in doc.blocks if amount in b.text)
    cell_box, block_box = cells[0].bbox, table_block.provenance.bbox
    assert block_box is not None
    assert cell_box.height < block_box.height / 10


@needs_pdf
def test_words_follow_the_blocks_reading_order() -> None:
    """Same order as the block text, so a run of words can be traced back to
    the block that contains it."""
    doc = PdfPlumberSource().load(invoice_pdf().path)
    joined_words = " ".join(w.text for w in doc.words)
    joined_blocks = " ".join(b.text for b in doc.blocks)
    assert joined_words == joined_blocks
    pages = [w.page for w in doc.words]
    assert pages == sorted(pages)


@needs_pdf
def test_word_count_is_reported() -> None:
    doc = PdfPlumberSource().load(invoice_pdf().path)
    assert doc.metadata["word_count"] == len(doc.words)


# --------------------------------------------------------------------------
# TK-2: the OCR source
# --------------------------------------------------------------------------


def test_tesseract_source_declares_what_it_can_read() -> None:
    """A scanned PDF and the three image formats people actually send. Nothing
    else: claiming support for a format it cannot read makes `default_sources`
    route files into it and fail late."""
    source = TesseractSource()
    assert source.supports("scan.pdf") and source.supports("SCAN.PDF")
    assert source.supports("page.png")
    assert source.supports("page.jpg") and source.supports("photo.JPEG")
    assert source.supports("page.tif") and source.supports("page.tiff")
    assert not source.supports("notes.md") and not source.supports("sheet.xlsx")


def test_screening_refuses_a_file_it_cannot_read(tmp_path) -> None:
    missing = str(tmp_path / "nope.png")
    result = TesseractSource().screen(missing)
    assert not result.passed


@needs_render
def test_an_absent_binary_names_the_extra_rather_than_crashing() -> None:
    """The whole point of an optional extra is lost if the error does not say
    how to satisfy it."""
    fixture = invoice_pdf()
    source = TesseractSource(tesseract_cmd="tesseract-that-is-not-installed")
    with pytest.raises(MissingDependency) as caught:
        source.load(fixture.path)
    assert caught.value.extra == "ocr"


@needs_render
@needs_tesseract
def test_a_rendered_invoice_comes_back_as_ocr_words_with_confidences() -> None:
    """The contract TK-1 defined, now populated by pixels rather than by a text
    layer: confidence in [0, 1], source OCR, boxes in PDF points."""
    fixture = invoice_pdf()
    doc = TesseractSource().load(fixture.path)

    assert doc.page_count == 2
    assert doc.page_sizes == {1: (612.0, 792.0), 2: (612.0, 792.0)}
    assert len(doc.words) > 150
    assert {w.page for w in doc.words} == {1, 2}
    assert all(w.source is WordSource.OCR for w in doc.words)
    assert all(w.confidence is not None for w in doc.words)
    assert all(0.0 <= w.confidence <= 1.0 for w in doc.words)

    # Pixel boxes converted back to points: every word inside its own page.
    for word in doc.words:
        width, height = doc.page_sizes[word.page]
        assert 0 <= word.bbox.x0 < word.bbox.x1 <= width
        assert 0 <= word.bbox.y0 < word.bbox.y1 <= height

    confidences = [w.confidence for w in doc.words]
    assert sum(confidences) / len(confidences) > 0.8
    assert doc.metadata["source"] == "tesseract+doc_layout"
    assert doc.metadata["word_count"] == len(doc.words)
    assert doc.metadata["dpi"] == 300


@needs_render
@needs_tesseract
def test_the_key_identifiers_are_read_off_the_pixels() -> None:
    """Measured, and not quite what the handoff predicted: tesseract reads the
    invoice number as two words, `INV-2026-041` and `7`, because the hyphenated
    run is wide enough to break. The identifier is recoverable, and the test
    says so honestly rather than asserting a single token that is not there -
    which is also why grounding matches *runs* of words."""
    fixture = invoice_pdf()
    doc = TesseractSource().load(fixture.path)

    page_one = "".join(w.text for w in doc.words_on(1))
    assert fixture.invoice_number.replace("-", "") in page_one.replace("-", "")
    assert fixture.tax_id in page_one
    assert fixture.po_number in page_one

    page_two = [w.text for w in doc.words_on(2)]
    assert f"{fixture.total:,.2f}" in page_two
    assert "MYR" in page_two


@needs_render
@needs_tesseract
def test_an_image_is_one_page_of_words(tmp_path) -> None:
    """Scans arrive as PNGs at least as often as PDFs. Pixel dimensions become
    points via the configured DPI, because an image file has no page size."""
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(invoice_pdf().path)
    path = tmp_path / "scan.png"
    document.get_page(0).render(scale=300 / 72).to_pil().save(str(path))

    doc = TesseractSource().load(str(path))
    assert doc.page_count == 1
    assert doc.words and all(w.page == 1 for w in doc.words)
    width, height = doc.page_sizes[1]
    assert 610 < width < 614 and 790 < height < 794


@needs_render
@needs_tesseract
def test_blocks_and_words_stay_in_the_same_reading_order() -> None:
    """Same invariant as the text-layer source, which is what lets a value found
    among the words be traced back to the block containing it.

    Compared with whitespace removed, not joined on spaces as the pdfplumber
    test does. `doc_layout` closes a sub-word gap when it sees one, so where
    tesseract split `INV-2026-0417` into two words the block text has it back
    as one token. The order is the invariant; the spacing is the layout
    deciding something the OCR engine got wrong.
    """
    doc = TesseractSource().load(invoice_pdf().path)
    squeeze = lambda text: "".join(text.split())  # noqa: E731
    assert squeeze("".join(w.text for w in doc.words)) == squeeze(
        "".join(b.text for b in doc.blocks)
    )
    pages = [w.page for w in doc.words]
    assert pages == sorted(pages)


@needs_render
@needs_tesseract
def test_an_ocr_word_can_be_the_evidence_for_an_extracted_value() -> None:
    """The reason the words exist at all: a value extracted from a scan points
    back at the pixels it was read from, and the evidence says it came from OCR
    and how confident the reading was."""
    fixture = invoice_pdf()
    doc = TesseractSource().load(fixture.path)
    schema = ExtractionSchema(
        "Invoice", [FieldSpec("total", FieldType.DECIMAL, "Total due")]
    )
    payload = '{"total": "MYR %s"}' % f"{fixture.total:,.2f}"

    result = ExtractionComponent(ScriptedLLM(responses=[payload])).execute(
        ExtractionRequest(
            schema=schema, source=doc, config=ExtractionConfig(max_repairs=0)
        )
    )

    total = result.field_map()["total"]
    assert total.grounded is True
    assert total.match is MatchClass.NORMALIZED
    evidence = total.evidence[0]
    assert evidence.page == 2
    assert evidence.source is WordSource.OCR
    assert evidence.min_confidence is not None and evidence.min_confidence > 0.5
    assert evidence.bbox is not None


@needs_render
@needs_tesseract
def test_a_per_page_timeout_is_enforced() -> None:
    """A hostile image can make an OCR engine run for minutes. The cap is
    per page, because that is the unit the subprocess runs on."""
    from toolkit.core import ScreeningRejected

    source = TesseractSource(timeout_seconds=0.001)
    with pytest.raises(ScreeningRejected) as caught:
        source.load(invoice_pdf().path)
    assert caught.value.reason.value == "too_slow"
