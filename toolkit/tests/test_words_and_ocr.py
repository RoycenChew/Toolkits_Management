"""The word-level evidence layer (TK-1) and the OCR source (TK-2).

Measured before this change: a 40-row line-item table reached downstream code as
one paragraph per page, each with a single box around the whole table. A value
extracted from row 23 could be pointed at only as "somewhere in this table".
Words now survive parsing, in the same reading order as the blocks, so evidence
can be a single cell.
"""
from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

from _invoice_pdf import invoice_pdf, pdfplumber_available  # noqa: E402

from toolkit.adapters import PdfPlumberSource, PlainTextSource  # noqa: E402
from toolkit.core import BBox, Document, Word, WordSource  # noqa: E402

needs_pdf = pytest.mark.skipif(not pdfplumber_available(), reason="pdfplumber not installed")


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
