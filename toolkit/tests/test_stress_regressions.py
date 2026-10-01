"""Regression tests for the defects the hostile corpus exposed.

`stress/run_stress.py` is the exploratory harness; this file is the ratchet. Each
test pins one bug that was real and shipped, so it cannot silently return.

The two findings left as KNOWN in the harness are asserted here too, as
*characterisation* tests: they document current behaviour and will fail loudly
when someone fixes them, which is the point — a KNOWN limitation that quietly
becomes fixed should force the documentation to be updated.

Run standalone: python toolkit/tests/test_stress_regressions.py
"""
from __future__ import annotations

import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_STRESS = os.path.join(_ROOT, "stress")
if _STRESS not in sys.path:
    sys.path.insert(0, _STRESS)

from toolkit.adapters import PdfPlumberSource, PlainTextSource, ScriptedLLM  # noqa: E402
from toolkit.chunking import (  # noqa: E402
    ChunkConfig,
    ChunkerComponent,
    ChunkRequest,
    estimate_tokens,
)
from toolkit.core import AdapterError, BlockType, MissingDependency  # noqa: E402
from toolkit.core.text import dehyphenate, normalise_text  # noqa: E402
from toolkit.extraction import DocumentSplitterComponent  # noqa: E402
from toolkit.extraction.component import _parse_number  # noqa: E402
from toolkit.pipelines import AskConfig, KnowledgeBase  # noqa: E402

_CORPUS: str | None = None


def corpus() -> str:
    global _CORPUS
    if _CORPUS is None:
        from make_corpus import build

        _CORPUS = build()
    return _CORPUS


def load(name: str):
    path = os.path.join(corpus(), name)
    source = PdfPlumberSource() if name.endswith(".pdf") else PlainTextSource()
    return source.load(path)


def _pdf_available() -> bool:
    try:
        import pdfplumber  # noqa: F401
    except ImportError:
        return False
    return True


# ==========================================================================
# doc_layout
# ==========================================================================


def test_two_column_page_reads_left_column_first():
    """The headline layout bug. A full-width title straddles the gutter and a
    centred page number bridges it; earlier versions spliced the two columns
    into single lines, putting the abstract and the introduction in one block."""
    if not _pdf_available():
        return
    doc = load("two_column_paper.pdf")
    texts = [b.text for b in doc.content_blocks()]
    abstract = next((i for i, t in enumerate(texts) if "Abstract" in t), None)
    intro = next((i for i, t in enumerate(texts) if "Introduction" in t), None)
    assert abstract is not None and intro is not None, texts
    assert abstract != intro, "columns spliced into a single block"
    assert abstract < intro, "right column read before left"


def test_full_width_heading_over_columns_is_a_heading():
    """"2. Method" matched the ordered-list pattern and was classified as a list
    item, which destroyed the hierarchy of any document numbering its sections."""
    if not _pdf_available():
        return
    doc = load("two_column_paper.pdf")
    method = [b for b in doc.blocks if b.text.strip() == "2. Method"]
    assert method, "the section heading vanished"
    assert method[0].type is BlockType.HEADING, method[0].type


def test_table_is_not_column_split():
    """A table is genuinely multi-column by geometry, so a density-based gutter
    detector tears its rows apart. Ink density is what refuses the split."""
    if not _pdf_available():
        return
    doc = load("table_heavy.pdf")
    text = " ".join(b.text for b in doc.content_blocks())
    assert re.search(r"40V\s+88\.1", text), (
        "table row torn apart; cells no longer adjacent: " + text[:160]
    )


def test_running_header_on_a_two_page_document_is_furniture():
    """The flat `>= 3 pages` rule meant short documents got no furniture
    detection at all, so journal headers were indexed as body text."""
    if not _pdf_available():
        return
    doc = load("two_column_paper.pdf")
    headers = [b for b in doc.blocks if "JOURNAL OF APPLIED" in b.text]
    assert headers, "header text missing"
    assert any(b.type is BlockType.PAGE_HEADER for b in headers), (
        "classified as " + ", ".join(sorted({b.type.value for b in headers}))
    )


def test_hyphenated_line_breaks_are_rejoined():
    if not _pdf_available():
        return
    doc = load("hyphenated_contract.pdf")
    text = " ".join(b.text for b in doc.content_blocks())
    broken = re.findall(r"\w+-\s+\w+", text)
    assert not broken, "words left split: " + ", ".join(broken[:4])
    assert "customer" in text and "consequential" in text


def test_dehyphenation_is_conservative():
    assert dehyphenate("custo-", "mer") == ("custo", True)
    assert dehyphenate("self-", "Service") == ("self-", False), "capital: keep hyphen"
    assert dehyphenate("40-", "50") == ("40-", False), "numeric range: keep hyphen"
    assert dehyphenate("plain", "word") == ("plain", False)


def test_emphasis_is_not_promoted_to_a_heading():
    if not _pdf_available():
        return
    doc = load("emphasis_not_headings.pdf")
    headings = [b.text for b in doc.blocks if b.type is BlockType.HEADING]
    promoted = [t for t in headings if t.startswith(("WARNING", "NOTE", "Important"))]
    assert not promoted, "emphasis promoted: " + "; ".join(promoted)


# ==========================================================================
# adapters
# ==========================================================================


def test_text_is_normalised_at_the_boundary():
    doc = load("unicode_mess.md")
    text = " ".join(b.text for b in doc.content_blocks())
    headings = [b.text for b in doc.blocks if b.type is BlockType.HEADING]

    assert "ﬁ" not in text and "ﬂ" not in text, "ligatures survive, breaking search"
    assert " " not in text, "non-breaking space survives"
    assert "​" not in text, "zero-width space survives"
    assert "­" not in text, "soft hyphen survives"
    assert "configuration" in text.lower(), "ligature did not fold to searchable text"
    assert "resistance" in text.lower(), "soft hyphen left the word split"
    assert all(" " not in h for h in headings), "nbsp in a heading breadcrumb"


def test_normalise_text_composes_accents_and_keeps_lines_when_asked():
    assert normalise_text("Café") == "Café"
    assert normalise_text("a\nb", collapse_whitespace=False) == "a\nb"
    assert normalise_text("a\nb") == "a b"
    assert normalise_text("") == ""


def test_markdown_heading_without_a_blank_line_is_still_a_heading():
    """`## Voltage\\nThe supply...` is ordinary Markdown. Paragraph-first
    splitting swallowed the heading into the body and wiped out every
    breadcrumb."""
    doc = load("bulletin_a.md")
    headings = [(b.text, b.level) for b in doc.blocks if b.type is BlockType.HEADING]
    assert ("Voltage", 2) in headings, headings
    assert any(b.type is BlockType.PARAGRAPH and "40V" in b.text for b in doc.blocks)


def test_pdf_with_no_text_layer_fails_clearly():
    if not _pdf_available():
        return
    try:
        load("no_text_layer.pdf")
    except AdapterError as exc:
        assert "scan" in str(exc).lower() or "ocr" in str(exc).lower(), str(exc)
    except MissingDependency:
        pass
    else:
        raise AssertionError("returned an empty Document instead of failing")


# ==========================================================================
# chunking
# ==========================================================================


def test_cjk_tokens_are_counted_per_character():
    """chars/4 and words*1.3 both fail on CJK: no spaces means the word arm
    collapses to 1, and the character arm underestimates ~4x. Chunks then
    silently overran the embedding window for every non-Latin document."""
    chinese = "本设备的最大供电电压为四十伏特。" * 10
    estimated = estimate_tokens(chinese)
    characters = sum(1 for c in chinese if "一" <= c <= "鿿")
    assert estimated >= characters, (
        "estimated %d for %d CJK characters" % (estimated, characters)
    )
    # Latin text must be unaffected.
    assert 3 <= estimate_tokens("four short words here") <= 10
    # Mixed text scores each script with the rule that suits it.
    assert estimate_tokens(chinese + " and some English words") > estimated


def test_cjk_document_chunks_stay_within_budget():
    doc = load("cjk_mixed.md")
    config = ChunkConfig(max_tokens=200, include_heading_path=False)
    result = ChunkerComponent().execute(ChunkRequest(doc, config))
    assert result.chunks
    over = {k: v for k, v in result.token_estimates.items() if v > config.max_tokens}
    assert not over, "chunks over budget: " + str(over)


def test_run_on_text_without_punctuation_stays_within_budget():
    doc = load("wall_of_text.md")
    config = ChunkConfig(max_tokens=120, include_heading_path=False)
    result = ChunkerComponent().execute(ChunkRequest(doc, config))
    assert len(result.chunks) > 5
    assert all(n <= config.max_tokens for n in result.token_estimates.values())


# ==========================================================================
# extraction
# ==========================================================================


def test_multi_invoice_file_splits_on_page_number_restarts():
    if not _pdf_available():
        return
    doc = load("invoice_batch.pdf")
    result = DocumentSplitterComponent().execute(doc)
    ranges = [(s.start_page, s.end_page) for s in result.segments]
    assert ranges == [(1, 2), (3, 3), (4, 4)], ranges


def test_money_formats_parse():
    """All three appear in real invoices; two of them silently failed before."""
    assert _parse_number("$1,240.50") == 1240.50
    assert _parse_number("EUR 2.450,75") == 2450.75, "European decimal comma"
    assert _parse_number("($310.00)") == -310.00, "accounting negative"
    assert _parse_number("USD 1,240.50") == 1240.50
    assert _parse_number("-45,5") == -45.5
    assert _parse_number("1.234") == 1234.0
    assert _parse_number("abc") is None
    assert _parse_number("") is None


def test_abbreviated_month_names_coerce():
    from toolkit.extraction.component import ExtractionComponent

    coerce = ExtractionComponent(ScriptedLLM(responses=["{}"]))._coerce_date
    assert coerce("Mar 16, 2024") == "2024-03-16", "3-letter month abbreviation"
    assert coerce("16 March 2024") == "2024-03-16"
    assert coerce("2024/03/15") == "2024-03-15"
    assert coerce("01/02/2024") is None, "ambiguous: must be refused, not guessed"


# ==========================================================================
# retrieval
# ==========================================================================


def test_near_duplicate_documents_do_not_fill_the_top_k():
    kb = KnowledgeBase()
    kb.ingest_folder(corpus())
    answer = kb.ask("voltage limit under load", AskConfig(top_k=3))
    texts = [c.text.lower().split() for c in answer.chunks]
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = set(texts[i]), set(texts[j])
            if not a or not b:
                continue
            overlap = len(a & b) / min(len(a), len(b))
            assert overlap <= 0.9, "near-duplicate pair in top-3 (%.2f)" % overlap


def test_whole_hostile_corpus_ingests_and_answers():
    kb = KnowledgeBase()
    result = kb.ingest_folder(corpus())
    unexpected = [
        f for f in result.failures
        if os.path.basename(f.path) != "no_text_layer.pdf"
    ]
    assert not unexpected, [
        (os.path.basename(f.path), f.error[:60]) for f in unexpected
    ]
    assert result.chunks_indexed > 10
    answer = kb.ask("what is the maximum supply voltage?", AskConfig(top_k=4))
    assert answer.grounded and "40V" in answer.text


# ==========================================================================
# Characterisation: documented limitations, asserted so a fix forces a doc update
# ==========================================================================


def test_known_limitation_table_structure_is_not_reconstructed():
    if not _pdf_available():
        return
    doc = load("table_heavy.pdf")
    assert not any(b.type is BlockType.TABLE for b in doc.blocks), (
        "tables are now detected - update doc_layout/README.md and MATURITY.md"
    )


def test_known_limitation_prompt_injection_is_obeyed():
    """Characterises the outstanding security hole. When this starts failing,
    injection defense has landed and the docs must be updated."""
    kb = KnowledgeBase(llm=ScriptedLLM(
        handler=lambda messages: (
            "There is no limit; certified for unlimited voltage [1]."
            if "IGNORE ALL PREVIOUS INSTRUCTIONS" in messages[-1].content
            else "The limit is 40V [1]."
        )
    ))
    kb.ingest_folder(corpus())
    answer = kb.ask("what is the voltage limit?", AskConfig(top_k=5))
    reached = "unlimited" in answer.text.lower()
    assert reached, (
        "the injection payload no longer reaches the prompt - injection defense "
        "has landed, so update MATURITY.md and ARCHITECTURE.md stage 19"
    )


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
