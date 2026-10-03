"""Regressions for defects found by real use, not by authored tests.

Every test here corresponds to a numbered finding in
`validation/paper_triage/FINDINGS.md`, discovered by running the toolkit over
49 uncurated arXiv PDFs. The whole suite of 273 authored tests missed all of
them, so each test states which finding it guards and why the earlier tests
could not have caught it.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
STRESS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "stress"
)
if STRESS not in sys.path:
    sys.path.insert(0, STRESS)

from toolkit.adapters import (  # noqa: E402
    HashingEmbedder,
    InMemoryVectorStore,
    PdfPlumberSource,
    SqliteFtsIndex,
)
from toolkit.adapters.sources import WORD_GAP_RATIO  # noqa: E402
from toolkit.pipelines import IngestConfig, KnowledgeBase  # noqa: E402

pdfplumber = pytest.importorskip("pdfplumber")


def _tight_pdf(directory: str) -> str:
    from make_corpus import tex_tight_spacing

    return tex_tight_spacing(os.path.join(str(directory), "tex_tight_spacing.pdf"))


# --------------------------------------------------------------------------- #
# F1 - word spaces destroyed by pdfplumber's default x_tolerance
# --------------------------------------------------------------------------- #
def test_f1_tex_tight_spacing_keeps_words_separate(tmp_path) -> None:
    """Words placed 1.8pt apart must not be glued into one token.

    Real cause: `extract_words` defaults to an absolute `x_tolerance=3`, while
    LaTeX's Computer Modern sets inter-word space below 3pt at body sizes. 49 of
    49 arXiv PDFs were affected, median 22.9% of characters.
    """
    path = _tight_pdf(tmp_path)
    document = PdfPlumberSource().load(path)
    text = " ".join(block.text for block in document.blocks).lower()

    assert "supersingular abelian surfaces are essential in" in text, text[:300]
    glued = [w for w in text.split() if len(w) > 24]
    assert not glued, "words were merged: " + repr(glued[:3])


def test_f1_the_old_default_still_reproduces_the_defect(tmp_path) -> None:
    """The guard above must be able to fail.

    A regression test for a parser setting is worthless if it passes at every
    setting. This pins the defect to the tolerance: at the ratio that
    approximates pdfplumber's old absolute default, the line glues again. If
    this test ever stops failing to split, the fixture has drifted and
    `test_f1_tex_tight_spacing_keeps_words_separate` no longer proves anything.
    """
    path = _tight_pdf(tmp_path)
    document = PdfPlumberSource(x_tolerance_ratio=0.35).load(path)
    text = " ".join(block.text for block in document.blocks)

    assert any(len(w) > 24 for w in text.split()), (
        "expected the wide tolerance to glue words, got: " + text[:300]
    )


def test_f1_whole_line_text_is_unaffected_by_the_fix(tmp_path) -> None:
    """The fix must not over-split text that was already correct.

    The paragraph in the fixture is emitted as whole lines, so it carries
    Helvetica's own space advance. Lowering the tolerance must leave it alone -
    the measured plateau (word count flat below ratio 0.15, short-token share
    steady at 20.4%) is what this asserts in miniature.
    """
    path = _tight_pdf(tmp_path)
    document = PdfPlumberSource().load(path)
    text = " ".join(block.text for block in document.blocks).lower()

    assert "keeps the native" in text, text[:300]
    assert "own space advance" in text, text[:300]
    singles = [w for w in text.split() if len(w) == 1 and w.isalpha()]
    assert len(singles) <= 4, "over-split into single letters: " + repr(singles[:12])


def test_f1_default_ratio_is_font_relative_not_absolute(tmp_path) -> None:
    """A ratio, so the threshold scales with font size.

    An absolute tolerance that fixes 9pt body text would split a 20pt heading
    with wide tracking. The heading in the fixture is 16pt bold and must survive.
    """
    assert 0.0 < WORD_GAP_RATIO < 0.3
    path = _tight_pdf(tmp_path)
    document = PdfPlumberSource().load(path)
    text = " ".join(block.text for block in document.blocks).lower()
    assert "supersingular abelian surfaces" in text


# --------------------------------------------------------------------------- #
# F2 - checkpoint replay restored a return value but no side effects
# --------------------------------------------------------------------------- #
def _kb() -> KnowledgeBase:
    return KnowledgeBase(
        embedder=HashingEmbedder(),
        vector_store=InMemoryVectorStore(),
        lexical_index=SqliteFtsIndex(),
        sources=[PdfPlumberSource()],
    )


def test_f2_replay_into_a_fresh_knowledge_base_still_indexes(tmp_path) -> None:
    """A checkpoint that outlives its process must not yield an empty index.

    Real symptom: a 49-document corpus reported 49 'replayed', 0 failures and
    `count() == 0`, after which every question was refused. The earlier tests
    could not catch it because they build the KnowledgeBase and the checkpoint
    database together in one process, where the chunk map is still populated.
    """
    path = _tight_pdf(tmp_path)
    database = os.path.join(str(tmp_path), "checkpoints.db")
    if True:
        config = IngestConfig(durable_db=database)

        first = _kb()
        first_result = first.ingest([path], config)
        assert first.count() > 0
        assert [o.status for o in first_result.documents] == ["ingested"]

        # A new KnowledgeBase, as a new process would have: same checkpoint
        # database, empty stores.
        second = _kb()
        second_result = second.ingest([path], config)

        assert second.count() > 0, (
            "replay left the index empty while reporting "
            + str([o.status for o in second_result.documents])
        )
        assert second.count() == first.count()
        assert [o.status for o in second_result.documents] == ["reingested"]


def test_f2_replay_is_still_skipped_when_the_document_is_present(tmp_path) -> None:
    """The fix must not disable the checkpoint it is protecting.

    Re-ingesting in the *same* KnowledgeBase has its side effects intact, so the
    work must still be skipped - otherwise crash-resume has become a no-op.
    """
    path = _tight_pdf(tmp_path)
    if True:
        config = IngestConfig(durable_db=os.path.join(str(tmp_path), "checkpoints.db"))

        kb = _kb()
        kb.ingest([path], config)
        before = kb.count()
        again = kb.ingest([path], config)

        assert [o.status for o in again.documents] == ["replayed"]
        assert again.chunks_indexed == 0
        assert kb.count() == before


def test_f2_a_reingested_document_is_answerable(tmp_path) -> None:
    """The point of the fix: the index works, not merely that counts are non-zero."""
    path = _tight_pdf(tmp_path)
    if True:
        config = IngestConfig(durable_db=os.path.join(str(tmp_path), "checkpoints.db"))
        _kb().ingest([path], config)

        fresh = _kb()
        fresh.ingest([path], config)
        answer = fresh.ask("What are supersingular abelian surfaces essential in?")

    assert answer.citations, "a reingested corpus produced no citations"
    assert answer.chunks
