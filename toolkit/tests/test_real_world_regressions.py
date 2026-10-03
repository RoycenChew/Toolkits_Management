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


class _CountingEmbedder:
    """A real `Embedder` that records how many texts it was asked to encode.

    The only way to assert that a load reused saved vectors rather than quietly
    recomputing them - which would throw away most of what persistence buys.
    """

    def __init__(self, dimension: int = 256, version: str | None = None) -> None:
        self._inner = HashingEmbedder(dimension=dimension)
        # Mirror the wrapped embedder's version by default, so "matching
        # embedder" really matches. Pass an explicit version to simulate a
        # different model.
        self._version = version or self._inner.model_version
        self.embedded = 0

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    @property
    def model_version(self) -> str:
        return self._version

    def embed(self, texts):
        batch = list(texts)
        self.embedded += len(batch)
        return self._inner.embed(batch)


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


# --------------------------------------------------------------------------- #
# F10 - rotated marginal text interleaved into body text
# --------------------------------------------------------------------------- #
def _stamped_pdf(directory) -> str:
    from make_corpus import rotated_margin_stamp

    return rotated_margin_stamp(os.path.join(str(directory), "rotated_margin_stamp.pdf"))


def test_f10_a_rotated_margin_stamp_does_not_corrupt_body_text(tmp_path) -> None:
    """Every arXiv PDF carries a rotated identifier down its left edge.

    Reading order is recovered by sorting spans on position, which only means
    something within one orientation: a vertical stamp has no common reading
    order with the lines beside it. Including it interleaved its characters into
    words on real papers ("an tc abelian surface", "routinely extc ceeding") and
    dropped an entire line of one abstract.
    """
    path = _stamped_pdf(tmp_path)
    document = PdfPlumberSource().load(path)
    text = " ".join(block.text for block in document.blocks)

    assert "essential in isogeny-based cryptography. Despite this" in text, text[:300]
    assert "efficient algorithm" in text, "a body line went missing: " + text[:300]
    for fragment in ("tcO", "6202", "viXra", "htam"):
        assert fragment not in text, "stamp fragment leaked into body text: " + fragment


def test_f10_the_fixture_really_is_adversarial(tmp_path) -> None:
    """The guard above must be able to fail.

    Asserts the raw extractor still reports the stamp in the left margin, so the
    test is proving that the source excludes it rather than that the fixture
    never contained it. These are the exact reversed fragments observed on real
    arXiv papers.
    """
    path = _stamped_pdf(tmp_path)
    with pdfplumber.open(path) as pdf:
        page = pdf.pages[0]
        rotated = [c for c in page.chars if not c.get("upright", True)]
        margin = [
            w["text"]
            for w in page.extract_words(x_tolerance_ratio=0.15)
            if w["x0"] < 45
        ]

    assert rotated, "fixture has no rotated glyphs at all"
    assert any("tcO" in w or "viXra" in w for w in margin), margin


def test_f10_excluded_rotated_text_is_counted_not_hidden(tmp_path) -> None:
    """Dropping content silently would be worse than the defect.

    The count is how a caller notices that a landscape page - whose whole body is
    rotated - lost its text and needs a different source.
    """
    path = _stamped_pdf(tmp_path)
    document = PdfPlumberSource().load(path)

    assert document.metadata["rotated_glyphs_excluded"] > 0
    assert document.metadata["rotated_glyphs_excluded"] == 41


def test_f10_a_document_without_rotation_reports_zero(tmp_path) -> None:
    """The counter must mean something: no rotation, no exclusions."""
    path = _tight_pdf(tmp_path)
    document = PdfPlumberSource().load(path)

    assert document.metadata["rotated_glyphs_excluded"] == 0


# --------------------------------------------------------------------------- #
# F3 - the heuristic token counter under-counted tables of numbers
# --------------------------------------------------------------------------- #
def test_f3_numeric_tables_are_not_under_counted() -> None:
    """A results table costs far more tokens than its character count implies.

    BPE gives most digits and punctuation marks a token each while packing about
    four letters per token. The worst real case was a 1,534-character results
    table estimated at 383 tokens and actually tokenised at 1,005 - a silent 2x
    overflow of a 512-token budget.
    """
    from toolkit.chunking import estimate_tokens

    prose = "The experiment measured a modest improvement in retrieval quality. " * 6
    table = "5.77 6.42 3.88 1.78 5.43 7.39 9.62 10.41 2.13 5.92 2.10 1.70 4.68 " * 6

    assert abs(len(table) - len(prose)) < 0.4 * len(prose), (
        "fixture should compare texts of similar length"
    )
    assert estimate_tokens(table) > estimate_tokens(prose), (
        "a table of numbers must be estimated higher than prose of the same length"
    )


def test_f3_estimate_errs_on_the_high_side_for_symbol_heavy_text() -> None:
    """Over-estimating is the safe error for a budget; under-estimating overflows."""
    from toolkit.chunking import estimate_tokens

    symbols = "f(x) = a*x^2 + b*x + c; dy/dx = 2*a*x + b; |x| <= 1e-6, " * 8
    assert estimate_tokens(symbols) > len(symbols) / 4.0


def test_f3_plain_prose_is_not_inflated() -> None:
    """The symbol arm must not fire on ordinary text and shrink every chunk.

    Asserted as "the estimate is unchanged from the two original arms" rather
    than against an absolute bound, because prose is dominated by the
    characters-over-four arm and that arm already over-estimates it by about a
    third. That is pre-existing behaviour and not what this fix touched; the
    claim here is only that the new arm adds nothing for prose.
    """
    from toolkit.chunking import estimate_tokens

    prose = (
        "This paragraph is ordinary English prose with normal punctuation. "
        "It should be estimated by the character and word arms, not by the "
        "symbol arm, because inflating prose wastes the context window. "
    ) * 4
    stripped = prose.strip()  # estimate_tokens strips before measuring
    original_arms = max(len(stripped) / 4.0, len(stripped.split()) * 1.3)

    assert estimate_tokens(prose) == int(original_arms)


# --------------------------------------------------------------------------- #
# F5 - unresolvable ground truth was scored as a retrieval miss
# --------------------------------------------------------------------------- #
def test_f5_a_snippet_in_no_chunk_is_a_dataset_error_not_a_miss(tmp_path) -> None:
    """Blaming the ranker for an upstream fault sends you to tune the wrong thing.

    Real symptom: this harness reported hit_rate@10 of 0.42 on a corpus where
    direct measurement over the resolvable cases gave 0.95, because extraction
    had mangled the text the snippets were written against.
    """
    from toolkit.evaluation import EvalCase, EvalConfig, EvalDataset, EvalRunner

    path = _tight_pdf(tmp_path)
    kb = _kb()
    kb.ingest([path])

    dataset = EvalDataset(
        name="f5",
        cases=[
            EvalCase(
                case_id="resolvable",
                query="supersingular abelian surfaces essential",
                expected_snippets=["Supersingular abelian surfaces are essential in"],
            ),
            EvalCase(
                case_id="not-in-corpus",
                query="supersingular abelian surfaces essential",
                expected_snippets=["this phrase appears in no indexed chunk anywhere"],
            ),
        ],
    )
    report = EvalRunner(kb).execute(dataset, EvalConfig(evaluate_answers=False))

    by_id = {c.case_id: c for c in report.cases}
    assert by_id["not-in-corpus"].ground_truth_missing is True
    assert by_id["resolvable"].ground_truth_missing is False
    assert report.metrics["dataset_errors"] == 1.0
    assert report.metrics["scored_cases"] == 1.0
    # The unscoreable case must not drag the retrieval metric down to 0.5.
    assert report.metrics["hit_rate@1"] == 1.0, report.metrics


# --------------------------------------------------------------------------- #
# F6 - the relevance gate answered questions the corpus cannot answer
# --------------------------------------------------------------------------- #
CLEANING_QUERY = "What cleaning product is recommended for laminate kitchen surfaces?"


def test_f6_a_single_shared_word_no_longer_passes_the_gate(tmp_path) -> None:
    """Coverage is measured per chunk, so one incidental word is not enough.

    All eight pre-registered unanswerable queries were answered before this,
    with citations to real but irrelevant chunks.
    """
    from toolkit.pipelines import AskConfig

    kb = _kb()
    kb.ingest([_tight_pdf(tmp_path)])

    answer = kb.ask(CLEANING_QUERY, AskConfig())

    assert not answer.citations, "answered an unanswerable question: " + answer.text[:200]
    assert not answer.grounded


def test_f6_a_real_question_is_still_answered(tmp_path) -> None:
    """The gate must not buy refusal accuracy with false refusals."""
    from toolkit.pipelines import AskConfig

    kb = _kb()
    kb.ingest([_tight_pdf(tmp_path)])

    answer = kb.ask("What are supersingular abelian surfaces essential in?", AskConfig())

    assert answer.citations, "refused a question the corpus answers"


def test_f6_the_old_behaviour_is_still_reachable(tmp_path) -> None:
    """`min_term_coverage=0.0` restores the any-word gate.

    This also proves the test above measures the threshold rather than something
    incidental about the fixture.
    """
    from toolkit.pipelines import AskConfig

    kb = _kb()
    kb.ingest([_tight_pdf(tmp_path)])

    lenient = kb.ask(CLEANING_QUERY, AskConfig(min_term_coverage=0.0))

    assert lenient.citations, "the lenient gate should still answer anything"


# --------------------------------------------------------------------------- #
# F7 / F8 / F9 - the ergonomic gaps a real consumer hit
# --------------------------------------------------------------------------- #
def test_f7_indexed_chunks_are_publicly_enumerable(tmp_path) -> None:
    """Auditing the index used to require reaching into a private dict."""
    path = _tight_pdf(tmp_path)
    kb = _kb()
    kb.ingest([path])

    chunks = kb.chunks()

    assert chunks
    assert len(chunks) == kb.count()
    assert all(c.provenances for c in chunks)
    assert isinstance(chunks, tuple), "must be a snapshot, not the live mapping"

    documents = kb.documents()
    assert len(documents) == 1
    assert documents[0][1] == path


def test_f8_eval_config_carries_the_fusion_weights(tmp_path) -> None:
    """Without these the harness could not ablate the fusion it exists to measure."""
    from toolkit.evaluation import EvalCase, EvalConfig, EvalDataset, EvalRunner

    kb = _kb()
    kb.ingest([_tight_pdf(tmp_path)])
    dataset = EvalDataset(
        name="f8",
        cases=[
            EvalCase(
                case_id="c1",
                query="supersingular abelian surfaces essential",
                expected_snippets=["Supersingular abelian surfaces are essential in"],
            )
        ],
    )

    report = EvalRunner(kb).execute(
        dataset, EvalConfig(evaluate_answers=False, lexical_weight=0.0)
    )

    assert report.config["lexical_weight"] == 0.0
    assert report.config["dense_weight"] == 1.0


def test_f9_ingest_reports_progress_per_document(tmp_path) -> None:
    """A 38-minute call that prints nothing is indistinguishable from a hang."""
    from toolkit.pipelines import IngestConfig

    seen: list[tuple[int, int, str]] = []
    kb = _kb()
    kb.ingest(
        [_tight_pdf(tmp_path)],
        IngestConfig(
            on_document=lambda index, total, outcome: seen.append(
                (index, total, outcome.status)
            )
        ),
    )

    assert seen == [(1, 1, "ingested")]


def test_f9_progress_fires_for_failures_too(tmp_path) -> None:
    """A reporter that goes quiet exactly when something breaks is worthless."""
    from toolkit.pipelines import IngestConfig

    broken = os.path.join(str(tmp_path), "broken.pdf")
    with open(broken, "wb") as handle:
        handle.write(b"this is not a PDF at all")

    seen: list[str] = []
    kb = _kb()
    result = kb.ingest(
        [broken], IngestConfig(on_document=lambda i, t, o: seen.append(o.status))
    )

    assert [o.status for o in result.documents] == ["failed"]
    assert seen == ["failed"]


# --------------------------------------------------------------------------- #
# Persistence - the index must survive the process that built it
# --------------------------------------------------------------------------- #
QUESTION = "What are supersingular abelian surfaces essential in?"


def _two_pdfs(directory) -> list[str]:
    from make_corpus import rotated_margin_stamp, tex_tight_spacing

    return [
        tex_tight_spacing(os.path.join(str(directory), "tight.pdf")),
        rotated_margin_stamp(os.path.join(str(directory), "stamped.pdf")),
    ]


def test_save_load_answers_identically(tmp_path) -> None:
    """The test that matters: a reloaded index behaves like the one saved.

    Counts matching proves nothing on its own - an index can be the right size
    and the wrong content. This asserts the same chunks are retrieved in the
    same order and the same answer comes out.
    """
    paths = _two_pdfs(tmp_path)
    original = _kb()
    original.ingest(paths)
    before = original.ask(QUESTION)

    saved = os.path.join(str(tmp_path), "saved.kb")
    original.save(saved)
    restored = KnowledgeBase.load(
        saved,
        embedder=HashingEmbedder(),
        vector_store=InMemoryVectorStore(),
        lexical_index=SqliteFtsIndex(),
        sources=[PdfPlumberSource()],
    )
    after = restored.ask(QUESTION)

    assert restored.count() == original.count()
    assert [c.chunk_id for c in after.chunks] == [c.chunk_id for c in before.chunks]
    assert after.text == before.text
    assert restored.documents() == original.documents()


def test_save_preserves_provenance_exactly(tmp_path) -> None:
    """Page and bbox are the core contract; a round-trip that loses them is useless."""
    paths = _two_pdfs(tmp_path)
    original = _kb()
    original.ingest(paths)

    saved = os.path.join(str(tmp_path), "saved.kb")
    original.save(saved)
    restored = KnowledgeBase.load(
        saved,
        embedder=HashingEmbedder(),
        vector_store=InMemoryVectorStore(),
        lexical_index=SqliteFtsIndex(),
    )

    before = {c.chunk_id: c for c in original.chunks()}
    for chunk in restored.chunks():
        source = before[chunk.chunk_id]
        assert chunk.text == source.text
        assert len(chunk.provenances) == len(source.provenances)
        for restored_prov, source_prov in zip(chunk.provenances, source.provenances):
            assert restored_prov.page == source_prov.page
            assert (restored_prov.bbox is None) == (source_prov.bbox is None)
            if source_prov.bbox is not None and restored_prov.bbox is not None:
                assert restored_prov.bbox.x0 == pytest.approx(source_prov.bbox.x0)
                assert restored_prov.bbox.y1 == pytest.approx(source_prov.bbox.y1)


def test_load_does_not_reparse_the_pdfs(tmp_path) -> None:
    """The whole point: parsing is 85% of ingest and must not happen again.

    Proven by deleting the source PDFs before loading. If `load` touched them it
    would fail.
    """
    paths = _two_pdfs(tmp_path)
    original = _kb()
    original.ingest(paths)
    saved = os.path.join(str(tmp_path), "saved.kb")
    original.save(saved)

    for path in paths:
        os.remove(path)

    restored = KnowledgeBase.load(
        saved,
        embedder=HashingEmbedder(),
        vector_store=InMemoryVectorStore(),
        lexical_index=SqliteFtsIndex(),
    )

    assert restored.count() == original.count()
    assert restored.ask(QUESTION).citations


def test_saved_vectors_are_reused_not_recomputed(tmp_path) -> None:
    """A load that silently re-embeds has thrown away most of the saving."""
    paths = _two_pdfs(tmp_path)
    original = _kb()
    original.ingest(paths)
    saved = os.path.join(str(tmp_path), "saved.kb")
    manifest = original.save(saved)

    assert manifest["has_vectors"] is True
    assert manifest["dimension"] > 0
    expected_bytes = manifest["chunks"] * manifest["dimension"] * 4
    assert os.path.getsize(os.path.join(saved, "vectors.bin")) == expected_bytes

    counting = _CountingEmbedder()
    restored = KnowledgeBase.load(
        saved,
        embedder=counting,
        vector_store=InMemoryVectorStore(),
        lexical_index=SqliteFtsIndex(),
    )

    assert restored.count() == original.count()
    assert counting.embedded == 0, "re-embedded despite a matching embedder"


def test_a_different_embedder_triggers_a_reembed_rather_than_a_refusal(tmp_path) -> None:
    """An index must outlive the model that built it.

    Mixing vectors from two embedders is what is never allowed; re-embedding
    every chunk from saved text is consistent and therefore fine. Slower, but it
    still skips the parser, which is the expensive half.
    """
    paths = _two_pdfs(tmp_path)
    original = _kb()
    original.ingest(paths)
    saved = os.path.join(str(tmp_path), "saved.kb")
    original.save(saved)

    other = _CountingEmbedder(dimension=64, version="other-embedder-v9")
    restored = KnowledgeBase.load(
        saved,
        embedder=other,
        vector_store=InMemoryVectorStore(),
        lexical_index=SqliteFtsIndex(),
    )

    assert restored.count() == original.count()
    assert other.embedded == original.count(), "should have re-embedded every chunk"
    assert restored.index_model_version == "other-embedder-v9"


def test_a_corrupt_chunk_line_names_the_line(tmp_path) -> None:
    """Silent truncation of an index would be the worst possible failure here."""
    paths = _two_pdfs(tmp_path)
    original = _kb()
    original.ingest(paths)
    saved = os.path.join(str(tmp_path), "saved.kb")
    original.save(saved)

    chunks_file = os.path.join(saved, "chunks.jsonl")
    with open(chunks_file, encoding="utf-8") as handle:
        lines = handle.readlines()
    lines.insert(1, '{"chunk_id": "broken", "text": "no doc_id field"}\n')
    with open(chunks_file, "w", encoding="utf-8", newline="\n") as handle:
        handle.writelines(lines)

    with pytest.raises(Exception) as caught:
        KnowledgeBase.load(
            saved,
            embedder=HashingEmbedder(),
            vector_store=InMemoryVectorStore(),
            lexical_index=SqliteFtsIndex(),
        )
    assert "line 2" in str(caught.value), caught.value


def test_a_future_format_is_refused(tmp_path) -> None:
    """Reading a newer layout with an older reader would corrupt silently."""
    import json

    paths = _two_pdfs(tmp_path)
    original = _kb()
    original.ingest(paths)
    saved = os.path.join(str(tmp_path), "saved.kb")
    original.save(saved)

    manifest_path = os.path.join(saved, "manifest.json")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["format"] = KnowledgeBase.SAVE_FORMAT + 1
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle)

    with pytest.raises(Exception) as caught:
        KnowledgeBase.load(
            saved,
            embedder=HashingEmbedder(),
            vector_store=InMemoryVectorStore(),
            lexical_index=SqliteFtsIndex(),
        )
    assert "format" in str(caught.value)


def test_loading_a_missing_index_says_so(tmp_path) -> None:
    with pytest.raises(Exception) as caught:
        KnowledgeBase.load(
            os.path.join(str(tmp_path), "nothing-here"),
            embedder=HashingEmbedder(),
            vector_store=InMemoryVectorStore(),
            lexical_index=SqliteFtsIndex(),
        )
    assert "manifest.json" in str(caught.value)


def test_forget_then_save_does_not_write_the_removed_vectors(tmp_path) -> None:
    """`_chunks` is authoritative, so a removed chunk must not survive in the file."""
    paths = _two_pdfs(tmp_path)
    original = _kb()
    original.ingest(paths)
    doc_id = original.documents()[0][0]
    original.forget(doc_id)

    saved = os.path.join(str(tmp_path), "saved.kb")
    manifest = original.save(saved)

    assert manifest["chunks"] == original.count()
    expected_bytes = manifest["chunks"] * manifest["dimension"] * 4
    assert os.path.getsize(os.path.join(saved, "vectors.bin")) == expected_bytes
    restored = KnowledgeBase.load(
        saved,
        embedder=HashingEmbedder(),
        vector_store=InMemoryVectorStore(),
        lexical_index=SqliteFtsIndex(),
    )
    assert all(c.doc_id != doc_id for c in restored.chunks())
