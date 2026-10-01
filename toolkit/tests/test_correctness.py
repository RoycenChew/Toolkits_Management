"""Tests for the three correctness defects found in the maturity review.

Each test reproduces the original bug first, so it fails against the old code
rather than merely passing against the new.

Run standalone: python toolkit/tests/test_correctness.py
"""
from __future__ import annotations

import contextlib
import os
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from toolkit.adapters import (  # noqa: E402
    Bm25sIndex,
    FastEmbedEmbedder,
    HashingEmbedder,
    InMemoryVectorStore,
    LanceDBStore,
    PdfPlumberSource,
    PlainTextSource,
    SqliteFtsIndex,
    load_document,
)
from toolkit.cache import CachedEmbedder, SqliteCache  # noqa: E402
from toolkit.chunking import ChunkConfig  # noqa: E402
from toolkit.core import (  # noqa: E402
    AdapterError,
    BBox,
    Chunk,
    MissingDependency,
    Provenance,
    ScreeningFailure,
    ScreeningLimits,
    ScreeningRejected,
)
from toolkit.pipelines import AskConfig, IngestConfig, KnowledgeBase  # noqa: E402


class Skip(Exception):
    pass


def _chunks(doc_id: str, texts: list[str], start: int = 0) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=doc_id + "#" + str(i),
            text=text,
            doc_id=doc_id,
            index=i,
            provenances=[Provenance(page=1, bbox=BBox(0, i * 10, 100, i * 10 + 9))],
        )
        for i, text in enumerate(texts, start=start)
    ]


# ==========================================================================
# FIX 1 — the delete path, and the orphan bug it exposes
# ==========================================================================


def delete_contract(make_store, is_vector: bool) -> None:
    """One contract, every store adapter. Deletion semantics that differ between
    the dense and lexical index would corrupt retrieval asymmetrically."""
    embedder = HashingEmbedder(32)
    a = _chunks("doc:a", ["alpha one", "alpha two", "alpha three"])
    b = _chunks("doc:b", ["beta one", "beta two"])

    try:
        store = make_store()
        if is_vector:
            store.upsert(a, embedder.embed([c.text for c in a]))
            store.upsert(b, embedder.embed([c.text for c in b]))
        else:
            store.index(a)
            store.index(b)
    except MissingDependency as exc:
        raise Skip(str(exc)) from exc

    assert store.count() == 5

    # delete by explicit ids
    assert store.delete(["doc:a#1"]) == 1
    assert store.count() == 4
    assert store.delete(["doc:a#1"]) == 0, "deleting twice must be a no-op, not an error"
    assert store.delete([]) == 0

    # delete_by_doc with keep=None removes the whole document
    assert store.delete_by_doc("doc:a") == 2
    assert store.count() == 2
    assert store.delete_by_doc("doc:nonexistent") == 0

    # delete_by_doc with keep=[...] is the orphan sweep
    assert store.delete_by_doc("doc:b", keep=["doc:b#0"]) == 1
    assert store.count() == 1

    # the survivor is still searchable
    if is_vector:
        hits = store.search(embedder.embed(["beta one"])[0], top_k=5)
    else:
        hits = store.search("beta", top_k=5)
    assert [h.chunk_id for h in hits] == ["doc:b#0"]


def test_delete_in_memory_vector_store():
    delete_contract(InMemoryVectorStore, is_vector=True)


def test_delete_lancedb_store():
    # Skipped rather than failed when lancedb is absent; CI's backend job runs it.
    with contextlib.suppress(Skip):
        delete_contract(
            lambda: LanceDBStore(uri=os.path.join(tempfile.mkdtemp(), "db")),
            is_vector=True,
        )


def test_delete_sqlite_fts_index():
    delete_contract(SqliteFtsIndex, is_vector=False)


def test_delete_bm25s_index():
    with contextlib.suppress(Skip):
        delete_contract(Bm25sIndex, is_vector=False)


def _corpus(body: str, name: str = "doc.md") -> str:
    directory = tempfile.mkdtemp()
    with open(os.path.join(directory, name), "w", encoding="utf-8") as handle:
        handle.write(body)
    return directory


_LONG = """# Manual

## Voltage
The supply must not exceed 40V under any load condition whatsoever.

## Grounding
Bond the chassis to earth before energising the circuit at all.

## Cooling
Maintain airflow above two metres per second across the heatsink.

## Disposal
Return the unit to an approved recycling centre when decommissioned.
"""

_SHORT = """# Manual

## Voltage
The supply must not exceed 40V under any load condition whatsoever.
"""


def test_re_ingesting_a_shorter_document_leaves_no_orphans():
    """THE BUG. chunk_id is a `doc_id#index` slot, so a document that shrinks
    leaves high-numbered slots behind: stale content, still retrievable, still
    citable with a real page number."""
    directory = _corpus(_LONG)
    path = os.path.join(directory, "doc.md")

    kb = KnowledgeBase(chunk_config=ChunkConfig(max_tokens=48))
    first = kb.ingest_folder(directory)
    assert first.chunks_indexed >= 4, first.chunks_indexed
    assert "recycling" in " ".join(c.text for c in kb._chunks.values())

    # Rewrite the document with only the first section, then re-ingest.
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(_SHORT)
    second = kb.ingest_folder(directory)

    assert second.chunks_indexed < first.chunks_indexed
    assert kb.vector_store.count() == second.chunks_indexed, (
        "stale slots survived in the vector store"
    )
    assert kb.lexical_index.count() == second.chunks_indexed, (
        "stale slots survived in the lexical index"
    )
    assert kb.count() == second.chunks_indexed

    # The removed sections must no longer be retrievable at all.
    for gone in ("recycling", "heatsink", "airflow"):
        answer = kb.ask(gone, AskConfig(top_k=5))
        assert gone not in answer.text.lower(), (
            "deleted content is still retrievable: " + gone
        )


def test_forget_removes_a_document_everywhere():
    kb = KnowledgeBase()
    kb.ingest_folder(_corpus(_LONG))
    doc_id = next(iter({c.doc_id for c in kb._chunks.values()}))
    before = kb.vector_store.count()
    assert before > 0

    removed = kb.forget(doc_id)
    assert removed > 0
    assert kb.vector_store.count() == 0
    assert kb.lexical_index.count() == 0
    assert kb.count() == 0

    answer = kb.ask("voltage")
    assert not answer.grounded, "a forgotten document must not be retrievable"

    assert kb.forget(doc_id) == 0, "forgetting twice is a no-op"


def test_forget_leaves_other_documents_intact():
    directory = tempfile.mkdtemp()
    for name, body in (("a.md", _LONG), ("b.md", _SHORT)):
        with open(os.path.join(directory, name), "w", encoding="utf-8") as handle:
            handle.write(body)

    kb = KnowledgeBase()
    kb.ingest_folder(directory)
    by_doc: dict[str, int] = {}
    for chunk in kb._chunks.values():
        by_doc[chunk.doc_id] = by_doc.get(chunk.doc_id, 0) + 1
    assert len(by_doc) == 2

    victim = max(by_doc, key=lambda d: by_doc[d])
    survivor_count = sum(n for d, n in by_doc.items() if d != victim)
    kb.forget(victim)

    assert kb.count() == survivor_count
    assert kb.vector_store.count() == survivor_count
    assert all(c.doc_id != victim for c in kb._chunks.values())


def test_emptied_document_is_swept():
    """A document that now produces zero chunks must stop being retrievable."""
    directory = _corpus(_LONG)
    path = os.path.join(directory, "doc.md")
    kb = KnowledgeBase()
    kb.ingest_folder(directory)
    assert kb.count() > 0

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("   \n\n  \n")
    kb.ingest_folder(directory)
    assert kb.vector_store.count() == 0
    assert kb.lexical_index.count() == 0


def test_store_without_delete_support_is_rejected_clearly():
    class LegacyStore:
        def upsert(self, chunks, vectors): pass
        def search(self, vector, top_k=10): return []
        def count(self): return 0

    kb = KnowledgeBase(vector_store=LegacyStore(), lexical_index=None)
    try:
        kb.ingest_folder(_corpus(_SHORT), IngestConfig(skip_failed=False))
    except AdapterError as exc:
        assert "delete_by_doc" in str(exc)
        assert "LegacyStore" in str(exc)
    else:
        raise AssertionError("expected a clear AdapterError, not a silent orphan leak")


# ==========================================================================
# FIX 2 — embedding model identity
# ==========================================================================


def test_every_embedder_reports_a_model_version():
    assert HashingEmbedder(64).model_version == "hashing-v1-d64-tri1"
    # Parameters that change output must change the version.
    assert HashingEmbedder(128).model_version != HashingEmbedder(64).model_version
    assert (
        HashingEmbedder(64, use_trigrams=False).model_version
        != HashingEmbedder(64).model_version
    )
    assert "bge-small" in FastEmbedEmbedder().model_version


def test_cached_embedder_delegates_its_version_and_keys_on_it():
    cache = SqliteCache()
    inner = HashingEmbedder(32)
    wrapped = CachedEmbedder(inner, cache)
    assert wrapped.model_version == inner.model_version

    wrapped.embed(["shared text"])
    assert wrapped.calls == 1

    # A different model behind the same cache must miss, not serve the first
    # model's vectors.
    other = CachedEmbedder(HashingEmbedder(32, use_trigrams=False), cache)
    other.embed(["shared text"])
    assert other.calls == 1, "a different model version must not hit the cache"


def test_swapping_embedders_is_refused_rather_than_silently_corrupting():
    """THE BUG. Same dimension, different model: the store accepts the vectors,
    search returns results, and quality is silently gone."""
    directory = _corpus(_LONG)
    store = InMemoryVectorStore()
    lexical = SqliteFtsIndex()

    kb = KnowledgeBase(
        embedder=HashingEmbedder(64), vector_store=store, lexical_index=lexical
    )
    kb.ingest_folder(directory)
    assert kb.index_model_version == "hashing-v1-d64-tri1"

    # Same dimension, different model. Dimension checks cannot catch this.
    swapped = KnowledgeBase(
        embedder=HashingEmbedder(64, use_trigrams=False),
        vector_store=store,
        lexical_index=lexical,
    )
    swapped._index_model_version = "hashing-v1-d64-tri1"

    try:
        swapped.ingest_folder(directory, IngestConfig(skip_failed=False))
    except AdapterError as exc:
        assert "not comparable" in str(exc)
        assert "tri1" in str(exc) and "tri0" in str(exc)
    else:
        raise AssertionError("expected AdapterError on an embedder swap")

    # Querying a stale index must be refused too, not just ingesting into it.
    try:
        swapped.ask("voltage")
    except AdapterError as exc:
        assert "not comparable" in str(exc)
    else:
        raise AssertionError("expected AdapterError when querying with a swapped embedder")


def test_same_embedder_reingest_is_allowed():
    directory = _corpus(_LONG)
    kb = KnowledgeBase(embedder=HashingEmbedder(64))
    kb.ingest_folder(directory)
    kb.ingest_folder(directory)
    assert kb.ask("voltage").grounded


# ==========================================================================
# FIX 3 — parse guards on untrusted input
# ==========================================================================


def test_size_cap_refuses_before_parsing():
    directory = _corpus("x" * 5000, name="big.txt")
    path = os.path.join(directory, "big.txt")

    assert PlainTextSource().screen(path, ScreeningLimits()).passed
    result = PlainTextSource().screen(path, ScreeningLimits(max_bytes=100))
    assert not result.passed
    assert result.reason is ScreeningFailure.TOO_LARGE
    assert result.size_bytes == 5000

    try:
        load_document(path, limits=ScreeningLimits(max_bytes=100))
    except ScreeningRejected as exc:
        assert exc.reason is ScreeningFailure.TOO_LARGE
    else:
        raise AssertionError("expected ScreeningRejected")


def test_empty_and_unreadable_files_are_refused():
    directory = tempfile.mkdtemp()
    empty = os.path.join(directory, "empty.txt")
    open(empty, "w", encoding="utf-8").close()
    result = PlainTextSource().screen(empty, ScreeningLimits())
    assert not result.passed and result.reason is ScreeningFailure.EMPTY

    missing = os.path.join(directory, "nope.txt")
    result = PlainTextSource().screen(missing, ScreeningLimits())
    assert not result.passed and result.reason is ScreeningFailure.UNREADABLE


def _pdf(pages: int) -> str:
    """Minimal multi-page PDF, generated so the page cap is testable offline."""
    content = b"BT /F1 12 Tf 72 700 Td (page) Tj ET\n"
    objects: list[bytes] = [b"", b"", b""]
    kids = " ".join("%d 0 R" % (4 + i * 2) for i in range(pages))
    objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[1] = (
        b"<< /Type /Pages /Kids [" + kids.encode() + b"] /Count " + str(pages).encode() + b" >>"
    )
    objects[2] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    for i in range(pages):
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 3 0 R >> >> /Contents "
            + str(5 + i * 2).encode()
            + b" 0 R >>"
        )
        objects.append(
            b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"endstream"
        )

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += str(number).encode() + b" 0 obj\n" + body + b"\nendobj\n"
    xref_at = len(out)
    out += b"xref\n0 " + str(len(objects) + 1).encode() + b"\n0000000000 65535 f \n"
    for offset in offsets:
        out += ("%010d 00000 n \n" % offset).encode()
    out += (
        b"trailer\n<< /Size "
        + str(len(objects) + 1).encode()
        + b" /Root 1 0 R >>\nstartxref\n"
        + str(xref_at).encode()
        + b"\n%%EOF\n"
    )
    path = os.path.join(tempfile.mkdtemp(), "multi.pdf")
    with open(path, "wb") as handle:
        handle.write(bytes(out))
    return path


def test_page_cap_is_the_check_that_matters_for_pdfs():
    """A small file can declare many pages, so a size cap alone does not bound
    the work."""
    path = _pdf(6)
    source = PdfPlumberSource()
    try:
        ok = source.screen(path, ScreeningLimits())
    except MissingDependency:
        return  # pdfplumber absent; the size check is covered elsewhere
    assert ok.passed and ok.page_count == 6

    capped = source.screen(path, ScreeningLimits(max_pages=3))
    assert not capped.passed
    assert capped.reason is ScreeningFailure.TOO_MANY_PAGES
    assert capped.page_count == 6
    assert capped.size_bytes < 4000, "a tiny file still declared six pages"

    try:
        source.load(path, ScreeningLimits(max_pages=3))
    except ScreeningRejected as exc:
        assert exc.reason is ScreeningFailure.TOO_MANY_PAGES
    else:
        raise AssertionError("load must screen before parsing")


def test_time_budget_aborts_a_long_document():
    path = _pdf(8)
    try:
        PdfPlumberSource().load(path, ScreeningLimits(max_seconds=0.0001))
    except ScreeningRejected as exc:
        assert exc.reason is ScreeningFailure.TOO_SLOW
    except MissingDependency:
        return
    else:
        raise AssertionError("expected the time budget to abort")


def test_guards_are_on_by_default_in_the_pipeline():
    directory = _corpus("y" * 20000, name="big.txt")
    kb = KnowledgeBase()

    permissive = kb.ingest_folder(directory)
    assert permissive.ok, "a 20 KB file is well within the default cap"

    strict = KnowledgeBase().ingest_folder(
        directory, IngestConfig(limits=ScreeningLimits(max_bytes=1000))
    )
    assert not strict.ok
    assert "too_large" in strict.failures[0].error


def test_unlimited_is_explicit():
    directory = _corpus("z" * 100, name="t.txt")
    path = os.path.join(directory, "t.txt")
    limits = ScreeningLimits.unlimited()
    assert PlainTextSource().screen(path, limits).passed
    assert limits.max_seconds == float("inf")

    for kwargs in ({"max_bytes": 0}, {"max_pages": 0}, {"max_seconds": 0}):
        try:
            ScreeningLimits(**kwargs)
        except ValueError:
            continue
        raise AssertionError("expected ValueError for " + str(kwargs))


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
        except Skip as exc:
            lines.append("SKIP " + name + ": " + str(exc)[:60])
        except Exception as exc:  # noqa: BLE001 - runner
            failures += 1
            lines.append("FAIL " + name + ": " + repr(exc))
    lines.append("")
    lines.append(str(len(functions) - failures) + "/" + str(len(functions)) + " passed")
    sys.stdout.write("\n".join(lines) + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
