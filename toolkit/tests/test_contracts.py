"""Contract tests: every adapter for a port runs through the same assertions.

This file is the mechanism behind the rule in ROADMAP.md — a port is not trusted
until two implementations satisfy it. The suites below are written against the
*protocol*, never against a specific backend, and each is parameterised over
every available adapter. Backends whose optional package is not installed are
reported as SKIP rather than failing, so a bare checkout still proves the
stdlib path end to end.

Run standalone:  python toolkit/tests/test_contracts.py
Or via pytest:   python -m pytest toolkit/tests -q
"""
from __future__ import annotations

import os
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from toolkit import (  # noqa: E402
    LLM,
    Block,
    BlockType,
    Chunk,
    Document,
    DocumentSource,
    Embedder,
    LexicalIndex,
    Message,
    MissingDependency,
    Provenance,
    Reranker,
    Usage,
    VectorStore,
)
from toolkit.adapters import (  # noqa: E402
    Bm25sIndex,
    CrossEncoderReranker,
    DoclingSource,
    FastEmbedEmbedder,
    HashingEmbedder,
    InMemoryVectorStore,
    LanceDBStore,
    LexicalOverlapReranker,
    LiteLLMClient,
    LLMReranker,
    PdfPlumberSource,
    PlainTextSource,
    ScriptedLLM,
    SqliteFtsIndex,
    load_document,
)
from toolkit.core import BBox  # noqa: E402


class Skip(Exception):
    """Raised when an optional backend is not installed."""


def _chunks(*texts: str) -> list[Chunk]:
    return [
        Chunk(
            chunk_id="c%d" % i,
            text=text,
            doc_id="doc:1",
            index=i,
            provenances=[Provenance(page=1, bbox=BBox(0, i * 10, 100, i * 10 + 9))],
        )
        for i, text in enumerate(texts)
    ]


# --------------------------------------------------------------------------
# Port: Embedder
# --------------------------------------------------------------------------


def embedder_contract(make) -> None:
    try:
        embedder = make()
        dimension = embedder.dimension
    except MissingDependency as exc:
        raise Skip(str(exc)) from exc

    assert isinstance(embedder, Embedder), "must satisfy the Embedder protocol"
    assert dimension > 0

    vectors = embedder.embed(["the cat sat on the mat", "quarterly revenue report"])
    assert len(vectors) == 2, "one vector per input, in input order"
    assert all(len(v) == dimension for v in vectors), "dimension must be consistent"
    assert all(isinstance(x, float) for x in vectors[0])

    # Determinism: the same text must embed identically across calls, or caching
    # and reproducible runs are both impossible.
    again = embedder.embed(["the cat sat on the mat"])
    assert again[0] == vectors[0], "embedding must be deterministic"

    # Similar text must be closer than unrelated text. This is the weakest
    # assertion that still means the embedder is doing its job.
    near = embedder.embed(["the cat sat on a mat"])[0]
    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))
    assert dot(vectors[0], near) > dot(vectors[0], vectors[1])

    assert embedder.embed([]) == [], "an empty batch must not error"


# --------------------------------------------------------------------------
# Port: VectorStore
# --------------------------------------------------------------------------


def vector_store_contract(make) -> None:
    embedder = HashingEmbedder(dimension=64)
    chunks = _chunks(
        "refund policy for returned goods",
        "employee onboarding checklist",
        "refunds are processed within ten days",
    )
    vectors = embedder.embed([c.text for c in chunks])

    try:
        store = make()
        store.upsert(chunks, vectors)
    except MissingDependency as exc:
        raise Skip(str(exc)) from exc

    assert isinstance(store, VectorStore)
    assert store.count() == 3

    query = embedder.embed(["how do refunds work"])[0]
    hits = store.search(query, top_k=2)
    assert len(hits) == 2
    assert hits[0].score >= hits[1].score, "hits must be ordered best-first"
    assert hits[0].chunk_id in {"c0", "c2"}, "a refund chunk should win"
    assert hits[0].text, "hits must carry their text"

    # Upsert is keyed on chunk_id: re-ingesting must update, never duplicate.
    store.upsert(chunks[:1], vectors[:1])
    assert store.count() == 3, "upsert must be idempotent on chunk_id"

    assert len(store.search(query, top_k=10)) <= 3, "top_k must not invent rows"


# --------------------------------------------------------------------------
# Port: LexicalIndex
# --------------------------------------------------------------------------


def lexical_index_contract(make) -> None:
    chunks = _chunks(
        "invoice INV-40192 was paid in full",
        "the quarterly onboarding handbook for new staff",
        "invoice INV-88213 remains outstanding",
    )
    try:
        index = make()
        index.index(chunks)
    except MissingDependency as exc:
        raise Skip(str(exc)) from exc

    assert isinstance(index, LexicalIndex)
    assert index.count() == 3

    hits = index.search("invoice", top_k=5)
    assert hits, "a term present in the corpus must return hits"
    assert all(h.score >= hits[-1].score for h in hits), "ordered best-first"
    assert {h.chunk_id for h in hits} <= {"c0", "c1", "c2"}

    # Exact identifiers are precisely what lexical search is for.
    exact = index.search("INV-88213", top_k=3)
    assert exact and exact[0].chunk_id == "c2"

    # Punctuation must not be interpreted as query syntax.
    index.search('what about "invoice" (urgent)? -- now', top_k=3)
    assert index.search("", top_k=3) == [], "an empty query returns nothing"
    assert index.search("zzzznonexistent", top_k=3) == []


# --------------------------------------------------------------------------
# Port: LLM
# --------------------------------------------------------------------------


def llm_contract(make) -> None:
    messages = [
        Message("system", "You are terse."),
        Message("user", "Say hello."),
    ]
    try:
        model = make()
        # The dependency loads lazily at call time, not construction, so the
        # MissingDependency guard has to cover the call as well.
        completion = model.complete(messages, temperature=0.0)
    except MissingDependency as exc:
        raise Skip(str(exc)) from exc

    assert isinstance(model, LLM)
    assert isinstance(completion.text, str) and completion.text
    assert isinstance(completion.usage, Usage)
    assert completion.usage.input_tokens > 0, "usage must be reported for budgeting"
    assert completion.model


# --------------------------------------------------------------------------
# Port: Reranker
# --------------------------------------------------------------------------


class _Candidate:
    """Stands in for hybrid_ranker.FusedItem without importing it, so the
    contract tests the protocol rather than one concrete producer."""

    def __init__(self, identifier: str, text: str) -> None:
        self.id = identifier
        self.payload = {"text": text}


def reranker_contract(make) -> None:
    items = [
        _Candidate("a", "Employee onboarding checklist for new starters."),
        _Candidate("b", "Refunds are processed within ten business days."),
        _Candidate("c", "The chassis must be bonded to earth before use."),
    ]
    query = "how long does a refund take"

    try:
        reranker = make()
        scores = list(reranker.score(query, items))
    except MissingDependency as exc:
        raise Skip(str(exc)) from exc

    assert isinstance(reranker, Reranker)
    assert len(scores) == len(items), "one score per item, in input order"
    assert all(isinstance(s, float) for s in scores)

    # The relevant passage must win. This is the entire job.
    best = max(range(len(scores)), key=lambda i: scores[i])
    assert items[best].id == "b", list(zip([i.id for i in items], scores))

    # Deterministic: the same inputs must rank the same way twice, or a cascade
    # becomes unreproducible and an eval diff becomes meaningless.
    assert list(reranker.score(query, items)) == scores

    assert list(reranker.score(query, [])) == [], "an empty candidate set is not an error"
    assert len(list(reranker.score("", items))) == len(items), (
        "an empty query must still return one score per item"
    )


# --------------------------------------------------------------------------
# Port: DocumentSource
# --------------------------------------------------------------------------


def document_source_contract(make, path: str) -> None:
    if not path or not os.path.exists(path):
        raise Skip("no fixture available; set TOOLKIT_TEST_PDF to a real PDF")
    try:
        source = make()
        if not source.supports(path):
            raise Skip("source does not claim " + os.path.basename(path))
        document = source.load(path)
    except MissingDependency as exc:
        raise Skip(str(exc)) from exc

    assert isinstance(source, DocumentSource)
    assert document.doc_id and document.doc_id.startswith("doc:")
    assert document.blocks, "a non-empty file must yield blocks"
    assert document.page_count >= 1
    assert document.source_uri

    for block in document.blocks:
        assert isinstance(block, Block)
        assert block.text.strip(), "empty blocks must be filtered out by the adapter"
        assert isinstance(block.type, BlockType)
        assert block.provenance is not None, "every block must know its page"
        assert block.provenance.page >= 1

    # The doc_id is a content hash, so loading the same file twice is stable.
    assert source.load(path).doc_id == document.doc_id

    assert document.text
    assert all(not b.is_furniture for b in document.content_blocks())


# --------------------------------------------------------------------------
# Core contracts (no backend involved)
# --------------------------------------------------------------------------


def test_document_model_basics():
    doc = Document(
        doc_id=Document.id_from_text("hello"),
        blocks=[
            Block("Title", BlockType.HEADING, Provenance(1), level=1),
            Block("Body text.", BlockType.PARAGRAPH, Provenance(1)),
            Block("Page 1 of 9", BlockType.PAGE_FOOTER, Provenance(1)),
        ],
        page_count=1,
    )
    assert Document.id_from_text("hello") == Document.id_from_text("hello")
    assert Document.id_from_text("hello") != Document.id_from_text("world")
    assert len(doc.content_blocks()) == 2, "furniture excluded from content"
    assert "Page 1 of 9" not in doc.text
    assert doc.to_markdown().startswith("# Title")


def test_provenance_merge_refuses_to_span_pages():
    a = Provenance(page=1, bbox=BBox(0, 0, 10, 10))
    b = Provenance(page=1, bbox=BBox(5, 5, 20, 20))
    merged = a.merge(b)
    assert merged.bbox == BBox(0, 0, 20, 20)

    cross = a.merge(Provenance(page=2, bbox=BBox(0, 0, 5, 5)))
    assert cross.page == 1
    assert cross.bbox is None, "a box spanning a page break would be a fiction"


def test_usage_is_summable():
    total = Usage(10, 5, 0.01) + Usage(20, 7, 0.02)
    assert (total.input_tokens, total.output_tokens) == (30, 12)
    assert abs(total.cost_usd - 0.03) < 1e-9


def test_bbox_rejects_inverted():
    try:
        BBox(10, 0, 5, 10)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_missing_dependency_names_the_extra():
    error = MissingDependency("lancedb", "store")
    assert "lancedb" in str(error) and "toolkit[store]" in str(error)


def test_scripted_llm_records_what_the_pipeline_sent():
    model = ScriptedLLM(responses=["first", "second"])
    a = model.complete([Message("user", "one")])
    b = model.complete([Message("user", "two")])
    assert (a.text, b.text) == ("first", "second")
    assert len(model.calls) == 2
    assert model.calls[1][0].content == "two"
    # Falls back to echo once the script runs out, so a pipeline that makes more
    # calls than expected does not crash mid-debug.
    assert model.complete([Message("user", "three")]).text == "three"


def test_cross_encoder_adapter_normalises_its_model_output():
    """Exercises the adapter's own logic without the optional dependency: pair
    construction, float conversion, and the length check that catches a model
    returning the wrong number of scores."""

    class FakeCrossEncoder:
        def __init__(self):
            self.seen = None

        def predict(self, pairs, batch_size=32):
            self.seen = pairs
            return [0.1, 0.9]  # numpy-like values arrive as plain floats here

    model = FakeCrossEncoder()
    reranker = CrossEncoderReranker(model=model)
    items = [_Candidate("a", "irrelevant text"), _Candidate("b", "the refund text")]
    scores = reranker.score("refund", items)

    assert scores == [0.1, 0.9]
    assert model.seen == [("refund", "irrelevant text"), ("refund", "the refund text")]
    assert reranker.score("q", []) == [], "no model call for an empty candidate set"

    class WrongCount:
        def predict(self, pairs, batch_size=32):
            return [1.0]

    from toolkit.core.errors import AdapterError

    try:
        CrossEncoderReranker(model=WrongCount()).score("q", items)
    except AdapterError as exc:
        assert "2 items" in str(exc)
    else:
        raise AssertionError("expected AdapterError on a score/item count mismatch")


def test_lexical_reranker_stems_so_plurals_match():
    """Without stemming this scores zero on 'refund' vs 'refunds', which is not
    an edge case — it is most real queries."""
    reranker = LexicalOverlapReranker()
    items = [
        _Candidate("a", "Onboarding checklist for new starters."),
        _Candidate("b", "Refunds are processed within ten business days."),
    ]
    scores = reranker.score("how long does a refund take", items)
    assert scores[1] > scores[0] > -1e-9
    assert scores[1] > 0.0, "the plural form must still match the singular query"


def test_llm_reranker_survives_a_partial_reply():
    """A malformed or truncated reply should degrade the ranking, not destroy it."""
    reranker = LLMReranker(ScriptedLLM(handler=lambda _: "2: 8\nsorry, cut off"))
    items = [_Candidate("a", "one"), _Candidate("b", "two"), _Candidate("c", "three")]
    scores = reranker.score("q", items)
    assert scores == [0.0, 8.0, 0.0]

    out_of_range = LLMReranker(ScriptedLLM(handler=lambda _: "9: 10\n1: 99"))
    assert out_of_range.score("q", items) == [10.0, 0.0, 0.0], (
        "an index outside the candidate set must be ignored and a score clamped"
    )


def test_plain_text_source_structure():
    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "note.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Heading\n\nA paragraph of text.\n\n- one\n- two\n")
    doc = PlainTextSource().load(path)
    types = [b.type for b in doc.blocks]
    assert types[0] is BlockType.HEADING
    assert doc.blocks[0].level == 1
    assert types.count(BlockType.LIST_ITEM) == 2
    assert all(b.provenance.bbox is None for b in doc.blocks), (
        "a source with no geometry must say so rather than invent boxes"
    )


def test_load_document_dispatches_and_refuses_unknown():
    from toolkit.core.errors import AdapterError

    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "a.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("hello world")
    assert load_document(path).blocks

    unknown = os.path.join(directory, "a.xyz")
    with open(unknown, "w", encoding="utf-8") as handle:
        handle.write("x")
    try:
        load_document(unknown)
    except AdapterError:
        pass
    else:
        raise AssertionError("expected AdapterError for an unsupported extension")


# --------------------------------------------------------------------------
# Parameterised registry: this is where "two per port" is enforced
# --------------------------------------------------------------------------


def _text_fixture() -> str:
    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "fixture.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("Heading line\n\nFirst paragraph of the fixture document.\n")
    return path


_PDF_LINES = [
    (18, 720, "Quarterly Report"),
    (11, 690, "First paragraph of the fixture document."),
    (11, 675, "It continues onto a second line here."),
    (11, 645, "A second paragraph begins after a gap."),
]


def _pdf_fixture() -> str:
    """Write a minimal, valid, uncompressed PDF with known text positions.

    Generating the fixture rather than checking one in means the PDF path is
    exercised on any machine with no external file and no extra dependency. The
    text is drawn in PDF's native y-up space; pdfplumber converts to y-down,
    which is exactly the conversion the adapter has to get right.
    """
    content = "".join(
        "BT /F1 %d Tf 72 %d Td (%s) Tj ET\n" % (size, y, text)
        for size, y, text in _PDF_LINES
    ).encode("latin-1")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += str(number).encode() + b" 0 obj\n" + body + b"\nendobj\n"

    xref_at = len(out)
    out += b"xref\n0 " + str(len(objects) + 1).encode() + b"\n"
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += ("%010d 00000 n \n" % offset).encode()
    out += (
        b"trailer\n<< /Size "
        + str(len(objects) + 1).encode()
        + b" /Root 1 0 R >>\nstartxref\n"
        + str(xref_at).encode()
        + b"\n%%EOF\n"
    )

    path = os.path.join(tempfile.mkdtemp(), "fixture.pdf")
    with open(path, "wb") as handle:
        handle.write(bytes(out))
    return path


def _fake_litellm_response(**kwargs):
    """A provider response in the dict shape LiteLLM commonly returns.

    Exercising the adapter this way tests the part that actually breaks — the
    response normalisation — without needing an API key or a network call.
    """
    return {
        "model": kwargs.get("model", "test-model"),
        "choices": [
            {"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3},
        "response_cost": 0.0001,
    }


PORT_SUITES: list[tuple[str, str, object]] = [
    ("Embedder", "HashingEmbedder", lambda: embedder_contract(lambda: HashingEmbedder(64))),
    ("Embedder", "FastEmbedEmbedder", lambda: embedder_contract(FastEmbedEmbedder)),
    ("VectorStore", "InMemoryVectorStore", lambda: vector_store_contract(InMemoryVectorStore)),
    (
        "VectorStore",
        "LanceDBStore",
        lambda: vector_store_contract(
            lambda: LanceDBStore(uri=os.path.join(tempfile.mkdtemp(), "db"))
        ),
    ),
    ("LexicalIndex", "SqliteFtsIndex", lambda: lexical_index_contract(SqliteFtsIndex)),
    ("LexicalIndex", "Bm25sIndex", lambda: lexical_index_contract(Bm25sIndex)),
    ("LLM", "ScriptedLLM", lambda: llm_contract(lambda: ScriptedLLM(responses=["ok"]))),
    (
        "LLM",
        "LiteLLMClient",
        lambda: llm_contract(lambda: LiteLLMClient(completion_fn=_fake_litellm_response)),
    ),
    (
        "Reranker",
        "LexicalOverlapReranker",
        lambda: reranker_contract(LexicalOverlapReranker),
    ),
    (
        "Reranker",
        "LLMReranker",
        # A handler rather than a response queue: the contract calls score()
        # several times to check determinism, and a queue would be exhausted.
        lambda: reranker_contract(
            lambda: LLMReranker(ScriptedLLM(handler=lambda _: "1: 1\n2: 9\n3: 0"))
        ),
    ),
    (
        "Reranker",
        "CrossEncoderReranker",
        lambda: reranker_contract(CrossEncoderReranker),
    ),
    (
        "DocumentSource",
        "PlainTextSource",
        lambda: document_source_contract(PlainTextSource, _text_fixture()),
    ),
    (
        "DocumentSource",
        "PdfPlumberSource",
        lambda: document_source_contract(PdfPlumberSource, _pdf_path()),
    ),
    (
        "DocumentSource",
        "DoclingSource",
        lambda: document_source_contract(DoclingSource, _pdf_path()),
    ),
]


def _pdf_path() -> str:
    """A real PDF if one is provided, otherwise the generated fixture."""
    return os.environ.get("TOOLKIT_TEST_PDF") or _pdf_fixture()


def _run_port_suites() -> tuple[int, int, int, list[str]]:
    passed = skipped = failed = 0
    lines: list[str] = []
    seen_ports: dict[str, int] = {}
    for port, name, suite in PORT_SUITES:
        try:
            suite()
        except Skip as exc:
            skipped += 1
            lines.append("SKIP " + port + "/" + name + ": " + str(exc)[:70])
            continue
        except Exception as exc:  # noqa: BLE001 - this is the runner
            failed += 1
            lines.append("FAIL " + port + "/" + name + ": " + repr(exc))
            continue
        passed += 1
        seen_ports[port] = seen_ports.get(port, 0) + 1
        lines.append("PASS " + port + "/" + name)

    lines.append("")
    for port in sorted({p for p, _, _ in PORT_SUITES}):
        count = seen_ports.get(port, 0)
        status = "ok" if count >= 2 else "only " + str(count) + " verified here"
        lines.append("  port " + port + ": " + str(count) + " implementation(s) passing - " + status)
    return passed, skipped, failed, lines


def test_port_contracts_across_every_adapter():
    """Runs every port suite, so `pytest` covers them and not just the
    standalone runner. Adapters whose optional package is absent are skipped,
    which is a reported outcome rather than a silent pass.
    """
    passed, skipped, failed, lines = _run_port_suites()
    assert failed == 0, "port contract failures:\n" + "\n".join(
        line for line in lines if line.startswith("FAIL")
    )
    assert passed >= 5, "expected at least the stdlib adapters to run, got " + str(passed)


def test_every_port_has_at_least_two_implementations_registered():
    """The rule, enforced structurally rather than by memory.

    Counts registered adapters, not passing ones, because whether an optional
    backend is installed in this environment says nothing about whether the
    abstraction was designed against more than one.
    """
    counts: dict[str, int] = {}
    for port, _, _ in PORT_SUITES:
        counts[port] = counts.get(port, 0) + 1
    thin = {port: n for port, n in counts.items() if n < 2}
    assert not thin, "these ports are validated against too few adapters: " + str(thin)


def _main() -> int:
    unit = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    lines: list[str] = ["--- core contracts ---"]
    failures = 0
    for name, fn in unit:
        try:
            fn()
            lines.append("PASS " + name)
        except Exception as exc:  # noqa: BLE001 - runner
            failures += 1
            lines.append("FAIL " + name + ": " + repr(exc))

    lines.append("")
    lines.append("--- port contracts (same assertions, every adapter) ---")
    passed, skipped, failed, port_lines = _run_port_suites()
    lines.extend(port_lines)
    lines.append("")
    lines.append(
        "unit "
        + str(len(unit) - failures)
        + "/"
        + str(len(unit))
        + " passed | adapters "
        + str(passed)
        + " passed, "
        + str(skipped)
        + " skipped (not installed), "
        + str(failed)
        + " failed"
    )
    sys.stdout.write("\n".join(lines) + "\n")
    return 1 if (failures or failed) else 0


if __name__ == "__main__":
    raise SystemExit(_main())
