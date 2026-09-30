"""End-to-end tests for the ingest and ask pipelines.

These are the tests that prove the components compose: a real folder of files goes
in, and a cited answer comes out, with every citation resolvable back to a page.

Run standalone: python toolkit/tests/test_pipelines.py
"""
from __future__ import annotations

import os
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from toolkit.adapters import HashingEmbedder, ScriptedLLM  # noqa: E402
from toolkit.cache import CachedEmbedder, SqliteCache  # noqa: E402
from toolkit.chunking import ChunkConfig  # noqa: E402
from toolkit.core import AdapterError  # noqa: E402
from toolkit.pipelines import AskConfig, IngestConfig, KnowledgeBase  # noqa: E402

_MANUAL = """# Safety Manual

## Voltage

The supply must not exceed 40V. Exceeding this limit voids the warranty
immediately and may damage the controller board.

## Grounding

Bond the chassis to earth before energising the circuit. Use a conductor rated
for at least 16 amps.
"""

_FINANCE = """# Refund Policy

## Eligibility

Refunds are available for goods returned within 30 days of delivery.

## Processing

Approved refunds are processed within ten business days to the original payment
method. Reference number INV-88213 is required.
"""


def _corpus() -> str:
    directory = tempfile.mkdtemp()
    with open(os.path.join(directory, "manual.md"), "w", encoding="utf-8") as handle:
        handle.write(_MANUAL)
    with open(os.path.join(directory, "finance.md"), "w", encoding="utf-8") as handle:
        handle.write(_FINANCE)
    # A file no source claims, to prove discovery filters rather than fails.
    with open(os.path.join(directory, "notes.bin"), "wb") as handle:
        handle.write(b"\x00\x01binary")
    return directory


# --------------------------------------------------------------------------
# Ingest
# --------------------------------------------------------------------------


def test_ingest_folder_discovers_and_indexes():
    kb = KnowledgeBase()
    result = kb.ingest_folder(_corpus())
    assert result.ok, result.failures
    assert len(result.documents) == 2, "the unsupported .bin must be filtered out"
    assert result.chunks_indexed > 0
    assert kb.count() == result.chunks_indexed
    assert kb.vector_store.count() == result.chunks_indexed
    assert kb.lexical_index.count() == result.chunks_indexed
    for outcome in result.documents:
        assert outcome.status == "ingested"
        assert outcome.doc_id.startswith("doc:")
        assert outcome.chunks > 0


def test_ingest_is_idempotent_on_re_run():
    """Re-ingesting the same corpus must not duplicate rows. chunk_id is a stable
    slot, so upsert overwrites."""
    folder = _corpus()
    kb = KnowledgeBase()
    first = kb.ingest_folder(folder)
    second = kb.ingest_folder(folder)
    assert second.chunks_indexed == first.chunks_indexed
    assert kb.vector_store.count() == first.chunks_indexed, "no duplicated rows"
    assert kb.lexical_index.count() == first.chunks_indexed


def test_cached_embedder_makes_a_second_ingest_free():
    folder = _corpus()
    cache = SqliteCache()
    embedder = CachedEmbedder(HashingEmbedder(128), cache)
    kb = KnowledgeBase(embedder=embedder)

    first = kb.ingest_folder(folder)
    assert first.embedding_calls > 0
    second = kb.ingest_folder(folder)
    assert second.embedding_calls == 0, "unchanged text must not be re-embedded"


def test_one_bad_file_does_not_stop_the_corpus():
    folder = _corpus()
    # An empty PDF: the source claims it, then fails to find text.
    with open(os.path.join(folder, "broken.pdf"), "wb") as handle:
        handle.write(b"%PDF-1.4\nnot really a pdf\n%%EOF\n")

    kb = KnowledgeBase()
    result = kb.ingest_folder(folder)
    assert not result.ok, "the broken file must be reported"
    assert len(result.failures) == 1
    assert result.failures[0].path.endswith("broken.pdf")
    assert result.failures[0].error
    assert result.chunks_indexed > 0, "the other documents still ingested"

    strict = KnowledgeBase()
    try:
        strict.ingest_folder(folder, IngestConfig(skip_failed=False))
    except Exception:
        pass
    else:
        raise AssertionError("skip_failed=False should surface the failure")


def test_durable_ingest_replays_completed_documents():
    folder = _corpus()
    db = os.path.join(tempfile.mkdtemp(), "ingest.db")
    config = IngestConfig(durable_db=db)

    first = KnowledgeBase()
    first_result = first.ingest_folder(folder, config)
    assert all(d.status == "ingested" for d in first_result.documents)

    # A fresh KnowledgeBase over the same checkpoint database: the work is
    # recognised as already done rather than repeated.
    second = KnowledgeBase()
    second_result = second.ingest_folder(folder, config)
    assert all(d.status == "replayed" for d in second_result.documents), [
        d.status for d in second_result.documents
    ]
    assert second_result.embedding_calls == 0
    assert all(d.doc_id for d in second_result.documents), (
        "a replayed document must still report which document it was"
    )


def test_ingest_rejects_a_missing_folder():
    try:
        KnowledgeBase().ingest_folder(os.path.join(tempfile.mkdtemp(), "nope"))
    except AdapterError:
        pass
    else:
        raise AssertionError("expected AdapterError for a missing directory")


# --------------------------------------------------------------------------
# Ask - extractive (no LLM)
# --------------------------------------------------------------------------


def test_extractive_ask_returns_cited_passages_with_geometry():
    kb = KnowledgeBase()
    kb.ingest_folder(_corpus())
    answer = kb.ask("what is the maximum supply voltage?")

    assert answer.grounded
    assert answer.citations, "extractive mode must still cite"
    assert "40V" in answer.text, answer.text[:200]
    for citation in answer.citations:
        assert citation.page >= 1
        assert citation.chunk_id and citation.doc_id
        assert citation.quote
        assert kb.chunk(citation.chunk_id) is not None, "citations must resolve"
    assert answer.pages


def test_retrieval_prefers_the_right_document():
    kb = KnowledgeBase()
    kb.ingest_folder(_corpus())
    voltage = kb.ask("voltage limit", AskConfig(top_k=2))
    refund = kb.ask("how long do refunds take", AskConfig(top_k=2))
    assert "40V" in voltage.text
    assert "ten business days" in refund.text, refund.text[:200]


def test_lexical_retrieval_finds_an_exact_identifier():
    """Dense retrieval is weakest exactly here, which is why the hybrid exists."""
    kb = KnowledgeBase()
    kb.ingest_folder(_corpus())
    answer = kb.ask("INV-88213", AskConfig(top_k=3))
    assert answer.citations
    assert any("INV-88213" in c.quote for c in answer.citations)
    assert answer.trace is not None
    assert answer.trace.lexical_hits, "the keyword index should have contributed"


def test_empty_index_refuses_rather_than_inventing():
    kb = KnowledgeBase()
    answer = kb.ask("anything at all")
    assert not answer.grounded
    assert answer.citations == []
    assert "do not contain" in answer.text

    silent = KnowledgeBase().ask("x", AskConfig(refuse_without_context=False))
    assert silent.text == "" and not silent.grounded


def test_off_topic_question_is_refused():
    """Rank fusion always returns something from a non-empty index, so without a
    relevance gate a knowledge base answers every question — including ones its
    corpus knows nothing about."""
    kb = KnowledgeBase()
    kb.ingest_folder(_corpus())

    refused = kb.ask("what is the airspeed velocity of an unladen swallow?")
    assert not refused.grounded
    assert refused.citations == []
    assert "do not contain" in refused.text

    # On-topic questions must still pass the same gate.
    assert kb.ask("what is the voltage limit?").grounded
    assert kb.ask("how long do refunds take?").grounded
    assert kb.ask("INV-88213").grounded


def test_relevance_gate_can_be_disabled():
    kb = KnowledgeBase()
    kb.ingest_folder(_corpus())
    answer = kb.ask(
        "airspeed velocity of an unladen swallow",
        AskConfig(relevance_gate=False),
    )
    assert answer.grounded, "with the gate off, retrieval returns its best guess"


def test_dense_arm_of_the_gate_is_opt_in():
    """A cosine floor is not portable across embedders, so it stays off until the
    caller calibrates one. Measured: under HashingEmbedder an irrelevant query can
    out-score a relevant one, which is why term overlap is the default signal."""
    kb = KnowledgeBase()
    kb.ingest_folder(_corpus())
    assert AskConfig().min_dense_similarity is None

    # A floor above every score in this corpus leaves only term overlap, so an
    # overlapping query still passes and a non-overlapping one still fails.
    strict = AskConfig(min_dense_similarity=0.99)
    assert kb.ask("voltage", strict).grounded
    assert not kb.ask("airspeed swallow", strict).grounded

    # A floor below the off-topic query's own score admits it, which is precisely
    # the mis-calibration the default avoids.
    loose = AskConfig(min_dense_similarity=0.05)
    assert kb.ask("airspeed velocity of an unladen swallow", loose).grounded


def test_stopword_only_query_is_not_refused_on_a_technicality():
    kb = KnowledgeBase()
    kb.ingest_folder(_corpus())
    answer = kb.ask("what is the")
    assert answer.grounded, "no content terms means no testable signal, not a refusal"


def test_empty_query_rejected():
    kb = KnowledgeBase()
    for query in ("", "   "):
        try:
            kb.ask(query)
        except ValueError:
            continue
        raise AssertionError("expected ValueError for an empty query")


# --------------------------------------------------------------------------
# Ask - generated (with an LLM)
# --------------------------------------------------------------------------


def test_generated_answer_resolves_citations_to_pages():
    kb = KnowledgeBase(llm=ScriptedLLM(responses=["The limit is 40V [1]."]))
    kb.ingest_folder(_corpus())
    answer = kb.ask("what is the voltage limit?")

    assert answer.grounded
    assert len(answer.citations) == 1
    citation = answer.citations[0]
    assert citation.marker == 1
    assert citation.chunk_id == answer.chunks[0].chunk_id
    assert citation.page >= 1
    assert answer.unverified_markers == []
    assert answer.usage.input_tokens > 0


def test_invented_source_markers_are_surfaced_not_rendered():
    """The most important failure to catch: the model citing a source it was
    never shown."""
    kb = KnowledgeBase(llm=ScriptedLLM(responses=["Per the appendix [9], it is 60V."]))
    kb.ingest_folder(_corpus())
    answer = kb.ask("voltage limit", AskConfig(top_k=2))

    assert answer.unverified_markers == [9]
    assert answer.citations == [], "an invented marker must not become a citation"
    assert not answer.grounded, "an answer citing nothing real is not grounded"


def test_an_uncited_answer_is_reported_as_ungrounded():
    kb = KnowledgeBase(llm=ScriptedLLM(responses=["It is definitely 40V."]))
    kb.ingest_folder(_corpus())
    answer = kb.ask("voltage limit")
    assert answer.citations == []
    assert not answer.grounded
    assert answer.text  # the text is still returned; the caller decides


def test_the_model_only_sees_numbered_sources():
    captured: list[str] = []

    def handler(messages):
        captured.append(messages[-1].content)
        return "Answer [1]."

    kb = KnowledgeBase(llm=ScriptedLLM(handler=handler))
    kb.ingest_folder(_corpus())
    kb.ask("voltage limit", AskConfig(top_k=2))

    prompt = captured[0]
    assert prompt.startswith("Sources:")
    assert "[1]" in prompt and "page" in prompt
    assert "Question: voltage limit" in prompt


def test_context_char_limit_is_enforced():
    captured: list[str] = []
    kb = KnowledgeBase(
        llm=ScriptedLLM(handler=lambda m: (captured.append(m[-1].content), "ok [1]")[1]),
        chunk_config=ChunkConfig(max_tokens=64),
    )
    kb.ingest_folder(_corpus())
    kb.ask("voltage", AskConfig(top_k=20, context_char_limit=200))
    assert len(captured[0]) < 900, len(captured[0])


def test_reranker_is_applied_when_supplied():
    class PreferRefunds:
        def score(self, query, items):
            return [1.0 if "refund" in i.payload.get("text", "").lower() else 0.0
                    for i in items]

    kb = KnowledgeBase(reranker=PreferRefunds())
    kb.ingest_folder(_corpus())
    answer = kb.ask("voltage limit", AskConfig(top_k=1, rerank_budget=10))
    assert answer.trace is not None and answer.trace.reranked
    assert "refund" in answer.text.lower(), (
        "the reranker should have overridden first-stage retrieval"
    )


def test_trace_explains_the_answer():
    kb = KnowledgeBase(llm=ScriptedLLM(responses=["ok [1]"]))
    kb.ingest_folder(_corpus())
    answer = kb.ask("grounding conductor rating", AskConfig(top_k=3))
    trace = answer.trace
    assert trace is not None
    assert trace.query == "grounding conductor rating"
    assert trace.fused_ids
    assert trace.used_ids == [c.chunk_id for c in answer.chunks]
    assert set(trace.used_ids) <= set(trace.fused_ids)


def test_components_can_be_swapped_wholesale():
    """The spine's payoff: a different embedder and no lexical index, same code."""
    kb = KnowledgeBase(embedder=HashingEmbedder(64), lexical_index=None)
    result = kb.ingest_folder(_corpus())
    assert result.ok
    answer = kb.ask("voltage limit")
    assert answer.citations
    assert answer.trace is not None and answer.trace.lexical_hits == []


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
