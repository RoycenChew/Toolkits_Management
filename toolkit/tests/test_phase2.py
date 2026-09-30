"""Tests for chunking, caching, the governor and bounded concurrency.

Each test asserts the property that makes the component worth having, not merely
that it runs. Run standalone: python toolkit/tests/test_phase2.py
"""
from __future__ import annotations

import os
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from toolkit.adapters import HashingEmbedder, ScriptedLLM  # noqa: E402
from toolkit.cache import CachedEmbedder, CachedLLM, SqliteCache, make_key  # noqa: E402
from toolkit.chunking import (  # noqa: E402
    ChunkConfig,
    ChunkerComponent,
    ChunkRequest,
    estimate_tokens,
    split_sentences,
)
from toolkit.concurrency import (  # noqa: E402
    MapConfig,
    MapFailed,
    batched,
    bounded_map,
    embed_all,
)
from toolkit.core import (  # noqa: E402
    AdapterError,
    BBox,
    Block,
    BlockType,
    Document,
    Message,
    Provenance,
    RateLimited,
    Usage,
)
from toolkit.governor import BudgetExceeded, GovernedLLM, GovernorConfig  # noqa: E402

# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------


def _doc() -> Document:
    def prov(page: int, y: float) -> Provenance:
        return Provenance(page=page, bbox=BBox(50, y, 500, y + 12))

    return Document(
        doc_id="doc:test",
        source_uri="/tmp/manual.pdf",
        page_count=2,
        blocks=[
            Block("Safety Manual", BlockType.HEADING, prov(1, 40), level=1),
            Block("Voltage", BlockType.HEADING, prov(1, 80), level=2),
            Block(
                "The supply must not exceed 40V. Exceeding it voids the warranty. "
                "Always verify with a calibrated meter before connecting the load.",
                BlockType.PARAGRAPH,
                prov(1, 100),
            ),
            Block("Check the meter.", BlockType.LIST_ITEM, prov(1, 140)),
            Block("Log the reading.", BlockType.LIST_ITEM, prov(1, 155)),
            Block("Grounding", BlockType.HEADING, prov(2, 40), level=2),
            Block(
                "Bond the chassis to earth before energising the circuit.",
                BlockType.PARAGRAPH,
                prov(2, 60),
            ),
            Block("Page 2 of 2", BlockType.PAGE_FOOTER, prov(2, 770)),
        ],
    )


def test_chunks_carry_page_and_bbox_provenance():
    """The property this component exists for. Without it, no citation."""
    result = ChunkerComponent().execute(ChunkRequest(_doc(), ChunkConfig(max_tokens=64)))
    assert result.chunks
    for chunk in result.chunks:
        assert chunk.provenances, "every chunk must know where it came from"
        for prov in chunk.provenances:
            assert prov.page in (1, 2)
            assert prov.bbox is not None
            assert prov.bbox.width > 0 and prov.bbox.height > 0
    covered = {page for chunk in result.chunks for page in chunk.pages}
    assert covered == {1, 2}, "no page may be lost during chunking"


def test_provenance_is_one_region_per_page_not_one_impossible_box():
    doc = Document(
        doc_id="doc:multi",
        page_count=2,
        blocks=[
            Block("A sentence on page one.", BlockType.PARAGRAPH,
                  Provenance(1, BBox(0, 700, 100, 712))),
            Block("A sentence on page two.", BlockType.PARAGRAPH,
                  Provenance(2, BBox(0, 40, 100, 52))),
        ],
    )
    # A generous budget forces both pages into one chunk.
    result = ChunkerComponent().execute(ChunkRequest(doc, ChunkConfig(max_tokens=512)))
    spanning = [c for c in result.chunks if len(c.pages) > 1]
    assert spanning, "the budget should have merged both pages into one chunk"
    assert len(spanning[0].provenances) == 2, "one region per page, never a merged box"


def test_heading_breadcrumb_is_prepended():
    result = ChunkerComponent().execute(ChunkRequest(_doc(), ChunkConfig(max_tokens=64)))
    voltage = [c for c in result.chunks if "40V" in c.text]
    assert voltage, "the voltage text must survive"
    assert voltage[0].text.startswith("Safety Manual > Voltage"), voltage[0].text[:60]
    assert voltage[0].metadata["heading_path"] == ["Safety Manual", "Voltage"]


def test_sections_are_not_packed_together():
    """split_on_heading must win over filling the token budget."""
    result = ChunkerComponent().execute(ChunkRequest(_doc(), ChunkConfig(max_tokens=2048)))
    for chunk in result.chunks:
        assert not ("40V" in chunk.text and "Bond the chassis" in chunk.text), (
            "Voltage and Grounding are different sections and must not share a chunk"
        )

    merged = ChunkerComponent().execute(
        ChunkRequest(_doc(), ChunkConfig(max_tokens=2048, split_on_heading=False))
    )
    assert len(merged.chunks) < len(result.chunks), (
        "disabling the heading split should produce fewer, larger chunks"
    )


def test_token_budget_is_respected():
    long_doc = Document(
        doc_id="doc:long",
        page_count=1,
        blocks=[
            Block(
                " ".join("Sentence number %d is here." % i for i in range(120)),
                BlockType.PARAGRAPH,
                Provenance(1, BBox(0, 0, 100, 10)),
            )
        ],
    )
    config = ChunkConfig(max_tokens=100, overlap_tokens=20, min_tokens=10)
    result = ChunkerComponent().execute(ChunkRequest(long_doc, config))
    assert len(result.chunks) > 3
    assert not result.oversized, result.oversized
    for chunk_id, tokens in result.token_estimates.items():
        assert tokens <= config.max_tokens, (chunk_id, tokens)


def test_unsplittable_long_unit_is_word_split_not_dropped():
    """A 3,000-token table row must not become an oversized chunk that an
    embedding endpoint would reject, and must not lose text either."""
    wall = " ".join("token%d" % i for i in range(800))
    doc = Document(
        doc_id="doc:wall",
        page_count=1,
        blocks=[Block(wall, BlockType.TABLE, Provenance(1, BBox(0, 0, 10, 10)))],
    )
    config = ChunkConfig(max_tokens=64, overlap_tokens=0, min_tokens=8)
    result = ChunkerComponent().execute(ChunkRequest(doc, config))
    assert len(result.chunks) > 5
    assert all(t <= config.max_tokens for t in result.token_estimates.values())
    # No text lost: every original word appears somewhere.
    combined = " ".join(c.text for c in result.chunks)
    assert "token0" in combined and "token799" in combined


def test_overlap_repeats_whole_sentences():
    doc = Document(
        doc_id="doc:ov",
        page_count=1,
        blocks=[
            Block(
                "Alpha one here. Beta two here. Gamma three here. Delta four here. "
                "Epsilon five here. Zeta six here.",
                BlockType.PARAGRAPH,
                Provenance(1, BBox(0, 0, 100, 10)),
            )
        ],
    )
    result = ChunkerComponent().execute(
        ChunkRequest(doc, ChunkConfig(max_tokens=16, overlap_tokens=6, min_tokens=2,
                                      include_heading_path=False))
    )
    assert len(result.chunks) >= 2, [c.text for c in result.chunks]
    texts = [c.text for c in result.chunks]
    shared = any(
        any(sentence and sentence in texts[i + 1] for sentence in split_sentences(texts[i]))
        for i in range(len(texts) - 1)
    )
    assert shared, "consecutive chunks should share at least one whole sentence"
    # Overlap must never sever a sentence mid-way.
    for text in texts:
        assert not text.endswith(("the", "a", "of", "and")), text[-30:]


def test_furniture_excluded_but_reachable():
    result = ChunkerComponent().execute(ChunkRequest(_doc(), ChunkConfig()))
    assert all("Page 2 of 2" not in c.text for c in result.chunks)
    kept = ChunkerComponent().execute(
        ChunkRequest(_doc(), ChunkConfig(keep_furniture=True))
    )
    assert any("Page 2 of 2" in c.text for c in kept.chunks)


def test_chunk_ids_are_stable_across_runs():
    a = ChunkerComponent().execute(ChunkRequest(_doc(), ChunkConfig(max_tokens=64)))
    b = ChunkerComponent().execute(ChunkRequest(_doc(), ChunkConfig(max_tokens=64)))
    assert [c.chunk_id for c in a.chunks] == [c.chunk_id for c in b.chunks]
    assert all(c.chunk_id.startswith("doc:test#") for c in a.chunks)
    # Content hash is metadata, so an edit replaces slot N rather than orphaning it.
    assert all("content_hash" in c.metadata for c in a.chunks)


def test_list_items_stay_whole_and_render_as_a_list():
    result = ChunkerComponent().execute(ChunkRequest(_doc(), ChunkConfig(max_tokens=2048)))
    listy = [c for c in result.chunks if "Check the meter" in c.text]
    assert listy
    assert "- Check the meter." in listy[0].text
    assert "- Log the reading." in listy[0].text


def test_semantic_boundaries_use_the_embedder_when_configured():
    doc = Document(
        doc_id="doc:sem",
        page_count=1,
        blocks=[
            Block(
                "Refund policy covers returned goods. Refunds take ten days. "
                "Hydraulic pressure valves require annual calibration.",
                BlockType.PARAGRAPH,
                Provenance(1, BBox(0, 0, 100, 10)),
            )
        ],
    )
    plain = ChunkerComponent().execute(
        ChunkRequest(doc, ChunkConfig(max_tokens=2048, include_heading_path=False))
    )
    semantic = ChunkerComponent(embedder=HashingEmbedder(128)).execute(
        ChunkRequest(
            doc,
            ChunkConfig(
                max_tokens=2048,
                include_heading_path=False,
                semantic_threshold=0.35,
                min_tokens=1,
            ),
        )
    )
    assert len(plain.chunks) == 1
    assert len(semantic.chunks) > 1, "a topic shift should force a boundary"


def test_empty_document_and_config_validation():
    empty = ChunkerComponent().execute(
        ChunkRequest(Document(doc_id="doc:empty", blocks=[]))
    )
    assert empty.chunks == [] and empty.token_estimates == {}
    for kwargs in (
        {"max_tokens": 8},
        {"max_tokens": 100, "overlap_tokens": 100},
        {"max_tokens": 100, "min_tokens": 100},
        {"semantic_threshold": 1.5},
    ):
        try:
            ChunkConfig(**kwargs)
        except ValueError:
            continue
        raise AssertionError("expected ValueError for " + str(kwargs))


def test_sentence_splitter_keeps_abbreviations_whole():
    assert split_sentences("Dr. Chen signed it. Then we shipped.") == [
        "Dr. Chen signed it.",
        "Then we shipped.",
    ]
    assert split_sentences("Use approx. 40V here. Verify it.") == [
        "Use approx. 40V here.",
        "Verify it.",
    ]
    assert split_sentences("") == []
    assert estimate_tokens("") == 0
    assert estimate_tokens("four short words here") > 0


def test_custom_token_counter_is_honoured():
    calls = {"n": 0}

    def counter(text: str) -> int:
        calls["n"] += 1
        return len(text.split())

    ChunkerComponent().execute(
        ChunkRequest(_doc(), ChunkConfig(max_tokens=20, token_counter=counter))
    )
    assert calls["n"] > 0, "a supplied token_counter must actually be used"


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------


def test_embeddings_cache_per_text_not_per_batch():
    """The property that makes the cache useful: editing one document must not
    re-embed the whole corpus."""
    cache = SqliteCache()
    embedder = CachedEmbedder(HashingEmbedder(64), cache)

    first = embedder.embed(["alpha", "beta", "gamma"])
    assert embedder.calls == 3

    # A different batch that overlaps: only the new text should be computed.
    second = embedder.embed(["gamma", "delta", "alpha"])
    assert embedder.calls == 4, "only 'delta' was new"
    assert second[0] == first[2] and second[2] == first[0], "order must be preserved"

    assert embedder.embed(["alpha"])[0] == first[0]
    assert embedder.calls == 4
    assert cache.hits > 0 and cache.count() == 4


def test_llm_cache_keys_on_everything_that_changes_the_answer():
    cache = SqliteCache()
    inner = ScriptedLLM(handler=lambda messages: "reply:" + messages[-1].content)
    llm = CachedLLM(inner, cache)

    a = llm.complete([Message("user", "hello")])
    b = llm.complete([Message("user", "hello")])
    assert a.text == b.text
    assert llm.calls == 1, "the identical request must hit the cache"
    assert b.usage.cached is True and b.usage.cost_usd == 0.0

    llm.complete([Message("user", "hello")], max_tokens=16)
    assert llm.calls == 2, "max_tokens is part of the key"

    llm.complete([Message("system", "be terse"), Message("user", "hello")])
    assert llm.calls == 3, "the message list is part of the key"


def test_nonzero_temperature_bypasses_the_cache():
    cache = SqliteCache()
    llm = CachedLLM(ScriptedLLM(handler=lambda m: "x"), cache)
    llm.complete([Message("user", "q")], temperature=0.7)
    llm.complete([Message("user", "q")], temperature=0.7)
    assert llm.calls == 2, "the caller asked for variation; caching would deny it"
    assert cache.count() == 0


def test_cache_survives_a_new_store_object():
    path = os.path.join(tempfile.mkdtemp(), "cache.db")
    store = SqliteCache(path)
    first = CachedEmbedder(HashingEmbedder(32), store)
    vectors = first.embed(["persisted text"])
    assert first.calls == 1
    store.close()

    reopened = SqliteCache(path)
    second = CachedEmbedder(HashingEmbedder(32), reopened)
    assert second.embed(["persisted text"])[0] == vectors[0]
    assert second.calls == 0, "a restart must not re-pay for embeddings"
    reopened.close()


def test_make_key_is_order_insensitive_for_dicts():
    assert make_key("ns", {"a": 1, "b": 2}) == make_key("ns", {"b": 2, "a": 1})
    assert make_key("ns", "x") != make_key("ns", "y")
    assert make_key("other", "x") != make_key("ns", "x")


def test_cache_clear_by_namespace():
    cache = SqliteCache()
    CachedEmbedder(HashingEmbedder(16), cache).embed(["a", "b"])
    CachedLLM(ScriptedLLM(responses=["r"]), cache).complete([Message("user", "q")])
    assert cache.count() == 3
    assert cache.clear("embed") == 2
    assert cache.count() == 1


# --------------------------------------------------------------------------
# Governor
# --------------------------------------------------------------------------


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def test_budget_is_checked_before_the_call_not_after():
    clock = _FakeClock()
    inner = ScriptedLLM(handler=lambda m: "ok")
    llm = GovernedLLM(
        inner,
        GovernorConfig(max_total_tokens=200, estimate_output_tokens=100),
        clock=clock.time,
        sleep=clock.sleep,
    )
    llm.complete([Message("user", "short question")])
    try:
        llm.complete([Message("user", "x" * 4000)])
    except BudgetExceeded as exc:
        assert "token budget" in str(exc)
        assert len(inner.calls) == 1, "the over-budget call must never reach the model"
    else:
        raise AssertionError("expected BudgetExceeded")


def test_call_and_cost_ceilings():
    clock = _FakeClock()
    llm = GovernedLLM(
        ScriptedLLM(handler=lambda m: "ok"),
        GovernorConfig(max_calls=2),
        clock=clock.time,
        sleep=clock.sleep,
    )
    llm.complete([Message("user", "a")])
    llm.complete([Message("user", "b")])
    try:
        llm.complete([Message("user", "c")])
    except BudgetExceeded as exc:
        assert "call limit" in str(exc)
    else:
        raise AssertionError("expected BudgetExceeded on the third call")
    assert llm.state.calls == 2


def test_rate_limit_uses_a_sliding_window():
    clock = _FakeClock()
    llm = GovernedLLM(
        ScriptedLLM(handler=lambda m: "ok"),
        GovernorConfig(max_requests_per_minute=2),
        clock=clock.time,
        sleep=clock.sleep,
    )
    llm.complete([Message("user", "1")])
    llm.complete([Message("user", "2")])
    assert clock.slept == [], "the first two calls are within the limit"

    llm.complete([Message("user", "3")])
    assert clock.slept, "the third call must wait for the window to slide"
    assert abs(clock.slept[0] - 60.0) < 1e-6
    assert llm.state.throttled_seconds > 0


def test_rate_limited_is_retried_and_honours_retry_after():
    clock = _FakeClock()
    attempts = {"n": 0}

    class Flaky:
        def complete(self, messages, temperature=0.0, max_tokens=None):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RateLimited("slow down", retry_after=7.0)
            return ScriptedLLM(responses=["recovered"]).complete(messages)

    llm = GovernedLLM(
        Flaky(),
        GovernorConfig(max_attempts=4),
        clock=clock.time,
        sleep=clock.sleep,
    )
    completion = llm.complete([Message("user", "q")])
    assert completion.text == "recovered"
    assert attempts["n"] == 3
    assert clock.slept == [7.0, 7.0], "the provider's Retry-After must be honoured"
    assert llm.state.retries == 2


def test_retries_are_bounded_and_then_surface():
    clock = _FakeClock()

    class AlwaysLimited:
        def complete(self, messages, temperature=0.0, max_tokens=None):
            raise RateLimited("nope")

    llm = GovernedLLM(
        AlwaysLimited(),
        GovernorConfig(max_attempts=2, initial_backoff=0.1, jitter=0.0),
        clock=clock.time,
        sleep=clock.sleep,
    )
    try:
        llm.complete([Message("user", "q")])
    except AdapterError as exc:
        assert "2 attempts" in str(exc)
    else:
        raise AssertionError("expected AdapterError after exhausting attempts")


def test_governor_accumulates_usage_and_composes_with_the_cache():
    clock = _FakeClock()
    inner = ScriptedLLM(handler=lambda m: "answer")
    cache = SqliteCache()
    # Governor outside, cache inside: a cache hit costs no budget and no
    # rate-limit slot, which is the composition you want.
    llm = GovernedLLM(
        CachedLLM(inner, cache),
        GovernorConfig(max_requests_per_minute=60),
        clock=clock.time,
        sleep=clock.sleep,
    )
    llm.complete([Message("user", "same")])
    spent_after_first = llm.state.usage.input_tokens
    llm.complete([Message("user", "same")])
    assert len(inner.calls) == 1, "the second call was served from cache"
    assert isinstance(llm.state.usage, Usage)
    assert llm.state.usage.input_tokens >= spent_after_first
    assert llm.state.calls == 2


# --------------------------------------------------------------------------
# Concurrency
# --------------------------------------------------------------------------


def test_results_stay_in_input_order():
    import time as real_time

    def slow_for_early_items(n: int) -> int:
        # The first items finish last, so any completion-order implementation
        # would scramble the output here.
        real_time.sleep(0.02 if n < 3 else 0.0)
        return n * n

    outcome = bounded_map(slow_for_early_items, list(range(10)), MapConfig(max_workers=5))
    assert outcome.ok
    assert outcome.results == [n * n for n in range(10)]


def test_failures_are_attributed_by_index_and_do_not_cancel_the_batch():
    def fn(n: int) -> int:
        if n == 3:
            raise ValueError("bad item")
        return n

    outcome = bounded_map(fn, list(range(6)), MapConfig(max_workers=3))
    assert not outcome.ok
    assert [index for index, _ in outcome.failures] == [3]
    assert outcome.results[3] is None
    assert outcome.values() == [0, 1, 2, 4, 5]


def test_fail_fast_raises_with_the_first_index():
    def fn(n: int) -> int:
        if n in (2, 4):
            raise AdapterError("boom")
        return n

    try:
        bounded_map(fn, list(range(6)), MapConfig(max_workers=2, fail_fast=True))
    except MapFailed as exc:
        assert [index for index, _ in exc.failures] == [2, 4]
        assert "index 2" in str(exc)
    else:
        raise AssertionError("expected MapFailed")


def test_only_listed_exceptions_are_retried():
    counts = {"retryable": 0, "programming": 0}

    def retryable(_: int) -> int:
        counts["retryable"] += 1
        raise RateLimited("later")

    def programming(_: int) -> int:
        counts["programming"] += 1
        raise KeyError("typo")

    config = MapConfig(max_attempts=3, initial_backoff=0.0, jitter=0.0)
    bounded_map(retryable, [0], config, sleep=lambda _: None)
    bounded_map(programming, [0], config, sleep=lambda _: None)
    assert counts["retryable"] == 3, "a rate limit is worth retrying"
    assert counts["programming"] == 1, "a KeyError is a bug, not a transient failure"


def test_embed_all_batches_and_preserves_alignment():
    texts = ["text number %d" % i for i in range(37)]
    embedder = HashingEmbedder(32)
    vectors = embed_all(embedder, texts, batch_size=8)
    assert len(vectors) == 37
    assert all(len(v) == 32 for v in vectors)
    # Alignment is the thing that breaks silently, so check it against a direct
    # single-item embedding.
    assert vectors[20] == embedder.embed([texts[20]])[0]
    assert embed_all(embedder, [], batch_size=8) == []


def test_batched_and_config_validation():
    assert [list(b) for b in batched([1, 2, 3, 4, 5], 2)] == [[1, 2], [3, 4], [5]]
    for kwargs in ({"max_workers": 0}, {"max_attempts": 0}):
        try:
            MapConfig(**kwargs)
        except ValueError:
            continue
        raise AssertionError("expected ValueError for " + str(kwargs))
    try:
        batched([1], 0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for size 0")


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
