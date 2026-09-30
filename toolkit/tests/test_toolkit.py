"""Behavioural checks for the four extracted components.

Run with: python -m pytest toolkit/tests -q
Or standalone: python toolkit/tests/test_toolkit.py
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from toolkit.doc_layout import (  # noqa: E402
    BBox,
    BlockType,
    DocLayoutComponent,
    LayoutConfig,
    LayoutRequest,
    TextSpan,
)
from toolkit.durable_steps import (  # noqa: E402
    DurableStepsComponent,
    LeaseNotAcquired,
    NonRetryableError,
    RetryPolicy,
    RunStatus,
    SqliteCheckpointStore,
    Step,
    WorkflowRequest,
)
from toolkit.entity_resolution import (  # noqa: E402
    ComparisonLevel,
    EntityResolutionComponent,
    FieldComparison,
    ResolutionConfig,
    ResolutionRequest,
    affine_gap_similarity,
)
from toolkit.hybrid_ranker import (  # noqa: E402
    FusionConfig,
    FusionMethod,
    FusionRequest,
    HybridRankerComponent,
    RankedItem,
    RankedList,
)

# --------------------------------------------------------------------------
# 1. Hybrid ranker
# --------------------------------------------------------------------------


def test_rrf_beats_incomparable_scales():
    """An item ranked well by both retrievers must outrank one ranked top by
    only one, even though BM25 and cosine scores share no scale."""
    bm25 = RankedList("bm25", [RankedItem("a", 14.2), RankedItem("b", 9.1), RankedItem("c", 3.0)])
    dense = RankedList("dense", [RankedItem("c", 0.91), RankedItem("a", 0.80), RankedItem("d", 0.55)])
    result = HybridRankerComponent().execute(
        FusionRequest("q", [bm25, dense], FusionConfig(top_k=4))
    )
    ids = [i.id for i in result.items]
    assert ids[0] == "a", ids
    assert result.candidates_considered == 4
    assert result.items[0].ranks == {"bm25": 1, "dense": 2}
    # b appears in one list only and must rank below both dual-list items.
    assert ids.index("b") > ids.index("c")


def test_weighting_and_normalisation_modes():
    a = RankedList("a", [RankedItem("x", 1.0), RankedItem("y", 0.9)], weight=3.0)
    b = RankedList("b", [RankedItem("y", 100.0), RankedItem("x", 1.0)], weight=1.0)
    rrf = HybridRankerComponent().execute(FusionRequest("q", [a, b], FusionConfig()))
    assert rrf.items[0].id == "x", "the heavily weighted list should decide"
    rel = HybridRankerComponent().execute(
        FusionRequest("q", [a, b], FusionConfig(method=FusionMethod.RELATIVE_SCORE))
    )
    assert {i.id for i in rel.items} == {"x", "y"}
    dist = HybridRankerComponent().execute(
        FusionRequest("q", [a, b], FusionConfig(method=FusionMethod.DISTRIBUTION))
    )
    assert {i.id for i in dist.items} == {"x", "y"}


def test_reranker_overrides_retrieval_and_is_budgeted():
    lists = [
        RankedList("bm25", [RankedItem("a", 9), RankedItem("b", 8), RankedItem("c", 7)]),
        RankedList("dense", [RankedItem("a", 0.9), RankedItem("d", 0.5)]),
    ]

    class OnlyDMatters:
        def score(self, query, items):
            return [1.0 if i.id == "d" else 0.0 for i in items]

    result = HybridRankerComponent().execute(
        FusionRequest("q", lists, FusionConfig(top_k=4, rerank_budget=4)),
        reranker=OnlyDMatters(),
    )
    assert result.reranked is True
    assert result.items[0].id == "d"
    assert "_rerank" in result.items[0].contributions

    # A budget of 1 can only reorder the single head item, so d stays put.
    narrow = HybridRankerComponent().execute(
        FusionRequest("q", lists, FusionConfig(top_k=4, rerank_budget=1)),
        reranker=OnlyDMatters(),
    )
    assert narrow.items[0].id == "a"


def test_reranker_length_mismatch_is_rejected():
    class Broken:
        def score(self, query, items):
            return [1.0]

    try:
        HybridRankerComponent().execute(
            FusionRequest(
                "q",
                [RankedList("s", [RankedItem("a"), RankedItem("b")])],
                FusionConfig(rerank_budget=2),
            ),
            reranker=Broken(),
        )
    except ValueError as exc:
        assert "2 items" in str(exc)
    else:
        raise AssertionError("expected ValueError on score/item count mismatch")


def test_mmr_suppresses_a_near_duplicate():
    lists = [
        RankedList(
            "s",
            [RankedItem("a", 1.0), RankedItem("a_dup", 0.99), RankedItem("far", 0.5)],
        )
    ]
    vectors = {"a": [1.0, 0.0], "a_dup": [0.99, 0.01], "far": [0.0, 1.0]}
    diverse = HybridRankerComponent().execute(
        FusionRequest("q", lists, FusionConfig(top_k=2, mmr_lambda=0.3), vectors=vectors)
    )
    assert [i.id for i in diverse.items] == ["a", "far"]
    plain = HybridRankerComponent().execute(
        FusionRequest("q", lists, FusionConfig(top_k=2), vectors=vectors)
    )
    assert [i.id for i in plain.items] == ["a", "a_dup"]


def test_empty_and_degenerate_input():
    empty = HybridRankerComponent().execute(FusionRequest("q", []))
    assert empty.items == [] and empty.candidates_considered == 0
    zero_weight = HybridRankerComponent().execute(
        FusionRequest("q", [RankedList("s", [RankedItem("a")], weight=0.0)])
    )
    assert zero_weight.items == []


# --------------------------------------------------------------------------
# 2. Entity resolution
# --------------------------------------------------------------------------


def test_affine_gap_prefers_one_omission_over_scattered_typos():
    """The whole reason to use affine gap: 'Robert J Smith' vs 'Robert Smith'
    (one clean omission) must score higher than an equal-length string with the
    same number of differing characters spread around."""
    omission = affine_gap_similarity("robert j smith", "robert smith")
    scattered = affine_gap_similarity("robert j smith", "rXbert j smXth")
    assert omission > scattered, (omission, scattered)
    assert affine_gap_similarity("abc", "abc") == 1.0
    assert affine_gap_similarity("abc", "") == 0.0
    assert affine_gap_similarity("", "") == 1.0
    assert 0.0 <= affine_gap_similarity("acme corp", "zzz ltd") <= 1.0


def test_resolution_finds_duplicates_without_labels():
    records = {
        "1": {"name": "Robert Smith", "city": "London", "postcode": "SW1A 1AA"},
        "2": {"name": "Robert J Smith", "city": "London", "postcode": "SW1A 1AA"},
        "3": {"name": "Bob Smith", "city": "Londn", "postcode": "SW1A 1AA"},
        "4": {"name": "Alice Nakamura", "city": "Osaka", "postcode": "530-0001"},
        "5": {"name": "Alice Nakamura", "city": "Osaka", "postcode": "530-0001"},
        "6": {"name": "Wei Chen", "city": "Taipei", "postcode": "100"},
    }
    config = ResolutionConfig(
        comparisons=[
            FieldComparison("name"),
            FieldComparison(
                "city",
                levels=(ComparisonLevel("exact", 1.0), ComparisonLevel("close", 0.85)),
            ),
            FieldComparison("postcode", levels=(ComparisonLevel("exact", 1.0),)),
        ],
        match_threshold=0.85,
    )
    result = EntityResolutionComponent().execute(
        ResolutionRequest(records=records, config=config)
    )

    clusters = {frozenset(c.record_ids) for c in result.clusters}
    assert frozenset({"4", "5"}) in clusters, clusters
    # The exact-duplicate pair must score above the near-duplicate pair.
    scores = {
        (min(p.left, p.right), max(p.left, p.right)): p.match_probability
        for p in result.scored_pairs
    }
    assert scores[("4", "5")] >= scores.get(("1", "3"), 0.0)
    # Wei Chen shares nothing and must stay alone.
    assert frozenset({"6"}) in clusters
    # Blocking must actually have avoided work.
    assert result.pairs_compared < 6 * 5 // 2
    assert result.pairs_avoided > 0
    assert result.selected_predicates


def test_match_weight_is_interpretable_and_explains_itself():
    records = {
        "a": {"name": "Acme Holdings", "ref": "X-1000"},
        "b": {"name": "Acme Holdings", "ref": "X-1000"},
        "c": {"name": "Zenith Foods", "ref": "Q-77"},
        "d": {"name": "Zenith Foods", "ref": "Q-77"},
        "e": {"name": "Acme Holdings", "ref": "Y-2222"},
    }
    config = ResolutionConfig(
        comparisons=[
            FieldComparison("name", levels=(ComparisonLevel("exact", 1.0),)),
            FieldComparison("ref", levels=(ComparisonLevel("exact", 1.0),)),
        ]
    )
    result = EntityResolutionComponent().execute(
        ResolutionRequest(records=records, config=config)
    )
    by_pair = {(p.left, p.right): p for p in result.scored_pairs}
    full = by_pair.get(("a", "b"))
    assert full is not None
    # Every scored pair carries the level reached per field: that is the audit
    # trail, and it must cover every configured comparison.
    assert set(full.pattern) == {"name", "ref"}
    assert full.pattern["name"] == "exact" and full.pattern["ref"] == "exact"
    partial = by_pair.get(("a", "e"))
    if partial is not None:
        assert partial.match_weight < full.match_weight
        assert partial.pattern["ref"] == "none"
    assert result.model.iterations >= 1
    assert 0.0 < result.model.lambda_prior < 1.0


def test_average_linkage_resists_chaining():
    """A~B and B~C must not silently merge A with C when A and C differ."""
    records = {
        "a": {"name": "aaaaaaaa"},
        "b": {"name": "aaaabbbb"},
        "c": {"name": "bbbbbbbb"},
    }
    comparisons = [
        FieldComparison(
            "name",
            levels=(ComparisonLevel("exact", 1.0), ComparisonLevel("similar", 0.55)),
        )
    ]
    avg = EntityResolutionComponent().execute(
        ResolutionRequest(
            records,
            ResolutionConfig(comparisons=comparisons, cluster_link="average", match_threshold=0.6),
        )
    )
    conn = EntityResolutionComponent().execute(
        ResolutionRequest(
            records,
            ResolutionConfig(comparisons=comparisons, cluster_link="connected", match_threshold=0.6),
        )
    )
    biggest_avg = max(len(c.record_ids) for c in avg.clusters)
    biggest_conn = max(len(c.record_ids) for c in conn.clusters)
    assert biggest_avg <= biggest_conn
    assert all(0.0 <= c.cohesion <= 1.0 for c in avg.clusters)


def test_labelled_pairs_drive_predicate_selection():
    records = {str(i): {"name": "person " + str(i), "id": "ID%04d" % i} for i in range(30)}
    records["99"] = {"name": "person 1", "id": "ID0001"}
    config = ResolutionConfig(
        comparisons=[FieldComparison("name"), FieldComparison("id")],
        max_predicates=3,
    )
    result = EntityResolutionComponent().execute(
        ResolutionRequest(records, config, labelled_pairs=[("1", "99", True)])
    )
    assert len(result.selected_predicates) <= 3
    pairs = {(p.left, p.right) for p in result.scored_pairs}
    assert ("1", "99") in pairs, "the labelled duplicate must survive blocking"


def test_empty_comparisons_rejected():
    try:
        EntityResolutionComponent().execute(
            ResolutionRequest({"a": {"x": "1"}}, ResolutionConfig(comparisons=[]))
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for empty comparisons")


# --------------------------------------------------------------------------
# 3. Document layout
# --------------------------------------------------------------------------


def _span(text, x0, y0, x1, y1, page=1, size=10.0, bold=False):
    return TextSpan(text, BBox(x0, y0, x1, y1), page, size, bold)


def test_two_column_reading_order():
    """Left column must be read fully before the right, not line-interleaved."""
    spans = [
        _span("Introduction", 50, 40, 200, 58, size=16, bold=True),
        _span("Left line one of the body text here.", 50, 80, 280, 92),
        _span("Left line two continues the same idea.", 50, 96, 280, 108),
        _span("Left line three still the same block.", 50, 112, 280, 124),
        _span("Right line one of the second column.", 330, 80, 560, 92),
        _span("Right line two of the second column.", 330, 96, 560, 108),
        _span("Right line three of that column also.", 330, 112, 560, 124),
    ]
    result = DocLayoutComponent().execute(
        LayoutRequest(spans, page_sizes={1: (600.0, 800.0)})
    )
    body = [b for b in result.blocks if b.type is BlockType.PARAGRAPH]
    order = "".join("L" if "Left" in b.text else "R" for b in body)
    assert order in ("LR", "L", "LRR", "LLR"), (order, [b.text for b in body])
    assert result.columns_per_page[1] == 2
    assert result.blocks[0].type is BlockType.HEADING
    assert result.blocks[0].reading_order == 0


def test_heading_levels_from_font_statistics():
    spans = [
        _span("Document Title", 50, 40, 300, 66, size=22, bold=True),
        _span("Chapter One", 50, 90, 250, 108, size=16, bold=True),
        _span("Body text at the normal size for this document.", 50, 120, 400, 132),
        _span("Body text continues along in the same paragraph.", 50, 136, 400, 148),
        _span("Subsection A", 50, 170, 220, 184, size=13, bold=True),
        _span("More body text under the subsection heading.", 50, 196, 400, 208),
    ]
    result = DocLayoutComponent().execute(
        LayoutRequest(spans, page_sizes={1: (600.0, 800.0)})
    )
    headings = [(b.text, b.level) for b in result.blocks if b.type is BlockType.HEADING]
    assert headings == [("Document Title", 1), ("Chapter One", 2), ("Subsection A", 3)], headings
    assert result.body_font_size == 10.0
    assert result.markdown.startswith("# Document Title")
    assert "### Subsection A" in result.markdown


def test_duplicate_spans_from_ocr_over_text_layer_are_dropped():
    spans = [
        _span("Invoice Number 4471", 50, 50, 250, 64),
        _span("invoice number 4471", 50.4, 50.3, 250.2, 64.1),
        _span("Payable within thirty days of receipt.", 50, 80, 320, 94),
    ]
    result = DocLayoutComponent().execute(LayoutRequest(spans))
    assert result.spans_deduplicated == 1
    assert sum("4471" in b.text for b in result.blocks) == 1


def test_recurring_headers_and_footers_are_classified():
    spans = []
    for page in range(1, 5):
        y = 0.0
        spans.append(_span("ACME CONFIDENTIAL", 50, 10, 200, 22, page=page, size=8))
        spans.append(
            _span("Body content unique to page " + str(page) + " of this report.", 50, 200, 400, 214, page=page)
        )
        spans.append(_span("Page " + str(page) + " of 4", 250, 770, 330, 782, page=page, size=8))
        del y
    result = DocLayoutComponent().execute(
        LayoutRequest(spans, page_sizes={p: (600.0, 800.0) for p in range(1, 5)})
    )
    kinds = {b.type for b in result.blocks}
    assert BlockType.PAGE_HEADER in kinds
    assert BlockType.PAGE_FOOTER in kinds
    # Markdown must exclude furniture even when blocks retain it.
    assert "CONFIDENTIAL" not in result.markdown
    assert "Page 1 of 4" not in result.markdown

    dropped = DocLayoutComponent().execute(
        LayoutRequest(
            spans,
            page_sizes={p: (600.0, 800.0) for p in range(1, 5)},
            config=LayoutConfig(drop_headers_footers=True),
        )
    )
    assert all(
        b.type not in (BlockType.PAGE_HEADER, BlockType.PAGE_FOOTER)
        for b in dropped.blocks
    )


def test_lists_captions_and_provenance():
    spans = [
        _span("- First bullet point in the list.", 60, 100, 300, 114),
        _span("- Second bullet point in the list.", 60, 118, 300, 132),
        _span("Figure 3: Throughput over time.", 60, 160, 290, 174, size=9),
    ]
    result = DocLayoutComponent().execute(LayoutRequest(spans))
    types = [b.type for b in result.blocks]
    assert types.count(BlockType.LIST_ITEM) == 2, [(b.type, b.text) for b in result.blocks]
    assert BlockType.CAPTION in types
    assert result.markdown.count("- ") >= 2
    # Every block must trace back to real input spans on a real page.
    for block in result.blocks:
        assert block.provenance.page == 1
        assert block.provenance.span_indices
        assert all(0 <= i < len(spans) for i in block.provenance.span_indices)
        assert block.provenance.bbox.area > 0


def test_single_column_page_is_not_split_on_a_table_gutter():
    spans = [_span("Narrow left cell", 50, 100 + 20 * i, 140, 112 + 20 * i) for i in range(5)]
    spans += [_span("Narrow right cell", 160, 100 + 20 * i, 250, 112 + 20 * i) for i in range(5)]
    result = DocLayoutComponent().execute(
        LayoutRequest(spans, page_sizes={1: (600.0, 800.0)})
    )
    assert result.columns_per_page[1] == 1, "a table gutter is not a column break"


def test_empty_document():
    result = DocLayoutComponent().execute(LayoutRequest([]))
    assert result.blocks == [] and result.markdown == ""


def test_bbox_rejects_inverted_boxes():
    try:
        BBox(10, 10, 5, 20)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for inverted bbox")


# --------------------------------------------------------------------------
# 4. Durable steps
# --------------------------------------------------------------------------


def test_completed_steps_are_replayed_not_rerun():
    store = SqliteCheckpointStore()
    calls: list[str] = []

    def make(name, value):
        def fn(ctx):
            calls.append(name)
            return value

        return Step(name, fn)

    steps = [make("fetch", {"n": 1}), make("transform", [1, 2]), make("load", "done")]
    first = DurableStepsComponent(store).execute(WorkflowRequest("run-1", steps))
    assert first.status is RunStatus.COMPLETED
    assert first.executed == ["fetch", "transform", "load"]
    assert calls == ["fetch", "transform", "load"]

    second = DurableStepsComponent(store).execute(WorkflowRequest("run-1", steps))
    assert second.replayed == ["fetch", "transform", "load"]
    assert second.executed == []
    assert calls == ["fetch", "transform", "load"], "no step may run twice"
    assert second.context["transform"] == [1, 2]


def test_crash_resumes_at_the_failed_step_only():
    store = SqliteCheckpointStore()
    calls: list[str] = []
    state = {"explode": True}

    def ok(ctx):
        calls.append("ok")
        return "first"

    def flaky(ctx):
        calls.append("flaky")
        assert ctx["ok"] == "first", "context must carry earlier results"
        if state["explode"]:
            raise RuntimeError("upstream is down")
        return "second"

    steps = [
        Step("ok", ok),
        Step("flaky", flaky, retry=RetryPolicy(max_attempts=2, initial_backoff=0.0)),
    ]
    failed = DurableStepsComponent(store).execute(WorkflowRequest("run-2", steps))
    assert failed.status is RunStatus.FAILED
    assert failed.failed_step == "flaky"
    assert "upstream is down" in (failed.error or "")
    assert calls == ["ok", "flaky", "flaky"], "retried twice, ok ran once"

    state["explode"] = False
    resumed = DurableStepsComponent(store).execute(WorkflowRequest("run-2", steps))
    assert resumed.status is RunStatus.COMPLETED
    assert resumed.replayed == ["ok"], "the completed step must not re-run"
    assert resumed.executed == ["flaky"]
    assert calls.count("ok") == 1


def test_non_retryable_error_skips_the_retry_budget():
    store = SqliteCheckpointStore()
    attempts: list[int] = []

    def fatal(ctx):
        attempts.append(1)
        raise NonRetryableError("input is malformed")

    result = DurableStepsComponent(store).execute(
        WorkflowRequest(
            "run-3",
            [Step("validate", fatal, retry=RetryPolicy(max_attempts=5, initial_backoff=0.0))],
        )
    )
    assert result.status is RunStatus.FAILED
    assert len(attempts) == 1, "a non-retryable failure must not be retried"


def test_lease_blocks_a_second_worker_and_releases_on_completion():
    store = SqliteCheckpointStore()
    request = WorkflowRequest("run-4", [Step("noop", lambda ctx: 1)], lease_seconds=60)
    holder = DurableStepsComponent(store, owner="worker-a")
    assert store.acquire_lease("run-4", "worker-a", 60) is True
    assert store.acquire_lease("run-4", "worker-b", 60) is False

    try:
        DurableStepsComponent(store, owner="worker-b").execute(request)
    except LeaseNotAcquired:
        pass
    else:
        raise AssertionError("expected LeaseNotAcquired for a contending worker")

    assert holder.execute(request).status is RunStatus.COMPLETED
    # Released on completion, so any worker may now claim it.
    assert store.acquire_lease("run-4", "worker-b", 60) is True


def test_non_serialisable_result_fails_loudly():
    store = SqliteCheckpointStore()
    try:
        DurableStepsComponent(store).execute(
            WorkflowRequest("run-5", [Step("bad", lambda ctx: object())])
        )
    except TypeError as exc:
        assert "JSON" in str(exc)
    else:
        raise AssertionError("expected TypeError for a non-serialisable result")


def test_duplicate_step_names_rejected():
    try:
        WorkflowRequest("run-6", [Step("a", lambda c: 1), Step("a", lambda c: 2)])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for duplicate step names")


def test_sqlite_store_survives_a_new_process_object(tmp_path=None):
    import tempfile

    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "runs.db")
    store = SqliteCheckpointStore(path)
    calls: list[int] = []

    def work(ctx):
        calls.append(1)
        return "value"

    DurableStepsComponent(store).execute(WorkflowRequest("run-7", [Step("s", work)]))
    store.close()

    # A brand new store object over the same file is the durability claim.
    reopened = SqliteCheckpointStore(path)
    result = DurableStepsComponent(reopened).execute(
        WorkflowRequest("run-7", [Step("s", work)])
    )
    reopened.close()
    assert result.replayed == ["s"]
    assert calls == [1], "the step must not run again after a restart"


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
        except Exception as exc:  # noqa: BLE001 - this is the test runner
            failures += 1
            lines.append("FAIL " + name + ": " + repr(exc))
    lines.append("")
    lines.append(str(len(functions) - failures) + "/" + str(len(functions)) + " passed")
    sys.stdout.write("\n".join(lines) + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
