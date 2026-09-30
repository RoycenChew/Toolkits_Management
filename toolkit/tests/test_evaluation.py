"""Tests for the evaluation harness.

The metric tests use hand-computed expected values, because a metric that is only
checked against its own implementation is not checked at all.

Run standalone: python toolkit/tests/test_evaluation.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from toolkit.adapters import ScriptedLLM  # noqa: E402
from toolkit.core import BBox, Chunk, Provenance  # noqa: E402
from toolkit.evaluation import (  # noqa: E402
    EvalCase,
    EvalConfig,
    EvalDataset,
    EvalReport,
    EvalRunner,
    average_precision,
    diff_reports,
    hit_rate,
    judge_relevance,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from toolkit.pipelines import KnowledgeBase  # noqa: E402

_MANUAL = """# Safety Manual

## Voltage

The supply must not exceed 40V. Exceeding this limit voids the warranty and may
damage the controller board.

## Grounding

Bond the chassis to earth before energising the circuit. Use a conductor rated
for at least 16 amps.
"""

_FINANCE = """# Refund Policy

## Processing

Approved refunds are processed within ten business days to the original payment
method. Quote reference INV-88213 when contacting support.
"""


def _corpus() -> str:
    directory = tempfile.mkdtemp()
    for name, body in (("manual.md", _MANUAL), ("finance.md", _FINANCE)):
        with open(os.path.join(directory, name), "w", encoding="utf-8") as handle:
            handle.write(body)
    return directory


def _dataset() -> EvalDataset:
    return EvalDataset(
        name="sample",
        cases=[
            EvalCase("voltage", "what is the maximum supply voltage?",
                     expected_snippets=["must not exceed 40V"]),
            EvalCase("grounding", "how should the chassis be grounded?",
                     expected_snippets=["Bond the chassis to earth"]),
            EvalCase("refunds", "how long do refunds take?",
                     expected_snippets=["ten business days"]),
            EvalCase("invoice", "INV-88213",
                     expected_snippets=["INV-88213"]),
            EvalCase("offtopic", "what is the airspeed velocity of a swallow?",
                     unanswerable=True),
        ],
    )


# --------------------------------------------------------------------------
# Metrics - checked against hand-computed values
# --------------------------------------------------------------------------


def test_hit_rate_and_precision():
    assert hit_rate([0, 0, 1, 0]) == 1.0
    assert hit_rate([0, 0, 1, 0], k=2) == 0.0
    assert hit_rate([0, 0, 0]) == 0.0
    assert precision_at_k([1, 0, 1, 0], 4) == 0.5
    assert precision_at_k([1, 1, 0], 2) == 1.0
    assert precision_at_k([], 3) == 0.0


def test_reciprocal_rank_positions():
    assert reciprocal_rank([1, 0, 0]) == 1.0
    assert reciprocal_rank([0, 1, 0]) == 0.5
    assert abs(reciprocal_rank([0, 0, 1]) - 1.0 / 3.0) < 1e-12
    assert reciprocal_rank([0, 0, 0]) == 0.0
    assert reciprocal_rank([0, 1], k=1) == 0.0


def test_ndcg_matches_hand_computation():
    # relevance [0, 1]: DCG = 1/log2(3); ideal for 1 relevant item = 1/log2(2) = 1
    expected = (1.0 / math.log2(3)) / 1.0
    assert abs(ndcg_at_k([0, 1], 2) - expected) < 1e-12
    assert ndcg_at_k([1, 0], 2) == 1.0
    assert ndcg_at_k([0, 0], 2) == 0.0
    # With total_relevant known, retrieving 1 of 2 must not score a perfect 1.0.
    assert ndcg_at_k([1, 0], 2, total_relevant=2) < 1.0


def test_recall_falls_back_honestly_when_the_denominator_is_unknown():
    """Calling hit_rate 'recall' would overstate the system, so the fallback is
    explicit."""
    assert recall_at_k([1, 0], 2) == hit_rate([1, 0], 2)
    assert recall_at_k([1, 0, 1], 3, total_relevant=4) == 0.5
    assert recall_at_k([1, 1], 2, total_relevant=1) == 1.0, "must clamp at 1.0"
    assert recall_at_k([1], 1, total_relevant=0) == 0.0


def test_average_precision():
    # Relevant at ranks 1 and 3: (1/1 + 2/3) / 2
    assert abs(average_precision([1, 0, 1]) - (1.0 + 2.0 / 3.0) / 2.0) < 1e-12
    assert average_precision([0, 0]) == 0.0


# --------------------------------------------------------------------------
# Relevance judgement
# --------------------------------------------------------------------------


def _chunk(text: str, doc_id: str = "doc:a", page: int = 1) -> Chunk:
    return Chunk(
        chunk_id="c1",
        text=text,
        doc_id=doc_id,
        index=0,
        provenances=[Provenance(page=page, bbox=BBox(0, 0, 10, 10))],
    )


def test_relevance_is_judged_on_content_not_chunk_id():
    """The design decision that keeps a golden set valid across re-chunking."""
    case = EvalCase("c", "q", expected_snippets=["must not exceed 40V"])
    assert judge_relevance(_chunk("The supply must not exceed 40V today."), case)
    assert not judge_relevance(_chunk("Something else entirely."), case)


def test_snippet_matching_is_whitespace_and_case_insensitive():
    case = EvalCase("c", "q", expected_snippets=["MUST   not\nexceed 40v"])
    assert judge_relevance(_chunk("the supply must not exceed 40V."), case)


def test_page_and_doc_expectations():
    by_page = EvalCase("c", "q", expected_pages=[7])
    assert judge_relevance(_chunk("anything", page=7), by_page)
    assert not judge_relevance(_chunk("anything", page=8), by_page)

    by_doc = EvalCase("c", "q", expected_doc_ids=["doc:a"])
    assert judge_relevance(_chunk("anything", doc_id="doc:a"), by_doc)
    assert not judge_relevance(_chunk("anything", doc_id="doc:b"), by_doc)

    both = EvalCase("c", "q", expected_doc_ids=["doc:a"], expected_pages=[3])
    assert judge_relevance(_chunk("x", doc_id="doc:a", page=3), both)
    assert not judge_relevance(_chunk("x", doc_id="doc:a", page=4), both)


def test_answerable_case_without_expectations_is_rejected():
    try:
        EvalCase("bad", "a question with no expectation")
    except ValueError as exc:
        assert "no expectation" in str(exc)
    else:
        raise AssertionError("expected ValueError")
    # Marked unanswerable, the same case is valid.
    assert EvalCase("ok", "a question", unanswerable=True)


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def test_runner_produces_metrics_and_per_case_results():
    kb = KnowledgeBase()
    kb.ingest_folder(_corpus())
    report = EvalRunner(kb).execute(_dataset(), EvalConfig(top_k=3))

    assert report.metrics["cases"] == 5
    assert report.metrics["unanswerable_cases"] == 1
    assert 0.0 <= report.metrics["hit_rate@3"] <= 1.0
    assert report.metrics["hit_rate@3"] > 0.5, report.metrics
    assert report.metrics["refusal_accuracy"] == 1.0, "the off-topic case must refuse"
    assert report.metrics["hallucinated_citation"] == 0.0
    assert len(report.cases) == 5
    assert report.config["embedder"] == "HashingEmbedder"
    assert report.created_at


def test_answerable_and_unanswerable_are_scored_separately():
    """Mixing them hides the trade: loosening the gate raises hit rate and lowers
    refusal accuracy at the same time."""
    kb = KnowledgeBase()
    kb.ingest_folder(_corpus())
    report = EvalRunner(kb).execute(_dataset(), EvalConfig(top_k=3))
    offtopic = [c for c in report.cases if c.case_id == "offtopic"][0]
    assert offtopic.refused and offtopic.correct
    # The refusal must not be counted as a retrieval miss on the answerable set.
    assert report.metrics["false_refusal"] < 1.0


def test_answer_substring_checking_and_citation_relevance():
    kb = KnowledgeBase(llm=ScriptedLLM(handler=lambda m: "The limit is 40V [1]."))
    kb.ingest_folder(_corpus())
    dataset = EvalDataset(
        "answers",
        [
            EvalCase("v", "voltage limit",
                     expected_snippets=["must not exceed 40V"],
                     expected_answer_contains=["40V"]),
            EvalCase("miss", "voltage limit",
                     expected_snippets=["must not exceed 40V"],
                     expected_answer_contains=["60V"]),
        ],
    )
    report = EvalRunner(kb).execute(dataset, EvalConfig(top_k=3))
    by_id = {c.case_id: c for c in report.cases}
    assert by_id["v"].answer_hits == ["40V"] and by_id["v"].answer_misses == []
    assert by_id["v"].correct
    assert by_id["miss"].answer_misses == ["60V"]
    assert not by_id["miss"].correct, "a missing expected string fails the case"
    assert report.metrics["answer_match"] == 0.5
    assert report.metrics["answer_cases"] == 2.0
    assert by_id["v"].cited_relevant, "the cited chunk was the relevant one"


def test_answer_match_is_absent_when_nothing_was_checked():
    """Reporting 0.000 for 'not measured' reads as 'failed everything'."""
    kb = KnowledgeBase()
    kb.ingest_folder(_corpus())
    report = EvalRunner(kb).execute(_dataset(), EvalConfig(top_k=3))
    assert "answer_match" not in report.metrics
    assert "answer_cases" not in report.metrics


def test_hallucinated_citations_are_measured():
    kb = KnowledgeBase(llm=ScriptedLLM(handler=lambda m: "It is 40V [9]."))
    kb.ingest_folder(_corpus())
    dataset = EvalDataset(
        "hallucination",
        [EvalCase("v", "voltage limit", expected_snippets=["must not exceed 40V"])],
    )
    report = EvalRunner(kb).execute(dataset, EvalConfig(top_k=2))
    assert report.metrics["hallucinated_citation"] == 1.0
    assert report.cases[0].unverified_markers == [9]


# --------------------------------------------------------------------------
# Regression diff - the point of the whole harness
# --------------------------------------------------------------------------


def test_diff_names_the_cases_a_change_broke():
    kb = KnowledgeBase()
    kb.ingest_folder(_corpus())
    dataset = _dataset()
    runner = EvalRunner(kb)

    before = runner.execute(dataset, EvalConfig(top_k=3))

    # Disabling the relevance gate is the classic trade: the off-topic case stops
    # refusing. The diff must name it rather than just moving an average.
    after = runner.execute(dataset, EvalConfig(top_k=3, relevance_gate=False))

    diff = diff_reports(before, after)
    assert "offtopic" in diff.broken, diff.render()
    assert diff.net < 0
    rendered = diff.render()
    assert "refusal_accuracy" in rendered
    assert "broken:" in rendered


def test_diff_reports_config_changes():
    kb = KnowledgeBase()
    kb.ingest_folder(_corpus())
    dataset = _dataset()
    before = EvalRunner(kb).execute(dataset, EvalConfig(top_k=3))
    after = EvalRunner(kb).execute(dataset, EvalConfig(top_k=5))
    diff = diff_reports(before, after)
    assert "top_k: 3 -> 5" in diff.render()


def test_diff_only_compares_metrics_present_in_both():
    kb = KnowledgeBase()
    kb.ingest_folder(_corpus())
    dataset = _dataset()
    before = EvalRunner(kb).execute(dataset, EvalConfig(k_values=(1, 3)))
    after = EvalRunner(kb).execute(dataset, EvalConfig(k_values=(1, 3, 10)))
    names = {d.name for d in diff_reports(before, after).deltas}
    assert "hit_rate@10" not in names, "a new cut-off must not fabricate a delta"
    assert "hit_rate@1" in names


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def test_report_round_trips_through_json():
    kb = KnowledgeBase()
    kb.ingest_folder(_corpus())
    report = EvalRunner(kb).execute(_dataset(), EvalConfig(top_k=3))

    path = os.path.join(tempfile.mkdtemp(), "report.json")
    report.to_json(path)
    loaded = EvalReport.from_json(path)
    assert loaded.dataset == report.dataset
    assert loaded.metrics == report.metrics
    assert [c.case_id for c in loaded.cases] == [c.case_id for c in report.cases]
    # A reloaded report must diff cleanly against its own source.
    assert diff_reports(report, loaded).net == 0


def test_dataset_round_trips_through_jsonl():
    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "golden.jsonl")
    _dataset().to_jsonl(path)
    loaded = EvalDataset.from_jsonl(path, name="sample")
    assert [c.case_id for c in loaded.cases] == [c.case_id for c in _dataset().cases]
    assert loaded.cases[0].expected_snippets == ["must not exceed 40V"]


def test_jsonl_tolerates_comments_and_reports_bad_lines():
    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "golden.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# a comment line\n\n")
        handle.write(json.dumps({"query": "q", "expected_snippets": ["x"]}) + "\n")
    loaded = EvalDataset.from_jsonl(path)
    assert len(loaded.cases) == 1
    assert loaded.cases[0].case_id == "case-3", "ids default to the line number"

    broken = os.path.join(directory, "broken.jsonl")
    with open(broken, "w", encoding="utf-8") as handle:
        handle.write("{not json}\n")
    try:
        EvalDataset.from_jsonl(broken)
    except ValueError as exc:
        assert "line 1" in str(exc)
    else:
        raise AssertionError("expected ValueError naming the line")


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
