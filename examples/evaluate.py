"""Answer 'did that change help?' with a table instead of a vibe.

Runs the same golden set against two configurations and diffs them. Offline, no
API key.

    python examples/evaluate.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quickstart import sample_corpus  # noqa: E402

from toolkit.chunking import ChunkConfig  # noqa: E402
from toolkit.evaluation import (  # noqa: E402
    EvalCase,
    EvalConfig,
    EvalDataset,
    EvalRunner,
    diff_reports,
)
from toolkit.pipelines import KnowledgeBase  # noqa: E402

GOLDEN = EvalDataset(
    name="sample-corpus",
    cases=[
        EvalCase(
            "voltage",
            "what is the maximum supply voltage?",
            expected_snippets=["must not exceed 40V"],
        ),
        EvalCase(
            "grounding",
            "how should the chassis be grounded?",
            expected_snippets=["Bond the chassis to earth"],
        ),
        EvalCase(
            "conductor",
            "what conductor rating is required?",
            expected_snippets=["rated for at least 16 amps"],
        ),
        EvalCase(
            "refund-window",
            "how many days do I have to return goods?",
            expected_snippets=["within 30 days of delivery"],
        ),
        EvalCase(
            "refund-time",
            "how long does a refund take to process?",
            expected_snippets=["ten business days"],
        ),
        EvalCase("invoice", "INV-88213", expected_snippets=["INV-88213"]),
        EvalCase(
            "offtopic",
            "what is the airspeed velocity of an unladen swallow?",
            unanswerable=True,
        ),
        EvalCase("offtopic2", "who won the 1998 world cup?", unanswerable=True),
    ],
)


CORPUS = sample_corpus()


def build(**kwargs) -> KnowledgeBase:
    kb = KnowledgeBase(**kwargs)
    kb.ingest_folder(CORPUS)
    return kb


def experiment(title: str, before_kb, before_cfg, after_kb, after_cfg) -> None:
    out = sys.stdout.write
    before = EvalRunner(before_kb).execute(GOLDEN, before_cfg)
    after = EvalRunner(after_kb).execute(GOLDEN, after_cfg)
    diff = diff_reports(before, after)
    out("\n" + "=" * 62 + "\n" + title + "\n" + "=" * 62 + "\n")
    # Only the metrics that moved, plus the headline ones. A wall of +0.000 rows
    # hides the two numbers that changed.
    interesting = [
        d
        for d in diff.deltas
        if abs(d.change) > 1e-9
        or d.name in ("hit_rate@1", "mrr", "refusal_accuracy", "overall_correct")
    ]
    out("metric                     before    after     change\n")
    for d in interesting:
        out(
            d.name.ljust(26)
            + format(d.before, ".3f").rjust(7)
            + format(d.after, ".3f").rjust(10)
            + format(d.change, "+.3f").rjust(11)
            + "\n"
        )
    out(
        "\ncases fixed: "
        + str(len(diff.fixed))
        + "  broken: "
        + str(len(diff.broken))
        + "  unchanged: "
        + str(diff.unchanged)
        + "\n"
    )
    if diff.broken:
        out("broken: " + ", ".join(diff.broken) + "\n")
    if diff.fixed:
        out("fixed:  " + ", ".join(diff.fixed) + "\n")
    if not diff.fixed and not diff.broken and before.metrics["overall_correct"] >= 1.0:
        out(
            "note: the baseline already scores 1.000, so this corpus cannot tell\n"
            "      these two settings apart. That is a fact about the golden set,\n"
            "      not evidence that the change is safe. Add harder cases.\n"
        )


def main() -> int:
    base = EvalConfig(k_values=(1, 3, 5), top_k=5)
    hybrid = build()

    experiment(
        "Does the relevance gate earn its keep?",
        hybrid,
        base,
        hybrid,
        EvalConfig(k_values=(1, 3, 5), top_k=5, relevance_gate=False),
    )

    experiment(
        "Does the keyword index earn its keep?",
        build(lexical_index=None),
        base,
        hybrid,
        base,
    )

    experiment(
        "Does a smaller chunk size help?",
        build(chunk_config=ChunkConfig(max_tokens=512)),
        base,
        build(chunk_config=ChunkConfig(max_tokens=96)),
        base,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
