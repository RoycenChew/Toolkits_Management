"""Run a golden set against a KnowledgeBase and diff runs against each other.

The harness, not the metrics, is the part worth owning. Ragas and DeepEval have
good metrics; what they cannot have is your golden set, your relevance
judgement, and a regression diff that tells you *which of your questions* a
change broke. That last part is what turns "retrieval feels better" into a
decision.
"""
from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from ..core.models import Chunk
from ..pipelines.models import Answer, AskConfig
from . import metrics as m
from .models import (
    CaseResult,
    EvalCase,
    EvalConfig,
    EvalDataset,
    EvalReport,
    MetricDelta,
    RegressionDiff,
)


def judge_relevance(chunk: Chunk, case: EvalCase) -> bool:
    """Is this retrieved chunk relevant to this case?

    Deliberately expressed over chunk *content and location*, never chunk id, so
    a golden set stays valid when the chunking strategy changes. Comparison is
    case-insensitive and whitespace-normalised, because a snippet copied out of a
    PDF rarely matches the extracted text byte for byte.
    """
    if case.expected_snippets:
        haystack = " ".join(chunk.text.lower().split())
        for snippet in case.expected_snippets:
            needle = " ".join(snippet.lower().split())
            if needle and needle in haystack:
                return True
    if case.expected_doc_ids and chunk.doc_id in case.expected_doc_ids:
        # A doc-level expectation combined with pages means both must hold;
        # alone, the document is the whole claim.
        if not case.expected_pages:
            return True
        if any(page in case.expected_pages for page in chunk.pages):
            return True
    elif case.expected_pages and not case.expected_doc_ids:
        if any(page in case.expected_pages for page in chunk.pages):
            return True
    return False


class EvalRunner:
    """Executes a dataset against a `KnowledgeBase` and reports."""

    def __init__(self, knowledge_base: Any) -> None:
        self._kb = knowledge_base

    def execute(
        self, dataset: EvalDataset, config: EvalConfig | None = None
    ) -> EvalReport:
        cfg = config or EvalConfig()
        ask_config = AskConfig(
            top_k=cfg.top_k,
            candidates_per_retriever=cfg.candidates_per_retriever,
            relevance_gate=cfg.relevance_gate,
            min_dense_similarity=cfg.min_dense_similarity,
            lexical_weight=cfg.lexical_weight,
            dense_weight=cfg.dense_weight,
            rrf_k=cfg.rrf_k,
            rerank_budget=cfg.rerank_budget,
        )
        resolvable = self._resolvable(dataset)
        results = [
            self._run_case(case, ask_config, cfg, resolvable) for case in dataset.cases
        ]
        return EvalReport(
            dataset=dataset.name,
            metrics=self._aggregate(results, dataset.cases, cfg),
            cases=results,
            config=self._snapshot(cfg, ask_config),
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    def _resolvable(self, dataset: EvalDataset) -> dict[str, bool]:
        """Which cases have ground truth that some indexed chunk satisfies?

        Needs to enumerate the corpus, which `KnowledgeBase.chunks()` provides.
        When the knowledge base does not expose it - any duck-typed stand-in -
        every case is assumed resolvable, which is the previous behaviour and
        errs towards reporting a retrieval miss rather than silently dropping a
        case from the denominator.
        """
        enumerate_chunks = getattr(self._kb, "chunks", None)
        if not callable(enumerate_chunks):
            return {}
        try:
            corpus = list(enumerate_chunks())
        except Exception:  # noqa: BLE001 - a stand-in that raises is not a failure here
            return {}
        if not corpus:
            return {}
        out: dict[str, bool] = {}
        for case in dataset.cases:
            if case.unanswerable:
                continue
            if not (
                case.expected_snippets or case.expected_doc_ids or case.expected_pages
            ):
                continue
            out[case.case_id] = any(judge_relevance(c, case) for c in corpus)
        return out

    def _run_case(
        self,
        case: EvalCase,
        ask_config: AskConfig,
        cfg: EvalConfig,
        resolvable: dict[str, bool] | None = None,
    ) -> CaseResult:
        started = time.monotonic()
        answer: Answer = self._kb.ask(case.query, ask_config)
        elapsed = time.monotonic() - started

        relevance = [1 if judge_relevance(c, case) else 0 for c in answer.chunks]
        refused = not answer.grounded and not answer.citations

        hits: list[str] = []
        misses: list[str] = []
        if cfg.evaluate_answers and case.expected_answer_contains:
            lowered = " ".join(answer.text.lower().split())
            for expected in case.expected_answer_contains:
                needle = " ".join(expected.lower().split())
                (hits if needle in lowered else misses).append(expected)

        cited_relevant = False
        if answer.citations:
            by_id = {c.chunk_id: index for index, c in enumerate(answer.chunks)}
            cited_relevant = any(
                relevance[by_id[citation.chunk_id]]
                for citation in answer.citations
                if citation.chunk_id in by_id
            )

        return CaseResult(
            case_id=case.case_id,
            query=case.query,
            retrieved_ids=[c.chunk_id for c in answer.chunks],
            relevance=relevance,
            unanswerable=case.unanswerable,
            refused=refused,
            grounded=answer.grounded,
            answer=answer.text,
            answer_hits=hits,
            answer_misses=misses,
            unverified_markers=list(answer.unverified_markers),
            ground_truth_missing=(resolvable or {}).get(case.case_id, True) is False,
            cited_relevant=cited_relevant,
            seconds=elapsed,
        )

    def _aggregate(
        self,
        results: Sequence[CaseResult],
        cases: Sequence[EvalCase],
        cfg: EvalConfig,
    ) -> dict[str, float]:
        """Answerable and unanswerable cases are scored separately.

        Mixing them hides the trade every retrieval system makes: loosening the
        relevance gate raises hit rate and lowers refusal accuracy at the same
        time. Averaged together those cancel out and the report says nothing
        changed.
        """
        unanswerable = [r for r in results if r.unanswerable]
        # A case whose expected snippet exists in NO indexed chunk cannot be
        # retrieved by anything, so scoring it as a retrieval miss blames the
        # ranker for a fault upstream of it. Real use made this concrete: this
        # harness reported hit_rate@10 of 0.42 on a corpus where direct
        # measurement over the resolvable cases gave 0.95, because extraction
        # had mangled the text the snippets were written against. Someone
        # trusting the number would have gone and tuned the ranker.
        #
        # Such cases are reported as `dataset_errors` and excluded from the
        # retrieval metrics. They are still counted in `cases`, and still
        # scored for refusal, because refusing is the correct response to a
        # question the corpus cannot answer.
        answerable = [
            r for r in results if not r.unanswerable and not r.ground_truth_missing
        ]
        unresolvable = [
            r for r in results if not r.unanswerable and r.ground_truth_missing
        ]
        out: dict[str, float] = {}

        for k in cfg.k_values:
            out["hit_rate@" + str(k)] = m.mean(
                [m.hit_rate(r.relevance, k) for r in answerable]
            )
            out["precision@" + str(k)] = m.mean(
                [m.precision_at_k(r.relevance, k) for r in answerable]
            )
            out["ndcg@" + str(k)] = m.mean(
                [m.ndcg_at_k(r.relevance, k) for r in answerable]
            )

        out["mrr"] = m.mean([m.reciprocal_rank(r.relevance) for r in answerable])
        out["map"] = m.mean([m.average_precision(r.relevance) for r in answerable])
        out["cited_relevant"] = m.mean(
            [1.0 if r.cited_relevant else 0.0 for r in answerable]
        )
        out["false_refusal"] = m.mean([1.0 if r.refused else 0.0 for r in answerable])
        out["refusal_accuracy"] = m.mean(
            [1.0 if r.refused else 0.0 for r in unanswerable]
        )
        out["hallucinated_citation"] = m.mean(
            [1.0 if r.unverified_markers else 0.0 for r in results]
        )
        # Only report answer_match when cases actually declared expected strings.
        # Emitting 0.000 for "not measured" is indistinguishable from "measured
        # and failed everything", and a metric that can be read as a catastrophe
        # when nothing was checked is worse than an absent one.
        scored = [r for r in answerable if r.answer_hits or r.answer_misses]
        if cfg.evaluate_answers and scored:
            out["answer_match"] = m.mean(
                [1.0 if not r.answer_misses else 0.0 for r in scored]
            )
            out["answer_cases"] = float(len(scored))
        out["overall_correct"] = m.mean([1.0 if r.correct else 0.0 for r in results])
        out["mean_seconds"] = m.mean([r.seconds for r in results])
        out["cases"] = float(len(results))
        out["unanswerable_cases"] = float(len(unanswerable))
        out["scored_cases"] = float(len(answerable))
        out["dataset_errors"] = float(len(unresolvable))
        return out

    def _snapshot(self, cfg: EvalConfig, ask: AskConfig) -> dict[str, Any]:
        kb = self._kb
        return {
            "top_k": ask.top_k,
            "candidates_per_retriever": ask.candidates_per_retriever,
            "rrf_k": ask.rrf_k,
            "relevance_gate": ask.relevance_gate,
            "min_dense_similarity": ask.min_dense_similarity,
            "min_term_coverage": ask.min_term_coverage,
            # Recorded so `diff_reports` can attribute a metric change to the
            # ablation that caused it. An ablation whose settings are not in the
            # snapshot is indistinguishable from a regression.
            "lexical_weight": ask.lexical_weight,
            "dense_weight": ask.dense_weight,
            "rerank_budget": ask.rerank_budget,
            "evaluate_answers": cfg.evaluate_answers,
            "embedder": type(getattr(kb, "embedder", None)).__name__,
            "vector_store": type(getattr(kb, "vector_store", None)).__name__,
            "lexical_index": type(getattr(kb, "lexical_index", None)).__name__,
            "reranker": type(getattr(kb, "reranker", None)).__name__,
            "llm": type(getattr(kb, "llm", None)).__name__,
            "chunk_max_tokens": getattr(
                getattr(kb, "chunk_config", None), "max_tokens", None
            ),
        }


def diff_reports(before: EvalReport, after: EvalReport) -> RegressionDiff:
    """Compare two runs at the aggregate and per-case level.

    Only metrics present in both are compared, so adding a new k-value to the
    config does not fabricate a delta against a run that never measured it.
    """
    shared = [key for key in after.metrics if key in before.metrics]
    deltas = [
        MetricDelta(name=key, before=float(before.metrics[key]), after=float(after.metrics[key]))
        for key in sorted(shared)
    ]

    before_by_id = {c.case_id: c for c in before.cases}
    fixed: list[str] = []
    broken: list[str] = []
    unchanged = 0
    for case in after.cases:
        previous = before_by_id.get(case.case_id)
        if previous is None:
            continue
        if case.correct and not previous.correct:
            fixed.append(case.case_id)
        elif previous.correct and not case.correct:
            broken.append(case.case_id)
        else:
            unchanged += 1

    return RegressionDiff(
        deltas=deltas,
        fixed=sorted(fixed),
        broken=sorted(broken),
        unchanged=unchanged,
        before_config=dict(before.config),
        after_config=dict(after.config),
    )


__all__ = ["EvalRunner", "diff_reports", "judge_relevance"]
