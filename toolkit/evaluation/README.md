# Evaluation Component

## What It Does

Runs a golden set against a `KnowledgeBase`, reports retrieval and answer metrics,
and **diffs two runs to name the cases a change broke**.

## Why It Is Useful

This is what separates a toolkit from a pile of code. Without it, every retrieval
decision is argued from vibes: "that feels better." With it, changing `rrf_k` prints
a table and a list of case ids.

Ragas and DeepEval have good metrics — use them. What they cannot have is *your*
golden set, *your* relevance judgement, and a regression diff over *your* questions.
The harness is the part worth owning; the metrics here are small enough to own too,
and being able to read their definitions in twenty seconds beats trusting a
dependency you argue with.

## The decision that makes a golden set survive

**Ground truth is never expressed as chunk ids.** The obvious design — list the
`chunk_id`s that should be retrieved — invalidates itself the moment it becomes
useful, because chunk ids change whenever chunking changes, and re-tuning chunking is
exactly the experiment you most want to run.

So relevance is expressed in terms that survive re-chunking:

- `expected_snippets` — text that must appear in a relevant passage. Easiest to author
  (copy it out of the document) and stable under any chunking strategy.
- `expected_pages` / `expected_doc_ids` — for tables and figures you would rather not
  transcribe.

Matching is case-insensitive and whitespace-normalised, because a snippet copied from
a PDF rarely matches extracted text byte for byte.

## Two more decisions worth knowing

**Unanswerable cases are first-class.** Mark a case `unanswerable=True` and a correct
system *refuses* it. Without these you cannot measure the relevance gate at all, and a
system that answers everything scores perfectly on a set of answerable questions.

**Answerable and unanswerable cases are aggregated separately.** Mixing them hides the
trade every retrieval system makes: loosening the gate raises hit rate and lowers
refusal accuracy simultaneously, and averaged together those cancel to "nothing
changed".

## Architecture

```
EvalDataset (JSONL)  ->  EvalRunner  ->  EvalReport  ->  diff_reports  ->  RegressionDiff
                              |                                              |
                        per case:                                    metric deltas
                        ask -> judge -> metrics                      cases fixed / broken
                                                                     config changes
```

## Installation

```bash
pip install -e .
```

Python 3.10+, standard library only. This unit is **not** copy-one-folder: the
runner imports `core` and `pipelines`, so install the package rather than copying
the directory. The metrics in `metrics.py` are the exception — that module imports
nothing and can be lifted on its own.

## Dependencies

Standard library.

## Input Schema

`EvalCase`: `case_id`, `query`, `expected_snippets`, `expected_doc_ids`,
`expected_pages`, `expected_answer_contains`, `unanswerable`, `tags`.

An answerable case with no expectation at all is rejected at construction — a case
that cannot fail is not a test.

Golden sets load from JSONL (`#` comments and blank lines allowed; `case_id` defaults
to the line number):

```jsonl
{"case_id": "voltage", "query": "max supply voltage?", "expected_snippets": ["must not exceed 40V"]}
{"case_id": "offtopic", "query": "who won the 1998 world cup?", "unanswerable": true}
```

`EvalConfig`: `k_values`, `top_k`, `candidates_per_retriever`, `evaluate_answers`,
`relevance_gate`, `min_dense_similarity`. The last two are surfaced here so a sweep
can vary them and the diff records what moved.

## Output Schema

`EvalReport.metrics` — `hit_rate@k`, `precision@k`, `ndcg@k`, `mrr`, `map`,
`cited_relevant`, `false_refusal`, `refusal_accuracy`, `hallucinated_citation`,
`overall_correct`, `mean_seconds`, plus `answer_match` **only when cases actually
declared expected strings**. Reporting 0.000 for "not measured" is indistinguishable
from "failed everything", so that key is absent instead.

`CaseResult` carries the retrieved ids, the binary relevance vector, the answer, hits
and misses, unverified markers, and a single `correct` verdict used by the diff.

`RegressionDiff.render()` prints the metric table, the fixed/broken case ids, and any
config keys that differ between runs.

## Usage

```python
from toolkit.evaluation import EvalConfig, EvalDataset, EvalRunner, diff_reports
from toolkit.pipelines import KnowledgeBase

golden = EvalDataset.from_jsonl("golden.jsonl")
kb = KnowledgeBase(); kb.ingest_folder("./docs")
runner = EvalRunner(kb)

baseline = runner.execute(golden, EvalConfig(top_k=5))
baseline.to_json("baseline.json")

candidate = runner.execute(golden, EvalConfig(top_k=5, relevance_gate=False))
print(diff_reports(baseline, candidate).render())
```

`python examples/evaluate.py` runs three such comparisons on the sample corpus.

## Limitations

- **Answer scoring is substring matching**, not semantic. It catches the regressions
  that matter and needs no judge model, but it will not notice a correct answer phrased
  differently. Bring Ragas or DeepEval when you need faithfulness and semantic scoring.
- **`recall@k` needs a denominator you usually do not have.** With `total_relevant`
  unknown it falls back to `hit_rate` and says so, because a recall figure with an
  invented denominator overstates the system.
- Relevance is binary. No graded judgements.
- **A saturated benchmark proves nothing.** If the baseline already scores 1.000, a
  diff of all zeros is a fact about your golden set, not evidence the change is safe.
  The example script prints this warning explicitly when it detects saturation — the
  sample corpus triggers it on two of three experiments.
- Cases run sequentially. A large golden set against a hosted model is slow.
- `mean_seconds` measures wall clock including the model call; it is a smell detector,
  not a benchmark.
- No statistical significance testing. On twenty cases, a 0.05 move is noise.

## Integration Guide

1. **Write 20 cases before tuning anything.** Ten answerable, ten unanswerable. This
   takes an hour and it is the hour that makes every later decision cheap.
2. Author `expected_snippets` by copying from the source document. Never chunk ids.
3. Commit `baseline.json` next to the golden set. A diff needs something to diff.
4. Re-run before and after every retrieval change, and read the **broken** list first.
5. Watch `refusal_accuracy` alongside `hit_rate`. A change that improves one while
   destroying the other is usually a bad trade, and only the split reporting shows it.
6. If everything scores 1.000, your cases are too easy. Add the questions your users
   actually ask badly.

## Extraction Notes

- **Preserved:** the standard IR metric definitions (MRR, nDCG, MAP, precision@k).
- **Added:** chunk-id-free relevance judgement, first-class unanswerable cases,
  split aggregation, hallucinated-citation measurement, the config snapshot in every
  report, and the per-case regression diff.
- **Isolated:** no dependency on any eval framework, and the runner only needs an
  object with `.ask(query, config)` — so it evaluates anything shaped like the
  pipeline, not just this one.
