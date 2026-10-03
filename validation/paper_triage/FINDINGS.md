# Validation findings — paper triage over a real arXiv corpus

**Run date:** 2026-10-03 · **Corpus:** 49 arXiv PDFs, 212 MB, 9 fields, published
1986–2026 · **Cost:** zero (no LLM; `llm=None` gives extractive answers)

This is the first time any component in this repository met input that nobody
here authored. Until now every adversarial test, including `stress/`, ran
against a generator written to match failure modes already predicted — so it
could not find a failure mode nobody had thought of.

It found several. The first one alone justifies the exercise.

---

## F1 — CRITICAL — `adapters`: word spaces are destroyed in every document

`PdfPlumberSource` calls `page.extract_words(...)` without `x_tolerance`, taking
pdfplumber's default of **3 points**. LaTeX's Computer Modern fonts set
inter-word spaces narrower than that, so pdfplumber merges whole lines into one
"word".

Measured over all 49 documents:

| Metric | Value |
|---|---|
| Documents with at least one glued run (>24 letters) | **49 / 49** |
| Median share of characters inside glued runs | **22.9%** |
| Worst document (`2610.01463v1`) | **84.2%** |
| Worst-affected cohort | modern LaTeX (2026 papers, 60–84%) |
| Least affected | 1991 `hep-th` papers (~1%) |

Observed text:

```
Supersingularabeliansurfacesareessentialinisogeny-based   (size 9.0, OINENH+CMR9)
Wederivethisalgorithmbyacarefulanalysisonthestructureofsuper-
SincethespectacularbreakofSIDH/SIKE[10,25,33],higher-dimensionaltech-
```

**Verified fix** — on page 1 of `2610.01924v1`:

| `x_tolerance` | words found | glued runs (>24 ch) |
|---|---|---|
| 3 (current default) | 412 | **5** |
| 1.5 | 412 | **0** |
| 1.0 | 412 | **0** |

At `x_tolerance=1.5` the word count is unchanged and the glued runs vanish; the
golden snippet `"supersingular abelian surfaces are essential in isogeny-based
cryptography"` matches exactly.

**Why this matters more than the percentage suggests.** Glued text cannot be
matched by BM25 at all — the tokens do not exist — and embedding a 60-character
run yields a degraded vector. The damage concentrates in abstracts, titles and
affiliations, which are the highest-value regions of a paper. This is a
retrieval-quality defect disguised as an extraction detail.

**Why 273 tests missed it.** `stress/make_corpus.py` writes PDFs by placing
text with generous explicit spacing. Real LaTeX places glyphs individually with
kerning and leaves the extractor to infer spaces from x-gaps. No synthetic
corpus written by the same person who wrote the parser will probe that
inference.

---

## F2 — CRITICAL — `pipelines` + `durable_steps`: resuming yields a silently empty index

With `IngestConfig(durable_db=...)`, re-running ingest in a **new process**
reported:

```
documents      : 49
chunks indexed : 0  (kb.count=0)
failures       : 0
status counts  : {'replayed': 49}
```

49 documents "replayed", zero failures, **and an empty index**. Subsequent
`ask()` calls then refuse every question — correct behaviour on an empty index,
which is exactly what hides the problem: the user concludes the corpus has no
answer rather than that nothing was indexed.

The assumption is stated in `_ingest_one` itself:

> *"Replayed from a previous run: the work did not happen again, and the store
> was populated then."*

That holds only if the stores outlive the process. `KnowledgeBase._chunks` is
**always** a process-local dict with no persistent option, and the default
stores are in-memory. So checkpoint replay restores a step's *return value* but
never its *side effects*.

**Why the tests missed it.** `test_completed_steps_are_replayed_not_rerun`,
`test_crash_resumes_at_the_failed_step_only` and
`test_ingest_is_idempotent_on_re_run` all pass, because each creates the
KnowledgeBase and the checkpoint DB together in one process — where `_chunks`
is still populated. The defect needs the checkpoint to outlive the process,
which is the entire point of durability.

**Operational consequence during this very run:** every re-measurement required
deleting `results/ingest.db` first.

---

## F3 — MAJOR — `chunking`: the heuristic token counter under-counts by 15–74%

Against `tiktoken` `cl100k_base`, with `ChunkConfig(max_tokens=512)`:

| Metric | Value |
|---|---|
| Chunks over the 512-token budget | **1166 / 2626 (44.4%)** |
| Worst overshoot | **2.65× budget** |
| real/estimated ratio, median | 1.15 |
| real/estimated ratio, p95 | 1.74 |

The heuristic counter is a documented limitation. This is the first measurement
of its *cost* on scientific prose: nearly half of all chunks would overflow a
512-token context window, and the worst would need 1357 tokens. Any caller
sizing a prompt from `estimate_tokens` is exposed.

---

## F4 — MAJOR — real documents contain tokenizer special-token literals

28 of 2626 chunks contain a literal `<|...|>` sequence such as `<|endoftext|>`.
`tiktoken.encode()` refuses these by default and raises `ValueError`, which
crashed this harness mid-run and cost the `eval` and `ablation` steps on the
first attempt.

Any component that counts tokens with a real tokenizer must pass
`disallowed_special=()`. This is also a prompt-injection-adjacent surface: a
document can carry a control token into a prompt.

---

## F5 — MAJOR — `evaluation` scores a broken golden set as a retrieval failure

The report read `hit_rate@10 = 0.4167`, with 16 of 26 cases "retrieving nothing
relevant". Measured directly against the retrievers on the cases whose ground
truth actually resolves, recall@10 was **0.95**.

The gap is F1: 13 of 24 answerable snippets appear in **no** indexed chunk,
mostly because the text was glued. `EvalRunner` cannot distinguish

- *no chunk in the corpus contains the expected snippet* (dataset or extraction
  problem), from
- *the relevant chunk exists and retrieval missed it* (a retrieval problem).

Both score zero. A harness whose purpose is to detect regressions reported a
retrieval failure when the real fault was upstream, and would have sent anyone
trusting it to optimise the ranker.

Unmatched `expected_snippets` should be reported as a dataset error and excluded
from the denominator, not counted as misses.

---

## F6 — MAJOR — the relevance gate does not refuse unanswerable questions

`refusal_accuracy = 0.0000`, `false_refusal = 0.0000`: both pre-registered
unanswerable queries ("melting point of gallium nitride", "configure a
Kubernetes ingress controller with TLS") were **answered**, with citations to
real but irrelevant chunks. `hallucinated_citation = 0.0` — the citations point
at text that exists, it simply does not answer the question.

`test_empty_index_refuses_rather_than_inventing` passes because an *empty*
index has nothing to retrieve. On a real 2626-chunk corpus there is always
something lexically similar, and the gate lets it through.

---

## F7 — MINOR — `pipelines`: no public way to enumerate what was indexed

Auditing the index requires reaching into the private `_chunks` dict. Any
consumer wanting to verify provenance, count per-document chunks, or export the
index has to break encapsulation.

## F8 — MINOR — `evaluation`: `EvalConfig` exposes no fusion weights

`AskConfig` has `lexical_weight` / `dense_weight`; `EvalConfig` does not. The
central claim of `hybrid_ranker` — that fusion beats either retriever alone —
therefore **cannot be ablated through the evaluation harness**. It had to be
measured by calling `HybridRankerComponent` directly.

## F9 — MINOR — no progress reporting on a 38-minute operation

`kb.ingest()` over 49 PDFs took 2263 s (uncached) and printed nothing. The only
way to tell it apart from a hang was to query the durable checkpoint table from
another process. `durable_steps` accidentally provided the observability the
facade lacks.

---

## What held up well

- **Extraction robustness: 49/49 documents ingested, 0 failures**, including
  1986–1996 papers, malformed font descriptors (`Could not get FontBBox…`
  warnings throughout), and files from 32 KB to 43 MB. The pre-registered
  threshold was >10% failure. This is a genuine pass.
- **Provenance: 18/20 sampled chunk bboxes verified** by re-opening the PDF and
  cropping to the union of the chunk's regions. 0 out-of-bounds. The 2
  mismatches were a rotated figure label and a screenshot-style code block.
  Provenance — the toolkit's core contract — holds on real PDFs.
- **`dag` fan-out: 49/49 completed, 0 failed, 2626 chunks, 49 distinct doc_ids —
  byte-identical document coverage to sequential ingest.** No thread-safety
  defect. `KnowledgeBase` survived 4-way concurrent ingest.
- **The documented GIL limitation, now measured:** fan-out with 4 workers took
  **201.8 s vs 183.8 s sequential** — 10% *slower*. `dag`'s README says
  CPU-bound nodes will not speed up; that is now a number, not a claim.
- **Refusal on an empty index** behaved correctly (F2's silent failure was
  masked precisely because this part works).
- **`durable_steps` on real data:** 49 checkpoints written and replayed exactly
  as designed. The composition is wrong (F2); the component is not.

---

## Two corrections to my own measurements

Recorded because the method matters as much as the result.

1. **Provenance: first reported 10/20 failures, actual 2/20.** The first check
   compared each chunk's *entire* text against only its *first* bbox, which any
   multi-region chunk fails. Fixed to union every region per page.
2. **`dag` fan-out: first reported 2345 chunks vs 2626, suggesting ~11% silent
   loss.** That run was contaminated — an earlier process had not been killed
   (`pkill` is unavailable on this machine) and two runs raced on the same parse
   cache and report file. Re-measured cleanly in a single process: **2626 and
   2626, identical.** There is no concurrency defect.

---

## Verdicts

| Component | Verdict | Evidence |
|---|---|---|
| `adapters` | **NEEDS IMPROVEMENT** | F1, one-line fix verified; but 0 ingest failures on 212 MB |
| `doc_layout` | **USED SUCCESSFULLY** | reading order and provenance held on real two-column PDFs; 18/20 bboxes verified |
| `chunking` | **NEEDS IMPROVEMENT** | F3: 44% of chunks over budget against a real tokenizer |
| `pipelines` | **NEEDS IMPROVEMENT** | F2 (critical), F6, F7, F9 |
| `durable_steps` | **USED SUCCESSFULLY** | worked exactly as documented; F2 is a composition defect in `pipelines` |
| `dag` | **USED SUCCESSFULLY** | 49-node fan-out, 0 failures, identical coverage, GIL cost measured |
| `graph` | **USED SUCCESSFULLY** | via `dag`; no issues at 49 nodes |
| `hybrid_ranker` | **NOT PROVEN** | recall@10 — lexical 0.955, dense 0.909, RRF 0.955. Tied, did not beat. See caveat below |
| `evaluation` | **NEEDS IMPROVEMENT** | F5 (misattributes upstream faults), F8 |
| `guardrails` | **NOT EXERCISED** | no injection payload in this corpus; Phase 7 work |
| `core` | **USED SUCCESSFULLY** | contracts carried provenance through 5 components without issue |
| `concurrency`, `cache`, `governor`, `extraction`, `entity_resolution`, `ports` | **NOT EXERCISED** | no LLM in this run; `entity_resolution` deferred to a dedicated project |

**`hybrid_ranker` caveat, stated plainly:** this ablation cannot fairly judge
it. Ground truth was defined as *a chunk containing the expected snippet
verbatim*, which structurally favours lexical retrieval — BM25 is being scored
on exactly the task it is built for. RRF matching the better retriever while
never losing to it is a reasonable outcome under a biased metric, on n=11. A
fair test needs semantic relevance judgements, not substring ground truth.
Recorded as **NOT PROVEN**, not as a defect.

---

## Reproducing

```bash
python validation/paper_triage/fetch_corpus.py              # ~5 min, 212 MB
rm -f validation/paper_triage/results/ingest.db             # required, see F2
python validation/paper_triage/run_validation.py --step all --no-parse-cache
```

`--no-parse-cache` is the honest extraction measurement (~38 min). Without it,
parsed documents are reused from `results/parse_cache` via
`CachedDocumentSource`, which implements the `DocumentSource` port rather than
bypassing it — the port doing the job it exists for.
