# Toolkits Management

[![CI](https://github.com/RoycenChew/Toolkits_Management/actions/workflows/ci.yml/badge.svg)](https://github.com/RoycenChew/Toolkits_Management/actions/workflows/ci.yml)

A reusable engineering toolkit for document intelligence and retrieval, built on one
principle:

> **Own the algorithms and the contracts. Borrow anything with weights or a server.**

The base install is **standard library only**. Every optional backend (Docling,
LiteLLM, FastEmbed, LanceDB, bm25s, sentence-transformers) is imported lazily inside
a single adapter file, so `import toolkit` pulls in nothing outside Python itself and
the whole thing runs offline with no API key.

```python
from toolkit.pipelines import KnowledgeBase

kb = KnowledgeBase()                       # offline defaults, no API key
kb.ingest_folder("./documents")
answer = kb.ask("what is the voltage limit?")

for c in answer.citations:
    print(c.source_uri, "page", c.page, c.bbox)   # every claim traces to a region
```

| | |
|---|---|
| **Tests** | 498 passing |
| **Type checking** | `mypy` clean across 47 source files |
| **Coverage** | 94% of the library, floor enforced at 90% |
| **Lint** | `ruff` clean |
| **Components** | 21 units |
| **Ports / adapters** | 6 ports, 13 adapters (≥2 per port) |
| **Base dependencies** | none |
| **Python** | 3.10+ |

---

## Table of contents

- [Why this exists](#why-this-exists)
- [Quick start](#quick-start)
- [Architecture](#architecture)
- [The components](#the-components)
- [Ports and adapters](#ports-and-adapters)
- [Build phases, in detail](#build-phases-in-detail)
- [Design decisions worth knowing](#design-decisions-worth-knowing)
- [Bugs the tests caught](#bugs-the-tests-caught)
- [Buy vs build: the standing decision list](#buy-vs-build-the-standing-decision-list)
- [Testing](#testing)
- [Licensing](#licensing)
- [Known limitations](#known-limitations)
- [What is not built](#what-is-not-built)
- [Repository layout](#repository-layout)

---

## Why this exists

The goal was a personal library of high-leverage components that accelerate
hackathons, AI projects and production prototypes — not another framework.

The research phase found that the open-source ecosystem is now strong enough that
**most capability should be borrowed**. Docling parses PDFs better than I will,
LiteLLM normalises 100+ providers, LanceDB is an embedded vector store with no server.
Rebuilding those is waste.

What the ecosystem does *not* ship is the narrow band of things this repo owns:

- **Chunking that carries provenance.** Chonkie, LangChain and LlamaIndex chunk
  *text* — a string goes in, strings come out, and which page it came from is gone.
  Nothing chunks *blocks with geometry* and propagates page + bounding box into every
  chunk. That propagation is what makes citation possible.
- **A repair loop that converges.** Every framework has structured extraction. Almost
  none re-prompt with the *specific* validation error, so the model reproduces the
  same mistake.
- **A relevance gate.** Rank fusion always returns something from a non-empty index,
  so without a gate a knowledge base answers every question ever asked of it.
- **An eval harness over your own golden set** with a regression diff that names the
  cases a change broke.
- **One document model**, so backends are swappable rather than load-bearing.

---

## Quick start

```bash
git clone https://github.com/RoycenChew/Toolkits_Management.git
cd Toolkits_Management

pip install -e .            # stdlib only — works offline
pip install -e ".[dev]"     # + pytest, ruff, mypy

python -m pytest toolkit/tests -q
python -m pytest toolkit/tests -q --cov   # 94%, fails below 90%
```

### Use it from the shell

The index is built once and reloaded in under a second, so the second command
is the one you run a hundred times:

```bash
toolkit ingest ./papers --save papers.kb       # or: python -m toolkit ...
toolkit ask papers.kb "what limits the throughput?"
toolkit eval papers.kb golden.jsonl --out metrics.json
toolkit inspect papers.kb
```

Measured on 49 real arXiv PDFs (212 MB): **38 minutes to ingest, 0.7 seconds to
reload.** `ask` exits 0 when it answered and 1 when it refused, so a script can
tell the difference. A CLI is also the most portable interface here - every
coding agent can run a shell command, including those that support neither MCP
nor the Agent Skills format.

Three runnable demos, all offline:

```bash
python examples/quickstart.py   # folder of documents -> cited answers
python examples/extract.py      # document -> validated object, repair loop visible
python examples/evaluate.py     # "did that change help?" as a table
```

### Optional backends

Each extra is independent. A missing package raises `MissingDependency` naming the
exact install command, never a bare `ImportError`.

```bash
pip install -e ".[docs]"     # pdfplumber, docling      — PDF parsing, OCR, tables
pip install -e ".[llm]"      # litellm                  — 100+ model providers
pip install -e ".[embed]"    # fastembed                — ONNX embeddings on CPU
pip install -e ".[store]"    # lancedb, bm25s           — persistent vector + BM25
pip install -e ".[rerank]"   # sentence-transformers    — cross-encoder reranking
pip install -e ".[eval]"     # ragas, deepeval          — semantic answer metrics
pip install -e ".[all]"      # everything above
```

### Production wiring — same two calls

```python
from toolkit.adapters import (
    Bm25sIndex, CrossEncoderReranker, FastEmbedEmbedder, LanceDBStore, LiteLLMClient,
)
from toolkit.cache import CachedEmbedder, CachedLLM, SqliteCache
from toolkit.governor import GovernedLLM, GovernorConfig
from toolkit.pipelines import IngestConfig, KnowledgeBase

cache = SqliteCache("runs.db")

kb = KnowledgeBase(
    embedder=CachedEmbedder(FastEmbedEmbedder(), cache),
    vector_store=LanceDBStore("./.lancedb"),
    lexical_index=Bm25sIndex(),
    reranker=CrossEncoderReranker(),
    llm=GovernedLLM(
        CachedLLM(LiteLLMClient("gpt-4o-mini"), cache),
        GovernorConfig(max_total_tokens=500_000, max_requests_per_minute=60),
    ),
)

kb.ingest_folder("./documents", IngestConfig(durable_db="ingest.db"))
```

With `durable_db` set, re-running after a crash reports finished documents as
`replayed` and re-embeds nothing.

---

## Architecture

```
┌─ L4  HARNESS ─────────────────────────────────────────────────────────────┐
│  evaluation/  golden sets · metrics · regression diff                      │
├─ L3  PIPELINES ───────────────────────────────────────────────────────────┤
│  pipelines/   KnowledgeBase.ingest_folder()  ·  .ask()                     │
├─ L2  ALGORITHMS (own) ────────────────────────────────────────────────────┤
│  doc_layout · chunking · hybrid_ranker · entity_resolution                 │
│  extraction · cache · governor · concurrency · durable_steps               │
├─ L1  PORTS (own contracts, borrowed implementations) ─────────────────────┤
│  ports.py     DocumentSource · Embedder · LLM · VectorStore                │
│               LexicalIndex · Reranker · Cache                              │
│  adapters/    the ONLY place a vendor SDK may be imported                  │
├─ L0  CORE CONTRACTS (own) ────────────────────────────────────────────────┤
│  core/        Document · Block · Chunk · Provenance · Usage · SearchHit    │
│               BBox · Message · Completion · error taxonomy                 │
└───────────────────────────────────────────────────────────────────────────┘
```

**Rules the codebase actually enforces:**

1. Only `toolkit/adapters/` may import a vendor SDK, and every such import is lazy.
   CI asserts that importing the toolkit pulls in none of them.
2. Every port is validated against **two or more** implementations. A test fails if
   any port drops below two.
3. Each component directory stays independently copyable — none of them import the
   `toolkit` package, so any one can be lifted into an unrelated project alone.

### End-to-end data flow

```
PDF / DOCX / MD / TXT
   │
   ▼  DocumentSource (pdfplumber | Docling | plaintext)
Document  ── blocks in reading order, each with page + bbox
   │
   ▼  doc_layout      reading order, heading levels, page furniture, OCR dedup
   ▼  chunking        provenance-carrying chunks + heading breadcrumbs
   ▼  Embedder        batched, cached, bounded concurrency
   ▼  VectorStore + LexicalIndex
   │
   ▼  hybrid_ranker   RRF fusion → rerank cascade → MMR
   ▼  relevance gate  refuse when nothing is actually about the question
   ▼  LLM             cite-or-refuse prompt
   ▼  verification    every [n] checked against the sources shown
Answer  ── text + citations resolving to page and bbox + retrieval trace
```

---

## The components

Not every unit is the same shape, and pretending otherwise would be the kind of
claim this repo tries not to make. There are eight kinds, recorded per unit in
[`REGISTRY.json`](REGISTRY.json):

| Kind | Shape | Units |
|---|---|---|
| **component** | one class, `execute(input_data) -> output` | `doc_layout` `chunking` `hybrid_ranker` `entity_resolution` `extraction` `guardrails` `durable_steps` `dag` `llm_http` |
| **wrapper** | satisfies the port it wraps, so it composes by construction | `cache` `governor` |
| **functions** | plain functions, no state to hold | `concurrency` `graph` `provider` |
| **contracts** | types and protocols only | `core` `ports` |
| **adapters** | the vendor containment boundary | `adapters` |
| **facade** | wires the rest; `ingest_folder()` + `ask()` | `pipelines` |
| **harness** | measures everything else | `evaluation` `extraction_eval` |
| **interface** | how a human or a script reaches the rest | `cli` |

`copy_tier` in the registry says what you must copy to reuse each one — `standalone`,
`needs_core`, or `needs_package` — and `toolkit/tests/test_packaging.py` verifies every
one of those claims by copying the unit out and importing it in a subprocess with the
repository off `sys.path`.

| Component | What it does | Prior art |
|---|---|---|
| **`core/`** | The shared contracts: `Document`, `Block`, `Chunk`, `Provenance`, `Usage`, `SearchHit`, `BBox`, error taxonomy | own |
| **`ports.py`** | Seven `Protocol`s the toolkit needs from the outside world: `DocumentSource`, `Embedder`, `LLM`, `VectorStore`, `LexicalIndex`, `Reranker`, `Cache` | own |
| **`adapters/`** | 15 adapters, ≥2 per port, every vendor import lazy | — |
| **`doc_layout/`** | Positioned text spans → ordered semantic blocks. Recursive XY-cut for columns, IoU overlap resolution, font-statistics heading levels, recurring-line boilerplate detection | XY-cut (Nagy & Seth 1984), Docling layout post-processing, Marker heading heuristics |
| **`chunking/`** | `Document` → citable chunks. Structure-aware, heading breadcrumbs, whole-sentence overlap, word-split fallback, rendered-text budget enforcement | own (Chonkie's taxonomy for reference) |
| **`hybrid_ranker/`** | Merge incomparable retriever scores. RRF, min-max / z-score fusion, budgeted rerank cascade, MMR diversification | RRF (Cormack 2009), Qdrant/Weaviate fusion, ColBERT/SPLADE cascades |
| **`entity_resolution/`** | Match records with no shared key, unsupervised and explainably. Affine-gap distance, greedy set-cover blocking, Fellegi–Sunter + EM, average-linkage clustering | dedupe, Splink, Gotoh 1982, Fellegi & Sunter 1969 |
| **`extraction/`** | Document + schema → validated object. Targeted repair loop, per-field provenance, grounding. Plus a multi-document splitter | own (shape common to ExtractThinker / Instructor) |
| **`guardrails/`** | Indirect prompt-injection defense: source delimiting, pattern neutralisation, output policy. Closes the hole where an injected answer passed citation verification as *grounded* | own work; delimiting from the injection literature |
| **[`graph/`](toolkit/graph/README.md)** | Deterministic graph algorithms: topological layers, cycle reporting with the actual cycle, SCC, critical path, transitive reduction | own work; Kahn, Tarjan, Dijkstra |
| **[`dag/`](toolkit/dag/README.md)** | DAG execution on a ready queue. A failed node's descendants are *skipped*, not failed, and every skip names its cause | own scheduler; Temporal's retry shape |
| **`cache/`** | Content-addressed caching. Per-**text** embedding keys, whole-request LLM keys | own |
| **`governor/`** | Pre-flight token/cost budget, sliding-window rate limit, selective retry | Temporal's retry shape + own window |
| **`durable_steps/`** | Crash-resumable multi-step execution on a plain SQL checkpoint table, with leasing | DBOS Transact, Hatchet leasing, Temporal replay |
| **`concurrency.py`** | Order-preserving bounded parallel map with index-attributed failures | own |
| **`pipelines/`** | `KnowledgeBase`: `ingest_folder()` + `ask()` with verified citations | own — the wiring |
| **`evaluation/`** | Golden sets, IR metrics, regression diff naming broken cases | own harness; standard IR metrics |
| **[`extraction_eval/`](toolkit/extraction_eval/README.md)** | Golden sets keyed on field paths. Field accuracy, line-item P/R/F1, grounding rate, and the **silent error rate**: accepted results that were wrong where it mattered | own |
| **[`provider/`](toolkit/provider/README.md)** | Resolve an AI provider from the environment — key, endpoint, model, API style — or an error naming what to set. Redacts the key in `repr` | own; shortcut-table shape from PRism |
| **[`llm_http/`](toolkit/llm_http/README.md)** | The `LLM` port over plain HTTP, no vendor SDK. Classifies failures without retrying them, and refuses to return a truncated answer as a success | own; two-shape split from PRism |
| **`cli.py`** | `ingest` / `ask` / `eval` / `inspect` over a saved index. 38-minute corpus, 0.7-second reload | own |

Each has its own README with architecture, input/output schema, limitations and
integration notes.

---

## Ports and adapters

The stdlib implementation comes first in each row — that one is why the whole toolkit
runs offline.

| Port | stdlib | real backends |
|---|---|---|
| `DocumentSource` | `PlainTextSource` | `PdfPlumberSource`, `TesseractSource`, `DoclingSource` |
| `Embedder` | `HashingEmbedder` | `FastEmbedEmbedder` |
| `VectorStore` | `InMemoryVectorStore` | `LanceDBStore` |
| `LexicalIndex` | `SqliteFtsIndex` (FTS5) | `Bm25sIndex` |
| `LLM` | `ScriptedLLM` | `LiteLLMClient` |
| `Reranker` | `LexicalOverlapReranker` | `LLMReranker`, `CrossEncoderReranker` |

**Two implementations per port is a hard rule, enforced by a test.** A protocol written
against a single backend is just that vendor's interface with the names changed, and
you discover this at the worst possible moment: when you try to swap it.

`toolkit/tests/test_contracts.py` runs every adapter for a port through the *same*
assertions and reports which are verified:

```
port DocumentSource: 3 implementation(s) passing - ok
port Embedder:       2 implementation(s) passing - ok
port LLM:            2 implementation(s) passing - ok
port LexicalIndex:   2 implementation(s) passing - ok
port Reranker:       2 implementation(s) passing - ok
port VectorStore:    2 implementation(s) passing - ok
```

The stdlib adapters are not only test doubles:

- `HashingEmbedder` — hashed n-gram bag, deterministic, a real (weak) retrieval signal
  with no download.
- `SqliteFtsIndex` — BM25 via SQLite FTS5, which ships with CPython.
- `InMemoryVectorStore` — exact brute-force cosine. Under ~50k chunks this is faster
  than an ANN index *and* exact, which removes a variable while debugging retrieval.
- `ScriptedLLM` — deterministic offline model. Debug a pipeline's control flow, the
  retries and the repair loop without spending a token.
- `LexicalOverlapReranker` — BM25 scored over *the candidate set itself*, so its IDF
  comes from the candidates rather than the corpus. A genuinely different signal from
  a global-IDF first stage.

---

## Build phases, in detail

### Phase 0 — Packaging ✅

`pyproject.toml` with independent optional extras, `ruff` + `mypy` + `pytest` config,
editable install. The four original components moved under the `toolkit` package and
`sys.path` hacks removed.

**Verified:** `pip install -e .` works; `import toolkit` pulls in nothing outside the
standard library.

---

### Phase 1 — The spine ✅

The phase that makes everything after it cheap.

**Built:** `core/` contracts (`Document`, `Block`, `Chunk`, `Provenance`, `Usage`,
`Completion`, `SearchHit`, `BBox`, `Message`, error taxonomy) and six ports as
`Protocol`s, each with two or more working adapters.

`Provenance` is part of the *core*, not an optional extra. A chunk that cannot say
which page and region it came from cannot be cited, highlighted or verified — and an
answer you cannot verify is a demo rather than a system.

**Verified:** the same `document_source_contract` passes on pdfplumber *and* on
Docling — the swap this phase exists to enable. The contract suite generates its own
minimal valid PDF, so the PDF path is exercised with no checked-in fixture.

**Defects the contract tests caught:**

1. `Bm25sIndex` returned arbitrary zero-scored hits for an empty query where
   `SqliteFtsIndex` returned none — exactly the divergence a port exists to prevent.
2. The `LiteLLMClient` suite passed despite a missing dependency, because the import
   is lazy at *call* time, not construction.
3. A `str`/`bytes` concat in the PDF fixture generator.

---

### Phase 1b — CI, types, reranker adapters ✅

- **CI** (`.github/workflows/ci.yml`), two jobs. The first installs **no** optional
  backends and asserts that importing the toolkit pulls in none of them, so the
  stdlib-only promise is verified rather than assumed. The second installs them, so
  adapters that report SKIP in the first are exercised somewhere.
- **mypy** clean across all 42 source files. Scope was widened from just the contracts
  once the seven errors it found were fixed.
- **Reranker adapters**, closing the one port that violated the two-implementation
  rule.

---

### Phase 2 — Chunking, cache, governor, concurrency ✅

**`chunking/`** — structure-aware, provenance-carrying:

- Sections beat budgets. A heading is the author telling you where a topic starts;
  packing across it to fill a 512-token quota discards that for nothing.
- Heading breadcrumbs are prepended. `"must not exceed 40V"` is useless alone;
  `"Safety Manual > Voltage"` attached to it is not. This repairs the most common RAG
  failure mode for one line of cost.
- Overlap is in whole sentences. Overlapping by raw token count severs sentences, and
  a half-sentence embeds to noise.
- Provenance is **one region per page**, never a single box straddling a page break —
  that box does not exist.

**`cache/`** — `SqliteCache`, `CachedEmbedder`, `CachedLLM`:

- Embeddings cached **per text, not per batch**. A batch-level cache is nearly useless
  in practice: the next run never sends an identical batch, so everything misses.
  Per-text keys mean editing one document in a 1,000-document corpus costs one
  embedding, not a thousand.
- Keys include everything that changes the answer — model, temperature, `max_tokens`,
  the full message list. `sort_keys=True` in the hash, or dict ordering silently halves
  the hit rate.
- Non-zero temperature bypasses the cache: the caller asked for variation.

**`governor/`** — budget, rate limit, retry:

- Budget checked **before** the call. Checking afterwards makes the limit advisory.
- Sliding window over real timestamps, not a fixed bucket — a fixed bucket permits a
  double-rate burst across the boundary, precisely when providers refuse.
- Only `RateLimited` / `AdapterError` retry. A `KeyError` is a bug, not a transient
  failure, and retrying it three times just delays the real signal.

**`concurrency.py`** — bounded parallel map:

- Results stay in **input order**. `as_completed` scrambles them, and misaligning
  embeddings against their texts is a corruption no test notices until retrieval
  quality drops. The test deliberately makes early items slow so any completion-order
  implementation fails it.

---

### Phase 3 — `ingest()` + `ask()` ✅

`pipelines/KnowledgeBase` wires source → layout → chunking → embed → store + lexical →
fusion → cited answer, with each document ingested as one durable step when
`durable_db` is set.

**Citations are verified, not trusted.** Every `[n]` the model emits is checked against
the sources it was actually shown. An out-of-range marker lands in
`answer.unverified_markers` and is **never** rendered as a citation — the model
invented a source. An answer citing nothing real reports `grounded=False`.

**Extractive mode.** With no LLM configured, `ask` returns the best passages fully
cited. Genuinely useful, and the right way to debug retrieval: if the correct passage
is not in the extractive output, no model was going to rescue it.

**The relevance gate** — found by running the demo, not by a test. The off-topic
question *"what is the airspeed velocity of an unladen swallow?"* was answered
confidently, because RRF scores are positional and a non-empty index always returns
something. Without a gate, a knowledge base answers every question ever asked of it,
which is the failure mode that destroys trust fastest.

---

### Phase 4 — Structured extraction ✅

**`ExtractionComponent`** — schema → prompt → parse → coerce → validate → **targeted
repair** → ground.

The repair loop is the reason this exists. The common approach re-runs the whole
prompt and hopes; the model has no idea what was wrong, so it often reproduces the
same mistake. This one sends back only what failed:

```
Your JSON had these problems. Fix only these fields and return the complete JSON object again:
- issued: expected a date formatted YYYY-MM-DD (got '14/03/2024')
    issued means: Date the invoice was issued
- status: must be one of: paid, unpaid, overdue (got 'not paid')
    status means: Payment status
Leave every other field exactly as it was.
```

Fields that passed are not mentioned — re-sending the whole schema invites the model to
rewrite work that was already right, which is how a repair round makes things worse.

Then every field points back at the page:

```
invoice_number  page 1  bbox [50, 82, 420, 100]  <- Invoice number INV-88213
total           page 1  bbox [50, 192, 420, 210] <- Total due: $1,240.50
issued          (not located verbatim)
```

**`DocumentSplitterComponent`** — for the other half: a 40-page scan that is really
six invoices. Detects page-number restarts (the strongest signal there is, and free),
repeated first-page headings, and top-level headings after body text. It reads page
numbers only from header/footer blocks, because *"see page 1 of the appendix"* in prose
is a reference, not a boundary.

---

### Phase 5 — Evaluation harness ✅

What separates a toolkit from a pile of code. Without it, every retrieval decision is
argued from vibes.

```
Does the relevance gate earn its keep?
metric                     before    after     change
overall_correct             1.000     0.750     -0.250
refusal_accuracy            1.000     0.000     -1.000

cases fixed: 0  broken: 2  unchanged: 6
broken: offtopic, offtopic2
```

**Ground truth is never chunk ids.** The obvious design — list the `chunk_id`s that
should be retrieved — invalidates itself the moment it becomes useful, because chunk
ids change whenever chunking changes, and re-tuning chunking is exactly the experiment
you most want to run. Relevance is judged on `expected_snippets` (copy text out of the
document), pages, or doc ids — all stable under any chunking strategy.

**Unanswerable cases are first-class, and scored separately.** Without them the
relevance gate is unmeasurable, and a system that answers everything scores perfectly
on a set of answerable questions. Aggregated together, the gate trade cancels out to
"nothing changed" — which is the wrong conclusion.

Metrics are hand-computed in the tests rather than checked against the implementation.
A metric verified against itself is not verified.

---

## Design decisions worth knowing

### Measurement killed a design: the relevance gate

My first gate used an absolute cosine floor of 0.25. Then I measured, on the sample
corpus with the default `HashingEmbedder`:

| query | dense score | actually relevant? |
|---|---|---|
| "what is the maximum supply voltage?" | 0.457 | yes |
| "airspeed velocity of an unladen swallow?" | **0.293** | **no** |
| "how long do refunds take?" | **0.248** | **yes** |

The irrelevant query **out-scores a relevant one**, because `HashingEmbedder` compares
character trigrams rather than meaning. Sentence embedders fail the opposite way:
bge-class models put unrelated text at 0.6–0.8, so a 0.25 floor would admit everything.

**No single threshold is portable.** So the gate runs on term overlap,
`min_dense_similarity` defaults to `None`, and the calibration procedure plus these
numbers live in the docstring. Three tests lock in that the dense arm stays opt-in and
that a mis-calibrated floor admits junk.

### A metric that reads as catastrophe when nothing was measured

`answer_match` originally reported `0.000` when no case declared an expected string.
That is indistinguishable from "measured and failed everything". The key is now absent
instead. An eval harness that emits a misleading number has failed at its only job.

### A signal that fires on correct work is worse than no signal

`require_grounding` flagged a correct `"2024-03-14"` as fabricated, because the source
document says *"Issued 14 March 2024"*. A `DATE` field is *required* to be ISO, so it
can never appear verbatim. Dates and booleans are now exempt; strings and numbers stay
in scope, and that is where fabrication actually happens.

### Ambiguous dates are refused, not guessed

`01/02/2024` is 1 February or 2 January depending on where the document came from.
Guessing silently corrupts data, so it costs a repair round and the model is asked.
Unambiguous rewrites (`2024/03/14`, `14 March 2024`) are done locally, because spending
a network round-trip on formatting is waste.

### Local coercion before repair

Models return `"$1,240.50"` and `"yes"`. Fixing that locally is free; a repair round is
not. In the demo, three defects in the first attempt produce only two repair items.

### Config defaults derive from each other

`overlap_tokens` and `min_tokens` originally had fixed defaults (64 and 48) that
crashed on any `max_tokens` below them. Lowering one field should not invalidate a
config, so both now derive from `max_tokens`. I fixed the design rather than the tests
that exposed it.

### A saturated benchmark proves nothing

`examples/evaluate.py` detects when the baseline already scores 1.000 and says so:

> *the baseline already scores 1.000, so this corpus cannot tell these two settings
> apart. That is a fact about the golden set, not evidence that the change is safe.*

---

## Bugs the tests caught

Kept as a record, because each was a real defect that would have shipped.

| # | Bug | Why it mattered |
|---|---|---|
| 1 | Line assembly spliced left- and right-column text at matching y coordinates | Column decomposition must run *before* line grouping — which is why the original XY-cut is ordered that way. A silent, hard-to-spot corruption. |
| 2 | `min_column_width_ratio=0.15` promoted a table gutter to a column break | Shattered reading order on any page with a table. Fixed by the insight that a body column narrower than a quarter page is implausible. |
| 3 | Token budget enforced on per-unit sums, not the rendered chunk | Separators, bullets and breadcrumbs add tokens the sum never saw; the default heuristic also switches which term dominates on concatenation. Oversized chunks get rejected by embedding endpoints. |
| 4 | `overlap_tokens` / `min_tokens` fixed defaults contradicted a lowered `max_tokens` | A reasonable override became a crash. |
| 5 | `Bm25sIndex` returned zero-scored hits for an empty query | Diverged from `SqliteFtsIndex` — exactly what a port exists to prevent. |
| 6 | `KnowledgeBase(lexical_index=None)` could not disable the keyword index | `None` was indistinguishable from "not supplied". Needed a sentinel. |
| 7 | Grounding read `1240.5` against `$1,240.50` as fabricated | Noise exactly where the signal matters most. Needed a separator-stripped view. |
| 8 | `require_grounding` flagged correct ISO dates | See above — a signal that fires on correct work gets ignored. |
| 9 | `LexicalOverlapReranker` scored **zero** on "refund" vs "refunds" | Not an edge case; most real queries. A reranker returning all zeros is worse than none, since it destroys first-stage ordering. |
| 10 | `FieldComparison.field` shadowed the `dataclasses.field` import | Works at runtime (an annotation without assignment does not bind) but a trap for any reader. |
| 11 | Off-topic questions answered confidently | Found by running the demo. Led to the relevance gate. |
| 12 | `answer_match` reported `0.000` when nothing was measured | Found by running the demo. |

---

## Buy vs build: the standing decision list

Written down so it stops being re-litigated at 2am during a hackathon. Research current
as of September 2026.

### Borrow — do not reimplement

| Capability | Default choice | Licence | Why |
|---|---|---|---|
| PDF → structured doc | **Docling** | MIT | Only major parser with a clean licence *and* a typed doc model with provenance |
| OCR (printed, multilingual) | **PaddleOCR-VL** | Apache-2.0 | Tops OmniDocBench v1.6 (~96.3%) at 0.9B params |
| OCR (messy / handwriting) | **dots.ocr** | MIT | Grounding ties text back to page position |
| Table structure | Docling TableFormer; **Camelot** for ruled tables | MIT | `microsoft/table-transformer` was archived Sep 2026 — prefer wrappers that vendor it |
| LLM calling | **LiteLLM** | MIT | Provider auth, parameter naming, error shapes. Never hand-roll this |
| Structured output (local) | **LLGuidance** | MIT | Earley parser over regex derivatives, ~50µs/token. Research-grade — never reimplement |
| Embeddings | **FastEmbed** | Apache-2.0 | ONNX on CPU, no torch — matters for laptop demos |
| Vector store | **LanceDB** | Apache-2.0 | Embedded, file-based, no server. Hackathon-correct |
| Lexical search | **bm25s** | MIT | Fast and dependency-light |
| Reranking | bge-reranker / **PyLate** | MIT | PyLate is the maintained late-interaction path, not the original ColBERT repo |
| Agent orchestration | **Pydantic AI** (typed) or LangGraph (graph state) | MIT | Pydantic AI for stability; LangGraph when you need durable graph execution |
| Entity resolution at scale | **Splink** | MIT | Pushes blocking into DuckDB/Spark when you outgrow in-memory |
| Eval metrics | **Ragas** + **DeepEval** | Apache-2.0 | Complementary, not competitors |
| Tracing | **Langfuse** / OpenLLMetry | MIT / Apache-2.0 | Both OTel-native and self-hostable |
| Web extraction | **Crawl4AI** | Apache-2.0 | Emits markdown locally, no API |
| ASR | **faster-whisper**; WhisperX for word timestamps | MIT | 4× faster than reference Whisper |
| Workflow at scale | **Hatchet** / DBOS | MIT | When `durable_steps` outgrows one machine |

### Build — nobody ships these well

The document model and ports · provenance-preserving chunking · the schema-extraction
repair loop · the cost/cache/governor layer · MinHash + LSH near-duplicate detection
*(not yet built)* · the eval harness · the bounded-concurrency async map.

---

## Testing

```bash
python -m pytest toolkit/tests -q          # 498 tests
python -m ruff check toolkit examples      # lint
python -m mypy toolkit                     # types, 42 files

# Standalone runners, no pytest needed:
python toolkit/tests/test_contracts.py     # prints which adapters verified each port
python toolkit/tests/test_toolkit.py
python toolkit/tests/test_phase2.py
python toolkit/tests/test_pipelines.py
python toolkit/tests/test_extraction.py
python toolkit/tests/test_evaluation.py
```

| Suite | Tests | Covers |
|---|---|---|
| `test_toolkit.py` | 27 | doc_layout, hybrid_ranker, entity_resolution, durable_steps |
| `test_contracts.py` | 13 + 13 adapters | core contracts + every port against every adapter |
| `test_phase2.py` | 32 | chunking, cache, governor, concurrency |
| `test_pipelines.py` | 23 | ingest, ask, citations, relevance gate |
| `test_extraction.py` | 24 | schema validation, repair loop, grounding, splitter |
| `test_evaluation.py` | 20 | metrics, relevance judgement, regression diff |

The tests assert the properties that make each component worth having, not that the
code runs. Examples:

- RRF ranks a dual-retriever hit above a single-retriever top hit, across incomparable
  score scales.
- A rerank budget of 1 provably cannot reorder beyond the head.
- Average linkage refuses a chain that connected components accepts.
- A two-column page reads the left column fully before the right.
- A table gutter is not mistaken for a column break.
- A killed workflow resumes at the failed step and re-runs nothing that completed —
  verified across a fresh store object over the same database file.
- An invented citation marker never becomes a citation.

---

## Licensing

This repository is MIT licensed. **No source was copied from any upstream project** —
every component was reimplemented from published algorithms and documented behaviour.

That was a deliberate constraint, because the strongest document parsers are copyleft:

- **Marker** — GPL-family, with a ~$5M revenue threshold on its model weights.
- **MinerU** — AGPL.

Both are excellent and worth reading; neither can be vendored into a permissive
toolkit. The permissively licensed prior art (Docling, Splink, dedupe, DBOS, LiteLLM,
LanceDB — all MIT or Apache-2.0) is credited in each component's README under
*Original Source*.

**Verify licences yourself before commercial use.** Parser-family licensing has been
shifting, and model weights routinely carry terms separate from the code.

---

## Known limitations

Stated plainly, per component. None of these are hidden in the code.

**Retrieval quality with the default embedder is weak** — `HashingEmbedder` is lexical,
not semantic. On the sample corpus, *"how long do refunds take?"* retrieves the
*Eligibility* section rather than *Processing*. Swap in `FastEmbedEmbedder` for real
semantics; the default exists so the pipeline runs with nothing installed.

**Tables are not reconstructed.** `doc_layout` passes table text through as prose;
`chunking` keeps a `TABLE` block whole where it fits and word-splits it where it does
not. Neither preserves row/column structure — that genuinely needs a model. Use
Docling or Camelot for table structure and feed non-table regions here.

**The relevance gate trades recall for precision.** Term overlap refuses genuine
paraphrases that share no vocabulary with the source. Set `min_dense_similarity` once
calibrated against your embedder, or `relevance_gate=False` if your queries are
paraphrase-heavy.

**Chunks are mirrored in memory** in `KnowledgeBase` so answers can carry full
provenance. A million-chunk corpus will not fit; for that, read provenance back from
the store instead.

**Extraction makes one LLM call per attempt over the whole document.** No map-reduce;
`context_char_limit` truncates. For a 200-page contract, split or retrieve first.

**Grounding is substring matching.** A value the model correctly summed or paraphrased
reads as ungrounded, so `require_grounding` suits verbatim extraction and not derived
fields.

**The splitter is heuristic.** Documents with no footer numbering and no headings give
one segment. It needs a `DocumentSource` that classifies page furniture —
`PdfPlumberSource` does via `doc_layout`; a raw text dump does not.

**Governor state is per-process.** Eight workers with `max_requests_per_minute=60` will
collectively issue 480. A cluster-wide budget needs a shared counter.

**The cache has no expiry or eviction.** It grows until you `clear()` it. Identity comes
from the wrapped model's `model_version`; a model without one falls back to its class
name and can collide.

**`durable_steps` is not a scheduler.** No queue, no worker pool, no cron. It makes a
run resumable; deciding when to run it is yours. `idempotent=False` gives *detection*,
not exactly-once.

**`entity_resolution` assumes conditional independence between fields.** Correlated
fields (city and postcode) double-count their evidence and inflate the weight. Blocking
is in-memory; for millions of records push blocking into SQL and feed candidate pairs
to the scoring half.

**Evaluation answer scoring is substring matching**, not semantic. Relevance is binary.
No statistical significance testing — on twenty cases, a 0.05 move is noise.

**No async anywhere.** Threads via `concurrency.py`, because every backend here is a
synchronous IO-bound call and an asyncio-first design forces the whole call stack above
it to become async.

---

## What is not built

Deliberate omissions, so nobody goes looking.

| Not built | Why |
|---|---|
| **LLGuidance backend** for constrained decoding | With a local model it would largely remove the need for the repair loop, but it is a dependency-level integration rather than a component — and the repair loop is what a hosted model needs regardless. |
| **MinHash + LSH** near-duplicate detection | Planned in the roadmap for corpus-scale dedup. `entity_resolution`'s blocking is pairwise and in-memory, which is a different problem. |
| **Tracing hook** (Langfuse / OpenLLMetry) | Pure adapter work, driven by a real project rather than completeness. |
| **Crawler, ASR, vision adapters** | Same — one adapter each, written when needed. |
| **Docling layout/table models as a component** | Model weights are not a component. Depend on Docling and feed its spans in. |
| **MinerU** | AGPL. Read it for the region-routing ideas; do not vendor it. |
| **ColBERT PLAID index** | Compiled extensions and research-organised code. Use PyLate. |

See [`toolkit/ROADMAP.md`](toolkit/ROADMAP.md) for the full plan and per-phase record.

---

## The highest-value next step is not code

Right now the sample corpus scores **1.000 on everything**, which the eval demo tells
you means it cannot detect a regression at all.

**Write 20 golden-set cases against documents you actually care about** — ten
answerable, ten unanswerable. It takes about an hour, and it is the hour that makes
every later tuning decision cheap:

```jsonl
{"case_id": "voltage", "query": "max supply voltage?", "expected_snippets": ["must not exceed 40V"]}
{"case_id": "offtopic", "query": "who won the 1998 world cup?", "unanswerable": true}
```

```python
from toolkit.evaluation import EvalDataset, EvalRunner, diff_reports

golden = EvalDataset.from_jsonl("golden.jsonl")
report = EvalRunner(kb).execute(golden)
report.to_json("baseline.json")     # commit this next to the golden set
```

Then every retrieval change prints a table and a list of broken case ids instead of a
feeling.

---

## Repository layout

```
.
├── REGISTRY.json                  the ledger: layers, deps, copy tiers, limitations
├── CHANGELOG.md                   what changed, and what each release got wrong
├── pyproject.toml                 optional extras, ruff / mypy / pytest config
├── .gitattributes                 LF everywhere; stops CRLF and BOM drift
├── .github/workflows/ci.yml       two jobs: stdlib-only, then with backends
├── docs/
│   ├── ARCHITECTURE.md            target design: profiles, planes, 22 stages, I/O
│   ├── MATURITY.md                honest readiness assessment and what it found
│   └── PLAYBOOK.md                how this is maintained + Definition of Done
├── examples/
│   ├── quickstart.py              folder of documents -> cited answers
│   ├── extract.py                 document -> validated object, repair loop
│   ├── evaluate.py                three A/B experiments with a regression diff
│   └── cookbook.py                one snippet per unit + the recipe catalogue
├── stress/                        hostile corpus + probe harness (exploratory)
│   ├── make_corpus.py             14 documents, each attacking one assumption
│   └── run_stress.py              probes; reports FAIL / KNOWN / PASS
└── toolkit/
    ├── py.typed                   PEP 561 marker, without which consumers get no types
    ├── core/                      L0 contracts, error taxonomy, text normalisation
    ├── ports.py                   L0 protocols
    ├── adapters/                  L2 — the only place a vendor SDK may be imported
    │   ├── sources.py             PlainText / PdfPlumber / Tesseract / Docling
    │   ├── embedders.py           Hashing / FastEmbed
    │   ├── llms.py                Scripted / LiteLLM
    │   ├── stores.py              InMemory / LanceDB / SqliteFts / Bm25s
    │   └── rerankers.py           LexicalOverlap / LLM / CrossEncoder
    ├── cache/                     L1 content-addressed model-call cache
    ├── governor/                  L1 budget, rate limit, retry
    ├── concurrency.py             L1 bounded parallel map
    ├── durable_steps/             L1 crash-resumable execution
    ├── provider/                  L1 find a key, endpoint and model (standalone)
    ├── graph/                     L2 graph algorithms (standalone)
    ├── dag/                       L2 DAG execution, ready queue, resumable
    ├── doc_layout/                L2 reading order, headings, furniture
    ├── chunking/                  L2 provenance-carrying chunks
    ├── hybrid_ranker/             L2 RRF, cascade, MMR
    ├── entity_resolution/         L2 blocking, Fellegi-Sunter, clustering
    ├── guardrails/                L2 injection defense, output policy
    ├── extraction/                L2 repair loop, word-level grounding, splitter
    ├── llm_http/                  L2 the LLM port over plain HTTP, no SDK
    ├── pipelines/                 L3 KnowledgeBase: ingest + ask
    ├── evaluation/                L4 golden sets, metrics, regression diff
    ├── extraction_eval/           L4 extraction metrics, silent error rate
    ├── cli.py                     L4 ingest / ask / eval / inspect
    ├── __main__.py                `python -m toolkit` -> cli.main
    └── tests/                     498 tests, seventeen suites
```

Every component directory carries its own `README.md` with architecture, input/output
schema, usage, limitations, integration guide and extraction notes.

---

## License

MIT
