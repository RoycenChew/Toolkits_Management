# Toolkit Roadmap — from four components to a real engineering kit

> **This is the original plan, written when the toolkit was four components.
> It is kept because the buy-vs-build reasoning still holds and is still
> re-read; its description of what exists is history, not status.**
>
> What has since shipped: the shared document model, ports with two
> implementations each, LLM and embedding adapters, persistence, caching, cost
> control, concurrency, packaging, evaluation — every gap this document opens by
> naming. The toolkit is now 20 units under a 25 cap.
>
> For current status see [`../README.md`](../README.md); for what changed and
> what each release got wrong see [`../CHANGELOG.md`](../CHANGELOG.md).

**The honest answer at the time, to "is this enough?": no, and not close.**

What exists today is four good algorithms with no spine. There is no shared document
model, no way to call an LLM, no embeddings, no persistence, no caching, no cost
control, no evaluation, no concurrency, and no packaging. In a hackathon you would
still spend the first three hours writing the same glue you always write, and the
four components would sit at the edge of it.

But the fix is *not* "add twenty more components". The biggest risk to this toolkit is
rebuilding what Docling, PaddleOCR, LiteLLM and LanceDB already do better than you
will. Research below says the ecosystem is now strong enough that **most capability
should be borrowed, and only a narrow band of things are worth owning.**

So this plan has three jobs:

1. Decide, per capability, **buy vs build** — and write the "buy" list down so you
   stop re-litigating it at 2am during a hackathon.
2. Build the **spine** that makes borrowed pieces interchangeable: one document model,
   one set of ports, one result contract.
3. Build only the **glue nobody ships**: provenance-preserving chunking, a cost/cache
   layer, a schema-extraction repair loop, and an eval harness wired to your own data.

---

## Part 1 — What is actually missing

Grouped by how much it hurts. "Hurt" is measured as: how often does its absence block
a real project or cost you hackathon hours.

### Tier 1 — Blocking. Without these the toolkit is not usable as a toolkit.

| Gap | Why it blocks | Own or borrow |
|---|---|---|
| **No shared document model** | `doc_layout` emits `Block`; nothing consumes it. Every new component invents its own types again. | **Own.** This is the spine. |
| **No LLM port** | Cannot call a model at all. Every project rewrites provider handling, retries, timeouts. | Borrow (LiteLLM) behind an own port. |
| **No embedding port** | Same, for vectors. Also needed by semantic chunking and MMR. | Borrow behind an own port. |
| **No vector store port** | `hybrid_ranker` takes ranked lists but nothing produces them. | Borrow (LanceDB) behind an own port. |
| **No chunking** | The gap between `doc_layout` blocks and retrieval. Currently a hand-written loop every time. | **Own** — see why below. |
| **No packaging** | `sys.path.insert` is not a toolkit. No `pyproject.toml`, no CI, no version. | **Own.** Half a day. |
| **No caching** | Re-running a pipeline re-pays for every LLM and embedding call. In a hackathon this is the difference between 30 and 300 iterations. | **Own** (thin, content-addressed). |

### Tier 2 — High value. These are what make it feel powerful rather than academic.

| Gap | Why it matters | Own or borrow |
|---|---|---|
| **No OCR / no parsing** | `doc_layout` starts *after* text has coordinates. Nothing produces coordinates. | Borrow (Docling / PaddleOCR-VL / pdfplumber) behind one `DocumentSource` port. |
| **No structured extraction** | "PDF → validated Pydantic object" is the single most common hackathon demand. Needs a schema→prompt→validate→**repair** loop. | **Own** the loop; borrow the model call. |
| **No table structure** | Tables come through as prose today. Genuinely needs a model. | Borrow (Docling TableFormer / Camelot / GMFT). |
| **No cost + rate control** | Token budgets, RPM limits, concurrency caps. Every provider fails differently. | **Own** (a governor around the LLM port). |
| **No evaluation** | You cannot tell whether a retrieval change helped. This is what separates a toolkit from a pile of code. | Borrow metrics (Ragas/DeepEval), **own** the harness + golden-set format. |
| **No near-duplicate detection at scale** | `entity_resolution` blocking is in-memory and pairwise. MinHash/LSH covers corpus-scale dedup, a different problem. | **Own** (MinHash + LSH is ~150 lines and has no good small dependency). |
| **No async / concurrency** | Everything is sequential. Embedding 5,000 chunks serially is unusable. | **Own** (a bounded-concurrency map with backpressure). |
| **No observability** | No way to see where a pipeline spent time or money. | Borrow (Langfuse / OpenLLMetry, both OTel-based) behind an optional hook. |

### Tier 3 — Domain reach. Add when a project needs it, not before.

Web extraction (Crawl4AI/Trafilatura) · ASR + diarization (faster-whisper/WhisperX) ·
Vision (supervision/Ultralytics) · Constrained decoding (LLGuidance) · Time-series ·
Geospatial. All of these are **pure borrow** — one adapter each, written when needed.

---

## Part 2 — Buy vs build: the standing decision list

The rule: **own the algorithms and the contracts, borrow the models and the
infrastructure.** Anything needing weights, a server, or a protocol implementation is
a borrow. Anything that is a page of maths or a data contract is a build.

### BORROW — do not reimplement these

| Capability | Default choice | Licence | Why this one | Fallback |
|---|---|---|---|---|
| PDF → structured doc | **Docling** | MIT | Only major parser with a clean licence *and* a typed doc model with provenance | pdfplumber (text-only, zero ML) |
| OCR (printed, multilingual) | **PaddleOCR-VL** | Apache-2.0 | Tops OmniDocBench v1.6 (~96.3%) at 0.9B params | Tesseract (CPU, no GPU, weak layout) |
| OCR (messy/handwriting) | **dots.ocr** (MIT) or olmOCR | MIT / varies | Grounding ties text back to page position | Surya (strong sub-3B, but GPL-family — check) |
| Table structure | Docling TableFormer; **Camelot** for ruled tables | MIT / MIT | TATR itself was archived Sep 2026 — prefer wrappers that vendor it | GMFT (~270MB TATR download) |
| LLM calling | **LiteLLM** | MIT | One interface, 100+ providers, the thing you must never hand-roll | Provider SDK direct |
| Structured output (local models) | **LLGuidance** | MIT | Earley parser over regex derivatives; ~50µs/token. Research-grade — never reimplement | Provider-native JSON mode |
| Embeddings | **FastEmbed** (ONNX, CPU) or sentence-transformers | Apache-2.0 | FastEmbed needs no torch — matters for laptop demos | Hosted embedding API |
| Vector store | **LanceDB** (embedded) | Apache-2.0 | Zero-server, file-based, survives restarts. Hackathon-correct | Qdrant (server, better hybrid) |
| Lexical search | **bm25s** or Tantivy | MIT / MIT | bm25s is fast and dependency-light | SQLite FTS5 (already on disk) |
| Reranking | bge-reranker / **PyLate** (ColBERT) | MIT | PyLate is the maintained late-interaction path, not the original ColBERT repo | Cross-encoder via sentence-transformers |
| Agent orchestration | **Pydantic AI** (typed) or LangGraph (complex/durable) | MIT | Pydantic AI for stability and types; LangGraph when you need real graph state | Your own loop + `durable_steps` |
| Chunking reference | **Chonkie** | MIT | Read it; its Recursive/Semantic/SDPM/Late chunkers are the state of the practice | — |
| Entity resolution at scale | **Splink** | MIT | When you outgrow in-memory: it pushes blocking into DuckDB/Spark | — |
| Eval metrics | **Ragas** (RAG) + **DeepEval** (broad/CI) | Apache-2.0 / Apache-2.0 | Complementary, not competitors; teams run both | promptfoo (red-teaming, Node) |
| Tracing | **Langfuse** or OpenLLMetry | MIT / Apache-2.0 | Both OTel-native, both self-hostable | stdout span printer |
| Web extraction | **Crawl4AI** (local) / Trafilatura (text only) | Apache-2.0 / GPL-ish — check | Crawl4AI emits markdown locally, no API | Firecrawl (hosted) |
| ASR | **faster-whisper**; **WhisperX** for word timestamps + diarization | MIT | 4× faster than reference Whisper, CTranslate2 backend | Hosted transcription |
| Workflow at scale | **Hatchet** / DBOS | MIT | When `durable_steps` outgrows one machine | — |

**Licence landmines to remember:** Marker is GPL-family with a ~$5M revenue threshold
on weights. MinerU is AGPL. Both are excellent; neither goes in a permissive toolkit.
Model weights routinely carry terms separate from the code — check both, every time.

### BUILD — these are yours, because nobody ships them well

1. **The document model + ports.** Every framework has its own `Document`. Owning one
   is what lets you swap Docling for PaddleOCR for pdfplumber without touching
   pipeline code. This is the highest-leverage thing in the entire plan.
2. **Provenance-preserving chunking.** Chonkie chunks *text*. Nothing chunks
   *blocks-with-bounding-boxes* and carries page + bbox through to the chunk. That
   carry-through is what makes citation and highlighting possible, and it is the
   difference between a demo and something believable.
3. **The schema-extraction repair loop.** Validation failure → targeted re-prompt with
   the specific error → re-validate, bounded. Every framework has extraction; almost
   none have a repair loop that converges, and it is where most of the accuracy is.
4. **Cost/cache/governor layer.** Content-addressed cache over every model call, plus
   token budget and concurrency limits. Saves real money and makes runs reproducible.
5. **MinHash + LSH near-duplicate detection.** Corpus-scale dedup. `datasketch` exists
   but is heavy for what is 150 lines of clean maths.
6. **The eval harness.** Not the metrics — Ragas has those. The *harness*: a golden-set
   format, a runner, and a regression diff so you can answer "did that change help?"
7. **Bounded-concurrency async map.** With retry, backoff and backpressure. Everyone
   rewrites this and most get the backpressure wrong.
8. The four existing components, kept as-is.

---

## Part 3 — Target architecture

```
┌─ L4  HARNESS ─────────────────────────────────────────────────────────┐
│  eval runner · golden sets · regression diff · tracing hook · fixtures │
├─ L3  PIPELINES ───────────────────────────────────────────────────────┤
│  ingest()  ·  ask()  ·  extract()  ·  resolve()                        │
│  each one composable, each one wrappable in durable_steps              │
├─ L2  ALGORITHMS (own) ────────────────────────────────────────────────┤
│  doc_layout · chunking · hybrid_ranker · entity_resolution             │
│  extraction repair loop · minhash/LSH · governor+cache · async map     │
├─ L1  PORTS (own contracts, borrowed implementations) ─────────────────┤
│  DocumentSource · Embedder · LLM · VectorStore · LexicalIndex          │
│  Reranker · Crawler · Transcriber                                      │
├─ L0  CORE CONTRACTS (own) ────────────────────────────────────────────┤
│  Document · Block · Chunk · Provenance · RankedList · Entity · Run     │
│  Error taxonomy · Result/Usage types                                   │
└───────────────────────────────────────────────────────────────────────┘
```

**The rule that keeps it honest:** L2 may import L0 and L1 *contracts* only. No
component ever imports a vendor SDK. Every borrow lives in one adapter file under L1,
and `pip install toolkit` with no extras must still import cleanly.

**Dependency extras**, so the base install stays tiny:

```
toolkit            -> stdlib only (L0, L2 algorithms, the four existing components)
toolkit[docs]      -> docling, pdfplumber
toolkit[ocr]       -> paddleocr / rapidocr
toolkit[llm]       -> litellm
toolkit[embed]     -> fastembed
toolkit[store]     -> lancedb, bm25s
toolkit[eval]      -> ragas, deepeval
toolkit[web]       -> crawl4ai
toolkit[audio]     -> faster-whisper
```

---

## Part 4 — Build order

Sequenced so that every phase ends with something demonstrable, and so the spine
exists before anything is hung on it.

### Phase 0 — Make it a package — **DONE**

`pyproject.toml` with optional extras, ruff + mypy config, pytest config. The four
components moved under the `toolkit` package. Editable install verified; `import
toolkit` pulls in nothing outside the standard library.

*Verified:* `pip install -e .` works, `python -m pytest toolkit/tests -q` passes,
`ruff check toolkit` is clean.

### Phase 1 — The spine — **DONE**

L0 contracts (`Document`, `Block`, `Chunk`, `Provenance`, `Usage`, `Completion`,
`SearchHit`, error taxonomy) and six ports as `Protocol`s, each with **two or more
working adapters**:

| Port | stdlib implementation | real backend |
|---|---|---|
| DocumentSource | `PlainTextSource` | `PdfPlumberSource`, `DoclingSource` |
| Embedder | `HashingEmbedder` | `FastEmbedEmbedder` |
| VectorStore | `InMemoryVectorStore` | `LanceDBStore` |
| LexicalIndex | `SqliteFtsIndex` (FTS5) | `Bm25sIndex` |
| LLM | `ScriptedLLM` | `LiteLLMClient` |
| Reranker | `LexicalOverlapReranker` | `LLMReranker`, `CrossEncoderReranker` |

`toolkit/tests/test_contracts.py` runs every adapter for a port through the *same*
assertions and refuses to let a port be registered with fewer than two. It generates
its own minimal PDF, so the PDF path is exercised with no checked-in fixture.

*Verified:* the same `document_source_contract` passes on pdfplumber and on Docling —
the swap this phase exists to enable.

### Phase 1b — **DONE**

- **CI** (`.github/workflows/ci.yml`): two jobs. The first installs *no* optional
  backends and asserts that importing the toolkit pulls in none of them, so the
  stdlib-only promise is verified rather than assumed; the second installs the
  backends so the adapters that SKIP in the first are exercised somewhere.
- **mypy** clean across all 42 source files, not just the contracts — the scope was
  widened once the seven errors it found were fixed.
- **Reranker adapters**, closing the port that violated this document's own rule:
  `LexicalOverlapReranker` (stdlib, BM25 over the candidate set), `LLMReranker`
  (one call for the whole set), `CrossEncoderReranker` (sentence-transformers).

Four things this turned up:

- `FieldComparison.field` shadowed the `dataclasses.field` import. It works at
  runtime — an annotation without assignment does not bind — but it is a trap for
  any reader and mypy flagged it. Aliased the import.
- The lexical reranker scored **zero** on "refund" against "refunds". Not an edge
  case; most real queries. Added conservative suffix stripping.
- `ChunkConfig.overlap_tokens` and `min_tokens` are `int | None` because None means
  "derive from max_tokens", which forced every call site to re-handle None. Added
  resolved `.overlap` / `.minimum` accessors.
- Two `self._llm.complete(...)` call sites were guarded by a caller-side `is not
  None` that the type checker could not see. Guarded locally instead.

### Phase 2 — Chunking + cache + governor — **DONE**

- `chunking/`: structure-aware, provenance-carrying, heading breadcrumbs, whole-
  sentence overlap, word-split fallback for oversized units, and a final
  `_enforce_budget` pass that measures the *rendered* chunk instead of trusting a sum
  of per-unit estimates. Optional semantic boundaries via the `Embedder` port.
- `cache/`: `SqliteCache` + `CachedEmbedder` (per-text keys) + `CachedLLM`
  (whole-request keys, temperature bypass, `cached=True` zero-cost usage).
- `governor/`: pre-flight budget projection, sliding-window rate limit, selective
  retry honouring `Retry-After`, injectable clock for testable timing.
- `concurrency.py`: `bounded_map` (order-preserving, index-attributed failures,
  selective retry), `batched`, `embed_all`.

*Verified:* 32 tests. Re-running an ingestion costs zero embedding calls for
unchanged text; every chunk names its page and bbox.

Three defects the tests caught during construction: the token budget was enforced on
per-unit sums while the shipped chunk is the rendered string (they disagree once
separators and breadcrumbs are added), and both `overlap_tokens` and `min_tokens` had
fixed defaults that contradicted any lowered `max_tokens` — now derived from it.

### Phase 3 — First real pipeline: `ingest()` + `ask()` — **DONE**

`pipelines/KnowledgeBase` wires `DocumentSource` → `doc_layout` → `chunking` →
`Embedder` → `VectorStore` + `LexicalIndex` → `hybrid_ranker` → cited answer, with
each document ingested as one durable step when `durable_db` is set.

*Verified:* `examples/quickstart.py` runs offline with nothing installed; 23 pipeline
tests. Every citation resolves to a chunk, a page and a bbox.

Added beyond the plan, because the tests and the demo forced it:

- **Citation verification.** Every `[n]` is checked against the sources the model was
  shown; invented markers land in `unverified_markers` and never render as citations.
- **A relevance gate.** Rank fusion always returns something from a non-empty index,
  so without one the knowledge base answered an off-topic question confidently. Found
  by running the demo, not by a test.
- **The gate is term-overlap based, not a cosine floor.** Measurement killed the first
  design: under the default embedder an irrelevant query scored 0.293 against a
  relevant one's 0.248, so no absolute threshold separates them.
  `min_dense_similarity` is opt-in with the calibration procedure documented.
- **An extractive mode** for when no LLM is configured — and the right way to debug
  retrieval.

Also fixed: `KnowledgeBase(lexical_index=None)` could not actually disable the keyword
index, because `None` was indistinguishable from "not supplied". Needed a sentinel.

### Phase 4 — Structured extraction (2 days)

Schema (Pydantic) → prompt → parse → validate → **repair on failure with the specific
validation error** → bounded retries → result with per-field provenance and confidence.
Plus document splitting/classification for the "this 40-page scan is really 6
documents" case. Optional LLGuidance backend for local models.

*Done when:* a contract PDF yields a validated object and you can click any field back
to the page region it came from.

### Phase 5 — Eval harness — **DONE**

`evaluation/` with own metrics (hit_rate, precision@k, nDCG, MRR, MAP), a runner over
any object shaped like `KnowledgeBase`, JSONL golden sets, JSON report persistence,
and `diff_reports` naming the cases a change fixed or broke.

*Verified:* 20 tests, metric values hand-computed rather than checked against the
implementation. `examples/evaluate.py` shows the gate experiment moving
`refusal_accuracy` 1.000 → 0.000 and naming both broken cases.

Three decisions that came out of building it:

- **Ground truth is never chunk ids.** They change whenever chunking changes, which is
  the experiment you most want to run — a golden set keyed on them invalidates itself
  the moment it becomes useful. Relevance is judged on snippets, pages and doc ids.
- **Unanswerable cases are first-class, and aggregated separately.** Without them the
  relevance gate is unmeasurable; mixed in with answerable cases, the gate trade
  cancels out to "nothing changed".
- **`answer_match` is absent rather than 0.000 when nothing was checked.** Found by
  running the example: a metric that reads as total failure when it was never measured
  is worse than no metric.

The example also warns when the baseline scores 1.000, because a diff of all zeros on
a saturated benchmark is a fact about the golden set, not evidence a change is safe.

### Phase 4 — Structured extraction — **DONE**

`extraction/` with `ExtractionComponent` (schema → prompt → parse → coerce → validate
→ targeted repair → ground) and `DocumentSplitterComponent` (page-number restarts,
repeated first-page headings, optional LLM labelling).

*Verified:* 24 tests. `examples/extract.py` shows a three-defect first attempt
converging in one repair round, with four of five fields located back to a bbox.

Decisions that came out of building it:

- **Local coercion before repair.** `"$1,240.50"` → `1240.50` and `"yes"` → `True`
  cost nothing; spending a network round-trip on formatting does.
- **Ambiguous dates are refused, not guessed.** `01/02/2024` is two different dates
  depending on the document's origin, so it becomes a repair rather than silent
  corruption. Unambiguous forms are rewritten locally.
- **Grounding exempts normalised types.** Found by a failing test: `require_grounding`
  flagged a correct `"2024-03-14"` because the source says "Issued 14 March 2024". A
  signal that fires on correct work is worse than no signal.
- **Grounding needed a separator-stripped view** of each passage, or `1240.5` against
  "$1,240.50" reads as fabricated — noise exactly where the signal matters most.

Not built: the LLGuidance backend. With a local model it would mostly remove the need
for the repair loop, but it is a dependency-level integration rather than a component,
and the repair loop is what a hosted model needs regardless.

### Phase 6 — Reach, as needed (ongoing)

MinHash/LSH · table adapters · crawler adapter · ASR adapter · tracing hook ·
vision adapter. One adapter at a time, each ~half a day, each driven by a real project
rather than by completeness.

**Total to a genuinely powerful kit: about 10 working days.** Phases 0–3 (4.5 days)
already give you something better than what most teams walk into a hackathon with.

---

## Part 5 — The hackathon layer

Separate from the architecture, and the thing you will actually feel at 2am:

- **`quickstart.py`** — one import, one call, a working RAG over a folder of PDFs.
  Defaults that need no API key: FastEmbed on CPU, LanceDB on disk, bm25s.
- **Offline mode.** Every default has a zero-network path. Conference wifi is a
  known adversary, and a demo that needs a hosted embedding API is a demo that dies.
- **Cache warm from disk.** Re-running after a crash must not re-pay for embeddings.
- **`FakeLLM` / `FakeEmbedder`.** Deterministic fixtures so pipelines are testable and
  demos are reproducible.
- **A recipes folder** — five copy-paste starting points: RAG over PDFs, extract to
  schema, dedupe a messy CSV, transcribe and search audio, crawl and index a site.

---

## Part 6 — Risks, stated plainly

| Risk | Mitigation |
|---|---|
| **Abstraction built on one implementation.** The most likely failure of this whole plan: ports that are really just LiteLLM's interface renamed. | Two adapters per port before the port is considered done. Non-negotiable. |
| **Scope creep into a framework.** The moment it has a config system and a plugin registry, it is LangChain and you will not maintain it. | Components stay independently copyable. No component may require the package to function. |
| **Licence contamination.** Marker/MinerU/Surya are easy to reach for under time pressure. | The borrow table above is the standing decision. Check weights separately from code. |
| **The four existing components rot.** They have no consumers yet. | Phase 3 makes two of them load-bearing. If a component has no pipeline using it by Phase 6, it is a candidate for deletion. |
| **Eval arrives too late to matter.** | Phase 5 is the last one that can slip. Everything after it is adapters, which are low-risk. |

---

## The one-line version

Own the **contracts, the chunking, the repair loop, the cache and the eval harness**.
Borrow **everything with weights or a server**. Build the spine before the reach, and
make every port prove itself against two implementations before you trust it.

---

### Sources

Research current as of September 2026.

- OCR benchmarks and licences: [Roboflow OCR rankings](https://blog.roboflow.com/best-open-source-ocr-models/) · [Spheron self-host guide](https://www.spheron.network/blog/best-open-source-ocr-vlm-self-host-gpu-cloud-2026/)
- Parser licence comparison: [particula.tech](https://particula.tech/blog/docling-vs-mineru-vs-marker-pdf-parser) · [PDF-to-markdown deep dive](https://jimmysong.io/blog/pdf-to-markdown-open-source-deep-dive/)
- Agent frameworks: [LangChain framework roundup](https://www.langchain.com/resources/ai-agent-frameworks) · [2026 decision guide](https://dev.to/linou518/the-2026-ai-agent-framework-decision-guide-langgraph-vs-crewai-vs-pydantic-ai-b2h)
- Eval frameworks: [DeepEval comparison](https://deepeval.com/blog/top-5-llm-evaluation-frameworks) · [promptfoo vs DeepEval vs Ragas](https://qaskills.sh/blog/promptfoo-vs-deepeval-vs-ragas-2026)
- Tables: [Camelot comparison](https://camelot-py.readthedocs.io/en/latest/user/comparison.html) · [table-transformer (archived Sep 2026)](https://github.com/microsoft/table-transformer)
- Chunking: [Chonkie](https://github.com/chonkie-inc/chonkie)
- Web extraction: [Crawl4AI](https://github.com/unclecode/crawl4ai) · [best open-source crawlers](https://www.firecrawl.dev/blog/best-open-source-web-crawler)
- Observability: [Langfuse](https://github.com/langfuse/langfuse) · [open-source LLM observability roundup](https://www.turingpost.com/p/llm-observability)
- ASR: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) · [WhisperX](https://github.com/m-bain/whisperx)
