# Reusable Engineering Toolkit

> **Status:** 21 units of a 25 cap, across five layers. The spine (contracts,
> ports, adapters), ingestion (chunking, cache, governor, concurrency), the
> `KnowledgeBase` pipeline with a saveable index, structured extraction,
> injection defense, graph and DAG execution, the evaluation harness, a CLI,
> and bring-your-own-key generation with no vendor SDK.
>
> It has been run against real input once: 49 uncurated arXiv PDFs, which found
> nine defects that 474 tests and a hand-built hostile corpus had all missed.
> See [`validation/paper_triage/FINDINGS.md`](../validation/paper_triage/FINDINGS.md).
> Every unit is still `documented` rather than `production_ready`; that gate
> needs use in a real project, not a test count.
>
> [ROADMAP.md](ROADMAP.md) is kept as the original plan and its outcome;
> [`../CHANGELOG.md`](../CHANGELOG.md) is the running record.
>
> ```python
> from toolkit.pipelines import KnowledgeBase
>
> kb = KnowledgeBase()            # offline defaults, no API key
> kb.ingest_folder("./pdfs")
> answer = kb.ask("what is the voltage limit?")
> for c in answer.citations:
>     print(c.source_uri, "page", c.page, c.bbox)
> ```
>
> `python examples/quickstart.py` runs this end to end with a built-in sample corpus.
>
> ```
> pip install -e .            # stdlib only, works offline
> pip install -e ".[all]"     # optional backends
> python -m pytest toolkit/tests -q
> ```
>
> **Architecture rule:** only `toolkit/adapters/` may import a vendor SDK, and every
> such import is lazy. `import toolkit` pulls in nothing outside the standard library.
> Every port is validated against two or more implementations by
> `toolkit/tests/test_contracts.py`, which fails if a port drops below two.
>
> **Ports and their adapters** (stdlib implementation first — that one is why the
> whole toolkit runs offline):
>
> | Port | stdlib | real backends |
> |---|---|---|
> | `DocumentSource` | `PlainTextSource` | `PdfPlumberSource`, `DoclingSource` |
> | `Embedder` | `HashingEmbedder` | `FastEmbedEmbedder` |
> | `VectorStore` | `InMemoryVectorStore` | `LanceDBStore` |
> | `LexicalIndex` | `SqliteFtsIndex` | `Bm25sIndex` |
> | `LLM` | `ScriptedLLM` | `LiteLLMClient` |
> | `Reranker` | `LexicalOverlapReranker` | `LLMReranker`, `CrossEncoderReranker` |
>
> CI runs the suite twice: once with **no** optional backends, asserting none leak
> into the base import, and once with them installed so every adapter is exercised.

Four standalone components, extracted as *algorithms* from strong open-source
projects and rebuilt with no application coupling. Each is a directory you copy into
a project. Each exposes one class with one method: `execute(input_data) -> output`.

**Zero third-party dependencies across all four.** Standard library, Python 3.10+.

| Component | Problem it removes | Extracted from |
|---|---|---|
| [`doc_layout/`](doc_layout/README.md) | Reading order, heading hierarchy, OCR duplicate spans, page furniture | XY-cut (Nagy 1984), Docling layout post-processing, Marker heading heuristics |
| [`chunking/`](chunking/README.md) | Chunks that can't be cited; sections packed together; severed sentences | Own work — provenance-carrying chunking has no upstream |
| [`hybrid_ranker/`](hybrid_ranker/README.md) | Merging incomparable retriever scores; spending a cross-encoder budget wisely | RRF (Cormack 2009), Qdrant/Weaviate fusion, ColBERT/SPLADE cascades |
| [`entity_resolution/`](entity_resolution/README.md) | Matching records with no shared key, without labels, explainably | dedupe (blocking cover, affine gap), Splink (Fellegi–Sunter + EM) |
| [`cache/`](cache/README.md) | Re-paying for every embedding and completion on each run | Own work — per-text keys, not per-batch |
| [`governor/`](governor/README.md) | Budget overruns, rate-limit refusals, retrying failures that can't succeed | Temporal retry shape + own sliding window and pre-flight budget |
| [`durable_steps/`](durable_steps/README.md) | Multi-step pipelines that must survive a crash without redoing work | DBOS Transact, Hatchet leasing, Temporal replay semantics |
| `concurrency.py` | Batch maps that reorder results or fan out unbounded | Own work |
| [`pipelines/`](pipelines/README.md) | `ingest()` + `ask()` with verified citations and a relevance gate | Own work — the wiring |
| [`extraction/`](extraction/README.md) | Document → validated object, repaired on the *specific* error, every field traceable to a bbox | Own work; shape common to ExtractThinker/Instructor |
| [`evaluation/`](evaluation/README.md) | "Did that change help?" answered with a table and a list of broken cases | Own harness; IR metrics (MRR, nDCG, MAP) |

**The spine** (`core/`, `ports.py`, `adapters/`) is what makes them compose: one
`Document` model, six ports, and 11 adapters — at least two per port.

## Layout

Twenty-one units across five layers. Each package directory carries the same five
files — `component.py` or equivalent, `models.py`, `__init__.py`, `README.md`,
`requirements.txt` — so a unit can be copied out whole.

```
toolkit/
├── core/                L0  contracts, error taxonomy, text normalisation
├── ports.py             L0  seven protocols, two implementations each
├── cache/               L1  content-addressed model-call cache
├── governor/            L1  budget, rate limit, selective retry
├── concurrency.py       L1  order-preserving bounded parallel map
├── durable_steps/       L1  crash-resumable steps, leasing
├── provider/            L1  find a key, endpoint and model      (standalone)
├── adapters/            L2  the only place a vendor SDK is imported
├── doc_layout/          L2  reading order, headings, furniture   (standalone)
├── chunking/            L2  provenance-carrying chunks
├── hybrid_ranker/       L2  RRF fusion, rerank cascade, MMR      (standalone)
├── entity_resolution/   L2  blocking, Fellegi-Sunter + EM        (standalone)
├── guardrails/          L2  indirect prompt-injection defense    (standalone)
├── extraction/          L2  schema repair loop, grounding
├── graph/               L2  deterministic graph algorithms       (standalone)
├── dag/                 L2  DAG execution on a ready queue
├── llm_http/            L2  the LLM port over plain HTTP, no SDK
├── pipelines/           L3  KnowledgeBase: ingest, ask, save/load
├── evaluation/          L4  golden sets, IR metrics, regression diff
├── cli.py               L4  ingest / ask / eval / inspect
└── tests/               thirteen suites
```

`(standalone)` means `copy_tier: standalone` — the directory can be copied out
on its own and imported, which `test_packaging.py` verifies by doing exactly
that in a subprocess with the repository off `sys.path`.

## Every unit has a runnable snippet

```bash
python examples/cookbook.py --list        # 21 unit snippets + 9 recipes
python examples/cookbook.py chunking      # just one
```

`REGISTRY.json` records, per unit, which snippet demonstrates it and which recipes
compose it. `toolkit/tests/test_packaging.py` asserts those references point at
functions that exist, so neither can be claimed without being true — and CI executes
the whole cookbook, so a snippet that rots fails the build.

## Validation

```
python toolkit/tests/test_toolkit.py      # standalone runner, no pytest needed
python -m pytest toolkit/tests -q         # or via pytest
```

27 behavioural tests, all passing. They assert the properties that make each
component worth having, not just that the code runs:

- RRF ranks a dual-retriever hit above a single-retriever top hit, across
  incomparable score scales.
- A rerank budget of 1 provably cannot reorder beyond the head.
- MMR suppresses a near-duplicate that plain relevance would keep.
- Affine-gap scores one clean omission above the same number of scattered typos.
- Average linkage refuses a chain that connected components accepts.
- A two-column page reads left column fully before right.
- A table gutter is not mistaken for a column break.
- A killed run resumes at the failed step and re-runs nothing that completed —
  verified across a fresh store object over the same database file.

Two real bugs were found and fixed by these tests during construction: line assembly
was splicing left- and right-column text at matching y coordinates (column
decomposition has to run *before* line grouping), and the column-width guard was
loose enough to promote a table gutter to a column break.

## Licensing

No source was copied from any upstream project — every component was reimplemented
from published algorithms and documented behaviour. This was a deliberate constraint,
because the strongest document parsers are copyleft: Marker is GPL-family with a
revenue threshold on its weights, and MinerU is AGPL. Both are excellent and worth
reading; neither can be vendored into a permissive toolkit. The permissively licensed
prior art (Docling, Splink, dedupe, DBOS, Temporal — all MIT) is credited in each
component's README under *Original Source*.

Verify licences yourself before commercial use; parser-family licensing has been
shifting, and model weights routinely carry terms separate from the code.

## What was deliberately not extracted

| Candidate | Why not |
|---|---|
| Docling layout/table models | Model weights are not a component. Use Docling as a dependency and feed its spans into `doc_layout`. |
| MinerU pipeline | AGPL. Read it for the region-routing ideas; do not vendor it. |
| LLGuidance constrained decoding | Research-grade Rust (Earley parser over regex derivatives). Depend on it, never reimplement it. |
| ColBERT PLAID index | Compiled extensions and research-organised code. Use PyLate if you want late interaction. |
| Hatchet engine | Infrastructure, not a component. The extractable idea is the lease, which `durable_steps` has. |

## Composition

The four are independent but designed to chain:

```
PDF ─▶ your extractor ─▶ doc_layout ─▶ chunks with (page, bbox) provenance
                                            │
                                            ▼
                     embed + BM25 ─▶ hybrid_ranker ─▶ cited answers
                                            │
        every stage above wrapped in ─▶ durable_steps (crash-resumable ingestion)

        messy records from several sources ─▶ entity_resolution ─▶ deduplicated entities
```
