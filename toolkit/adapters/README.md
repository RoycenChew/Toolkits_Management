# Adapters

**Layer 2 · depends on `core` + `doc_layout` · `copy_tier: needs_package`**

## What It Does

The only place in the toolkit a vendor SDK may be imported. Fourteen adapters
implementing six ports, with **at least two per port** — a rule enforced by a
failing test, not a convention.

## Why It Is Useful

A port validated against a single backend is just that vendor's interface with the
names renamed, and you discover this at the worst possible moment: when you try to
swap it. Two implementations per port is what makes the abstraction real.

Every vendor import is **lazy**, inside the method that needs it. `import toolkit`
therefore pulls in nothing outside the standard library — a property CI asserts on
every push, and which caught a real defect on its first run.

## The adapters

Stdlib implementation first in each row. That one is why the whole toolkit runs
offline with no API key.

| Port | stdlib | real backends |
|---|---|---|
| `DocumentSource` | `PlainTextSource` | `PdfPlumberSource`, `DoclingSource` |
| `Embedder` | `HashingEmbedder` | `FastEmbedEmbedder` |
| `VectorStore` | `InMemoryVectorStore` | `LanceDBStore` |
| `LexicalIndex` | `SqliteFtsIndex` | `Bm25sIndex` |
| `LLM` | `ScriptedLLM` | `LiteLLMClient` |
| `Reranker` | `LexicalOverlapReranker` | `LLMReranker`, `CrossEncoderReranker` |

**The stdlib adapters are not merely test doubles:**

- `HashingEmbedder` — hashed n-gram bag with signed buckets. No semantics, but
  deterministic, offline, and a real signal for lexically similar text.
- `SqliteFtsIndex` — BM25 through SQLite FTS5, which ships with CPython. Supports
  incremental indexing, which `Bm25sIndex` does not.
- `InMemoryVectorStore` — exact brute-force cosine. Under ~50k chunks it is faster
  than an ANN index *and* exact, which removes a variable while debugging retrieval.
- `ScriptedLLM` — deterministic, records every prompt in `.calls`. Lets you debug a
  repair loop or an agent's control flow without spending a token. Several tests
  assert on what the pipeline actually sent, which is usually the thing that's wrong.
- `LexicalOverlapReranker` — BM25 scored over *the candidate set itself*, so its IDF
  comes from the candidates rather than the corpus: a genuinely different signal from
  a global-IDF first stage.

## Installation

```bash
pip install -e ".[docs]"     # pdfplumber, docling
pip install -e ".[embed]"    # fastembed
pip install -e ".[store]"    # lancedb, bm25s
pip install -e ".[llm]"      # litellm
pip install -e ".[rerank]"   # sentence-transformers
pip install -e ".[all]"
```

This unit depends on `core` and `doc_layout`, so it is **not** a copy-one-folder
component — install the package. A missing optional package raises
`MissingDependency` naming the exact extra, never a bare `ImportError`.

## Notable behaviour

| Adapter | Worth knowing |
|---|---|
| `PdfPlumberSource` | Reads `top`/`bottom` (already y-down). Screens size **and page count** — a 2 KB file can declare 40,000 pages, so a size cap alone does not bound the work. Raises `AdapterError` naming OCR when there is no text layer. Keeps every word on `Document.words` with its own box, in block reading order |
| `DoclingSource` | Traverses via `iterate_items()` or `.texts` because the API moved between versions. Normalises bbox to y-down by ordering rather than trusting field names. Screens size only |
| `PlainTextSource` | Line-oriented, because Markdown needs no blank line after a heading — paragraph-first splitting silently destroyed every heading |
| `HashingEmbedder` | `model_version` encodes dimension *and* trigram setting, so a config change is as detectable as a model change |
| `LanceDBStore` | Delete-then-add for upsert (portable across versions); `delete_by_doc` pushes the exclusion into the predicate rather than round-tripping ids |
| `Bm25sIndex` | Rebuilds the whole index on any write. Suits ingest-then-query, not continuous deletion — use `SqliteFtsIndex` for that |
| `LiteLLMClient` | Reads the response through a `Mapping`/attribute shim because providers differ. Translates rate limits to `RateLimited` so the governor need not string-match |
| `CrossEncoderReranker` | Accepts an injected `model`, which is how its normalisation logic is tested without the dependency |

All adapters normalise text through `core.normalise_text` at the boundary: NFKC plus
removal of zero-width and soft hyphens. A ligature renders identically to the letters
it replaces, so `conﬁguration` is unsearchable as "configuration" and the bug is
invisible by eye.

## Usage

```python
from toolkit.adapters import HashingEmbedder, PlainTextSource, load_document

doc = load_document("notes.md")          # dispatches to the first source that claims it
vectors = HashingEmbedder(256).embed([b.text for b in doc.content_blocks()])
```

Swapping a backend changes one constructor argument and nothing else:

```python
from toolkit.adapters import FastEmbedEmbedder, LanceDBStore
from toolkit.pipelines import KnowledgeBase

kb = KnowledgeBase(embedder=FastEmbedEmbedder(), vector_store=LanceDBStore("./.lancedb"))
```

## Limitations

- **Lazy imports mean failures surface at call time, not construction.**
  `LiteLLMClient()` succeeds without litellm installed; the error appears on
  `complete()`. Deliberate — constructing a client should not require the network or
  the package — but it means a missing dependency can hide until the first call.
- `DoclingSource` cannot screen page count without opening the file twice, so it
  screens size only. Pre-screen with `PdfPlumberSource` when the input is a PDF.
- No OCR adapter yet. `DoclingSource` can OCR internally; PaddleOCR-VL and dots.ocr
  are the recommended borrows and are unimplemented.
- `LanceDBStore` converts L2 distance to `1/(1+d)` so larger is better. That is
  monotonic but not comparable to cosine from another store — fuse with RRF, which
  needs no score calibration, rather than raw scores.
- `ScriptedLLM`'s response queue is consumed; use `handler=` for anything called more
  than once. A contract test was silently broken by exactly this.
- No async variants anywhere.

## Integration Guide

1. **Write the span adapter first and verify one page visually** before trusting
   anything downstream. Print `result.markdown` and compare against the source.
2. Keep every vendor import lazy and inside the method. CI fails the build if one
   leaks into module scope.
3. Adding a backend means adding an **adapter**, never a competing component — and
   the port's contract test must pass unchanged.
4. Pass `model_name` explicitly in production. Several adapters default it to the
   class name, which collides when two models sit behind one class.

## Extraction Notes

- **Preserved:** nothing copied. Each adapter is a thin translation to its vendor's
  published API.
- **Added:** the stdlib implementation for every port, which is what makes the
  offline path real; `model_version` on embedders; delete semantics on both store
  ports; and `MissingDependency` carrying the install command.
- **Isolated:** this is the containment boundary. Every other unit imports ports, not
  SDKs, which is why `import toolkit` stays stdlib-only and why backends are
  swappable at all.
