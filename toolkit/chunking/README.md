# Chunker Component

## What It Does

Turns a `Document` (ordered blocks with page + bbox geometry) into retrievable
`Chunk`s that **carry that geometry through**, so any chunk can name the page and
region it came from.

## Why It Is Useful

Chonkie, LangChain and LlamaIndex all chunk *text*: a string goes in, strings come
out, and which page the text came from is gone. Nothing in the ecosystem chunks
*blocks that carry geometry* and propagates it.

That propagation is the entire point. A chunk that knows it came from page 7 at a
specific bounding box can be cited, highlighted and audited. A chunk that is just a
string can only be trusted.

Three decisions here matter more than the token arithmetic:

- **Sections beat budgets.** A heading is the author telling you where a topic
  starts. Packing across it to fill a 512-token quota discards that for nothing.
- **Heading breadcrumbs are prepended.** `"must not exceed 40V"` is useless alone;
  `"Safety Manual > Voltage"` attached to it is not. This repairs the most common
  RAG failure mode for one line of cost.
- **Overlap is in whole sentences.** Overlapping by raw token count severs
  sentences, and a half-sentence embeds to noise.

## Original Source

Own work, informed by [Chonkie](https://github.com/chonkie-inc/chonkie)'s chunker
taxonomy (Token / Sentence / Recursive / Semantic / SDPM / Late). No code copied —
and the provenance-carrying behaviour has no upstream to copy from.

## Architecture

```
INPUT     Document (blocks with BlockType, level, Provenance)
   |
UNITS     walk blocks, maintain heading stack by level, split into sentences
   |       oversized units word-split so nothing exceeds the budget or is lost
   |
SEMANTIC  optional: embed units, force a break where cosine similarity dips
   |
PACK      greedy fill to max_tokens; hard break at headings; sentence overlap
   |
MERGE     fold undersized trailing chunks back (never across a section)
   |
ENFORCE   re-measure the *rendered* text; re-split anything still over budget
   |
OUTPUT    Chunk[] with provenances (one region per page), heading_path, hashes
```

```
stdlib only (Embedder optional)  ->  Component  ->  Chunk[]
```

## Installation

Copy **two** directories, because this unit imports the shared contracts:

```bash
cp -r toolkit/chunking  your_project/
cp -r toolkit/core      your_project/
```

Both must sit under the same parent package so the relative import resolves.
Python 3.10+. Standard library only.

Earlier versions of this README said "copy the `chunking/` directory" full stop,
which does not work — `toolkit/tests/test_packaging.py` now copies each unit out
with its declared dependencies and imports it in a subprocess, so the instruction
is verified rather than asserted.


## Dependencies

Standard library. An `Embedder` is optional and only needed for
`semantic_threshold`.

## Input Schema

`ChunkRequest(document, config)` where `ChunkConfig` has:

| Field | Default | Meaning |
|---|---|---|
| `max_tokens` | 512 | hard ceiling per chunk |
| `min_tokens` | `max_tokens//8`, ≤48 | below this, merge back into predecessor |
| `overlap_tokens` | `max_tokens//4`, ≤64 | sentence overlap between chunks |
| `split_on_heading` | `True` | start a new chunk at every heading |
| `include_heading_path` | `True` | prepend the breadcrumb |
| `keep_furniture` | `False` | include page headers/footers |
| `token_counter` | `None` | supply `tiktoken` etc. for exact counts |
| `semantic_threshold` | `None` | 0..1; needs an embedder |

`min_tokens` and `overlap_tokens` derive from `max_tokens` rather than being fixed,
so lowering `max_tokens` alone can never produce a self-contradictory config.

## Output Schema

`ChunkResult`:
- `chunks`: `Chunk(chunk_id, text, doc_id, index, provenances, metadata)`.
  `metadata` carries `heading_path`, `block_types`, `token_estimate`,
  `content_hash`, `source_uri`.
- `oversized`: chunk_ids still over budget (a single unsplittable unit).
- `token_estimates`: chunk_id → token count.

`chunk_id` is `doc_id#index` — a stable slot, so re-ingesting an edited document
**overwrites** slot N rather than leaving an orphan row in your vector store. The
content hash lives in metadata for change detection.

## Usage

```python
from toolkit.adapters import PdfPlumberSource
from toolkit.chunking import ChunkConfig, ChunkerComponent, ChunkRequest

document = PdfPlumberSource().load("manual.pdf")
result = ChunkerComponent().execute(
    ChunkRequest(document, ChunkConfig(max_tokens=384))
)

for chunk in result.chunks:
    pages = chunk.pages
    box = chunk.provenances[0].bbox
    print(chunk.chunk_id, "pages", pages, "at", box.as_tuple())
    print(chunk.text[:120])
```

Exact token counts and semantic boundaries:

```python
import tiktoken
from toolkit.adapters import FastEmbedEmbedder

encoder = tiktoken.get_encoding("cl100k_base")
config = ChunkConfig(
    max_tokens=512,
    token_counter=lambda text: len(encoder.encode(text)),
    semantic_threshold=0.55,
)
result = ChunkerComponent(embedder=FastEmbedEmbedder()).execute(
    ChunkRequest(document, config)
)
```

## Limitations

- **Tables are chunked as text.** A `BlockType.TABLE` block is kept whole where it
  fits and word-split where it does not. Neither preserves row/column structure —
  that needs a table-aware serialisation upstream.
- Sentence splitting is regex plus an abbreviation blocklist, not a model. It
  handles `Dr.`, `approx.`, `e.g.` and the common cases; exotic punctuation and
  languages without Latin sentence-final punctuation will under-split.
- The default token counter is a heuristic (`max(chars/4, words*1.3)`), deliberately
  biased to overestimate. Supply `token_counter` when the budget is a hard API limit.
- `semantic_threshold` costs one embedding per sentence. On a large corpus that is
  the dominant cost of ingestion.
- Heading levels come from whatever the `DocumentSource` inferred. A document that
  signals structure only by numbering ("3.1.2") gives a flat breadcrumb.
- Single-pass and synchronous. Use `toolkit.concurrency.bounded_map` to chunk many
  documents in parallel.

## Integration Guide

1. Read `result.oversized` on your first real corpus. A non-empty list almost always
   means tables or code blocks that want different handling, not a bad budget.
2. Carry `chunk.provenances` into your vector store payload. Without it you cannot
   build the citation UI later, and retrofitting it means re-ingesting everything.
3. Set `max_tokens` from your *embedding* model's window, not your LLM's. That is
   the limit that actually truncates.
4. Leave `split_on_heading=True` unless your documents have no headings at all.
5. Sanity-check one document's `heading_path` values before ingesting thousands —
   if the breadcrumbs are wrong, every chunk is subtly mislabelled.

## Extraction Notes

- **Preserved:** the standard greedy-pack-with-overlap approach, and Chonkie's
  distinction between token, sentence and semantic strategies.
- **Removed:** nothing — written from scratch.
- **Added, and the reason this exists:** provenance propagation, per-page region
  merging, heading-stack breadcrumbs, the `_enforce_budget` pass that measures the
  *rendered* chunk rather than trusting a sum of per-unit estimates, and
  budget-relative defaults so partial config overrides cannot contradict themselves.
- **Isolated:** the embedder is duck-typed, so no model or vendor import appears in
  this component.
