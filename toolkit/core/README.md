# Core Contracts

**Layer 0 · imports nothing · `copy_tier: standalone`**

## What It Does

Defines the types every other unit speaks: `Document`, `Block`, `Chunk`,
`Provenance`, `BBox`, `Usage`, `Completion`, `SearchHit`, `Message`, the screening
limits, and the error taxonomy.

No algorithms, no I/O, no vendor imports. Data contracts and nothing else.

## Why It Is Useful

Every framework ships its own `Document`. Owning one here is what lets a pipeline
swap pdfplumber for Docling for an OCR engine **without a line of pipeline code
changing** — a property the contract tests verify by running the same assertions
against three different `DocumentSource` adapters.

This is the highest-leverage unit in the toolkit, and the cheapest to get wrong
permanently. Two decisions are load-bearing:

**Provenance is part of the core, not an optional extra.** A chunk that cannot say
which page and which region it came from cannot be cited, highlighted or verified —
and an answer you cannot verify is a demo rather than a system. Retrofitting
provenance into a shipped RAG system means re-ingesting everything.

**`BlockType` is deliberately small.** A vocabulary only one backend can populate is
not a shared contract. Backends that distinguish more categories map down; backends
that distinguish fewer leave everything `PARAGRAPH`.

## Architecture

```
BBox          axis-aligned region, y-down. Adapters normalise before constructing.
Provenance    page + bbox + source id. merge() refuses to span a page break.
Block         text + BlockType + level + provenance
Document      doc_id (content hash) + ordered blocks + page_count + metadata
Chunk         chunk_id (doc_id#index slot) + text + provenances[] + metadata
Usage         input/output tokens + cost + cached flag. Summable.
Message       role + content          Completion   text + usage + model
SearchHit     chunk_id + score + text. Score scale is backend-specific by design.
ScreeningLimits / ScreeningResult / ScreeningFailure    parse guards
ToolkitError  -> MissingDependency | AdapterError -> RateLimited | ValidationFailed
                 | ScreeningRejected
```

## Installation

Standalone — the only unit with no toolkit dependencies at all:

```bash
cp -r toolkit/core your_project/
```

Python 3.10+. Standard library only.

## Input / Output

Not a component; it has no `execute()`. It is the vocabulary other units use.

| Type | Key invariant |
|---|---|
| `BBox` | `x0 <= x1` and `y0 <= y1`, enforced in `__post_init__`. **y-down** |
| `Provenance.merge` | Across pages returns the earlier page with `bbox=None` rather than inventing a box that spans a page break |
| `Document.doc_id` | Content hash, so identical content yields an identical id. It identifies a *version*, not a file — the stable identity of "the document at this path" is the path |
| `Chunk.chunk_id` | `doc_id#index` — a stable **slot**, so re-ingesting an edited document overwrites rather than orphaning. Content hash lives in metadata |
| `Usage.__add__` | Summable so a pipeline can report a total; `cached` is the AND of both |
| `Document.content_blocks()` | Excludes page furniture; the usual input to chunking |

## Errors

| Error | Retry? | Meaning |
|---|---|---|
| `MissingDependency` | no | optional backend absent; carries the exact install command |
| `AdapterError` | sometimes | backend reachable but failed or returned something unusable |
| `RateLimited` | **yes** | distinguished precisely so the governor can back off without string-matching provider text |
| `ValidationFailed` | **no** | retrying identical input against a deterministic validator reproduces the failure |
| `ScreeningRejected` | **never** | poison. Retrying a decompression bomb is a second outage |

## Usage

```python
from toolkit.core import BBox, Block, BlockType, Document, Provenance

doc = Document(
    doc_id=Document.id_from_text("hello"),
    page_count=1,
    blocks=[
        Block("Safety Manual", BlockType.HEADING,
              Provenance(page=1, bbox=BBox(50, 40, 300, 60)), level=1),
        Block("The supply must not exceed 40V.", BlockType.PARAGRAPH,
              Provenance(page=1, bbox=BBox(50, 80, 400, 95))),
    ],
)

print(doc.to_markdown())          # "# Safety Manual\n\nThe supply must not exceed 40V."
print(len(doc.content_blocks()))  # furniture excluded
```

Text normalisation, applied by every `DocumentSource` at the boundary:

```python
from toolkit.core import normalise_text

normalise_text("Conﬁguration file")   # -> "Configuration file"
```

## Limitations

- **No tenant or lineage fields.** Multi-tenancy needs `tenant_id` threaded through
  `Chunk` and every retrieval filter; retrofitting it means touching every storage
  path. Specced in `docs/ARCHITECTURE.md` §6, deliberately not built.
- **No version keys on artefacts.** `parser_version`, `chunker_version` and
  `embedding_model_version` are specced; only the embedding one is implemented, and
  it is tracked by `KnowledgeBase` rather than stamped on the contract.
- **`y-down` is a convention, not a type.** Nothing prevents passing y-up
  coordinates; it inverts reading order silently. The most common integration bug.
- `Document` is not JSON-serialisable, which is why `durable_steps` checkpoints at
  document granularity rather than between pipeline stages.
- No `trust` field yet, so `guardrails` treats all retrieved content as untrusted
  rather than distinguishing sources.

## Integration Guide

1. **Normalise coordinates to y-down in your adapter**, not downstream. pdfplumber's
   `top`/`bottom` are already y-down; pdfminer's `y0`/`y1` are y-up and must be
   flipped.
2. Populate `Provenance` even when it feels unnecessary. It is the one field you
   cannot add later without re-ingesting.
3. Use `Document.id_from_bytes` for files and keep your own path→doc_id map if you
   need to supersede an edited document.
4. Raise the narrowest error that fits. The governor and the ingest loop both branch
   on type, so `AdapterError` where `RateLimited` was meant turns a retryable blip
   into a failure.

## Extraction Notes

- **Preserved:** the typed-document-with-provenance shape, closest in spirit to
  Docling's `DoclingDocument` (MIT).
- **Removed:** every backend concern — no converters, no pipeline options, no model
  references.
- **Added:** the `doc_id#index` slot scheme, `Provenance.merge`'s refusal to span a
  page break, the summable `Usage`, and the retryable/poison error split.
- **Isolated:** imports nothing. Verified by `test_packaging.py`, which imports this
  directory as a bare top-level package in a subprocess with the repository off
  `sys.path`.
