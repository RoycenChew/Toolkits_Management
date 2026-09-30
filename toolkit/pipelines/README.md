# Pipelines — `KnowledgeBase`

## What It Does

Wires the whole toolkit into two operations:

- `ingest_folder(path)` — discover files → parse → chunk → embed → index (dense + keyword)
- `ask(question)` — retrieve → fuse → optionally rerank → answer **with citations that
  resolve to a page and bounding box**

## Why It Is Useful

This is the phase that makes five components load-bearing at once, and it is the
hackathon demo:

```python
kb = KnowledgeBase()
kb.ingest_folder("./pdfs")
answer = kb.ask("what is the voltage limit?")
```

Every default is offline and needs no API key — `HashingEmbedder` on CPU,
`InMemoryVectorStore`, SQLite FTS5. Conference wifi is a known adversary, and a demo
that needs a hosted embedding endpoint is a demo that dies.

**Without an LLM, `ask` still works** and returns the best passages fully cited. That
extractive mode is genuinely useful on its own, and it is the right way to debug
retrieval: if the correct passage is not in the extractive output, no model was ever
going to rescue it.

## Architecture

```
ingest_folder(folder)
   |
DISCOVER   walk the tree, keep paths some DocumentSource claims
   |
per document, optionally as ONE durable step (crash costs this doc only)
   |  load    -> DocumentSource (pdfplumber / Docling / plaintext)
   |  chunk   -> ChunkerComponent (provenance carried through)
   |  embed   -> embed_all (batched, bounded concurrency, cache-aware)
   |  index   -> VectorStore.upsert + LexicalIndex.index
   |
OUTPUT     IngestResult(documents, chunks_indexed, embedding_calls)


ask(question)
   |
RETRIEVE   dense (cosine) + lexical (BM25), N candidates each
   |
GATE       refuse unless something is actually about the question
   |
FUSE       HybridRankerComponent, RRF
   |
RERANK     optional, budgeted
   |
CONTEXT    top_k chunks, numbered [1]..[k], each labelled with its page
   |
GENERATE   LLM with a cite-or-refuse system prompt   (or extractive, if no LLM)
   |
VERIFY     every [n] checked against what the model was shown
   |
OUTPUT     Answer(text, citations, chunks, usage, trace, grounded, unverified_markers)
```

## The three decisions that matter

**Citations are verified, not trusted.** Every `[n]` the model emits is checked
against the sources it was actually shown. A marker outside that range lands in
`answer.unverified_markers` and is *never* rendered as a citation — the model invented
a source, and that is the failure worth surfacing loudly. An answer citing nothing
real reports `grounded=False`.

**There is a relevance gate.** Rank fusion always returns something from a non-empty
index — RRF scores are positional — so without a gate a knowledge base answers every
question, including ones its corpus knows nothing about. That is the failure mode that
destroys trust fastest.

**The gate runs on term overlap, not a cosine floor.** A cosine threshold is not
portable. Measured on the sample corpus with the default embedder:

| query | dense score | actually relevant? |
|---|---|---|
| "what is the maximum supply voltage?" | 0.457 | yes |
| "airspeed velocity of an unladen swallow?" | **0.293** | **no** |
| "how long do refunds take?" | **0.248** | **yes** |

The irrelevant query out-scores a relevant one, because `HashingEmbedder` compares
character trigrams rather than meaning. Sentence embedders have the inverse problem:
bge-class models place unrelated text at 0.6–0.8, so a 0.25 floor would admit
everything. So `min_dense_similarity` defaults to `None` — calibrate it against your
own embedder, then set it.

## Usage

Offline, zero setup:

```python
from toolkit.pipelines import AskConfig, IngestConfig, KnowledgeBase

kb = KnowledgeBase()
result = kb.ingest_folder("./documents")
print(result.chunks_indexed, "chunks,", result.embedding_calls, "embedding calls")
for failure in result.failures:
    print("skipped", failure.path, failure.error)

answer = kb.ask("what is the refund window?", AskConfig(top_k=4))
print(answer.text)
for c in answer.citations:
    print(f"[{c.marker}] {c.source_uri} p{c.page} {c.bbox} — {' > '.join(c.heading_path)}")
```

Production wiring — same two calls:

```python
from toolkit.adapters import FastEmbedEmbedder, LanceDBStore, Bm25sIndex, LiteLLMClient
from toolkit.cache import CachedEmbedder, CachedLLM, SqliteCache
from toolkit.governor import GovernedLLM, GovernorConfig

cache = SqliteCache("runs.db")
kb = KnowledgeBase(
    embedder=CachedEmbedder(FastEmbedEmbedder(), cache),
    vector_store=LanceDBStore("./.lancedb"),
    lexical_index=Bm25sIndex(),
    llm=GovernedLLM(CachedLLM(LiteLLMClient("gpt-4o-mini"), cache),
                    GovernorConfig(max_total_tokens=500_000, max_requests_per_minute=60)),
)
kb.ingest_folder("./documents", IngestConfig(durable_db="ingest.db"))
```

With `durable_db` set, re-running after a crash reports finished documents as
`replayed` and re-embeds nothing.

## Limitations

- **The extractive mode does not summarise.** Without an LLM it returns passages. That
  is the honest output, not an answer.
- **The relevance gate trades recall for precision.** Term overlap refuses genuine
  paraphrases that share no vocabulary with the source. Set `min_dense_similarity`
  once calibrated, or `relevance_gate=False`, if your queries are paraphrase-heavy.
- **Retrieval quality with the default embedder is weak** — it is lexical, not
  semantic. In the sample corpus "how long do refunds take?" retrieves the
  *Eligibility* section rather than *Processing*. Swap in `FastEmbedEmbedder` for real
  semantics; the default exists so the pipeline runs with nothing installed.
- **Chunks are mirrored in memory** (`self._chunks`) so answers can carry full
  provenance. A million-chunk corpus will not fit; for that, read provenance back from
  the store instead.
- The durable unit is one whole document, not each stage inside it — a `Document` is
  not JSON-serialisable, and inventing a serialisation for it to checkpoint between
  parse and embed buys little.
- `_run_id` fingerprints path + size + mtime, not content. A file edited without
  changing either would be treated as already ingested.
- Ingest is sequential across documents (parallel only *within* a document's embedding
  batches). For a large corpus, drive `bounded_map` over `ingest()` calls yourself.
- No query rewriting, no multi-hop, no conversation memory. One question, one retrieval.

## Integration Guide

1. Start with the defaults and `ask` with no LLM. If the right passage is not in the
   extractive output, fix retrieval before adding a model — a model cannot fix recall.
2. Then add `FastEmbedEmbedder`. This is usually the single largest quality jump.
3. Add the LLM last, wrapped in `CachedLLM` inside `GovernedLLM`.
4. Check `answer.unverified_markers` in your UI. Non-empty means the model fabricated a
   source, and it should not render as a citation.
5. Calibrate `min_dense_similarity` with five known-irrelevant queries against your
   real embedder, then set it and keep the gate on.
6. Set `durable_db` for any corpus big enough that restarting would annoy you.

## Extraction Notes

- **Preserved:** the standard retrieve-fuse-rerank-generate shape common to every RAG
  framework.
- **Added:** marker verification against the shown sources, the portable relevance
  gate, the extractive fallback, per-document durable checkpointing, and
  `RetrievalTrace` so "why did it answer that?" is answerable after the fact.
- **Isolated:** this module imports only toolkit ports and components. No vendor SDK
  appears anywhere in it; every backend arrives as a constructor argument.
