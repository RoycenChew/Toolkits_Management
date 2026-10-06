# Cache Component

**Layer 1 · depends on `core` · `copy_tier: needs_core`**

## What It Does

Content-addressed caching for model calls. `CachedEmbedder` and `CachedLLM` wrap any
`Embedder` or `LLM` and satisfy the same port, so caching is added by construction
and nothing downstream knows.

## Why It Is Useful

Two design choices do the real work:

**Embeddings are cached per text, not per batch.** A batch-level cache is nearly
useless in practice: the next run almost never sends an identical batch — one
document changed, or the order differs — and the whole batch misses. Per-text keys
mean re-ingesting a 1,000-document corpus after editing one document costs one
embedding, not a thousand. In a hackathon this is the difference between 30
iterations and 300.

**The key includes everything that changes the answer.** Model, temperature,
`max_tokens`, and the full message list. Omit any one and you serve a result from a
different configuration — silently, which is worse than no cache at all.

## Original Source

Own work. The pattern is ubiquitous; the two decisions above are what make it worth
owning rather than reaching for a generic memoiser.

## Architecture

```
INPUT    texts / messages + model identity + parameters
   |
KEY      blake2b over json.dumps(..., sort_keys=True)     <- sort_keys matters
   |
LOOKUP   per-item for embeddings; whole-request for completions
   |
MISS     call the wrapped port, store the result
   |
OUTPUT   same type the wrapped port returns; Usage.cached=True on a hit
```

```
sqlite3 (stdlib)  ->  SqliteCache  ->  CachedEmbedder / CachedLLM
```

## Installation

Copy **two** directories, because this unit imports the shared contracts:

```bash
cp -r toolkit/cache  your_project/
cp -r toolkit/core      your_project/
```

Both must sit under the same parent package so the relative import resolves.
Python 3.10+. Standard library only.

Earlier versions of this README said "copy the `cache/` directory" full stop,
which does not work — `toolkit/tests/test_packaging.py` now copies each unit out
with its declared dependencies and imports it in a subprocess, so the instruction
is verified rather than asserted.


## Dependencies

Standard library (`sqlite3`, `hashlib`, `json`).

## Input / Output Schema

`SqliteCache(database=":memory:")` implements the toolkit's `Cache` port:
`get(key) -> Mapping | None`, `set(key, value)`, plus `clear(namespace=None)`,
`count()`, `close()`, and `.hits` / `.misses` counters.

`CachedEmbedder(embedder, cache, model_name=None)` — exposes `dimension`,
`embed(texts)`, and `.calls` (the number of texts that actually reached the wrapped
embedder, which is what a test should assert on).

`CachedLLM(llm, cache, model_name=None)` — exposes `complete(...)` and `.calls`.
A cache hit returns `Usage(cached=True, cost_usd=0.0)`: a cached call really did cost
nothing, and a budget that counts it again is lying.

## Usage

```python
from toolkit.adapters import FastEmbedEmbedder, LiteLLMClient
from toolkit.cache import CachedEmbedder, CachedLLM, SqliteCache

cache = SqliteCache("runs.db")            # survives restarts
embedder = CachedEmbedder(FastEmbedEmbedder(), cache)
llm = CachedLLM(LiteLLMClient("gpt-4o-mini"), cache)

embedder.embed(["alpha", "beta"])         # 2 calls
embedder.embed(["beta", "gamma"])         # 1 call - beta was cached
print(embedder.calls, cache.hits)
```

Compose with the governor, cache **inside**, so a hit costs no budget and no
rate-limit slot:

```python
from toolkit.governor import GovernedLLM, GovernorConfig

llm = GovernedLLM(CachedLLM(LiteLLMClient(), cache),
                  GovernorConfig(max_total_tokens=200_000))
```

## Limitations

- **No expiry or eviction.** The cache grows without bound. Call `clear(namespace)`
  when a model changes, or delete the file. Adding a TTL would need a policy that
  only you can choose.
- Cache identity is the wrapped model's `model_version` (or an explicit
  `model_name`). A wrapped object with no `model_version` falls back to its class
  name, and two such objects behind one class will collide. Every LLM in this
  toolkit has a `model_version`; give your own one too.
- Values must be JSON-serialisable, so embeddings are stored as float lists. That is
  roughly 20 bytes per dimension as text; a million 768-dim chunks is large.
- Non-zero temperature bypasses the cache entirely by design.
- SQLite with WAL handles several processes on one machine. It is not a shared
  network cache.
- Tool calls, streaming and structured outputs are not cached — the `LLM` port does
  not cover them.

## Integration Guide

1. Use one file-backed `SqliteCache` per project and pass it to every wrapper.
2. Make sure the wrapped model exposes `model_version` (all toolkit LLMs do), or
   pass `model_name` explicitly. Only then do two models never share entries.
3. Wrap the embedder before any batching helper, so `bounded_map` parallelises only
   the genuine misses.
4. Put the cache inside the governor, not outside.
5. `clear(namespace="embed")` after changing embedding model. Stale vectors of the
   wrong dimension are a confusing failure.

## Extraction Notes

- **Preserved:** nothing external; own implementation.
- **Added:** per-item embedding keys, `sort_keys` normalisation so dict ordering
  cannot halve the hit rate, `cached=True` / zero-cost usage reporting, and the
  deliberate temperature bypass.
- **Isolated:** both wrappers are structurally typed against the ports, so they work
  with any adapter — including the stdlib `HashingEmbedder` and `ScriptedLLM`.
