"""Content-addressed caching for expensive model calls.

Two things make this worth owning rather than reaching for a generic cache:

**Embeddings are cached per text, not per batch.** A batch-level cache is nearly
useless in practice, because the next run almost never sends the identical batch —
one document changed, or the order differs, and the whole batch misses. Caching
each text individually means re-ingesting a 1,000-document corpus after editing
one document costs one embedding, not a thousand.

**The key includes everything that changes the answer.** Model name, temperature,
max_tokens and the full input. Leave any of them out and you serve a cached result
from a different configuration, which is worse than no cache because it is silent.

Both wrappers satisfy the same port as the thing they wrap, so caching is added by
construction and nothing downstream knows or cares.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

from ..core.models import Completion, Message, Usage

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache_entries (
    key        TEXT PRIMARY KEY,
    namespace  TEXT NOT NULL,
    value_json TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS cache_namespace ON cache_entries (namespace);
"""


def make_key(namespace: str, *parts: Any) -> str:
    """Stable hash over a namespace and any JSON-serialisable parts.

    `sort_keys` matters: without it, two dicts that differ only in insertion
    order produce different keys and the cache quietly halves its hit rate.
    """
    payload = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()
    return namespace + ":" + digest


class SqliteCache:
    """Durable content-addressed store. `:memory:` for tests, a path for reuse."""

    def __init__(self, database: str = ":memory:") -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(database, timeout=30, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            if database != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Mapping[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value_json FROM cache_entries WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(row["value_json"])

    def set(self, key: str, value: Mapping[str, Any]) -> None:
        namespace = key.split(":", 1)[0]
        with self._lock:
            self._conn.execute(
                "INSERT INTO cache_entries (key, namespace, value_json, created_at)"
                " VALUES (?, ?, ?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json,"
                " created_at = excluded.created_at",
                (key, namespace, json.dumps(value, ensure_ascii=False), time.time()),
            )
            self._conn.commit()

    def clear(self, namespace: str | None = None) -> int:
        with self._lock:
            if namespace is None:
                cursor = self._conn.execute("DELETE FROM cache_entries")
            else:
                cursor = self._conn.execute(
                    "DELETE FROM cache_entries WHERE namespace = ?", (namespace,)
                )
            self._conn.commit()
            return cursor.rowcount

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM cache_entries").fetchone()
        return int(row["n"])

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class CachedEmbedder:
    """Wraps any Embedder, caching each text individually.

    `calls` counts how many texts actually reached the wrapped embedder, which is
    the number a test should assert on — batch counts hide the behaviour that
    matters.
    """

    def __init__(self, embedder: Any, cache: Any, model_name: str | None = None) -> None:
        self._embedder = embedder
        self._cache = cache
        self._model = model_name or type(embedder).__name__
        self.calls = 0

    @property
    def dimension(self) -> int:
        return self._embedder.dimension

    @property
    def model_version(self) -> str:
        """Delegated, never synthesised. A cache that reported its own identity
        would hide the wrapped model's, which is the thing that matters."""
        return str(getattr(self._embedder, "model_version", self._model))

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        version = self.model_version
        keys = [make_key("embed", version, self.dimension, text) for text in texts]
        results: list[list[float] | None] = []
        pending: list[int] = []

        for position, key in enumerate(keys):
            cached = self._cache.get(key)
            if cached is None:
                results.append(None)
                pending.append(position)
            else:
                results.append([float(v) for v in cached["vector"]])

        if pending:
            fresh = self._embedder.embed([texts[i] for i in pending])
            self.calls += len(pending)
            if len(fresh) != len(pending):
                raise ValueError(
                    "wrapped embedder returned "
                    + str(len(fresh))
                    + " vectors for "
                    + str(len(pending))
                    + " texts"
                )
            for position, vector in zip(pending, fresh):
                results[position] = [float(v) for v in vector]
                self._cache.set(keys[position], {"vector": list(vector)})

        return [vector for vector in results if vector is not None]


class CachedLLM:
    """Wraps any LLM, caching on the full request.

    Only deterministic requests are cached. At a non-zero temperature the caller
    has explicitly asked for variation, and serving a cached completion would
    defeat the parameter they set — so those calls pass straight through.
    """

    def __init__(self, llm: Any, cache: Any, model_name: str | None = None) -> None:
        self._llm = llm
        self._cache = cache
        # Identity comes from the wrapped model, never from its class. Every
        # HttpLLM shares a class name, so keying on it let two different models
        # behind one cache serve each other's answers with no error anywhere.
        self._model = model_name or str(
            getattr(llm, "model_version", None) or type(llm).__name__
        )
        self.calls = 0

    @property
    def model_version(self) -> str:
        """The identity the cache keys on: the explicit `model_name`, else the
        wrapped model's own `model_version`, else its class name."""
        return self._model

    def complete(
        self,
        messages: Sequence[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Completion:
        if temperature > 0.0:
            self.calls += 1
            return self._llm.complete(messages, temperature, max_tokens)

        key = make_key(
            "llm",
            self._model,
            temperature,
            max_tokens,
            [(m.role, m.content) for m in messages],
        )
        cached = self._cache.get(key)
        if cached is not None:
            return Completion(
                text=str(cached["text"]),
                # Reported as cached with zero cost: a cached call really did cost
                # nothing, and a budget that counts it again is lying.
                usage=Usage(
                    input_tokens=int(cached.get("input_tokens", 0)),
                    output_tokens=int(cached.get("output_tokens", 0)),
                    cost_usd=0.0,
                    cached=True,
                ),
                model=str(cached.get("model", self._model)),
                finish_reason=str(cached.get("finish_reason", "")),
            )

        completion = self._llm.complete(messages, temperature, max_tokens)
        self.calls += 1
        self._cache.set(
            key,
            {
                "text": completion.text,
                "input_tokens": completion.usage.input_tokens,
                "output_tokens": completion.usage.output_tokens,
                "model": completion.model,
                "finish_reason": completion.finish_reason,
            },
        )
        return completion


__all__ = ["CachedEmbedder", "CachedLLM", "SqliteCache", "make_key"]
