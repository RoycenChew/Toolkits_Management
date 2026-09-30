"""VectorStore and LexicalIndex adapters.

Two of each, and in both cases one of the two is standard library only:

* `InMemoryVectorStore` — exact brute-force cosine. For corpora under roughly
  50k chunks this is not a toy; it is faster than an ANN index at that scale and
  it returns exact results, which removes a variable while you are debugging
  retrieval quality.
* `LanceDBStore` — embedded, file-backed, survives restarts, no server.
* `SqliteFtsIndex` — BM25 via SQLite's FTS5, which ships with Python.
* `Bm25sIndex` — faster BM25 with proper tokenisation for larger corpora.

Having a stdlib implementation of each port is what makes the offline hackathon
path real rather than aspirational.
"""
from __future__ import annotations

import contextlib
import math
import re
import sqlite3
import threading
from collections.abc import Sequence
from typing import Any

from ..core.errors import AdapterError, MissingDependency
from ..core.models import Chunk, SearchHit

_WORD = re.compile(r"[^\w]+", re.UNICODE)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    num = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        num += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return num / (math.sqrt(na) * math.sqrt(nb))


class InMemoryVectorStore:
    """Exact cosine search over an in-process dict. Upsert is by `chunk_id`."""

    def __init__(self, dimension: int | None = None) -> None:
        self._dimension = dimension
        self._rows: dict[str, tuple[Chunk, list[float]]] = {}

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise AdapterError(
                "got " + str(len(vectors)) + " vectors for " + str(len(chunks)) + " chunks"
            )
        for chunk, vector in zip(chunks, vectors):
            if self._dimension is None:
                self._dimension = len(vector)
            elif len(vector) != self._dimension:
                raise AdapterError(
                    "vector of length "
                    + str(len(vector))
                    + " does not match store dimension "
                    + str(self._dimension)
                )
            self._rows[chunk.chunk_id] = (chunk, [float(v) for v in vector])

    def search(self, vector: Sequence[float], top_k: int = 10) -> list[SearchHit]:
        scored = [
            SearchHit(
                chunk_id=chunk_id,
                score=_cosine(vector, stored),
                text=chunk.text,
                doc_id=chunk.doc_id,
                metadata=dict(chunk.metadata),
            )
            for chunk_id, (chunk, stored) in self._rows.items()
        ]
        scored.sort(key=lambda h: (-h.score, h.chunk_id))
        return scored[:top_k]

    def count(self) -> int:
        return len(self._rows)


class LanceDBStore:
    """Embedded, file-backed vector store. No server to run or deploy."""

    def __init__(
        self,
        uri: str = "./.lancedb",
        table_name: str = "chunks",
        connection: Any | None = None,
    ) -> None:
        self._uri = uri
        self._table_name = table_name
        self._db = connection
        self._table: Any = None

    def _connect(self) -> Any:
        if self._db is None:
            try:
                import lancedb  # type: ignore
            except ImportError as exc:
                raise MissingDependency("lancedb", "store") from exc
            self._db = lancedb.connect(self._uri)
        return self._db

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise AdapterError(
                "got " + str(len(vectors)) + " vectors for " + str(len(chunks)) + " chunks"
            )
        if not chunks:
            return
        db = self._connect()
        rows = [
            {
                "chunk_id": chunk.chunk_id,
                "vector": [float(v) for v in vector],
                "text": chunk.text,
                "doc_id": chunk.doc_id,
                "pages": ",".join(str(p) for p in chunk.pages),
            }
            for chunk, vector in zip(chunks, vectors)
        ]
        if self._table is None:
            names = db.table_names()
            self._table = (
                db.open_table(self._table_name)
                if self._table_name in names
                else db.create_table(self._table_name, data=rows)
            )
            if self._table_name not in names:
                return
        # Delete-then-add is how LanceDB expresses upsert portably across
        # versions; merge_insert exists but its signature has moved.
        ids = ", ".join("'" + c.chunk_id.replace("'", "''") + "'" for c in chunks)
        # An empty table, or a LanceDB version that rejects the predicate, means
        # there is nothing to delete — not a failure worth propagating.
        with contextlib.suppress(Exception):
            self._table.delete("chunk_id IN (" + ids + ")")
        self._table.add(rows)

    def search(self, vector: Sequence[float], top_k: int = 10) -> list[SearchHit]:
        if self._table is None:
            db = self._connect()
            if self._table_name not in db.table_names():
                return []
            self._table = db.open_table(self._table_name)
        rows = self._table.search([float(v) for v in vector]).limit(top_k).to_list()
        hits = []
        for row in rows:
            # LanceDB returns L2 distance; convert so larger is better, which is
            # what every consumer of SearchHit assumes.
            distance = float(row.get("_distance", 0.0))
            hits.append(
                SearchHit(
                    chunk_id=str(row.get("chunk_id", "")),
                    score=1.0 / (1.0 + distance),
                    text=str(row.get("text", "")),
                    doc_id=str(row.get("doc_id", "")),
                    metadata={"pages": row.get("pages", "")},
                )
            )
        return hits

    def count(self) -> int:
        if self._table is None:
            db = self._connect()
            if self._table_name not in db.table_names():
                return 0
            self._table = db.open_table(self._table_name)
        return int(self._table.count_rows())


class SqliteFtsIndex:
    """BM25 keyword search using SQLite FTS5 — no third-party package at all.

    FTS5 ships with CPython on every platform this toolkit targets. Its `bm25()`
    ranking function returns *lower is better*, so scores are negated on the way
    out to match the rest of the toolkit's convention.
    """

    def __init__(self, database: str = ":memory:") -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(database, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            try:
                self._conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts"
                    " USING fts5(chunk_id UNINDEXED, doc_id UNINDEXED, text)"
                )
            except sqlite3.OperationalError as exc:
                raise AdapterError(
                    "this SQLite build lacks FTS5; use Bm25sIndex instead"
                ) from exc
            self._conn.commit()

    def index(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        with self._lock:
            for chunk in chunks:
                self._conn.execute(
                    "DELETE FROM chunk_fts WHERE chunk_id = ?", (chunk.chunk_id,)
                )
            self._conn.executemany(
                "INSERT INTO chunk_fts (chunk_id, doc_id, text) VALUES (?, ?, ?)",
                [(c.chunk_id, c.doc_id, c.text) for c in chunks],
            )
            self._conn.commit()

    def search(self, query: str, top_k: int = 10) -> list[SearchHit]:
        # FTS5 query syntax would treat punctuation as operators, so the query
        # is reduced to quoted terms joined by OR. Passing raw user text through
        # is a syntax-error generator.
        terms = [t for t in _WORD.split(query.lower()) if t]
        if not terms:
            return []
        expression = " OR ".join('"' + t + '"' for t in terms)
        with self._lock:
            rows = self._conn.execute(
                "SELECT chunk_id, doc_id, text, bm25(chunk_fts) AS rank"
                " FROM chunk_fts WHERE chunk_fts MATCH ?"
                " ORDER BY rank LIMIT ?",
                (expression, top_k),
            ).fetchall()
        return [
            SearchHit(
                chunk_id=row["chunk_id"],
                score=-float(row["rank"]),
                text=row["text"],
                doc_id=row["doc_id"],
            )
            for row in rows
        ]

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM chunk_fts").fetchone()
        return int(row["n"])

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class Bm25sIndex:
    """BM25 via the `bm25s` package. Faster than FTS5 on larger corpora.

    Rebuilds its index on every `index()` call, because bm25s is a static index
    by design. That is fine for ingest-then-query workloads and wrong for
    incremental ones — use SqliteFtsIndex if you need to add documents
    continuously.
    """

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._retriever: Any = None
        self._order: list[str] = []

    def _build(self) -> None:
        try:
            import bm25s  # type: ignore
        except ImportError as exc:
            raise MissingDependency("bm25s", "store") from exc
        self._order = list(self._chunks)
        corpus = [self._chunks[cid].text for cid in self._order]
        if not corpus:
            self._retriever = None
            return
        tokens = bm25s.tokenize(corpus, stopwords="en", show_progress=False)
        retriever = bm25s.BM25()
        retriever.index(tokens, show_progress=False)
        self._retriever = retriever

    def index(self, chunks: Sequence[Chunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk
        self._build()

    def search(self, query: str, top_k: int = 10) -> list[SearchHit]:
        if self._retriever is None or not self._order:
            return []
        # bm25s happily retrieves against an empty token list and returns the
        # arbitrary first k documents with zero scores. Guarding here keeps the
        # adapter's behaviour identical to SqliteFtsIndex, which is the point of
        # having a port at all.
        if not [t for t in _WORD.split(query.lower()) if t]:
            return []
        import bm25s  # type: ignore

        tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
        limit = min(top_k, len(self._order))
        indices, scores = self._retriever.retrieve(tokens, k=limit, show_progress=False)
        hits = []
        for position, score in zip(indices[0], scores[0]):
            if float(score) <= 0.0:
                continue
            chunk = self._chunks[self._order[int(position)]]
            hits.append(
                SearchHit(
                    chunk_id=chunk.chunk_id,
                    score=float(score),
                    text=chunk.text,
                    doc_id=chunk.doc_id,
                )
            )
        return hits

    def count(self) -> int:
        return len(self._chunks)


__all__ = [
    "Bm25sIndex",
    "InMemoryVectorStore",
    "LanceDBStore",
    "SqliteFtsIndex",
]
