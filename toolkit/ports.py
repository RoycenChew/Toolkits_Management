"""The ports: what the toolkit needs from the outside world.

Every one of these is a `Protocol`, so an implementation never imports the
toolkit to satisfy it — structural typing means your own class, a test double,
or a vendor adapter all work identically.

The discipline that keeps these honest is **two implementations per port before
the port is trusted**. A protocol written against a single backend is just that
vendor's interface with the names changed, and you discover this at the worst
possible moment: when you try to swap it. Each port below therefore ships with
one zero-dependency implementation and at least one real backend, and the
contract tests run both through the same assertions.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from .core.limits import ScreeningLimits, ScreeningResult
from .core.models import Chunk, Completion, Document, Message, SearchHit


@runtime_checkable
class DocumentSource(Protocol):
    """Anything that turns a file into a `Document`.

    Implementations must emit blocks in reading order and populate provenance
    wherever the backend knows geometry. A source with no geometry (plain text)
    sets `bbox=None` rather than fabricating one.
    """

    def load(self, path: str) -> Document: ...

    def supports(self, path: str) -> bool:
        """Whether this source can handle the given path. Cheap check on the
        extension or magic bytes; never opens a model to decide."""
        ...

    def screen(self, path: str, limits: ScreeningLimits) -> ScreeningResult:
        """Is this safe to parse? Called before `load`, always.

        Parsers run on whatever file they are given. Without a cap, a
        decompression bomb or a 40,000-page PDF takes the process down, and
        catching the exception does not save you from an OOM kill. A screening
        failure is **poison, not retryable** — retrying a bomb is a second
        outage.
        """
        ...


@runtime_checkable
class Embedder(Protocol):
    """Text to dense vectors.

    `embed` takes a batch because every real backend is far more efficient
    batched, and a single-item API encourages the slowest possible usage.
    Vectors must be returned in input order.
    """

    @property
    def dimension(self) -> int: ...

    @property
    def model_version(self) -> str:
        """Stable identity of whatever produced the vectors.

        Required, not optional. Dimension alone is not identity: swapping
        bge-small for a different model of the same size produces vectors that a
        store will happily accept, a search will happily return, and whose
        quality has silently collapsed with no error anywhere. The version
        string is what makes that detectable.
        """
        ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@runtime_checkable
class LLM(Protocol):
    """A chat-completion model.

    Kept to the narrowest useful surface. Tool calling, streaming and structured
    output are deliberately absent: they differ enough between providers that a
    lowest-common-denominator abstraction would be dishonest. Reach for the
    provider SDK when you need them.
    """

    def complete(
        self,
        messages: Sequence[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Completion: ...


@runtime_checkable
class VectorStore(Protocol):
    """Dense vector persistence and nearest-neighbour search.

    `upsert` must be idempotent on `chunk_id`, because re-ingesting a document
    is a normal event and duplicated rows silently corrupt retrieval.
    """

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None: ...

    def search(self, vector: Sequence[float], top_k: int = 10) -> list[SearchHit]: ...

    def count(self) -> int: ...

    def delete(self, chunk_ids: Sequence[str]) -> int: ...

    def delete_by_doc(self, doc_id: str, keep: Sequence[str] | None = None) -> int:
        """Remove a document's rows. Returns how many were deleted.

        `keep=None` deletes everything for the document — the retention and
        right-to-erasure path.

        `keep=[...]` deletes everything *except* those ids, which is the orphan
        sweep. Chunk ids are `doc_id#index` slots, so re-ingesting a document
        that now produces five chunks where it previously produced eight leaves
        slots #5..#7 behind: stale content, still retrievable, still citable.
        One method covers both jobs because they are the same query with a
        different exclusion set.
        """
        ...


@runtime_checkable
class LexicalIndex(Protocol):
    """Keyword search. The other half of hybrid retrieval.

    Worth keeping even when a vector store is present: exact identifiers, rare
    proper nouns and numbers are where dense retrieval is weakest and BM25 is
    strongest.
    """

    def index(self, chunks: Sequence[Chunk]) -> None: ...

    def search(self, query: str, top_k: int = 10) -> list[SearchHit]: ...

    def count(self) -> int: ...

    def delete(self, chunk_ids: Sequence[str]) -> int: ...

    def delete_by_doc(self, doc_id: str, keep: Sequence[str] | None = None) -> int: ...


@runtime_checkable
class Reranker(Protocol):
    """Scores (query, document) pairs more accurately than first-stage retrieval.

    Structurally compatible with `hybrid_ranker.Reranker`, so the same object
    satisfies both without importing either.
    """

    def score(self, query: str, items: Sequence[Any]) -> Sequence[float]: ...


@runtime_checkable
class Cache(Protocol):
    """Content-addressed storage for expensive call results (Phase 2)."""

    def get(self, key: str) -> Mapping[str, Any] | None: ...

    def set(self, key: str, value: Mapping[str, Any]) -> None: ...


__all__ = [
    "Cache",
    "DocumentSource",
    "Embedder",
    "LLM",
    "LexicalIndex",
    "Reranker",
    "VectorStore",
]
