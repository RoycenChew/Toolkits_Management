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

**Every method here is documented**, unlike most of the toolkit's internals. A
protocol method body is `...` — the docstring *is* the implementation contract,
so an undocumented one leaves an implementer guessing at semantics the type
signature cannot express: whether order is preserved, whether a repeat call is a
no-op, what a return of zero means, and what is safe to retry.
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

    def load(self, path: str) -> Document:
        """Parse a file into a `Document`.

        Blocks must be in reading order — the order a human reads them, which on
        a multi-column page is not the order the extractor emits them. Text
        should be normalised at this boundary (`core.normalise_text`), because a
        ligature renders identically to the letters it replaces and so breaks
        search invisibly downstream.

        Implementations are expected to screen first; `load` may assume the
        limits have been applied but must not rely on the caller having done so.

        Raises:
            ScreeningRejected: the file failed a resource limit. Never retry.
            MissingDependency: the backend package is not installed.
            AdapterError: the file is parseable in principle but yielded nothing
                usable — a scanned PDF with no text layer, for instance. The
                message should say what to do about it.
        """
        ...

    def supports(self, path: str) -> bool:
        """Whether this source can handle the given path.

        A cheap check on the extension or magic bytes. Must not open a model, hit
        the network, or read the whole file: `load_document` calls this on every
        registered source to pick one, so an expensive `supports` makes dispatch
        cost more than parsing.
        """
        ...

    def screen(self, path: str, limits: ScreeningLimits) -> ScreeningResult:
        """Is this safe to parse? Called before `load`, always.

        Parsers run on whatever file they are given. Without a cap, a
        decompression bomb or a 40,000-page PDF takes the process down, and
        catching the exception does not save you from an OOM kill. A screening
        failure is **poison, not retryable** — retrying a bomb is a second
        outage.

        Returns a result rather than raising, so a caller can audit a corpus
        without handling exceptions; call `.raise_if_rejected()` to convert.

        An implementation that cannot check a particular limit should return a
        passing result for the checks it *can* perform rather than failing
        closed — `DoclingSource` screens size but not page count, because
        counting units would mean opening every format twice. Document which
        limits apply.
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
    def dimension(self) -> int:
        """Length of every vector this embedder produces.

        Must be stable for the life of the instance: a vector store sizes its
        column from the first `upsert` and a later change is a corruption, not a
        resize. Determining it may require one probe call, which an
        implementation should cache.
        """
        ...

    @property
    def model_version(self) -> str:
        """Stable identity of whatever produced the vectors.

        Required, not optional. Dimension alone is not identity: swapping
        bge-small for a different model of the same size produces vectors that a
        store will happily accept, a search will happily return, and whose
        quality has silently collapsed with no error anywhere. The version
        string is what makes that detectable.

        Must change whenever the output changes — including configuration, not
        just weights. `HashingEmbedder` encodes its dimension *and* its trigram
        setting for exactly that reason.

        A wrapper must delegate rather than synthesise: reporting the wrapper's
        own identity hides the model's, which is the thing that matters.
        """
        ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch, returning one vector per input **in input order**.

        Order is part of the contract. Callers zip the result against the input,
        so reordering silently misaligns every vector with the wrong text — a
        corruption no exception reports and no test notices until retrieval
        quality drops.

        An empty input returns an empty list and must not error. Every vector
        must have length `dimension`.

        Raises:
            MissingDependency: the backend package is not installed.
            AdapterError: the backend failed, or returned a count that does not
                match the input.
        """
        ...


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
    ) -> Completion:
        """Generate one completion.

        `temperature=0.0` is the default because the toolkit's callers — the
        extraction repair loop, the grounded-answer prompt — want reproducible
        output, and because `CachedLLM` only caches deterministic requests. A
        non-zero temperature is an explicit request for variation and bypasses
        the cache.

        `Completion.usage` should carry real provider-reported token counts
        where available; the governor's budget is only as honest as this field.
        An implementation that cannot obtain them should estimate and say so in
        its documentation rather than returning zeros.

        Raises:
            RateLimited: the provider refused for rate or quota reasons. This is
                the one failure always worth retrying, which is why it has its
                own type rather than being a string inside `AdapterError`.
                Populate `retry_after` when the provider supplies it.
            MissingDependency: the backend package is not installed.
            AdapterError: any other backend failure, including a response with
                no choices.
        """
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Dense vector persistence and nearest-neighbour search.

    `upsert` must be idempotent on `chunk_id`, because re-ingesting a document
    is a normal event and duplicated rows silently corrupt retrieval.
    """

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        """Insert or replace rows, keyed on `chunk_id`.

        Idempotent by contract: calling twice with the same input must leave one
        row per chunk, not two. Re-ingestion is routine, and a store that
        appends turns it into silent duplication that inflates every subsequent
        search.

        `chunks` and `vectors` are positionally paired and must be the same
        length; an implementation should reject a mismatch rather than truncate,
        since truncating pairs vectors with the wrong text.

        Raises:
            AdapterError: length mismatch, or a vector whose length does not
                match the store's established dimension.
            MissingDependency: the backend package is not installed.
        """
        ...

    def search(self, vector: Sequence[float], top_k: int = 10) -> list[SearchHit]:
        """Nearest neighbours, **ordered best-first**.

        Returns at most `top_k`, and fewer when the store holds fewer — never
        padding. An empty store returns an empty list rather than raising.

        The score scale is backend-specific and deliberately **not** normalised
        here: cosine from one store and a converted L2 distance from another are
        not comparable, and normalising twice destroys information. Fuse across
        stores with `hybrid_ranker`'s rank-based RRF, which needs no calibration.
        Higher must mean better.
        """
        ...

    def count(self) -> int:
        """Total rows held. Zero for an empty or not-yet-created store."""
        ...

    def delete(self, chunk_ids: Sequence[str]) -> int:
        """Delete specific rows. Returns how many were actually removed.

        Deleting an absent id is a no-op, not an error: a sweep that has already
        partially run must be safe to repeat, because multi-store deletion is
        not transactional and recovery depends on re-running it. An empty input
        returns 0.
        """
        ...

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

        Note that `doc_id` is a content hash, so it identifies a *version*. To
        retire a document that was edited, the caller must track path → doc_id
        itself; `KnowledgeBase._supersede` does this.
        """
        ...


@runtime_checkable
class LexicalIndex(Protocol):
    """Keyword search. The other half of hybrid retrieval.

    Worth keeping even when a vector store is present: exact identifiers, rare
    proper nouns and numbers are where dense retrieval is weakest and BM25 is
    strongest.
    """

    def index(self, chunks: Sequence[Chunk]) -> None:
        """Add or replace chunks in the index, keyed on `chunk_id`.

        Idempotent, for the same reason as `VectorStore.upsert`.

        An implementation may rebuild its whole index on each call — `bm25s`
        does, because its index is static by design. That is acceptable, and it
        must be documented, because it makes continuous ingestion quadratic.
        """
        ...

    def search(self, query: str, top_k: int = 10) -> list[SearchHit]:
        """Keyword search, **ordered best-first**, higher score better.

        Raw user text must be tokenised rather than passed to a query engine as
        syntax: FTS5 treats punctuation as operators, so an unescaped question
        mark is a syntax error rather than a search.

        An empty query, or one consisting only of stopwords, returns an empty
        list — it must not fall back to returning arbitrary rows with zero
        scores, which looks like a result and is not. (`bm25s` does this by
        default; its adapter guards against it to match `SqliteFtsIndex`.)
        """
        ...

    def count(self) -> int:
        """Total chunks indexed."""
        ...

    def delete(self, chunk_ids: Sequence[str]) -> int:
        """Delete specific chunks. Returns how many were removed. Idempotent."""
        ...

    def delete_by_doc(self, doc_id: str, keep: Sequence[str] | None = None) -> int:
        """As `VectorStore.delete_by_doc`. Semantics must match exactly.

        Deletion that behaves differently between the dense and lexical index
        corrupts retrieval asymmetrically: a chunk gone from one and present in
        the other is still retrievable and no longer scoreable.
        """
        ...


@runtime_checkable
class Reranker(Protocol):
    """Scores (query, document) pairs more accurately than first-stage retrieval.

    Structurally compatible with `hybrid_ranker.Reranker`, so the same object
    satisfies both without importing either.
    """

    def score(self, query: str, items: Sequence[Any]) -> Sequence[float]:
        """Score each item against the query, **in input order**.

        Must return exactly one score per item; the caller zips the result and
        raises on a mismatch rather than silently misaligning.

        Must be deterministic for identical input. A cascade whose ordering
        varies between runs makes an eval diff meaningless.

        `items` are duck-typed — `hybrid_ranker.FusedItem` carries its text in
        `payload["text"]`, but a caller may pass chunks or plain strings.
        Reading the text defensively costs a few lines and removes a class of
        silent empty-result bugs.

        An empty `items` returns an empty sequence without calling the backend.
        """
        ...


@runtime_checkable
class Cache(Protocol):
    """Content-addressed storage for expensive call results.

    Keys are opaque; build them with `cache.make_key`, which sorts dict keys so
    two equivalent requests cannot produce different keys and halve the hit rate.
    """

    def get(self, key: str) -> Mapping[str, Any] | None:
        """Return the stored value, or None on a miss. Never raises on a miss."""
        ...

    def set(self, key: str, value: Mapping[str, Any]) -> None:
        """Store a value, replacing any existing entry for the key.

        `value` must be JSON-serialisable: the cache is expected to survive a
        process restart, so whatever goes in has to be storable.
        """
        ...


__all__ = [
    "Cache",
    "DocumentSource",
    "Embedder",
    "LLM",
    "LexicalIndex",
    "Reranker",
    "VectorStore",
]
