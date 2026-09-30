"""Data contracts for the ingest and ask pipelines."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..core.models import BBox, Chunk, SearchHit, Usage


@dataclass
class IngestConfig:
    batch_size: int = 32
    """Texts per embedding call. Larger is cheaper per text until the provider's
    payload limit; 32 is safe everywhere."""
    max_workers: int = 4
    """Parallel embedding batches per document."""
    durable_db: str | None = None
    """Path to a checkpoint database. When set, each document becomes one durable
    step, so a crash mid-corpus costs the document in flight and nothing else.
    Set to None for a single-shot run where restarting from zero is acceptable."""
    skip_failed: bool = True
    """Continue past a document that fails to parse. A corpus with one corrupt
    PDF should still ingest the other 999, with the failure reported."""
    lexical: bool = True
    """Also populate the keyword index. Worth keeping on: exact identifiers and
    rare proper nouns are where dense retrieval is weakest."""


@dataclass
class DocumentOutcome:
    path: str
    doc_id: str = ""
    chunks: int = 0
    pages: int = 0
    status: str = "ingested"
    """'ingested', 'replayed' (already done in an earlier run), or 'failed'."""
    error: str = ""


@dataclass
class IngestResult:
    documents: Sequence[DocumentOutcome]
    chunks_indexed: int
    embedding_calls: int
    """Texts that actually reached the embedder. With a cache in place this is
    the number that cost money, not the number of chunks."""

    @property
    def failures(self) -> list[DocumentOutcome]:
        return [d for d in self.documents if d.status == "failed"]

    @property
    def ok(self) -> bool:
        return not self.failures


@dataclass
class AskConfig:
    top_k: int = 5
    """Chunks placed in the prompt. More context is not better: it dilutes
    attention and costs tokens."""
    candidates_per_retriever: int = 20
    """Recall net per retriever before fusion. Wider costs almost nothing here
    because fusion is cheap; it is the reranker that has a budget."""
    rerank_budget: int = 0
    """Candidates passed to a reranker, when one is supplied. 0 disables."""
    rrf_k: int = 60
    lexical_weight: float = 1.0
    dense_weight: float = 1.0
    temperature: float = 0.0
    max_tokens: int | None = 700
    refuse_without_context: bool = True
    """When retrieval finds nothing relevant, answer that the corpus does not
    cover it rather than letting the model answer from its own weights. An
    unsourced answer in a document-grounded system is the failure users notice
    last and trust least."""

    relevance_gate: bool = True
    """Refuse when the best hit is not actually about the question.

    Necessary because rank fusion always returns *something* from a non-empty
    index: RRF scores are positional, so an off-topic query still yields a
    confidently-ranked top result. Without this gate a knowledge base answers
    every question, which is the failure mode that destroys trust fastest.

    The gate passes if a retrieved chunk literally shares a content word with the
    query, or — when `min_dense_similarity` is set — if dense similarity clears
    that floor."""

    min_dense_similarity: float | None = None
    """Cosine floor for the dense arm of the gate. `None` disables that arm, so
    the gate runs on term overlap alone.

    It defaults to off because **no single threshold is portable across
    embedders**, and a wrong one is worse than none. Measured on this toolkit's
    own sample corpus with the default `HashingEmbedder`:

        "what is the maximum supply voltage?"      -> 0.457  (relevant)
        "what is the airspeed velocity of a swallow?" -> 0.293  (irrelevant)
        "how long do refunds take?"               -> 0.248  (relevant)

    The irrelevant query scores *higher* than a relevant one, because
    `HashingEmbedder` compares character trigrams rather than meaning. Sentence
    embedders have the opposite problem: models like bge-small put unrelated text
    around 0.6-0.8, so a naive 0.25 floor would admit everything.

    Calibrate it once against your own embedder and a handful of known-irrelevant
    queries, then set it. Until you have, term overlap is the honest signal.

    Trade-off worth knowing: overlap alone refuses genuine paraphrases that share
    no vocabulary with the source. That is a precision-for-recall trade, and it is
    the right default for a document-grounded system — a wrong confident answer
    costs more than an admitted miss."""
    context_char_limit: int = 8000
    """Hard cap on assembled context, as a backstop against a huge top_k."""


@dataclass(frozen=True)
class Citation:
    marker: int
    """The [n] the model was shown and used."""
    chunk_id: str
    doc_id: str
    page: int
    bbox: BBox | None
    source_uri: str
    heading_path: Sequence[str]
    quote: str
    """The chunk text, truncated. What the claim was actually based on."""
    score: float


@dataclass
class RetrievalTrace:
    """Everything needed to explain or evaluate one answer.

    Kept because 'why did it return that?' is the question you will ask most
    often, and reconstructing it after the fact is impossible.
    """

    query: str
    dense_hits: Sequence[SearchHit] = field(default_factory=list)
    lexical_hits: Sequence[SearchHit] = field(default_factory=list)
    fused_ids: Sequence[str] = field(default_factory=list)
    used_ids: Sequence[str] = field(default_factory=list)
    reranked: bool = False


@dataclass
class Answer:
    text: str
    citations: Sequence[Citation]
    chunks: Sequence[Chunk]
    """The chunks placed in the prompt, in the order they were numbered."""
    usage: Usage = field(default_factory=Usage)
    trace: RetrievalTrace | None = None
    grounded: bool = True
    """False when the pipeline refused for lack of context, or when the model
    produced an answer citing nothing."""
    unverified_markers: Sequence[int] = field(default_factory=list)
    """Markers the model cited that were never shown to it. Non-empty means the
    model invented a source, which is the single most important thing to surface
    and the easiest to miss."""

    @property
    def pages(self) -> list[int]:
        return sorted({c.page for c in self.citations})
