"""Data contracts for the ingest and ask pipelines."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from ..core.limits import ScreeningLimits
from ..core.models import BBox, Chunk, SearchHit, Usage
from ..guardrails.models import GuardrailConfig


@dataclass
class IngestConfig:
    limits: ScreeningLimits = field(default_factory=ScreeningLimits)
    """Resource caps applied before any parser touches the file. Default-on,
    including in the hackathon profile: the habit of trusting your own documents
    is how an untrusted one eventually gets parsed unguarded."""

    batch_size: int = 32
    """Texts per embedding call. Larger is cheaper per text until the provider's
    payload limit; 32 is safe everywhere."""
    max_workers: int = 4
    """Parallel embedding batches per document."""
    on_document: Callable[[int, int, DocumentOutcome], None] | None = None
    """Called after each document with `(index, total, outcome)`, 1-based.

    Ingest is sequential across documents and a real corpus takes a long time -
    49 arXiv PDFs took 38 minutes - during which this call printed nothing, so a
    slow run was indistinguishable from a hung one. The only way to observe
    progress was to query the durable checkpoint table from another process,
    which only worked because `durable_db` happened to be set.

    A callback rather than built-in logging: a library that prints is a library
    you cannot embed. Raised exceptions are deliberately **not** caught - a
    broken progress reporter is a bug in the caller's code and hiding it would
    make it unfindable.
    """
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
    """How this document ended up in the index.

    - ``'ingested'``  — parsed, chunked, embedded and indexed by this call.
    - ``'replayed'``  — a checkpoint said it was already done *and* the document
      was verified to still be present in this KnowledgeBase, so the work was
      skipped.
    - ``'reingested'`` — a checkpoint said it was already done but the document
      was **not** present (the usual cause: the checkpoint database outlived the
      process that wrote it, and chunks are process-local). The work was redone
      rather than trusting a record whose side effects are gone.
    - ``'failed'``    — see ``error``.
    """
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

    The gate passes when one retrieved chunk covers enough of the query's
    content words (see `min_term_coverage`), or — when `min_dense_similarity`
    is set — if dense similarity clears that floor."""

    min_term_coverage: float = 0.5
    """Fraction of the query's content words one retrieved chunk must contain.

    Measured against a single chunk, not pooled across hits, and that is the
    whole point. This gate used to pass if *any* content word appeared in *any*
    top hit, which on a real corpus is no test at all: a question about
    PostgreSQL ports or ethanol boiling points shares "default", "server",
    "point" or "degrees" with some paper, so **all eight pre-registered
    unanswerable queries were answered**, with citations to real but irrelevant
    chunks. A relevant chunk contains several of the query's words *together*;
    an irrelevant corpus merely contains them somewhere.

    0.5 was measured on 49 real papers against 24 answerable and 8 unanswerable
    queries:

        gate                         answerable passed   unanswerable refused
        any term in any hit (old)         100%                    0%
        coverage >= 0.4 .. 0.6            100%                   88%
        coverage >= 0.7                    88%                  100%

    Thresholds from 0.4 to 0.6 all score identically - the lowest answerable
    query sits at 0.62 and seven of eight unanswerable ones at or below 0.33 -
    so 0.5 sits in the middle of a plateau rather than on a knife edge. It costs
    nothing in false refusals on that set.

    The one unanswerable query that still gets through scores 0.60: "the default
    port for a PostgreSQL server connection" genuinely shares most of its
    vocabulary with machine-learning prose. Term overlap cannot separate that,
    and raising the threshold to catch it starts refusing real questions, which
    is the worse error.

    Set to `0.0` for the old any-word behaviour, or to `1.0` to demand a chunk
    containing every content word.
    """

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
    dedupe_threshold: float = 0.9
    """Drop a selected chunk whose wording overlaps an already-selected one by
    more than this fraction. 1.0 disables it.

    Near-duplicate documents are normal — a revision B of a bulletin, a contract
    and its appendix copy, the same policy on two intranet pages. Without this,
    a top-3 fills with the same passage three times, which wastes the context
    budget and makes the answer look well-sourced when it has one source.

    Token-overlap based rather than MMR because it needs no vectors, so it works
    on the lexical-only path and costs nothing."""

    guardrails: GuardrailConfig = field(default_factory=GuardrailConfig)
    """Injection defense. On by default, because the threat arrives with the
    documents and a pipeline that trusts its corpus is only safe until the
    corpus changes."""

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
    injection_flags: Sequence[str] = field(default_factory=list)
    """Injection findings in the retrieved sources, rendered. Non-empty means a
    document tried to issue instructions to the model."""

    policy_flags: Sequence[str] = field(default_factory=list)
    """Output-policy violations: the answer echoing instructions, or emitting a
    URL that appeared in no source."""

    unverified_markers: Sequence[int] = field(default_factory=list)
    """Markers the model cited that were never shown to it. Non-empty means the
    model invented a source, which is the single most important thing to surface
    and the easiest to miss."""

    @property
    def pages(self) -> list[int]:
        return sorted({c.page for c in self.citations})
