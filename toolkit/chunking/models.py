"""Data contracts for provenance-preserving chunking."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from ..core.models import Chunk, Document


@dataclass
class ChunkConfig:
    max_tokens: int = 512
    """Hard ceiling per chunk. Only a single unsplittable sentence may exceed it,
    and when that happens the chunk is flagged in its metadata rather than
    silently truncated — losing text is worse than an oversized chunk."""

    min_tokens: int | None = None
    """A trailing chunk smaller than this is merged back into its predecessor,
    provided the merge stays under max_tokens. Orphan fragments retrieve badly:
    they carry too little context to be judged relevant.

    Defaults to an eighth of `max_tokens`, capped at 48, for the same reason
    `overlap_tokens` does: a fixed default that silently exceeds a lowered
    `max_tokens` turns a reasonable override into a crash."""

    overlap_tokens: int | None = None
    """Tail of the previous chunk repeated at the head of the next, so a fact
    sitting on a boundary appears whole in at least one chunk. Overlap is applied
    in whole sentences, never mid-sentence.

    Defaults to a quarter of `max_tokens`, capped at 64. A fixed default would be
    self-contradictory the moment someone lowered `max_tokens` below it, which is
    a footgun in a config object whose whole job is to be overridden piecemeal.
    Set 0 to disable."""

    split_on_heading: bool = True
    """Start a new chunk whenever a heading is encountered. Sections are the
    author's own semantic boundaries; ignoring them to fill a token budget
    throws away the best free signal in the document."""

    include_heading_path: bool = True
    """Prefix each chunk with its heading breadcrumb ('Manual > Safety >
    Voltage'). Cheap, and it repairs the single most common RAG failure: a chunk
    that says 'it must not exceed 40V' with no indication of what 'it' is."""

    heading_separator: str = " > "
    heading_prefix_separator: str = "\n\n"

    keep_furniture: bool = False
    """Include page headers and footers. Almost never wanted."""

    token_counter: Callable[[str], int] | None = None
    """Real tokenizer when accuracy matters (`tiktoken`, a HF tokenizer). The
    default is a character heuristic, which is adequate for budgeting and keeps
    this module dependency-free."""

    semantic_threshold: float | None = None
    """When set (0..1) and an embedder is supplied, force a chunk boundary where
    cosine similarity between adjacent sentences falls below this value. Costs
    one embedding per sentence, so it is off by default."""

    @property
    def overlap(self) -> int:
        """`overlap_tokens` after defaulting. Always an int.

        The public field stays `int | None` because None is the honest way to say
        "derive it from max_tokens"; this accessor is what the algorithm uses, so
        no call site has to re-handle the None case."""
        return self.overlap_tokens if self.overlap_tokens is not None else 0

    @property
    def minimum(self) -> int:
        """`min_tokens` after defaulting. Always an int."""
        return self.min_tokens if self.min_tokens is not None else 0

    def __post_init__(self) -> None:
        if self.max_tokens < 16:
            raise ValueError("max_tokens must be at least 16")
        if self.overlap_tokens is None:
            self.overlap_tokens = min(64, self.max_tokens // 4)
        if self.overlap_tokens < 0:
            raise ValueError("overlap_tokens must be non-negative")
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        if self.min_tokens is None:
            self.min_tokens = min(48, self.max_tokens // 8)
        if self.min_tokens < 0:
            raise ValueError("min_tokens must be non-negative")
        if self.min_tokens >= self.max_tokens:
            raise ValueError("min_tokens must be smaller than max_tokens")
        if self.semantic_threshold is not None and not 0.0 <= self.semantic_threshold <= 1.0:
            raise ValueError("semantic_threshold must be in [0, 1]")


@dataclass
class ChunkRequest:
    document: Document
    config: ChunkConfig = field(default_factory=ChunkConfig)


@dataclass
class ChunkResult:
    chunks: Sequence[Chunk]
    oversized: Sequence[str]
    """chunk_ids that exceeded max_tokens because a single sentence could not be
    split. Surfaced rather than hidden, because it usually means the source has
    a table or a code block that should be handled differently."""
    token_estimates: dict[str, int]
    """chunk_id -> token count, using whichever counter was configured."""
