"""The shared contracts. This module is the spine of the toolkit.

Every framework ships its own `Document`. Owning one here is what lets a
pipeline swap pdfplumber for Docling for an OCR engine without a single line of
pipeline code changing. Nothing in this module imports a vendor SDK, a model, or
a database; it is data contracts and nothing else.

Provenance is deliberately part of the core rather than an optional extra. A
chunk that cannot say which page and which region it came from cannot be cited,
highlighted, or verified, and an answer you cannot verify is a demo rather than
a system.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class BBox:
    """Axis-aligned region on a page, y-down (y0 is the top edge).

    Adapters are responsible for normalising into y-down before constructing
    one. Extractors disagree about this and a silent flip inverts reading order.
    """

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError("bbox must satisfy x0 <= x1 and y0 <= y1")

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    def merge(self, other: BBox) -> BBox:
        return BBox(
            min(self.x0, other.x0),
            min(self.y0, other.y0),
            max(self.x1, other.x1),
            max(self.y1, other.y1),
        )

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


class BlockType(str, Enum):
    """What a block is, independent of which backend produced it.

    Kept deliberately small. A backend that distinguishes more categories maps
    them down; a backend that distinguishes fewer leaves everything PARAGRAPH.
    A vocabulary that only one backend can populate is not a shared contract.
    """

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    CAPTION = "caption"
    PAGE_HEADER = "page_header"
    PAGE_FOOTER = "page_footer"
    CODE = "code"
    FORMULA = "formula"
    OTHER = "other"


FURNITURE = frozenset({BlockType.PAGE_HEADER, BlockType.PAGE_FOOTER})
"""Block types that are page decoration rather than content. Excluded from
chunking and markdown by default, but retained in the Document so that a caller
who wants them can still reach them."""


@dataclass(frozen=True)
class Provenance:
    """Where a piece of content came from.

    `page` is 1-based. `bbox` is None for sources that have no geometry at all
    (a plain text file, a database row); everything else should populate it.
    """

    page: int
    bbox: BBox | None = None
    source_id: str = ""
    """Backend-specific handle back to the original unit, when one exists."""

    def merge(self, other: Provenance) -> Provenance:
        """Combine two provenances from the same page into one covering region.

        Merging across pages is meaningless, so the earlier page wins and the
        geometry is dropped rather than inventing a box that spans a page break.
        """
        if self.page != other.page:
            earlier = self if self.page <= other.page else other
            return Provenance(page=earlier.page, bbox=None, source_id=earlier.source_id)
        if self.bbox is None or other.bbox is None:
            return Provenance(
                page=self.page,
                bbox=self.bbox or other.bbox,
                source_id=self.source_id or other.source_id,
            )
        return Provenance(
            page=self.page,
            bbox=self.bbox.merge(other.bbox),
            source_id=self.source_id or other.source_id,
        )


@dataclass(frozen=True)
class Block:
    """One semantic unit of a document, in reading order."""

    text: str
    type: BlockType = BlockType.PARAGRAPH
    provenance: Provenance | None = None
    level: int | None = None
    """Heading depth, 1-based. None for non-headings."""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_furniture(self) -> bool:
        return self.type in FURNITURE


@dataclass
class Document:
    """A parsed document: ordered blocks plus where they came from.

    `doc_id` should be stable across runs for the same input, because it is what
    a vector store row, a cache entry and a citation all key on. `from_bytes`
    derives one by content hash, which is the behaviour you almost always want.
    """

    doc_id: str
    blocks: Sequence[Block] = field(default_factory=list)
    source_uri: str = ""
    page_count: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @staticmethod
    def id_from_bytes(data: bytes, prefix: str = "doc") -> str:
        return prefix + ":" + hashlib.sha256(data).hexdigest()[:16]

    @staticmethod
    def id_from_text(text: str, prefix: str = "doc") -> str:
        return Document.id_from_bytes(text.encode("utf-8"), prefix)

    def content_blocks(self) -> list[Block]:
        """Blocks excluding page furniture. The usual input to chunking."""
        return [b for b in self.blocks if not b.is_furniture]

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.content_blocks())

    def to_markdown(self) -> str:
        parts: list[str] = []
        for block in self.content_blocks():
            if block.type is BlockType.HEADING:
                parts.append("#" * min(block.level or 1, 6) + " " + block.text)
            elif block.type is BlockType.LIST_ITEM:
                parts.append("- " + block.text)
            elif block.type is BlockType.CAPTION:
                parts.append("*" + block.text + "*")
            elif block.type is BlockType.CODE:
                parts.append("```\n" + block.text + "\n```")
            else:
                parts.append(block.text)
        return "\n\n".join(parts)


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit, carrying every provenance it was built from.

    `provenances` is a list because a chunk may legitimately span two blocks on
    two pages. Collapsing it to a single location would be a lie, and the lie
    surfaces exactly when a user clicks a citation.
    """

    chunk_id: str
    text: str
    doc_id: str
    index: int
    provenances: Sequence[Provenance] = field(default_factory=list)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def pages(self) -> list[int]:
        return sorted({p.page for p in self.provenances})


@dataclass(frozen=True)
class Usage:
    """What a model call cost. Summable, so a pipeline can report a total."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            cached=self.cached and other.cached,
        )


@dataclass(frozen=True)
class Message:
    role: str
    """'system', 'user' or 'assistant'."""
    content: str


@dataclass(frozen=True)
class Completion:
    text: str
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    finish_reason: str = ""


@dataclass(frozen=True)
class SearchHit:
    """One result from a vector store or lexical index.

    The score scale is backend-specific and deliberately not normalised here —
    that is `hybrid_ranker`'s job, and normalising twice loses information.
    """

    chunk_id: str
    score: float
    text: str = ""
    doc_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


__all__ = [
    "BBox",
    "Block",
    "BlockType",
    "Chunk",
    "Completion",
    "Document",
    "FURNITURE",
    "Message",
    "Provenance",
    "SearchHit",
    "Usage",
]
