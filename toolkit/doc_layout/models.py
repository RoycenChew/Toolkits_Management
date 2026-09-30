"""Data contracts for the document layout / reading-order component."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum


@dataclass(frozen=True)
class BBox:
    """Axis-aligned box in PDF-ish coordinates with y growing downwards.

    If your extractor emits y-up coordinates (pdfminer, some PyMuPDF modes),
    flip them before constructing spans, or reading order will come out upside
    down. This is the single most common integration mistake.
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

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2.0

    def merge(self, other: BBox) -> BBox:
        return BBox(
            min(self.x0, other.x0),
            min(self.y0, other.y0),
            max(self.x1, other.x1),
            max(self.y1, other.y1),
        )

    def intersection_area(self, other: BBox) -> float:
        dx = min(self.x1, other.x1) - max(self.x0, other.x0)
        dy = min(self.y1, other.y1) - max(self.y0, other.y0)
        if dx <= 0 or dy <= 0:
            return 0.0
        return dx * dy

    def iou(self, other: BBox) -> float:
        inter = self.intersection_area(other)
        union = self.area + other.area - inter
        return 0.0 if union <= 0 else inter / union

    def contained_fraction(self, other: BBox) -> float:
        """How much of *self* lies inside *other*."""
        if self.area <= 0:
            return 0.0
        return self.intersection_area(other) / self.area

    def vertical_overlap(self, other: BBox) -> float:
        """Shared vertical extent as a fraction of the shorter box's height."""
        dy = min(self.y1, other.y1) - max(self.y0, other.y0)
        shorter = min(self.height, other.height)
        if shorter <= 0:
            return 1.0 if dy >= 0 else 0.0
        return max(0.0, dy) / shorter


@dataclass(frozen=True)
class TextSpan:
    """One run of text with uniform styling, as emitted by a PDF text extractor
    or an OCR engine. This is the component's only input primitive."""

    text: str
    bbox: BBox
    page: int
    font_size: float = 10.0
    bold: bool = False
    italic: bool = False
    font_name: str = ""


class BlockType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    PAGE_HEADER = "page_header"
    PAGE_FOOTER = "page_footer"
    CAPTION = "caption"


@dataclass(frozen=True)
class Provenance:
    """Where a block came from. Without this you cannot cite, highlight, or
    verify an extraction, which is why it is part of the core contract rather
    than an optional extra."""

    page: int
    bbox: BBox
    span_indices: Sequence[int]
    """Indices into the original input span list."""


@dataclass(frozen=True)
class Block:
    text: str
    type: BlockType
    provenance: Provenance
    level: int | None = None
    """Heading depth, 1-based. None for non-headings."""
    reading_order: int = 0
    font_size: float = 0.0
    column: int = 0


@dataclass
class LayoutConfig:
    line_overlap_threshold: float = 0.5
    """Vertical overlap fraction above which two spans are the same line."""
    paragraph_gap_multiplier: float = 1.6
    """A vertical gap bigger than this many median line heights breaks a block."""
    column_gap_multiplier: float = 1.2
    """A vertical whitespace band wider than this many median char widths can
    split a region into columns during XY-cut."""
    min_column_width_ratio: float = 0.25
    """Reject a column split that would leave a region narrower than this
    fraction of the page. This single number is what separates a real text
    column from a table gutter: body columns are wide (a two-column A4 page
    gives roughly 0.4 each), table cells are narrow. Lower it only for layouts
    with three or more genuine text columns."""
    min_gutter_ratio: float = 0.02
    """A gutter narrower than this fraction of the page is inter-word spacing or
    a cell boundary, never a column break."""
    heading_size_ratio: float = 1.12
    """A line this much larger than body text is a heading candidate."""
    heading_max_words: int = 25
    """Long lines are prose even when they are bold or large."""
    header_footer_band: float = 0.08
    """Fraction of page height at top/bottom eligible to be header/footer."""
    header_footer_min_pages: int = 3
    """Repeat count needed before a recurring line is treated as boilerplate."""
    duplicate_iou: float = 0.85
    """Overlapping spans with the same text above this IoU are deduplicated.
    OCR-over-text-layer pipelines produce these constantly."""
    drop_headers_footers: bool = False
    """Classify only, or remove them from the output entirely."""
    caption_max_words: int = 30

    def __post_init__(self) -> None:
        if not 0.0 < self.line_overlap_threshold <= 1.0:
            raise ValueError("line_overlap_threshold must be in (0, 1]")
        if self.paragraph_gap_multiplier <= 0:
            raise ValueError("paragraph_gap_multiplier must be positive")


@dataclass
class LayoutRequest:
    spans: Sequence[TextSpan]
    page_sizes: dict[int, tuple[float, float]] = field(default_factory=dict)
    """page -> (width, height). Inferred from span extents when absent, which is
    slightly less accurate for header/footer banding."""
    config: LayoutConfig = field(default_factory=LayoutConfig)


@dataclass
class LayoutResult:
    blocks: Sequence[Block]
    markdown: str
    body_font_size: float
    """The inferred body text size, in points. The reference for every
    heading decision."""
    columns_per_page: dict[int, int]
    spans_deduplicated: int
