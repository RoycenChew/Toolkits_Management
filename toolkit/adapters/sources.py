"""DocumentSource adapters.

Three implementations, deliberately spanning the whole cost range:

* `PlainTextSource` — stdlib, no geometry. Proves the port works when the
  backend knows nothing about layout.
* `PdfPlumberSource` — text coordinates only, no ML, runs anywhere. Feeds the
  spans into `doc_layout`, which is where reading order and headings come from.
* `DoclingSource` — full ML pipeline with table structure.

The first two are what make the abstraction real. A port validated only against
Docling would have quietly inherited Docling's assumptions.
"""
from __future__ import annotations

import os
import re
import time
from collections.abc import Sequence
from typing import Any

from ..core.errors import AdapterError, MissingDependency
from ..core.limits import (
    ScreeningFailure,
    ScreeningLimits,
    ScreeningRejected,
    ScreeningResult,
)
from ..core.models import BBox, Block, BlockType, Document, Provenance
from ..core.text import normalise_text
from ..doc_layout import (
    WORD_GAP_RATIO,
    DocLayoutComponent,
    LayoutConfig,
    LayoutRequest,
    TextSpan,
)
from ..doc_layout import BBox as LayoutBBox
from ..doc_layout import BlockType as LayoutBlockType

_LAYOUT_TO_CORE = {
    LayoutBlockType.HEADING: BlockType.HEADING,
    LayoutBlockType.PARAGRAPH: BlockType.PARAGRAPH,
    LayoutBlockType.LIST_ITEM: BlockType.LIST_ITEM,
    LayoutBlockType.CAPTION: BlockType.CAPTION,
    LayoutBlockType.PAGE_HEADER: BlockType.PAGE_HEADER,
    LayoutBlockType.PAGE_FOOTER: BlockType.PAGE_FOOTER,
}


def _core_bbox(box: LayoutBBox) -> BBox:
    return BBox(box.x0, box.y0, box.x1, box.y1)


def _screen_file(path: str, limits: ScreeningLimits) -> ScreeningResult:
    """Size and readability checks every source shares.

    Done on the filesystem before a parser is handed the path, because the only
    cheap moment to refuse a 4 GB file is before anything opens it.
    """
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        return ScreeningResult(
            passed=False,
            reason=ScreeningFailure.UNREADABLE,
            detail=str(exc),
        )
    if size < limits.min_bytes:
        return ScreeningResult(
            passed=False, reason=ScreeningFailure.EMPTY, detail="file is empty",
            size_bytes=size,
        )
    if size > limits.max_bytes:
        return ScreeningResult(
            passed=False,
            reason=ScreeningFailure.TOO_LARGE,
            detail=str(size) + " bytes exceeds the " + str(limits.max_bytes) + " byte cap",
            size_bytes=size,
        )
    return ScreeningResult(passed=True, size_bytes=size)


class PlainTextSource:
    """Reads .txt and .md. No geometry, one synthetic page.

    Included because it is the honest lower bound of the port: a source that
    knows nothing about layout must still produce a usable Document. Markdown
    headings are recognised, because doing so costs one regex and makes the
    chunker behave sensibly on README-shaped input.
    """

    extensions = (".txt", ".md", ".markdown", ".rst")

    def supports(self, path: str) -> bool:
        return path.lower().endswith(self.extensions)

    def screen(self, path: str, limits: ScreeningLimits | None = None) -> ScreeningResult:
        return _screen_file(path, limits or ScreeningLimits())

    def load(self, path: str, limits: ScreeningLimits | None = None) -> Document:
        self.screen(path, limits or ScreeningLimits()).raise_if_rejected()
        with open(path, "rb") as handle:
            raw = handle.read()
        # Normalised at the boundary, keeping newlines because this source is
        # line-structured. Every downstream component then sees clean text.
        text = normalise_text(
            raw.decode("utf-8", errors="replace"), collapse_whitespace=False
        )
        blocks: list[Block] = []
        paragraph: list[str] = []

        def flush() -> None:
            if paragraph:
                joined = " ".join(" ".join(paragraph).split())
                if joined:
                    blocks.append(
                        Block(joined, BlockType.PARAGRAPH, Provenance(page=1))
                    )
                paragraph.clear()

        # Line-oriented rather than paragraph-oriented. Markdown does not require
        # a blank line after a heading — `## Voltage\nThe supply must not...` is
        # ordinary and extremely common. Splitting on blank lines first swallows
        # the heading into the paragraph, which silently destroys every heading
        # level and every breadcrumb downstream.
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                flush()
                continue

            heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
            if heading:
                flush()
                blocks.append(
                    Block(
                        text=heading.group(2).strip(),
                        type=BlockType.HEADING,
                        level=len(heading.group(1)),
                        provenance=Provenance(page=1),
                    )
                )
                continue

            item = re.match(r"^[-*+]\s+(.+)$", stripped)
            if item:
                flush()
                blocks.append(
                    Block(item.group(1).strip(), BlockType.LIST_ITEM, Provenance(page=1))
                )
                continue

            paragraph.append(stripped)
        flush()

        return Document(
            doc_id=Document.id_from_bytes(raw),
            blocks=blocks,
            source_uri=os.path.abspath(path),
            page_count=1,
            metadata={"source": "plaintext"},
        )


class PdfPlumberSource:
    """PDF text coordinates, structured by `doc_layout`. No models, no GPU.

    This is the default PDF path and the one that works on a laptop with no
    network. It cannot read scans (there is no OCR here) and it does not
    reconstruct tables — for either of those, use DoclingSource.
    """

    def __init__(
        self,
        config: LayoutConfig | None = None,
        x_tolerance_ratio: float = WORD_GAP_RATIO,
    ) -> None:
        self._config = config or LayoutConfig()
        self._layout = DocLayoutComponent()
        self._x_tolerance_ratio = x_tolerance_ratio

    def supports(self, path: str) -> bool:
        return path.lower().endswith(".pdf")

    def screen(self, path: str, limits: ScreeningLimits | None = None) -> ScreeningResult:
        """Size first, then page count once the container is open.

        Page count is the check that matters for PDFs: a 2 MB file can declare
        40,000 pages, so a size cap alone does not bound the work.
        """
        effective = limits or ScreeningLimits()
        result = _screen_file(path, effective)
        if not result.passed:
            return result
        try:
            import pdfplumber  # type: ignore
        except ImportError:
            # Cannot count pages without the backend; the size check still ran.
            return result
        try:
            with pdfplumber.open(path) as pdf:
                pages = len(pdf.pages)
        except Exception as exc:  # noqa: BLE001 - screening boundary
            message = str(exc).lower()
            reason = (
                ScreeningFailure.ENCRYPTED
                if "password" in message or "encrypt" in message
                else ScreeningFailure.UNREADABLE
            )
            return ScreeningResult(
                passed=False, reason=reason, detail=str(exc)[:200],
                size_bytes=result.size_bytes,
            )
        if pages > effective.max_pages:
            return ScreeningResult(
                passed=False,
                reason=ScreeningFailure.TOO_MANY_PAGES,
                detail=str(pages) + " pages exceeds the " + str(effective.max_pages) + " page cap",
                size_bytes=result.size_bytes,
                page_count=pages,
            )
        return ScreeningResult(
            passed=True, size_bytes=result.size_bytes, page_count=pages
        )

    def load(self, path: str, limits: ScreeningLimits | None = None) -> Document:
        effective = limits or ScreeningLimits()
        self.screen(path, effective).raise_if_rejected()
        try:
            import pdfplumber  # type: ignore
        except ImportError as exc:
            raise MissingDependency("pdfplumber", "docs") from exc

        spans: list[TextSpan] = []
        page_sizes: dict[int, tuple[float, float]] = {}
        rotated_glyphs = 0
        with open(path, "rb") as handle:
            raw = handle.read()

        started = time.monotonic()
        with pdfplumber.open(path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                # Between-page check. Cannot interrupt a single pathological
                # page — that needs a subprocess — but it bounds a document that
                # is slow because it is long.
                if time.monotonic() - started > effective.max_seconds:
                    raise ScreeningRejected(
                        ScreeningFailure.TOO_SLOW,
                        "exceeded "
                        + str(effective.max_seconds)
                        + "s after "
                        + str(page_number - 1)
                        + " pages",
                    )
                page_sizes[page_number] = (float(page.width), float(page.height))
                # Rotated glyphs are a SEPARATE text flow, not noise. Reading
                # order is recovered by sorting spans by position, which is only
                # meaningful within one orientation: a vertical stamp down the
                # left margin has no common reading order with the horizontal
                # lines beside it, so mixing them interleaves its characters
                # into words. Every arXiv PDF carries such a stamp, and it was
                # corrupting body text ("an tc abelian surface", "routinely extc
                # ceeding") and displacing a line of one abstract entirely.
                #
                # `doc_layout` has no notion of orientation, so the honest thing
                # is to exclude rotated text and report how much was excluded,
                # rather than silently weaving it into the prose. See
                # `rotated_glyphs_excluded` in this document's metadata.
                upright_page = page.filter(
                    lambda obj: obj.get("upright", True)
                )
                rotated_glyphs += sum(
                    1 for ch in page.chars if not ch.get("upright", True)
                )
                words = upright_page.extract_words(
                    extra_attrs=["size", "fontname"],
                    use_text_flow=False,
                    x_tolerance_ratio=self._x_tolerance_ratio,
                )
                for word in words:
                    text = normalise_text(str(word.get("text", "")))
                    if not text:
                        continue
                    font = str(word.get("fontname", ""))
                    # pdfplumber's `top`/`bottom` are already y-down, which is
                    # what doc_layout expects. Using x0/y0 here instead would
                    # silently invert the reading order.
                    spans.append(
                        TextSpan(
                            text=text,
                            bbox=LayoutBBox(
                                float(word["x0"]),
                                float(word["top"]),
                                float(word["x1"]),
                                float(word["bottom"]),
                            ),
                            page=page_number,
                            font_size=float(word.get("size") or 10.0),
                            bold="bold" in font.lower(),
                            italic="italic" in font.lower() or "oblique" in font.lower(),
                            font_name=font,
                        )
                    )

        if not spans:
            raise AdapterError(
                "no extractable text in "
                + os.path.basename(path)
                + "; this is probably a scanned PDF and needs OCR"
            )

        result = self._layout.execute(
            LayoutRequest(spans=spans, page_sizes=page_sizes, config=self._config)
        )
        blocks = [
            Block(
                text=block.text,
                type=_LAYOUT_TO_CORE.get(block.type, BlockType.OTHER),
                provenance=Provenance(
                    page=block.provenance.page,
                    bbox=_core_bbox(block.provenance.bbox),
                ),
                level=block.level,
                metadata={"column": block.column, "font_size": block.font_size},
            )
            for block in result.blocks
        ]
        return Document(
            doc_id=Document.id_from_bytes(raw),
            blocks=blocks,
            source_uri=os.path.abspath(path),
            page_count=len(page_sizes),
            metadata={
                "source": "pdfplumber+doc_layout",
                "body_font_size": result.body_font_size,
                "columns_per_page": result.columns_per_page,
                "spans_deduplicated": result.spans_deduplicated,
                # Non-zero means text was excluded: rotated stamps, margin
                # annotations, or a landscape page whose whole body is rotated.
                # A large count relative to the document's size means real
                # content was dropped and this is the wrong source for it.
                "rotated_glyphs_excluded": rotated_glyphs,
            },
        )


class DoclingSource:
    """Docling's full pipeline: layout models, OCR, and table structure.

    Handles scans and reconstructs tables, which the pdfplumber path cannot.
    Costs a model download on first use and is substantially slower. Docling is
    MIT-licensed, which is why it is the ML default here rather than Marker
    (GPL-family) or MinerU (AGPL).
    """

    _TYPE_HINTS = (
        ("title", BlockType.HEADING),
        ("section_header", BlockType.HEADING),
        ("header", BlockType.HEADING),
        ("list_item", BlockType.LIST_ITEM),
        ("table", BlockType.TABLE),
        ("caption", BlockType.CAPTION),
        ("formula", BlockType.FORMULA),
        ("code", BlockType.CODE),
        ("page_header", BlockType.PAGE_HEADER),
        ("page_footer", BlockType.PAGE_FOOTER),
        ("footnote", BlockType.OTHER),
    )

    def __init__(self, converter: Any | None = None) -> None:
        self._converter = converter

    def supports(self, path: str) -> bool:
        return path.lower().endswith(
            (".pdf", ".docx", ".pptx", ".html", ".png", ".jpg", ".jpeg", ".tiff")
        )

    def screen(self, path: str, limits: ScreeningLimits | None = None) -> ScreeningResult:
        """Size only. Docling opens many formats and counting units before
        conversion would mean opening each one twice; the size cap is the
        portable guard, and the caller can pre-screen with PdfPlumberSource for
        a page count when the input is a PDF."""
        return _screen_file(path, limits or ScreeningLimits())

    def _map_type(self, label: str) -> BlockType:
        lowered = label.lower()
        for needle, block_type in self._TYPE_HINTS:
            if needle in lowered:
                return block_type
        if "text" in lowered or "paragraph" in lowered:
            return BlockType.PARAGRAPH
        return BlockType.OTHER

    def load(self, path: str, limits: ScreeningLimits | None = None) -> Document:
        self.screen(path, limits or ScreeningLimits()).raise_if_rejected()
        converter = self._converter
        if converter is None:
            try:
                from docling.document_converter import DocumentConverter  # type: ignore
            except ImportError as exc:
                raise MissingDependency("docling", "docs") from exc
            converter = DocumentConverter()
            self._converter = converter

        try:
            result = converter.convert(path)
        except Exception as exc:  # noqa: BLE001 - adapter boundary
            raise AdapterError("docling failed on " + os.path.basename(path)) from exc

        doc = getattr(result, "document", result)
        blocks: list[Block] = []
        pages: set[int] = set()

        for item in self._iter_items(doc):
            text = normalise_text(getattr(item, "text", "") or "")
            if not text:
                continue
            label = str(getattr(item, "label", "") or "")
            page, bbox = self._locate(item)
            if page:
                pages.add(page)
            blocks.append(
                Block(
                    text=text,
                    type=self._map_type(label),
                    provenance=Provenance(page=page or 1, bbox=bbox),
                    level=getattr(item, "level", None),
                    metadata={"docling_label": label},
                )
            )

        with open(path, "rb") as handle:
            raw = handle.read()
        return Document(
            doc_id=Document.id_from_bytes(raw),
            blocks=blocks,
            source_uri=os.path.abspath(path),
            page_count=len(pages) or 1,
            metadata={"source": "docling"},
        )

    def _iter_items(self, doc: Any) -> Sequence[Any]:
        """Docling's traversal API has moved between versions, so try the
        documented entry points in order rather than pinning to one."""
        for attribute in ("iterate_items", "texts"):
            member = getattr(doc, attribute, None)
            if member is None:
                continue
            try:
                items = member() if callable(member) else member
            except TypeError:
                continue
            out = []
            for entry in items:
                # iterate_items yields (item, level) in some versions.
                out.append(entry[0] if isinstance(entry, tuple) else entry)
            if out:
                return out
        raise AdapterError(
            "could not traverse the Docling document; the installed docling"
            " version exposes neither iterate_items() nor .texts"
        )

    def _locate(self, item: Any) -> tuple[int, BBox | None]:
        provenance = getattr(item, "prov", None)
        if not provenance:
            return 0, None
        first = provenance[0] if isinstance(provenance, (list, tuple)) else provenance
        page = int(getattr(first, "page_no", 0) or 0)
        raw = getattr(first, "bbox", None)
        if raw is None:
            return page, None
        try:
            left = float(raw.l)
            top = float(raw.t)
            right = float(raw.r)
            bottom = float(raw.b)
        except (AttributeError, TypeError, ValueError):
            return page, None
        # Docling reports bottom-left origin for some backends, so normalise to
        # y-down by ordering rather than trusting the field names.
        y0, y1 = (top, bottom) if top <= bottom else (bottom, top)
        x0, x1 = (left, right) if left <= right else (right, left)
        return page, BBox(x0, y0, x1, y1)


def default_sources() -> list[Any]:
    """The zero-dependency-first ordering used by `load_document`."""
    return [PlainTextSource(), PdfPlumberSource()]


def load_document(
    path: str,
    sources: Sequence[Any] | None = None,
    limits: ScreeningLimits | None = None,
) -> Document:
    """Dispatch to the first source that claims the path, screening first."""
    for source in sources or default_sources():
        if source.supports(path):
            return source.load(path, limits or ScreeningLimits())
    raise AdapterError("no DocumentSource supports " + os.path.basename(path))


__all__ = [
    "DoclingSource",
    "PdfPlumberSource",
    "PlainTextSource",
    "default_sources",
    "load_document",
]
