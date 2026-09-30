"""Reading-order recovery and structure inference over positioned text spans.

This is the *algorithmic* half of a document parser, separated from the ML half.
Layout detection models (Docling's layout predictor, Surya, DocLayout-YOLO) tell
you where regions are; they do not tell you what order to read them in, which
overlapping boxes to discard, which lines are the same paragraph, or what heading
level a line is. Those steps are deterministic geometry and statistics, and they
are the part worth owning because they need no weights, no GPU and no network.

Reimplemented from the published algorithms:

* Recursive XY-cut (Nagy & Seth) for column and region decomposition, with the
  sliver guard that keeps table gutters from being read as column breaks.
* Region containment / overlap resolution, the job of Docling's layout
  post-processor: OCR run over a page that already had a text layer yields two
  copies of everything.
* Font-statistics heading inference, the idea behind Marker's heading
  processors: body size is the mode of character-weighted sizes, and heading
  levels are the distinct larger sizes ranked descending.
* Recurring-line detection for headers and footers, which is why page furniture
  does not end up interleaved with prose.

Input is a flat list of TextSpan. Anything that can emit text plus a bounding box
can drive it: pdfplumber, PyMuPDF, pdfminer, PaddleOCR, Tesseract TSV, Textract.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Sequence

from .models import (
    BBox,
    Block,
    BlockType,
    LayoutConfig,
    LayoutRequest,
    LayoutResult,
    Provenance,
    TextSpan,
)

_BULLET = re.compile(r"^\s*(?:[-â€¢â€£â—¦âƒâˆ™*]|\(?\d{1,2}[.)]|[a-z][.)])\s+")
_CAPTION = re.compile(
    r"^\s*(?:fig(?:ure)?|table|tbl|exhibit|chart|listing|appendix)\s*\.?\s*"
    r"(?:\d+|[ivxlc]+|[A-Z])\b",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"[.!?:;\"â€â€™)]\s*$")


class _Line:
    """A horizontal run of spans that belong to the same visual line."""

    __slots__ = ("spans", "indices", "bbox", "page")

    def __init__(self, span: TextSpan, index: int) -> None:
        self.spans: list[TextSpan] = [span]
        self.indices: list[int] = [index]
        self.bbox = span.bbox
        self.page = span.page

    def add(self, span: TextSpan, index: int) -> None:
        self.spans.append(span)
        self.indices.append(index)
        self.bbox = self.bbox.merge(span.bbox)

    def finalise(self) -> None:
        order = sorted(range(len(self.spans)), key=lambda i: self.spans[i].bbox.x0)
        self.spans = [self.spans[i] for i in order]
        self.indices = [self.indices[i] for i in order]

    @property
    def text(self) -> str:
        out = ""
        prev = None
        for span in self.spans:
            piece = span.text
            if prev is not None:
                gap = span.bbox.x0 - prev.bbox.x1
                # Insert a space only when the visual gap is wide enough to be
                # one. Extractors split mid-word constantly.
                needs_space = gap > 0.25 * max(prev.font_size, 1.0)
                if needs_space and not out.endswith(" ") and not piece.startswith(" "):
                    out += " "
            out += piece
            prev = span
        return " ".join(out.split())

    @property
    def font_size(self) -> float:
        # Character-weighted: a short bold run should not redefine the line.
        total = sum(len(s.text) for s in self.spans)
        if total == 0:
            return self.spans[0].font_size
        return sum(s.font_size * len(s.text) for s in self.spans) / total

    @property
    def bold_fraction(self) -> float:
        total = sum(len(s.text) for s in self.spans)
        if total == 0:
            return 0.0
        return sum(len(s.text) for s in self.spans if s.bold) / total


class DocLayoutComponent:
    """Positioned text spans in, ordered semantic blocks plus markdown out."""

    def execute(self, input_data: LayoutRequest) -> LayoutResult:
        cfg = input_data.config
        spans = [
            (i, s)
            for i, s in enumerate(input_data.spans)
            if s.text and s.text.strip() and s.bbox.width >= 0
        ]
        if not spans:
            return LayoutResult([], "", 0.0, {}, 0)

        kept, dropped = self._deduplicate(spans, cfg)
        body_size = self._body_font_size([s for _, s in kept])
        page_sizes = self._page_sizes(kept, input_data.page_sizes)

        blocks: list[Block] = []
        columns_per_page: dict[int, int] = {}
        by_page: dict[int, list[tuple[int, TextSpan]]] = defaultdict(list)
        for index, span in kept:
            by_page[span.page].append((index, span))

        # Columns must be separated before lines are assembled. Two spans in
        # different columns can sit at exactly the same y, so line grouping on a
        # whole page would splice the left and right columns into single lines
        # and destroy the reading order. Cut first, then group: that is the
        # original XY-cut ordering, and the reason it is ordered that way.
        page_columns = {
            page: self._xy_cut(items, page_sizes.get(page), cfg)
            for page, items in sorted(by_page.items())
        }
        page_column_lines = {
            page: [self._lines(column, cfg) for column in columns]
            for page, columns in page_columns.items()
        }
        # Header/footer detection is cross-page by nature, so it runs over every
        # page's lines before any page is finalised.
        boilerplate = self._boilerplate(
            {
                page: [line for column in columns for line in column]
                for page, columns in page_column_lines.items()
            },
            page_sizes,
            cfg,
        )

        for page in sorted(page_column_lines):
            columns = page_column_lines[page]
            columns_per_page[page] = len(columns)
            for column_index, column_lines in enumerate(columns):
                for group in self._group_paragraphs(column_lines, cfg):
                    block = self._classify(
                        group, body_size, page_sizes.get(page), boilerplate, cfg
                    )
                    if block is not None:
                        blocks.append(
                            Block(
                                text=block.text,
                                type=block.type,
                                provenance=block.provenance,
                                level=block.level,
                                reading_order=0,
                                font_size=block.font_size,
                                column=column_index,
                            )
                        )

        blocks = self._assign_heading_levels(blocks, body_size)
        if cfg.drop_headers_footers:
            blocks = [
                b
                for b in blocks
                if b.type not in (BlockType.PAGE_HEADER, BlockType.PAGE_FOOTER)
            ]
        blocks = [
            Block(
                b.text,
                b.type,
                b.provenance,
                b.level,
                i,
                b.font_size,
                b.column,
            )
            for i, b in enumerate(blocks)
        ]

        return LayoutResult(
            blocks=blocks,
            markdown=self._to_markdown(blocks),
            body_font_size=body_size,
            columns_per_page=columns_per_page,
            spans_deduplicated=dropped,
        )

    # --- overlap resolution ---------------------------------------------

    def _deduplicate(
        self, spans: Sequence[tuple[int, TextSpan]], cfg: LayoutConfig
    ) -> tuple[list[tuple[int, TextSpan]], int]:
        """Drop spans that duplicate another span at nearly the same place.

        Two independent sources of duplicates: an OCR pass layered over an
        existing text layer, and extractors that emit both a line-level and a
        word-level span for the same text. Both are resolved the same way, by
        keeping the first occurrence and discarding later near-identical boxes.
        Grouping by page and normalised text keeps this near-linear instead of
        comparing every span to every other.
        """
        buckets: dict[tuple[int, str], list[BBox]] = defaultdict(list)
        kept: list[tuple[int, TextSpan]] = []
        dropped = 0
        for index, span in spans:
            key = (span.page, " ".join(span.text.split()).lower())
            if any(
                span.bbox.iou(seen) >= cfg.duplicate_iou
                or span.bbox.contained_fraction(seen) >= 0.95
                for seen in buckets[key]
            ):
                dropped += 1
                continue
            buckets[key].append(span.bbox)
            kept.append((index, span))
        return kept, dropped

    # --- statistics ------------------------------------------------------

    def _body_font_size(self, spans: Sequence[TextSpan]) -> float:
        """Body size is the character-weighted mode, rounded to half a point.

        The mean is wrong here: a title page or a long footnote skews it. The
        mode of characters (not of spans) is robust because body text is, by
        definition, most of the characters on the page.
        """
        counter: Counter[float] = Counter()
        for span in spans:
            counter[round(span.font_size * 2) / 2] += len(span.text.strip())
        if not counter:
            return 10.0
        return counter.most_common(1)[0][0] or 10.0

    def _page_sizes(
        self,
        spans: Sequence[tuple[int, TextSpan]],
        supplied: dict[int, tuple[float, float]],
    ) -> dict[int, tuple[float, float]]:
        inferred: dict[int, tuple[float, float]] = {}
        extents: dict[int, BBox] = {}
        for _, span in spans:
            extents[span.page] = (
                span.bbox
                if span.page not in extents
                else extents[span.page].merge(span.bbox)
            )
        for page, box in extents.items():
            inferred[page] = supplied.get(page, (box.x1, box.y1))
        for page, size in supplied.items():
            inferred.setdefault(page, size)
        return inferred

    # --- line assembly ---------------------------------------------------

    def _lines(
        self, items: Sequence[tuple[int, TextSpan]], cfg: LayoutConfig
    ) -> list[_Line]:
        ordered = sorted(items, key=lambda p: (p[1].bbox.y0, p[1].bbox.x0))
        lines: list[_Line] = []
        for index, span in ordered:
            placed = False
            # Only the most recent few lines can plausibly match, and checking
            # them in reverse keeps this linear for normal documents.
            for line in reversed(lines[-4:]):
                if (
                    span.bbox.vertical_overlap(line.bbox) >= cfg.line_overlap_threshold
                    and abs(span.bbox.center_y - line.bbox.center_y)
                    <= 0.6 * max(span.bbox.height, 1.0)
                ):
                    line.add(span, index)
                    placed = True
                    break
            if not placed:
                lines.append(_Line(span, index))
        for line in lines:
            line.finalise()
        lines.sort(key=lambda ln: (ln.bbox.y0, ln.bbox.x0))
        return lines

    # --- recursive XY-cut ------------------------------------------------

    def _xy_cut(
        self,
        items: Sequence[tuple[int, TextSpan]],
        page_size: tuple[float, float] | None,
        cfg: LayoutConfig,
    ) -> list[list[tuple[int, TextSpan]]]:
        """Split a region into columns by looking for a vertical whitespace gutter.

        Classic XY-cut alternates horizontal and vertical projection cuts. Here
        only the vertical cut is needed, because the horizontal structure is
        recovered afterwards by line assembly and paragraph grouping. The guard
        that matters in practice is `min_column_width_ratio`: without it every
        table gutter and every indented block is promoted to a column and the
        reading order shatters.
        """
        if len(items) < 4:
            return [list(items)]
        page_width = page_size[0] if page_size else max(s.bbox.x1 for _, s in items)
        if page_width <= 0:
            return [list(items)]

        gutter = self._find_gutter(items, page_width, cfg)
        if gutter is None:
            return [list(items)]

        left = [(i, s) for i, s in items if s.bbox.x1 <= gutter]
        right = [(i, s) for i, s in items if s.bbox.x0 >= gutter]
        # Spans straddling the gutter (a full-width heading, a spanning rule)
        # mean this is not a clean multi-column region after all.
        if len(left) + len(right) != len(items) or not left or not right:
            return [list(items)]

        out: list[list[tuple[int, TextSpan]]] = []
        for side in (left, right):
            side_width = max(s.bbox.x1 for _, s in side) - min(
                s.bbox.x0 for _, s in side
            )
            out.extend(self._xy_cut(side, (max(side_width, 1.0), 0.0), cfg))
        return out

    def _find_gutter(
        self,
        items: Sequence[tuple[int, TextSpan]],
        page_width: float,
        cfg: LayoutConfig,
    ) -> float | None:
        intervals = sorted((s.bbox.x0, s.bbox.x1) for _, s in items)
        median_char = max(
            0.5 * (sum(s.font_size for _, s in items) / len(items)), 1.0
        )
        min_gap = max(
            cfg.column_gap_multiplier * median_char,
            cfg.min_gutter_ratio * page_width,
        )
        min_width = cfg.min_column_width_ratio * page_width

        cursor = intervals[0][1]
        best_gap = 0.0
        best_cut = None
        for x0, x1 in intervals[1:]:
            if x0 - cursor > best_gap:
                left_width = cursor - intervals[0][0]
                right_width = max(i[1] for i in intervals) - x0
                if (
                    x0 - cursor >= min_gap
                    and left_width >= min_width
                    and right_width >= min_width
                ):
                    best_gap = x0 - cursor
                    best_cut = (cursor + x0) / 2.0
            cursor = max(cursor, x1)
        return best_cut

    # --- paragraph grouping ----------------------------------------------

    def _group_paragraphs(
        self, lines: Sequence[_Line], cfg: LayoutConfig
    ) -> list[list[_Line]]:
        """Merge consecutive lines into blocks, breaking on the signals that
        actually indicate a new block: a vertical gap larger than normal leading,
        a font-size change, a new bullet, or a previous line that ended on
        sentence-final punctuation while being noticeably short.
        """
        if not lines:
            return []
        heights = sorted(ln.bbox.height for ln in lines)
        median_height = heights[len(heights) // 2] or 1.0

        groups: list[list[_Line]] = [[lines[0]]]
        for prev, line in zip(lines, lines[1:]):
            gap = line.bbox.y0 - prev.bbox.y1
            size_change = abs(line.font_size - prev.font_size) > 0.6
            new_bullet = bool(_BULLET.match(line.text))
            prev_bullet = bool(_BULLET.match(prev.text))
            wide = max(ln.bbox.width for ln in lines) or 1.0
            short_and_ended = (
                prev.bbox.width < 0.75 * wide and bool(_SENTENCE_END.search(prev.text))
            )
            breaks = (
                gap > cfg.paragraph_gap_multiplier * median_height
                or size_change
                or new_bullet
                or (prev_bullet and not new_bullet and gap > 0.3 * median_height)
                or short_and_ended
            )
            if breaks:
                groups.append([line])
            else:
                groups[-1].append(line)
        return groups

    # --- classification ---------------------------------------------------

    def _boilerplate(
        self,
        page_lines: dict[int, list[_Line]],
        page_sizes: dict[int, tuple[float, float]],
        cfg: LayoutConfig,
    ) -> dict[str, BlockType]:
        """Find lines that recur across pages inside the top or bottom band.

        Page numbers change every page, so the text is normalised by replacing
        digit runs with a placeholder before counting. That single
        transformation is what makes 'Page 3 of 12' detectable as furniture.
        """
        if len(page_lines) < cfg.header_footer_min_pages:
            return {}
        top: Counter[str] = Counter()
        bottom: Counter[str] = Counter()
        for page, lines in page_lines.items():
            size = page_sizes.get(page)
            height = size[1] if size and size[1] > 0 else max(
                (ln.bbox.y1 for ln in lines), default=0.0
            )
            if height <= 0:
                continue
            band = cfg.header_footer_band * height
            seen_top: set[str] = set()
            seen_bottom: set[str] = set()
            for line in lines:
                key = re.sub(r"\d+", "#", line.text.strip().lower())
                if not key:
                    continue
                if line.bbox.y1 <= band:
                    seen_top.add(key)
                elif line.bbox.y0 >= height - band:
                    seen_bottom.add(key)
            top.update(seen_top)
            bottom.update(seen_bottom)

        threshold = cfg.header_footer_min_pages
        out: dict[str, BlockType] = {}
        for key, count in top.items():
            if count >= threshold:
                out[key] = BlockType.PAGE_HEADER
        for key, count in bottom.items():
            if count >= threshold:
                out.setdefault(key, BlockType.PAGE_FOOTER)
        return out

    def _classify(
        self,
        group: Sequence[_Line],
        body_size: float,
        page_size: tuple[float, float] | None,
        boilerplate: dict[str, BlockType],
        cfg: LayoutConfig,
    ) -> Block | None:
        text = " ".join(line.text for line in group).strip()
        if not text:
            return None
        bbox = group[0].bbox
        for line in group[1:]:
            bbox = bbox.merge(line.bbox)
        indices = [i for line in group for i in line.indices]
        provenance = Provenance(page=group[0].page, bbox=bbox, span_indices=indices)
        size = sum(ln.font_size * len(ln.text) for ln in group) / max(
            sum(len(ln.text) for ln in group), 1
        )
        words = len(text.split())

        key = re.sub(r"\d+", "#", text.lower())
        if len(group) == 1 and key in boilerplate:
            return Block(text, boilerplate[key], provenance, None, 0, size)

        if _BULLET.match(group[0].text):
            return Block(text, BlockType.LIST_ITEM, provenance, None, 0, size)

        if _CAPTION.match(text) and words <= cfg.caption_max_words:
            return Block(text, BlockType.CAPTION, provenance, None, 0, size)

        bold = sum(ln.bold_fraction * len(ln.text) for ln in group) / max(
            sum(len(ln.text) for ln in group), 1
        )
        larger = size >= cfg.heading_size_ratio * body_size
        emphasised = bold >= 0.6 and size >= body_size
        is_heading = (
            words <= cfg.heading_max_words
            and (larger or emphasised)
            and not _SENTENCE_END.search(text[:-1] or text)
        )
        if is_heading:
            # Level is assigned later, once every heading size on the document
            # is known. A per-block decision cannot get the hierarchy right.
            return Block(text, BlockType.HEADING, provenance, None, 0, size)

        return Block(text, BlockType.PARAGRAPH, provenance, None, 0, size)

    def _assign_heading_levels(
        self, blocks: Sequence[Block], body_size: float
    ) -> list[Block]:
        """Rank the distinct heading sizes descending and map them to depths.

        Sizes are bucketed to the half point first, because extractors report
        11.999999 and 12.0 for the same font. Same-size-as-body headings (bold
        run-in headings) always land at the deepest level.
        """
        sizes = sorted(
            {round(b.font_size * 2) / 2 for b in blocks if b.type is BlockType.HEADING},
            reverse=True,
        )
        if not sizes:
            return list(blocks)
        level_of = {size: index + 1 for index, size in enumerate(sizes)}
        deepest = len(sizes)
        out: list[Block] = []
        for b in blocks:
            if b.type is not BlockType.HEADING:
                out.append(b)
                continue
            bucket = round(b.font_size * 2) / 2
            level = level_of.get(bucket, deepest)
            if bucket <= body_size:
                level = deepest
            out.append(
                Block(
                    b.text,
                    b.type,
                    b.provenance,
                    min(level, 6),
                    b.reading_order,
                    b.font_size,
                    b.column,
                )
            )
        return out

    # --- serialisation ----------------------------------------------------

    def _to_markdown(self, blocks: Sequence[Block]) -> str:
        parts: list[str] = []
        for b in blocks:
            if b.type is BlockType.HEADING:
                parts.append("#" * (b.level or 1) + " " + b.text)
            elif b.type is BlockType.LIST_ITEM:
                parts.append("- " + _BULLET.sub("", b.text).strip())
            elif b.type is BlockType.CAPTION:
                parts.append("*" + b.text + "*")
            elif b.type in (BlockType.PAGE_HEADER, BlockType.PAGE_FOOTER):
                continue
            else:
                parts.append(b.text)
        return "\n\n".join(parts)


__all__ = ["DocLayoutComponent"]
