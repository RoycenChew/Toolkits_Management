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

WORD_GAP_RATIO = 0.15
"""Smallest inter-span gap, as a fraction of font size, that means "a space".

Two stages have to agree on this number, and they used not to. A PDF extractor
decides where one word ends and the next begins; this component then rejoins
the spans it is handed and must put the spaces back. If the rejoining threshold
is *looser* than the splitting one it silently undoes the split and glues the
line back together:

    Supersingularabeliansurfacesareessentialinisogeny-based

This was 0.25 while `adapters.PdfPlumberSource` took pdfplumber's default
absolute tolerance, and between them every one of 49 real arXiv PDFs lost word
spaces: a median 22.9% of characters, worst case 84.2%. LaTeX's Computer Modern
sets inter-word space below 0.25 em at body sizes, so the gap was there and was
discarded. Glued text is unmatchable by any lexical index, because the tokens
do not exist.

0.15 em sits below every real space observed in that corpus and above the
intra-word kerning. Measured across 8 documents spanning 1986-2026: word counts
rise from 12,627 to ~21,980 and then plateau below 0.15, while the share of one-
and two-character tokens stays flat at 20.4% - so the extra splits are real
spaces being recovered, not words being broken up. `PdfPlumberSource` imports
this constant so the two stages cannot drift apart again.

A ratio rather than an absolute point value because the threshold has to scale
with type size: 1.35pt at 9pt body text, 3pt at a 20pt heading.
"""

_BULLET = re.compile(r"^\s*(?:[-â€¢â€£â—¦âƒâˆ™*]|\(?\d{1,2}[.)]|[a-z][.)])\s+")
_CAPTION = re.compile(
    r"^\s*(?:fig(?:ure)?|table|tbl|exhibit|chart|listing|appendix)\s*\.?\s*"
    r"(?:\d+|[ivxlc]+|[A-Z])\b",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"[.!?:;\"â€â€™)]\s*$")


def _hyphen_break(left: str, right: str) -> bool:
    """Is the trailing hyphen on `left` a justification break rather than part
    of a compound word?

    Inlined rather than imported from `core.text` so this component keeps its
    independently-copyable property — doc_layout imports nothing from the
    toolkit package. Conservative on purpose: only a lowercase continuation
    counts, so `self-
Service` keeps its hyphen and `40-
50` is untouched.
    """
    if not left.endswith("-") or len(left) < 2:
        return False
    if not left[-2].isalpha():
        return False
    return bool(right) and right[0].isalpha() and right[0].islower()


def _join_lines(lines: Sequence[str]) -> str:
    """Merge a block's lines, rejoining words split by justification.

    PDFs break `custo-
mer` and `conse-
quential` constantly. Joined with a
    space, neither half matches a query and the chunk reads as broken text; the
    stress corpus had five such words in a single short contract. Only a
    lowercase continuation is treated as a break, so genuine compounds survive.
    """
    out = ""
    for raw in lines:
        piece = raw.strip()
        if not piece:
            continue
        if not out:
            out = piece
            continue
        out = out[:-1] + piece if _hyphen_break(out, piece) else out + " " + piece
    return out.strip()


class _Row:
    """A provisional horizontal row of spans, used only for column geometry.

    Distinct from `_Line`: a row may legitimately span two columns while the
    geometry is still unknown, which is exactly the signal `_find_gutter` needs.
    Real lines are assembled per column once reading order has been decided.
    """

    __slots__ = ("items", "bbox")

    def __init__(self, index: int, span: TextSpan) -> None:
        self.items: list[tuple[int, TextSpan]] = [(index, span)]
        self.bbox = span.bbox

    def add(self, index: int, span: TextSpan) -> None:
        self.items.append((index, span))
        self.bbox = self.bbox.merge(span.bbox)

    @property
    def font_size(self) -> float:
        total = sum(len(span.text) for _, span in self.items)
        if total == 0:
            return self.items[0][1].font_size
        return sum(span.font_size * len(span.text) for _, span in self.items) / total


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
                needs_space = gap > WORD_GAP_RATIO * max(prev.font_size, 1.0)
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

    def _rows(
        self, items: Sequence[tuple[int, TextSpan]], cfg: LayoutConfig
    ) -> list[_Row]:
        """Group spans into provisional rows by vertical overlap.

        Column detection has to reason about rows, not spans. Extractors emit
        one span per *word*, so a full-width title is not a single wide object
        that obviously crosses a gutter - it is six narrow words, some left of
        the gutter and some right, which a span-level cut happily tears in half.
        Rows restore the geometry the page actually has.

        These rows are provisional and used only for geometry. Real line
        assembly happens per column afterwards, once reading order is known.
        """
        ordered = sorted(items, key=lambda pair: (pair[1].bbox.y0, pair[1].bbox.x0))
        rows: list[_Row] = []
        for index, span in ordered:
            placed = False
            for row in reversed(rows[-3:]):
                if span.bbox.vertical_overlap(row.bbox) >= cfg.line_overlap_threshold:
                    row.add(index, span)
                    placed = True
                    break
            if not placed:
                rows.append(_Row(index, span))
        rows.sort(key=lambda row: (row.bbox.y0, row.bbox.x0))
        return rows

    def _row_spans_gutter(self, row: _Row, gutter: float, min_gap: float) -> bool:
        """Does this row genuinely run across the gutter, or is it two columns?

        The distinction that makes two-column detection work. Both cases have
        ink on either side of the gutter, so extent alone cannot tell them
        apart:

        * A **full-width title** has inter-word gaps of a few points. The gutter
          falls inside one of them, or inside a word.
        * **Two column lines sharing a baseline** have a gap of tens of points
          at exactly that position.

        Comparing the gap containing the gutter against `min_gap` separates
        them. Without this test, a layout whose columns share baselines makes
        every row look full-width, and an earlier version abandoned the split
        for precisely that reason.
        """
        ordered = sorted((span for _, span in row.items), key=lambda s: s.bbox.x0)
        cursor = ordered[0].bbox.x1
        for span in ordered[1:]:
            if cursor <= gutter <= span.bbox.x0:
                return (span.bbox.x0 - cursor) < min_gap
            cursor = max(cursor, span.bbox.x1)
        # The gutter sits inside a word: unambiguously a spanning row.
        return True

    def _xy_cut(
        self,
        items: Sequence[tuple[int, TextSpan]],
        page_size: tuple[float, float] | None,
        cfg: LayoutConfig,
        depth: int = 0,
    ) -> list[list[tuple[int, TextSpan]]]:
        """Decompose a region into columns, in reading order.

        Classic XY-cut alternates vertical and horizontal projection cuts. Three
        earlier versions of this were wrong in ways the stress corpus exposed:

        1. **Vertical cuts only.** A full-width title straddles the gutter, the
           straddle check refused to split, and the two columns were spliced
           into single lines - the abstract and the introduction in one block.
        2. **Requiring a zero-ink gap.** A centred page number in the footer
           sits in the gutter and bridges it, so no clean whitespace column
           existed anywhere on the page and detection failed outright.
        3. **Treating any crossing row as full-width.** Columns that share
           baselines make every row cross, so the guard against over-splitting
           killed the very case it was meant to serve.

        What works: find the gutter by row *density*, then classify each crossing
        row by the size of the gap at the gutter. Genuinely full-width rows force
        a horizontal band cut; rows that merely share a baseline are split.
        """
        if len(items) < 4 or depth > 6:
            return [list(items)]
        page_width = page_size[0] if page_size else max(s.bbox.x1 for _, s in items)
        if page_width <= 0:
            return [list(items)]

        rows = self._rows(items, cfg)
        if len(rows) < 2:
            return [list(items)]

        gutter = self._find_gutter(rows, page_width, cfg)
        if gutter is None:
            return [list(items)]

        min_gap = self._min_gap(rows, page_width, cfg)
        spanning: list[_Row] = []
        left: list[tuple[int, TextSpan]] = []
        right: list[tuple[int, TextSpan]] = []

        for row in rows:
            if row.bbox.x1 <= gutter:
                left.extend(row.items)
            elif row.bbox.x0 >= gutter:
                right.extend(row.items)
            elif self._row_spans_gutter(row, gutter, min_gap):
                spanning.append(row)
            else:
                for index, span in row.items:
                    target = left if (span.bbox.x0 + span.bbox.x1) / 2 <= gutter else right
                    target.append((index, span))

        if not left or not right:
            return [list(items)]
        if len(spanning) > len(rows) // 2:
            # Predominantly full-width: a single-column region whose rows happen
            # to reach past the candidate.
            return [list(items)]

        if spanning:
            return self._band_cut(rows, spanning, page_size, cfg, depth)

        # Ink density is what separates text columns from a table. A table is
        # genuinely multi-column by geometry - whitespace gutters between cells
        # are real - so a density cut happily tears its rows apart, which is the
        # one thing a table must not have done to it. Text lines fill most of
        # their width; table cells occupy a small fraction of theirs. Extent
        # cannot tell them apart, because a row of two cells has the same extent
        # as a line of prose; ink can.
        if min(self._ink_ratio(left, cfg), self._ink_ratio(right, cfg)) < 0.6:
            return [list(items)]

        out: list[list[tuple[int, TextSpan]]] = []
        for side in (left, right):
            width = max(s.bbox.x1 for _, s in side) - min(s.bbox.x0 for _, s in side)
            out.extend(self._xy_cut(side, (max(width, 1.0), 0.0), cfg, depth + 1))
        return out

    def _ink_ratio(
        self, items: Sequence[tuple[int, TextSpan]], cfg: LayoutConfig
    ) -> float:
        """Mean fraction of each row occupied by actual glyphs.

        Prose runs about 0.8 or above: words with single spaces between them.
        Table rows run nearer 0.3, because most of the row is the whitespace
        between cells. Measured on the stress corpus, this is the cleanest
        available signal for refusing to column-split a table.
        """
        rows = self._rows(items, cfg)
        if not rows:
            return 0.0
        ratios = []
        for row in rows:
            extent = row.bbox.width
            if extent <= 0:
                continue
            ink = sum(span.bbox.width for _, span in row.items)
            ratios.append(min(1.0, ink / extent))
        return sum(ratios) / len(ratios) if ratios else 0.0

    def _band_cut(
        self,
        rows: Sequence[_Row],
        spanning: Sequence[_Row],
        page_size: tuple[float, float] | None,
        cfg: LayoutConfig,
        depth: int,
    ) -> list[list[tuple[int, TextSpan]]]:
        """Horizontal cut around full-width rows, then columns within each band.

        Emits regions top-to-bottom, which is the order a human reads: material
        above a full-width heading, then the heading, then material below it.
        Consecutive full-width rows are grouped so a wrapped title stays one
        region rather than becoming one region per line.
        """
        full = sorted(spanning, key=lambda row: row.bbox.y0)
        rest = [row for row in rows if row not in spanning]

        regions: list[list[tuple[int, TextSpan]]] = []
        cursor = float("-inf")
        index = 0
        while index < len(full):
            run = [full[index]]
            while index + 1 < len(full) and (
                full[index + 1].bbox.y0 - run[-1].bbox.y1
                < 1.5 * max(run[-1].bbox.height, 1.0)
            ):
                index += 1
                run.append(full[index])
            boundary = run[0].bbox.y0

            band = [r for r in rest if cursor <= r.bbox.center_y < boundary]
            if band:
                band_items = [pair for row in band for pair in row.items]
                regions.extend(self._xy_cut(band_items, page_size, cfg, depth + 1))
            regions.append([pair for row in run for pair in row.items])
            cursor = run[-1].bbox.y1
            index += 1

        tail = [r for r in rest if r.bbox.center_y >= cursor]
        if tail:
            tail_items = [pair for row in tail for pair in row.items]
            regions.extend(self._xy_cut(tail_items, page_size, cfg, depth + 1))
        return [region for region in regions if region]

    def _min_gap(
        self, rows: Sequence[_Row], page_width: float, cfg: LayoutConfig
    ) -> float:
        median_char = max(0.5 * (sum(r.font_size for r in rows) / len(rows)), 1.0)
        return max(
            cfg.column_gap_multiplier * median_char,
            cfg.min_gutter_ratio * page_width,
        )

    def _find_gutter(
        self, rows: Sequence[_Row], page_width: float, cfg: LayoutConfig
    ) -> float | None:
        """Widest vertical band that few rows reach into.

        Density rather than emptiness. Counting the *rows* that cover each x
        position means a single footer page number sitting in the gutter raises
        the count there to one, while a body column reaches a count of a dozen,
        so the gutter is still clearly the minimum. Requiring strict emptiness
        instead let one centred page number defeat column detection for a whole
        page.

        Coverage is counted from each row's spans, not its full extent, so a
        full-width title contributes to every bin it actually has ink in and the
        inter-word gaps inside it do not read as candidate gutters.
        """
        if not rows:
            return None
        bin_size = 4.0
        bins = max(1, int(page_width / bin_size))
        coverage = [0] * bins

        for row in rows:
            touched: set[int] = set()
            for _, span in row.items:
                start = max(0, int(span.bbox.x0 / bin_size))
                end = min(bins - 1, int(span.bbox.x1 / bin_size))
                touched.update(range(start, end + 1))
            for index in touched:
                coverage[index] += 1

        # A gutter may be reached by at most this many rows. The floor of two is
        # load-bearing: on a two-column page the gutter is routinely crossed by
        # both a full-width title and a centred page number, and a floor of one
        # rejected the only real gutter on the page.
        tolerance = max(2, int(len(rows) * 0.15))
        min_gap = self._min_gap(rows, page_width, cfg)
        min_width = cfg.min_column_width_ratio * page_width
        left_edge = min(r.bbox.x0 for r in rows)
        right_edge = max(r.bbox.x1 for r in rows)

        best: tuple[float, float] | None = None
        index = 0
        while index < bins:
            if coverage[index] > tolerance:
                index += 1
                continue
            start = index
            while index < bins and coverage[index] <= tolerance:
                index += 1
            low, high = start * bin_size, index * bin_size
            # Whitespace outside the text block is the page edge, not a gutter.
            if low <= left_edge or high >= right_edge:
                continue
            if high - low < min_gap:
                continue
            centre = (low + high) / 2.0
            if centre - left_edge < min_width or right_edge - centre < min_width:
                continue
            if best is None or (high - low) > best[1]:
                best = (centre, high - low)

        return None if best is None else best[0]


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
        pages_total = len(page_lines)
        if pages_total < 2:
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

        # On a document shorter than the normal threshold, require the line to
        # appear on *every* page instead. A running header on both pages of a
        # two-page paper is 100% recurrence inside the margin band, which is
        # strong evidence — and the old flat `>= 3` rule meant short documents
        # got no furniture detection at all, so journal headers were indexed as
        # body text.
        threshold = (
            cfg.header_footer_min_pages
            if pages_total >= cfg.header_footer_min_pages
            else pages_total
        )
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
        text = _join_lines([line.text for line in group])
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

        # Heading test runs BEFORE the bullet test, because a numbered section
        # heading looks exactly like an ordered list item: "2. Method" and
        # "3.1 Results" both match the bullet pattern. Testing bullets first
        # misclassified every numbered heading in the stress corpus as a list
        # item, which destroyed the heading hierarchy of any document that
        # numbers its sections — most specifications and papers.
        if _BULLET.match(group[0].text) and not is_heading:
            return Block(text, BlockType.LIST_ITEM, provenance, None, 0, size)

        if _CAPTION.match(text) and words <= cfg.caption_max_words:
            return Block(text, BlockType.CAPTION, provenance, None, 0, size)

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
