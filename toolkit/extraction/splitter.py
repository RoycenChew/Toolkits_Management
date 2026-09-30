"""Find where one document ends and the next begins inside a single file.

A 40-page scan is routinely six invoices, or a contract with four appendices that
each restart their own numbering. Extracting against the whole file produces one
mangled record instead of six good ones, and no amount of prompt engineering
fixes it — the segmentation has to happen first.

This is under-served precisely because it looks trivial and is not. The signals
that actually work are structural rather than semantic, and they are cheap:

* **A page-number reset.** 'Page 1 of 4' appearing on physical page 12 is the
  strongest boundary signal there is, and it needs no model.
* **A repeated first-page heading.** The same title recurring at the top of a
  page means a new copy of the same document type started there.
* **A top-level heading after a run of body text**, for structured reports.

An optional LLM classifier labels the segments afterwards. Labelling is genuinely
semantic; boundary detection mostly is not, so it stays free by default.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from ..core.models import Block, BlockType, Document, Message
from .models import Segment, SplitResult

_PAGE_OF = re.compile(r"\bpage\s+(\d{1,3})\s+(?:of|/)\s+(\d{1,3})\b", re.IGNORECASE)
_BARE_PAGE = re.compile(r"^\s*(?:page\s+)?(\d{1,3})\s*$", re.IGNORECASE)


class DocumentSplitterComponent:
    """Split a multi-document file into page ranges.

    Pass an LLM to have segments labelled; without one they are unlabelled but
    still correctly bounded.
    """

    def __init__(self, llm: Any | None = None) -> None:
        self._llm = llm

    def execute(
        self,
        document: Document,
        min_pages: int = 1,
        label_segments: bool = False,
    ) -> SplitResult:
        pages = self._pages(document)
        if not pages:
            return SplitResult(segments=[], page_count=0)
        ordered = sorted(pages)
        if len(ordered) == 1:
            return SplitResult(
                segments=[Segment(ordered[0], ordered[0], reason="single page")],
                page_count=1,
            )

        boundaries = self._boundaries(pages, ordered)
        segments = self._assemble(ordered, boundaries, min_pages)

        if label_segments and self._llm is not None:
            segments = self._label(segments, pages)
        return SplitResult(segments=segments, page_count=len(ordered))

    # --- page assembly ----------------------------------------------------

    def _pages(self, document: Document) -> dict[int, list[Block]]:
        pages: dict[int, list[Block]] = {}
        for block in document.blocks:
            if block.provenance is None or not block.text.strip():
                continue
            pages.setdefault(block.provenance.page, []).append(block)
        return pages

    # --- boundary detection -----------------------------------------------

    def _boundaries(
        self, pages: Mapping[int, Sequence[Block]], ordered: Sequence[int]
    ) -> dict[int, str]:
        """Pages that start a new document, with the reason each was chosen."""
        reasons: dict[int, str] = {}

        counters = {page: self._page_counter(pages[page]) for page in ordered}
        first_headings = {page: self._first_heading(pages[page]) for page in ordered}

        heading_counts: dict[str, int] = {}
        for heading in first_headings.values():
            if heading:
                heading_counts[heading] = heading_counts.get(heading, 0) + 1

        for index, page in enumerate(ordered[1:], start=1):
            previous = ordered[index - 1]

            current_counter = counters[page]
            previous_counter = counters[previous]
            if current_counter == 1 and previous_counter is not None and previous_counter >= 1:
                # 'Page 1 of N' after any other numbered page: a restart.
                reasons[page] = "page numbering restarted"
                continue

            heading = first_headings[page]
            if heading and heading_counts.get(heading, 0) > 1 and heading == first_headings[ordered[0]]:
                reasons[page] = "first-page heading repeated"
                continue

            if heading and self._is_top_level(pages[page]) and not self._is_top_level(
                pages[previous]
            ):
                reasons[page] = "top-level heading after body text"

        return reasons

    def _page_counter(self, blocks: Sequence[Block]) -> int | None:
        """Read 'page N of M' from this page's furniture.

        Restricted to header/footer blocks: 'see page 1 of the appendix' in body
        prose is a reference, not a page number, and treating it as one splits
        documents in the middle of a sentence.
        """
        for block in blocks:
            if block.type not in (BlockType.PAGE_HEADER, BlockType.PAGE_FOOTER):
                continue
            match = _PAGE_OF.search(block.text)
            if match:
                return int(match.group(1))
            bare = _BARE_PAGE.match(block.text.strip())
            if bare:
                return int(bare.group(1))
        return None

    def _first_heading(self, blocks: Sequence[Block]) -> str:
        for block in blocks:
            if block.type is BlockType.HEADING:
                return " ".join(block.text.lower().split())
            if block.type not in (BlockType.PAGE_HEADER, BlockType.PAGE_FOOTER):
                return ""
        return ""

    def _is_top_level(self, blocks: Sequence[Block]) -> bool:
        for block in blocks:
            if block.type is BlockType.HEADING:
                return (block.level or 1) <= 1
            if block.type not in (BlockType.PAGE_HEADER, BlockType.PAGE_FOOTER):
                return False
        return False

    def _assemble(
        self, ordered: Sequence[int], boundaries: Mapping[int, str], min_pages: int
    ) -> list[Segment]:
        """Turn boundary pages into contiguous ranges.

        A segment shorter than `min_pages` is folded into its predecessor rather
        than emitted, because a one-page fragment is far more often a false
        boundary than a real single-page document.
        """
        segments: list[Segment] = []
        start = ordered[0]
        reason = "start of file"
        for page in ordered[1:]:
            if page in boundaries:
                length = page - start
                if length >= min_pages or not segments:
                    segments.append(
                        Segment(start, page - 1, reason=reason, confidence=0.8)
                    )
                    start = page
                    reason = boundaries[page]
                # Too short: absorb it by leaving `start` where it is.
        segments.append(
            Segment(start, ordered[-1], reason=reason, confidence=0.8)
        )
        return segments

    # --- optional labelling -----------------------------------------------

    def _label(
        self, segments: Sequence[Segment], pages: Mapping[int, Sequence[Block]]
    ) -> list[Segment]:
        llm = self._llm
        if llm is None:
            return list(segments)
        labelled: list[Segment] = []
        for segment in segments:
            preview = self._preview(segment, pages)
            if not preview.strip():
                labelled.append(segment)
                continue
            completion = llm.complete(
                [
                    Message(
                        "system",
                        "You classify documents. Reply with a short lowercase label"
                        " of one to three words and nothing else.",
                    ),
                    Message("user", preview),
                ],
                0.0,
                24,
            )
            label = " ".join(completion.text.strip().lower().split())[:40]
            labelled.append(
                Segment(
                    start_page=segment.start_page,
                    end_page=segment.end_page,
                    label=label,
                    confidence=segment.confidence,
                    reason=segment.reason,
                )
            )
        return labelled

    def _preview(
        self, segment: Segment, pages: Mapping[int, Sequence[Block]], limit: int = 600
    ) -> str:
        """The first page's text is enough to classify a document, and keeping the
        preview small is what makes labelling affordable across many segments."""
        blocks = pages.get(segment.start_page, [])
        text = " ".join(
            b.text for b in blocks if b.type not in (BlockType.PAGE_HEADER, BlockType.PAGE_FOOTER)
        )
        return text[:limit]


__all__ = ["DocumentSplitterComponent"]
