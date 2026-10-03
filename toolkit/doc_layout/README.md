# Doc Layout Component

## What It Does

Takes positioned text spans from any PDF extractor or OCR engine and returns
semantic blocks in correct reading order — headings with inferred levels, paragraphs,
list items, captions, page headers and footers — each carrying page and bounding-box
provenance, plus a markdown rendering.

## Why It Is Useful

This is the half of a document parser that needs no machine learning, and it is the
half people skip. Layout models tell you *where* regions are. They do not tell you:

- what order to read a two-column page in;
- which of the two overlapping copies of a line to keep when OCR ran over an
  existing text layer;
- which consecutive lines are one paragraph;
- whether a large bold line is an `h1` or an `h3`;
- that "Page 3 of 12" is furniture, not content.

Those are deterministic geometry and font statistics. Owning them means you can swap
PDF backends freely, run on a laptop with no GPU, and — because every block keeps its
page and bbox — cite and highlight the exact source of any extracted answer.

## Original Source

Reimplemented from published algorithms and the behaviour of these projects. No code
copied, which matters here because two of the three are copyleft.

| Piece | Prior art | Licence of the original |
|---|---|---|
| Recursive XY-cut column decomposition | Nagy & Seth 1984 | — (paper) |
| Region containment / overlap resolution | [Docling](https://github.com/docling-project/docling) `layout_postprocessor.py` | MIT |
| Font-statistics heading inference, block merging | [Marker](https://github.com/datalab-to/marker) `processors/` | GPL-family — **not copied** |
| Region routing ideas | [MinerU](https://github.com/opendatalab/MinerU) | AGPL — **not copied** |

## Architecture

```
INPUT     list[TextSpan(text, bbox, page, font_size, bold, ...)]
   |
DEDUPE    drop near-identical overlapping spans (OCR over a text layer)
   |
STATS     body font size = character-weighted mode, rounded to 0.5pt
   |
XY-CUT    per page: find a vertical gutter -> recurse -> ordered columns
   |       (guarded by min_column_width_ratio so table gutters are not columns)
   |
LINES     per column: group spans by vertical overlap, order by x
   |
PARAGRAPH break on gap > 1.6x median leading, font change, bullet, short+terminated
   |
CLASSIFY  boilerplate (recurring across pages) / list / caption / heading / paragraph
   |
LEVELS    rank distinct heading sizes descending -> h1..h6
   |
OUTPUT    ordered Block list + markdown + body_font_size + columns_per_page
```

```
stdlib only.  no torch, no weights, no GPU, no network  ->  Component  ->  blocks
```

## Installation

Copy the `doc_layout/` directory into your project. Python 3.10+.

## Dependencies

Standard library only. You supply the span extractor; pdfplumber, PyMuPDF,
pdfminer.six, PaddleOCR, Tesseract TSV and Textract output all convert in a few lines.

## Input Schema

`LayoutRequest`:

| Field | Type | Meaning |
|---|---|---|
| `spans` | `Sequence[TextSpan]` | `text`, `bbox`, `page`, `font_size`, `bold`, `italic`, `font_name` |
| `page_sizes` | `dict[int, tuple[float, float]]` | page -> (width, height); inferred if omitted |
| `config` | `LayoutConfig` | see the dataclass; every threshold is documented inline |

**Coordinates must be y-down** (y0 = top). pdfplumber's `top`/`bottom` and PyMuPDF
rects are already y-down; pdfminer's default `y0`/`y1` are y-up and must be flipped.
Getting this wrong inverts the reading order and is the most common integration bug.

## Output Schema

`LayoutResult`:

- `blocks`: `Block(text, type, provenance, level, reading_order, font_size, column)`
  where `type` is one of `heading`, `paragraph`, `list_item`, `caption`,
  `page_header`, `page_footer`, and `provenance` is
  `Provenance(page, bbox, span_indices)`.
- `markdown`: headings as `#`, list items as `-`, captions italicised, furniture
  excluded.
- `body_font_size`, `columns_per_page`, `spans_deduplicated`.

`span_indices` index back into your original input list — that is the citation hook.

## Usage

```python
from doc_layout import BBox, DocLayoutComponent, LayoutRequest, TextSpan

# Adapter for pdfplumber (its coordinates are already y-down).
def spans_from_pdfplumber(pdf):
    out = []
    for page_number, page in enumerate(pdf.pages, start=1):
        for word in page.extract_words(extra_attrs=["size", "fontname"]):
            out.append(TextSpan(
                text=word["text"],
                bbox=BBox(word["x0"], word["top"], word["x1"], word["bottom"]),
                page=page_number,
                font_size=float(word.get("size", 10.0)),
                bold="Bold" in word.get("fontname", ""),
                font_name=word.get("fontname", ""),
            ))
    return out

result = DocLayoutComponent().execute(LayoutRequest(spans, page_sizes=sizes))
print(result.markdown)

# Grounded citation: every block knows exactly where it came from.
for block in result.blocks:
    if block.type.value == "heading":
        p = block.provenance
        print(block.level, block.text, "-> page", p.page, p.bbox)
```

## Limitations

- **No text orientation.** Every span is assumed horizontal. Reading order is
  recovered by sorting on position, which is only meaningful within one
  orientation, so rotated text cannot be placed in the same flow. Every arXiv
  PDF carries a rotated identifier down its left edge, and including it
  interleaved its characters into body words (`an tc abelian surface`) and
  displaced a whole line of one abstract. `PdfPlumberSource` therefore excludes
  non-upright glyphs and reports how many in
  `Document.metadata["rotated_glyphs_excluded"]`. A landscape page, whose entire
  body is rotated, will lose its text - check that count.
- **Tables are not reconstructed.** Table text comes through as paragraphs. Table
  *structure* recovery genuinely needs a model (TableFormer, Surya) — use Docling or
  Marker for that and feed non-table regions here.
- No OCR and no layout detection. This component starts after text has coordinates.
- Column detection handles the common one- and two-column cases and nests
  recursively. Exotic magazine layouts, sidebars and text wrapped around figures
  will need per-corpus tuning of `min_column_width_ratio`.
- Heading inference relies on font size and weight being present and meaningful. A
  document that signals structure only through numbering ("3.1.2") or indentation
  will land everything at one level.
- Reading order across pages is page-number order. Multi-page tables and articles
  continued elsewhere are not stitched.
- `header_footer_min_pages` defaults to 3, so a two-page document gets no
  boilerplate detection at all. That is deliberate: two samples cannot distinguish
  furniture from content.
- Formulas, code blocks and footnote linking are not classified.

## Integration Guide

1. Write the span adapter for your extractor first, and verify one page visually —
   print `result.markdown` and compare against the PDF before trusting anything.
2. Check `columns_per_page`. If a single-column document reports 2, raise
   `min_column_width_ratio`. If a two-column one reports 1, lower it.
3. Check `spans_deduplicated`. A large number on a born-digital PDF means your
   extractor is emitting both word- and line-level spans; that is fine, it is being
   handled, but it tells you something about the input.
4. Verify `body_font_size` looks like body text, not a title. If it is wrong, every
   heading decision downstream is wrong.
5. For RAG, chunk on `blocks` rather than on characters, and carry
   `provenance.page` and `provenance.bbox` into your vector store payload. That is
   what lets you show the user the source.

## Extraction Notes

- **Preserved:** recursive XY-cut with sliver guarding; containment/IoU overlap
  resolution; character-weighted font-mode statistics for body size; descending
  size-rank heading levels; digit-normalised recurring-line detection for
  headers and footers.
- **Removed:** all model inference, weight loading, and backend registries; the
  multi-format converter and pipeline-option plumbing; CLI and telemetry.
- **Rewritten:** everything, from the algorithms rather than the source, because
  Marker is GPL-family and MinerU is AGPL and neither can be vendored into a
  permissive toolkit. `Provenance` is an explicit first-class type here rather than
  an optional attribute.
- **Isolated:** `TextSpan` and `BBox` are the only input primitives, so no PDF
  library is imported anywhere in the component. Column decomposition deliberately
  runs *before* line assembly — the reverse order splices columns together at
  matching y coordinates, which is a silent and hard-to-spot corruption.
