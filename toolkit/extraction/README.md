# Extraction Component

**Layer 2 · depends on `core` · `copy_tier: needs_core`**

## What It Does

Document + schema → validated object, with **every value traceable back to the
words that state it** — including each cell of a line-item table — and a repair loop
that converges instead of re-guessing. Money comes back as `Decimal`, so thirty
amounts add up to the stated subtotal exactly.

Plus `DocumentSplitterComponent` for the other half of the problem: a 40-page scan
that is really six invoices.

## Why It Is Useful

"PDF → validated object" is the most common real request there is. Every framework has
extraction. Almost none have a repair loop that converges, and that is where most of
the accuracy lives.

The difference is what happens on a validation failure:

- The common approach re-runs the whole prompt and hopes. The model has no idea what
  was wrong, so it often reproduces the same mistake.
- This one sends back **only the fields that failed, with the specific error and the
  value it produced**:

```
Your JSON had these problems. Fix only these fields and return the complete JSON object again:
- issued: expected a date formatted YYYY-MM-DD (got '14/03/2024')
    issued means: Date the invoice was issued
- status: must be one of: paid, unpaid, overdue (got 'not paid')
    status means: Payment status
Leave every other field exactly as it was.
```

That turns a guess into a correction, which is why two repair rounds are usually
enough. Re-sending the whole schema invites the model to rewrite fields that were
already right — that is how a repair round makes things worse.

## Architecture

```
INPUT     Document | Chunk[] | str   +   ExtractionSchema
   |
PROMPT    field names, types, constraints, descriptions, page-labelled context
   |
PARSE     fence-stripping + brace matching that respects strings and escapes
   |
COERCE    "$1,240.50" -> 1240.50 ;  "yes" -> True ;  "14 March 2024" -> "2024-03-14"
   |       (local fixes must not cost a repair round)
   |
VALIDATE  types, enums, patterns, ranges, required, nested paths
   |       -> ValidationIssue("line_items[1].qty", "must be at least 1", 0)
   |
REPAIR    re-prompt with ONLY the failures     <-- loop, bounded by max_repairs
   |
GROUND    match each scalar against runs of consecutive whole words
   |       -> Evidence(page, merged bbox, words, source, min_confidence)
   |       -> MatchClass: exact | normalized | fuzzy_ocr | not_found | not_checked
   |
OUTPUT    ExtractionResult(data, fields, leaves, issues, attempts, usage,
          truncated, warnings)
```

## Grounding: what it actually checks

A value is matched against runs of **consecutive whole words** on one page, never as a
substring. That one rule is the whole difference between a useful signal and a
misleading one. Measured on a synthetic two-page invoice, the substring version this
replaces reported a fabricated `quantity: 7` as found, because `7` sits inside the
invoice number `INV-2026-0417` and inside the unit price `7.25`.

- **Numbers compare by parsed value, per token.** `7` equals neither `7.25` nor
  anything inside a reference code. A model returning `6511.05` still matches a page
  reading `MYR 6,511.05` — the match is `normalized`, and saying which it is matters.
- **Strings compare normalised token sequences**: case, and the punctuation around a
  token (`ACME,` is `acme`). Interior characters stay, so `INV-2026-0417` cannot match
  a different reference.
- **Dates are grounded through `date_order`.** With `date_order="DMY"`, the ISO value
  `2026-10-03` is looked for as the page own `03/10/2026`. Without a hint, only
  spellings with one possible reading are searched: claiming `03/10/2026` as evidence
  for 3 October when the document may have meant 10 March is worse than finding
  nothing.
- **Rows are local.** Inside a table, the most distinctive string cell (a description,
  in practice) is grounded first and becomes the row anchor; the other cells are only
  accepted on that line, with at least half the shorter box height overlapping. A
  `quantity` of 1 appears in five rows of the fixture, and without an anchor every one
  of them grounds on the first — a box pointing at the wrong row, which is worse than
  no box because it looks right.
- **One token is evidence for one value.** A claimed word is not offered again, so a
  row whose `unit_price` and `amount` are equal grounds each on its own column.
- **`fuzzy_ocr` is deliberately not grounded.** A low-confidence OCR word one edit
  from the value is the likeliest explanation of a near miss, and is reported as such,
  but it is not evidence that the document says what the model claims.

When a `Document` has no `words` (a plain-text source), the same rules are applied to
block text with `(?<!\w)…(?!\w)` boundaries and numeric tokens parsed. Evidence then
carries the *block* box, and a consumer can tell: every token in the block shares it.

## Three decisions worth knowing

**Local coercion before repair.** Models return `"$1,240.50"` and `"yes"`. Fixing that
locally is free; spending a network round-trip on formatting is not.

**Ambiguous dates are refused, not guessed.** `01/02/2024` is 1 February or 2 January
depending on where the document came from. Guessing silently corrupts data, so it
becomes a repair round and the model is asked. Unambiguous rewrites (`2024/03/14`,
`14 March 2024`) are done locally.

**Grounding exempts normalised types.** A `DATE` field is *required* to come back as
`YYYY-MM-DD`, so a document reading "Issued 14 March 2024" can never contain the value
verbatim. Requiring grounding there fires on correct extractions — and a signal that
flags correct work is worse than no signal, because people learn to ignore it. Strings
and numbers stay in scope; that is where fabrication actually happens.

## Installation

Copy **two** directories, because this unit imports the shared contracts:

```bash
cp -r toolkit/extraction  your_project/
cp -r toolkit/core      your_project/
```

Both must sit under the same parent package so the relative import resolves.
Python 3.10+. Standard library only.

Earlier versions of this README said "copy the `extraction/` directory" full stop,
which does not work — `toolkit/tests/test_packaging.py` now copies each unit out
with its declared dependencies and imports it in a subprocess, so the instruction
is verified rather than asserted.


## Dependencies

Standard library. Pydantic is optional and only for `ExtractionSchema.from_pydantic`.

## Input Schema

`FieldSpec(name, type, description, required, enum, pattern, minimum, maximum,
item_type, fields, examples)` where `type` is one of `string`, `integer`, `number`,
`boolean`, `date`, `array`, `object`.

**`description` is the highest-leverage text in the whole pipeline.** Most extraction
failures are underspecified fields, not weak models.

From Pydantic when you already have the model:

```python
class Invoice(BaseModel):
    invoice_number: str = Field(description="The invoice reference")
    total: float = Field(description="Total due", ge=0)

schema = ExtractionSchema.from_pydantic(Invoice)
```

`type` is one of `string`, `integer`, `number`, `decimal`, `boolean`, `date`,
`array`, `object`. Use `decimal` for anything that has to add up: a `float` cannot
hold 1240.50, and `Decimal` is built from the normalised string, never via `float`.
The type is deliberately domain-neutral — exact decimal arithmetic, not a currency.

`ExtractionConfig`: `max_repairs` (2), `temperature`, `max_tokens`,
`require_grounding`, `strict`, `context_char_limit`, `coerce`, `date_order`
(`"DMY" | "MDY" | "YMD" | None`), `ocr_confidence_threshold` (0.75).

## Output Schema

`ExtractionResult`: `data` (the object), `fields` (one `FieldResult` per top-level
field), `leaves` (every scalar in the result, cells inside arrays included), `issues`,
`attempts`, `usage`, `raw_responses`, `truncated`, `warnings`. `.valid`,
`.ungrounded`, `.ungrounded_paths`, `.field_map()` and `.leaf_map()` are conveniences.

`FieldResult`: `name`, `value`, `grounded`, `provenance`, `matched_text`, `path`
(`line_items[3].amount`), `match` (a `MatchClass`) and `evidence`. `Evidence` holds
`page`, the merged `bbox` of the matched words, `words`, `source`
(`text_layer` / `ocr`) and `min_confidence`. `provenance` stays the first evidence
page and box, so existing callers are unaffected.

A 30-row invoice has four `fields` and 123 `leaves`; `leaves` is the list to review.

## Usage

```python
from toolkit.adapters import LiteLLMClient, PdfPlumberSource
from toolkit.extraction import (
    ExtractionComponent, ExtractionRequest, ExtractionSchema, FieldSpec, FieldType,
)

schema = ExtractionSchema("Invoice", [
    FieldSpec("invoice_number", FieldType.STRING, "The invoice reference code"),
    FieldSpec("issued", FieldType.DATE, "Date the invoice was issued"),
    FieldSpec("total", FieldType.NUMBER, "Total amount due", minimum=0),
    FieldSpec("status", FieldType.STRING, "Payment status",
              enum=["paid", "unpaid", "overdue"]),
])

document = PdfPlumberSource().load("invoice.pdf")
result = ExtractionComponent(LiteLLMClient("gpt-4o-mini")).execute(
    ExtractionRequest(schema, document)
)

print(result.data, result.attempts, result.valid)
for field in result.fields:
    if field.provenance:
        print(field.name, "page", field.provenance.page, field.provenance.bbox)
for field in result.ungrounded:
    print("review:", field.name, field.value)
```

Reviewing a table, and reading the evidence a reviewer actually needs:

```python
request = ExtractionRequest(schema, document, ExtractionConfig(date_order="DMY"))
result = ExtractionComponent(llm).execute(request)

if result.truncated:
    print(result.warnings)          # the model did not see the whole document

for path in result.ungrounded_paths:
    print("fabricated or paraphrased:", path, result.leaf_map()[path].value)

cell = result.leaf_map()["line_items[3].amount"]
print(cell.match.value, cell.evidence[0].page, cell.evidence[0].bbox)
```

Splitting a multi-document file:

```python
from toolkit.extraction import DocumentSplitterComponent

split = DocumentSplitterComponent(llm).execute(document, label_segments=True)
for segment in split.segments:
    print(segment.start_page, segment.end_page, segment.label, segment.reason)
```

`python examples/extract.py` runs the repair loop offline.

## Limitations

- **One LLM call per attempt over the whole document.** No map-reduce over long
  documents; `context_char_limit` truncates. Truncation is now *reported* —
  `result.truncated` and a warning naming characters sent against characters
  available — but it is not repaired, because a repair round cannot recover text the
  model was never shown. For a 200-page contract, split or retrieve first.
- Grounding answers "is this on the page", not "is this right". A value the model
  correctly summed or paraphrased is `not_found`, so `require_grounding` suits
  verbatim extraction and not derived fields. Whether a field is legitimately derived
  is the application call, which is why there is no `derived` match class here.
- **Cell-precise evidence needs word-level geometry.** `PdfPlumberSource` and
  `TesseractSource` populate `Document.words`; a plain text dump does not, and there
  the evidence box is the block box.
- Row locality anchors on the most distinctive string cell. A row that states no
  string of three characters or more is grounded without an anchor, so a repeated
  number in it may ground on another row.
- Outside an array, a value appearing twice grounds on the first occurrence.
- No streaming, no tool-calling, no constrained decoding. With a local model, pair the
  schema with LLGuidance and the repair loop mostly stops firing.
- `enum` matching is exact and case-sensitive after coercion.
- **The splitter is heuristic.** It detects page-number restarts, repeated first-page
  headings, and top-level headings after body text. Documents with no footer numbering
  and no headings give one segment. It reads page numbers only from header/footer
  blocks, so it needs a `DocumentSource` that classifies furniture — `PdfPlumberSource`
  does via `doc_layout`; a plain text dump does not.
- Segments shorter than `min_pages` are absorbed into the previous one, so a genuine
  one-page document between two long ones can be swallowed.

## Integration Guide

1. **Write real `description` text for every field.** It costs a minute and it is
   worth more than any other change you can make here.
2. Start with `max_repairs=2`. If you routinely need more, the schema is ambiguous —
   read `result.raw_responses` to see what the model actually thought.
3. Show `result.ungrounded_paths` to a human before trusting a batch. It is a short
   list and every fabricated value is in it. `match` tells a reviewer which kind of
   problem each one is, and `fuzzy_ocr` in particular usually means "re-scan", not
   "re-prompt".
4. Turn on `require_grounding` only for verbatim fields. It exempts dates and booleans
   automatically, but a computed total will still trip it.
5. Split before extracting when a file holds several documents. Extracting a six-invoice
   scan as one record produces one mangled result, and no prompt fixes that.
6. Set `date_order` from what you know about the sender, not from the page. It is the
   one fact the document cannot tell you, and without it an ambiguous date costs a
   repair round every time.
7. Check `result.truncated` in any batch job. A silently truncated document looks
   exactly as confident as a complete one.
8. Carry `evidence[0].bbox` into your UI. Reviewers check extractions far faster when
   they can see the exact cell, and `Document.page_sizes` scales the box onto a
   render.

## Extraction Notes

- **Preserved:** the general schema-prompt-validate shape common to ExtractThinker,
  Instructor and LangChain's output parsers.
- **Added:** the targeted repair prompt (only failed fields, with their errors and
  descriptions), word-level grounding with token boundaries and match classes,
  per-cell evidence inside arrays, exact-decimal values, row locality, reported
  truncation, local coercion that refuses ambiguous dates unless told the convention,
  precise nested issue paths — all of them, not just the first — and the heuristic
  splitter.
- **Isolated:** no Pydantic requirement, no vendor SDK. The component needs only an
  object with `.complete(messages, temperature, max_tokens)`.
