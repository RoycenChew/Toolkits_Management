# Extraction Component

## What It Does

Document + schema → validated object, with **every field traceable back to the page
and bounding box it came from**, and a repair loop that converges instead of
re-guessing.

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
GROUND    locate each value in the source -> Provenance(page, bbox)
   |
OUTPUT    ExtractionResult(data, fields, issues, attempts, usage)
```

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

Copy the `extraction/` directory. Python 3.10+.

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

`ExtractionConfig`: `max_repairs` (2), `temperature`, `max_tokens`,
`require_grounding`, `strict`, `context_char_limit`, `coerce`.

## Output Schema

`ExtractionResult`: `data` (the object), `fields` (`FieldResult` with `value`,
`grounded`, `provenance`, `matched_text`), `issues`, `attempts`, `usage`,
`raw_responses`. `.valid`, `.ungrounded` and `.field_map()` are conveniences.

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
  documents; `context_char_limit` truncates. For a 200-page contract, split or
  retrieve first, then extract per section.
- Grounding is substring matching. A value the model correctly paraphrased or summed
  reads as ungrounded, so `require_grounding` suits verbatim extraction and not
  derived fields.
- `matched_text` returns the first passage containing the value. A number appearing
  twice grounds to whichever comes first, which may not be the one that was meant.
- Nested validation reports the **first** issue per field, not all of them. Keeps the
  repair prompt short; means a deeply broken array takes more rounds.
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
3. Show `result.ungrounded` to a human before trusting a batch. It is a short list and
   every fabricated value is in it.
4. Turn on `require_grounding` only for verbatim fields. It exempts dates and booleans
   automatically, but a computed total will still trip it.
5. Split before extracting when a file holds several documents. Extracting a six-invoice
   scan as one record produces one mangled result, and no prompt fixes that.
6. Carry `field.provenance` into your UI. Reviewers check extractions far faster when
   they can see the source region.

## Extraction Notes

- **Preserved:** the general schema-prompt-validate shape common to ExtractThinker,
  Instructor and LangChain's output parsers.
- **Added:** the targeted repair prompt (only failed fields, with their errors and
  descriptions), per-field provenance and grounding, local coercion that refuses
  ambiguous dates, precise nested issue paths, and the heuristic splitter.
- **Isolated:** no Pydantic requirement, no vendor SDK. The component needs only an
  object with `.complete(messages, temperature, max_tokens)`.
