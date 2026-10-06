# Extraction Eval

**Layer 4 · depends on `core` + `extraction` · `copy_tier: needs_package`**

## What It Does

A golden set of documents with expected values per path in, a report out: field
accuracy, line-item precision/recall/F1, grounding rate, schema validity, tokens,
cost — and the **silent error rate**, which is the one the others hide.

Plus a `diff_extraction_reports` regression diff that names the documents a change
broke, not just the average it moved.

## Why It Is Useful

`evaluation` answers "did retrieval improve". Nothing answered "did extraction
improve", and the two cannot share a harness: a retrieval case is a question with
relevant passages, an extraction case is a document with expected values per path,
and the metric that matters most here has no equivalent there.

**That metric is the silent error rate: results the pipeline accepted that have a
wrong required field.** Field accuracy counts every mistake equally, which flatters
a system that fails loudly. What a reader of the output actually suffers is the
subset that passed validation, raised no warning, and was wrong — and a change can
improve field accuracy while making that subset *larger*. A rejected wrong result
cost a retry; a silent one cost whatever acting on it costs.

Which is why the acceptance predicate is injected rather than assumed. Every other
metric here is a property of the extraction; `silent_error_rate` is a property of
the extraction **and the gate in front of it**, and that gate is a product decision.

```python
# the weakest honest gate, and the default
accept=lambda r: r.valid
# one that also refuses anything not found on the page
accept=lambda r: r.valid and not r.ungrounded_paths
```

The second turns silent errors into loud ones and rejects more work. Seeing that
trade is the point.

## Architecture

```
ExtractionGolden (JSONL)        path -> expected value, one case per document
   |                            `line_items`: a list of rows, scored by matching
LOAD      source_text inline, or load_source(case) for a real document
   |
EXTRACT   your ExtractionComponent, your schema, your ExtractionConfig
   |
COMPARE   normalised exact match per path, using extraction's own leaf paths
   |       so `grounded` and `match` come from the extractor, not re-derived
   |
ROWS      multiset match on (description, amount); micro-averaged P / R / F1
   |
GATE      accept(result) -> accepted?  ->  accepted AND wrong = SILENT ERROR
   |
REPORT    ExtractionEvalReport.to_json()  ->  diff_extraction_reports(before, after)
```

## Three decisions worth knowing

**Ground truth is paths, not an expected object.** `{"line_items[3].amount": "57.00"}`
rather than a full expected JSON document. The object form reads better and is worse
to work with: a case written that way must restate every field to assert one of
them, so the set grows faster than the attention available to keep it right, and a
schema change invalidates every case rather than the cases that mention the changed
field. Paths are also exactly what `extraction` puts on every `FieldResult`, so the
harness reads `grounded` and `match` off the extractor instead of recomputing them.

**Normalised exact match, and no fuzzy matching.** `"USD 35.00"`, `35`,
`Decimal("35.00")` and `"35.0"` are one value — a harness that scores three of them
wrong measures formatting rather than extraction. But `"ACME Corp"` and
`"ACME Corporation"` are different answers, and a similarity threshold that calls
them equal hides the error it was meant to find. `None` equals only `None`: a field
the model omitted is a wrong answer, not a missing measurement, or a model could
score well by answering less.

**Rows are matched on description *and* amount, as multisets.** Either field alone
is ambiguous — two rows of a real invoice routinely share an amount, and a repeated
description with a different amount is a different row. Multisets, not sets: a
correct row returned twice is one match and one false positive, otherwise precision
is unbounded above and a model that repeats every row outscores one that does not.

## Installation

```bash
pip install -e .
```

`copy_tier: needs_package`. This unit is standard library only, but its dependency
closure is `core` **and** `extraction`, and this repository reserves `needs_core`
for a closure of `core` alone — a unit that drags two others along is a package
install, not a copied folder.

It does copy out cleanly if you insist, and `toolkit/tests/test_packaging.py`
verifies exactly that by copying these three directories into an empty project and
importing them in a subprocess with the repository off `sys.path`:

```bash
cp -r toolkit/extraction_eval  your_project/
cp -r toolkit/extraction       your_project/
cp -r toolkit/core             your_project/
```

All three must sit under the same parent package so the relative imports resolve.
Python 3.10+.

## Dependencies

Standard library. `extraction` for the component it drives, `core` for `Usage`.
Resolving a case's `document` to a real file is deliberately *not* in here — that
needs an adapter, and an adapter needs a vendor package. It is the `load_source`
hook instead.

## Input Schema

`ExtractionCase(case_id, document, source_text, expected, required_paths, tags,
notes)`.

- `expected` maps a **path** to its expected value. A path whose expected value is a
  list of mappings is a table, scored by matching.
- `required_paths` are the paths that must be right for an accepted result not to be
  a silent error. Empty means *all* of them — a case that marked nothing required
  would make the silent error rate structurally zero.
- `source_text` holds the document inline; otherwise `document` is whatever the
  runner's `load_source` hook needs to find it.
- `notes` is for the person who finds this case failing in eight months.

Golden sets are JSONL, one case per line, `#` comments and blank lines skipped:

```json
{"case_id": "inv-1", "document": "invoices/inv-1.pdf",
 "expected": {"invoice_number": "INV-88213", "total": "35.00",
              "line_items": [{"description": "Hex bolt stainless", "amount": "20.00"}]},
 "required_paths": ["invoice_number", "total"]}
```

`ExtractionEvalConfig`: `line_items_path`, `description_keys`, `amount_keys`,
`record_raw_responses` (off by default — a report holding every completion is large,
and completions are the one part that cannot be diffed usefully).

## Output Schema

`ExtractionEvalReport`: `dataset`, `metrics`, `cases`, `config`, `created_at`, with
`to_json` / `from_json` and `case_map()`. `config` is a snapshot of the schema name,
the extraction settings and the model version, because without it a diff can tell
you something changed but never what.

`metrics`:

| Key | Meaning |
|---|---|
| `field_accuracy` | normalised exact match over every compared path |
| `line_item_precision` / `_recall` / `_f1` | rows, micro-averaged so a 40-row invoice outweighs a 2-row one |
| `grounding_rate` | share of non-null extracted values found in the document |
| `schema_validity_rate` | share of cases with no validation issue |
| `acceptance_rate` | share the gate let through |
| `silent_error_rate` | accepted **and** wrong where it mattered, over all cases |
| `silent_error_rate_of_accepted` | the same numerator over accepted cases: "of what you shipped, how much was wrong" |
| `mean_attempts` | repair rounds actually spent |
| `truncated_cases` | documents the model only partly saw |
| `total_input_tokens` / `total_output_tokens` / `total_cost_usd` | from `Usage` |
| `mean_seconds` | wall clock per case |

Both silent-error denominators are reported, named, because the ambiguity is
otherwise invisible: over all cases the number is comparable between runs, and over
accepted ones it is what a reader suffers. A stricter gate improves the first and
can worsen the second.

`ExtractionCaseResult` carries `outcomes` (a `FieldOutcome` per path: expected,
actual, correct, grounded, match, required), `line_items`, `schema_valid`,
`accepted`, `truncated`, token counts, and the properties `field_accuracy`,
`required_wrong`, `silent_error` and `correct`.

## Usage

```python
from toolkit.adapters import PdfPlumberSource
from toolkit.extraction import ExtractionComponent, ExtractionConfig
from toolkit.extraction_eval import (
    ExtractionEvalRunner, ExtractionGolden, diff_extraction_reports,
)

golden = ExtractionGolden.from_jsonl("golden/invoices.jsonl")
runner = ExtractionEvalRunner(
    ExtractionComponent(llm),
    schema,
    extraction_config=ExtractionConfig(date_order="DMY"),
    load_source=lambda case: PdfPlumberSource().load(case.document),
    accept=lambda result: result.valid and not result.ungrounded_paths,
)

report = runner.execute(golden)
print(report.metrics["field_accuracy"], report.metrics["silent_error_rate"])
print("shipped and wrong:", report.silent_errors)
report.to_json("reports/run-2026-10-06.json")
```

Comparing two runs:

```python
before = ExtractionEvalReport.from_json("reports/run-a.json")
after = runner.execute(golden)
diff = diff_extraction_reports(before, after)
print(diff.render())          # metric table, then which documents broke
if diff.broken:
    raise SystemExit("regression in: " + ", ".join(diff.broken))
```

`python examples/cookbook.py extraction_eval` runs it offline.

## Limitations

- **The golden set is the measurement.** Every number here is only as good as the
  values a human verified, and a mis-transcribed expectation reads as a model error
  forever. There is no cross-checking of the set against itself.
- **No statistical significance.** Two runs differing by one case on a 20-case set
  is noise, and nothing here says so. Same gap as `evaluation`.
- **Field accuracy weights every path equally.** An invoice number and a notes field
  count the same. Use `required_paths` and the silent error rate for the distinction
  that matters; there is no per-path weighting.
- **Rows match on two columns only.** A table whose rows are distinguished by a date
  or a line number, not by description and amount, needs `description_keys` and
  `amount_keys` pointed at the right fields — and if three columns are needed to
  identify a row, this will under-count matches.
- **No per-case repair-cost attribution.** `mean_attempts` is reported, but not
  which paths the repair rounds were spent on.
- **Cost depends on the adapter populating `Usage`.** `ScriptedLLM` and `HttpLLM`
  report 0.0, because neither has a price table; `total_cost_usd` is only meaningful
  behind an adapter that fills it in.
- **One extraction per case, sequentially.** Wrap the component in
  `concurrency.bounded_map` yourself for a large set; the runner does not fan out.

## Integration Guide

1. **Write the acceptance predicate first**, and make it the one your pipeline
   actually ships with. Every silent-error number is relative to it, and a harness
   measuring a gate you do not use measures nothing.
2. Mark `required_paths` honestly. Everything required makes the silent error rate
   pessimistic; nothing required makes it zero. The right answer is the fields
   someone acts on.
3. Start the set at ten documents chosen for *disagreement*, not coverage — the ones
   that already went wrong. A set of easy documents reports 1.0 and tells you nothing.
4. Keep every report JSON in the repository. The diff is the deliverable, and it
   needs a before.
5. Fail CI on `diff.broken`, not on an aggregate threshold. An average that holds
   while two documents break is the regression you most want to catch.
6. Re-read `grounding_rate` alongside `field_accuracy`. A rise in accuracy with a
   fall in grounding usually means the model started computing values rather than
   reading them, which is correct until the arithmetic changes.

## Extraction Notes

- **Preserved:** the golden-set / report / regression-diff shape of this
  repository's own `evaluation` unit, deliberately — a second harness that behaved
  differently would be a second thing to learn.
- **Added:** path-keyed ground truth, normalised-exact comparison with no fuzzy
  matching, multiset row matching on two columns, and the silent error rate with
  both of its denominators named.
- **Isolated:** no dependency on `pipelines` or on any adapter. The runner takes a
  component and a `load_source` hook, so it measures anything with
  `.execute(ExtractionRequest)`.
