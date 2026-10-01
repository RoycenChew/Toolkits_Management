# Stress harness

A deliberately hostile document corpus, and a probe harness that reports what breaks.

**This is not the test suite.** `toolkit/tests/` is the ratchet — it pins behaviour
that must not regress. This directory is the opposite: an exploratory tool whose job
is to *find* defects. The two have different success conditions, which is why they
live apart.

```bash
python stress/make_corpus.py            # generate the corpus into a temp dir
python stress/make_corpus.py ./corpus   # or a path you choose
python stress/run_stress.py             # generate, probe, print findings
```

Runs with nothing installed: the PDFs are written by hand rather than with reportlab.
Non-Latin scripts go in Markdown, because Helvetica is Latin-1 only.

## The corpus

Each file attacks one assumption. A stress corpus everything survives was built too
politely.

| File | Attacks |
|---|---|
| `two_column_paper.pdf` | XY-cut columns, a full-width title straddling the gutter, a centred page number *inside* the gutter, footnotes, a running header, a full-width heading mid-page |
| `invoice_batch.pdf` | multi-document splitting, page-number restarts, and number formats an extractor must coerce |
| `hyphenated_contract.pdf` | words split across line breaks by justification |
| `emphasis_not_headings.pdf` | bold run-in emphasis and ALL-CAPS warnings that are **not** headings |
| `table_heavy.pdf` | a text-grid table, which is genuinely multi-column by geometry |
| `no_text_layer.pdf` | a scan; must fail with a clear error, not an empty `Document` |
| `unicode_mess.md` | ligatures, smart quotes, non-breaking and zero-width spaces, a soft hyphen, a decomposed accent |
| `cjk_mixed.md` | the token estimator, which has no spaces to count |
| `injection.md` | a live indirect prompt-injection payload |
| `wall_of_text.md` | the sentence splitter and the word-split fallback |
| `bulletin_a.md` / `bulletin_b.md` | retrieval redundancy — 95% identical documents |

## Reading the output

```
STATUS   meaning
FAIL     a defect. The valuable output
KNOWN    a documented limitation, confirmed
PASS     the probe's assumption holds
CRASH    the probe itself raised — also a finding
```

## What it found

First run: **FAIL=10 KNOWN=2 PASS=8.** Current: **FAIL=0 KNOWN=1 PASS=18.**

The ten defects are listed in [`../docs/MATURITY.md`](../docs/MATURITY.md) and pinned
by `toolkit/tests/test_stress_regressions.py`. The most instructive:

- **Two-column pages were spliced into single lines.** Three wrong implementations
  before it worked: vertical cuts only, then requiring a zero-ink gap (a centred page
  number bridges it), then treating any crossing row as full-width (columns sharing
  baselines make every row cross).
- **The fix then tore tables apart**, because a table *is* multi-column by geometry.
  Extent cannot distinguish a table from prose; ink density can — prose fills ~0.8 of
  its row, table rows nearer 0.3.
- **CJK token estimates were 3.6× too low**, so every chunk silently overran the
  embedding window for non-Latin documents.
- **`PlainTextSource` discarded every heading not followed by a blank line** — which
  is ordinary Markdown, and had been degrading the sample corpus the whole time.

## Adding a probe

1. Add a file to `make_corpus.py` with a docstring naming the assumption it attacks.
2. Add a `probe_*` function to `run_stress.py` that `record(...)`s PASS, FAIL or KNOWN.
3. When a FAIL is fixed, **pin it** in `toolkit/tests/test_stress_regressions.py` —
   the harness finds defects, the suite stops them returning.
4. If a limitation is genuine and not being fixed, mark it KNOWN *and* add a
   characterisation test, so a future fix forces the docs to update.

## Relationship to the test suite

```
stress/           exploratory. Expect failures. Run by hand
toolkit/tests/    ratchet. Zero failures tolerated. Run in CI
```

The corpus generator is imported by `test_stress_regressions.py`, so the two stay in
step: a corpus change that breaks a pinned behaviour fails CI.
