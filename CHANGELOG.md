# Changelog

Notable changes, newest first. Each entry records what was built **and what it got
wrong**, because the defects are the more useful half — they say where the design was
weak, and several were only found by a structural guard rather than by a test anyone
thought to write.

---

## Version policy

Semantic versioning, with one clarification about what "the API" means here.

**Stable surface** — breaking changes require a major bump:

- the types in `toolkit/core` (`Document`, `Block`, `Chunk`, `Provenance`, `BBox`,
  `Usage`, `Completion`, `SearchHit`, `Message`)
- the protocols in `toolkit/ports.py`, including their documented invariants. Adding a
  **required** method to a port is breaking, because every existing implementation
  stops satisfying it — `delete` and `model_version` were both such changes
- each component's `execute(...)` signature and its request/result dataclasses
- `copy_tier` for a unit: promising less than before is breaking

**Unstable, may change in a minor release:**

- anything prefixed `_`
- default *values* in a config dataclass. Tuning a threshold is a minor change; the
  relevance gate's `min_dense_similarity` went from `0.25` to `None` in 0.2.0 because
  measurement showed no portable default exists
- which adapters ship, and their constructor keyword arguments
- the content of `REGISTRY.json` and the test suite

**Pre-1.0 caveat.** While the version is below 1.0, a minor bump may break the stable
surface if a contract is found to be wrong — `0.2.0` added required port methods for
exactly that reason. **1.0 will mean every unit has `production_ready` maturity in
`REGISTRY.json`**, which requires use in a real project, not a test count. No unit has
reached it yet.

---

## 0.3.0 — 2026-10-01

### Added
- **`examples/cookbook.py`** — one runnable offline snippet per unit, plus six named
  composition recipes. Executed by the test suite, so a documented example cannot rot
  into a lie.
- **`REGISTRY.json`** — the ledger. Layer, kind, capability, dependencies, copy tier,
  limitations, maturity, example and recipe membership per unit, plus the component cap
  and layer rule.
- **`toolkit/tests/test_packaging.py`** — 34 tests that verify the toolkit's claims
  about itself rather than restating them.
- **`docs/PLAYBOOK.md`** — nine operating rules, each tied to the incident that
  produced it, and a thirteen-point Component Definition of Done.
- `toolkit/core/README.md`, `toolkit/adapters/README.md`, `stress/README.md`.
- `py.typed`, so consumers actually receive type information.
- `.gitattributes` enforcing LF.
- Coverage configuration with a floor.

### Fixed
- **The central claim was false for 7 of 15 units.** Every README said "copy the
  directory"; `cache`, `governor`, `chunking` and `extraction` each import `core`, and
  `adapters`, `pipelines` and `evaluation` need most of the package. The *pattern* was
  fine — all 16 copy tests pass — the documentation was wrong, and nothing checked it.
- **Dead code**: `KnowledgeBase._context` was orphaned when guardrails were wired in
  and went on being maintained for nothing. Found by coverage, not by review.
- **An untested retry branch**: the governor's `AdapterError` path had no test, only
  the `RateLimited` one. They differ — the first falls back to exponential backoff, the
  second honours a provider's `retry_after` — and a transient backend failure is the
  more common case. Found by coverage.
- **A UTF-8 BOM in 6 files and CRLF in 11**, from PowerShell's `Set-Content -Encoding
  utf8`. Python tolerates a BOM on import, so 203 tests stayed green while tooling
  broke: it crashed `ast.parse` in the audit script that found it.
- Fully documented every `ports.py` protocol method. 16 of 27 had no docstring, and a
  protocol body is `...` — the docstring *is* the contract.
- The root README claimed every component exposes one `execute()`. False for 6 of 15;
  replaced with the seven honest shapes.
- `entity_resolution` appeared in no composition at all, making it exactly the
  speculative inventory the archiving rule targets. Given a reconciliation recipe.

### Changed
- Component count 11 → 15 units (counting contracts and adapters, which the earlier
  count silently omitted).

---

## 0.2.0 — 2026-09-30

### Added
- **`toolkit/guardrails/`** — indirect prompt-injection defense: source delimiting, a
  system preamble that describes the attack explicitly, pattern neutralisation with
  visible audit markers, and an output policy catching echoed instructions and URLs
  absent from every source.
- **`toolkit/evaluation/`** — golden sets keyed on snippets rather than chunk ids, IR
  metrics, JSON report persistence, and a regression diff that names broken cases.
- **`toolkit/extraction/`** — schema-driven extraction with a targeted repair loop,
  per-field provenance and grounding, plus a multi-document splitter.
- **`toolkit/pipelines/`** — `KnowledgeBase.ingest_folder()` and `.ask()` with verified
  citations, a relevance gate and an extractive mode.
- **`toolkit/chunking/`**, **`cache/`**, **`governor/`**, **`concurrency.py`**.
- **`stress/`** — a hostile 12-document corpus and probe harness.
- Delete path: `delete` and `delete_by_doc` on both store ports,
  `KnowledgeBase.forget()`, and an orphan sweep on re-ingest.
- `ScreeningLimits` — size, page and time caps applied before any parser sees a file.
- `model_version` on the `Embedder` port.
- CI: a stdlib-only job and a with-backends job.

### Fixed — the ten defects the hostile corpus exposed
- **Two-column pages spliced into single lines.** Three wrong implementations: vertical
  cuts only; then requiring a zero-ink gap, which a centred page number bridges; then
  treating any crossing row as full-width, which columns sharing baselines always are.
- **Tables torn apart** by the fix above. A table genuinely *is* multi-column by
  geometry; ink density distinguishes it from prose (0.8 versus 0.3), extent cannot.
- Numbered section headings (`2. Method`) classified as list items.
- Hyphenated line breaks never rejoined (`custo-` + `mer`).
- Running headers on short documents indexed as body text.
- Ligatures, non-breaking and zero-width spaces, and soft hyphens unnormalised —
  `conﬁguration` is unsearchable as "configuration" and looks identical.
- Markdown headings with no following blank line silently discarded.
- **CJK token estimates 3.6× too low**, so every chunk overran the embedding window for
  non-Latin documents.
- European decimal commas and accounting negatives unparsed — the second inverts the
  sign of every credit note.
- Near-duplicates filling the top-k.

### Fixed — correctness review
- **The orphan sweep could never have worked.** `doc_id` is a content hash, so an
  edited document arrives with a new id and `delete_by_doc` cannot reach the previous
  version. Identity of "the document at this path" is the path; added a supersede index.
- Swapping embedders at the same dimension silently corrupted retrieval. Now refused at
  ingest *and* query.
- Untrusted parsing had no caps. A 2 KB PDF can declare 40,000 pages.
- `KnowledgeBase(lexical_index=None)` could not disable the keyword index — `None` was
  indistinguishable from "not supplied". Needed a sentinel.

### Changed
- **`AskConfig.min_dense_similarity`: `0.25` → `None`.** Measurement killed the
  original design: on the sample corpus an irrelevant query scored **0.293** against a
  relevant one's **0.248**, because `HashingEmbedder` compares character trigrams, not
  meaning. No single cosine threshold is portable, so the dense arm of the relevance
  gate is now opt-in with a documented calibration procedure.
- `ChunkConfig.overlap_tokens` and `min_tokens` derive from `max_tokens` instead of
  being fixed, so lowering one field can no longer produce a self-contradictory config.
- `doc_layout` column decomposition now runs *before* line assembly. The reverse order
  splices columns at matching y coordinates — a silent corruption.
- `min_column_width_ratio` 0.15 → 0.25: a body column narrower than a quarter page is
  implausible, a table gutter is not.

---

## 0.1.0 — 2026-09-30

Initial four components, extracted as algorithms rather than copied as code:
`doc_layout`, `hybrid_ranker`, `entity_resolution`, `durable_steps`.

No source was taken from any upstream project. That was a licensing constraint, not a
preference: the strongest document parsers are copyleft — Marker is GPL-family with a
revenue threshold on its weights, MinerU is AGPL — and neither can be vendored into a
permissive toolkit. Everything was reimplemented from published algorithms, with
permissive prior art credited per component.

### Fixed during construction
- Line assembly spliced left- and right-column text at matching y coordinates.
- `min_column_width_ratio` promoted a table gutter to a column break.
- The token budget was enforced on per-unit sums while the shipped chunk is the
  rendered string, which includes separators and breadcrumbs the sum never saw.
