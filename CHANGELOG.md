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

## 0.8.1 - 2026-10-03

A documentation audit. No behaviour changed; several documents did, because
they were describing a repository that no longer existed.

### Corrected

- **`llm_http` was absent from `README.md` entirely** - shipped in 0.8.0 without
  ever being named on the front page. `provider` and `cli` were mentioned but
  missing from the component table.
- **"There are seven kinds"** - there are eight; `interface` arrived with the
  CLI. The kind table also omitted `graph` and `provider` from `functions` and
  `dag` and `llm_http` from `component`.
- **"Six `Protocol`s"** - there are seven, now named in the row rather than
  counted.
- **"13 adapters"** - there are 14.
- **"# 139 tests"** in a quick-start command - 368.
- **`docs/MATURITY.md`: "139 tests asserting properties"** - 368.
- **`docs/PLAYBOOK.md`: "All 15 units are currently documented"** - 20.
- **The layout tree** omitted `llm_http/`, `cli.py` and `__main__.py`, and said
  the stress corpus had 12 documents when `make_corpus.build()` emits 14.
- **`toolkit/README.md` claimed "Phases 0-5 complete ... remaining work is Phase
  6"**, and its layout section listed four units and one test file - the shape
  of the repository roughly twenty commits earlier. Rewritten to the current 20
  units across five layers, with `(standalone)` marked where `copy_tier` says
  so, and it advertised "15 unit snippets + 6 recipes" where the cookbook has
  20 and 9.
- **`toolkit/ROADMAP.md` opened by asserting** there was no shared document
  model, no way to call an LLM, no embeddings, no persistence, no caching, no
  cost control, no evaluation, no concurrency and no packaging. Every one of
  those shipped long ago. The buy-vs-build reasoning is still worth re-reading,
  so the document is kept and now says plainly that it is the original plan and
  what has since shipped, rather than reading as current status.
- **Eleven unit READMEs had no `**Layer N - dependencies - copy_tier**`
  header.** Six did, and all six were accurate. The other eleven now have one
  generated from the registry.

### Added: eight tests, so none of this can drift again

`test_packaging.py` now checks the documentation the way it already checked the
code:

- every unit is named in `README.md`
- the kind table's rows match the registry's kinds, **and** the prose count
  above it matches too
- no "N units" claim disagrees with the ledger
- the port and adapter counts in prose match the registry's `api` lists
- the layout tree lists every unit's path
- the package README's snippet and recipe counts match the registry
- no README still describes the four-component era
- every package README's header matches the registry's layer, dependencies and
  copy tier

### What the audit says about the earlier releases

Each of these numbers was correct when written. They rotted because the
repository grew and nothing checked them - the same failure mode as the
`durable_steps` limitation that still said "no DAG" after `dag` shipped, and the
install instructions that were wrong for 7 of 15 units until a test copied each
one out. **Prose is code that nothing executes.** The fix is never proofreading;
it is a test.

Verified by reintroducing five of the defects one at a time and confirming a
test fails for each. Two attempts initially missed:

* the "seven kinds" prose - the table-row check passed while the sentence above
  it lied, so a separate assertion on the word was added;
* `| **Tests** | 273 passing |` - the first guard searched for "N tests" and
  that row says "N passing". Worse, the fix silently did nothing: the regex was
  written through a shell heredoc, which turned `` into literal backspace
  bytes, so the pattern matched nothing and reported a pass. Found by scanning
  the repository for control characters, which is the same class of defect as
  the BOM and CRLF damage in 0.2.0 and has the same cause - generating code
  through a shell.

**Known friction, accepted deliberately.** The test count is now asserted
exactly, in four documents, so adding a single test requires updating all four.
That is annoying and it is the only version that is actually true; the
alternative is a number that is approximately right, which is how every one of
these claims rotted in the first place. If it becomes a nuisance the right fix
is to state the count in one place rather than to weaken the check.

---

## 0.8.0 - 2026-10-03

Two units extracted from PRism (same author, reused with permission) and
reimplemented against this toolkit's contracts. Together they close the gap
that made the generation half of the pipeline unreachable: the toolkit could
call a model but had no idea how to find one.

### Added

- **`toolkit/provider`** (layer 1, standalone, imports nothing) - resolve an AI
  provider from the environment: key, endpoint, model and API style, or a
  `SetupError` naming exactly what to set.

  One key is enough: set `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`, `GROQ_API_KEY`
  or any of thirteen shortcuts and the table supplies the endpoint, a default
  model and a label. A key with no model is still unusable, and making the
  caller look one up is how "I set the key and it still doesn't work" happens.

  Two API *styles*, not N providers. Almost every vendor exposes an
  OpenAI-compatible chat-completions endpoint, so DeepSeek, Gemini, Groq,
  Mistral, Together, xAI, Qwen, Kimi, GLM, vLLM, Ollama and LM Studio cost one
  adapter and a base URL; Anthropic's Messages API earns its own.

  `.toolkit.env` is read first and overlaid by the real environment, in that
  direction: the file is what you keep locally, the environment is what CI
  injects, and the injected value has to win or every deploy needs the file
  deleted first.

- **`toolkit/llm_http`** (layer 2, needs_package) - the `LLM` port over plain
  HTTP, no vendor SDK. Before this the only route to a real model was
  `LiteLLMClient`, so generation was unreachable on the stdlib-only base
  install. A chat completion is one POST with a JSON body; an SDK is
  convenience, not capability.

  It **classifies failures and never retries them**: `RateLimited` for 429 and
  the 5xx set, `AdapterError` for everything else, so `governor` can decide.
  Two components retrying the same call independently is how a rate limit
  becomes an outage.

  It also refuses to return a quietly wrong answer. An empty completion with
  `finish_reason=length` is an error naming `max_tokens`; an Anthropic
  `stop_reason=refusal` is an error; a `content_filter` is an error. A
  *partial* answer is returned intact with the reason on the `Completion`,
  because that is usable.

  Anthropic takes the system prompt as a top-level field rather than a message,
  and sending `role: "system"` inside `messages` produces a 400 whose text does
  not point at the cause - so the lifting happens in the client, not in every
  caller.

- Recipe **R9 `byok_generation`**: `provider -> llm_http -> governor -> cache ->
  KnowledgeBase`, the whole generation stack with no third-party package.

### The security change I made to the original

`Provider` overrides `__repr__` and `__str__` to redact the key, showing only
its length and last four characters. PRism's equivalent is a plain dataclass
whose generated `repr` prints `api_key` in full.

Nothing in that project appears to print it today, but the places a provider
object ends up - a debug log line, an unhandled traceback, a crash reporter, a
CI job's captured output - are all places a credential must never reach. One
`logging.debug(provider)` leaks a key into a log aggregator for its whole
retention period, and that is not retrofittable: the key has to be rotated.
`Diagnosis` reports variable names only, and no error message echoes a key.
Tests assert all of it, including that `"%s" % [provider]` is safe, because
formatting a container calls `repr` on its members.

### Testability as a design feature

`HttpLLM` takes an injectable `transport`, so all 43 tests for these two units
run offline with no key. The error translation **is** the valuable part of
`llm_http`, and error handling that can only be exercised against a live
provider is error handling nobody tests. PRism calls `urllib` directly, which
is why its own failure paths have no tests.

### What this release got wrong

- The composition test called `GovernorConfig(max_input_tokens=...)`, a field
  that does not exist - it is `max_total_tokens`. Written from memory of an API
  in this same repository rather than from the dataclass, which is the mistake
  that the registry's `test_declared_deps_match_the_code` exists to catch in
  the large and that a `TypeError` caught here in the small.
- `_raise_for_status` formatted a message with `status and label` where `label`
  was meant. It evaluated correctly because `status` is always truthy at that
  point, which is the worst kind of bug: right answer, wrong reason.
- A 503 with an empty body rendered as `"... (HTTP 503): {}"`. An empty JSON
  object carries no information and should not be appended to an otherwise
  clean message.

### Still open

Neither unit has `used_in_projects` yet. They are tested but not *used*, which
by this repository's own `production_ready` gate means they have not earned
anything - and the gap they close, validating the generation half against real
input, is exactly the project that has not been run. `Usage.cost_usd` stays
0.0 for want of a per-model price table, so `governor`'s cost ceiling only
works if the caller supplies prices.

---

## 0.7.0 - 2026-10-03

### Added

- **A command line interface** - `toolkit/cli.py`, reachable as `toolkit` or
  `python -m toolkit`, with `ingest`, `ask`, `eval` and `inspect`.

      toolkit ingest ./papers --save papers.kb
      toolkit ask papers.kb "what limits the throughput?"
      toolkit eval papers.kb golden.jsonl --out metrics.json
      toolkit inspect papers.kb

  On the 49-paper arXiv corpus: ingest 49/49 with 0 failures and 3,036 chunks,
  then **`ask` answers in 2.2 seconds** including interpreter start and model
  load. `eval` reproduces the bespoke validation runner's numbers exactly -
  `hit_rate@10` 0.9048, `refusal_accuracy` 0.8750, `dataset_errors` 3,
  `scored_cases` 21 - which is the cross-check that matters: the CLI is not a
  second code path with its own answers.

  `ask` exits 0 when it answered and 1 when it refused. Both are successful
  runs of the program and only one found something, so the difference has to
  reach the shell.

  Per-document progress comes from `IngestConfig.on_document`, added in 0.5.0
  for exactly this. Every vendor import sits inside a function, so `--help`
  works on the stdlib-only install and the CI job asserting no vendor SDK in
  `sys.modules` keeps passing.

  This had to wait for 0.6.0. Without `save`/`load` every invocation would
  re-parse the corpus - 38 minutes for 49 papers - which is no interface at all.
  The earlier recommendation to build the CLI before persistence had the
  dependency backwards.

- `manifest.json` now records `embedder_class`, so `ask` and `eval` rebuild a
  compatible embedder without being told which one. Informational only: `load`
  still keys vector reuse off `embedder_model_version`.

### What this release got wrong

- The vendor-SDK test asserted against `sys.modules` in-process. It passed
  alone and failed with the full suite, because by then other tests had
  imported pdfplumber and fastembed and the assertion blamed the CLI for their
  imports. It now runs in a fresh interpreter, which is the only place the
  claim means anything.
- `REGISTRY.json` declared `core` among the CLI's dependencies. `cli.py` does
  not import it, and `test_declared_deps_match_the_code` said so immediately -
  the ledger is checked against the import graph, not trusted.
- `toolkit/__main__.py` tripped `test_registry_covers_every_unit_on_disk`. The
  shim is four lines calling `cli.main` and carries no capability, so the test
  now skips it rather than the ledger gaining an entry with nothing to say.

---

## 0.6.0 - 2026-10-03

The index now survives the process that built it. Everything inconvenient about
this toolkit traced to one fact: a 38-minute ingest could not be reused.

### Added

- **`KnowledgeBase.save(directory)` and `KnowledgeBase.load(directory, **kwargs)`.**
  Measured on the 49-paper arXiv corpus:

      ingest (parse cache warm)     308.2 s     3036 chunks
      save                            0.3 s
      load                            0.7 s     3036 chunks

  A cold ingest of that corpus takes **38 minutes**, of which about 85% is the
  PDF parser. It now reloads in **under a second**, with identical chunk ids,
  an identical answer to the same question, and all 3,036 provenance regions
  intact. Total on disk is 10.8 MB for 212 MB of source PDFs.

  A directory of plain files rather than one opaque blob, and no pickle, because
  the index is the expensive artefact and has to survive a toolkit upgrade that
  changes a dataclass:

      manifest.json     format version, embedder version, counts, dimension
      chunks.jsonl      one chunk per line, provenance included
      documents.jsonl   doc_id -> source path, the supersede index
      vectors.bin       float32 via `array`, row-aligned to chunks.jsonl

  `load` reuses saved vectors when the embedder matches the one that produced
  them, and **re-embeds from saved text when it does not** - so an index outlives
  the model that built it, paying only the embedding cost and never the parser
  again. Mixing vectors from two embedders remains refused; that guard already
  existed and is what the manifest's `embedder_model_version` feeds.

  `load` is a classmethod taking the same keyword arguments as the constructor,
  so an index saved without an LLM can be loaded with one.

- `KnowledgeBase._vectors`, a chunk_id -> embedding mirror, because no
  `VectorStore` port method reads a vector back out: `search` returns
  neighbours, not a named row. About 4.6 MB for 3,000 chunks at 384 dimensions,
  against the tens of megabytes the chunk text already occupies.

### Design note: why the mirror cannot drift

Three code paths remove chunks - the stale sweep inside `_do_ingest`, `forget`,
and supersede - and a second mirror of the same data is exactly the kind of
thing that goes stale when a fourth appears. So `_chunks` is authoritative:
`save` iterates it, re-embeds anything whose vector is missing, and never writes
a vector whose chunk is gone. A stale or incomplete mirror therefore costs time,
never correctness. The pops were added at all three sites as well, and a test
asserts that forgetting a document before saving shrinks `vectors.bin` to match.

### What this release got wrong

- The first version of the reuse test asserted a hand-written version string
  (`"hashing-256"`) against `HashingEmbedder`'s real one
  (`"hashing-v1-d256-tri1"`), so it reported a re-embed and looked like an
  implementation bug. The fixture now mirrors the wrapped embedder's version by
  default and takes an explicit one only to simulate a *different* model. A test
  double whose identity does not match the thing it doubles will manufacture
  failures.
- mypy caught a real shadowing mistake: `save` bound `vector` twice with
  different types, once as `list[float]` from the re-embed path and once as
  `Sequence[float] | None` from the mirror lookup.

### Still open

The CLI this unblocks is not built yet. It was the obvious next convenience and
is now worth doing, because `toolkit ask papers.kb "..."` loads in a second
rather than re-parsing 212 MB per question - which is why persistence had to
come first, and why the earlier recommendation to build the CLI first was wrong.

---

## 0.5.0 - 2026-10-03

Closes the six remaining findings from the first real-world validation. Every
threshold here was measured on 49 real arXiv papers, not chosen.

### Fixed

- **Token estimates under-counted tables of numbers (F3).** `estimate_tokens`
  modelled prose and nothing else. BPE packs about four letters per token but
  gives most punctuation marks and digits a token each, so a results table costs
  far more than its character count implies - the worst real case was a
  1,534-character table estimated at 383 tokens and actually tokenised at 1,005,
  a silent 2x overflow of a 512-token budget. A third arm,
  `words*0.9 + punctuation + digits`, now competes with the existing two.
  Measured on 2,780 real chunks against `cl100k_base`, chunks the estimate
  called in-budget while they really were not fell from **959 to 231 (-76%)**;
  on the re-chunked corpus, chunks over a 512-token budget fell **35.0% to
  9.8%** and the worst overshoot from **2.65x to 1.59x**. Coefficients are 1.0
  rather than tuned decimals so this does not specialise to the corpus it was
  measured on.
- **The relevance gate answered questions the corpus cannot answer (F6).** It
  pooled term overlap across the top ten hits, so one shared word passed - and
  on a real corpus something always shares a word. All eight pre-registered
  unanswerable queries were answered, with citations to real but irrelevant
  chunks. Coverage is now measured **per chunk**, best chunk deciding, against
  the new `AskConfig.min_term_coverage` (default 0.5). Swept over 24 answerable
  and 8 unanswerable queries: thresholds 0.4-0.6 all give 100% of answerable
  passing and 88% of unanswerable refused, so 0.5 sits mid-plateau rather than
  on a knife edge. On the full run `refusal_accuracy` went **0.0000 to 0.8750**
  with `false_refusal` unchanged at **0.0000**. Set it to 0.0 for the old
  behaviour.
- **`EvalRunner` scored unreachable ground truth as a retrieval miss (F5).** A
  case whose expected snippet exists in no indexed chunk cannot be retrieved by
  anything, so counting it as a miss blames the ranker for an upstream fault -
  this harness reported `hit_rate@10` of 0.42 where direct measurement over the
  resolvable cases gave 0.95. Such cases are now flagged
  `CaseResult.ground_truth_missing`, counted as `dataset_errors`, and excluded
  from the retrieval metrics, with `scored_cases` naming the real denominator.
  They are still scored for refusal, because refusing is the correct response to
  a question the corpus cannot answer.

### Added

- `KnowledgeBase.chunks()` and `.documents()` (F7): public snapshots of what is
  indexed. Auditing provenance, checking whether a phrase is in the corpus, or
  exporting the index previously meant reaching into the private `_chunks` dict.
  F5 is only implementable because this exists.
- `EvalConfig.lexical_weight`, `.dense_weight`, `.rrf_k`, `.rerank_budget` (F8),
  all recorded in the report's config snapshot so `diff_reports` can attribute a
  metric change to the ablation that caused it. Without them the harness could
  not answer the one question `hybrid_ranker` exists to settle, because there
  was no way to turn a retriever off between runs.
- `IngestConfig.on_document`, called with `(index, total, outcome)` after each
  document including failures (F9). A 49-document ingest took 38 minutes and
  printed nothing, so a slow run was indistinguishable from a hung one; the only
  way to observe progress was to query the durable checkpoint table from another
  process. A callback rather than logging, because a library that prints is a
  library you cannot embed, and exceptions from it are deliberately not caught.
- `DocumentOutcome.status` can now be `'reingested'` (0.4.0) and the docstring
  defines all four values.

### What this release got wrong

- **A reported improvement was a measurement artefact.** `hit_rate@10` reads
  0.7917 before F5 and 0.9048 after, but 19 cases hit in both runs: 0.7917 x 24
  and 0.9048 x 21 are both 19. F5 changed the denominator, not the numerator.
  Presenting that as a retrieval gain would have been exactly the error F5
  exists to prevent, and it is recorded here because the temptation was real.
  The genuine retrieval gains are the earlier extraction fixes, 0.4167 ->
  0.7917.
- **A regression test asserted the wrong arm.** The first version of
  `test_f3_plain_prose_is_not_inflated` bounded prose against `words*1.45` and
  failed, because prose is dominated by the characters-over-four arm, which
  already over-estimates English by about a third. That is pre-existing and
  untouched by this change; the test now asserts the estimate is identical to
  the two original arms, which is the actual claim.
- The validation harness went on reporting F7 and F8 as findings after both were
  fixed. It now uses the public accessor and runs its ablation through
  `EvalRunner`, which is the only honest proof those additions work.

### Still open

`F4` 29 chunks carry literal `<|endoftext|>` sequences - corpus reality, and any
caller counting tokens with a real tokenizer must pass `disallowed_special=()`.
`F11` three golden snippets match no chunk, because they were drawn from arXiv
metadata abstracts whose wording differs from the PDF bodies - a flaw in the
golden set, not the toolkit. One of eight unanswerable queries still gets
through the gate at 0.60 coverage ("the default port for a PostgreSQL server
connection" shares most of its vocabulary with ML prose); term overlap cannot
separate that and raising the threshold starts refusing real questions.
`hybrid_ranker` remains **NOT PROVEN** - lexical alone wins on this metric by
both measurement paths, but ground truth is verbatim snippet containment, which
scores BM25 on precisely its own task. Settling it needs semantic relevance
judgements, which is the one thing this validation project has shown it cannot
provide.

---

## 0.4.1 - 2026-10-03
### 0.4.1 - 2026-10-03

- **Rotated text is excluded as a separate flow (F10).** Every arXiv PDF carries
  a rotated identifier down its left edge. `extract_words` returned it as
  reversed fragments positioned in the margin, and `doc_layout` - which has no
  notion of orientation - interleaved them into body lines, producing
  `an tc abelian surface` and `routinely extc ceeding` and displacing an entire
  line of one abstract. Reading order is recovered by sorting spans on position,
  which is only meaningful within one orientation, so rotated text cannot share
  the flow. `PdfPlumberSource` now filters to upright glyphs and reports the
  count in `Document.metadata["rotated_glyphs_excluded"]` - excluding content
  silently would be worse than the defect, and that count is how a caller
  notices a landscape page has lost its body. `doc_layout`'s limitations and
  registry entry now say so.
- Cumulative effect of the three extraction fixes on the real corpus, with no
  change to retrieval, ranking or chunking: `hit_rate@10` **0.4167 -> 0.7917**
  (+90% relative), `ndcg@10` 0.3347 -> 0.6390, `overall_correct` 0.3846 ->
  0.7308, provenance 18/20 -> 20/20, golden snippets matching no chunk 13 -> 3.
- The validation parse cache now keys on a hash of the parser's own source
  rather than on a list of its settings. Naming `WORD_GAP_RATIO` in the key
  caught the first behaviour change and missed the second; hashing the code
  catches both without having to remember to.
- `hybrid_ranker` remains **NOT PROVEN**, now because the experiment saturated:
  lexical recall@10 reached 1.000 against verbatim-snippet ground truth, so
  fusion can only dilute it. That is arithmetic, not evidence. Settling it needs
  semantic relevance judgements.

---

## 0.4.0 - 2026-10-03

First release driven by real-world use rather than by authored tests. The toolkit was
run over 49 uncurated arXiv PDFs (212 MB, nine fields, published 1986-2026) as a paper
triage tool; the full write-up is `validation/paper_triage/FINDINGS.md`. Nine findings,
two critical. **273 tests, 94% coverage and a hostile corpus had missed all of them.**

### Fixed

- **Word spaces were destroyed in every real PDF.** Two stages disagreed about how wide
  a gap has to be before it means "a space". `PdfPlumberSource` took pdfplumber's
  default absolute `x_tolerance=3`, and `doc_layout` rejoined spans at
  `gap > 0.25 * font_size`. LaTeX's Computer Modern sets inter-word space below both at
  body sizes, so whole lines were merged into one token:

      Supersingularabeliansurfacesareessentialinisogeny-based

  Present in **49 of 49** documents, a median **22.9%** of characters and up to
  **84.2%** in the worst case, concentrated in abstracts and titles. Glued text is
  unmatchable by any lexical index because its tokens do not exist, and it embeds
  poorly, so this was a retrieval defect wearing an extraction defect's clothes.

  Both stages now share one constant, `doc_layout.WORD_GAP_RATIO = 0.15`, which
  `PdfPlumberSource` imports so they cannot drift apart again. A *ratio* rather than an
  absolute tolerance so the threshold scales with type size: 1.35pt at 9pt body text,
  3pt at a 20pt heading. The value was measured, not guessed - across 8 documents
  spanning 1986-2026, word count rises from 12,627 to ~21,980 and then plateaus below
  0.15 while the share of one- and two-character tokens stays flat at 20.4%, so the
  extra splits are real spaces recovered rather than words broken apart.
  `PdfPlumberSource(x_tolerance_ratio=...)` overrides it.

- **Resuming an ingest produced a silently empty index.** With `IngestConfig.durable_db`
  set, re-running in a *new* process reported 49 documents replayed, 0 failures and
  `count() == 0`; every subsequent question was then refused, which is correct behaviour
  on an empty index and is exactly what hid the fault. A checkpoint records a step's
  return value, never its side effects, and `KnowledgeBase._chunks` is always
  process-local. `_ingest_one` now verifies the document is actually present before
  trusting the record, and redoes the work when it is not, reporting the new status
  `reingested`. Replay within one process still skips the work, so crash-resume is
  unaffected.

### Changed

- `DocumentOutcome.status` gains `'reingested'`, and its docstring now defines all four
  values.
- `REGISTRY.json`: ten units record `used_in_projects: ["validation/paper_triage"]`.
  Maturity is deliberately **unchanged** - whether a validation harness counts as "a
  real project" under the `production_ready` gate is a judgement for the owner, not
  something this release should assume.
- `stress/make_corpus.py` gains `tex_tight_spacing`, which positions words with an exact
  gap using a `TJ` array with explicit kerning, the way a TeX engine does. The existing
  fixtures emit a whole line as one string and so inherit Helvetica's comfortably wide
  space glyph (~3.06pt at 11pt, just *above* pdfplumber's old threshold) - which is
  precisely why nothing in the hostile corpus had ever reproduced this.

### What this release got wrong

- **A test asserted the defect.** `test_durable_ingest_replays_completed_documents`
  checked that a fresh `KnowledgeBase` over an existing checkpoint database reports
  every document `replayed` with zero embedding calls - and passed. It never asserted
  `count()`, so it certified the silent-data-loss path as correct behaviour. It is now
  two tests, one per process model, both asserting the index is actually populated.
- **The synthetic corpus could not have found either defect**, and that is the real
  lesson. `make_corpus.py` was written by the same author as the parser, against the
  failure modes that author had already imagined. It found ten genuine defects and was
  worth building, but no such corpus probes the assumptions its author did not know they
  were making. Nine of these nine findings needed input nobody here wrote.
- Four of the author's own assumptions about the public API were wrong while writing the
  consumer (`IngestResult.failures`, a public chunk accessor, the metadata path key, the
  fusion result shape). Only one was a real gap - there is still no public way to
  enumerate indexed chunks - but a facade whose shape cannot be guessed is a
  documentation problem.
- Two measurements were reported wrongly before being corrected: a provenance check that
  compared whole chunks against only their first bounding box (10/20 "failures" were
  really 2/20), and a `dag` fan-out comparison contaminated by a stray process racing on
  a shared cache (the apparent 11% chunk loss did not exist). Both corrections are
  recorded in `FINDINGS.md`.

### Still open from this round

`F3` the heuristic token counter puts 44% of chunks over a 512-token budget against
`cl100k_base` (worst 2.65x); `F5` `EvalRunner` scores an unmatched `expected_snippet` as
a retrieval miss, so it reported `hit_rate@10` of 0.42 where direct measurement gave
0.95; `F6` the relevance gate answered both pre-registered unanswerable queries with
real-but-irrelevant citations; `F7`-`F9` no public chunk accessor, no fusion weights in
`EvalConfig`, no progress reporting on a 38-minute ingest.

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
