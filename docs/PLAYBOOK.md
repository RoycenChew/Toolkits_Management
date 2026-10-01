# Playbook — how this toolkit is maintained

Operating rules for a reusable component library. Every one was earned from
something that actually went wrong in this repository, and each is stated with the
incident so it is arguable rather than dogma.

Companions: [`ARCHITECTURE.md`](ARCHITECTURE.md) (target design) ·
[`MATURITY.md`](MATURITY.md) (honest readiness) · [`../REGISTRY.json`](../REGISTRY.json)
(the ledger).

---

## 1. A claim without a test is a liability

Unverified claims rot into lies without anyone lying. Every promise in a README
should have a mechanical guard.

| Claim | Guard | Found |
|---|---|---|
| "stdlib-only base install" | CI job asserting no vendor SDK in `sys.modules` | **a real bug on run #1** |
| "two adapters per port" | test fails below two | the `Reranker` port had zero |
| "independently copyable" | copy each unit out, import in a subprocess | **the claim was wrong for 7 of 15 units** |
| "core never imports platform" | layer rule over the registry graph | — |
| "no BOM, LF endings" | byte check over tracked sources | 6 files with a BOM, 11 with CRLF |

The copy test is the clearest case. Ten READMEs said *"copy the directory"*; four of
those units import `core` and three need most of the package. The instruction did not
work, and nothing noticed for days.

**Rule:** when you write a claim in a README, write the test in the same commit.

## 2. Structural guards beat authored tests

The stdlib-only CI job caught a defect **203 hand-written tests could not** — because
those tests were written on a machine where every optional backend was installed. A
green local suite said nothing about the base install.

Prefer tests that assert properties of *the whole*: import graphs, dependency
closures, encoding, packaging, counts. They catch what you cannot imagine, which is
exactly the category your unit tests miss.

## 3. Usage is the only completion gate

Tests prove a component does what you thought. A real project proves you thought the
right thing.

`REGISTRY.json` carries `used_in_projects` for every unit. Nothing reaches
`production_ready` without an entry. **All 15 units are currently `documented` and
none are `production_ready`** — that is the honest state, and the model exists so the
two are not confused.

## 4. One ledger; generate the rest

`REGISTRY.json` is the single source of truth. Hand-maintained duplicates drift: the
root README's component count and test count both had to be corrected twice before
the registry existed.

JSON rather than YAML deliberately — the packaging tests parse it, and they run in the
stdlib-only job where no YAML parser is installed. A ledger the bare job cannot read
is a ledger the bare job cannot verify.

## 5. Never build a layer without a consumer

Each new unit ships with a recipe that uses it and an example that runs. A component
with no consumer is speculative inventory: it costs CI time, review time and
attention, and returns nothing.

Six of the ten composition recipes in `ARCHITECTURE.md` need a DAG engine. That is a
stronger argument for building one than any priority score.

## 6. A hard cap forces the trade-off

**25 units.** The 26th requires retiring one; `test_component_cap_not_exceeded`
enforces it.

Every individual addition always looks justified. Only a cap makes the comparison
actually happen.

## 7. Characterisation tests for known limits

Assert what *doesn't* work, so fixing it forces the documentation to change.

When injection defense landed, `test_known_limitation_prompt_injection_is_obeyed`
failed — exactly as designed — and forced the revision of `MATURITY.md` and
`ARCHITECTURE.md` stage 19. The guardrails suite still carries
`test_known_gap_obfuscated_payloads_evade_patterns`, which asserts that obfuscated
payloads get through.

**A limitation that isn't asserted is a limitation you'll forget you have.**

## 8. The cold-start test

Can you use a component after two weeks away, from its README alone, without opening
the source? No CI can run this. Try it on `entity_resolution` — the one with EM and
learned blocking predicates — and you will learn more than any audit produces.

## 9. Coverage is a defect-finder, not a score

94% of the library, floored at 90%. The number is not the point — what it found is:

- **`KnowledgeBase._context` was dead code**, orphaned when guardrails were wired in
  and still being maintained for nothing. No review caught it.
- **The governor's `AdapterError` retry branch had no test**, only the `RateLimited`
  one. The two differ — one honours a provider's `retry_after`, the other falls back to
  exponential backoff — and the untested branch is the more common failure.

Chase uncovered *branches*, not the percentage. The floor sits below the current figure
on purpose: a floor at the measurement fails on any honest refactor, and a floor far
below enforces nothing.

## 10. Make maintenance cost visible

Per unit, in the registry: dependencies, declared limitations, `last_reviewed`. Cost
is invisible by default, which is how libraries accumulate a hundred components nobody
dares delete.

---

## Component Definition of Done

A unit is **complete** when every line is true. Not a feeling — a checklist.

```
[ ] Entry point declared, and its API shape is honest
    (component / wrapper / functions / contracts / adapters / facade / harness)
[ ] Input and output contracts are explicit types
[ ] Error behaviour documented: what raises, what degrades,
    what is poison versus retryable
[ ] Tests assert PROPERTIES, not merely that the code runs
[ ] Characterisation test for every known limitation
[ ] Imports standalone with its declared deps — verified by the copy test
[ ] No dependency on a higher layer — verified by the layer rule
[ ] README: purpose, architecture, I/O schema, limitations, integration guide
[ ] Installation instruction names every directory the reader must copy
[ ] One runnable example, offline
[ ] REGISTRY.json entry: layer, kind, deps, copy_tier, limitations, maturity
[ ] Appears in at least one recipe
[ ] Type-complete: py.typed shipped, mypy clean
```

**Current state: every unit passes all thirteen.** The last two items closed on
2026-10-01: `examples/cookbook.py` gives each unit a runnable offline snippet, and six
named recipes in `REGISTRY.json` give each one a consumer. Both are enforced —
`test_packaging.py` asserts the declared references point at functions that exist, and
CI executes the whole cookbook, so a documented example cannot rot into a lie.

The one exemption is deliberate: `contracts` units need no recipe of their own, because
a protocol is consumed implicitly by every recipe using an implementation of it, and
inventing a recipe to satisfy a checklist is the cargo cult this rule exists to
prevent.

---

## Not-doing list

Stated so these stop occupying attention:

| Not doing | Why |
|---|---|
| Blanket docstring coverage | 193 of 326 public symbols have none, but most are `count()`, `delete()` and trivial accessors. Forcing prose onto those buries the docs that matter. Protocol methods are the exception — there a docstring *is* the contract |
| Filling empty domains for symmetry | Computer Vision is empty and nothing depends on it; Code Intelligence is empty and the architecture's own boundary rule needs it. Same emptiness, opposite priority |
| A second implementation of an existing capability | Ports take adapters, not competing components |
| Wrapping mature libraries | `pip install ultralytics` is the correct answer to "I need object detection". Borrow weights and servers; own algorithms and contracts |

---

## Intake: adding something new

```
0  GATE        check REGISTRY.json. Capability exists and no trigger? STOP
1  Rule of Three   would you rebuild it 3+ times in 3 years?
                   does no mature library own it?
                   can you state input -> output in one sentence?
                   any "no" -> STUDY or BORROW, do not EXTRACT
2  PRE-MORTEM  write down what bad search results will look like, before searching
3  Discover    timeboxed to 45 minutes per capability
4  Triage      kill criteria first (licence, coupling, no algorithm)
5  Decide      extract | borrow | study | DECLINE   <- DECLINE is a success
6  Mode        depend | vendor | reimplement, decided by licence
7  Build       to the Definition of Done above
8  Register    ledger entry, recipe membership, example
```

**Expect ~40% of discovery sessions to end in DECLINE or BORROW.** That is a
successful session: it saved you a maintenance obligation.

## Review cadence

| When | What |
|---|---|
| Per change | CI: stdlib-only job, packaging tests, lint, types |
| Per change | `pytest --cov` locally; floor 90%, currently 94%. Not in `addopts` because coverage tracing triples the run time and a slow suite is a suite nobody runs |
| Per new unit | Definition of Done, registry entry, cap check |
| Monthly | sort registry by `last_reviewed`, action the oldest three |
| Quarterly | archive any unit with empty `used_in_projects` after 6 months |

**Archiving will feel wrong.** A component you built, tested and documented but never
used is speculative inventory: it costs review time, CI time and attention and returns
nothing. Git keeps the history; what you remove is the obligation.
