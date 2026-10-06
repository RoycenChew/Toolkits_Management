# Entity Resolution Component

**Layer 2 · imports nothing · `copy_tier: standalone`**

## What It Does

Takes a set of records with no shared identifier and returns clusters of records
that refer to the same real-world entity, plus an interpretable match weight and a
per-field explanation for every pair it scored.

And then `score_record`, for the question an application asks at runtime: one new
record against a list of candidates, using weights that already exist.

No labels required. No SQL engine. No model download.

## Why It Is Useful

"Merge these two messy datasets that share no key" is a problem that appears in
almost every data project and is consistently underestimated. Doing it properly
needs three separate pieces of machinery that are each easy to get wrong:

1. **Blocking** — you cannot compare all n(n−1)/2 pairs, so you need cheap keys
   that put probable duplicates in the same bucket. Choosing a *good set* of keys
   is a weighted set-cover problem, not a matter of taste.
2. **Scoring** — you need P(match) from field agreements, without labelled data.
   Fellegi–Sunter with EM does this and, crucially, produces a match weight in bits
   that a human can audit.
3. **Clustering** — pairwise scores are not entities. Naive transitive closure
   chains unrelated records together through a chain of weak links.

## Original Source

Reimplemented from the algorithms behind these projects. No code copied.

| Piece | Prior art | Licence of the original |
|---|---|---|
| Affine-gap string distance | [dedupe](https://github.com/dedupeio/dedupe) / its `affinegap` extension; Gotoh 1982 | MIT |
| Greedy blocking-predicate cover | [dedupe](https://github.com/dedupeio/dedupe) `training.py` | MIT |
| Fellegi–Sunter + EM estimation | [Splink](https://github.com/moj-analytical-services/splink); Fellegi & Sunter 1969 | MIT |

## Architecture

```
INPUT     {record_id: {field: value}}  +  FieldComparison per field
   |
BLOCKING  predicate library -> greedy weighted set cover -> selected predicates
   |      (labelled duplicate pairs, if supplied, are what the cover must cover)
   |
CANDIDATES pairs sharing at least one block key, oversized blocks discarded
   |
COMPARE   per field: similarity -> discrete ComparisonLevel -> pattern
   |
EM        estimate lambda, m[field][level], u[field][level] from the patterns
   |
SCORE     match_weight = log2(lambda/(1-lambda)) + sum log2(m/u)  -> probability
   |
CLUSTER   average linkage (default) or connected components, at match_threshold
   |
OUTPUT    EntityCluster list + ScoredPair list + TrainedModel
                                                     |
                                      keep it ───────┘
                                                     |
                                                     v
          score_record(record, candidates, config, model) -> ScoredPair list
          same weight arithmetic, no EM, no blocking, no batch
```

```
stdlib only  ->  Component  ->  clusters, scored pairs, learned model
```

## Installation

Copy the `entity_resolution/` directory into your project. Python 3.10+.

## Dependencies

Standard library only. No numpy, no scipy, no database.

## Input Schema

| Field | Type | Meaning |
|---|---|---|
| `records` | `Mapping[str, Mapping[str, Any]]` | id -> record |
| `config.comparisons` | `Sequence[FieldComparison]` | which fields to compare, at what thresholds, with which comparator |
| `config.predicates` | `Sequence[Predicate] \| None` | `None` uses the built-in library |
| `config.max_block_size` | `int` | blocks above this are dropped as degenerate |
| `config.max_predicates` | `int` | cap on the greedy cover |
| `config.match_threshold` | `float` | minimum probability to link |
| `config.cluster_link` | `"average"` \| `"connected"` | linkage strategy |
| `labelled_pairs` | `Sequence[tuple[str, str, bool]]` | optional, trains blocking only |

## Output Schema

- `clusters`: `EntityCluster(cluster_id, record_ids, cohesion)`, largest first.
- `score_record` returns the same `ScoredPair` type, best first.
- `scored_pairs`: `ScoredPair(left, right, match_probability, match_weight, pattern)`.
- `model`: `TrainedModel(lambda_prior, m_probabilities, u_probabilities, iterations, converged)`.
- `selected_predicates`, `pairs_compared`, `pairs_avoided`.

`match_weight` is a log2 Bayes factor: +4 means the evidence makes a match 16× more
likely than chance. `pattern` names the level each field reached, which is the audit
trail — you can point at the field that decided any given pair.

## Usage

```python
from entity_resolution import (
    ComparisonLevel, EntityResolutionComponent, FieldComparison,
    ResolutionConfig, ResolutionRequest,
)

records = {
    "1": {"name": "Robert Smith",   "city": "London", "postcode": "SW1A 1AA"},
    "2": {"name": "Robert J Smith", "city": "London", "postcode": "SW1A 1AA"},
    "3": {"name": "Alice Nakamura", "city": "Osaka",  "postcode": "530-0001"},
}

config = ResolutionConfig(
    comparisons=[
        FieldComparison("name"),                                   # affine-gap default
        FieldComparison("city", levels=(ComparisonLevel("exact", 1.0),
                                        ComparisonLevel("close", 0.85))),
        FieldComparison("postcode", levels=(ComparisonLevel("exact", 1.0),)),
    ],
    match_threshold=0.9,
)

result = EntityResolutionComponent().execute(ResolutionRequest(records, config))

for cluster in result.clusters:
    print(cluster.cluster_id, cluster.record_ids, round(cluster.cohesion, 3))
for pair in result.scored_pairs[:5]:
    print(pair.left, pair.right, round(pair.match_weight, 2), pair.pattern)
```

Custom comparator (use whatever similarity your domain needs):

```python
FieldComparison("phone", comparator=lambda a, b: 1.0 if a[-7:] == b[-7:] else 0.0)
```

## Scoring one record

`execute` resolves a batch, and its EM step is what makes it a batch operation: m
and u probabilities are estimated from the distribution of comparison patterns
across many pairs. EM over a batch of one has nothing to estimate from — it would
return the seed parameters while looking like a measurement.

So the model is separated from the scoring. Train once, keep the `TrainedModel`,
score single records against candidates for as long as it holds:

```python
from entity_resolution import EntityResolutionComponent, default_model

component = EntityResolutionComponent()
model = component.execute(ResolutionRequest(records=corpus, config=config)).model

matches = component.score_record(incoming, candidates, config, model, record_id="new")
best = matches[0]
print(best.right, round(best.match_weight, 1), dict(best.pattern))
```

The weight arithmetic is the same function the batch path uses — a test asserts
that a pair scored this way is bit-for-bit what `execute` gave it — so a cached
model cannot quietly come to mean something else than the run that produced it.

Points worth knowing:

- **No blocking.** The caller chooses the candidates, which is the point: at
  runtime they usually come from a database query you already have. Passing the
  whole corpus works and is O(n) comparisons; the record is never scored against
  itself.
- **`default_model(config, match_rate=0.01)`** gives fixed weights when there is
  nothing to train on yet — the same seed EM starts from, most of the m mass on the
  strongest level and most of the u mass on no-match. It reports `iterations=0` and
  `converged=False`, because those parameters were asserted rather than measured.
- **A model missing one of the configured fields raises.** `_score` falls back to
  epsilon for an unknown field, and epsilon over epsilon contributes exactly zero
  bits — so adding a comparison and forgetting to retrain would score as if the new
  field did not exist, and look like it worked.

## Comparing numbers and dates

The default comparator is normalised affine-gap distance, which is right for a name
and the wrong *question* for a number or a date. `100` and `1000` share three
characters and score high; `2026-01-31` and `2026-02-01` are one day apart and look
nothing alike.

```python
from entity_resolution import (
    ComparisonLevel, FieldComparison, date_comparator, numeric_comparator,
    numeric_similarity,
)

FieldComparison("amount", comparator=numeric_similarity,
                levels=(ComparisonLevel("exact", 1.0), ComparisonLevel("close", 0.98)))
FieldComparison("invoiced_on", comparator=date_comparator(window_days=7),
                levels=(ComparisonLevel("same_day", 1.0),
                        ComparisonLevel("within_3_days", 0.57)))
```

- `numeric_similarity` is **relative** difference: `1 - |a - b| / max(|a|, |b|)`. A
  difference of 10 is nothing on a million and everything on a dozen, and one
  threshold cannot serve both scales otherwise. Opposite signs score 0.0 however
  close the magnitudes — +500 and -500 are a credit and a debit, and calling them
  similar is how a reconciliation pairs the wrong rows.
- `numeric_comparator(scale=...)` switches to an **absolute** scale, which is the
  right choice for a quantity that legitimately passes through zero, where a
  relative difference is undefined.
- `date_comparator(window_days=...)` decays **linearly** over the window, so a
  `ComparisonLevel` threshold reads back as a number of days: at a 7-day window,
  0.57 is "within three days", which someone can check. An exponential decay would
  put a number nobody can picture on every level, and a level nobody can picture is
  one nobody will tune. The window is per field because fields differ: a date of
  birth three days out is a transcription error, an invoice date three days out is a
  different invoice.
- `days_apart(a, b)` is exposed separately, because it is usually the number a
  human wants in a report and "similarity 0.97" is not.
- Both return **0.0** for a value they cannot read, rather than raising. A missing or
  junk field is the normal case in this data, and a comparator that raises takes the
  batch down with it. Fellegi-Sunter already means "we learned nothing here" by
  `NO_MATCH_LEVEL`.

## Limitations

- **Conditional independence** between fields is assumed by Fellegi–Sunter. Highly
  correlated fields (city and postcode) double-count their evidence and inflate the
  weight. Either merge them into one comparison, or read the weights as ordering
  rather than calibrated probability.
- EM is unsupervised and can settle in a poor local optimum on small or
  low-signal datasets. Check `model.converged` and eyeball the m/u tables.
- Average linkage is O(k³)-ish in the number of records within a connected region.
  Fine up to a few thousand records per region; beyond that, use `"connected"` or
  partition first.
- Blocking is in-memory. For millions of records, push the blocking keys into SQL
  or Spark and feed only the candidate pairs to the scoring half.
- `labelled_pairs` trains the *blocking* cover only. Matching stays unsupervised
  by design.
- Non-Latin scripts work (the comparators are character-based), but affine-gap
  similarity is less meaningful for logographic text; supply a custom comparator.
- **`score_record` does no blocking**, so the candidate list is the caller's
  problem and its quality bounds the result: a correct match that was never a
  candidate cannot be found, and nothing here will say so.
- **A kept model goes stale silently.** m and u describe the data they were
  estimated from, and nothing detects drift. Retrain on a schedule, and treat
  `default_model` weights as a bootstrap rather than a destination.
- **The date comparator refuses an ambiguous numeric date.** `03/10/2026` has no
  document-level hint to resolve it in a record, unlike in `extraction`, so it
  parses as nothing rather than as a guess - which scores it 0.0, not an error.

## Integration Guide

1. Normalise before you resolve: casefold, strip punctuation, expand known
   abbreviations. The component lowercases and trims, nothing more — domain
   normalisation is yours and it matters more than any parameter here.
2. Start with the default predicate library. Check `pairs_compared` against
   `pairs_avoided`; if you compared almost everything, your keys are too loose.
3. Label 20–50 known duplicate pairs and pass them as `labelled_pairs`. This
   usually cuts the candidate count substantially at the same recall.
4. Read `scored_pairs` near your threshold before trusting it. The `pattern` field
   tells you exactly why each borderline pair landed where it did.
5. Keep `model` if you want stable behaviour across runs — re-running EM on a
   different data slice will give slightly different parameters.

## Extraction Notes

- **Preserved:** the affine-gap recurrence (Gotoh), the greedy weighted set-cover
  selection, the Fellegi–Sunter likelihood and its EM update, average-linkage
  agglomeration.
- **Removed:** dedupe's console active-learning UI, its settings pickling and
  `Gazetteer`/`StaticDedupe` persistence classes; Splink's entire SQL backend
  abstraction, its DuckDB/Spark dialect layer and its charting.
- **Rewritten:** affine-gap distance in pure Python with O(min) space instead of a
  C extension; the labelling interface reduced to a plain `labelled_pairs`
  argument, so it can be driven by a script or an LLM rather than a terminal
  prompt; comparison levels made an explicit first-class type.
- **Isolated:** no database, no dataframe library, no compiled extension. The
  three stages (blocking, scoring, clustering) are separate methods and can be
  used independently if you already have one of them.
