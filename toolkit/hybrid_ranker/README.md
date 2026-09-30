# Hybrid Ranker Component

## What It Does

Merges several independently-ranked result lists into one ranking, then optionally
reranks the head of that list with an expensive model and diversifies the output.

## Why It Is Useful

Every retrieval system ends up with more than one retriever — keyword and vector
at minimum, often sparse-expansion and metadata filters too. The scores are not
comparable: BM25 returns 14.2, cosine returns 0.81. Reciprocal Rank Fusion sidesteps
the problem entirely by using ranks instead of scores, which is why it needs no
tuning and no calibration. It is the single highest quality-per-line change you can
make to a RAG pipeline.

The cascade is the second half: cheap retrieval casts a wide net for recall, then a
cross-encoder or late-interaction scorer is spent only on a bounded candidate set.

## Original Source

Not a code extraction. Reimplemented from the published algorithms:

- Reciprocal Rank Fusion — Cormack, Clarke & Buettcher, SIGIR 2009.
- Relative-score and distribution-based fusion — as documented by
  [Qdrant](https://qdrant.tech/documentation/advanced-tutorials/reranking-hybrid-search/)
  and Weaviate.
- Retrieve-then-rerank cascade — the pattern behind
  [ColBERT/PLAID](https://github.com/stanford-futuredata/ColBERT) and
  [SPLADE](https://github.com/naver/splade) production pipelines.
- Maximal Marginal Relevance — Carbonell & Goldstein, SIGIR 1998.

No upstream code was copied, so no upstream licence applies.

## Architecture

```
INPUT    list[RankedList]  (one per retriever, best-first, each with a weight)
   |
CONTRIB  RRF: 1/(k+rank)    |  RELATIVE_SCORE: min-max  |  DISTRIBUTION: z-score
   |
FUSE     weighted sum per id, keeping per-source contributions and ranks
   |
CASCADE  top `rerank_budget` -> Reranker.score() -> blend by `rerank_weight`
   |
MMR      greedy relevance-minus-redundancy selection (needs vectors)
   |
OUTPUT   FusionResult(items=top_k FusedItem, ...)
```

```
numpy? no.  models? no.  network? no.  ->  Component  ->  ranked results
```

## Installation

Copy the `hybrid_ranker/` directory into your project. Python 3.10+. No install step.

## Dependencies

Standard library only. A reranker, if you use one, is yours to supply and brings
its own dependencies.

## Input Schema

| Field | Type | Meaning |
|---|---|---|
| `query` | `str` | Passed through to the reranker; unused by fusion |
| `ranked_lists` | `Sequence[RankedList]` | `source`, `items` (best-first), `weight` |
| `config` | `FusionConfig` | method, `rrf_k`, `rerank_budget`, `rerank_weight`, `top_k`, `mmr_lambda` |
| `vectors` | `Mapping[str, Sequence[float]] \| None` | Required only when `mmr_lambda < 1` |

## Output Schema

`FusionResult`: `items` (`FusedItem`: `id`, `score`, `payload`, `contributions`,
`ranks`), `method`, `candidates_considered`, `reranked`.

`contributions` and `ranks` are the debugging surface — they tell you which
retriever is carrying a result and which is dead weight.

## Usage

```python
from hybrid_ranker import (
    FusionConfig, FusionRequest, HybridRankerComponent, RankedItem, RankedList,
)

bm25 = RankedList("bm25", [RankedItem("doc7", 14.2), RankedItem("doc2", 9.1)])
dense = RankedList("dense", [RankedItem("doc2", 0.91), RankedItem("doc7", 0.80)])

result = HybridRankerComponent().execute(
    FusionRequest(query="refund policy", ranked_lists=[bm25, dense],
                  config=FusionConfig(top_k=10))
)
for item in result.items:
    print(item.id, round(item.score, 4), item.ranks)
```

With a cascade:

```python
class CrossEncoderReranker:
    def __init__(self, model): self.model = model
    def score(self, query, items):
        pairs = [(query, i.payload["text"]) for i in items]
        return self.model.predict(pairs)          # one score per item, same order

result = HybridRankerComponent().execute(
    FusionRequest(query, lists, FusionConfig(top_k=10, rerank_budget=50)),
    reranker=CrossEncoderReranker(model),
)
```

## Limitations

- Fusion is pure ordering arithmetic. It cannot recover a document no retriever
  returned — fix recall upstream.
- `rrf_k=60` is the literature default, not a tuned value. Larger `k` flattens the
  weight of top ranks; tune it on your own labelled queries if you have them.
- `RELATIVE_SCORE` and `DISTRIBUTION` assume each list's scores are internally
  meaningful. If a retriever returns near-constant scores they contribute nothing.
- MMR is O(n²·d) in the candidate pool. Fine for a top-100 pool, not for 10,000.
- No async. Wrap the reranker call yourself if it needs concurrency.

## Integration Guide

1. Have each retriever return `RankedList(source=..., items=[RankedItem(id, score, payload)])`.
   Use the *same* id space across retrievers, or nothing will fuse.
2. Start with RRF, equal weights, `rerank_budget=0`. Measure.
3. Add `rerank_budget` at 3–5× your `top_k`. Measure again — this is usually the
   largest single gain.
4. Only then tune weights, and only against labelled queries.
5. Set `mmr_lambda` to about 0.7 if results are repetitive; leave it at 1.0 otherwise.

## Extraction Notes

- **Preserved:** the RRF, normalisation, cascade blending and MMR algorithms.
- **Removed:** nothing — this was written from the algorithms, not lifted.
- **Rewritten:** per-source attribution (`contributions`, `ranks`) is an addition;
  most framework implementations discard it, and it is what makes tuning possible.
- **Isolated:** the reranker is a `Protocol`, so no model, framework or vector
  store is imported anywhere in the component.
