"""Reranker adapters.

A reranker re-scores a small candidate set with a signal the first stage could
not afford. That is the whole idea of the cascade: cheap retrieval casts a wide
net, and something expensive is spent only on the head of it.

Three implementations, because the trade between them is real and depends on what
you have installed:

* `LexicalOverlapReranker` — stdlib. Scores BM25 over *the candidate set itself*,
  so its IDF comes from the candidates rather than the whole corpus. That is a
  genuinely different signal from a global-IDF first stage, and it needs nothing.
* `CrossEncoderReranker` — the real thing. A cross-encoder reads the query and
  document together, which is why it beats any bi-encoder and why it cannot be
  used for first-stage retrieval.
* `LLMReranker` — when you have a model but no reranker weights. Judges the whole
  candidate set in one call, which keeps it affordable.

All three satisfy the same `Reranker` protocol and are interchangeable, which the
contract tests enforce.
"""
from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any

from ..core.errors import AdapterError, MissingDependency
from ..core.models import Message

_TOKEN = re.compile(r"[a-z0-9]+")
_SCORE_LINE = re.compile(r"(\d{1,3})\s*[:.)\-]\s*(\d{1,3}(?:\.\d+)?)")


def _text_of(item: Any) -> str:
    """Read the document text off whatever shape the caller passed.

    `hybrid_ranker` yields `FusedItem` with the text in `payload`; a caller
    wiring this up by hand is more likely to pass chunks or plain strings.
    Accepting all three costs six lines and removes a whole class of confusing
    empty-result bugs.
    """
    if isinstance(item, str):
        return item
    payload = getattr(item, "payload", None)
    if isinstance(payload, dict):
        for key in ("text", "content", "chunk"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
    for attribute in ("text", "content"):
        value = getattr(item, attribute, None)
        if isinstance(value, str) and value:
            return value
    return ""


def _stem(token: str) -> str:
    """Conservative suffix stripping.

    Without it a lexical reranker scores zero on "refund" against a document
    saying "refunds", which is not an edge case — it is most real queries. A full
    Porter stemmer would be more correct and is a dependency; these four rules
    recover the overwhelming majority of the benefit in ten lines. Length guards
    keep short words ("is", "as", "does") from being mangled into noise.
    """
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokens(text: str) -> list[str]:
    return [_stem(token) for token in _TOKEN.findall(text.lower())]


class LexicalOverlapReranker:
    """BM25 scored over the candidate set, with no dependencies.

    Because IDF is computed from the candidates rather than the corpus, a term
    that is common overall but rare *among these candidates* is weighted highly —
    which is exactly the discrimination a reranking stage should add. It will not
    match a cross-encoder, but it is free, deterministic, and runs offline.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b

    def score(self, query: str, items: Sequence[Any]) -> list[float]:
        if not items:
            return []
        documents = [_tokens(_text_of(item)) for item in items]
        lengths = [len(d) for d in documents]
        average = (sum(lengths) / len(lengths)) if lengths else 0.0
        terms = set(_tokens(query))
        if not terms or average <= 0:
            return [0.0] * len(items)

        document_frequency = {
            term: sum(1 for doc in documents if term in doc) for term in terms
        }
        total = len(documents)

        scores: list[float] = []
        for doc, length in zip(documents, lengths):
            score = 0.0
            for term in terms:
                frequency = doc.count(term)
                if not frequency:
                    continue
                df = document_frequency[term]
                # Standard BM25 IDF with the +0.5 smoothing that keeps a term
                # present in every candidate from going negative.
                idf = math.log(1.0 + (total - df + 0.5) / (df + 0.5))
                denominator = frequency + self._k1 * (
                    1.0 - self._b + self._b * (length / average)
                )
                score += idf * (frequency * (self._k1 + 1.0)) / denominator
            scores.append(score)
        return scores


class CrossEncoderReranker:
    """A sentence-transformers cross-encoder. The accurate option.

    Reads query and document jointly, so it cannot be precomputed and cannot be
    used for first-stage retrieval — which is precisely why it belongs behind a
    budget in a cascade.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        model: Any | None = None,
        batch_size: int = 32,
    ) -> None:
        self._model_name = model_name
        self._model = model
        self._batch_size = batch_size

    def _ensure(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder  # type: ignore
            except ImportError as exc:
                raise MissingDependency("sentence-transformers", "rerank") from exc
            self._model = CrossEncoder(self._model_name)
        return self._model

    def score(self, query: str, items: Sequence[Any]) -> list[float]:
        if not items:
            return []
        model = self._ensure()
        pairs = [(query, _text_of(item)) for item in items]
        try:
            raw = model.predict(pairs, batch_size=self._batch_size)
        except Exception as exc:  # noqa: BLE001 - adapter boundary
            raise AdapterError("cross-encoder scoring failed") from exc
        scores = [float(value) for value in raw]
        if len(scores) != len(items):
            raise AdapterError(
                "cross-encoder returned "
                + str(len(scores))
                + " scores for "
                + str(len(items))
                + " items"
            )
        return scores


class LLMReranker:
    """Judge relevance with a model you already have.

    Scores the whole candidate set in one call rather than one call per document:
    a per-document loop is a rate-limit and latency disaster at any useful budget,
    and the model judges relative relevance better when it can see the
    alternatives anyway.

    Any candidate the model fails to score keeps `default_score`, so a partial or
    malformed reply degrades the ranking instead of destroying it.
    """

    def __init__(
        self,
        llm: Any,
        snippet_chars: int = 500,
        default_score: float = 0.0,
        max_tokens: int | None = 400,
    ) -> None:
        if llm is None:
            raise ValueError("LLMReranker requires an LLM")
        self._llm = llm
        self._snippet_chars = snippet_chars
        self._default = default_score
        self._max_tokens = max_tokens

    def score(self, query: str, items: Sequence[Any]) -> list[float]:
        if not items:
            return []
        listing = "\n\n".join(
            "[" + str(index) + "] " + _text_of(item)[: self._snippet_chars]
            for index, item in enumerate(items, start=1)
        )
        messages = [
            Message(
                "system",
                "You rate how well each passage answers a question. Reply with one"
                " line per passage in the form 'number: score', where score is 0"
                " to 10. No other text.",
            ),
            Message("user", "Question: " + query + "\n\nPassages:\n" + listing),
        ]
        completion = self._llm.complete(messages, 0.0, self._max_tokens)

        scores = [self._default] * len(items)
        for raw_index, raw_score in _SCORE_LINE.findall(completion.text or ""):
            index = int(raw_index) - 1
            if 0 <= index < len(items):
                scores[index] = max(0.0, min(10.0, float(raw_score)))
        return scores


__all__ = ["CrossEncoderReranker", "LLMReranker", "LexicalOverlapReranker"]
