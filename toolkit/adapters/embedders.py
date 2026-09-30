"""Embedder adapters.

`HashingEmbedder` exists for a reason beyond testing: it is a real, if weak,
retrieval signal that needs no model, no download and no network. That makes the
whole pipeline runnable on conference wifi, and it means the contract tests can
assert behaviour rather than skipping.

`FastEmbedEmbedder` is the production default because it runs ONNX on CPU and
does not drag in torch, which matters when the demo machine is a laptop.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Any

from ..core.errors import AdapterError, MissingDependency

_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Deterministic hashed bag-of-n-grams, L2-normalised.

    The classic hashing trick: map each token and character trigram into one of
    `dimension` buckets by hash, accumulate, then normalise. It has no semantic
    understanding at all — it cannot match 'car' to 'automobile' — but it is
    stable across runs and processes, needs nothing installed, and gives the
    cosine metric something meaningful to work with for lexically similar text.

    Use it as the offline default and as the fixture in tests. Do not ship it as
    your only retriever.
    """

    def __init__(self, dimension: int = 256, use_trigrams: bool = True) -> None:
        if dimension < 8:
            raise ValueError("dimension must be at least 8")
        self._dimension = dimension
        self._use_trigrams = use_trigrams

    @property
    def dimension(self) -> int:
        return self._dimension

    def _bucket(self, feature: str) -> int:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self._dimension

    def _features(self, text: str) -> list[str]:
        tokens = _TOKEN.findall(text.lower())
        features = list(tokens)
        if self._use_trigrams:
            for token in tokens:
                padded = "^" + token + "$"
                features.extend(
                    padded[i : i + 3] for i in range(max(0, len(padded) - 2))
                )
        return features

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self._dimension
            for feature in self._features(text or ""):
                # Sign from a second hash bit keeps unrelated collisions from
                # always adding constructively.
                bucket = self._bucket(feature)
                sign = 1.0 if self._bucket("s|" + feature) % 2 == 0 else -1.0
                vector[bucket] += sign
            norm = math.sqrt(sum(v * v for v in vector))
            out.append([v / norm for v in vector] if norm > 0 else vector)
        return out


class FastEmbedEmbedder:
    """FastEmbed (ONNX, CPU). Real semantics without a torch install."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        model: Any | None = None,
        batch_size: int = 64,
    ) -> None:
        self._model_name = model_name
        self._model = model
        self._batch_size = batch_size
        self._dimension: int | None = None

    def _ensure(self) -> Any:
        if self._model is None:
            try:
                from fastembed import TextEmbedding  # type: ignore
            except ImportError as exc:
                raise MissingDependency("fastembed", "embed") from exc
            self._model = TextEmbedding(model_name=self._model_name)
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            # The only reliable way to learn the dimension across fastembed
            # versions is to embed something trivial once.
            self._dimension = len(self.embed(["dimension probe"])[0])
        return self._dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure()
        try:
            vectors = [
                [float(v) for v in vector]
                for vector in model.embed(list(texts), batch_size=self._batch_size)
            ]
        except TypeError:
            vectors = [[float(v) for v in vector] for vector in model.embed(list(texts))]
        except Exception as exc:  # noqa: BLE001 - adapter boundary
            raise AdapterError("fastembed failed to embed a batch") from exc
        if len(vectors) != len(texts):
            raise AdapterError(
                "fastembed returned "
                + str(len(vectors))
                + " vectors for "
                + str(len(texts))
                + " inputs"
            )
        self._dimension = len(vectors[0])
        return vectors


__all__ = ["FastEmbedEmbedder", "HashingEmbedder"]
