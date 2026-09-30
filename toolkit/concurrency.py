"""Bounded parallel map with retry.

Threads rather than asyncio, because every backend in this toolkit is a
synchronous IO-bound call. Threads are the right tool for that and they do not
force the entire call stack above them to become async, which is the hidden cost
of an asyncio-first design.

Two properties that are easy to get wrong and are the reason this exists:

* **Results stay in input order.** `as_completed` returns whichever finished
  first; silently reordering a batch of embeddings against its texts is a
  corruption that no test notices until retrieval quality drops.
* **Failures are bounded and attributed.** A failing item does not cancel the
  batch by default; it is collected with its index so the caller can decide. The
  index is what makes a failure actionable.
"""
from __future__ import annotations

import random
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from .core.errors import AdapterError, RateLimited, ToolkitError

T = TypeVar("T")
R = TypeVar("R")


class MapFailed(ToolkitError):
    """Raised when `fail_fast` is set and at least one item failed."""

    def __init__(self, failures: Sequence[tuple[int, BaseException]]) -> None:
        self.failures = list(failures)
        first = failures[0]
        super().__init__(
            str(len(failures))
            + " item(s) failed; first at index "
            + str(first[0])
            + ": "
            + repr(first[1])
        )


@dataclass
class MapResult(Generic[R]):
    results: list[R | None]
    """One slot per input, in input order. `None` where that item failed."""
    failures: list[tuple[int, BaseException]] = field(default_factory=list)
    attempts: int = 0

    @property
    def ok(self) -> bool:
        return not self.failures

    def values(self) -> list[R]:
        """Successful results only, order preserved."""
        return [r for r in self.results if r is not None]


@dataclass
class MapConfig:
    max_workers: int = 8
    max_attempts: int = 1
    initial_backoff: float = 0.5
    backoff_multiplier: float = 2.0
    max_backoff: float = 20.0
    jitter: float = 0.1
    fail_fast: bool = False
    retry_on: tuple[type[BaseException], ...] = (RateLimited, AdapterError)
    """Only these are retried. A programming error should surface immediately
    rather than being attempted three times."""

    def __post_init__(self) -> None:
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")


def bounded_map(
    fn: Callable[[T], R],
    items: Iterable[T],
    config: MapConfig | None = None,
    sleep: Callable[[float], None] | None = None,
) -> MapResult[R]:
    """Apply `fn` across `items` with at most `max_workers` in flight.

    The worker count is the backpressure: the pool never holds more than
    `max_workers` futures doing work, so a 100,000-item input does not
    materialise 100,000 concurrent requests. Submitting everything at once to a
    bounded pool would queue the whole input in memory, which is the usual
    mistake in this pattern.
    """
    cfg = config or MapConfig()
    pause = sleep or time.sleep
    source = list(items)
    if not source:
        return MapResult(results=[], failures=[], attempts=0)

    results: list[R | None] = [None] * len(source)
    failures: list[tuple[int, BaseException]] = []
    total_attempts = 0

    def run(index: int) -> tuple[int, R | None, BaseException | None, int]:
        attempts = 0
        last: BaseException | None = None
        for attempt in range(1, cfg.max_attempts + 1):
            attempts += 1
            try:
                return index, fn(source[index]), None, attempts
            except cfg.retry_on as exc:
                last = exc
                if attempt >= cfg.max_attempts:
                    break
                delay = min(
                    cfg.initial_backoff * (cfg.backoff_multiplier ** (attempt - 1)),
                    cfg.max_backoff,
                )
                if isinstance(exc, RateLimited) and exc.retry_after:
                    delay = exc.retry_after
                if cfg.jitter:
                    delay *= 1.0 + random.uniform(-cfg.jitter, cfg.jitter)
                pause(max(0.0, delay))
            except BaseException as exc:  # noqa: BLE001 - not retryable, reported
                return index, None, exc, attempts
        return index, None, last, attempts

    with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
        for index, value, error, attempts in pool.map(run, range(len(source))):
            total_attempts += attempts
            if error is not None:
                failures.append((index, error))
            else:
                results[index] = value

    if failures and cfg.fail_fast:
        raise MapFailed(sorted(failures))

    return MapResult(results=results, failures=sorted(failures), attempts=total_attempts)


def batched(items: Sequence[T], size: int) -> list[Sequence[T]]:
    """Split into batches of at most `size`. Useful before `bounded_map` when the
    backend is itself batch-oriented, such as an embedding endpoint."""
    if size < 1:
        raise ValueError("size must be at least 1")
    return [items[i : i + size] for i in range(0, len(items), size)]


def embed_all(
    embedder: Any,
    texts: Sequence[str],
    batch_size: int = 32,
    config: MapConfig | None = None,
) -> list[list[float]]:
    """Embed a large list by batching, then mapping the batches in parallel.

    Flattens back in input order, which is the property that makes this safe to
    zip against the original texts.
    """
    if not texts:
        return []
    batches = batched(list(texts), batch_size)
    outcome = bounded_map(
        lambda batch: embedder.embed(list(batch)),
        batches,
        config or MapConfig(fail_fast=True),
    )
    vectors: list[list[float]] = []
    for batch_vectors in outcome.results:
        if batch_vectors is None:
            raise AdapterError("an embedding batch failed; see MapResult.failures")
        vectors.extend([float(v) for v in vector] for vector in batch_vectors)
    if len(vectors) != len(texts):
        raise AdapterError(
            "embedded " + str(len(vectors)) + " vectors for " + str(len(texts)) + " texts"
        )
    return vectors


__all__ = [
    "MapConfig",
    "MapFailed",
    "MapResult",
    "batched",
    "bounded_map",
    "embed_all",
]
