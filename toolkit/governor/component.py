"""Cost budget, rate limiting and retry around an LLM port.

Three separate concerns that are usually tangled together and all three are easy
to get subtly wrong:

* **Budget.** Enforced *before* the call, not after, so a run cannot blow past
  its ceiling and then report it. Checking afterwards is the common mistake and
  it makes the limit advisory rather than real.
* **Rate limiting.** A sliding window over actual call timestamps, not a fixed
  bucket. A fixed per-minute bucket permits a double-rate burst across the
  boundary, which is exactly when providers start refusing.
* **Retry.** Only `RateLimited` and transient adapter errors are retried.
  `ValidationFailed` and a budget breach are not: retrying a deterministic
  failure just burns the budget more slowly.

The clock and the sleep function are injectable, so the tests verify the timing
logic without spending real seconds on it.
"""
from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from ..core.errors import AdapterError, RateLimited, ToolkitError
from ..core.models import Completion, Message, Usage


class BudgetExceeded(ToolkitError):
    """The configured token or cost ceiling would be breached by this call."""

    def __init__(self, message: str, usage: Usage) -> None:
        self.usage = usage
        super().__init__(message)


@dataclass
class GovernorConfig:
    max_total_tokens: int | None = None
    """Ceiling across the governor's lifetime, input plus output."""
    max_cost_usd: float | None = None
    max_calls: int | None = None
    max_requests_per_minute: float | None = None
    max_attempts: int = 3
    initial_backoff: float = 0.5
    backoff_multiplier: float = 2.0
    max_backoff: float = 30.0
    jitter: float = 0.1
    estimate_output_tokens: int = 512
    """Assumed output size when pre-checking the budget, since the real figure is
    unknown until the call returns. Pre-flight checks have to be pessimistic or
    they are not limits."""

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.max_requests_per_minute is not None and self.max_requests_per_minute <= 0:
            raise ValueError("max_requests_per_minute must be positive")


@dataclass
class GovernorState:
    usage: Usage = field(default_factory=Usage)
    calls: int = 0
    retries: int = 0
    throttled_seconds: float = 0.0


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class GovernedLLM:
    """Wraps any LLM with a budget, a rate limit and a retry policy.

    Satisfies the `LLM` port itself, so it composes with `CachedLLM` in either
    order. Putting the cache *inside* the governor (governor wrapping cache) is
    usually right: a cache hit then costs no budget and no rate-limit slot, which
    is the behaviour you want.
    """

    def __init__(
        self,
        llm: object,
        config: GovernorConfig | None = None,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._llm = llm
        self._config = config or GovernorConfig()
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep
        self._lock = threading.Lock()
        self._timestamps: list[float] = []
        self.state = GovernorState()

    @property
    def config(self) -> GovernorConfig:
        return self._config

    def complete(
        self,
        messages: Sequence[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Completion:
        projected_in = sum(_estimate_tokens(m.content) for m in messages)
        projected_out = max_tokens or self._config.estimate_output_tokens
        self._check_budget(projected_in + projected_out)
        self._throttle()

        cfg = self._config
        last_error: Exception | None = None
        for attempt in range(1, cfg.max_attempts + 1):
            try:
                completion = self._llm.complete(messages, temperature, max_tokens)  # type: ignore[attr-defined]
            except RateLimited as exc:
                last_error = exc
                if attempt >= cfg.max_attempts:
                    break
                self.state.retries += 1
                # Honour the provider's own Retry-After when it gives one; it
                # knows its window better than an exponential guess does.
                delay = exc.retry_after if exc.retry_after else self._backoff(attempt)
                self._sleep(delay)
                continue
            except AdapterError as exc:
                last_error = exc
                if attempt >= cfg.max_attempts:
                    break
                self.state.retries += 1
                self._sleep(self._backoff(attempt))
                continue

            with self._lock:
                self.state.usage = self.state.usage + completion.usage
                self.state.calls += 1
            return completion

        raise AdapterError(
            "giving up after " + str(cfg.max_attempts) + " attempts: " + str(last_error)
        )

    # --- budget -----------------------------------------------------------

    def _check_budget(self, projected_tokens: int) -> None:
        cfg = self._config
        with self._lock:
            state = self.state
            if cfg.max_calls is not None and state.calls >= cfg.max_calls:
                raise BudgetExceeded(
                    "call limit of " + str(cfg.max_calls) + " reached", state.usage
                )
            if cfg.max_total_tokens is not None:
                spent = state.usage.input_tokens + state.usage.output_tokens
                if spent + projected_tokens > cfg.max_total_tokens:
                    raise BudgetExceeded(
                        "token budget of "
                        + str(cfg.max_total_tokens)
                        + " would be exceeded (spent "
                        + str(spent)
                        + ", this call needs about "
                        + str(projected_tokens)
                        + ")",
                        state.usage,
                    )
            if cfg.max_cost_usd is not None and state.usage.cost_usd >= cfg.max_cost_usd:
                raise BudgetExceeded(
                    "cost budget of $"
                    + format(cfg.max_cost_usd, ".4f")
                    + " reached (spent $"
                    + format(state.usage.cost_usd, ".4f")
                    + ")",
                    state.usage,
                )

    # --- rate limiting ----------------------------------------------------

    def _throttle(self) -> None:
        """Sliding-window rate limit.

        Keeps the timestamps of calls inside the last 60 seconds and sleeps only
        until the oldest of them ages out. A fixed bucket would allow twice the
        configured rate across a window boundary.
        """
        limit = self._config.max_requests_per_minute
        if limit is None:
            return
        while True:
            with self._lock:
                now = self._clock()
                self._timestamps = [t for t in self._timestamps if now - t < 60.0]
                if len(self._timestamps) < limit:
                    self._timestamps.append(now)
                    return
                wait = 60.0 - (now - self._timestamps[0])
            if wait <= 0:
                continue
            self.state.throttled_seconds += wait
            self._sleep(wait)

    def _backoff(self, attempt: int) -> float:
        cfg = self._config
        delay = min(
            cfg.initial_backoff * (cfg.backoff_multiplier ** (attempt - 1)),
            cfg.max_backoff,
        )
        if cfg.jitter:
            delay *= 1.0 + random.uniform(-cfg.jitter, cfg.jitter)
        return max(0.0, delay)


__all__ = ["BudgetExceeded", "GovernedLLM", "GovernorConfig", "GovernorState"]
