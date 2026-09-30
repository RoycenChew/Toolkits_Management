# Governor Component

## What It Does

Wraps any `LLM` with a spend budget, a rate limit and a retry policy. Satisfies the
`LLM` port itself, so it composes with `CachedLLM` in either order.

## Why It Is Useful

Three concerns usually tangled together, each easy to get subtly wrong:

- **Budget is checked *before* the call.** Checking afterwards is the common mistake
  and it makes the limit advisory: the run blows past its ceiling and then reports
  it. The pre-flight check is pessimistic about output size, because a limit that
  assumes the best case is not a limit.
- **Rate limiting is a sliding window** over real call timestamps, not a fixed
  per-minute bucket. A fixed bucket permits a double-rate burst across the window
  boundary — precisely when providers start refusing.
- **Only transient failures are retried.** `RateLimited` and `AdapterError` are;
  `ValidationFailed` and a budget breach are not. Retrying a deterministic failure
  just burns the budget more slowly.

The clock and sleep function are injectable, so timing logic is tested without
spending real seconds.

## Original Source

Own work. The retry/backoff shape follows Temporal's `RetryPolicy`; the sliding
window and pre-flight budget check are not from any one upstream.

## Architecture

```
INPUT    messages + temperature + max_tokens
   |
BUDGET   project input tokens + assumed output; raise BudgetExceeded if over
   |
THROTTLE sliding 60s window; sleep only until the oldest call ages out
   |
CALL     wrapped LLM
   |  RateLimited  -> honour Retry-After, else exponential backoff + jitter
   |  AdapterError -> exponential backoff + jitter
   |  other        -> surface immediately
   |
ACCRUE   add real Usage to GovernorState
   |
OUTPUT   Completion
```

```
stdlib only  ->  GovernedLLM  ->  Completion (+ GovernorState)
```

## Installation

Copy the `governor/` directory. Python 3.10+.

## Dependencies

Standard library.

## Input Schema

`GovernorConfig`:

| Field | Default | Meaning |
|---|---|---|
| `max_total_tokens` | `None` | lifetime ceiling, input + output |
| `max_cost_usd` | `None` | lifetime cost ceiling |
| `max_calls` | `None` | lifetime call ceiling |
| `max_requests_per_minute` | `None` | sliding-window rate limit |
| `max_attempts` | 3 | total attempts per call |
| `initial_backoff` / `backoff_multiplier` / `max_backoff` / `jitter` | 0.5 / 2.0 / 30.0 / 0.1 | retry schedule |
| `estimate_output_tokens` | 512 | assumed output for the pre-flight check |

## Output Schema

`complete(...)` returns a `Completion`. Inspect `governor.state`:
`GovernorState(usage, calls, retries, throttled_seconds)`.

Raises `BudgetExceeded` (carries `.usage` at the point of refusal) or `AdapterError`
once attempts are exhausted.

## Usage

```python
from toolkit.adapters import LiteLLMClient
from toolkit.governor import BudgetExceeded, GovernedLLM, GovernorConfig

llm = GovernedLLM(
    LiteLLMClient("gpt-4o-mini"),
    GovernorConfig(max_total_tokens=200_000, max_cost_usd=5.0,
                   max_requests_per_minute=60),
)

try:
    answer = llm.complete([Message("user", "summarise this")])
except BudgetExceeded as exc:
    print("stopped at", exc.usage)

print(llm.state.calls, llm.state.retries, llm.state.usage.cost_usd)
```

## Limitations

- **`estimate_output_tokens` is a guess** (default 512). Too low and a run can
  overshoot `max_total_tokens` by roughly one call's output; too high and it refuses
  work it could afford. Pass `max_tokens` on each call to make the check exact.
- Input token projection uses `len(text)//4`, not a real tokenizer. It is a budget
  signal, not an invoice.
- `max_cost_usd` depends on the adapter reporting cost. `LiteLLMClient` forwards
  `response_cost` when the provider supplies it; otherwise cost stays 0.0 and that
  ceiling never triggers.
- **State is per-instance and in-memory.** Budgets are not shared across processes.
  For a cluster-wide budget you need a shared counter, which this does not attempt.
- The rate limiter is thread-safe but the window is local: eight processes with
  `max_requests_per_minute=60` will collectively issue 480.
- `max_cost_usd` and `max_calls` are checked against *spent* totals, so the call that
  crosses the line is allowed to complete. Only `max_total_tokens` projects forward.
- Retry sleeps block the calling thread. Combine with `bounded_map` for concurrency.

## Integration Guide

1. Put the cache **inside** the governor: `GovernedLLM(CachedLLM(real, cache))`. A
   cache hit then costs no budget and no rate-limit slot.
2. Pass `max_tokens` per call so the pre-flight budget check stops guessing.
3. Set `max_requests_per_minute` below your provider's published limit, not at it —
   the window is local and your own retries also consume slots.
4. Log `state.throttled_seconds` after a batch run. Large values mean the rate limit,
   not the model, is your bottleneck.
5. Treat `BudgetExceeded` as a stop signal, not an error to retry.

## Extraction Notes

- **Preserved:** Temporal's retry-policy shape (attempts, multiplier, cap, jitter).
- **Added:** the pre-flight budget projection, the sliding-window limiter, honouring
  a provider's `Retry-After` over the computed backoff, and injectable clock/sleep so
  the timing is testable.
- **Isolated:** no provider SDK, no server, no third-party package. Wraps anything
  satisfying the `LLM` port, including `ScriptedLLM`.
