# Durable Steps Component

**Layer 1 · imports nothing · `copy_tier: standalone`**

## What It Does

Runs a sequence of steps so that each completed step is checkpointed. Re-invoking
with the same `run_id` replays completed steps from storage instead of re-running
them, and resumes at the first step that has not finished. Includes retries with
exponential backoff and jitter, a non-retryable error class, and a lease so two
workers cannot process the same run concurrently.

## Why It Is Useful

Any multi-step pipeline that touches the network — an agent loop, a document
ingestion job, an ETL run, a multi-call LLM chain — will die partway through. Without
checkpoints you restart from zero, which wastes the expensive steps and, worse,
repeats the side-effecting ones. This is the durability primitive that Temporal,
DBOS and Hatchet each wrap in their own runtime; here it is about 300 lines over a
SQL table, with no server to run.

The lease is the part most homegrown versions omit, and it is what makes resumption
safe once more than one worker exists.

## Original Source

Reimplemented from the durable-execution pattern. No code copied.

| Piece | Prior art | Licence of the original |
|---|---|---|
| Durable step + checkpoint table | [DBOS Transact](https://github.com/dbos-inc/dbos-transact-py) | MIT |
| Lease-based run claiming | [Hatchet](https://github.com/hatchet-dev/hatchet) Postgres queue | MIT (verify current terms) |
| Replay semantics, retry policy shape | [Temporal](https://github.com/temporalio) | MIT |

## Architecture

```
INPUT     WorkflowRequest(run_id, steps, initial_context, lease_seconds)
   |
LEASE     acquire_lease(run_id, owner) -> False means another worker owns it
   |
LOAD      load_steps(run_id) -> existing checkpoints
   |
per step:
   |  checkpoint exists & COMPLETED   -> replay result into context, skip
   |  non-idempotent & interrupted    -> raise StepNotReplayable
   |  otherwise                       -> run, retry on failure, checkpoint on success
   |
RELEASE   release_lease(run_id, COMPLETED | FAILED)
   |
OUTPUT    WorkflowResult(status, context, executed, replayed, failed_step, error)
```

```
sqlite3 (stdlib) or your own CheckpointStore  ->  Component  ->  result + durable state
```

## Installation

Copy the `durable_steps/` directory into your project. Python 3.10+.

## Dependencies

Standard library only — `sqlite3` ships with Python. For Postgres, implement the
four-method `CheckpointStore` protocol; the SQL is nearly identical (`ON CONFLICT`
for the upsert, a conditional `UPDATE ... WHERE lease_expiry < now()` for the claim).

## Input Schema

| Field | Type | Meaning |
|---|---|---|
| `run_id` | `str` | Idempotency key. **Derive it from the business event**, not from `uuid4()` or a timestamp, or you lose resumption |
| `steps` | `Sequence[Step]` | `name` (unique), `fn`, `retry`, `idempotent` |
| `initial_context` | `Mapping[str, Any]` | Seed values available to the first step |
| `lease_seconds` | `float` | How long this worker claims the run |

`Step.fn` receives the accumulated context (`step name -> result`) and must return a
**JSON-serialisable** value. That is the cost of durability: the result has to
survive a process restart, so it has to be storable. A non-serialisable return
raises `TypeError` immediately rather than failing silently later.

## Output Schema

`WorkflowResult`: `run_id`, `status` (`running`/`completed`/`failed`), `context`,
`executed` (ran in this invocation), `replayed` (skipped via checkpoint),
`failed_step`, `error`.

Exceptions: `LeaseNotAcquired`, `StepNotReplayable`, `TypeError` (non-serialisable
result). `NonRetryableError` is for *you* to raise inside a step.

## Usage

```python
from durable_steps import (
    DurableStepsComponent, NonRetryableError, RetryPolicy,
    SqliteCheckpointStore, Step, WorkflowRequest,
)

store = SqliteCheckpointStore("runs.db")

def fetch(ctx):
    return {"pages": 42}                      # checkpointed as JSON

def extract(ctx):
    if ctx["fetch"]["pages"] == 0:
        raise NonRetryableError("empty document")   # skips the retry budget
    return ["clause one", "clause two"]

def load(ctx):
    return {"written": len(ctx["extract"])}

result = DurableStepsComponent(store).execute(
    WorkflowRequest(
        run_id="ingest:contract-8841",           # stable, business-derived
        steps=[
            Step("fetch", fetch, retry=RetryPolicy(max_attempts=5)),
            Step("extract", extract),
            Step("load", load, idempotent=False),  # has an external side effect
        ],
    )
)
print(result.status, result.executed, result.replayed)

# Kill the process mid-run and call the exact same thing again:
# completed steps land in `replayed`, and execution resumes where it stopped.
```

## Limitations

- **Not a scheduler.** No queue, no worker pool, no cron, no cross-process
  signalling. It makes a run resumable; deciding when to run it is yours.
- Steps run sequentially in the calling thread. No parallel fan-out. When the work
  is a graph rather than a list, use `toolkit/dag`, which keeps this component's
  checkpoint-and-replay semantics by composing with it rather than replacing it.
- Step results must be JSON-serialisable. Pass large payloads by reference (an S3
  key, a row id), not by value.
- `idempotent=False` gives you *detection*, not exactly-once. If a non-idempotent
  step is interrupted, the component refuses to guess and raises
  `StepNotReplayable` for a human to resolve. True exactly-once needs an
  idempotency key at the external service.
- The lease is time-based. A worker that stalls past `lease_seconds` without dying
  can still be joined by a second worker. Set `lease_seconds` comfortably above your
  worst realistic step duration.
- SQLite is fine for a single process or a few processes on one machine. For real
  concurrency across hosts, implement the Postgres store.
- Changing a step's `name` orphans its checkpoint; changing the *order* of steps
  mid-run is not detected. Version your `run_id` if the workflow shape changes.
- Retry sleeps block the calling thread. No async variant.

## Integration Guide

1. Split your pipeline at the points where a crash would be expensive. One step per
   network boundary is the usual right granularity — too fine and you pay storage
   overhead, too coarse and a crash costs real work.
2. Make `run_id` deterministic from the input (`"ingest:" + document_hash`). This is
   the single decision that determines whether resumption works at all.
3. Return small, serialisable summaries from each step. Write bulk output to
   external storage and return its location.
4. Raise `NonRetryableError` for anything a retry cannot fix — validation failures,
   404s, malformed input. Retrying those burns the budget and delays the real signal.
5. Mark side-effecting steps `idempotent=False`, and make them idempotent at the
   service level where you can (an idempotency key on the payment, a deterministic
   message id on the email). Then you can mark them idempotent honestly.
6. For Postgres, implement `CheckpointStore` and pass it in. Nothing else changes.

## Extraction Notes

- **Preserved:** the replay-from-checkpoint semantics, exponential backoff with
  jitter, lease-based single-owner claiming, and the record-intent-before-acting
  pattern for non-idempotent work.
- **Removed:** the entire distributed runtime — queues, worker registration,
  heartbeating, scheduling, dashboards, gRPC transport, and the decorator/DSL layer
  that couples business logic to a framework.
- **Rewritten:** the store is a four-method `Protocol` rather than a fixed backend,
  so SQLite ships as the reference and Postgres is a drop-in. The lease claim is one
  conditional `UPDATE` inside a single `BEGIN IMMEDIATE` transaction, so two workers
  racing on an expired lease cannot both win.
- **Isolated:** no server, no daemon, no network, no third-party package. Steps are
  plain callables taking a mapping, so nothing about your business logic has to
  import this component's types to be testable.
