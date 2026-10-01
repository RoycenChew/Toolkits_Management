# DAG Executor Component

**Layer 2 · depends on `core`, `durable_steps`, `graph` · `copy_tier: needs_package`**

## What It Does

Runs a graph of tasks: parallel wherever the dependencies allow, with per-node retry,
conditional branches, honest failure propagation, and optional checkpointed resume.

## Why It Is Useful

`durable_steps` runs a **list**. An agent plan, a fan-out over documents, and an event
pipeline are all **graphs**. That documented limitation is the reason this exists:
stages 5-7 of the pipeline in `docs/ARCHITECTURE.md` fan out per document, and a list
cannot express it. `recipe:parallel_document_pipeline` is the consumer.

What it is not: a scheduler. There is no cron, no queue, no worker registration and no
distributed coordination. It executes one graph in one process. That is the part that
is hard to get right and reusable; deciding *when* to run is yours.

## The three decisions that matter

These are where DAG executors usually go wrong.

**A ready queue, not layer-by-layer.** Running topological layers in lockstep is
simpler, but one slow node stalls every node in the next layer even when their own
dependencies finished long ago. A node here is submitted the moment its last
dependency completes. `topological_layers` is still *reported* as the plan, because the
layer count is the floor on sequential rounds and that is worth seeing — it is just not
how work is dispatched. There is a test asserting a ready node does not wait behind an
unrelated slow one.

**A failed node's descendants are `SKIPPED`, not `FAILED`.** They did not go wrong;
there was nothing for them to do. Collapsing the two turns one real failure into a page
of red and buries the cause. Every skip records *which* node caused it.

**Fail-fast stops scheduling; it does not cancel.** A thread cannot be interrupted
safely, and killing a node mid-write is how partial state gets left behind. In-flight
work finishes, nothing new starts, and everything unrun is still accounted for.

## Architecture

```
INPUT     Node[] — id, fn, depends_on, retry, when, idempotent
   |
BUILD     Graph.from_dependencies, then validate_dag
   |       duplicate id / unknown dep / self-dep / cycle -> raise BEFORE anything runs
   |
PLAN      topological_layers (reported) + critical_path (reported)
   |
SCHEDULE  ready queue over a thread pool
   |         node completes -> decrement successors -> submit newly ready
   |         node fails     -> mark descendants SKIPPED with the cause
   |
PER NODE  checkpoint hit? -> REPLAYED, not re-run
   |       when() false?   -> SKIPPED, descendants skipped, run still OK
   |       retry on transient errors only; bugs surface immediately
   |
OUTPUT    DagResult — outcomes, results, layers, critical_path, seconds
```

## Installation

```bash
pip install -e .
```

Standard library only, but it composes three other units (`core`, `durable_steps`,
`graph`), so install the package rather than copying the directory. If you only need
the algorithms, `toolkit/graph` **is** standalone.

## Input Schema

`Node(id, fn, depends_on=(), retry=RetryPolicy(), when=None, idempotent=True)`

**`fn` receives its declared dependencies' results plus the initial context — not
everything completed so far.** That is deliberate: reading a result you did not declare
makes the node silently order-dependent, which works until the scheduler's timing
shifts and then fails unreproducibly. Restricting the context is what forces the
dependency to be declared. There is a test asserting an undeclared sibling is invisible.

`DagConfig(max_workers=8, on_failure=CONTINUE, checkpoint_db=None, node_timeout=None)`

| Field | Note |
|---|---|
| `on_failure` | `CONTINUE` (default) runs independent branches; `FAIL_FAST` stops scheduling |
| `checkpoint_db` | enables resume. Node results must then be JSON-serialisable |
| `node_timeout` | **advisory** — bounds how long the scheduler *waits*, not how long a node runs |

`DagRequest(nodes, run_id="dag", initial_context={}, config=DagConfig())` — derive
`run_id` from the work, not a timestamp, or resumption never finds the previous attempt.

## Output Schema

`DagResult`: `status`, `outcomes` (id → `NodeOutcome` with status, attempts, seconds,
error, skip reason), `results` (id → return value), `layers`, `critical_path`,
`seconds`. Convenience lists: `.completed`, `.replayed`, `.skipped`, `.failed`, `.ok`,
and `.render()` for a one-screen summary.

Raises `DagError` for a malformed graph and `CycleError` for a cyclic one — both before
any node executes.

## Usage

```python
from toolkit.dag import DagConfig, DagExecutorComponent, DagRequest, Node

nodes = [
    Node("extract", extract),
    Node("clean",   clean,   ["extract"]),
    Node("enrich",  enrich,  ["extract"]),      # runs alongside clean
    Node("index",   index,   ["clean", "enrich"]),
]
result = DagExecutorComponent().execute(
    DagRequest(nodes, config=DagConfig(max_workers=4))
)
print(result.render())
print(result.results["index"])
```

Resumable, with retry on transient failures:

```python
from toolkit.dag import RetryPolicy

nodes = [
    Node("fetch", fetch, retry=RetryPolicy(max_attempts=5)),
    Node("load",  load,  ["fetch"], idempotent=False),
]
request = DagRequest(nodes, run_id="etl:2026-10-01",
                     config=DagConfig(checkpoint_db="runs.db"))
DagExecutorComponent().execute(request)   # crash, then run the same thing again
```

A conditional branch:

```python
Node("reindex", reindex, ["diff"], when=lambda ctx: ctx["diff"]["changed"] > 0)
```

`python examples/cookbook.py dag` and
`python examples/cookbook.py recipe:parallel_document_pipeline` both run.

## Limitations

- **Threads, not processes.** Correct for IO-bound work, which is what this toolkit
  does; CPU-bound nodes contend on the GIL and will not speed up.
- **`node_timeout` cannot stop a node.** Python threads are not interruptible, so the
  scheduler stops *waiting* and reports a timeout while the node keeps running. For a
  hard bound, run the work in a subprocess inside the node.
- **The graph is fixed before execution.** No dynamic fan-out — a node cannot spawn
  nodes based on its own result. Build the graph once you know the inputs, as the
  parallel-document recipe does.
- **Single machine.** No distributed coordination. `durable_steps` has leasing; this
  does not, so two processes running the same `run_id` concurrently will both work.
- **Checkpointed results must be JSON-serialisable**, and a non-serialisable return
  raises `TypeError` rather than silently skipping the checkpoint.
- `idempotent=False` gives *detection*, not exactly-once: an interrupted node is
  reported for a human to resolve, because whether the side effect landed is unknowable
  from here.
- No priorities, no resource limits beyond `max_workers`, no per-node concurrency keys.
- No visualisation; `graph` has no serialisation format either.

## Integration Guide

1. **Declare dependencies, do not read undeclared results.** The restricted context
   enforces it, but the failure mode it prevents is worth understanding.
2. Derive `run_id` from the work (`"ingest:" + corpus_hash`). This single decision
   determines whether resume works at all.
3. Return small, serialisable summaries. Write bulk output to storage and return its
   location.
4. Keep `CONTINUE` unless a partial result is worse than none. In a fan-out over
   documents, one bad file should not abandon the other nine hundred.
5. Read `result.critical_path` before adding workers. If the longest chain is the
   bottleneck, more parallelism buys nothing.
6. Mark side-effecting nodes `idempotent=False`, and make them idempotent at the
   service level where you can — then you can honestly mark them `True`.

## Extraction Notes

- **Preserved:** the retry-policy shape from Temporal, and the checkpoint-and-replay
  semantics already in `durable_steps`, reused rather than reimplemented.
- **Removed:** the entire distributed runtime that Airflow, Prefect and Temporal wrap
  this in — queues, workers, heartbeats, scheduling, dashboards, and the DSL that
  couples business logic to a framework.
- **Added:** the ready-queue scheduler, `SKIPPED` as distinct from `FAILED` with the
  cause recorded, the dependency-restricted context, conditional nodes, up-front graph
  validation, and checkpointing by composition with `durable_steps` rather than by
  rewriting it.
- **Isolated:** no server, no daemon, no third-party package. Nodes are plain callables
  taking a mapping, so business logic needs no import from here to be testable.
