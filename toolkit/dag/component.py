"""DAG execution: parallel where the graph allows, with honest failure propagation.

`durable_steps` runs a list. This runs a graph, which is what an agent plan, a
fan-out ingestion and an event pipeline all actually are. That documented
limitation of `durable_steps` is the reason this exists;
`recipe:parallel_document_pipeline` is the consumer.

Three decisions worth stating, because they are where DAG executors usually go
wrong:

**A ready queue, not layer-by-layer.** Running topological layers in lockstep is
simpler, but one slow node in a layer stalls every node in the next even when
its own dependencies finished long ago. The scheduler here submits a node the
moment its last dependency completes. `topological_layers` is still reported as
the *plan*, because the layer count is the floor on sequential rounds and that
is useful to see — it is just not how work is dispatched.

**A failed node's descendants are SKIPPED, not FAILED.** They did not go wrong;
there was nothing for them to do. Collapsing the two turns one real failure into
a page of red and buries the cause.

**Fail-fast stops scheduling, it does not cancel.** A thread cannot be
interrupted safely, and killing a node mid-write is how partial state gets left
behind. In-flight work finishes; nothing new starts.
"""
from __future__ import annotations

import json
import random
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any

from ..core.errors import AdapterError, RateLimited
from ..durable_steps import SqliteCheckpointStore, StepRecord, StepStatus
from ..graph import Graph, critical_path, descendants, topological_layers, validate_dag
from .models import (
    DagConfig,
    DagError,
    DagRequest,
    DagResult,
    FailurePolicy,
    Node,
    NodeOutcome,
    NodeStatus,
    RetryPolicy,
    RunStatus,
)

_RETRYABLE = (RateLimited, AdapterError, TimeoutError, ConnectionError, OSError)
"""Only transient classes retry. A `KeyError` or a `TypeError` is a bug, and
attempting it three times delays the stack trace without improving the odds."""


class DagExecutorComponent:
    """Nodes in, results out. Stateless; safe to reuse across runs."""

    def execute(self, input_data: DagRequest) -> DagResult:
        started = time.monotonic()
        nodes = self._index(input_data.nodes)
        graph = Graph.from_dependencies({n.id: list(n.depends_on) for n in nodes.values()})
        validate_dag(graph)

        plan = topological_layers(graph)
        _, path = critical_path(graph)

        store = (
            SqliteCheckpointStore(input_data.config.checkpoint_db)
            if input_data.config.checkpoint_db
            else None
        )
        records = dict(store.load_steps(input_data.run_id)) if store else {}

        state = _RunState(
            graph=graph,
            nodes=nodes,
            config=input_data.config,
            initial_context=dict(input_data.initial_context),
            store=store,
            run_id=input_data.run_id,
            records=records,
        )
        self._run(state)

        return DagResult(
            status=RunStatus.FAILED if state.outcomes_failed() else RunStatus.COMPLETED,
            outcomes=state.outcomes,
            results=state.results,
            layers=plan,
            critical_path=path,
            seconds=time.monotonic() - started,
        )

    # --- validation -------------------------------------------------------

    def _index(self, nodes: Sequence[Node]) -> dict[str, Node]:
        """Reject a malformed graph before anything runs.

        A half-executed DAG is worse than a refused one: the side effects
        happened and the reason is buried in a traceback, so both checks are
        done up front where the error can name the problem.
        """
        indexed: dict[str, Node] = {}
        for node in nodes:
            if node.id in indexed:
                raise DagError("duplicate node id: " + repr(node.id))
            indexed[node.id] = node
        if not indexed:
            raise DagError("at least one node is required")

        for node in indexed.values():
            for dependency in node.depends_on:
                if dependency not in indexed:
                    raise DagError(
                        "node %r depends on %r, which was never declared"
                        % (node.id, dependency)
                    )
            if node.id in node.depends_on:
                raise DagError("node %r depends on itself" % node.id)
        return indexed

    # --- scheduling -------------------------------------------------------

    def _run(self, state: _RunState) -> None:
        pending = {
            node_id: state.graph.in_degree(node_id) for node_id in state.graph.nodes
        }
        ready = [node_id for node_id, degree in sorted(pending.items()) if degree == 0]
        in_flight: dict[Future[_NodeResult], str] = {}
        halted = False

        with ThreadPoolExecutor(max_workers=state.config.max_workers) as pool:
            while ready or in_flight:
                while ready and not halted:
                    node_id = ready.pop(0)
                    pending.pop(node_id, None)
                    decided = self._precheck(state, node_id)
                    if decided is not None:
                        # Skipped or replayed without running: release its
                        # dependents immediately rather than waiting a round.
                        self._settle(state, node_id, decided, pending, ready)
                        continue
                    future = pool.submit(self._run_node, state, node_id)
                    in_flight[future] = node_id

                if not in_flight:
                    break

                done, _ = wait(
                    list(in_flight),
                    timeout=state.config.node_timeout,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    # The wait timed out. A thread cannot be interrupted, so the
                    # node keeps running; the scheduler simply stops waiting on
                    # it and reports the timeout. Documented as advisory.
                    for future, node_id in list(in_flight.items()):
                        state.record_outcome(
                            NodeOutcome(
                                node_id, NodeStatus.FAILED, error="timed out waiting"
                            )
                        )
                        in_flight.pop(future, None)
                        self._propagate_skip(state, node_id, pending, ready)
                    halted = state.config.on_failure is FailurePolicy.FAIL_FAST
                    continue

                for future in done:
                    node_id = in_flight.pop(future)
                    outcome = future.result()
                    self._settle(state, node_id, outcome, pending, ready)
                    if (
                        outcome.outcome.status is NodeStatus.FAILED
                        and state.config.on_failure is FailurePolicy.FAIL_FAST
                    ):
                        halted = True

        # Anything still pending when the queue drains was cut off by fail-fast
        # or by an upstream failure that had already been propagated.
        for node_id in sorted(pending):
            if node_id not in state.outcomes:
                state.record_outcome(
                    NodeOutcome(
                        node_id,
                        NodeStatus.SKIPPED,
                        reason="not scheduled: run halted after a failure",
                    )
                )

    def _precheck(self, state: _RunState, node_id: str) -> _NodeResult | None:
        """Decide a node without running it: skipped, replayed, or neither."""
        node = state.nodes[node_id]

        failed_dependency = next(
            (
                dependency
                for dependency in sorted(node.depends_on)
                if state.outcomes.get(dependency)
                and state.outcomes[dependency].status
                in (NodeStatus.FAILED, NodeStatus.SKIPPED)
            ),
            None,
        )
        if failed_dependency:
            reason = "dependency %r did not complete" % failed_dependency
            return _NodeResult(NodeOutcome(node_id, NodeStatus.SKIPPED, reason=reason), None)

        record = state.records.get(node_id)
        if record is not None and record.status is StepStatus.COMPLETED:
            value = json.loads(record.result_json) if record.result_json else None
            return _NodeResult(
                NodeOutcome(node_id, NodeStatus.REPLAYED, attempts=record.attempts), value
            )
        if (
            record is not None
            and not node.idempotent
            and record.attempts > 0
            and record.status is not StepStatus.FAILED
        ):
            return _NodeResult(
                NodeOutcome(
                    node_id,
                    NodeStatus.FAILED,
                    attempts=record.attempts,
                    error=(
                        "interrupted after %d attempt(s) and not idempotent;"
                        " resolve manually" % record.attempts
                    ),
                ),
                None,
            )

        if node.when is not None and not node.when(state.context_for(node)):
            return _NodeResult(
                NodeOutcome(node_id, NodeStatus.SKIPPED, reason="when() returned False"),
                None,
            )
        return None

    def _run_node(self, state: _RunState, node_id: str) -> _NodeResult:
        node = state.nodes[node_id]
        context = state.context_for(node)
        started = time.monotonic()
        attempts = 0
        last_error = ""

        for attempt in range(1, node.retry.max_attempts + 1):
            attempts += 1
            if not node.idempotent and state.store is not None:
                # Record the intent before acting, so an interruption is
                # detectable on the next run rather than silently retried.
                state.store.record_step(
                    StepRecord(state.run_id, node_id, StepStatus.PENDING, attempts, None, None)
                )
            try:
                value = node.fn(context)
            except _RETRYABLE as exc:
                last_error = type(exc).__name__ + ": " + str(exc)
                if attempt >= node.retry.max_attempts:
                    break
                time.sleep(_backoff(node.retry, attempt, exc))
                continue
            except BaseException as exc:  # noqa: BLE001 - not retryable, reported
                return _NodeResult(
                    NodeOutcome(
                        node_id,
                        NodeStatus.FAILED,
                        attempts=attempts,
                        seconds=time.monotonic() - started,
                        error=type(exc).__name__ + ": " + str(exc),
                    ),
                    None,
                )

            return _NodeResult(
                NodeOutcome(
                    node_id,
                    NodeStatus.COMPLETED,
                    attempts=attempts,
                    seconds=time.monotonic() - started,
                ),
                value,
            )

        return _NodeResult(
            NodeOutcome(
                node_id,
                NodeStatus.FAILED,
                attempts=attempts,
                seconds=time.monotonic() - started,
                error=last_error,
            ),
            None,
        )

    # --- bookkeeping ------------------------------------------------------

    def _settle(
        self,
        state: _RunState,
        node_id: str,
        result: _NodeResult,
        pending: dict[str, int],
        ready: list[str],
    ) -> None:
        state.record_outcome(result.outcome)

        if result.outcome.status in (NodeStatus.COMPLETED, NodeStatus.REPLAYED):
            state.results[node_id] = result.value
            if state.store is not None and result.outcome.status is NodeStatus.COMPLETED:
                state.checkpoint(node_id, result)
            for successor in state.graph.out_edges(node_id):
                if successor in pending:
                    pending[successor] -= 1
                    if pending[successor] == 0:
                        _insert_sorted(ready, successor)
            return

        self._propagate_skip(state, node_id, pending, ready)

    def _propagate_skip(
        self,
        state: _RunState,
        node_id: str,
        pending: dict[str, int],
        ready: list[str],
    ) -> None:
        """Mark everything downstream of a failure as skipped.

        Computed from the graph in one pass rather than discovered node by node
        as the queue drains, so the result is complete even under fail-fast and
        every skip names the node that caused it.
        """
        for downstream in sorted(descendants(state.graph, node_id)):
            if downstream in state.outcomes:
                continue
            state.record_outcome(
                NodeOutcome(
                    downstream,
                    NodeStatus.SKIPPED,
                    reason="upstream %r did not complete" % node_id,
                )
            )
            pending.pop(downstream, None)
            if downstream in ready:
                ready.remove(downstream)


def _insert_sorted(ready: list[str], node_id: str) -> None:
    position = 0
    while position < len(ready) and ready[position] < node_id:
        position += 1
    ready.insert(position, node_id)


def _backoff(policy: RetryPolicy, attempt: int, exc: BaseException) -> float:
    hint = getattr(exc, "retry_after", None)
    if hint:
        return float(hint)
    delay = min(
        policy.initial_backoff * (policy.backoff_multiplier ** (attempt - 1)),
        policy.max_backoff,
    )
    if policy.jitter:
        delay *= 1.0 + random.uniform(-policy.jitter, policy.jitter)
    return max(0.0, delay)


class _NodeResult:
    __slots__ = ("outcome", "value")

    def __init__(self, outcome: NodeOutcome, value: Any) -> None:
        self.outcome = outcome
        self.value = value


class _RunState:
    """Mutable run state, deliberately separate from the component.

    The component stays stateless and reusable; everything that changes during a
    run lives here, which is what makes concurrent runs of the same component
    safe.
    """

    def __init__(
        self,
        graph: Graph,
        nodes: Mapping[str, Node],
        config: DagConfig,
        initial_context: dict[str, Any],
        store: Any,
        run_id: str,
        records: Mapping[str, StepRecord],
    ) -> None:
        self.graph = graph
        self.nodes = nodes
        self.config = config
        self.initial_context = initial_context
        self.store = store
        self.run_id = run_id
        self.records = records
        self.outcomes: dict[str, NodeOutcome] = {}
        self.results: dict[str, Any] = {}

    def context_for(self, node: Node) -> dict[str, Any]:
        """Initial context plus the results of this node's declared dependencies.

        Not everything completed so far — see `Node` for why restricting it is
        the point rather than an inconvenience.
        """
        context = dict(self.initial_context)
        for dependency in node.depends_on:
            if dependency in self.results:
                context[dependency] = self.results[dependency]
        return context

    def record_outcome(self, outcome: NodeOutcome) -> None:
        self.outcomes[outcome.id] = outcome

    def outcomes_failed(self) -> bool:
        return any(o.status is NodeStatus.FAILED for o in self.outcomes.values())

    def checkpoint(self, node_id: str, result: _NodeResult) -> None:
        try:
            payload = json.dumps(result.value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "node %r returned a value that is not JSON serialisable; a"
                " checkpointed DAG must return storable results" % node_id
            ) from exc
        self.store.record_step(
            StepRecord(
                self.run_id, node_id, StepStatus.COMPLETED, result.outcome.attempts,
                payload, None,
            )
        )


__all__ = ["DagExecutorComponent"]
