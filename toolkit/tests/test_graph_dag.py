"""Tests for graph algorithms and DAG execution.

The properties that matter here are mostly about *failure*: what happens to the
rest of a graph when one node dies, whether a cycle is reported usefully, and
whether a 2,000-node chain survives. A DAG executor that only works on the happy
path is a for-loop with extra steps.

Run standalone: python toolkit/tests/test_graph_dag.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from toolkit.core import AdapterError, RateLimited  # noqa: E402
from toolkit.dag import (  # noqa: E402
    DagConfig,
    DagError,
    DagExecutorComponent,
    DagRequest,
    FailurePolicy,
    Node,
    NodeStatus,
    RetryPolicy,
    RunStatus,
)
from toolkit.graph import (  # noqa: E402
    CycleError,
    Graph,
    NodeNotFound,
    ancestors,
    critical_path,
    descendants,
    find_cycle,
    is_dag,
    shortest_path,
    strongly_connected_components,
    topological_layers,
    topological_order,
    transitive_reduction,
)

PIPELINE = {
    "build": ["fetch"],
    "lint": ["fetch"],
    "test": ["build"],
    "ship": ["test", "lint"],
}


def _pipeline() -> Graph:
    return Graph.from_dependencies(PIPELINE)


# ==========================================================================
# Graph construction
# ==========================================================================


def test_dependencies_and_successors_are_opposite_directions():
    """Mixing these up gives a reversed execution order that still looks
    plausible, which is why both constructors exist and are named."""
    deps = Graph.from_dependencies({"b": ["a"]})
    succ = Graph.from_successors({"a": ["b"]})
    assert deps.successors == succ.successors
    assert deps.roots() == ["a"] and deps.leaves() == ["b"]

    # The wrong one produces a graph that is valid and backwards.
    wrong = Graph.from_successors({"b": ["a"]})
    assert wrong.roots() == ["b"], "reversed, and nothing complains"


def test_nodes_mentioned_only_as_targets_are_added():
    graph = Graph.from_successors({"a": ["b", "c"]})
    assert graph.nodes == frozenset({"a", "b", "c"})
    assert graph.edge_count == 2


def test_unknown_node_is_rejected_at_construction_and_on_query():
    try:
        Graph(nodes=frozenset({"a"}), successors={"a": frozenset({"ghost"})})
    except NodeNotFound as exc:
        assert exc.node == "ghost"
    else:
        raise AssertionError("expected NodeNotFound")

    graph = _pipeline()
    for call in (lambda: graph.out_edges("nope"), lambda: descendants(graph, "nope")):
        try:
            call()
        except NodeNotFound:
            continue
        raise AssertionError("expected NodeNotFound")


def test_reverse_and_subgraph():
    graph = _pipeline()
    assert graph.reverse().roots() == ["ship"]
    sub = graph.subgraph(["fetch", "build", "test"])
    assert sub.nodes == frozenset({"fetch", "build", "test"})
    assert sub.out_edges("test") == [], "edges to dropped nodes are removed"


# ==========================================================================
# Ordering
# ==========================================================================


def test_topological_order_respects_dependencies_and_is_deterministic():
    order = topological_order(_pipeline())
    position = {node: i for i, node in enumerate(order)}
    for node, dependencies in PIPELINE.items():
        for dependency in dependencies:
            assert position[dependency] < position[node], (dependency, node)
    # Stable across runs: a non-deterministic order makes an execution log
    # undiffable and a flaky test unattributable.
    assert order == topological_order(_pipeline())
    assert order == ["fetch", "build", "lint", "test", "ship"]


def test_layers_are_parallel_batches():
    layers = topological_layers(_pipeline())
    assert layers == [["fetch"], ["build", "lint"], ["test"], ["ship"]]
    # Layer count is the longest dependency chain, the floor on sequential
    # rounds however many workers are available.
    assert len(layers) == critical_path(_pipeline())[0]


def test_cycle_is_reported_with_the_actual_cycle():
    graph = Graph.from_successors({"a": ["b"], "b": ["c"], "c": ["a"]})
    assert not is_dag(graph)
    try:
        topological_order(graph)
    except CycleError as exc:
        assert exc.cycle[0] == exc.cycle[-1], "should read as a loop"
        assert set(exc.cycle) == {"a", "b", "c"}
        assert "->" in str(exc)
    else:
        raise AssertionError("expected CycleError")

    try:
        topological_layers(graph)
    except CycleError:
        pass
    else:
        raise AssertionError("layers must also refuse a cyclic graph")


def test_self_loop_and_two_node_cycle():
    assert find_cycle(Graph.from_successors({"a": ["a"]})) == ["a", "a"]
    cycle = find_cycle(Graph.from_successors({"a": ["b"], "b": ["a"]}))
    assert cycle is not None and len(cycle) == 3


def test_find_cycle_returns_none_on_a_dag():
    assert find_cycle(_pipeline()) is None
    assert is_dag(_pipeline())


def test_deep_chain_does_not_exhaust_the_stack():
    """A 2,000-node dependency chain is an ordinary monorepo. Recursive
    implementations of these algorithms die here, and only on real input."""
    size = 2000
    chain = {"n%d" % i: ["n%d" % (i - 1)] for i in range(1, size)}
    graph = Graph.from_dependencies(chain)
    assert len(topological_order(graph)) == size
    assert len(topological_layers(graph)) == size
    assert find_cycle(graph) is None
    assert len(strongly_connected_components(graph)) == size
    assert len(descendants(graph, "n0")) == size - 1
    assert critical_path(graph)[0] == size


# ==========================================================================
# Traversal and paths
# ==========================================================================


def test_descendants_and_ancestors_exclude_the_node():
    graph = _pipeline()
    assert descendants(graph, "fetch") == {"build", "lint", "test", "ship"}
    assert descendants(graph, "ship") == set()
    assert ancestors(graph, "ship") == {"fetch", "build", "lint", "test"}
    assert ancestors(graph, "fetch") == set()


def test_strongly_connected_components():
    assert strongly_connected_components(_pipeline()) == [
        ["build"], ["fetch"], ["lint"], ["ship"], ["test"],
    ] or all(len(c) == 1 for c in strongly_connected_components(_pipeline()))

    tangled = Graph.from_successors({"a": ["b"], "b": ["c"], "c": ["a"], "d": ["a"]})
    components = strongly_connected_components(tangled)
    multi = [c for c in components if len(c) > 1]
    assert multi == [["a", "b", "c"]], components


def test_shortest_path_unweighted_and_weighted():
    graph = _pipeline()
    assert shortest_path(graph, "fetch", "ship") == ["fetch", "lint", "ship"]
    assert shortest_path(graph, "fetch", "fetch") == ["fetch"]
    assert shortest_path(graph, "ship", "fetch") is None, "edges are directed"

    # Weighting makes the longer hop count cheaper.
    costs = {("fetch", "lint"): 100.0, ("lint", "ship"): 100.0}
    weighted = shortest_path(
        graph, "fetch", "ship", weight=lambda a, b: costs.get((a, b), 1.0)
    )
    assert weighted == ["fetch", "build", "test", "ship"]


def test_negative_weights_are_refused_rather_than_mishandled():
    graph = Graph.from_successors({"a": ["b"], "b": ["c"]})
    try:
        shortest_path(graph, "a", "c", weight=lambda x, y: -1.0)
    except ValueError as exc:
        assert "negative" in str(exc)
    else:
        raise AssertionError("Dijkstra requires non-negative weights")


def test_critical_path_with_durations():
    graph = _pipeline()
    # Make the lint branch the expensive one.
    durations = {"fetch": 1.0, "lint": 50.0, "build": 2.0, "test": 2.0, "ship": 1.0}
    total, path = critical_path(graph, durations)
    assert path == ["fetch", "lint", "ship"]
    assert total == 52.0

    total_default, _ = critical_path(graph)
    assert total_default == 4.0, "no durations means every node costs 1"

    try:
        critical_path(Graph.from_successors({"a": ["b"], "b": ["a"]}))
    except CycleError:
        pass
    else:
        raise AssertionError("longest path is unbounded on a cyclic graph")


def test_transitive_reduction_drops_implied_edges():
    graph = Graph.from_successors({"a": ["b", "c"], "b": ["c"]})
    reduced = transitive_reduction(graph)
    assert reduced.out_edges("a") == ["b"], "a->c is implied by a->b->c"
    assert reduced.out_edges("b") == ["c"]
    # Reachability is preserved, which is the whole point.
    assert descendants(reduced, "a") == descendants(graph, "a")


# ==========================================================================
# DAG execution
# ==========================================================================


def _recorder():
    log: list[str] = []
    lock = threading.Lock()

    def make(name: str, fail: type[BaseException] | None = None, delay: float = 0.0):
        def fn(ctx):
            with lock:
                log.append(name)
            if delay:
                time.sleep(delay)
            if fail is not None:
                raise fail("failing in " + name)
            return name.upper()

        return fn

    return log, make


def test_runs_a_graph_and_wires_results():
    log, make = _recorder()
    nodes = [
        Node("fetch", make("fetch")),
        Node("build", make("build"), ["fetch"]),
        Node("lint", make("lint"), ["fetch"]),
        Node("test", make("test"), ["build"]),
        Node("ship", make("ship"), ["test", "lint"]),
    ]
    result = DagExecutorComponent().execute(DagRequest(nodes))

    assert result.ok and result.status is RunStatus.COMPLETED
    assert sorted(result.completed) == ["build", "fetch", "lint", "ship", "test"]
    assert result.results["ship"] == "SHIP"
    assert result.layers == [["fetch"], ["build", "lint"], ["test"], ["ship"]]
    assert list(result.critical_path) == ["fetch", "build", "test", "ship"]
    assert log.index("fetch") < log.index("build") < log.index("test")


def test_a_node_only_sees_its_declared_dependencies():
    """Reading an undeclared result makes a node silently order-dependent: it
    works until the scheduler's timing shifts, then fails unreproducibly."""
    seen: dict[str, set[str]] = {}

    def spy(name: str):
        def fn(ctx):
            seen[name] = set(ctx)
            return name

        return fn

    nodes = [
        Node("a", spy("a")),
        Node("b", spy("b")),
        Node("c", spy("c"), ["a"]),
    ]
    DagExecutorComponent().execute(DagRequest(nodes, initial_context={"seed": 1}))

    assert seen["c"] == {"seed", "a"}, "c declared only a, so b must be invisible"
    assert seen["a"] == {"seed"}
    assert "b" not in seen["c"]


def test_independent_branches_run_in_parallel():
    _, make = _recorder()
    nodes = [
        Node("root", make("root")),
        Node("slow_a", make("slow_a", delay=0.25), ["root"]),
        Node("slow_b", make("slow_b", delay=0.25), ["root"]),
        Node("slow_c", make("slow_c", delay=0.25), ["root"]),
    ]
    started = time.monotonic()
    result = DagExecutorComponent().execute(
        DagRequest(nodes, config=DagConfig(max_workers=4))
    )
    elapsed = time.monotonic() - started

    assert result.ok
    assert elapsed < 0.6, (
        "three 0.25s independent nodes took %.2fs; they ran sequentially" % elapsed
    )


def test_a_ready_node_does_not_wait_for_an_unrelated_slow_node():
    """The reason this is a ready queue rather than lockstep layers: a slow node
    must not stall a node whose own dependencies have finished."""
    finished: list[str] = []
    lock = threading.Lock()

    def record(name: str, delay: float = 0.0):
        def fn(ctx):
            if delay:
                time.sleep(delay)
            with lock:
                finished.append(name)
            return name

        return fn

    nodes = [
        Node("slow_root", record("slow_root", delay=0.3)),
        Node("fast_root", record("fast_root")),
        Node("after_fast", record("after_fast"), ["fast_root"]),
    ]
    result = DagExecutorComponent().execute(
        DagRequest(nodes, config=DagConfig(max_workers=4))
    )
    assert result.ok
    assert finished.index("after_fast") < finished.index("slow_root"), (
        "after_fast waited for an unrelated slow node: %s" % finished
    )


# --- failure semantics -----------------------------------------------------


def test_failure_skips_descendants_and_continues_independent_branches():
    log, make = _recorder()
    nodes = [
        Node("fetch", make("fetch")),
        Node("build", make("build", fail=RuntimeError), ["fetch"]),
        Node("lint", make("lint"), ["fetch"]),
        Node("test", make("test"), ["build"]),
        Node("ship", make("ship"), ["test", "lint"]),
    ]
    result = DagExecutorComponent().execute(DagRequest(nodes))

    assert result.status is RunStatus.FAILED
    assert result.failed == ["build"]
    assert result.skipped == ["ship", "test"], "descendants skipped, not failed"
    assert "lint" in result.completed, "an independent branch must still run"
    assert "test" not in log and "ship" not in log, "skipped nodes must not execute"

    # Every skip names its cause; an unexplained skip is a gap in the run.
    assert "build" in result.outcomes["test"].reason
    assert result.outcomes["build"].status is NodeStatus.FAILED
    assert "RuntimeError" in result.outcomes["build"].error


def test_fail_fast_stops_scheduling_new_work():
    log, make = _recorder()
    nodes = [
        Node("boom", make("boom", fail=RuntimeError)),
        Node("later_a", make("later_a"), ["boom"]),
        Node("unrelated", make("unrelated", delay=0.2)),
        Node("after_unrelated", make("after_unrelated"), ["unrelated"]),
    ]
    result = DagExecutorComponent().execute(
        DagRequest(nodes, config=DagConfig(max_workers=1,
                                           on_failure=FailurePolicy.FAIL_FAST))
    )
    assert result.status is RunStatus.FAILED
    assert result.failed == ["boom"]
    # Nothing new was scheduled, and every unrun node is accounted for.
    assert set(result.skipped) | set(result.completed) | {"boom"} == {
        "boom", "later_a", "unrelated", "after_unrelated"
    }
    assert "later_a" not in log


def test_continue_is_the_default_policy():
    assert DagConfig().on_failure is FailurePolicy.CONTINUE


def test_retryable_errors_retry_and_bugs_do_not():
    attempts = {"flaky": 0, "buggy": 0}

    def flaky(ctx):
        attempts["flaky"] += 1
        if attempts["flaky"] < 3:
            raise RateLimited("slow down", retry_after=0.0)
        return "recovered"

    def buggy(ctx):
        attempts["buggy"] += 1
        raise KeyError("typo")

    policy = RetryPolicy(max_attempts=4, initial_backoff=0.0, jitter=0.0)
    result = DagExecutorComponent().execute(
        DagRequest([
            Node("flaky", flaky, retry=policy),
            Node("buggy", buggy, retry=policy),
        ])
    )
    assert attempts["flaky"] == 3 and result.results["flaky"] == "recovered"
    assert attempts["buggy"] == 1, "a KeyError is a bug, not a transient failure"
    assert result.outcomes["buggy"].status is NodeStatus.FAILED


def test_retries_are_bounded():
    attempts = {"n": 0}

    def always(ctx):
        attempts["n"] += 1
        raise AdapterError("backend down")

    result = DagExecutorComponent().execute(
        DagRequest([Node("x", always,
                         retry=RetryPolicy(max_attempts=3, initial_backoff=0.0,
                                           jitter=0.0))])
    )
    assert attempts["n"] == 3
    assert result.outcomes["x"].attempts == 3
    assert "AdapterError" in result.outcomes["x"].error


# --- conditional nodes -----------------------------------------------------


def test_when_predicate_skips_a_branch_without_failing_the_run():
    log, make = _recorder()
    nodes = [
        Node("check", lambda ctx: {"needed": False}),
        Node("maybe", make("maybe"), ["check"],
             when=lambda ctx: ctx["check"]["needed"]),
        Node("after", make("after"), ["maybe"]),
        Node("always", make("always"), ["check"]),
    ]
    result = DagExecutorComponent().execute(DagRequest(nodes))

    assert result.ok, "a skipped branch is not a failure: " + result.render()
    assert result.skipped == ["after", "maybe"]
    assert "when()" in result.outcomes["maybe"].reason
    assert "always" in result.completed
    assert "maybe" not in log


# --- graph validation ------------------------------------------------------


def test_malformed_graphs_are_refused_before_anything_runs():
    log, make = _recorder()
    cases = {
        "duplicate": [Node("a", make("a")), Node("a", make("a2"))],
        "unknown dependency": [Node("a", make("a"), ["ghost"])],
        "self dependency": [Node("a", make("a"), ["a"])],
        "empty": [],
    }
    for label, nodes in cases.items():
        try:
            DagExecutorComponent().execute(DagRequest(nodes))
        except DagError:
            pass
        else:
            raise AssertionError("expected DagError for " + label)
    assert log == [], "a malformed graph must not execute anything"


def test_a_cyclic_graph_raises_with_the_cycle():
    _, make = _recorder()
    nodes = [
        Node("a", make("a"), ["c"]),
        Node("b", make("b"), ["a"]),
        Node("c", make("c"), ["b"]),
    ]
    try:
        DagExecutorComponent().execute(DagRequest(nodes))
    except CycleError as exc:
        assert set(exc.cycle) >= {"a", "b", "c"}
    else:
        raise AssertionError("expected CycleError")


# --- checkpointing ---------------------------------------------------------


def test_a_crashed_run_replays_completed_nodes():
    database = os.path.join(tempfile.mkdtemp(), "dag.db")
    config = DagConfig(checkpoint_db=database, max_workers=2)
    calls: list[str] = []
    state = {"explode": True}

    def fetch(ctx):
        calls.append("fetch")
        return {"rows": 3}

    def transform(ctx):
        calls.append("transform")
        return [1, 2, 3]

    def load(ctx):
        calls.append("load")
        if state["explode"]:
            raise AdapterError("warehouse unavailable")
        return "loaded"

    def build(nodes_fail: bool):
        return [
            Node("fetch", fetch),
            Node("transform", transform, ["fetch"]),
            Node("load", load, ["transform"],
                 retry=RetryPolicy(max_attempts=1)),
        ]

    first = DagExecutorComponent().execute(
        DagRequest(build(True), run_id="etl:2026-10-01", config=config)
    )
    assert first.status is RunStatus.FAILED and first.failed == ["load"]
    assert sorted(first.completed) == ["fetch", "transform"]
    assert calls == ["fetch", "transform", "load"]

    state["explode"] = False
    second = DagExecutorComponent().execute(
        DagRequest(build(False), run_id="etl:2026-10-01", config=config)
    )
    assert second.ok, second.render()
    assert sorted(second.replayed) == ["fetch", "transform"]
    assert second.completed == ["load"]
    assert calls == ["fetch", "transform", "load", "load"], (
        "completed nodes must not re-run: %s" % calls
    )
    assert second.results["transform"] == [1, 2, 3], "replayed value is restored"


def test_checkpointed_nodes_must_return_serialisable_values():
    database = os.path.join(tempfile.mkdtemp(), "dag.db")
    try:
        DagExecutorComponent().execute(
            DagRequest(
                [Node("bad", lambda ctx: object())],
                config=DagConfig(checkpoint_db=database),
            )
        )
    except TypeError as exc:
        assert "JSON" in str(exc)
    else:
        raise AssertionError("expected TypeError for an unstorable result")


def test_interrupted_non_idempotent_node_is_reported_not_retried():
    """Whether the side effect landed is unknowable from here, so the executor
    refuses to guess."""
    from toolkit.durable_steps import SqliteCheckpointStore, StepRecord, StepStatus

    database = os.path.join(tempfile.mkdtemp(), "dag.db")
    store = SqliteCheckpointStore(database)
    store.record_step(
        StepRecord("run", "charge", StepStatus.PENDING, 1, None, None)
    )
    store.close()

    calls: list[str] = []
    result = DagExecutorComponent().execute(
        DagRequest(
            [Node("charge", lambda ctx: calls.append("charge"), idempotent=False)],
            run_id="run",
            config=DagConfig(checkpoint_db=database),
        )
    )
    assert result.failed == ["charge"]
    assert "not idempotent" in result.outcomes["charge"].error
    assert calls == [], "must not re-run a possibly-completed side effect"


def test_render_summarises_a_run():
    result = DagExecutorComponent().execute(
        DagRequest([
            Node("ok", lambda ctx: 1),
            Node("bad", lambda ctx: (_ for _ in ()).throw(RuntimeError("x"))),
            Node("after", lambda ctx: 2, ["bad"]),
        ])
    )
    text = result.render()
    assert "completed" in text and "ok" in text
    assert "bad failed" in text
    assert "after skipped" in text
    assert result.seconds >= 0.0


def _main() -> int:
    functions = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    failures = 0
    lines = []
    for name, fn in functions:
        try:
            fn()
            lines.append("PASS " + name)
        except Exception as exc:  # noqa: BLE001 - runner
            failures += 1
            lines.append("FAIL " + name + ": " + repr(exc)[:300])
    lines.append("")
    lines.append(str(len(functions) - failures) + "/" + str(len(functions)) + " passed")
    sys.stdout.write("\n".join(lines) + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
