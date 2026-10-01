"""Graph algorithms, all pure and all deterministic.

Three properties are deliberate throughout:

**Deterministic.** Every choice point sorts. Two runs over the same graph give
identical output, so an execution log can be diffed and a failure attributed.
Non-determinism here would surface as flaky tests in whatever consumes it.

**Iterative, not recursive.** Tarjan's SCC and the depth-first traversals are
written with explicit stacks. A 2,000-node dependency chain is an ordinary
monorepo and would blow CPython's default recursion limit, which is the kind of
failure that only appears on real input.

**Actionable errors.** `topological_order` raises `CycleError` carrying the
actual cycle rather than reporting that one exists. On a large graph the
difference is between a five-minute fix and an afternoon.
"""
from __future__ import annotations

import heapq
from collections.abc import Callable, Mapping, Sequence

from .models import CycleError, Graph, NodeNotFound


def topological_order(graph: Graph) -> list[str]:
    """Nodes in dependency order: every node after everything it depends on.

    Kahn's algorithm with a sorted ready-set, so the order is stable.

    Raises:
        CycleError: carrying one concrete cycle.
    """
    indegree = {node: graph.in_degree(node) for node in graph.nodes}
    ready = sorted(node for node, degree in indegree.items() if degree == 0)
    order: list[str] = []

    while ready:
        node = ready.pop(0)
        order.append(node)
        for successor in graph.out_edges(node):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                # Insert in sorted position rather than append-then-sort: the
                # ready set stays small and this keeps the order stable.
                position = 0
                while position < len(ready) and ready[position] < successor:
                    position += 1
                ready.insert(position, successor)

    if len(order) != len(graph.nodes):
        cycle = find_cycle(graph)
        raise CycleError(cycle or sorted(set(graph.nodes) - set(order)))
    return order


def topological_layers(graph: Graph) -> list[list[str]]:
    """Generations: each layer depends only on earlier layers.

    This is the function a scheduler wants. Everything within a layer is
    mutually independent and can run in parallel, and the layer count is the
    longest dependency chain — so it is also the minimum number of sequential
    rounds any execution needs, however many workers you have.

    Raises:
        CycleError: if the graph is not a DAG.
    """
    indegree = {node: graph.in_degree(node) for node in graph.nodes}
    layers: list[list[str]] = []
    remaining = dict(indegree)

    while remaining:
        layer = sorted(node for node, degree in remaining.items() if degree == 0)
        if not layer:
            cycle = find_cycle(graph.subgraph(remaining))
            raise CycleError(cycle or sorted(remaining))
        layers.append(layer)
        for node in layer:
            del remaining[node]
            for successor in graph.out_edges(node):
                if successor in remaining:
                    remaining[successor] -= 1
    return layers


def find_cycle(graph: Graph) -> list[str] | None:
    """One concrete cycle as a node list that repeats its first element, or None.

    Iterative depth-first search with an explicit colour map. Returns the first
    cycle found in sorted traversal order, which makes the result stable rather
    than merely correct.
    """
    WHITE, GREY, BLACK = 0, 1, 2
    colour = dict.fromkeys(graph.nodes, WHITE)

    for start in sorted(graph.nodes):
        if colour[start] != WHITE:
            continue
        stack: list[tuple[str, list[str]]] = [(start, [])]
        path: list[str] = []
        while stack:
            node, pending = stack[-1]
            if colour[node] == WHITE:
                colour[node] = GREY
                path.append(node)
                stack[-1] = (node, graph.out_edges(node))
                continue
            if pending:
                successor = pending.pop(0)
                if colour.get(successor) == GREY:
                    index = path.index(successor)
                    return path[index:] + [successor]
                if colour.get(successor) == WHITE:
                    stack.append((successor, []))
                continue
            colour[node] = BLACK
            if path and path[-1] == node:
                path.pop()
            stack.pop()
    return None


def is_dag(graph: Graph) -> bool:
    return find_cycle(graph) is None


def strongly_connected_components(graph: Graph) -> list[list[str]]:
    """Tarjan's SCC, iterative. Components in reverse topological order.

    On a DAG every component is a single node, so this is mainly a diagnostic:
    a component with more than one member *is* a dependency cycle, and seeing
    all of them at once is more useful than fixing one at a time.
    """
    index_of: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: dict[str, bool] = {}
    stack: list[str] = []
    components: list[list[str]] = []
    counter = 0

    for root in sorted(graph.nodes):
        if root in index_of:
            continue
        work: list[tuple[str, list[str]]] = [(root, graph.out_edges(root))]
        index_of[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack[root] = True

        while work:
            node, pending = work[-1]
            if pending:
                successor = pending.pop(0)
                if successor not in index_of:
                    index_of[successor] = low[successor] = counter
                    counter += 1
                    stack.append(successor)
                    on_stack[successor] = True
                    work.append((successor, graph.out_edges(successor)))
                elif on_stack.get(successor):
                    low[node] = min(low[node], index_of[successor])
                continue

            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index_of[node]:
                component: list[str] = []
                while True:
                    member = stack.pop()
                    on_stack[member] = False
                    component.append(member)
                    if member == node:
                        break
                components.append(sorted(component))
    return components


def descendants(graph: Graph, node: str) -> set[str]:
    """Everything reachable from `node`, excluding itself.

    The set a scheduler must skip when a node fails: they depend on output that
    does not exist.
    """
    if node not in graph.nodes:
        raise NodeNotFound(node)
    seen: set[str] = set()
    frontier = list(graph.out_edges(node))
    while frontier:
        current = frontier.pop()
        if current in seen:
            continue
        seen.add(current)
        frontier.extend(graph.out_edges(current))
    return seen


def ancestors(graph: Graph, node: str) -> set[str]:
    """Everything `node` transitively depends on, excluding itself."""
    return descendants(graph.reverse(), node)


def reachable_from(graph: Graph, node: str) -> set[str]:
    """`node` plus all its descendants."""
    return {node} | descendants(graph, node)


def shortest_path(
    graph: Graph,
    start: str,
    goal: str,
    weight: Callable[[str, str], float] | None = None,
) -> list[str] | None:
    """Cheapest path, or None if unreachable.

    Dijkstra when `weight` is supplied, breadth-first otherwise — unweighted
    breadth-first is both faster and exactly correct, so paying for a heap when
    every edge costs 1 would be waste.

    Negative weights are rejected rather than silently mishandled; Dijkstra's
    correctness depends on their absence and failing loudly beats a wrong path.
    """
    for node in (start, goal):
        if node not in graph.nodes:
            raise NodeNotFound(node)
    if start == goal:
        return [start]

    if weight is None:
        previous: dict[str, str] = {}
        seen = {start}
        queue = [start]
        while queue:
            node = queue.pop(0)
            for successor in graph.out_edges(node):
                if successor in seen:
                    continue
                seen.add(successor)
                previous[successor] = node
                if successor == goal:
                    return _rebuild(previous, start, goal)
                queue.append(successor)
        return None

    distance = {start: 0.0}
    previous = {}
    heap: list[tuple[float, str]] = [(0.0, start)]
    settled: set[str] = set()
    while heap:
        cost, node = heapq.heappop(heap)
        if node in settled:
            continue
        settled.add(node)
        if node == goal:
            return _rebuild(previous, start, goal)
        for successor in graph.out_edges(node):
            edge = weight(node, successor)
            if edge < 0:
                raise ValueError(
                    "negative weight on %s -> %s; Dijkstra requires non-negative"
                    % (node, successor)
                )
            candidate = cost + edge
            if candidate < distance.get(successor, float("inf")):
                distance[successor] = candidate
                previous[successor] = node
                heapq.heappush(heap, (candidate, successor))
    return None


def _rebuild(previous: Mapping[str, str], start: str, goal: str) -> list[str]:
    path = [goal]
    while path[-1] != start:
        path.append(previous[path[-1]])
    path.reverse()
    return path


def critical_path(
    graph: Graph, duration: Mapping[str, float] | None = None
) -> tuple[float, list[str]]:
    """Longest-duration path through a DAG: total cost and the path itself.

    The lower bound on wall-clock time no amount of parallelism removes, and the
    chain to attack if the pipeline is too slow. With no durations every node
    counts as 1, so the result is the longest dependency chain — the same number
    as `len(topological_layers(graph))`.

    Raises:
        CycleError: if the graph is not a DAG. "Longest path" is unbounded on a
            cyclic graph, so there is no sensible answer to return.
    """
    order = topological_order(graph)
    costs = duration or {}
    best: dict[str, float] = {}
    came_from: dict[str, str] = {}

    for node in order:
        own = float(costs.get(node, 1.0))
        incoming = graph.in_edges(node)
        if not incoming:
            best[node] = own
            continue
        predecessor = max(incoming, key=lambda n: (best[n], n))
        best[node] = best[predecessor] + own
        came_from[node] = predecessor

    if not best:
        return 0.0, []
    end = max(sorted(best), key=lambda n: best[n])
    path = [end]
    while path[-1] in came_from:
        path.append(came_from[path[-1]])
    path.reverse()
    return best[end], path


def transitive_reduction(graph: Graph) -> Graph:
    """Remove edges implied by a longer path, preserving reachability.

    `a -> b`, `b -> c` and `a -> c` means the third edge says nothing: `c`
    already waits for `a` through `b`. Dropping it makes a declared dependency
    graph readable, and makes it obvious which dependencies are real.

    Raises:
        CycleError: reduction is only well-defined on a DAG.
    """
    topological_order(graph)  # raises on a cycle
    reachability = {node: descendants(graph, node) for node in graph.nodes}
    reduced: dict[str, set[str]] = {}
    for node in sorted(graph.nodes):
        direct = set(graph.out_edges(node))
        implied = {
            far
            for near in direct
            for far in reachability[near]
        }
        reduced[node] = direct - implied
    return Graph(
        nodes=graph.nodes,
        successors={node: frozenset(targets) for node, targets in reduced.items()},
    )


def validate_dag(graph: Graph, required: Sequence[str] | None = None) -> None:
    """Raise unless the graph is a usable DAG.

    Convenience for a scheduler's entry point: one call that produces an
    actionable error instead of a partial execution.

    Raises:
        CycleError: the graph has a cycle, named.
        NodeNotFound: a required node is absent.
    """
    for node in required or ():
        if node not in graph.nodes:
            raise NodeNotFound(node)
    cycle = find_cycle(graph)
    if cycle:
        raise CycleError(cycle)


__all__ = [
    "ancestors",
    "critical_path",
    "descendants",
    "find_cycle",
    "is_dag",
    "reachable_from",
    "shortest_path",
    "strongly_connected_components",
    "topological_layers",
    "topological_order",
    "transitive_reduction",
    "validate_dag",
]
