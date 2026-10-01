# Graph Component

**Layer 2 · imports nothing · `copy_tier: standalone`**

## What It Does

Deterministic graph algorithms: topological order and layers, cycle reporting, strongly
connected components, reachability, shortest and critical path, transitive reduction.

## Why It Is Useful

Anything that runs steps in a declared order needs these: `dag` schedules with them,
and the toolkit's own layer rule is an acyclicity check. The algorithms are the part
nobody should rewrite -- well understood, easy to get subtly wrong, and `networkx` is
a large dependency for the ten functions anyone actually uses.

Three properties are the reason to own this rather than reach for a snippet:

**Deterministic.** Every choice point sorts, so two runs over the same graph give
byte-identical output. That sounds cosmetic until an execution log needs diffing or a
flaky test needs attributing — a non-deterministic topological order makes both
impossible.

**Iterative, not recursive.** Tarjan's SCC and the depth-first traversals use explicit
stacks. A 2,000-node dependency chain is an ordinary monorepo and blows CPython's
recursion limit, which is a failure that only ever appears on real input. There is a
test for exactly that size.

**Actionable errors.** `topological_order` raises `CycleError` carrying the cycle
itself — `a -> b -> c -> a` — not a boolean. On a 200-node graph that is the difference
between a five-minute fix and an afternoon.

## Architecture

```
Graph.from_dependencies({"test": ["build"], ...})   each node names what it needs
Graph.from_successors({"build": ["test"], ...})     each node names what follows
   |
   ├── topological_order   -> list, dependency order, raises CycleError
   ├── topological_layers  -> list[list], each layer runs in parallel
   ├── find_cycle          -> the actual cycle, or None
   ├── strongly_connected_components -> cycles, all of them at once
   ├── descendants / ancestors / reachable_from
   ├── shortest_path       -> BFS, or Dijkstra when weighted
   ├── critical_path       -> (cost, path), the floor on wall-clock time
   └── transitive_reduction -> drops edges a longer path already implies
```

```
stdlib only. no numpy, no networkx, no C extension.
```

## Installation

Standalone — copy one directory:

```bash
cp -r toolkit/graph your_project/
```

Python 3.10+. Standard library only. Verified by `test_packaging.py`, which imports
this directory as a bare top-level package in a subprocess with the repository off
`sys.path`.

It defines its own `GraphError` / `CycleError` / `NodeNotFound` rather than importing
`core.errors`, which is what keeps it standalone — graph algorithms are the most
portable thing in the toolkit, and coupling them to a document-oriented error taxonomy
would be a poor trade for three lines of reuse.

## Input / Output

`Graph` is frozen. Two constructors, and **picking the wrong one gives a reversed
execution order that still looks plausible**:

| Constructor | Mapping means | Use when |
|---|---|---|
| `from_dependencies` | node → what it **needs** | declaring a pipeline. Usually this one |
| `from_successors` | node → what **follows** it | you already have adjacency |

Nodes mentioned only as targets are added automatically, so leaves need no declaration.

| Function | Returns | Raises |
|---|---|---|
| `topological_order(g)` | `list[str]`, stable | `CycleError` with the cycle |
| `topological_layers(g)` | `list[list[str]]` | `CycleError` |
| `find_cycle(g)` | `list[str]` ending where it began, or `None` | — |
| `is_dag(g)` | `bool` | — |
| `strongly_connected_components(g)` | `list[list[str]]`, reverse topological | — |
| `descendants(g, n)` / `ancestors(g, n)` | `set[str]`, excluding `n` | `NodeNotFound` |
| `shortest_path(g, a, b, weight=None)` | `list[str]` or `None` | `NodeNotFound`, `ValueError` on a negative weight |
| `critical_path(g, duration=None)` | `(float, list[str])` | `CycleError` |
| `transitive_reduction(g)` | `Graph` | `CycleError` |
| `validate_dag(g, required=None)` | `None` | `CycleError`, `NodeNotFound` |

`Graph` also offers `out_edges`, `in_edges`, `in_degree`, `out_degree`, `roots`,
`leaves`, `reverse`, `subgraph`, `edge_count`, `len()` and `in`.

## Usage

```python
from toolkit.graph import Graph, critical_path, topological_layers

build = Graph.from_dependencies({
    "compile": ["checkout"],
    "unit_tests": ["compile"],
    "lint": ["checkout"],
    "package": ["unit_tests", "lint"],
})

for layer in topological_layers(build):
    print("run in parallel:", layer)        # ['checkout'] / ['compile', 'lint'] / ...

cost, path = critical_path(build, {"unit_tests": 20.0, "compile": 5.0})
print(cost, path)        # the chain to attack if the pipeline is too slow
```

Diagnosing a cycle:

```python
from toolkit.graph import CycleError, strongly_connected_components, topological_order

try:
    topological_order(graph)
except CycleError as exc:
    print(" -> ".join(exc.cycle))          # a -> b -> c -> a
    # Every cycle at once, rather than fixing them one at a time:
    for component in strongly_connected_components(graph):
        if len(component) > 1:
            print("tangled:", component)
```

`python examples/cookbook.py graph` runs a worked example.

## Limitations

- **Unweighted except in `shortest_path`.** `critical_path` takes per-node durations,
  but there is no general edge-weight model.
- **Immutable.** No `add_node` or `remove_edge`; build a new `Graph`. Fine for
  declared pipelines, wrong for an incrementally-built graph of any size.
- **`transitive_reduction` is O(V·E)** — it computes full reachability. Comfortable to
  a few thousand nodes, not a million.
- **No serialisation.** No DOT, JSON or GraphML output, so no visualisation.
- Node ids are strings. Attach your own payload in a dict beside the graph;
  `dag.Node` does exactly that.
- `shortest_path` rejects negative weights rather than switching to Bellman-Ford.
  Failing loudly beats a silently wrong path.
- No flow, matching, colouring or spanning-tree algorithms. Added when a consumer
  needs one, per playbook rule 5.

## Integration Guide

1. **Use `from_dependencies`** unless you already hold adjacency. It matches how
   pipelines are declared and avoids the reversed-order trap.
2. Call `validate_dag` once at your entry point. One actionable error beats a
   half-finished traversal.
3. Use `topological_layers` to *plan* and report, but schedule from a ready queue —
   see `dag`'s README for why lockstep layers waste time.
4. When a cycle appears in generated graphs, reach for
   `strongly_connected_components` rather than `find_cycle`: it shows every tangle at
   once.
5. `critical_path` is the question "why is this slow?" — the answer is never the sum
   of all node durations, it is this chain.

## Extraction Notes

- **Preserved:** the textbook algorithms — Kahn's topological sort, Tarjan's SCC,
  Dijkstra, DAG longest path. Named so they can be verified against their sources.
- **Removed:** nothing; written from the algorithms, not adapted from a library.
- **Added:** determinism everywhere, iterative rewrites of the recursive ones, cycles
  reported as data rather than as an exception message, and the
  `from_dependencies` / `from_successors` split that names the direction explicitly.
- **Isolated:** imports nothing at all, including from `core`.
