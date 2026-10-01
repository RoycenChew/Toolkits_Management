"""Graph types.

Defines its own exceptions rather than importing `core.errors`, which keeps this
unit `copy_tier: standalone` — graph algorithms are the most portable thing in
the toolkit, and making them depend on a document-oriented error taxonomy would
be a poor trade for three lines of reuse.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass


class GraphError(Exception):
    """Base class for graph problems."""


class CycleError(GraphError):
    """A cycle was found where a DAG was required.

    Carries the actual cycle, because "your graph has a cycle" is not actionable
    on a graph of two hundred nodes. The node list is in traversal order and
    repeats its first element last, so it reads as a loop.
    """

    def __init__(self, cycle: Iterable[str]) -> None:
        self.cycle = list(cycle)
        super().__init__("cycle: " + " -> ".join(self.cycle))


class NodeNotFound(GraphError):
    def __init__(self, node: str) -> None:
        self.node = node
        super().__init__("no such node: " + repr(node))


@dataclass(frozen=True)
class Graph:
    """A directed graph, immutable.

    Edges point from a node to its **successors**. Task graphs are more often
    *declared* the other way round — as each task's dependencies — so
    `from_dependencies` exists and is usually the constructor you want. Mixing
    the two directions up is the easiest way to get a reversed execution order
    that still looks plausible.

    Every traversal sorts, so results are reproducible run to run. That matters
    more than it sounds: a non-deterministic topological order makes an
    execution log impossible to diff and a flaky test impossible to attribute.
    """

    nodes: frozenset[str]
    successors: Mapping[str, frozenset[str]]

    def __post_init__(self) -> None:
        unknown = {
            target for sources in self.successors.values() for target in sources
        } - set(self.nodes)
        if unknown:
            raise NodeNotFound(sorted(unknown)[0])

    # --- construction -----------------------------------------------------

    @staticmethod
    def from_successors(mapping: Mapping[str, Iterable[str]]) -> Graph:
        """Build from `node -> nodes it points at`.

        Nodes mentioned only as targets are added automatically, so a caller
        need not declare leaves separately.
        """
        nodes: set[str] = set(mapping)
        for targets in mapping.values():
            nodes.update(targets)
        return Graph(
            nodes=frozenset(nodes),
            successors={
                node: frozenset(mapping.get(node, ())) for node in sorted(nodes)
            },
        )

    @staticmethod
    def from_dependencies(mapping: Mapping[str, Iterable[str]]) -> Graph:
        """Build from `node -> nodes it depends on`.

        The natural way to declare a pipeline: each step names what it needs.
        Edges end up pointing dependency to dependent, which is execution order.
        """
        nodes: set[str] = set(mapping)
        for sources in mapping.values():
            nodes.update(sources)
        successors: dict[str, set[str]] = {node: set() for node in nodes}
        for node, sources in mapping.items():
            for source in sources:
                successors[source].add(node)
        return Graph(
            nodes=frozenset(nodes),
            successors={node: frozenset(successors[node]) for node in sorted(nodes)},
        )

    # --- queries ----------------------------------------------------------

    def out_edges(self, node: str) -> list[str]:
        if node not in self.nodes:
            raise NodeNotFound(node)
        return sorted(self.successors.get(node, frozenset()))

    def in_edges(self, node: str) -> list[str]:
        if node not in self.nodes:
            raise NodeNotFound(node)
        return sorted(
            source for source, targets in self.successors.items() if node in targets
        )

    def in_degree(self, node: str) -> int:
        return len(self.in_edges(node))

    def out_degree(self, node: str) -> int:
        return len(self.out_edges(node))

    @property
    def edge_count(self) -> int:
        return sum(len(targets) for targets in self.successors.values())

    def roots(self) -> list[str]:
        """Nodes with no dependencies — where execution starts."""
        return sorted(node for node in self.nodes if self.in_degree(node) == 0)

    def leaves(self) -> list[str]:
        """Nodes nothing depends on — where execution ends."""
        return sorted(node for node in self.nodes if self.out_degree(node) == 0)

    def reverse(self) -> Graph:
        reversed_edges: dict[str, set[str]] = {node: set() for node in self.nodes}
        for source, targets in self.successors.items():
            for target in targets:
                reversed_edges[target].add(source)
        return Graph(
            nodes=self.nodes,
            successors={
                node: frozenset(targets) for node, targets in reversed_edges.items()
            },
        )

    def subgraph(self, keep: Iterable[str]) -> Graph:
        retained = frozenset(keep) & self.nodes
        return Graph(
            nodes=retained,
            successors={
                node: frozenset(self.successors.get(node, frozenset())) & retained
                for node in retained
            },
        )

    def __len__(self) -> int:
        return len(self.nodes)

    def __contains__(self, node: object) -> bool:
        return node in self.nodes


__all__ = ["CycleError", "Graph", "GraphError", "NodeNotFound"]
