"""Data contracts for DAG execution."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..core.errors import ToolkitError


class NodeStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    """Never ran. Either a dependency failed, or `when` returned False.

    Distinct from FAILED on purpose. A node whose dependency failed did not go
    wrong — there was nothing for it to do. Collapsing the two turns one real
    failure into a page of red and buries the cause.
    """
    REPLAYED = "replayed"
    """Found already complete in the checkpoint store and not re-run."""


class RunStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"


class FailurePolicy(str, Enum):
    CONTINUE = "continue"
    """Keep running branches that do not depend on the failure. The default:
    in a fan-out over documents, one bad file should not abandon the other
    nine hundred."""
    FAIL_FAST = "fail_fast"
    """Stop scheduling new work after the first failure. In-flight nodes are
    still allowed to finish — cancelling mid-write is how partial state gets
    left behind."""


class DagError(ToolkitError):
    """The graph itself is unusable: a duplicate id, or a dependency on a node
    that was never declared. Raised before anything executes, because a
    half-run DAG is worse than a refused one."""


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 1
    initial_backoff: float = 0.5
    backoff_multiplier: float = 2.0
    max_backoff: float = 30.0
    jitter: float = 0.1

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")


@dataclass(frozen=True)
class Node:
    """One unit of work in the graph.

    `fn` receives a mapping of **its declared dependencies' results** plus the
    run's initial context — not everything completed so far. That is deliberate:
    reading a result you did not declare makes the node silently
    order-dependent, which works until the scheduler's timing changes and then
    fails in a way nobody can reproduce. Declaring the dependency is the fix,
    and restricting the context is what forces it.
    """

    id: str
    fn: Callable[[Mapping[str, Any]], Any]
    depends_on: Sequence[str] = ()
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    when: Callable[[Mapping[str, Any]], bool] | None = None
    """Optional guard evaluated just before the node runs, with the same context
    `fn` would receive. Returning False marks the node SKIPPED and skips its
    descendants — a conditional branch, not a failure."""
    idempotent: bool = True
    """False for a node with an external side effect that must not happen twice.
    With a checkpoint store, an interrupted non-idempotent node is reported
    rather than silently retried."""

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise ValueError("node id must not be empty")


@dataclass(frozen=True)
class NodeOutcome:
    id: str
    status: NodeStatus
    attempts: int = 0
    seconds: float = 0.0
    error: str = ""
    reason: str = ""
    """Why a node was skipped: which dependency failed, or that `when` was
    False. A skip with no reason is an unexplained gap in the run."""


@dataclass
class DagConfig:
    max_workers: int = 8
    on_failure: FailurePolicy = FailurePolicy.CONTINUE
    checkpoint_db: str | None = None
    """Path to a checkpoint database. When set, each node's result is recorded
    on success and replayed on a re-run, so a crash costs the nodes in flight
    rather than the whole graph. Results must then be JSON-serialisable."""
    node_timeout: float | None = None
    """Advisory only — see the README. A thread cannot be interrupted, so this
    bounds how long the scheduler *waits*, not how long a node runs."""

    def __post_init__(self) -> None:
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")


@dataclass
class DagRequest:
    nodes: Sequence[Node]
    run_id: str = "dag"
    """Checkpoint identity. Derive it from the work, not from a timestamp, or
    resumption never finds the previous attempt."""
    initial_context: Mapping[str, Any] = field(default_factory=dict)
    config: DagConfig = field(default_factory=DagConfig)


@dataclass
class DagResult:
    status: RunStatus
    outcomes: Mapping[str, NodeOutcome]
    results: Mapping[str, Any]
    """node id -> return value, for nodes that completed or were replayed."""
    layers: Sequence[Sequence[str]]
    """The execution plan. Layer count is the longest dependency chain, which is
    the floor on sequential rounds no matter how many workers are available."""
    critical_path: Sequence[str] = ()
    seconds: float = 0.0

    def _ids(self, status: NodeStatus) -> list[str]:
        return sorted(i for i, o in self.outcomes.items() if o.status is status)

    @property
    def completed(self) -> list[str]:
        return self._ids(NodeStatus.COMPLETED)

    @property
    def replayed(self) -> list[str]:
        return self._ids(NodeStatus.REPLAYED)

    @property
    def failed(self) -> list[str]:
        return self._ids(NodeStatus.FAILED)

    @property
    def skipped(self) -> list[str]:
        return self._ids(NodeStatus.SKIPPED)

    @property
    def ok(self) -> bool:
        return self.status is RunStatus.COMPLETED

    def render(self) -> str:
        lines = [
            "%-10s %s" % (status, ", ".join(getattr(self, status)) or "-")
            for status in ("completed", "replayed", "skipped", "failed")
        ]
        for outcome in sorted(self.outcomes.values(), key=lambda o: o.id):
            if outcome.status is NodeStatus.FAILED:
                lines.append("  %s failed after %d attempt(s): %s" % (
                    outcome.id, outcome.attempts, outcome.error))
            elif outcome.status is NodeStatus.SKIPPED and outcome.reason:
                lines.append("  %s skipped: %s" % (outcome.id, outcome.reason))
        return "\n".join(lines)
