"""Data contracts for the durable step execution component."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class StepStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    initial_backoff: float = 0.5
    backoff_multiplier: float = 2.0
    max_backoff: float = 30.0
    jitter: float = 0.1
    """Fraction of the delay to randomise, so retrying workers do not
    synchronise into a thundering herd."""

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.initial_backoff < 0:
            raise ValueError("initial_backoff must be non-negative")


@dataclass(frozen=True)
class Step:
    """One unit of work that is checkpointed on success.

    `fn` receives the accumulated context (step name -> that step's result) and
    returns a JSON-serialisable value. That constraint is the price of
    durability: whatever the step returns must survive a process restart, so it
    has to be storable.

    Set `idempotent=False` for a step with an external side effect that must not
    happen twice (charging a card, sending an email). Such a step records an
    intent marker before running, so a crash mid-step is detected on replay and
    surfaced rather than silently retried.
    """

    name: str
    fn: Callable[[Mapping[str, Any]], Any]
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    idempotent: bool = True

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("step name must be non-empty")


@dataclass(frozen=True)
class StepRecord:
    run_id: str
    name: str
    status: StepStatus
    attempts: int
    result_json: str | None
    error: str | None


@dataclass
class WorkflowRequest:
    run_id: str
    """Caller-supplied idempotency key. Re-running with the same key resumes
    instead of starting over. Derive it from the business event, not from a
    timestamp or a uuid4, or you lose the resumption guarantee."""
    steps: Sequence[Step]
    initial_context: Mapping[str, Any] = field(default_factory=dict)
    lease_seconds: float = 300.0
    """How long this worker claims the run. Another worker may take it over only
    after the lease expires."""

    def __post_init__(self) -> None:
        names = [s.name for s in self.steps]
        if len(names) != len(set(names)):
            raise ValueError("step names must be unique within a workflow")
        if not self.steps:
            raise ValueError("at least one step is required")


@dataclass
class WorkflowResult:
    run_id: str
    status: RunStatus
    context: Mapping[str, Any]
    """step name -> result, including results replayed from the store."""
    executed: Sequence[str]
    """Steps that actually ran in this invocation."""
    replayed: Sequence[str]
    """Steps skipped because a checkpoint already existed."""
    failed_step: str | None = None
    error: str | None = None


class NonRetryableError(Exception):
    """Raise inside a step to fail the run immediately, skipping all retries.

    Use it for anything a retry cannot fix: a validation failure, a 404, a
    malformed input. Retrying those burns the budget and delays the real signal.
    """


class CheckpointStore(Protocol):
    """Persistence for run and step state.

    The only real requirement is that `acquire_lease` and `record_step` are
    atomic. A SQLite implementation ships with the component; a Postgres one is
    the same SQL with `ON CONFLICT` and a `FOR UPDATE SKIP LOCKED` lease claim.
    """

    def acquire_lease(self, run_id: str, owner: str, lease_seconds: float) -> bool: ...

    def release_lease(self, run_id: str, status: RunStatus) -> None: ...

    def load_steps(self, run_id: str) -> Mapping[str, StepRecord]: ...

    def record_step(self, record: StepRecord) -> None: ...
