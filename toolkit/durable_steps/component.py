"""Crash-resumable multi-step execution on a plain SQL checkpoint table.

Reimplemented from the durable-execution pattern that DBOS Transact, Temporal
and Hatchet all express differently: record each completed step's output, and on
restart replay the recorded outputs instead of re-running the work. That one idea
turns a fragile multi-step agent or ETL run into something you can kill and
restart without duplicating side effects.

What this deliberately is not: a distributed scheduler. There is no queue, no
worker pool, no cross-process signalling. It is the durability primitive alone,
which is the part that is hard to get right and the part that is reusable.

The lease is what makes it safe under more than one worker: a run is claimed for
a bounded window, and a second worker can only take over after that window
lapses. Without it, two workers resuming the same run would both re-execute the
same pending step.
"""
from __future__ import annotations

import json
import os
import random
import sqlite3
import threading
import time
import traceback
import uuid
from collections.abc import Mapping
from typing import Any

from .models import (
    CheckpointStore,
    NonRetryableError,
    RetryPolicy,
    RunStatus,
    Step,
    StepRecord,
    StepStatus,
    WorkflowRequest,
    WorkflowResult,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS durable_runs (
    run_id       TEXT PRIMARY KEY,
    status       TEXT NOT NULL,
    lease_owner  TEXT,
    lease_expiry REAL NOT NULL DEFAULT 0,
    updated_at   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS durable_steps (
    run_id      TEXT NOT NULL,
    name        TEXT NOT NULL,
    status      TEXT NOT NULL,
    attempts    INTEGER NOT NULL DEFAULT 0,
    result_json TEXT,
    error       TEXT,
    updated_at  REAL NOT NULL,
    PRIMARY KEY (run_id, name)
);
"""


class SqliteCheckpointStore:
    """Reference CheckpointStore. Durable on disk, usable in memory for tests.

    WAL is enabled so a reader does not block the writer, and a busy timeout is
    set because a lease contest is a normal event here, not an error.
    """

    def __init__(self, database: str = ":memory:") -> None:
        self._database = database
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(database, timeout=30, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            if database != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=30000")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def acquire_lease(self, run_id: str, owner: str, lease_seconds: float) -> bool:
        """Claim the run if it is unclaimed, already ours, or the lease expired.

        The whole claim is a single conditional UPDATE inside one transaction, so
        two workers racing cannot both see an expired lease and both win.
        """
        now = time.time()
        expiry = now + lease_seconds
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("BEGIN IMMEDIATE")
            try:
                cur.execute(
                    "SELECT status, lease_owner, lease_expiry FROM durable_runs"
                    " WHERE run_id = ?",
                    (run_id,),
                )
                row = cur.fetchone()
                if row is None:
                    cur.execute(
                        "INSERT INTO durable_runs"
                        " (run_id, status, lease_owner, lease_expiry, updated_at)"
                        " VALUES (?, ?, ?, ?, ?)",
                        (run_id, RunStatus.RUNNING.value, owner, expiry, now),
                    )
                    self._conn.commit()
                    return True
                if row["status"] == RunStatus.COMPLETED.value:
                    # Already finished: nothing to claim, and nothing to redo.
                    self._conn.commit()
                    return True
                held_by_other = (
                    row["lease_owner"] not in (None, owner)
                    and float(row["lease_expiry"]) > now
                )
                if held_by_other:
                    self._conn.commit()
                    return False
                cur.execute(
                    "UPDATE durable_runs SET status = ?, lease_owner = ?,"
                    " lease_expiry = ?, updated_at = ? WHERE run_id = ?",
                    (RunStatus.RUNNING.value, owner, expiry, now, run_id),
                )
                self._conn.commit()
                return True
            except Exception:
                self._conn.rollback()
                raise

    def release_lease(self, run_id: str, status: RunStatus) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE durable_runs SET status = ?, lease_owner = NULL,"
                " lease_expiry = 0, updated_at = ? WHERE run_id = ?",
                (status.value, time.time(), run_id),
            )
            self._conn.commit()

    def load_steps(self, run_id: str) -> Mapping[str, StepRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT run_id, name, status, attempts, result_json, error"
                " FROM durable_steps WHERE run_id = ?",
                (run_id,),
            ).fetchall()
        return {
            r["name"]: StepRecord(
                run_id=r["run_id"],
                name=r["name"],
                status=StepStatus(r["status"]),
                attempts=r["attempts"],
                result_json=r["result_json"],
                error=r["error"],
            )
            for r in rows
        }

    def record_step(self, record: StepRecord) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO durable_steps"
                " (run_id, name, status, attempts, result_json, error, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(run_id, name) DO UPDATE SET"
                " status = excluded.status, attempts = excluded.attempts,"
                " result_json = excluded.result_json, error = excluded.error,"
                " updated_at = excluded.updated_at",
                (
                    record.run_id,
                    record.name,
                    record.status.value,
                    record.attempts,
                    record.result_json,
                    record.error,
                    time.time(),
                ),
            )
            self._conn.commit()


class LeaseNotAcquired(RuntimeError):
    """Another worker holds an unexpired lease on this run."""


class StepNotReplayable(RuntimeError):
    """A non-idempotent step was interrupted mid-flight and cannot be retried
    automatically, because it may already have caused its side effect."""


class DurableStepsComponent:
    """Run a sequence of steps so that a crash costs at most one step.

    Re-invoking with the same `run_id` skips every step that already has a
    completed checkpoint and resumes at the first that does not.
    """

    def __init__(self, store: CheckpointStore | None = None, owner: str | None = None) -> None:
        self._store = store or SqliteCheckpointStore()
        # Default owner identifies the process, which is what a lease is really
        # scoped to.
        self._owner = owner or (str(os.getpid()) + "-" + uuid.uuid4().hex[:8])

    @property
    def store(self) -> CheckpointStore:
        return self._store

    def execute(self, input_data: WorkflowRequest) -> WorkflowResult:
        if not self._store.acquire_lease(
            input_data.run_id, self._owner, input_data.lease_seconds
        ):
            raise LeaseNotAcquired(
                "run " + input_data.run_id + " is leased by another worker"
            )

        records = dict(self._store.load_steps(input_data.run_id))
        context: dict[str, Any] = dict(input_data.initial_context)
        executed: list[str] = []
        replayed: list[str] = []

        for step in input_data.steps:
            existing = records.get(step.name)
            if existing is not None and existing.status is StepStatus.COMPLETED:
                context[step.name] = (
                    json.loads(existing.result_json)
                    if existing.result_json is not None
                    else None
                )
                replayed.append(step.name)
                continue

            # A non-idempotent step with attempts already recorded but no result
            # means the process died between starting it and checkpointing it.
            # Whether the side effect landed is unknowable from here, so stop
            # rather than guess.
            if (
                existing is not None
                and not step.idempotent
                and existing.attempts > 0
                and existing.status is not StepStatus.FAILED
            ):
                error = (
                    "step " + step.name + " was interrupted after "
                    + str(existing.attempts)
                    + " attempt(s) and is not idempotent; resolve manually"
                )
                self._store.release_lease(input_data.run_id, RunStatus.FAILED)
                raise StepNotReplayable(error)

            outcome = self._run_step(step, input_data.run_id, context, existing)
            if outcome[0] is False:
                self._store.release_lease(input_data.run_id, RunStatus.FAILED)
                return WorkflowResult(
                    run_id=input_data.run_id,
                    status=RunStatus.FAILED,
                    context=context,
                    executed=executed,
                    replayed=replayed,
                    failed_step=step.name,
                    error=str(outcome[1]),
                )
            context[step.name] = outcome[1]
            executed.append(step.name)

        self._store.release_lease(input_data.run_id, RunStatus.COMPLETED)
        return WorkflowResult(
            run_id=input_data.run_id,
            status=RunStatus.COMPLETED,
            context=context,
            executed=executed,
            replayed=replayed,
        )

    # --- step execution ---------------------------------------------------

    def _run_step(
        self,
        step: Step,
        run_id: str,
        context: Mapping[str, Any],
        existing: StepRecord | None,
    ) -> tuple[bool, Any]:
        attempts = existing.attempts if existing else 0
        last_error = existing.error if existing else None

        for attempt in range(1, step.retry.max_attempts + 1):
            attempts += 1
            if not step.idempotent:
                # Write the intent before doing the work, so an interruption is
                # detectable on the next replay.
                self._store.record_step(
                    StepRecord(run_id, step.name, StepStatus.PENDING, attempts, None, None)
                )
            try:
                result = step.fn(context)
            except NonRetryableError as exc:
                last_error = self._describe(exc)
                self._store.record_step(
                    StepRecord(
                        run_id, step.name, StepStatus.FAILED, attempts, None, last_error
                    )
                )
                return False, last_error
            except Exception as exc:  # noqa: BLE001 - the retry boundary
                last_error = self._describe(exc)
                self._store.record_step(
                    StepRecord(
                        run_id, step.name, StepStatus.PENDING, attempts, None, last_error
                    )
                )
                if attempt >= step.retry.max_attempts:
                    break
                time.sleep(self._backoff(step.retry, attempt))
                continue

            payload = self._serialise(step.name, result)
            self._store.record_step(
                StepRecord(run_id, step.name, StepStatus.COMPLETED, attempts, payload, None)
            )
            return True, result

        self._store.record_step(
            StepRecord(run_id, step.name, StepStatus.FAILED, attempts, None, last_error)
        )
        return False, last_error

    def _serialise(self, name: str, result: Any) -> str:
        try:
            return json.dumps(result)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "step '" + name + "' returned a value that is not JSON"
                " serialisable; durable steps must return storable results"
            ) from exc

    def _backoff(self, policy: RetryPolicy, attempt: int) -> float:
        delay = min(
            policy.initial_backoff * (policy.backoff_multiplier ** (attempt - 1)),
            policy.max_backoff,
        )
        if policy.jitter:
            delay *= 1.0 + random.uniform(-policy.jitter, policy.jitter)
        return max(0.0, delay)

    def _describe(self, exc: BaseException) -> str:
        detail = "".join(
            traceback.format_exception_only(type(exc), exc)
        ).strip()
        return detail or type(exc).__name__


__all__ = [
    "DurableStepsComponent",
    "SqliteCheckpointStore",
    "LeaseNotAcquired",
    "StepNotReplayable",
]
