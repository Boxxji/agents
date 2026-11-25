"""Core schemas, deterministic idempotency keys, and exceptions for durable workflow replay.

Mirrors the canonical Hanzo control-plane durable execution contract
(``control-plane/internal/durable/``) and ``hanzo_agents.agent_workflow`` lifecycle:
- Deterministic idempotency keys: ``sha256(org:run_id:round:step_id)``
- Append-only event taxonomy: ``WorkflowStarted``, ``ActivityScheduled``,
  ``ActivityCompleted``, ``ActivityFailed``, ``CheckpointSaved``,
  ``WorkflowSuspended``, ``WorkflowResumed``, ``WorkflowCompleted``, ``WorkflowFailed``
- Zero always-on daemon or background polling processes.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class DurableReplayError(Exception):
    """Base error for durable workflow replay failures."""


class InvalidIdentifierError(DurableReplayError):
    """Raised when an org, workflow_id, run_id, actor_id, or step_id is unsafe or empty."""


class NonDeterministicReplayError(DurableReplayError):
    """Raised fail-closed when workflow replay diverges from persisted history."""


class TenantIsolationError(DurableReplayError):
    """Raised when an activation violates tenant, actor, or workflow ownership boundaries."""


class WorkflowSuspendedSignal(Exception):
    """Control-flow signal indicating a clean event-driven suspension awaiting a signal."""

    def __init__(self, signal_name: str, step_id: str) -> None:
        self.signal_name = signal_name
        self.step_id = step_id
        super().__init__(
            f"Workflow suspended awaiting signal {signal_name!r} at step {step_id!r}"
        )


class EventType(str, Enum):
    """Append-only durable workflow journal event taxonomy."""

    WORKFLOW_STARTED = "WorkflowStarted"
    ACTIVITY_SCHEDULED = "ActivityScheduled"
    ACTIVITY_COMPLETED = "ActivityCompleted"
    ACTIVITY_FAILED = "ActivityFailed"
    CHECKPOINT_SAVED = "CheckpointSaved"
    WORKFLOW_SUSPENDED = "WorkflowSuspended"
    WORKFLOW_RESUMED = "WorkflowResumed"
    WORKFLOW_COMPLETED = "WorkflowCompleted"
    WORKFLOW_FAILED = "WorkflowFailed"


class RunStatus(str, Enum):
    """Persisted lifecycle status for a workflow run."""

    PENDING = "pending"
    RUNNING = "running"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    FAILED = "failed"


class BackendKind(str, Enum):
    """Checkpoint persistence backend under evaluation."""

    SQLITE_WAL = "sqlite_wal"
    ATOMIC_JSONL = "atomic_jsonl"
    EPHEMERAL_BASELINE = "ephemeral_baseline"


class CrashMode(str, Enum):
    """Subprocess hard-crash injection point (``os._exit(99)``)."""

    NONE = "none"
    BEFORE_ACTIVITY_COMMIT = "before_activity_commit"
    AFTER_EFFECT_BEFORE_ACK = "after_effect_before_ack"
    AFTER_STEP_COMMIT = "after_step_commit"


CRASH_EXIT_CODE = 99

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}$")


def validate_identifier(field_name: str, value: str) -> str:
    """Validate an identifier against empty values, control chars, and path traversal."""
    if not isinstance(value, str):
        raise InvalidIdentifierError(f"{field_name} must be a string")
    stripped = value.strip()
    if not stripped or stripped != value:
        raise InvalidIdentifierError(
            f"{field_name} must be non-empty without leading/trailing whitespace"
        )
    if ".." in value or "/" in value or "\\" in value or "\x00" in value:
        raise InvalidIdentifierError(
            f"{field_name} contains forbidden path-traversal or separator characters: {value!r}"
        )
    if not _SAFE_ID_RE.match(value):
        raise InvalidIdentifierError(
            f"{field_name} contains disallowed characters: {value!r}"
        )
    return value


def canonical_json(data: Any) -> str:
    """Serialize data deterministically for content hashing and wire persistence."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def hash_payload(data: Any) -> str:
    """Compute SHA-256 hex digest of a canonicalized JSON payload."""
    encoded = canonical_json(data).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compute_idempotency_key(org: str, run_id: str, round_idx: int, step_id: str) -> str:
    """Compute ``sha256(org:run_id:round:step_id)`` matching ``control-plane/internal/durable``."""
    org_clean = validate_identifier("org", org)
    run_clean = validate_identifier("run_id", run_id)
    step_clean = validate_identifier("step_id", step_id)
    if not isinstance(round_idx, int) or isinstance(round_idx, bool) or round_idx < 0:
        raise InvalidIdentifierError(
            f"round_idx must be a non-negative integer, got {round_idx!r}"
        )
    material = f"{org_clean}:{run_clean}:{round_idx}:{step_clean}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


@dataclass(frozen=True)
class StepSpec:
    """Declarative specification of a single durable workflow activity step."""

    step_id: str
    activity_type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    round_idx: int = 0
    await_signal: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "activity_type": self.activity_type,
            "payload": self.payload,
            "round_idx": self.round_idx,
            "await_signal": self.await_signal,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "StepSpec":
        return cls(
            step_id=str(raw["step_id"]),
            activity_type=str(raw["activity_type"]),
            payload=dict(raw.get("payload") or {}),
            round_idx=int(raw.get("round_idx", 0)),
            await_signal=raw.get("await_signal"),
        )


@dataclass(frozen=True)
class CrashPlan:
    """Deterministic hard-crash specification for a single subprocess attempt."""

    target_step_index: int = -1
    mode: CrashMode = CrashMode.NONE
    enable_idempotency_guard: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_step_index": self.target_step_index,
            "mode": self.mode.value,
            "enable_idempotency_guard": self.enable_idempotency_guard,
        }

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "CrashPlan":
        if not raw:
            return cls()
        return cls(
            target_step_index=int(raw.get("target_step_index", -1)),
            mode=CrashMode(raw.get("mode", CrashMode.NONE.value)),
            enable_idempotency_guard=bool(raw.get("enable_idempotency_guard", True)),
        )


@dataclass
class JournalEvent:
    """Immutable append-only event persisted in the workflow journal."""

    sequence: int
    previous_sequence: int
    event_type: EventType
    org: str
    workflow_id: str
    run_id: str
    execution_id: str
    actor_id: str
    round_idx: int
    step_id: str = ""
    activity_type: str = ""
    idempotency_key: str = ""
    input_hash: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    pid: int = 0
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence": self.sequence,
            "previous_sequence": self.previous_sequence,
            "event_type": self.event_type.value,
            "org": self.org,
            "workflow_id": self.workflow_id,
            "run_id": self.run_id,
            "execution_id": self.execution_id,
            "actor_id": self.actor_id,
            "round_idx": self.round_idx,
            "step_id": self.step_id,
            "activity_type": self.activity_type,
            "idempotency_key": self.idempotency_key,
            "input_hash": self.input_hash,
            "payload": self.payload,
            "error": self.error,
            "pid": self.pid,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "JournalEvent":
        return cls(
            sequence=int(raw["sequence"]),
            previous_sequence=int(raw["previous_sequence"]),
            event_type=EventType(raw["event_type"]),
            org=str(raw["org"]),
            workflow_id=str(raw["workflow_id"]),
            run_id=str(raw["run_id"]),
            execution_id=str(raw["execution_id"]),
            actor_id=str(raw["actor_id"]),
            round_idx=int(raw.get("round_idx", 0)),
            step_id=str(raw.get("step_id", "")),
            activity_type=str(raw.get("activity_type", "")),
            idempotency_key=str(raw.get("idempotency_key", "")),
            input_hash=str(raw.get("input_hash", "")),
            payload=dict(raw.get("payload") or {}),
            error=str(raw.get("error", "")),
            pid=int(raw.get("pid", 0)),
            timestamp=str(raw.get("timestamp", "")),
        )


@dataclass
class StepSnapshot:
    """Persisted step-boundary state snapshot for fast recovery inspection."""

    org: str
    run_id: str
    round_idx: int
    step_id: str
    sequence: int
    state: Dict[str, Any]
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "org": self.org,
            "run_id": self.run_id,
            "round_idx": self.round_idx,
            "step_id": self.step_id,
            "sequence": self.sequence,
            "state": self.state,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "StepSnapshot":
        return cls(
            org=str(raw["org"]),
            run_id=str(raw["run_id"]),
            round_idx=int(raw.get("round_idx", 0)),
            step_id=str(raw["step_id"]),
            sequence=int(raw["sequence"]),
            state=dict(raw.get("state") or {}),
            created_at=str(raw.get("created_at", "")),
        )


@dataclass
class LedgerRecord:
    """Transactional idempotency ledger entry indexed by ``(org, idempotency_key)``."""

    idempotency_key: str
    org: str
    run_id: str
    round_idx: int
    step_id: str
    activity_type: str
    status: str
    input_hash: str
    output: Dict[str, Any]
    error: str = ""
    attempt: int = 1
    pid: int = 0
    completed_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "idempotency_key": self.idempotency_key,
            "org": self.org,
            "run_id": self.run_id,
            "round_idx": self.round_idx,
            "step_id": self.step_id,
            "activity_type": self.activity_type,
            "status": self.status,
            "input_hash": self.input_hash,
            "output": self.output,
            "error": self.error,
            "attempt": self.attempt,
            "pid": self.pid,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "LedgerRecord":
        return cls(
            idempotency_key=str(raw["idempotency_key"]),
            org=str(raw["org"]),
            run_id=str(raw["run_id"]),
            round_idx=int(raw.get("round_idx", 0)),
            step_id=str(raw["step_id"]),
            activity_type=str(raw["activity_type"]),
            status=str(raw["status"]),
            input_hash=str(raw.get("input_hash", "")),
            output=dict(raw.get("output") or {}),
            error=str(raw.get("error", "")),
            attempt=int(raw.get("attempt", 1)),
            pid=int(raw.get("pid", 0)),
            completed_at=str(raw.get("completed_at", "")),
        )


@dataclass
class ActivationTelemetry:
    """Measured telemetry from a single stateless workflow activation."""

    org: str
    workflow_id: str
    run_id: str
    execution_id: str
    actor_id: str
    backend: str
    pid: int
    status: str
    executed_steps: int
    replayed_steps: int
    deduplicated_hits: int
    awaiting_signal: str = ""
    error: str = ""
    wall_ms: float = 0.0
    replay_load_ms: float = 0.0
    checkpoint_write_ms: List[float] = field(default_factory=list)
    proc_io_rchar: int = 0
    proc_io_wchar: int = 0
    proc_io_write_bytes: int = 0
    proc_io_syscw: int = 0
    outputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "org": self.org,
            "workflow_id": self.workflow_id,
            "run_id": self.run_id,
            "execution_id": self.execution_id,
            "actor_id": self.actor_id,
            "backend": self.backend,
            "pid": self.pid,
            "status": self.status,
            "executed_steps": self.executed_steps,
            "replayed_steps": self.replayed_steps,
            "deduplicated_hits": self.deduplicated_hits,
            "awaiting_signal": self.awaiting_signal,
            "error": self.error,
            "wall_ms": round(self.wall_ms, 4),
            "replay_load_ms": round(self.replay_load_ms, 4),
            "checkpoint_write_ms": [round(x, 4) for x in self.checkpoint_write_ms],
            "proc_io_rchar": self.proc_io_rchar,
            "proc_io_wchar": self.proc_io_wchar,
            "proc_io_write_bytes": self.proc_io_write_bytes,
            "proc_io_syscw": self.proc_io_syscw,
            "outputs": self.outputs,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ActivationTelemetry":
        return cls(
            org=str(raw["org"]),
            workflow_id=str(raw["workflow_id"]),
            run_id=str(raw["run_id"]),
            execution_id=str(raw["execution_id"]),
            actor_id=str(raw["actor_id"]),
            backend=str(raw["backend"]),
            pid=int(raw["pid"]),
            status=str(raw["status"]),
            executed_steps=int(raw["executed_steps"]),
            replayed_steps=int(raw["replayed_steps"]),
            deduplicated_hits=int(raw["deduplicated_hits"]),
            awaiting_signal=str(raw.get("awaiting_signal", "")),
            error=str(raw.get("error", "")),
            wall_ms=float(raw.get("wall_ms", 0.0)),
            replay_load_ms=float(raw.get("replay_load_ms", 0.0)),
            checkpoint_write_ms=[float(x) for x in raw.get("checkpoint_write_ms", [])],
            proc_io_rchar=int(raw.get("proc_io_rchar", 0)),
            proc_io_wchar=int(raw.get("proc_io_wchar", 0)),
            proc_io_write_bytes=int(raw.get("proc_io_write_bytes", 0)),
            proc_io_syscw=int(raw.get("proc_io_syscw", 0)),
            outputs=dict(raw.get("outputs") or {}),
        )
