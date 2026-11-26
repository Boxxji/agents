"""Persisted checkpoint and journal stores for durable workflow replay.

Implements three backends behind a single interface for controlled comparison:
1. ``SqliteWalCheckpointStore``: SQLite with ``PRAGMA journal_mode=WAL`` and
   ``PRAGMA synchronous=FULL`` persisting event history, transactional
   ``idempotency_key`` ledger, run metadata, and step snapshots atomically.
2. ``AtomicJsonCheckpointStore``: Append+fsync JSON-lines journal with atomic
   ``os.replace`` + ``os.fsync`` state/ledger/snapshot files (mirroring
   ``control-plane/internal/durable/journal.go``).
3. ``EphemeralCheckpointStore``: Unpersisted in-memory Python dictionary control
   baseline demonstrating state loss and duplicate side effects across OS restarts.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .schemas import (
    BackendKind,
    EventType,
    InvalidIdentifierError,
    JournalEvent,
    LedgerRecord,
    RunStatus,
    StepSnapshot,
    TenantIsolationError,
    canonical_json,
    validate_identifier,
)


def _stable_hash(value: str) -> str:
    """Return 32-char SHA-256 prefix for path-traversal-proof directory sharding."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def _fsync_dir(dir_path: Path) -> None:
    """Best-effort fsync on parent directory after atomic rename."""
    try:
        fd = os.open(str(dir_path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _atomic_write_bytes(target_path: Path, data: bytes) -> None:
    """Write ``data`` to ``target_path`` via same-dir temp file + fsync + ``os.replace``."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", dir=str(target_path.parent))
    tmp_path = Path(tmp_name)
    try:
        try:
            os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(str(tmp_path), str(target_path))
        _fsync_dir(target_path.parent)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _status_for_event(event_type: EventType) -> Optional[RunStatus]:
    if event_type in (
        EventType.WORKFLOW_STARTED,
        EventType.WORKFLOW_RESUMED,
        EventType.ACTIVITY_SCHEDULED,
        EventType.ACTIVITY_COMPLETED,
        EventType.CHECKPOINT_SAVED,
    ):
        return RunStatus.RUNNING
    if event_type == EventType.WORKFLOW_SUSPENDED:
        return RunStatus.SUSPENDED
    if event_type == EventType.WORKFLOW_COMPLETED:
        return RunStatus.COMPLETED
    if event_type in (EventType.WORKFLOW_FAILED, EventType.ACTIVITY_FAILED):
        return RunStatus.FAILED
    return None


class CheckpointStore(ABC):
    """Abstract durable checkpoint and event-history store."""

    backend_kind: BackendKind

    @abstractmethod
    def load_events(self, org: str, run_id: str) -> List[JournalEvent]:
        """Return ordered persisted event history for ``(org, run_id)``."""

    @abstractmethod
    def get_run_metadata(self, org: str, run_id: str) -> Optional[Dict[str, Any]]:
        """Return persisted run ownership/status metadata or ``None`` if unseen."""

    @abstractmethod
    def append_event(
        self,
        event: JournalEvent,
        *,
        snapshot: Optional[StepSnapshot] = None,
        ledger_record: Optional[LedgerRecord] = None,
    ) -> None:
        """Atomically persist a journal event plus optional snapshot and ledger record."""

    @abstractmethod
    def get_ledger_record(self, org: str, idempotency_key: str) -> Optional[LedgerRecord]:
        """Lookup a committed activity record scoped strictly to ``org``."""

    @abstractmethod
    def list_snapshots(self, org: str, run_id: str) -> List[StepSnapshot]:
        """Return persisted step snapshots for ``(org, run_id)`` ordered by sequence."""

    @abstractmethod
    def storage_bytes(self) -> int:
        """Return real on-disk footprint in bytes (0 for ephemeral stores)."""

    @abstractmethod
    def close(self) -> None:
        """Release underlying file descriptors / database connections."""


class SqliteWalCheckpointStore(CheckpointStore):
    """Production-grade SQLite WAL + ``synchronous=FULL`` checkpoint store."""

    backend_kind = BackendKind.SQLITE_WAL

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._configure()
        self._init_schema()

    def _configure(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA synchronous=FULL;")
            cur.execute("PRAGMA busy_timeout=30000;")
            cur.execute("PRAGMA foreign_keys=ON;")
            cur.close()

    def _init_schema(self) -> None:
        self._run_tx(
            lambda cur: (
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS workflow_runs (
                        org TEXT NOT NULL,
                        run_id TEXT NOT NULL,
                        workflow_id TEXT NOT NULL,
                        actor_id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        last_sequence INTEGER NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (org, run_id)
                    )
                    """
                ),
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS workflow_events (
                        org TEXT NOT NULL,
                        run_id TEXT NOT NULL,
                        sequence INTEGER NOT NULL,
                        previous_sequence INTEGER NOT NULL,
                        event_type TEXT NOT NULL,
                        workflow_id TEXT NOT NULL,
                        execution_id TEXT NOT NULL,
                        actor_id TEXT NOT NULL,
                        round_idx INTEGER NOT NULL,
                        step_id TEXT NOT NULL,
                        activity_type TEXT NOT NULL,
                        idempotency_key TEXT NOT NULL,
                        input_hash TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        error TEXT NOT NULL,
                        pid INTEGER NOT NULL,
                        timestamp TEXT NOT NULL,
                        PRIMARY KEY (org, run_id, sequence)
                    )
                    """
                ),
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS activity_ledger (
                        org TEXT NOT NULL,
                        idempotency_key TEXT NOT NULL,
                        run_id TEXT NOT NULL,
                        round_idx INTEGER NOT NULL,
                        step_id TEXT NOT NULL,
                        activity_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        input_hash TEXT NOT NULL,
                        output_json TEXT NOT NULL,
                        error TEXT NOT NULL,
                        attempt INTEGER NOT NULL,
                        pid INTEGER NOT NULL,
                        completed_at TEXT NOT NULL,
                        PRIMARY KEY (org, idempotency_key)
                    )
                    """
                ),
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS step_snapshots (
                        org TEXT NOT NULL,
                        run_id TEXT NOT NULL,
                        round_idx INTEGER NOT NULL,
                        step_id TEXT NOT NULL,
                        sequence INTEGER NOT NULL,
                        state_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY (org, run_id, round_idx, step_id)
                    )
                    """
                ),
            )
        )

    def _run_tx(self, fn: Any) -> Any:
        last_err: Exception | None = None
        for attempt in range(25):
            with self._lock:
                cur = self._conn.cursor()
                try:
                    cur.execute("BEGIN IMMEDIATE")
                    result = fn(cur)
                    cur.execute("COMMIT")
                    cur.close()
                    return result
                except sqlite3.OperationalError as exc:
                    try:
                        cur.execute("ROLLBACK")
                    except sqlite3.Error:
                        pass
                    cur.close()
                    if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                        last_err = exc
                    else:
                        raise
                except Exception:
                    try:
                        cur.execute("ROLLBACK")
                    except sqlite3.Error:
                        pass
                    cur.close()
                    raise
            time.sleep(0.005 * (attempt + 1))
        raise RuntimeError(f"SqliteWalCheckpointStore lock contention: {last_err}")

    def pragma_state(self) -> Dict[str, str]:
        """Return active SQLite journal_mode and synchronous settings for audit tests."""
        with self._lock:
            jm = self._conn.execute("PRAGMA journal_mode;").fetchone()[0]
            sync = self._conn.execute("PRAGMA synchronous;").fetchone()[0]
        return {"journal_mode": str(jm).lower(), "synchronous": str(sync)}

    def load_events(self, org: str, run_id: str) -> List[JournalEvent]:
        validate_identifier("org", org)
        validate_identifier("run_id", run_id)
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM workflow_events
                WHERE org = ? AND run_id = ?
                ORDER BY sequence ASC
                """,
                (org, run_id),
            ).fetchall()
        return [
            JournalEvent(
                sequence=int(r["sequence"]),
                previous_sequence=int(r["previous_sequence"]),
                event_type=EventType(r["event_type"]),
                org=str(r["org"]),
                workflow_id=str(r["workflow_id"]),
                run_id=str(r["run_id"]),
                execution_id=str(r["execution_id"]),
                actor_id=str(r["actor_id"]),
                round_idx=int(r["round_idx"]),
                step_id=str(r["step_id"]),
                activity_type=str(r["activity_type"]),
                idempotency_key=str(r["idempotency_key"]),
                input_hash=str(r["input_hash"]),
                payload=json.loads(r["payload_json"]),
                error=str(r["error"]),
                pid=int(r["pid"]),
                timestamp=str(r["timestamp"]),
            )
            for r in rows
        ]

    def get_run_metadata(self, org: str, run_id: str) -> Optional[Dict[str, Any]]:
        validate_identifier("org", org)
        validate_identifier("run_id", run_id)
        with self._lock:
            row = self._conn.execute(
                """
                SELECT org, run_id, workflow_id, actor_id, status, last_sequence, updated_at
                FROM workflow_runs
                WHERE org = ? AND run_id = ?
                """,
                (org, run_id),
            ).fetchone()
        if row is None:
            return None
        return dict(row)

    def append_event(
        self,
        event: JournalEvent,
        *,
        snapshot: Optional[StepSnapshot] = None,
        ledger_record: Optional[LedgerRecord] = None,
    ) -> None:
        validate_identifier("org", event.org)
        validate_identifier("run_id", event.run_id)
        validate_identifier("workflow_id", event.workflow_id)
        validate_identifier("actor_id", event.actor_id)

        def _tx(cur: sqlite3.Cursor) -> None:
            existing = cur.execute(
                """
                SELECT workflow_id, actor_id, status, last_sequence
                FROM workflow_runs WHERE org = ? AND run_id = ?
                """,
                (event.org, event.run_id),
            ).fetchone()

            if existing is not None:
                if existing["actor_id"] != event.actor_id:
                    raise TenantIsolationError(
                        f"Actor mismatch for run {event.run_id!r}: "
                        f"expected {existing['actor_id']!r}, got {event.actor_id!r}"
                    )
                if existing["workflow_id"] != event.workflow_id:
                    raise TenantIsolationError(
                        f"Workflow mismatch for run {event.run_id!r}: "
                        f"expected {existing['workflow_id']!r}, got {event.workflow_id!r}"
                    )

            cur.execute(
                """
                INSERT INTO workflow_events (
                    org, run_id, sequence, previous_sequence, event_type,
                    workflow_id, execution_id, actor_id, round_idx,
                    step_id, activity_type, idempotency_key, input_hash,
                    payload_json, error, pid, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.org,
                    event.run_id,
                    event.sequence,
                    event.previous_sequence,
                    event.event_type.value,
                    event.workflow_id,
                    event.execution_id,
                    event.actor_id,
                    event.round_idx,
                    event.step_id,
                    event.activity_type,
                    event.idempotency_key,
                    event.input_hash,
                    canonical_json(event.payload),
                    event.error,
                    event.pid,
                    event.timestamp,
                ),
            )

            next_status = _status_for_event(event.event_type)
            status_val = (
                next_status.value
                if next_status is not None
                else (existing["status"] if existing else RunStatus.RUNNING.value)
            )
            cur.execute(
                """
                INSERT INTO workflow_runs (
                    org, run_id, workflow_id, actor_id, status, last_sequence, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(org, run_id) DO UPDATE SET
                    status = excluded.status,
                    last_sequence = excluded.last_sequence,
                    updated_at = excluded.updated_at
                """,
                (
                    event.org,
                    event.run_id,
                    event.workflow_id,
                    event.actor_id,
                    status_val,
                    event.sequence,
                    event.timestamp,
                ),
            )

            if ledger_record is not None:
                cur.execute(
                    """
                    INSERT OR IGNORE INTO activity_ledger (
                        org, idempotency_key, run_id, round_idx, step_id,
                        activity_type, status, input_hash, output_json,
                        error, attempt, pid, completed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ledger_record.org,
                        ledger_record.idempotency_key,
                        ledger_record.run_id,
                        ledger_record.round_idx,
                        ledger_record.step_id,
                        ledger_record.activity_type,
                        ledger_record.status,
                        ledger_record.input_hash,
                        canonical_json(ledger_record.output),
                        ledger_record.error,
                        ledger_record.attempt,
                        ledger_record.pid,
                        ledger_record.completed_at,
                    ),
                )

            if snapshot is not None:
                cur.execute(
                    """
                    INSERT OR REPLACE INTO step_snapshots (
                        org, run_id, round_idx, step_id, sequence, state_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot.org,
                        snapshot.run_id,
                        snapshot.round_idx,
                        snapshot.step_id,
                        snapshot.sequence,
                        canonical_json(snapshot.state),
                        snapshot.created_at,
                    ),
                )

        self._run_tx(_tx)

    def get_ledger_record(self, org: str, idempotency_key: str) -> Optional[LedgerRecord]:
        validate_identifier("org", org)
        with self._lock:
            row = self._conn.execute(
                """
                SELECT * FROM activity_ledger
                WHERE org = ? AND idempotency_key = ?
                """,
                (org, idempotency_key),
            ).fetchone()
        if row is None:
            return None
        return LedgerRecord(
            idempotency_key=str(row["idempotency_key"]),
            org=str(row["org"]),
            run_id=str(row["run_id"]),
            round_idx=int(row["round_idx"]),
            step_id=str(row["step_id"]),
            activity_type=str(row["activity_type"]),
            status=str(row["status"]),
            input_hash=str(row["input_hash"]),
            output=json.loads(row["output_json"]),
            error=str(row["error"]),
            attempt=int(row["attempt"]),
            pid=int(row["pid"]),
            completed_at=str(row["completed_at"]),
        )

    def list_snapshots(self, org: str, run_id: str) -> List[StepSnapshot]:
        validate_identifier("org", org)
        validate_identifier("run_id", run_id)
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM step_snapshots
                WHERE org = ? AND run_id = ?
                ORDER BY sequence ASC
                """,
                (org, run_id),
            ).fetchall()
        return [
            StepSnapshot(
                org=str(r["org"]),
                run_id=str(r["run_id"]),
                round_idx=int(r["round_idx"]),
                step_id=str(r["step_id"]),
                sequence=int(r["sequence"]),
                state=json.loads(r["state_json"]),
                created_at=str(r["created_at"]),
            )
            for r in rows
        ]

    def storage_bytes(self) -> int:
        total = 0
        for suffix in ("", "-wal", "-shm"):
            p = Path(f"{self.db_path}{suffix}")
            if p.exists():
                total += p.stat().st_size
        return total

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class AtomicJsonCheckpointStore(CheckpointStore):
    """Append+fsync JSONL journal + atomic ``os.replace`` state/ledger/snapshot store.

    Mirrors ``control-plane/internal/durable/journal.go`` (``FileJournalStore``):
    directory names are SHA-256 sharded to guarantee zero path-traversal exposure,
    every path is verified under ``self.root_dir``, and advisory ``fcntl.flock``
    serializes concurrent multi-process writers on the same run.
    """

    backend_kind = BackendKind.ATOMIC_JSONL

    def __init__(self, root_dir: Path | str) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _safe_subpath(self, *parts: str) -> Path:
        candidate = (self.root_dir.joinpath(*parts)).resolve()
        try:
            candidate.relative_to(self.root_dir)
        except ValueError as exc:
            raise InvalidIdentifierError(
                f"Resolved checkpoint path escapes root directory: {candidate}"
            ) from exc
        return candidate

    def _org_dir(self, org: str) -> Path:
        validate_identifier("org", org)
        return self._safe_subpath("orgs", _stable_hash(org))

    def _run_dir(self, org: str, run_id: str) -> Path:
        validate_identifier("org", org)
        validate_identifier("run_id", run_id)
        return self._safe_subpath("orgs", _stable_hash(org), "runs", _stable_hash(run_id))

    def _ledger_path(self, org: str, idempotency_key: str) -> Path:
        validate_identifier("org", org)
        validate_identifier("idempotency_key", idempotency_key)
        return self._safe_subpath(
            "orgs", _stable_hash(org), "ledger", f"{idempotency_key}.json"
        )

    def _lock_run(self, org: str, run_id: str) -> Tuple[int, Path]:
        rdir = self._run_dir(org, run_id)
        rdir.mkdir(parents=True, exist_ok=True)
        lock_path = rdir / ".lock"
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
        fcntl.flock(fd, fcntl.LOCK_EX)
        return fd, rdir

    @staticmethod
    def _unlock_run(fd: int) -> None:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def _read_journal_unlocked(self, journal_path: Path) -> List[JournalEvent]:
        if not journal_path.exists():
            return []
        events: List[JournalEvent] = []
        raw_bytes = journal_path.read_bytes()
        total_len = len(raw_bytes)
        pos = 0
        last_valid_end = 0
        needs_truncate = False

        while pos < total_len:
            nl = raw_bytes.find(b"\n", pos)
            if nl == -1:
                # Unterminated torn tail left by a hard OS crash — truncate before
                # any future O_APPEND can concatenate a new record onto the torn bytes.
                needs_truncate = True
                break
            line_bytes = raw_bytes[pos : nl + 1]
            pos = nl + 1
            try:
                text = line_bytes.decode("utf-8").strip()
            except UnicodeDecodeError:
                if pos == total_len:
                    needs_truncate = True
                continue
            if not text:
                last_valid_end = pos
                continue
            try:
                events.append(JournalEvent.from_dict(json.loads(text)))
                last_valid_end = pos
            except (json.JSONDecodeError, KeyError, ValueError):
                if pos == total_len:
                    needs_truncate = True
                continue

        if needs_truncate and last_valid_end < total_len:
            tfd = os.open(str(journal_path), os.O_WRONLY)
            try:
                os.ftruncate(tfd, last_valid_end)
                os.fsync(tfd)
            finally:
                os.close(tfd)

        events.sort(key=lambda e: e.sequence)
        return events

    def _repair_torn_tail_unlocked(self, journal_path: Path) -> None:
        """Ensure journal.jsonl ends cleanly on a newline before O_APPEND writes."""
        if not journal_path.exists():
            return
        size = journal_path.stat().st_size
        if size == 0:
            return
        with journal_path.open("rb") as fh:
            fh.seek(size - 1)
            last_byte = fh.read(1)
        if last_byte != b"\n":
            self._read_journal_unlocked(journal_path)

    def load_events(self, org: str, run_id: str) -> List[JournalEvent]:
        rdir = self._run_dir(org, run_id)
        journal_path = rdir / "journal.jsonl"
        if not journal_path.exists():
            return []
        fd, _ = self._lock_run(org, run_id)
        try:
            return self._read_journal_unlocked(journal_path)
        finally:
            self._unlock_run(fd)

    def get_run_metadata(self, org: str, run_id: str) -> Optional[Dict[str, Any]]:
        rdir = self._run_dir(org, run_id)
        state_path = rdir / "state.json"
        journal_path = rdir / "journal.jsonl"
        if not state_path.exists() and not journal_path.exists():
            return None
        fd, _ = self._lock_run(org, run_id)
        try:
            meta: Optional[Dict[str, Any]] = None
            if state_path.exists():
                meta = json.loads(state_path.read_text(encoding="utf-8"))
            events = self._read_journal_unlocked(journal_path)
            if meta is None and events:
                last = events[-1]
                status = _status_for_event(last.event_type) or RunStatus.RUNNING
                meta = {
                    "org": last.org,
                    "run_id": last.run_id,
                    "workflow_id": last.workflow_id,
                    "actor_id": last.actor_id,
                    "status": status.value,
                    "last_sequence": last.sequence,
                    "updated_at": last.timestamp,
                }
            elif meta is not None and events and events[-1].sequence > int(meta.get("last_sequence", 0)):
                last = events[-1]
                status = _status_for_event(last.event_type) or RunStatus.RUNNING
                meta["status"] = status.value
                meta["last_sequence"] = last.sequence
                meta["updated_at"] = last.timestamp
            return meta
        finally:
            self._unlock_run(fd)

    def append_event(
        self,
        event: JournalEvent,
        *,
        snapshot: Optional[StepSnapshot] = None,
        ledger_record: Optional[LedgerRecord] = None,
    ) -> None:
        validate_identifier("org", event.org)
        validate_identifier("run_id", event.run_id)
        validate_identifier("workflow_id", event.workflow_id)
        validate_identifier("actor_id", event.actor_id)

        fd, rdir = self._lock_run(event.org, event.run_id)
        try:
            state_path = rdir / "state.json"
            existing: Optional[Dict[str, Any]] = None
            if state_path.exists():
                existing = json.loads(state_path.read_text(encoding="utf-8"))
                if existing["actor_id"] != event.actor_id:
                    raise TenantIsolationError(
                        f"Actor mismatch for run {event.run_id!r}: "
                        f"expected {existing['actor_id']!r}, got {event.actor_id!r}"
                    )
                if existing["workflow_id"] != event.workflow_id:
                    raise TenantIsolationError(
                        f"Workflow mismatch for run {event.run_id!r}: "
                        f"expected {existing['workflow_id']!r}, got {event.workflow_id!r}"
                    )

            if ledger_record is not None:
                lpath = self._ledger_path(ledger_record.org, ledger_record.idempotency_key)
                if not lpath.exists():
                    _atomic_write_bytes(
                        lpath, canonical_json(ledger_record.to_dict()).encode("utf-8")
                    )

            if snapshot is not None:
                snap_name = f"r{snapshot.round_idx}_{_stable_hash(snapshot.step_id)}.json"
                snap_path = rdir / "snapshots" / snap_name
                _atomic_write_bytes(
                    snap_path, canonical_json(snapshot.to_dict()).encode("utf-8")
                )

            journal_path = rdir / "journal.jsonl"
            self._repair_torn_tail_unlocked(journal_path)
            line = (canonical_json(event.to_dict()) + "\n").encode("utf-8")
            jfd = os.open(
                str(journal_path),
                os.O_CREAT | os.O_WRONLY | os.O_APPEND,
                0o644,
            )
            try:
                os.write(jfd, line)
                os.fsync(jfd)
            finally:
                os.close(jfd)

            next_status = _status_for_event(event.event_type)
            status_val = (
                next_status.value
                if next_status is not None
                else (existing["status"] if existing else RunStatus.RUNNING.value)
            )
            state_doc = {
                "org": event.org,
                "run_id": event.run_id,
                "workflow_id": event.workflow_id,
                "actor_id": event.actor_id,
                "status": status_val,
                "last_sequence": event.sequence,
                "updated_at": event.timestamp,
            }
            _atomic_write_bytes(
                state_path, canonical_json(state_doc).encode("utf-8")
            )
        finally:
            self._unlock_run(fd)

    def get_ledger_record(self, org: str, idempotency_key: str) -> Optional[LedgerRecord]:
        lpath = self._ledger_path(org, idempotency_key)
        if not lpath.exists():
            return None
        return LedgerRecord.from_dict(json.loads(lpath.read_text(encoding="utf-8")))

    def list_snapshots(self, org: str, run_id: str) -> List[StepSnapshot]:
        snap_dir = self._run_dir(org, run_id) / "snapshots"
        if not snap_dir.exists():
            return []
        snaps = [
            StepSnapshot.from_dict(json.loads(p.read_text(encoding="utf-8")))
            for p in snap_dir.glob("*.json")
        ]
        snaps.sort(key=lambda s: s.sequence)
        return snaps

    def storage_bytes(self) -> int:
        total = 0
        if not self.root_dir.exists():
            return 0
        for p in self.root_dir.rglob("*"):
            if p.is_file() and p.name != ".lock":
                try:
                    total += p.stat().st_size
                except OSError:
                    continue
        return total

    def close(self) -> None:
        return None


class EphemeralCheckpointStore(CheckpointStore):
    """Unpersisted in-memory dictionary baseline (control group).

    Demonstrates why an in-memory-only journal cannot survive OS process crashes:
    when a new OS process restarts after ``os._exit(99)``, this store starts empty
    and re-executes every step from scratch, causing duplicate external side effects.
    """

    backend_kind = BackendKind.EPHEMERAL_BASELINE

    def __init__(self) -> None:
        self._events: Dict[Tuple[str, str], List[JournalEvent]] = {}
        self._meta: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._ledger: Dict[Tuple[str, str], LedgerRecord] = {}
        self._snapshots: Dict[Tuple[str, str], Dict[Tuple[int, str], StepSnapshot]] = {}

    def load_events(self, org: str, run_id: str) -> List[JournalEvent]:
        validate_identifier("org", org)
        validate_identifier("run_id", run_id)
        return list(self._events.get((org, run_id), []))

    def get_run_metadata(self, org: str, run_id: str) -> Optional[Dict[str, Any]]:
        validate_identifier("org", org)
        validate_identifier("run_id", run_id)
        meta = self._meta.get((org, run_id))
        return dict(meta) if meta is not None else None

    def append_event(
        self,
        event: JournalEvent,
        *,
        snapshot: Optional[StepSnapshot] = None,
        ledger_record: Optional[LedgerRecord] = None,
    ) -> None:
        validate_identifier("org", event.org)
        validate_identifier("run_id", event.run_id)
        validate_identifier("workflow_id", event.workflow_id)
        validate_identifier("actor_id", event.actor_id)
        key = (event.org, event.run_id)
        existing = self._meta.get(key)
        if existing is not None:
            if existing["actor_id"] != event.actor_id:
                raise TenantIsolationError(f"Actor mismatch for run {event.run_id!r}")
            if existing["workflow_id"] != event.workflow_id:
                raise TenantIsolationError(f"Workflow mismatch for run {event.run_id!r}")

        self._events.setdefault(key, []).append(event)
        next_status = _status_for_event(event.event_type) or RunStatus.RUNNING
        self._meta[key] = {
            "org": event.org,
            "run_id": event.run_id,
            "workflow_id": event.workflow_id,
            "actor_id": event.actor_id,
            "status": next_status.value,
            "last_sequence": event.sequence,
            "updated_at": event.timestamp,
        }
        if ledger_record is not None:
            self._ledger.setdefault(
                (ledger_record.org, ledger_record.idempotency_key), ledger_record
            )
        if snapshot is not None:
            self._snapshots.setdefault(key, {})[(snapshot.round_idx, snapshot.step_id)] = (
                snapshot
            )

    def get_ledger_record(self, org: str, idempotency_key: str) -> Optional[LedgerRecord]:
        validate_identifier("org", org)
        return self._ledger.get((org, idempotency_key))

    def list_snapshots(self, org: str, run_id: str) -> List[StepSnapshot]:
        validate_identifier("org", org)
        validate_identifier("run_id", run_id)
        snaps = list(self._snapshots.get((org, run_id), {}).values())
        snaps.sort(key=lambda s: s.sequence)
        return snaps

    def storage_bytes(self) -> int:
        return 0

    def close(self) -> None:
        return None


def create_checkpoint_store(
    backend: BackendKind, store_path: Optional[Path | str] = None
) -> CheckpointStore:
    """Factory creating the requested checkpoint store backend."""
    if backend == BackendKind.SQLITE_WAL:
        if store_path is None:
            raise ValueError("store_path is required for sqlite_wal backend")
        return SqliteWalCheckpointStore(store_path)
    if backend == BackendKind.ATOMIC_JSONL:
        if store_path is None:
            raise ValueError("store_path is required for atomic_jsonl backend")
        return AtomicJsonCheckpointStore(store_path)
    if backend == BackendKind.EPHEMERAL_BASELINE:
        return EphemeralCheckpointStore()
    raise ValueError(f"Unsupported backend: {backend!r}")
