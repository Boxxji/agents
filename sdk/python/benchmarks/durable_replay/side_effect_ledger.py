"""Persisted external side-effect ledger for cross-process exactly-once verification.

Simulates a real external downstream service (billing API, artifact publisher,
repository commit service) backed by its own SQLite WAL database independent of
the workflow checkpoint store. Every physical invocation from any OS process is
durably recorded with its PID, ``idempotency_key``, and whether the side effect
was physically applied (``applied=1``) or deduplicated (``deduplicated=1``).
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .schemas import canonical_json, hash_payload, validate_identifier


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExternalSideEffectLedger:
    """Independent SQLite WAL database tracking real cross-process side effects."""

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
                    CREATE TABLE IF NOT EXISTS effect_invocations (
                        invocation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        org TEXT NOT NULL,
                        run_id TEXT NOT NULL,
                        step_id TEXT NOT NULL,
                        activity_type TEXT NOT NULL,
                        idempotency_key TEXT NOT NULL,
                        pid INTEGER NOT NULL,
                        applied INTEGER NOT NULL,
                        deduplicated INTEGER NOT NULL,
                        payload_hash TEXT NOT NULL,
                        result_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                ),
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS applied_effects (
                        effect_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        org TEXT NOT NULL,
                        run_id TEXT NOT NULL,
                        step_id TEXT NOT NULL,
                        activity_type TEXT NOT NULL,
                        idempotency_key TEXT NOT NULL,
                        pid INTEGER NOT NULL,
                        result_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                ),
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_applied_org_key
                    ON applied_effects (org, idempotency_key)
                    """
                ),
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_applied_org_run
                    ON applied_effects (org, run_id)
                    """
                ),
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_invocations_org_run
                    ON effect_invocations (org, run_id)
                    """
                ),
            )
        )

    def _run_tx(self, fn: Any) -> Any:
        last_err: Exception | None = None
        for attempt in range(20):
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
        raise RuntimeError(f"ExternalSideEffectLedger lock contention: {last_err}")

    def apply_side_effect(
        self,
        *,
        org: str,
        run_id: str,
        step_id: str,
        activity_type: str,
        idempotency_key: str,
        payload: Dict[str, Any],
        enforce_idempotency: bool = True,
    ) -> Tuple[Dict[str, Any], bool]:
        """Apply a side effect to the external system.

        Returns ``(result_dict, deduplicated)`` where ``deduplicated=True`` means
        the external system recognized ``(org, idempotency_key)`` and returned the
        previously committed result without applying a duplicate mutation.
        """
        validate_identifier("org", org)
        validate_identifier("run_id", run_id)
        validate_identifier("step_id", step_id)
        validate_identifier("activity_type", activity_type)

        pid = os.getpid()
        payload_digest = hash_payload(payload)
        now = _utc_now()

        def _tx(cur: sqlite3.Cursor) -> Tuple[Dict[str, Any], bool]:
            if enforce_idempotency:
                row = cur.execute(
                    """
                    SELECT result_json FROM applied_effects
                    WHERE org = ? AND idempotency_key = ?
                    ORDER BY effect_id ASC LIMIT 1
                    """,
                    (org, idempotency_key),
                ).fetchone()
                if row is not None:
                    cached = json.loads(row["result_json"])
                    cur.execute(
                        """
                        INSERT INTO effect_invocations (
                            org, run_id, step_id, activity_type, idempotency_key,
                            pid, applied, deduplicated, payload_hash, result_json, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, 0, 1, ?, ?, ?)
                        """,
                        (
                            org,
                            run_id,
                            step_id,
                            activity_type,
                            idempotency_key,
                            pid,
                            payload_digest,
                            row["result_json"],
                            now,
                        ),
                    )
                    return cached, True

            result_obj = {
                "step_id": step_id,
                "activity_type": activity_type,
                "idempotency_key": idempotency_key,
                "payload_hash": payload_digest,
                "effect_digest": hash_payload(
                    {
                        "org": org,
                        "run_id": run_id,
                        "step_id": step_id,
                        "activity_type": activity_type,
                        "payload_hash": payload_digest,
                    }
                ),
                "applied_by_pid": pid,
            }
            result_json = canonical_json(result_obj)
            cur.execute(
                """
                INSERT INTO applied_effects (
                    org, run_id, step_id, activity_type, idempotency_key,
                    pid, result_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    org,
                    run_id,
                    step_id,
                    activity_type,
                    idempotency_key,
                    pid,
                    result_json,
                    now,
                ),
            )
            cur.execute(
                """
                INSERT INTO effect_invocations (
                    org, run_id, step_id, activity_type, idempotency_key,
                    pid, applied, deduplicated, payload_hash, result_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, 1, 0, ?, ?, ?)
                """,
                (
                    org,
                    run_id,
                    step_id,
                    activity_type,
                    idempotency_key,
                    pid,
                    payload_digest,
                    result_json,
                    now,
                ),
            )
            return result_obj, False

        return self._run_tx(_tx)

    def get_applied_counts(self, org: str, run_id: str) -> Dict[str, int]:
        """Return ``{step_id: applied_count}`` across all OS processes for ``(org, run_id)``."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT step_id, COUNT(*) AS cnt
                FROM applied_effects
                WHERE org = ? AND run_id = ?
                GROUP BY step_id
                ORDER BY step_id ASC
                """,
                (org, run_id),
            ).fetchall()
        return {str(r["step_id"]): int(r["cnt"]) for r in rows}

    def get_invocation_stats(self, org: str, run_id: str) -> Dict[str, Any]:
        """Return physical invocation counts, deduplication hits, and distinct OS PIDs."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT step_id, pid, applied, deduplicated
                FROM effect_invocations
                WHERE org = ? AND run_id = ?
                ORDER BY invocation_id ASC
                """,
                (org, run_id),
            ).fetchall()
        distinct_pids = sorted({int(r["pid"]) for r in rows})
        total_invocations = len(rows)
        total_applied = sum(int(r["applied"]) for r in rows)
        total_dedup = sum(int(r["deduplicated"]) for r in rows)
        applied_by_step: Dict[str, int] = {}
        for r in rows:
            if int(r["applied"]) > 0:
                sid = str(r["step_id"])
                applied_by_step[sid] = applied_by_step.get(sid, 0) + int(r["applied"])
        duplicate_effects = sum(max(0, c - 1) for c in applied_by_step.values())
        return {
            "total_invocations": total_invocations,
            "total_applied": total_applied,
            "total_deduplicated": total_dedup,
            "duplicate_side_effects": duplicate_effects,
            "distinct_pids": distinct_pids,
            "applied_by_step": applied_by_step,
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()
