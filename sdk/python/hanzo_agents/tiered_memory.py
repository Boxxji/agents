"""
Unified Tiered Memory API (Episodic, Semantic, Procedural) with separate
retention, retrieval, compaction, stateless restart recovery, and derived index rebuildability.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

MemoryTier = Literal["episodic", "semantic", "procedural"]
TierRecordState = Literal["ACTIVE", "COMPACTED", "ARCHIVED"]

TIER_RECORD_PREFIX = "tiered_record:"
TIER_INDEX_PREFIX = "tiered_index:"


def _norm(value: str) -> str:
    return " ".join(value.strip().lower().split())


def compute_tiered_record_id(
    scope: str,
    scope_id: str,
    tier: MemoryTier,
    source_identity: str,
    content: str,
    version: int = 1,
) -> str:
    canonical = (
        f"{scope}:{scope_id}:{tier}:{_norm(source_identity)}:{_norm(content)}:v{version}"
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class TieredMemoryRecord:
    id: str
    tier: MemoryTier
    source_identity: str
    source_version: int
    version: int
    scope: str
    scope_id: str
    state: TierRecordState
    content: str
    episodic: Optional[Dict[str, Any]] = None
    semantic: Optional[Dict[str, Any]] = None
    procedural: Optional[Dict[str, Any]] = None
    lineage_ids: List[str] = field(default_factory=list)
    compacted_into_id: Optional[str] = None
    acl_roles: List[str] = field(default_factory=list)
    importance: float = 1.0
    metadata: Optional[Dict[str, Any]] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TieredMemoryRecord":
        return cls(**data)


class TieredMemory:
    """
    Three memory classes ('episodic', 'semantic', 'procedural') behind one API
    with separate retention, retrieval, and compaction over persistent storage.
    """

    def __init__(
        self,
        default_scope: str = "workflow",
        default_scope_id: str = "default",
        backing_store: Optional[Dict[str, Any]] = None,
    ):
        self.default_scope = default_scope
        self.default_scope_id = default_scope_id
        self._store = backing_store if backing_store is not None else {"records": {}, "vectors": {}}
        self._records: Dict[str, Dict[str, Any]] = self._store.setdefault("records", {})
        self._vectors: Dict[str, Tuple[str, str, List[float], Dict[str, Any]]] = (
            self._store.setdefault("vectors", {})
        )

    def record_episode(
        self,
        run_id: str,
        step_index: int,
        action: str,
        observation: str,
        source_identity: str,
        outcome: str = "observation",
        tokens_used: Optional[int] = None,
        acl_roles: Optional[List[str]] = None,
        scope: Optional[str] = None,
        scope_id: Optional[str] = None,
        embedding: Optional[Sequence[float]] = None,
    ) -> TieredMemoryRecord:
        sc = scope or self.default_scope
        sid = scope_id or self.default_scope_id
        content = f"[Run {run_id} #{step_index}] {action} -> {observation}"
        rec_id = compute_tiered_record_id(sc, sid, "episodic", source_identity, content, 1)
        rec = TieredMemoryRecord(
            id=rec_id,
            tier="episodic",
            source_identity=source_identity,
            source_version=1,
            version=1,
            scope=sc,
            scope_id=sid,
            state="ACTIVE",
            content=content,
            episodic={
                "run_id": run_id,
                "step_index": step_index,
                "action": action,
                "observation": observation,
                "outcome": outcome,
                "tokens_used": tokens_used,
            },
            acl_roles=list(acl_roles or []),
        )
        self._persist(rec, embedding)
        return rec

    def record_semantic(
        self,
        topic: str,
        statement: str,
        source_identity: str,
        lineage_ids: Optional[List[str]] = None,
        acl_roles: Optional[List[str]] = None,
        scope: Optional[str] = None,
        scope_id: Optional[str] = None,
        embedding: Optional[Sequence[float]] = None,
    ) -> TieredMemoryRecord:
        sc = scope or self.default_scope
        sid = scope_id or self.default_scope_id
        content = f"{topic}: {statement}"
        rec_id = compute_tiered_record_id(sc, sid, "semantic", source_identity, content, 1)
        rec = TieredMemoryRecord(
            id=rec_id,
            tier="semantic",
            source_identity=source_identity,
            source_version=1,
            version=1,
            scope=sc,
            scope_id=sid,
            state="ACTIVE",
            content=content,
            semantic={"topic": topic, "statement": statement},
            lineage_ids=list(lineage_ids or []),
            acl_roles=list(acl_roles or []),
        )
        self._persist(rec, embedding)
        return rec

    def record_procedure(
        self,
        name: str,
        trigger_pattern: str,
        steps: List[str],
        source_identity: str,
        recovery_steps: Optional[List[str]] = None,
        lineage_ids: Optional[List[str]] = None,
        acl_roles: Optional[List[str]] = None,
        scope: Optional[str] = None,
        scope_id: Optional[str] = None,
        embedding: Optional[Sequence[float]] = None,
    ) -> TieredMemoryRecord:
        sc = scope or self.default_scope
        sid = scope_id or self.default_scope_id
        content = f"{name} (trigger: {trigger_pattern}): {' -> '.join(steps)}"
        rec_id = compute_tiered_record_id(sc, sid, "procedural", source_identity, content, 1)
        rec = TieredMemoryRecord(
            id=rec_id,
            tier="procedural",
            source_identity=source_identity,
            source_version=1,
            version=1,
            scope=sc,
            scope_id=sid,
            state="ACTIVE",
            content=content,
            procedural={
                "name": name,
                "trigger_pattern": trigger_pattern,
                "steps": list(steps),
                "recovery_steps": list(recovery_steps or []),
            },
            lineage_ids=list(lineage_ids or []),
            acl_roles=list(acl_roles or []),
        )
        self._persist(rec, embedding)
        return rec

    def compact_episodic(
        self,
        run_id: Optional[str] = None,
        topic: str = "compacted-summary",
        scope: Optional[str] = None,
        scope_id: Optional[str] = None,
    ) -> Tuple[Optional[TieredMemoryRecord], List[str]]:
        sc = scope or self.default_scope
        sid = scope_id or self.default_scope_id
        episodes = self.retrieve(
            tiers=["episodic"],
            scope=sc,
            scope_id=sid,
            run_id=run_id,
            include_states=["ACTIVE"],
        )
        if not episodes:
            return None, []

        episodes.sort(key=lambda r: (r.episodic or {}).get("step_index", 0))
        compacted_ids = [e.id for e in episodes]
        summary_stmt = " | ".join(e.content for e in episodes)
        merged_acl = sorted({role for e in episodes for role in e.acl_roles})

        sem = self.record_semantic(
            topic=topic,
            statement=summary_stmt,
            source_identity=f"compaction:{compacted_ids[0]}",
            lineage_ids=compacted_ids,
            acl_roles=merged_acl,
            scope=sc,
            scope_id=sid,
        )

        now_iso = datetime.now(timezone.utc).isoformat()
        for ep in episodes:
            key = f"{sc}:{sid}:{TIER_RECORD_PREFIX}{ep.id}"
            updated = TieredMemoryRecord(
                **{
                    **ep.to_dict(),
                    "state": "COMPACTED",
                    "compacted_into_id": sem.id,
                    "updated_at": now_iso,
                }
            )
            self._records[key] = updated.to_dict()
            self._vectors.pop(key, None)

        return sem, compacted_ids

    def extract_procedure_from_run(
        self,
        run_id: str,
        name: str,
        trigger_pattern: str,
        scope: Optional[str] = None,
        scope_id: Optional[str] = None,
    ) -> TieredMemoryRecord:
        sc = scope or self.default_scope
        sid = scope_id or self.default_scope_id
        episodes = self.retrieve(
            tiers=["episodic"],
            scope=sc,
            scope_id=sid,
            run_id=run_id,
            include_states=["ACTIVE", "COMPACTED"],
        )
        episodes.sort(key=lambda r: (r.episodic or {}).get("step_index", 0))
        steps = [(e.episodic or {}).get("action", e.content) for e in episodes]
        lineage_ids = [e.id for e in episodes]
        merged_acl = sorted({role for e in episodes for role in e.acl_roles})
        return self.record_procedure(
            name=name,
            trigger_pattern=trigger_pattern,
            steps=steps,
            source_identity=f"procedure:{run_id}",
            lineage_ids=lineage_ids,
            acl_roles=merged_acl,
            scope=sc,
            scope_id=sid,
        )

    def recover_run_state(
        self,
        run_id: str,
        trigger_pattern: Optional[str] = None,
        caller_roles: Optional[Sequence[str]] = None,
        scope: Optional[str] = None,
        scope_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        sc = scope or self.default_scope
        sid = scope_id or self.default_scope_id
        steps = self.retrieve(
            tiers=["episodic"],
            scope=sc,
            scope_id=sid,
            run_id=run_id,
            caller_roles=caller_roles,
            include_states=["ACTIVE", "COMPACTED"],
        )
        steps.sort(key=lambda r: (r.episodic or {}).get("step_index", 0))
        semantic_ctx = self.retrieve(
            tiers=["semantic"], scope=sc, scope_id=sid, caller_roles=caller_roles
        )
        procedures = self.retrieve(
            tiers=["procedural"],
            scope=sc,
            scope_id=sid,
            trigger_pattern=trigger_pattern,
            caller_roles=caller_roles,
        )
        last_step = max(((s.episodic or {}).get("step_index", -1) for s in steps), default=-1)
        return {
            "run_id": run_id,
            "last_step_index": last_step,
            "steps": steps,
            "semantic_context": semantic_ctx,
            "applicable_procedures": procedures,
        }

    def retrieve(
        self,
        tiers: Optional[Sequence[MemoryTier]] = None,
        scope: Optional[str] = None,
        scope_id: Optional[str] = None,
        caller_roles: Optional[Sequence[str]] = None,
        run_id: Optional[str] = None,
        trigger_pattern: Optional[str] = None,
        include_states: Optional[Sequence[TierRecordState]] = None,
    ) -> List[TieredMemoryRecord]:
        sc = scope or self.default_scope
        sid = scope_id or self.default_scope_id
        allowed_tiers = set(tiers or ["episodic", "semantic", "procedural"])
        allowed_states = set(include_states or ["ACTIVE"])
        prefix = f"{sc}:{sid}:{TIER_RECORD_PREFIX}"

        results: List[TieredMemoryRecord] = []
        for key, raw in self._records.items():
            if not key.startswith(prefix):
                continue
            rec = TieredMemoryRecord.from_dict(raw)
            if rec.scope != sc or rec.scope_id != sid:
                continue
            if rec.tier not in allowed_tiers or rec.state not in allowed_states:
                continue
            if run_id and (rec.episodic or {}).get("run_id") != run_id:
                continue
            if trigger_pattern and rec.procedural:
                if _norm(trigger_pattern) not in _norm(rec.procedural.get("trigger_pattern", "")):
                    continue
            if not self._is_authorized(rec, caller_roles):
                continue
            results.append(rec)

        return results

    def rebuild_derived_index(
        self,
        embed_fn: Callable[[str], Sequence[float]],
        scope: Optional[str] = None,
        scope_id: Optional[str] = None,
    ) -> Dict[str, int]:
        sc = scope or self.default_scope
        sid = scope_id or self.default_scope_id
        prefix = f"{sc}:{sid}:{TIER_RECORD_PREFIX}"
        for key in list(self._vectors.keys()):
            if key.startswith(prefix):
                self._vectors.pop(key, None)

        rebuilt = 0
        skipped = 0
        for key, raw in self._records.items():
            if not key.startswith(prefix):
                continue
            rec = TieredMemoryRecord.from_dict(raw)
            if rec.state != "ACTIVE":
                skipped += 1
                continue
            vec = [float(x) for x in embed_fn(rec.content)]
            self._vectors[key] = (sc, sid, vec, {"record_id": rec.id, "tier": rec.tier})
            rebuilt += 1

        return {"rebuilt_count": rebuilt, "skipped_count": skipped}

    def _persist(
        self, record: TieredMemoryRecord, embedding: Optional[Sequence[float]]
    ) -> None:
        key = f"{record.scope}:{record.scope_id}:{TIER_RECORD_PREFIX}{record.id}"
        self._records[key] = record.to_dict()
        if record.state == "ACTIVE" and embedding is not None:
            self._vectors[key] = (
                record.scope,
                record.scope_id,
                [float(x) for x in embedding],
                {"record_id": record.id, "tier": record.tier},
            )

    @staticmethod
    def _is_authorized(
        record: TieredMemoryRecord, caller_roles: Optional[Sequence[str]]
    ) -> bool:
        if not record.acl_roles:
            return True
        if not caller_roles:
            return False
        caller_set = set(caller_roles)
        return any(r in caller_set for r in record.acl_roles)
