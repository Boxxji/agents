"""
Hindsight-style Reflective Memory (retain, recall, reflect) with strict belief
separation between immutable observed evidence ('OBSERVATION') and inferred
conclusions ('CONCLUSION') or evolving hypotheses ('HYPOTHESIS').
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

EpistemicKind = Literal["OBSERVATION", "CONCLUSION", "HYPOTHESIS"]
BeliefStatus = Literal["ACTIVE", "CONTESTED", "REFUTED"]

EPISTEMIC_RECORD_PREFIX = "epistemic_record:"


def _norm(val: str) -> str:
    return " ".join(val.strip().lower().split())


def compute_epistemic_id(
    scope: str,
    scope_id: str,
    kind: EpistemicKind,
    topic: str,
    statement: str,
    version: int = 1,
) -> str:
    canonical = f"{scope}:{scope_id}:{kind}:{_norm(topic)}:{_norm(statement)}:v{version}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class EpistemicMemoryRecord:
    id: str
    kind: EpistemicKind
    status: BeliefStatus
    topic: str
    statement: str
    source_identity: str
    source_version: int
    version: int
    scope: str
    scope_id: str
    confidence: float
    supporting_observation_ids: List[str] = field(default_factory=list)
    contradicting_observation_ids: List[str] = field(default_factory=list)
    acl_roles: List[str] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpistemicMemoryRecord":
        return cls(**data)


class ReflectiveMemory:
    """
    Separates observed evidence from inferred conclusions and hypotheses:
    - retain(): stores immutable OBSERVATION records.
    - reflect(): creates CONCLUSION / HYPOTHESIS records linked to verified observations.
    - challenge_belief(): attaches contradicting observations and recalibrates confidence/status.
    - recall(): queries epistemic memory with strict tenant & RBAC pre-filtering.
    """

    def __init__(
        self,
        default_scope: str = "workflow",
        default_scope_id: str = "default",
    ):
        self.default_scope = default_scope
        self.default_scope_id = default_scope_id
        self._records: Dict[str, Dict[str, Any]] = {}
        self._vectors: Dict[str, Tuple[str, str, List[float], Dict[str, Any]]] = {}

    def retain(
        self,
        topic: str,
        statement: str,
        source_identity: str,
        source_version: int = 1,
        acl_roles: Optional[List[str]] = None,
        scope: Optional[str] = None,
        scope_id: Optional[str] = None,
        embedding: Optional[Sequence[float]] = None,
    ) -> EpistemicMemoryRecord:
        sc = scope or self.default_scope
        sid = scope_id or self.default_scope_id
        rec_id = compute_epistemic_id(sc, sid, "OBSERVATION", topic, statement, 1)
        existing = self.get_record_by_id(rec_id, scope=sc, scope_id=sid)
        if existing is not None:
            return existing

        rec = EpistemicMemoryRecord(
            id=rec_id,
            kind="OBSERVATION",
            status="ACTIVE",
            topic=topic.strip(),
            statement=statement.strip(),
            source_identity=source_identity,
            source_version=source_version,
            version=1,
            scope=sc,
            scope_id=sid,
            confidence=1.0,
            acl_roles=list(acl_roles or []),
        )
        self._persist(rec, embedding)
        return rec

    def reflect(
        self,
        kind: Literal["CONCLUSION", "HYPOTHESIS"],
        topic: str,
        statement: str,
        supporting_observation_ids: Sequence[str],
        source_identity: str,
        contradicting_observation_ids: Optional[Sequence[str]] = None,
        confidence: float = 0.8,
        acl_roles: Optional[List[str]] = None,
        scope: Optional[str] = None,
        scope_id: Optional[str] = None,
        embedding: Optional[Sequence[float]] = None,
    ) -> EpistemicMemoryRecord:
        sc = scope or self.default_scope
        sid = scope_id or self.default_scope_id
        if not supporting_observation_ids:
            raise ValueError(
                "Unsupported belief rejected: conclusions and hypotheses require supporting_observation_ids"
            )

        verified_support: List[str] = []
        inherited_roles = set(acl_roles or [])
        for oid in supporting_observation_ids:
            obs = self.get_record_by_id(oid, scope=sc, scope_id=sid)
            if obs is None or obs.kind != "OBSERVATION":
                raise ValueError(
                    f"Invalid supporting observation '{oid}': must exist as OBSERVATION in the same scope"
                )
            verified_support.append(obs.id)
            inherited_roles.update(obs.acl_roles)

        verified_contra: List[str] = []
        for cid in contradicting_observation_ids or []:
            obs = self.get_record_by_id(cid, scope=sc, scope_id=sid)
            if obs is None or obs.kind != "OBSERVATION":
                raise ValueError(
                    f"Invalid contradicting observation '{cid}': must exist as OBSERVATION in the same scope"
                )
            verified_contra.append(obs.id)

        cal_conf, status = self._calibrate(
            confidence, len(verified_support), len(verified_contra)
        )
        rec_id = compute_epistemic_id(sc, sid, kind, topic, statement, 1)
        existing = self.get_record_by_id(rec_id, scope=sc, scope_id=sid)

        rec = EpistemicMemoryRecord(
            id=rec_id,
            kind=kind,
            status=status,
            topic=topic.strip(),
            statement=statement.strip(),
            source_identity=source_identity,
            source_version=1,
            version=(existing.version + 1) if existing else 1,
            scope=sc,
            scope_id=sid,
            confidence=cal_conf,
            supporting_observation_ids=list(dict.fromkeys(verified_support)),
            contradicting_observation_ids=list(dict.fromkeys(verified_contra)),
            acl_roles=sorted(inherited_roles),
        )
        self._persist(rec, embedding)
        return rec

    def challenge_belief(
        self,
        belief_id: str,
        contradicting_observation_id: str,
        scope: Optional[str] = None,
        scope_id: Optional[str] = None,
    ) -> EpistemicMemoryRecord:
        sc = scope or self.default_scope
        sid = scope_id or self.default_scope_id
        belief = self.get_record_by_id(belief_id, scope=sc, scope_id=sid)
        if belief is None:
            raise ValueError(f"Belief '{belief_id}' not found")
        if belief.kind == "OBSERVATION":
            raise ValueError("Observed facts (OBSERVATION) are immutable and cannot be overwritten")

        obs = self.get_record_by_id(contradicting_observation_id, scope=sc, scope_id=sid)
        if obs is None or obs.kind != "OBSERVATION":
            raise ValueError("Contradicting record must be an existing OBSERVATION in the same scope")

        contra_ids = list(
            dict.fromkeys([*belief.contradicting_observation_ids, obs.id])
        )
        cal_conf, status = self._calibrate(
            belief.confidence, len(belief.supporting_observation_ids), len(contra_ids)
        )

        updated = EpistemicMemoryRecord(
            **{
                **belief.to_dict(),
                "status": status,
                "confidence": cal_conf,
                "version": belief.version + 1,
                "contradicting_observation_ids": contra_ids,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        key = f"{sc}:{sid}:{EPISTEMIC_RECORD_PREFIX}{updated.id}"
        self._records[key] = updated.to_dict()
        if status == "REFUTED":
            self._vectors.pop(key, None)
        return updated

    def recall(
        self,
        kinds: Optional[Sequence[EpistemicKind]] = None,
        statuses: Optional[Sequence[BeliefStatus]] = None,
        topic: Optional[str] = None,
        caller_roles: Optional[Sequence[str]] = None,
        scope: Optional[str] = None,
        scope_id: Optional[str] = None,
    ) -> List[EpistemicMemoryRecord]:
        sc = scope or self.default_scope
        sid = scope_id or self.default_scope_id
        allowed_kinds = set(kinds or ["OBSERVATION", "CONCLUSION", "HYPOTHESIS"])
        allowed_statuses = set(statuses or ["ACTIVE"])
        prefix = f"{sc}:{sid}:{EPISTEMIC_RECORD_PREFIX}"

        results: List[EpistemicMemoryRecord] = []
        for key, raw in self._records.items():
            if not key.startswith(prefix):
                continue
            rec = EpistemicMemoryRecord.from_dict(raw)
            if rec.scope != sc or rec.scope_id != sid:
                continue
            if rec.kind not in allowed_kinds or rec.status not in allowed_statuses:
                continue
            if topic and _norm(rec.topic) != _norm(topic):
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
        prefix = f"{sc}:{sid}:{EPISTEMIC_RECORD_PREFIX}"
        for key in list(self._vectors.keys()):
            if key.startswith(prefix):
                self._vectors.pop(key, None)

        rebuilt = 0
        skipped = 0
        for key, raw in self._records.items():
            if not key.startswith(prefix):
                continue
            rec = EpistemicMemoryRecord.from_dict(raw)
            if rec.status != "ACTIVE":
                skipped += 1
                continue
            vec = [float(x) for x in embed_fn(f"{rec.topic}: {rec.statement}")]
            self._vectors[key] = (sc, sid, vec, {"record_id": rec.id, "kind": rec.kind})
            rebuilt += 1
        return {"rebuilt_count": rebuilt, "skipped_count": skipped}

    def get_record_by_id(
        self,
        record_id: str,
        scope: Optional[str] = None,
        scope_id: Optional[str] = None,
    ) -> Optional[EpistemicMemoryRecord]:
        sc = scope or self.default_scope
        sid = scope_id or self.default_scope_id
        raw = self._records.get(f"{sc}:{sid}:{EPISTEMIC_RECORD_PREFIX}{record_id}")
        if raw is None:
            return None
        rec = EpistemicMemoryRecord.from_dict(raw)
        if rec.scope != sc or rec.scope_id != sid:
            return None
        return rec

    @staticmethod
    def _calibrate(
        base_conf: float, support_count: int, contra_count: int
    ) -> Tuple[float, BeliefStatus]:
        if contra_count == 0:
            boost = min(0.15, (support_count - 1) * 0.05)
            return round(min(1.0, max(0.05, base_conf + boost)), 4), "ACTIVE"
        ratio = support_count / (support_count + contra_count)
        cal = round(max(0.05, base_conf * ratio), 4)
        if contra_count >= support_count or cal < 0.35:
            return cal, "REFUTED"
        return cal, "CONTESTED"

    def _persist(
        self, rec: EpistemicMemoryRecord, embedding: Optional[Sequence[float]]
    ) -> None:
        key = f"{rec.scope}:{rec.scope_id}:{EPISTEMIC_RECORD_PREFIX}{rec.id}"
        self._records[key] = rec.to_dict()
        if rec.status == "ACTIVE" and embedding is not None:
            self._vectors[key] = (
                rec.scope,
                rec.scope_id,
                [float(x) for x in embedding],
                {"record_id": rec.id, "kind": rec.kind},
            )

    @staticmethod
    def _is_authorized(
        rec: EpistemicMemoryRecord, caller_roles: Optional[Sequence[str]]
    ) -> bool:
        if not rec.acl_roles:
            return True
        if not caller_roles:
            return False
        caller_set = set(caller_roles)
        return any(r in caller_set for r in rec.acl_roles)
