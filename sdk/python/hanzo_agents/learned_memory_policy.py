"""
Memory-R1 style Learned Memory Policy Controller & Store:
Selects ADD / UPDATE / DELETE / FORGET / NOOP with downstream task outcome
calibration while strictly enforcing hard authorization, revocation, TTL
forgetting, sensitive-data isolation, and 'INSUFFICIENT_EVIDENCE' abstention.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

PolicyAction = Literal["ADD", "UPDATE", "DELETE", "FORGET", "NOOP"]
QueryAbstentionStatus = Literal["ANSWER", "INSUFFICIENT_EVIDENCE"]

POLICY_RECORD_PREFIX = "policy_rec:"


def _norm(val: str) -> str:
    return " ".join(val.strip().lower().split())


def compute_policy_record_id(
    scope: str, scope_id: str, key_slot: str, content: str, version: int
) -> str:
    canonical = f"{scope}:{scope_id}:{_norm(key_slot)}:{_norm(content)}:v{version}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class PolicyManagedRecord:
    id: str
    key_slot: str
    content: str
    source_identity: str
    source_version: int
    version: int
    scope: str
    scope_id: str
    confidence: float
    utility_score: float
    expires_at: Optional[str] = None
    acl_roles: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyManagedRecord":
        return cls(**data)


@dataclass(frozen=True)
class PolicyDecisionTrace:
    decision_id: str
    action: PolicyAction
    overridden_by_hard_policy: bool
    reason: str
    key_slot: str
    features: Tuple[float, float, float, float]
    record: Optional[PolicyManagedRecord] = None


class PolicyControlledMemory:
    def __init__(
        self,
        default_scope: str = "workflow",
        default_scope_id: str = "default",
        learning_rate: float = 0.15,
        min_confidence_for_add: float = 0.35,
        min_evidence_count: int = 1,
        min_answer_confidence: float = 0.6,
    ):
        self.default_scope = default_scope
        self.default_scope_id = default_scope_id
        self.learning_rate = learning_rate
        self.min_confidence_for_add = min_confidence_for_add
        self.min_evidence_count = min_evidence_count
        self.min_answer_confidence = min_answer_confidence

        self.action_weights: Dict[PolicyAction, List[float]] = {
            "ADD": [0.9, 0.1, 0.6, 0.8],
            "UPDATE": [0.2, 0.9, 0.7, 0.8],
            "DELETE": [-0.5, 0.4, -0.6, -0.5],
            "FORGET": [-0.4, 0.2, -0.8, -0.6],
            "NOOP": [-0.6, -0.6, 0.2, -0.2],
        }
        self._records: Dict[str, Dict[str, Any]] = {}
        self._slot_maps: Dict[str, Dict[str, str]] = {}
        self._revoked_sources: Dict[str, set] = {}
        self._vectors: Dict[str, Tuple[str, str, List[float], Dict[str, Any]]] = {}
        self._traces: Dict[str, PolicyDecisionTrace] = {}

    def apply_policy(
        self,
        key_slot: str,
        content: str,
        source_identity: str,
        confidence: float = 0.85,
        utility_hint: float = 0.75,
        expires_at: Optional[str] = None,
        revoke_key: bool = False,
        contains_restricted_sensitive_data: bool = False,
        acl_roles: Optional[Sequence[str]] = None,
        now_iso: Optional[str] = None,
        scope: Optional[str] = None,
        scope_id: Optional[str] = None,
        embedding: Optional[Sequence[float]] = None,
    ) -> PolicyDecisionTrace:
        sc = scope or self.default_scope
        sid = scope_id or self.default_scope_id
        now = now_iso or datetime.now(timezone.utc).isoformat()
        slot = _norm(key_slot)
        scope_key = f"{sc}:{sid}"
        slot_map = self._slot_maps.setdefault(scope_key, {})
        revoked = self._revoked_sources.setdefault(scope_key, set())
        decision_id = uuid.uuid4().hex[:24]

        existing_id = slot_map.get(slot)
        existing = self.get_record_by_id(existing_id, scope=sc, scope_id=sid) if existing_id else None

        novelty = 0.0 if existing else 1.0
        contradiction = (
            1.0 if (existing and _norm(existing.content) != _norm(content)) else 0.0
        )
        features = (novelty, contradiction, confidence, utility_hint)

        # 1. Hard sensitive-data isolation override
        if contains_restricted_sensitive_data:
            trace = PolicyDecisionTrace(
                decision_id=decision_id,
                action="NOOP",
                overridden_by_hard_policy=True,
                reason="Hard policy override: restricted sensitive data rejected",
                key_slot=slot,
                features=features,
            )
            self._traces[decision_id] = trace
            return trace

        # 2. Hard revocation override
        if revoke_key or source_identity in revoked:
            if existing:
                self._remove_record(existing.id, slot, slot_map, sc, sid)
            trace = PolicyDecisionTrace(
                decision_id=decision_id,
                action="DELETE",
                overridden_by_hard_policy=True,
                reason="Hard policy override: revoked source or key",
                key_slot=slot,
                features=features,
            )
            self._traces[decision_id] = trace
            return trace

        # 3. Hard retention / TTL override
        if expires_at and expires_at <= now:
            if existing:
                self._remove_record(existing.id, slot, slot_map, sc, sid)
            trace = PolicyDecisionTrace(
                decision_id=decision_id,
                action="FORGET",
                overridden_by_hard_policy=True,
                reason=f"Hard retention override: expired at {expires_at}",
                key_slot=slot,
                features=features,
            )
            self._traces[decision_id] = trace
            return trace

        # 4. Exact duplicate -> NOOP
        if existing and _norm(existing.content) == _norm(content):
            trace = PolicyDecisionTrace(
                decision_id=decision_id,
                action="NOOP",
                overridden_by_hard_policy=False,
                reason="Identical content already active",
                key_slot=slot,
                features=features,
                record=existing,
            )
            self._traces[decision_id] = trace
            return trace

        action = self._select_action(features, existing is not None)
        if action in ("NOOP", "FORGET", "DELETE"):
            if action in ("FORGET", "DELETE") and existing:
                self._remove_record(existing.id, slot, slot_map, sc, sid)
            trace = PolicyDecisionTrace(
                decision_id=decision_id,
                action=action,
                overridden_by_hard_policy=False,
                reason=f"Learned policy selected {action}",
                key_slot=slot,
                features=features,
            )
            self._traces[decision_id] = trace
            return trace

        next_version = (existing.version + 1) if existing else 1
        if existing:
            self._remove_record(existing.id, slot, slot_map, sc, sid)

        rec_id = compute_policy_record_id(sc, sid, slot, content, next_version)
        rec = PolicyManagedRecord(
            id=rec_id,
            key_slot=slot,
            content=content.strip(),
            source_identity=source_identity,
            source_version=1,
            version=next_version,
            scope=sc,
            scope_id=sid,
            confidence=confidence,
            utility_score=utility_hint,
            expires_at=expires_at,
            acl_roles=list(acl_roles or []),
            created_at=existing.created_at if existing else now,
            updated_at=now,
        )
        r_key = f"{sc}:{sid}:{POLICY_RECORD_PREFIX}{rec.id}"
        self._records[r_key] = rec.to_dict()
        slot_map[slot] = rec.id
        if embedding is not None:
            self._vectors[r_key] = (
                sc,
                sid,
                [float(x) for x in embedding],
                {"record_id": rec.id, "key_slot": slot},
            )

        final_action: PolicyAction = "UPDATE" if existing else "ADD"
        trace = PolicyDecisionTrace(
            decision_id=decision_id,
            action=final_action,
            overridden_by_hard_policy=False,
            reason=f"Learned policy selected {final_action}",
            key_slot=slot,
            features=features,
            record=rec,
        )
        self._traces[decision_id] = trace
        return trace

    def record_task_outcome(self, decision_id: str, outcome_reward: float) -> bool:
        trace = self._traces.get(decision_id)
        if not trace or trace.overridden_by_hard_policy:
            return False
        w = self.action_weights[trace.action]
        self.action_weights[trace.action] = [
            round(w[i] + self.learning_rate * outcome_reward * trace.features[i], 4)
            for i in range(4)
        ]
        return True

    def revoke_source(
        self,
        source_identity: str,
        scope: Optional[str] = None,
        scope_id: Optional[str] = None,
    ) -> int:
        sc = scope or self.default_scope
        sid = scope_id or self.default_scope_id
        scope_key = f"{sc}:{sid}"
        self._revoked_sources.setdefault(scope_key, set()).add(source_identity)
        slot_map = self._slot_maps.setdefault(scope_key, {})

        removed = 0
        for slot, rid in list(slot_map.items()):
            rec = self.get_record_by_id(rid, scope=sc, scope_id=sid)
            if rec and rec.source_identity == source_identity:
                self._remove_record(rec.id, slot, slot_map, sc, sid)
                removed += 1
        return removed

    def forget_stale(
        self,
        now_iso: str,
        scope: Optional[str] = None,
        scope_id: Optional[str] = None,
    ) -> int:
        sc = scope or self.default_scope
        sid = scope_id or self.default_scope_id
        slot_map = self._slot_maps.setdefault(f"{sc}:{sid}", {})
        forgotten = 0
        for slot, rid in list(slot_map.items()):
            rec = self.get_record_by_id(rid, scope=sc, scope_id=sid)
            if rec and rec.expires_at and rec.expires_at <= now_iso:
                self._remove_record(rec.id, slot, slot_map, sc, sid)
                forgotten += 1
        return forgotten

    def query_with_abstention(
        self,
        key_slots: Optional[Sequence[str]] = None,
        caller_roles: Optional[Sequence[str]] = None,
        now_iso: Optional[str] = None,
        min_evidence_count: Optional[int] = None,
        min_confidence: Optional[float] = None,
        scope: Optional[str] = None,
        scope_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        sc = scope or self.default_scope
        sid = scope_id or self.default_scope_id
        now = now_iso or datetime.now(timezone.utc).isoformat()
        min_cnt = min_evidence_count if min_evidence_count is not None else self.min_evidence_count
        min_conf = min_confidence if min_confidence is not None else self.min_answer_confidence
        slot_map = self._slot_maps.get(f"{sc}:{sid}", {})
        targets = [_norm(k) for k in key_slots] if key_slots else list(slot_map.keys())

        valid: List[PolicyManagedRecord] = []
        for sk in targets:
            rid = slot_map.get(sk)
            if not rid:
                continue
            rec = self.get_record_by_id(rid, scope=sc, scope_id=sid)
            if not rec:
                continue
            if rec.expires_at and rec.expires_at <= now:
                continue
            if not self._is_authorized(rec.acl_roles, caller_roles):
                continue
            if rec.confidence < min_conf:
                continue
            valid.append(rec)

        if len(valid) < min_cnt:
            return {
                "status": "INSUFFICIENT_EVIDENCE",
                "reason": f"Found {len(valid)} record(s) meeting threshold (required >= {min_cnt})",
                "records": [],
            }
        return {
            "status": "ANSWER",
            "reason": f"Verified {len(valid)} supporting record(s)",
            "records": valid,
        }

    def rebuild_derived_index(
        self,
        embed_fn: Callable[[str], Sequence[float]],
        scope: Optional[str] = None,
        scope_id: Optional[str] = None,
    ) -> Dict[str, int]:
        sc = scope or self.default_scope
        sid = scope_id or self.default_scope_id
        prefix = f"{sc}:{sid}:{POLICY_RECORD_PREFIX}"
        for k in list(self._vectors.keys()):
            if k.startswith(prefix):
                self._vectors.pop(k, None)

        slot_map = self._slot_maps.get(f"{sc}:{sid}", {})
        rebuilt = 0
        for rid in slot_map.values():
            rec = self.get_record_by_id(rid, scope=sc, scope_id=sid)
            if not rec:
                continue
            r_key = f"{sc}:{sid}:{POLICY_RECORD_PREFIX}{rec.id}"
            self._vectors[r_key] = (
                sc,
                sid,
                [float(x) for x in embed_fn(rec.content)],
                {"record_id": rec.id, "key_slot": rec.key_slot},
            )
            rebuilt += 1
        return {"rebuilt_count": rebuilt}

    def get_record_by_id(
        self,
        record_id: Optional[str],
        scope: Optional[str] = None,
        scope_id: Optional[str] = None,
    ) -> Optional[PolicyManagedRecord]:
        if not record_id:
            return None
        sc = scope or self.default_scope
        sid = scope_id or self.default_scope_id
        raw = self._records.get(f"{sc}:{sid}:{POLICY_RECORD_PREFIX}{record_id}")
        return PolicyManagedRecord.from_dict(raw) if raw else None

    def _select_action(
        self, features: Tuple[float, float, float, float], has_existing: bool
    ) -> PolicyAction:
        if features[2] < self.min_confidence_for_add:
            return "NOOP"
        candidates: List[PolicyAction] = (
            ["UPDATE", "NOOP", "FORGET"] if has_existing else ["ADD", "NOOP"]
        )
        best_act = candidates[0]
        best_score = float("-inf")
        for act in candidates:
            w = self.action_weights[act]
            score = sum(w[i] * features[i] for i in range(4))
            if score > best_score:
                best_score = score
                best_act = act
        return best_act

    def _remove_record(
        self,
        record_id: str,
        slot_key: str,
        slot_map: Dict[str, str],
        scope: str,
        scope_id: str,
    ) -> None:
        r_key = f"{scope}:{scope_id}:{POLICY_RECORD_PREFIX}{record_id}"
        self._records.pop(r_key, None)
        self._vectors.pop(r_key, None)
        slot_map.pop(slot_key, None)

    @staticmethod
    def _is_authorized(
        acl_roles: List[str], caller_roles: Optional[Sequence[str]]
    ) -> bool:
        if not acl_roles:
            return True
        if not caller_roles:
            return False
        caller_set = set(caller_roles)
        return any(r in caller_set for r in acl_roles)
