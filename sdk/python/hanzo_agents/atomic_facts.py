"""
Mem0-style Atomic Fact Memory with deduplication, selective writes,
strict pre-retrieval tenant/RBAC isolation, and derived index rebuildability.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, Sequence, Tuple

FactLifecycle = Literal["ACTIVE", "SUPERSEDED", "PENDING_REVIEW"]
FactDecisionAction = Literal["ADD", "UPDATE", "NOOP", "REVIEW"]

FACT_KEY_PREFIX = "fact_record:"
FACT_SLOT_PREFIX = "fact_slot:"


def _normalize_token(value: str) -> str:
    return " ".join(value.strip().lower().split())


def compute_slot_hash(scope: str, scope_id: str, subject: str, predicate: str) -> str:
    canonical = f"{scope}:{scope_id}:{_normalize_token(subject)}:{_normalize_token(predicate)}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


def compute_fact_id(
    scope: str,
    scope_id: str,
    subject: str,
    predicate: str,
    obj: str,
    version: int,
) -> str:
    canonical = (
        f"{scope}:{scope_id}:{_normalize_token(subject)}:"
        f"{_normalize_token(predicate)}:{_normalize_token(obj)}:v{version}"
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    length = min(len(a), len(b))
    if length == 0:
        return 0.0
    dot = sum(float(a[i]) * float(b[i]) for i in range(length))
    norm_a = math.sqrt(sum(float(a[i]) ** 2 for i in range(length)))
    norm_b = math.sqrt(sum(float(b[i]) ** 2 for i in range(length)))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass(frozen=True)
class AtomicFactInput:
    subject: str
    predicate: str
    object: str
    source_identity: str
    source_version: int = 1
    confidence: float = 1.0
    acl_roles: List[str] = field(default_factory=list)
    summary: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class AtomicFactRecord:
    id: str
    slot_key: str
    subject: str
    predicate: str
    object: str
    source_identity: str
    source_version: int
    scope: str
    scope_id: str
    confidence: float
    version: int
    lifecycle: FactLifecycle
    acl_roles: List[str] = field(default_factory=list)
    supersedes_id: Optional[str] = None
    summary: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AtomicFactRecord":
        return cls(**data)


@dataclass(frozen=True)
class FactDecision:
    action: FactDecisionAction
    reason: str
    target_fact_id: Optional[str] = None
    record: Optional[AtomicFactRecord] = None
    superseded_record: Optional[AtomicFactRecord] = None


class FactDecisionModel(Protocol):
    def decide(
        self,
        candidate: AtomicFactInput,
        existing_slot_facts: List[AtomicFactRecord],
        similar_facts: List[Tuple[AtomicFactRecord, float]],
    ) -> Tuple[FactDecisionAction, str, Optional[AtomicFactRecord]]: ...


class DefaultFactDecisionModel:
    """Deterministic structured-decision model ('Kai' reconciliation judge)."""

    def __init__(
        self,
        review_confidence_threshold: float = 0.5,
        semantic_duplicate_threshold: float = 0.96,
    ):
        self.review_threshold = review_confidence_threshold
        self.duplicate_threshold = semantic_duplicate_threshold

    def decide(
        self,
        candidate: AtomicFactInput,
        existing_slot_facts: List[AtomicFactRecord],
        similar_facts: List[Tuple[AtomicFactRecord, float]],
    ) -> Tuple[FactDecisionAction, str, Optional[AtomicFactRecord]]:
        if candidate.confidence < self.review_threshold:
            return (
                "REVIEW",
                f"Candidate confidence ({candidate.confidence}) below threshold ({self.review_threshold})",
                None,
            )

        norm_s = _normalize_token(candidate.subject)
        norm_p = _normalize_token(candidate.predicate)
        norm_o = _normalize_token(candidate.object)

        active_slot = [f for f in existing_slot_facts if f.lifecycle == "ACTIVE"]

        for existing in active_slot:
            if (
                _normalize_token(existing.subject) == norm_s
                and _normalize_token(existing.predicate) == norm_p
                and _normalize_token(existing.object) == norm_o
            ):
                return ("NOOP", "Identical active atomic fact already exists", existing)

        for sim_rec, score in similar_facts:
            if (
                sim_rec.lifecycle == "ACTIVE"
                and score >= self.duplicate_threshold
                and _normalize_token(sim_rec.subject) == norm_s
                and _normalize_token(sim_rec.predicate) == norm_p
                and _normalize_token(sim_rec.object) == norm_o
            ):
                return ("NOOP", f"Semantic duplicate detected (score={score:.4f})", sim_rec)

        if active_slot:
            latest = max(active_slot, key=lambda r: r.version)
            if candidate.confidence < latest.confidence * 0.7:
                return (
                    "REVIEW",
                    f"Contradictory update has much lower confidence ({candidate.confidence}) than v{latest.version}",
                    latest,
                )
            return (
                "UPDATE",
                f"Reconciling (subject, predicate) slot from v{latest.version} to v{latest.version + 1}",
                latest,
            )

        return ("ADD", "Novel atomic fact for scope", None)


class AtomicFactMemory:
    """
    Event-driven atomic fact writer and retriever with deduplication, selective writes,
    strict tenant/RBAC pre-filtering, and derived vector index rebuildability.
    """

    def __init__(
        self,
        default_scope: str = "workflow",
        default_scope_id: str = "default",
        decision_model: Optional[FactDecisionModel] = None,
        review_confidence_threshold: float = 0.5,
        semantic_duplicate_threshold: float = 0.96,
    ):
        self.default_scope = default_scope
        self.default_scope_id = default_scope_id
        self.decision_model: FactDecisionModel = decision_model or DefaultFactDecisionModel(
            review_confidence_threshold=review_confidence_threshold,
            semantic_duplicate_threshold=semantic_duplicate_threshold,
        )
        # Durable record ledger vs derived vector index
        self._records: Dict[str, Dict[str, Any]] = {}
        self._slots: Dict[str, List[str]] = {}
        self._vectors: Dict[str, Tuple[str, str, List[float], Dict[str, Any]]] = {}

        self.events_processed = 0
        self.durable_writes = 0
        self.add_count = 0
        self.update_count = 0
        self.noop_count = 0
        self.review_count = 0

    def ingest_fact(
        self,
        candidate: AtomicFactInput,
        scope: Optional[str] = None,
        scope_id: Optional[str] = None,
        embedding: Optional[Sequence[float]] = None,
    ) -> FactDecision:
        self.events_processed += 1
        sc = scope or self.default_scope
        sid = scope_id or self.default_scope_id
        slot_hash = compute_slot_hash(sc, sid, candidate.subject, candidate.predicate)
        slot_storage_key = f"{sc}:{sid}:{FACT_SLOT_PREFIX}{slot_hash}"

        slot_ids = list(self._slots.get(slot_storage_key, []))
        existing_slot_facts: List[AtomicFactRecord] = []
        for fid in slot_ids:
            rec = self.get_fact_by_id(fid, scope=sc, scope_id=sid)
            if rec is not None:
                existing_slot_facts.append(rec)

        similar_facts: List[Tuple[AtomicFactRecord, float]] = []
        if embedding:
            for v_key, (v_sc, v_sid, v_emb, v_meta) in self._vectors.items():
                if v_sc != sc or v_sid != sid:
                    continue
                fid = v_meta.get("fact_id")
                if not fid:
                    continue
                rec = self.get_fact_by_id(fid, scope=sc, scope_id=sid)
                if rec is not None:
                    score = _cosine_similarity(embedding, v_emb)
                    similar_facts.append((rec, score))
            similar_facts.sort(key=lambda item: item[1], reverse=True)
            similar_facts = similar_facts[:5]

        action, reason, target_fact = self.decision_model.decide(
            candidate, existing_slot_facts, similar_facts
        )

        if action == "NOOP":
            self.noop_count += 1
            return FactDecision(
                action="NOOP",
                reason=reason,
                target_fact_id=target_fact.id if target_fact else None,
                record=target_fact,
            )

        now_iso = datetime.now(timezone.utc).isoformat()

        if action == "REVIEW":
            self.review_count += 1
            version = (target_fact.version if target_fact else 0) + 1
            fid = compute_fact_id(
                sc, sid, candidate.subject, candidate.predicate, candidate.object, version
            )
            review_rec = AtomicFactRecord(
                id=fid,
                slot_key=slot_hash,
                subject=candidate.subject.strip(),
                predicate=candidate.predicate.strip(),
                object=candidate.object.strip(),
                source_identity=candidate.source_identity,
                source_version=candidate.source_version,
                scope=sc,
                scope_id=sid,
                confidence=candidate.confidence,
                version=version,
                lifecycle="PENDING_REVIEW",
                acl_roles=list(candidate.acl_roles),
                supersedes_id=target_fact.id if target_fact else None,
                summary=candidate.summary,
                metadata=candidate.metadata,
                created_at=now_iso,
                updated_at=now_iso,
            )
            self._persist(review_rec, slot_storage_key, slot_ids, None)
            return FactDecision(
                action="REVIEW",
                reason=reason,
                target_fact_id=target_fact.id if target_fact else None,
                record=review_rec,
            )

        if action == "UPDATE" and target_fact is not None:
            self.update_count += 1
            superseded_rec = AtomicFactRecord(
                **{
                    **target_fact.to_dict(),
                    "lifecycle": "SUPERSEDED",
                    "updated_at": now_iso,
                }
            )
            prior_storage_key = f"{sc}:{sid}:{FACT_KEY_PREFIX}{target_fact.id}"
            self._records[prior_storage_key] = superseded_rec.to_dict()
            self.durable_writes += 1
            self._vectors.pop(prior_storage_key, None)

            next_version = target_fact.version + 1
            new_id = compute_fact_id(
                sc, sid, candidate.subject, candidate.predicate, candidate.object, next_version
            )
            updated_rec = AtomicFactRecord(
                id=new_id,
                slot_key=slot_hash,
                subject=candidate.subject.strip(),
                predicate=candidate.predicate.strip(),
                object=candidate.object.strip(),
                source_identity=candidate.source_identity,
                source_version=candidate.source_version,
                scope=sc,
                scope_id=sid,
                confidence=candidate.confidence,
                version=next_version,
                lifecycle="ACTIVE",
                acl_roles=list(candidate.acl_roles),
                supersedes_id=target_fact.id,
                summary=candidate.summary,
                metadata=candidate.metadata,
                created_at=now_iso,
                updated_at=now_iso,
            )
            self._persist(updated_rec, slot_storage_key, slot_ids, embedding)
            return FactDecision(
                action="UPDATE",
                reason=reason,
                target_fact_id=target_fact.id,
                record=updated_rec,
                superseded_record=superseded_rec,
            )

        # Default: ADD
        self.add_count += 1
        new_id = compute_fact_id(
            sc, sid, candidate.subject, candidate.predicate, candidate.object, 1
        )
        added_rec = AtomicFactRecord(
            id=new_id,
            slot_key=slot_hash,
            subject=candidate.subject.strip(),
            predicate=candidate.predicate.strip(),
            object=candidate.object.strip(),
            source_identity=candidate.source_identity,
            source_version=candidate.source_version,
            scope=sc,
            scope_id=sid,
            confidence=candidate.confidence,
            version=1,
            lifecycle="ACTIVE",
            acl_roles=list(candidate.acl_roles),
            summary=candidate.summary,
            metadata=candidate.metadata,
            created_at=now_iso,
            updated_at=now_iso,
        )
        self._persist(added_rec, slot_storage_key, slot_ids, embedding)
        return FactDecision(action="ADD", reason=reason, record=added_rec)

    def query_facts(
        self,
        scope: Optional[str] = None,
        scope_id: Optional[str] = None,
        caller_roles: Optional[Sequence[str]] = None,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        query_embedding: Optional[Sequence[float]] = None,
        top_k: int = 10,
        include_lifecycle: Optional[Sequence[FactLifecycle]] = None,
    ) -> List[AtomicFactRecord]:
        sc = scope or self.default_scope
        sid = scope_id or self.default_scope_id
        allowed_lifecycle = set(include_lifecycle or ["ACTIVE"])

        candidates: List[AtomicFactRecord] = []
        if query_embedding is not None:
            scored: List[Tuple[float, AtomicFactRecord]] = []
            for _, (v_sc, v_sid, v_emb, v_meta) in self._vectors.items():
                if v_sc != sc or v_sid != sid:
                    continue
                fid = v_meta.get("fact_id")
                if not fid:
                    continue
                rec = self.get_fact_by_id(fid, scope=sc, scope_id=sid)
                if rec is not None:
                    scored.append((_cosine_similarity(query_embedding, v_emb), rec))
            scored.sort(key=lambda item: item[0], reverse=True)
            candidates = [rec for _, rec in scored]
        else:
            prefix = f"{sc}:{sid}:{FACT_KEY_PREFIX}"
            for key, raw in self._records.items():
                if key.startswith(prefix):
                    candidates.append(AtomicFactRecord.from_dict(raw))

        results: List[AtomicFactRecord] = []
        for rec in candidates:
            if rec.scope != sc or rec.scope_id != sid:
                continue
            if rec.lifecycle not in allowed_lifecycle:
                continue
            if subject and _normalize_token(rec.subject) != _normalize_token(subject):
                continue
            if predicate and _normalize_token(rec.predicate) != _normalize_token(predicate):
                continue
            if not self._is_authorized(rec, caller_roles):
                continue
            results.append(rec)

        return results[:top_k]

    def rebuild_derived_index(
        self,
        embed_fn: Callable[[str], Sequence[float]],
        scope: Optional[str] = None,
        scope_id: Optional[str] = None,
    ) -> Dict[str, int]:
        sc = scope or self.default_scope
        sid = scope_id or self.default_scope_id
        prefix = f"{sc}:{sid}:{FACT_KEY_PREFIX}"

        # Clear existing derived vectors for scope
        for key in list(self._vectors.keys()):
            if key.startswith(prefix):
                self._vectors.pop(key, None)

        rebuilt = 0
        skipped = 0
        for key, raw in self._records.items():
            if not key.startswith(prefix):
                continue
            rec = AtomicFactRecord.from_dict(raw)
            if rec.lifecycle != "ACTIVE":
                skipped += 1
                continue
            vec = list(embed_fn(f"{rec.subject} {rec.predicate} {rec.object}"))
            self._vectors[key] = (
                sc,
                sid,
                vec,
                {"fact_id": rec.id, "slot_key": rec.slot_key, "version": rec.version},
            )
            rebuilt += 1

        return {"rebuilt_count": rebuilt, "skipped_count": skipped}

    def get_fact_by_id(
        self,
        fact_id: str,
        scope: Optional[str] = None,
        scope_id: Optional[str] = None,
    ) -> Optional[AtomicFactRecord]:
        sc = scope or self.default_scope
        sid = scope_id or self.default_scope_id
        raw = self._records.get(f"{sc}:{sid}:{FACT_KEY_PREFIX}{fact_id}")
        if raw is None:
            return None
        rec = AtomicFactRecord.from_dict(raw)
        if rec.scope != sc or rec.scope_id != sid:
            return None
        return rec

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "events_processed": self.events_processed,
            "durable_writes": self.durable_writes,
            "add_count": self.add_count,
            "update_count": self.update_count,
            "noop_count": self.noop_count,
            "review_count": self.review_count,
            "write_amplification": (
                round(self.durable_writes / self.events_processed, 4)
                if self.events_processed > 0
                else 0.0
            ),
        }

    def _persist(
        self,
        record: AtomicFactRecord,
        slot_storage_key: str,
        slot_ids: List[str],
        embedding: Optional[Sequence[float]],
    ) -> None:
        storage_key = f"{record.scope}:{record.scope_id}:{FACT_KEY_PREFIX}{record.id}"
        self._records[storage_key] = record.to_dict()
        self.durable_writes += 1

        if record.id not in slot_ids:
            slot_ids.append(record.id)
        self._slots[slot_storage_key] = slot_ids

        if record.lifecycle == "ACTIVE" and embedding is not None:
            self._vectors[storage_key] = (
                record.scope,
                record.scope_id,
                [float(x) for x in embedding],
                {"fact_id": record.id, "slot_key": record.slot_key, "version": record.version},
            )

    @staticmethod
    def _is_authorized(
        record: AtomicFactRecord, caller_roles: Optional[Sequence[str]]
    ) -> bool:
        if not record.acl_roles:
            return True
        if not caller_roles:
            return False
        caller_set = set(caller_roles)
        return any(role in caller_set for role in record.acl_roles)
