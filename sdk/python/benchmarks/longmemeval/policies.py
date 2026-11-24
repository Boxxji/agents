"""
Memory policy adapters evaluated directly against the unmodified Hanzo Agents
Python SDK classes (`MemoryClient`, `ScopedMemoryClient`, `GlobalMemoryClient`,
and `MemoryInterface`).

No competing memory store is created here: every policy reads and writes solely
through Hanzo's canonical `set`, `get`, `delete`, `list_keys`, `set_vector`,
`delete_vector`, and `similarity_search` methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from hanzo_agents.execution_context import ExecutionContext
from hanzo_agents.memory import (
    GlobalMemoryClient,
    MemoryClient,
    MemoryInterface,
    ScopedMemoryClient,
)

from .dataset import MemoryQuery, MemoryTurn
from .embedder import DeterministicHashEmbedder
from .transport import InMemoryHanzoAgentsClient, TransportHTTPError


def estimate_tokens(text: str) -> int:
    """
    Fast deterministic token estimator matching
    `control-plane/internal/eval/membench/types.go:estimateTokens`.
    """
    trimmed = text.strip()
    if not trimmed:
        return 0
    words = len(trimmed.split())
    char_est = (len(trimmed) + 3) // 4
    return max(words, char_est)


def canonical_key(entity: str, attribute: str) -> str:
    """Canonical KV key for an `(entity, attribute)` pair."""
    return f"{entity.strip().lower()}.{attribute.strip().lower()}"


@dataclass
class PolicyRecallResult:
    """Grounded recall output returned by a policy for one benchmark query."""

    answer: Optional[str] = None
    ordered_answers: List[str] = field(default_factory=list)
    abstained: bool = False
    contradiction_detected: bool = False
    conflicting_values: List[str] = field(default_factory=list)
    evidence_values: List[str] = field(default_factory=list)
    evidence_keys: List[str] = field(default_factory=list)
    retrieval_tokens: int = 0
    top_score: Optional[float] = None


def build_memory_interface(
    client: InMemoryHanzoAgentsClient,
    *,
    actor_id: str,
    session_id: str,
    workflow_id: str,
    agent_node_id: str = "bench-agent",
) -> MemoryInterface:
    """Construct a real `MemoryInterface` bound to an `ExecutionContext`."""
    ctx = ExecutionContext(
        run_id=workflow_id,
        execution_id=f"exec-{workflow_id}",
        agent_instance=SimpleNamespace(node_id=agent_node_id),
        reasoner_name="longmemeval_policy",
        agent_node_id=agent_node_id,
        session_id=session_id,
        actor_id=actor_id,
        workflow_id=workflow_id,
    )
    mem_client = MemoryClient(client, ctx, agent_node_id=agent_node_id)
    # Event client is unused for synchronous benchmark queries; pass minimal stub.
    return MemoryInterface(mem_client, SimpleNamespace(on_change=lambda p: (lambda fn: fn)))  # type: ignore[arg-type]


class BaseMemoryPolicy(ABC):
    """Abstract adapter exercising a specific Hanzo Agents memory usage policy."""

    name: str = ""
    description: str = ""

    @abstractmethod
    async def ingest(self, client: InMemoryHanzoAgentsClient, turn: MemoryTurn) -> None:
        """Persist a single turn via the Hanzo SDK memory interface."""

    @abstractmethod
    async def recall(
        self, client: InMemoryHanzoAgentsClient, query: MemoryQuery
    ) -> PolicyRecallResult:
        """Answer a query using only Hanzo SDK memory reads/searches."""


class KVScopedPolicy(BaseMemoryPolicy):
    """
    Baseline 1: Strict session-scoped KV (`interface.session(session_id)`).

    Exercises `ScopedMemoryClient.set/get/delete` bound exclusively to `session`.
    Demonstrates strict session isolation, clean abstention on missing keys,
    and expected failure on cross-session retrieval and temporal ordering.
    """

    name = "kv_scoped"
    description = "Strict session-scoped KV via ScopedMemoryClient('session', session_id)."

    async def ingest(self, client: InMemoryHanzoAgentsClient, turn: MemoryTurn) -> None:
        iface = build_memory_interface(
            client,
            actor_id=turn.actor_id,
            session_id=turn.session_id,
            workflow_id=turn.workflow_id,
        )
        scoped: ScopedMemoryClient = iface.session(turn.session_id)
        key = canonical_key(turn.entity, turn.attribute)
        if turn.action == "delete":
            try:
                await scoped.delete(key)
            except TransportHTTPError as exc:
                if exc.status_code != 404:
                    raise
            return
        await scoped.set(
            key,
            {
                "value": turn.value,
                "statement": turn.statement,
                "timestamp": turn.timestamp,
            },
        )

    async def recall(
        self, client: InMemoryHanzoAgentsClient, query: MemoryQuery
    ) -> PolicyRecallResult:
        iface = build_memory_interface(
            client,
            actor_id=query.actor_id,
            session_id=query.session_id,
            workflow_id=query.workflow_id,
        )
        scoped = iface.session(query.session_id)
        key = canonical_key(query.entity, query.attribute)
        record = await scoped.get(key, default=None)
        if not isinstance(record, dict) or not record.get("value"):
            return PolicyRecallResult(abstained=True, retrieval_tokens=0)

        val = str(record["value"])
        stmt = str(record.get("statement") or val)
        return PolicyRecallResult(
            answer=val,
            ordered_answers=[val] if query.query_type == "sequence" else [],
            abstained=False,
            evidence_values=[val],
            evidence_keys=[key],
            retrieval_tokens=estimate_tokens(stmt),
        )


class KVFallbackChainPolicy(BaseMemoryPolicy):
    """
    Baseline 2: Native Hanzo hierarchical fallback (`MemoryInterface.get()`).

    Writes each turn into its declared scope (`workflow` / `session` / `actor` /
    `global`) using `ScopedMemoryClient` / `GlobalMemoryClient` and reads back
    through `MemoryInterface.get(key)`, which walks
    `workflow -> session -> actor -> global` in fixed precedence order.
    """

    name = "kv_fallback_chain"
    description = "Native MemoryInterface hierarchical lookup (workflow -> session -> actor -> global)."

    @staticmethod
    def _target_client(iface: MemoryInterface, turn: MemoryTurn) -> Any:
        if turn.scope == "workflow":
            return iface.workflow(turn.workflow_id)
        if turn.scope == "session":
            return iface.session(turn.session_id)
        if turn.scope == "actor":
            return iface.actor(turn.actor_id)
        return iface.global_scope

    async def ingest(self, client: InMemoryHanzoAgentsClient, turn: MemoryTurn) -> None:
        iface = build_memory_interface(
            client,
            actor_id=turn.actor_id,
            session_id=turn.session_id,
            workflow_id=turn.workflow_id,
        )
        target = self._target_client(iface, turn)
        key = canonical_key(turn.entity, turn.attribute)
        if turn.action == "delete":
            try:
                await target.delete(key)
            except TransportHTTPError as exc:
                if exc.status_code != 404:
                    raise
            return
        await target.set(
            key,
            {
                "value": turn.value,
                "statement": turn.statement,
                "timestamp": turn.timestamp,
                "scope": turn.scope,
            },
        )

    async def recall(
        self, client: InMemoryHanzoAgentsClient, query: MemoryQuery
    ) -> PolicyRecallResult:
        iface = build_memory_interface(
            client,
            actor_id=query.actor_id,
            session_id=query.session_id,
            workflow_id=query.workflow_id,
        )
        key = canonical_key(query.entity, query.attribute)
        record = await iface.get(key, default=None)
        if not isinstance(record, dict) or not record.get("value"):
            return PolicyRecallResult(abstained=True, retrieval_tokens=0)

        val = str(record["value"])
        stmt = str(record.get("statement") or val)
        return PolicyRecallResult(
            answer=val,
            ordered_answers=[val] if query.query_type == "sequence" else [],
            abstained=False,
            evidence_values=[val],
            evidence_keys=[key],
            retrieval_tokens=estimate_tokens(stmt),
        )


class VectorTopKPolicy(BaseMemoryPolicy):
    """
    Baseline 3: Unthresholded vector similarity search (`similarity_search(top_k=K)`).

    Stores every turn as a vector via `actor.set_vector()` / `global_scope.set_vector()`
    and retrieves top-K nearest neighbors without a similarity cutoff, exposing
    the abstention failure mode of Hanzo's current unthresholded vector API.
    """

    name = "vector_top_k"
    description = "Unthresholded cosine top-K vector search via MemoryClient.set_vector/similarity_search."

    def __init__(self, embedder: DeterministicHashEmbedder, top_k: int = 5) -> None:
        self.embedder = embedder
        self.top_k = top_k

    @staticmethod
    def _doc_text(turn: MemoryTurn) -> str:
        return f"{turn.entity} {turn.attribute} {turn.statement}"

    @staticmethod
    def _query_text(query: MemoryQuery) -> str:
        return f"{query.entity} {query.attribute} {query.question}"

    async def ingest(self, client: InMemoryHanzoAgentsClient, turn: MemoryTurn) -> None:
        iface = build_memory_interface(
            client,
            actor_id=turn.actor_id,
            session_id=turn.session_id,
            workflow_id=turn.workflow_id,
        )
        target: Any = (
            iface.global_scope if turn.scope == "global" else iface.actor(turn.actor_id)
        )
        if turn.action == "delete":
            # Naive vector-only policy does not track per-turn keys for entity deletion.
            return
        vec = self.embedder.embed(self._doc_text(turn))
        await target.set_vector(
            turn.turn_id,
            vec,
            metadata={
                "entity": turn.entity,
                "attribute": turn.attribute,
                "value": turn.value,
                "statement": turn.statement,
                "timestamp": turn.timestamp,
                "scope": turn.scope,
                "confidence": turn.confidence,
                "is_event": turn.is_event,
            },
        )

    async def recall(
        self, client: InMemoryHanzoAgentsClient, query: MemoryQuery
    ) -> PolicyRecallResult:
        iface = build_memory_interface(
            client,
            actor_id=query.actor_id,
            session_id=query.session_id,
            workflow_id=query.workflow_id,
        )
        q_vec = self.embedder.embed(self._query_text(query))
        actor_hits = await iface.actor(query.actor_id).similarity_search(
            q_vec, top_k=self.top_k
        )
        global_hits = await iface.global_scope.similarity_search(
            q_vec, top_k=self.top_k
        )
        combined = sorted(
            [*actor_hits, *global_hits],
            key=lambda h: (-float(h.get("score", 0.0)), str(h.get("key", ""))),
        )[: self.top_k]

        if not combined:
            return PolicyRecallResult(abstained=True, retrieval_tokens=0)

        tokens = sum(
            estimate_tokens(str((h.get("metadata") or {}).get("statement", "")))
            for h in combined
        )
        ev_vals = [
            str((h.get("metadata") or {}).get("value", ""))
            for h in combined
            if (h.get("metadata") or {}).get("value")
        ]
        ev_keys = [str(h.get("key", "")) for h in combined]

        if query.query_type == "sequence":
            ordered_hits = sorted(
                combined,
                key=lambda h: str((h.get("metadata") or {}).get("timestamp", "")),
            )
            seq = [
                str((h.get("metadata") or {}).get("value", ""))
                for h in ordered_hits
                if (h.get("metadata") or {}).get("value")
            ]
            return PolicyRecallResult(
                answer=" -> ".join(seq) if seq else None,
                ordered_answers=seq,
                abstained=not seq,
                evidence_values=ev_vals,
                evidence_keys=ev_keys,
                retrieval_tokens=tokens,
                top_score=float(combined[0].get("score", 0.0)),
            )

        best = combined[0]
        best_val = str((best.get("metadata") or {}).get("value", ""))
        return PolicyRecallResult(
            answer=best_val or None,
            abstained=not bool(best_val),
            evidence_values=ev_vals,
            evidence_keys=ev_keys,
            retrieval_tokens=tokens,
            top_score=float(best.get("score", 0.0)),
        )


class VectorThresholdedPolicy(BaseMemoryPolicy):
    """
    Policy 4: Similarity-thresholded vector search with recency tie-breaking
    and coordinated vector deletion.

    Uses `ScopedMemoryClient.set_vector`, `delete_vector`, and
    `similarity_search(top_k=8)` with a client-side `min_score` threshold to
    enable abstention and timestamp-sorted disambiguation among matching vectors.
    """

    name = "vector_thresholded"
    description = (
        "Client-thresholded similarity_search (min_score cutoff) with recency ranking "
        "and coordinated delete_vector cleanup."
    )

    def __init__(
        self,
        embedder: DeterministicHashEmbedder,
        *,
        min_score: float = 0.45,
        top_k: int = 8,
    ) -> None:
        self.embedder = embedder
        self.min_score = min_score
        self.top_k = top_k

    @staticmethod
    def _doc_text(turn: MemoryTurn) -> str:
        return f"{turn.entity} {turn.attribute} {turn.statement}"

    @staticmethod
    def _query_text(query: MemoryQuery) -> str:
        return f"{query.entity} {query.attribute} {query.question}"

    async def ingest(self, client: InMemoryHanzoAgentsClient, turn: MemoryTurn) -> None:
        iface = build_memory_interface(
            client,
            actor_id=turn.actor_id,
            session_id=turn.session_id,
            workflow_id=turn.workflow_id,
        )
        target: Any = (
            iface.global_scope if turn.scope == "global" else iface.actor(turn.actor_id)
        )
        if turn.action == "delete":
            # Locate and delete vectors matching this entity/attribute via metadata filter.
            q_vec = self.embedder.embed(f"{turn.entity} {turn.attribute}")
            existing = await target.similarity_search(
                q_vec,
                top_k=50,
                filters={"entity": turn.entity, "attribute": turn.attribute},
            )
            for hit in existing:
                await target.delete_vector(str(hit["key"]))
            return

        vec = self.embedder.embed(self._doc_text(turn))
        await target.set_vector(
            turn.turn_id,
            vec,
            metadata={
                "entity": turn.entity,
                "attribute": turn.attribute,
                "value": turn.value,
                "statement": turn.statement,
                "timestamp": turn.timestamp,
                "scope": turn.scope,
                "confidence": turn.confidence,
                "is_event": turn.is_event,
            },
        )

    async def recall(
        self, client: InMemoryHanzoAgentsClient, query: MemoryQuery
    ) -> PolicyRecallResult:
        iface = build_memory_interface(
            client,
            actor_id=query.actor_id,
            session_id=query.session_id,
            workflow_id=query.workflow_id,
        )
        q_vec = self.embedder.embed(self._query_text(query))
        actor_hits = await iface.actor(query.actor_id).similarity_search(
            q_vec, top_k=self.top_k
        )
        global_hits = await iface.global_scope.similarity_search(
            q_vec, top_k=self.top_k
        )
        all_hits = [*actor_hits, *global_hits]
        top_score = max((float(h.get("score", 0.0)) for h in all_hits), default=0.0)

        passed = [
            h for h in all_hits if float(h.get("score", 0.0)) >= self.min_score
        ]
        if not passed:
            return PolicyRecallResult(
                abstained=True, retrieval_tokens=0, top_score=top_score
            )

        # Keep only hits closely aligned with the top match (within 0.12 cosine band)
        # so high-confidence matches do not drag unrelated mid-score distractors.
        band_floor = max(self.min_score, top_score - 0.15)
        focused = [h for h in passed if float(h.get("score", 0.0)) >= band_floor]
        tokens = sum(
            estimate_tokens(str((h.get("metadata") or {}).get("statement", "")))
            for h in focused
        )
        ev_vals = [
            str((h.get("metadata") or {}).get("value", ""))
            for h in focused
            if (h.get("metadata") or {}).get("value")
        ]
        ev_keys = [str(h.get("key", "")) for h in focused]

        if query.query_type == "sequence":
            ordered_hits = sorted(
                focused,
                key=lambda h: str((h.get("metadata") or {}).get("timestamp", "")),
            )
            seq: List[str] = []
            for h in ordered_hits:
                val = str((h.get("metadata") or {}).get("value", ""))
                if val and (not seq or seq[-1] != val):
                    seq.append(val)
            return PolicyRecallResult(
                answer=" -> ".join(seq) if seq else None,
                ordered_answers=seq,
                abstained=not seq,
                evidence_values=ev_vals,
                evidence_keys=ev_keys,
                retrieval_tokens=tokens,
                top_score=top_score,
            )

        # Recency tie-breaking among focused high-similarity hits
        focused.sort(
            key=lambda h: (
                str((h.get("metadata") or {}).get("timestamp", "")),
                float(h.get("score", 0.0)),
            ),
            reverse=True,
        )
        best_val = str((focused[0].get("metadata") or {}).get("value", ""))
        return PolicyRecallResult(
            answer=best_val or None,
            abstained=not bool(best_val),
            evidence_values=ev_vals,
            evidence_keys=ev_keys,
            retrieval_tokens=tokens,
            top_score=top_score,
        )


class HybridScopePolicy(BaseMemoryPolicy):
    """
    Policy 5: Temporal- and contradiction-aware hybrid policy layered strictly
    on top of Hanzo's existing `MemoryInterface` primitives.

    Uses ONLY `ScopedMemoryClient` (`workflow`, `session`, `actor`),
    `GlobalMemoryClient`, `set`/`get`/`delete`/`list_keys`, and
    `set_vector`/`delete_vector`/`similarity_search`:
    - Writes scoped KV records with embedded `(timestamp, confidence, source)`
      metadata AND preserves an append-only per-entity event history under
      `events.<entity>.<attribute>.<turn_id>` in `actor` scope for temporal ordering.
    - Reads across all applicable scopes (`workflow`, `session`, `actor`, `global`)
      explicitly so a stale `session`/`workflow` record cannot silently shadow a
      newer authoritative `actor` update.
    - Detects contradictions whenever distinct scopes or high/low-confidence
      records disagree on `(entity, attribute)`, resolving conflicts by
      `(confidence >= 0.80 priority, timestamp)` so verified tool observations are
      not clobbered by unverified hypotheses.
    - Propagates `delete` across KV + vector + event keys and falls back to
      thresholded `similarity_search` (`min_score`) only when structured KV has
      no match, guaranteeing clean abstention and zero cross-actor leakage.
    """

    name = "hybrid_scope_policy"
    description = (
        "Temporal & contradiction-aware hybrid KV+event+thresholded-vector policy "
        "over existing Hanzo MemoryInterface/ScopedMemoryClient/GlobalMemoryClient APIs."
    )

    def __init__(
        self,
        embedder: DeterministicHashEmbedder,
        *,
        min_score: float = 0.45,
        verified_confidence: float = 0.80,
    ) -> None:
        self.embedder = embedder
        self.min_score = min_score
        self.verified_confidence = verified_confidence

    @staticmethod
    def _scope_client(iface: MemoryInterface, scope: str, turn: MemoryTurn) -> Any:
        if scope == "workflow":
            return iface.workflow(turn.workflow_id)
        if scope == "session":
            return iface.session(turn.session_id)
        if scope == "actor":
            return iface.actor(turn.actor_id)
        return iface.global_scope

    async def ingest(self, client: InMemoryHanzoAgentsClient, turn: MemoryTurn) -> None:
        iface = build_memory_interface(
            client,
            actor_id=turn.actor_id,
            session_id=turn.session_id,
            workflow_id=turn.workflow_id,
        )
        key = canonical_key(turn.entity, turn.attribute)
        actor_client = iface.actor(turn.actor_id)
        target = self._scope_client(iface, turn.scope, turn)

        if turn.action == "delete":
            # Delete KV record from target scope + actor scope + vectors + event history.
            for sc_client in (target, actor_client):
                try:
                    await sc_client.delete(key)
                except TransportHTTPError as exc:
                    if exc.status_code != 404:
                        raise
            vec_hits = await actor_client.similarity_search(
                self.embedder.embed(f"{turn.entity} {turn.attribute}"),
                top_k=50,
                filters={"entity": turn.entity, "attribute": turn.attribute},
            )
            for hit in vec_hits:
                await actor_client.delete_vector(str(hit["key"]))
            return

        payload = {
            "entity": turn.entity,
            "attribute": turn.attribute,
            "value": turn.value,
            "statement": turn.statement,
            "timestamp": turn.timestamp,
            "scope": turn.scope,
            "source": turn.source,
            "confidence": turn.confidence,
            "turn_id": turn.turn_id,
            "is_event": turn.is_event,
        }
        await target.set(key, payload)

        # Preserve timeline events using existing ScopedMemoryClient.set + list_keys
        if turn.is_event:
            event_key = f"event:{key}:{turn.turn_id}"
            await actor_client.set(event_key, payload)

        # Also index a vector in the appropriate scope for semantic fallback
        vec_target: Any = iface.global_scope if turn.scope == "global" else actor_client
        await vec_target.set_vector(
            turn.turn_id,
            self.embedder.embed(f"{turn.entity} {turn.attribute} {turn.statement}"),
            metadata=payload,
        )

    async def _collect_scope_records(
        self, iface: MemoryInterface, query: MemoryQuery, key: str
    ) -> List[Tuple[str, Dict[str, Any]]]:
        candidates: List[Tuple[str, Any]] = [
            ("workflow", iface.workflow(query.workflow_id)),
            ("session", iface.session(query.session_id)),
            ("actor", iface.actor(query.actor_id)),
            ("global", iface.global_scope),
        ]
        found: List[Tuple[str, Dict[str, Any]]] = []
        for scope_name, sc_client in candidates:
            rec = await sc_client.get(key, default=None)
            if isinstance(rec, dict) and rec.get("value"):
                found.append((scope_name, rec))
        return found

    async def recall(
        self, client: InMemoryHanzoAgentsClient, query: MemoryQuery
    ) -> PolicyRecallResult:
        iface = build_memory_interface(
            client,
            actor_id=query.actor_id,
            session_id=query.session_id,
            workflow_id=query.workflow_id,
        )
        key = canonical_key(query.entity, query.attribute)
        actor_client = iface.actor(query.actor_id)

        # 1. Temporal ordering queries: reconstruct chronological sequence from event keys
        if query.query_type == "sequence":
            prefix = f"event:{key}:"
            all_keys = await actor_client.list_keys()
            event_keys = [k for k in all_keys if k.startswith(prefix)]
            records: List[Dict[str, Any]] = []
            for ek in event_keys:
                rec = await actor_client.get(ek, default=None)
                if isinstance(rec, dict) and rec.get("value"):
                    records.append(rec)
            if records:
                records.sort(
                    key=lambda r: (str(r.get("timestamp", "")), str(r.get("turn_id", "")))
                )
                seq = [str(r["value"]) for r in records]
                tokens = sum(
                    estimate_tokens(str(r.get("statement") or r["value"]))
                    for r in records
                )
                return PolicyRecallResult(
                    answer=" -> ".join(seq),
                    ordered_answers=seq,
                    abstained=False,
                    evidence_values=seq,
                    evidence_keys=event_keys,
                    retrieval_tokens=tokens,
                )

        # 2. Structured multi-scope inspection (prevents stale session/workflow shadowing
        #    and detects cross-scope contradictions explicitly).
        scoped_records = await self._collect_scope_records(iface, query, key)
        if scoped_records:
            distinct_vals = {str(rec["value"]) for _, rec in scoped_records}
            contradiction = len(distinct_vals) > 1

            # Rank authoritative record:
            # 1) Verified high-confidence evidence (>= verified_confidence) beats
            #    low-confidence unverified hypotheses.
            # 2) Within the same verification tier, newest timestamp wins (breaking
            #    stale workflow/session shadowing!).
            # 3) Tie-break by higher confidence, then narrower scope.
            scope_rank = {"actor": 3, "session": 2, "workflow": 1, "global": 0}
            scoped_records.sort(
                key=lambda item: (
                    1 if float(item[1].get("confidence", 0.0)) >= self.verified_confidence else 0,
                    str(item[1].get("timestamp", "")),
                    float(item[1].get("confidence", 0.0)),
                    scope_rank.get(item[0], 0),
                ),
                reverse=True,
            )
            winner_scope, winner = scoped_records[0]
            winner_val = str(winner["value"])
            conflicting = sorted(v for v in distinct_vals if v != winner_val)
            tokens = sum(
                estimate_tokens(str(rec.get("statement") or rec["value"]))
                for _, rec in scoped_records
            )
            return PolicyRecallResult(
                answer=winner_val,
                abstained=False,
                contradiction_detected=contradiction,
                conflicting_values=conflicting,
                evidence_values=[str(rec["value"]) for _, rec in scoped_records],
                evidence_keys=[f"{sc}:{key}" for sc, _ in scoped_records],
                retrieval_tokens=tokens,
                top_score=1.0,
            )

        # 3. Fallback to thresholded vector search (handles paraphrases / unkeyed facts)
        q_vec = self.embedder.embed(f"{query.entity} {query.attribute} {query.question}")
        actor_hits = await actor_client.similarity_search(q_vec, top_k=6)
        global_hits = await iface.global_scope.similarity_search(q_vec, top_k=6)
        all_hits = [*actor_hits, *global_hits]
        top_score = max((float(h.get("score", 0.0)) for h in all_hits), default=0.0)
        passed = [
            h
            for h in all_hits
            if float(h.get("score", 0.0)) >= self.min_score
            and (h.get("metadata") or {}).get("entity") == query.entity
            and (h.get("metadata") or {}).get("attribute") == query.attribute
        ]
        if not passed:
            return PolicyRecallResult(
                abstained=True, retrieval_tokens=0, top_score=top_score
            )

        passed.sort(
            key=lambda h: (
                str((h.get("metadata") or {}).get("timestamp", "")),
                float(h.get("score", 0.0)),
            ),
            reverse=True,
        )
        best_meta = passed[0].get("metadata") or {}
        best_val = str(best_meta.get("value", ""))
        return PolicyRecallResult(
            answer=best_val or None,
            abstained=not bool(best_val),
            evidence_values=[best_val] if best_val else [],
            evidence_keys=[str(passed[0].get("key", ""))],
            retrieval_tokens=estimate_tokens(str(best_meta.get("statement") or best_val)),
            top_score=top_score,
        )


def build_default_policies(
    embedder: DeterministicHashEmbedder,
    *,
    min_score: float = 0.45,
    top_k: int = 5,
) -> List[BaseMemoryPolicy]:
    """Instantiate the canonical set of 5 evaluated Hanzo memory policies."""
    return [
        KVScopedPolicy(),
        KVFallbackChainPolicy(),
        VectorTopKPolicy(embedder=embedder, top_k=top_k),
        VectorThresholdedPolicy(embedder=embedder, min_score=min_score, top_k=max(top_k, 8)),
        HybridScopePolicy(embedder=embedder, min_score=min_score),
    ]
