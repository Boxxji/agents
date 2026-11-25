"""
Clean-room Apache-2.0 Self-Organising Associative Memory with Bounded Consolidation.

Raw events stay immutable on a fast O(1) write path; asynchronous bounded
consolidation enriches derived associative nodes with contextual descriptions,
tags, and degree-bounded bidirectional links with lineage back to raw events.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

EventCategory = Literal["incident", "decision", "code_change", "lesson"]

RAW_EVENT_PREFIX = "amem_raw_event:"
ASSOC_NODE_PREFIX = "amem_node:"


def _norm(val: str) -> str:
    return " ".join(val.strip().lower().split())


def _extract_auto_tags(content: str, existing: Optional[Sequence[str]] = None) -> List[str]:
    words = [w for w in re.sub(r"[^a-z0-9_-]+", " ", content.lower()).split() if len(w) >= 4]
    stop = {"that", "this", "with", "from", "when", "were", "have", "been"}
    tags = [_norm(t) for t in (existing or [])] + [w for w in words if w not in stop][:8]
    return list(dict.fromkeys(tags))


def compute_raw_event_id(
    scope: str, scope_id: str, category: EventCategory, source_identity: str, content: str
) -> str:
    canonical = f"{scope}:{scope_id}:{category}:{_norm(source_identity)}:{_norm(content)}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class ImmutableRawEvent:
    id: str
    category: EventCategory
    content: str
    tags: List[str]
    source_identity: str
    source_version: int
    scope: str
    scope_id: str
    acl_roles: List[str] = field(default_factory=list)
    context_description: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImmutableRawEvent":
        return cls(**data)


@dataclass
class AssociativeMemoryNode:
    id: str
    event_id: str
    category: EventCategory
    content: str
    context_description: str
    tags: List[str]
    links: List[Dict[str, Any]]
    evidence_event_ids: List[str]
    scope: str
    scope_id: str
    acl_roles: List[str] = field(default_factory=list)
    version: int = 1
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AssociativeMemoryNode":
        return cls(**data)


class AssociativeMemory:
    def __init__(
        self,
        default_scope: str = "workflow",
        default_scope_id: str = "default",
        max_edges_per_node: int = 6,
        min_association_score: float = 0.2,
    ):
        self.default_scope = default_scope
        self.default_scope_id = default_scope_id
        self.max_edges_per_node = max_edges_per_node
        self.min_association_score = min_association_score
        self._raw_events: Dict[str, Dict[str, Any]] = {}
        self._pending_queues: Dict[str, List[str]] = {}
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._vectors: Dict[str, Tuple[str, str, List[float], Dict[str, Any]]] = {}

    def ingest_event(
        self,
        category: EventCategory,
        content: str,
        source_identity: str,
        tags: Optional[Sequence[str]] = None,
        context_description: Optional[str] = None,
        acl_roles: Optional[Sequence[str]] = None,
        scope: Optional[str] = None,
        scope_id: Optional[str] = None,
    ) -> ImmutableRawEvent:
        sc = scope or self.default_scope
        sid = scope_id or self.default_scope_id
        ev_id = compute_raw_event_id(sc, sid, category, source_identity, content)
        raw_key = f"{sc}:{sid}:{RAW_EVENT_PREFIX}{ev_id}"
        if raw_key in self._raw_events:
            return ImmutableRawEvent.from_dict(self._raw_events[raw_key])

        ev = ImmutableRawEvent(
            id=ev_id,
            category=category,
            content=content.strip(),
            tags=_extract_auto_tags(content, tags),
            source_identity=source_identity,
            source_version=1,
            scope=sc,
            scope_id=sid,
            acl_roles=list(acl_roles or []),
            context_description=context_description,
        )
        self._raw_events[raw_key] = ev.to_dict()
        q = self._pending_queues.setdefault(f"{sc}:{sid}", [])
        if ev_id not in q:
            q.append(ev_id)
        return ev

    def consolidate_bounded(
        self,
        max_batch_size: int = 25,
        max_edges_per_node: Optional[int] = None,
        embed_fn: Optional[Callable[[str], Sequence[float]]] = None,
        scope: Optional[str] = None,
        scope_id: Optional[str] = None,
    ) -> Dict[str, int]:
        sc = scope or self.default_scope
        sid = scope_id or self.default_scope_id
        max_edges = max_edges_per_node or self.max_edges_per_node
        q_key = f"{sc}:{sid}"
        queue = self._pending_queues.get(q_key, [])
        if not queue:
            return {
                "processed_events": 0,
                "nodes_updated": 0,
                "edges_created_or_updated": 0,
                "edges_pruned_by_bound": 0,
            }

        batch = queue[:max_batch_size]
        self._pending_queues[q_key] = queue[max_batch_size:]

        prefix = f"{sc}:{sid}:{ASSOC_NODE_PREFIX}"
        node_map: Dict[str, AssociativeMemoryNode] = {
            raw["id"]: AssociativeMemoryNode.from_dict(raw)
            for k, raw in self._nodes.items()
            if k.startswith(prefix)
        }

        edges_updated = 0
        edges_pruned = 0
        touched = set()

        for ev_id in batch:
            raw_ev_dict = self._raw_events.get(f"{sc}:{sid}:{RAW_EVENT_PREFIX}{ev_id}")
            if not raw_ev_dict:
                continue
            ev = ImmutableRawEvent.from_dict(raw_ev_dict)
            nid = f"node_{ev.id}"
            ctx_desc = (
                ev.context_description
                or f"[{ev.category.upper()}] {ev.content} (tags: {', '.join(ev.tags)})"
            )
            node = node_map.get(nid) or AssociativeMemoryNode(
                id=nid,
                event_id=ev.id,
                category=ev.category,
                content=ev.content,
                context_description=ctx_desc,
                tags=list(ev.tags),
                links=[],
                evidence_event_ids=[ev.id],
                scope=sc,
                scope_id=sid,
                acl_roles=list(ev.acl_roles),
            )

            for other in list(node_map.values()):
                if other.id == node.id:
                    continue
                score, shared = self._score(node, other)
                if score < self.min_association_score:
                    continue
                rel = f"{node.category}<->{other.category}"
                evidence = list(dict.fromkeys([ev.id, other.event_id]))
                edges_pruned += self._upsert_link(
                    node,
                    {
                        "target_node_id": other.id,
                        "relation_type": rel,
                        "weight": score,
                        "shared_tags": shared,
                        "evidence_event_ids": evidence,
                    },
                    max_edges,
                )
                edges_pruned += self._upsert_link(
                    other,
                    {
                        "target_node_id": node.id,
                        "relation_type": rel,
                        "weight": score,
                        "shared_tags": shared,
                        "evidence_event_ids": evidence,
                    },
                    max_edges,
                )
                edges_updated += 2
                other.version += 1
                touched.add(other.id)

            node_map[node.id] = node
            touched.add(node.id)

        for nid in touched:
            n = node_map[nid]
            n_key = f"{sc}:{sid}:{ASSOC_NODE_PREFIX}{nid}"
            self._nodes[n_key] = n.to_dict()
            if embed_fn is not None:
                self._vectors[n_key] = (
                    sc,
                    sid,
                    [float(x) for x in embed_fn(n.context_description)],
                    {"node_id": n.id, "event_id": n.event_id},
                )

        return {
            "processed_events": len(batch),
            "nodes_updated": len(touched),
            "edges_created_or_updated": edges_updated,
            "edges_pruned_by_bound": edges_pruned,
        }

    def recall_associative(
        self,
        tags: Optional[Sequence[str]] = None,
        categories: Optional[Sequence[EventCategory]] = None,
        expand_hops: int = 0,
        caller_roles: Optional[Sequence[str]] = None,
        scope: Optional[str] = None,
        scope_id: Optional[str] = None,
    ) -> List[AssociativeMemoryNode]:
        sc = scope or self.default_scope
        sid = scope_id or self.default_scope_id
        prefix = f"{sc}:{sid}:{ASSOC_NODE_PREFIX}"
        req_tags = [_norm(t) for t in (tags or [])]
        allowed_cats = set(categories) if categories else None

        all_nodes: Dict[str, AssociativeMemoryNode] = {
            raw["id"]: AssociativeMemoryNode.from_dict(raw)
            for k, raw in self._nodes.items()
            if k.startswith(prefix)
        }

        seeds: List[AssociativeMemoryNode] = []
        for n in all_nodes.values():
            if not self._is_authorized(n.acl_roles, caller_roles):
                continue
            if allowed_cats and n.category not in allowed_cats:
                continue
            if req_tags and not any(t in n.tags for t in req_tags):
                continue
            seeds.append(n)

        result_map = {s.id: s for s in seeds}
        frontier = list(seeds)
        for _ in range(expand_hops):
            next_frontier = []
            for node in frontier:
                for link in node.links:
                    tid = link["target_node_id"]
                    if tid in result_map or tid not in all_nodes:
                        continue
                    neighbor = all_nodes[tid]
                    if not self._is_authorized(neighbor.acl_roles, caller_roles):
                        continue
                    result_map[tid] = neighbor
                    next_frontier.append(neighbor)
            frontier = next_frontier

        return list(result_map.values())

    def rebuild_derived_graph_and_index(
        self,
        max_edges_per_node: Optional[int] = None,
        embed_fn: Optional[Callable[[str], Sequence[float]]] = None,
        scope: Optional[str] = None,
        scope_id: Optional[str] = None,
    ) -> Dict[str, int]:
        sc = scope or self.default_scope
        sid = scope_id or self.default_scope_id
        node_prefix = f"{sc}:{sid}:{ASSOC_NODE_PREFIX}"
        for k in list(self._nodes.keys()):
            if k.startswith(node_prefix):
                self._nodes.pop(k, None)
                self._vectors.pop(k, None)

        raw_prefix = f"{sc}:{sid}:{RAW_EVENT_PREFIX}"
        ev_ids = [
            raw["id"] for k, raw in self._raw_events.items() if k.startswith(raw_prefix)
        ]
        self._pending_queues[f"{sc}:{sid}"] = list(ev_ids)
        return self.consolidate_bounded(
            max_batch_size=max(1, len(ev_ids)),
            max_edges_per_node=max_edges_per_node,
            embed_fn=embed_fn,
            scope=sc,
            scope_id=sid,
        )

    @staticmethod
    def _score(
        a: AssociativeMemoryNode, b: AssociativeMemoryNode
    ) -> Tuple[float, List[str]]:
        shared = [t for t in a.tags if t in set(b.tags)]
        union_sz = len(set(a.tags) | set(b.tags))
        jaccard = len(shared) / union_sz if union_sz else 0.0
        bonus = 0.15 if (a.category != b.category and shared) else 0.0
        return round(min(1.0, jaccard + bonus), 4), shared

    @staticmethod
    def _upsert_link(
        node: AssociativeMemoryNode, new_link: Dict[str, Any], max_edges: int
    ) -> int:
        links = [
            l for l in node.links if l["target_node_id"] != new_link["target_node_id"]
        ]
        links.append(new_link)
        links.sort(key=lambda item: item["weight"], reverse=True)
        pruned = max(0, len(links) - max_edges)
        node.links = links[:max_edges]
        node.evidence_event_ids = list(
            dict.fromkeys(
                [node.event_id, *(eid for l in node.links for eid in l["evidence_event_ids"])]
            )
        )
        return pruned

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
