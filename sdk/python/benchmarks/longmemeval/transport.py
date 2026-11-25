"""
Deterministic in-memory HTTP transport mirroring the Hanzo Agents control-plane
memory REST surface (`control-plane/internal/handlers/memory.go` and
`vector_memory.go`).

This transport plugs into the unmodified `hanzo_agents.memory.MemoryClient`,
`ScopedMemoryClient`, `GlobalMemoryClient`, and `MemoryInterface` via
`HanzoAgentsClient._async_request`, exercising the real SDK request, header,
serialization, scope-resolution, and response-decoding code paths without
requiring network sockets or external storage services.
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import unquote, urlparse


class TransportHTTPError(RuntimeError):
    """Raised by `TransportResponse.raise_for_status()` on 4xx/5xx responses."""

    def __init__(self, status_code: int, message: str, payload: Any = None) -> None:
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code
        self.payload = payload


@dataclass
class TransportResponse:
    """Minimal HTTP response object compatible with `MemoryClient`."""

    status_code: int
    _payload: Any = None

    def json(self) -> Any:
        return copy.deepcopy(self._payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            msg = ""
            if isinstance(self._payload, dict):
                msg = str(
                    self._payload.get("message")
                    or self._payload.get("error")
                    or self._payload
                )
            raise TransportHTTPError(self.status_code, msg, self._payload)


@dataclass
class StoredKVRecord:
    """Canonical KV record matching `types.Memory` in the Go control plane."""

    scope: str
    scope_id: str
    key: str
    data_json: str
    created_at: str
    updated_at: str


@dataclass
class StoredVectorRecord:
    """Canonical vector record matching `types.VectorRecord` in the Go control plane."""

    scope: str
    scope_id: str
    key: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    updated_at: str = ""


@dataclass
class MemoryEventRecord:
    """Change event emitted by set/delete operations, mirroring `memory.go`."""

    action: str
    scope: str
    scope_id: str
    key: str
    data_json: Optional[str]
    previous_data_json: Optional[str]
    timestamp: str
    actor_id: str = ""
    workflow_id: str = ""
    agent_id: str = ""


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity matching `base_memory._cosine` and Go vector storage."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _metadata_matches(
    metadata: Optional[Dict[str, Any]], filters: Optional[Dict[str, Any]]
) -> bool:
    if not filters:
        return True
    meta = metadata or {}
    return all(meta.get(k) == v for k, v in filters.items())


class ControlPlaneMemoryTransport:
    """
    In-memory simulation of the Hanzo Agents control-plane `/v1/memory/*` API.

    Mirrors the exact Go handler semantics in:
    - `control-plane/internal/handlers/memory.go` (`SetMemoryHandler`,
      `GetMemoryHandler`, `DeleteMemoryHandler`, `ListMemoryHandler`,
      `resolveScope`, `getScopeID`)
    - `control-plane/internal/handlers/vector_memory.go` (`SetVectorHandler`,
      `DeleteVectorHandler`, `SimilaritySearchHandler`)
    """

    SCOPE_HIERARCHY: Tuple[str, ...] = ("workflow", "session", "actor", "global")

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._kv: Dict[Tuple[str, str, str], StoredKVRecord] = {}
        self._vectors: Dict[Tuple[str, str, str], StoredVectorRecord] = {}
        self._events: List[MemoryEventRecord] = []
        self._api_calls: Dict[str, int] = {
            "set": 0,
            "get": 0,
            "delete": 0,
            "list": 0,
            "set_vector": 0,
            "delete_vector": 0,
            "similarity_search": 0,
        }

    @property
    def api_calls(self) -> Dict[str, int]:
        return dict(self._api_calls)

    @property
    def events(self) -> List[MemoryEventRecord]:
        return list(self._events)

    def reset_stats(self) -> None:
        for k in self._api_calls:
            self._api_calls[k] = 0

    @staticmethod
    def _header(headers: Optional[Dict[str, str]], name: str) -> str:
        if not headers:
            return ""
        if name in headers:
            return headers[name] or ""
        lower = name.lower()
        for k, v in headers.items():
            if k.lower() == lower:
                return v or ""
        return ""

    @classmethod
    def _get_scope_id(cls, headers: Optional[Dict[str, str]], scope: str) -> str:
        if scope == "workflow":
            return cls._header(headers, "X-Workflow-ID")
        if scope == "session":
            return cls._header(headers, "X-Session-ID")
        if scope == "actor":
            return cls._header(headers, "X-Actor-ID")
        if scope == "global":
            return "global"
        return ""

    @classmethod
    def _resolve_scope(
        cls, headers: Optional[Dict[str, str]], explicit_scope: Optional[str]
    ) -> Tuple[str, str]:
        if explicit_scope:
            return explicit_scope, cls._get_scope_id(headers, explicit_scope)
        for scope in ("workflow", "session", "actor"):
            sid = cls._get_scope_id(headers, scope)
            if sid:
                return scope, sid
        return "global", "global"

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    async def handle_request(
        self,
        method: str,
        url: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> TransportResponse:
        del timeout  # Not used in deterministic in-memory transport.
        parsed = urlparse(url)
        path = parsed.path
        method_upper = method.upper()

        async with self._lock:
            if method_upper == "POST" and path.endswith("/memory/set"):
                return self._handle_set(json or {}, headers)
            if method_upper == "POST" and path.endswith("/memory/get"):
                return self._handle_get(json or {}, headers)
            if method_upper == "POST" and path.endswith("/memory/delete"):
                return self._handle_delete(json or {}, headers)
            if method_upper == "GET" and path.endswith("/memory/list"):
                return self._handle_list(params or {}, headers)
            if method_upper == "POST" and path.endswith("/memory/vector"):
                return self._handle_set_vector(json or {}, headers)
            if method_upper == "POST" and path.endswith("/memory/vector/search"):
                return self._handle_vector_search(json or {}, headers)
            if method_upper == "DELETE" and "/memory/vector/" in path:
                key_encoded = path.split("/memory/vector/", 1)[1]
                key = unquote(key_encoded)
                return self._handle_delete_vector(key, params or {}, headers)

        return TransportResponse(
            404,
            {"error": "not_found", "message": f"Unhandled route {method_upper} {path}"},
        )

    @staticmethod
    def _serialize_kv_record(rec: StoredKVRecord) -> Dict[str, Any]:
        """Serialize a stored KV record matching Go's `json.RawMessage` wire format."""
        return {
            "key": rec.key,
            "data": json.loads(rec.data_json),
            "scope": rec.scope,
            "scope_id": rec.scope_id,
            "created_at": rec.created_at,
            "updated_at": rec.updated_at,
        }

    def _handle_set(
        self, payload: Dict[str, Any], headers: Optional[Dict[str, str]]
    ) -> TransportResponse:
        self._api_calls["set"] += 1
        key = payload.get("key")
        if not key or "data" not in payload:
            return TransportResponse(
                400, {"error": "invalid_request", "message": "key and data are required"}
            )
        scope, scope_id = self._resolve_scope(headers, payload.get("scope"))
        now = self._now_iso()
        data_json = json.dumps(payload["data"])
        composite = (scope, scope_id, str(key))
        prev = self._kv.get(composite)
        rec = StoredKVRecord(
            scope=scope,
            scope_id=scope_id,
            key=str(key),
            data_json=data_json,
            created_at=prev.created_at if prev else now,
            updated_at=now,
        )
        self._kv[composite] = rec
        self._events.append(
            MemoryEventRecord(
                action="set",
                scope=scope,
                scope_id=scope_id,
                key=str(key),
                data_json=data_json,
                previous_data_json=prev.data_json if prev else None,
                timestamp=now,
                actor_id=self._header(headers, "X-Actor-ID"),
                workflow_id=self._header(headers, "X-Workflow-ID"),
                agent_id=self._header(headers, "X-Agent-Node-ID"),
            )
        )
        return TransportResponse(200, self._serialize_kv_record(rec))

    def _handle_get(
        self, payload: Dict[str, Any], headers: Optional[Dict[str, str]]
    ) -> TransportResponse:
        self._api_calls["get"] += 1
        key = payload.get("key")
        if not key:
            return TransportResponse(
                400, {"error": "invalid_request", "message": "key is required"}
            )
        explicit_scope = payload.get("scope")
        if explicit_scope:
            scope, scope_id = self._resolve_scope(headers, explicit_scope)
            rec = self._kv.get((scope, scope_id, str(key)))
            if rec is None:
                return TransportResponse(
                    404, {"error": "not_found", "message": "Memory key not found"}
                )
            return TransportResponse(200, self._serialize_kv_record(rec))

        # Exact hierarchical fallback order from control-plane/internal/handlers/memory.go:189
        for scope in self.SCOPE_HIERARCHY:
            scope_id = self._get_scope_id(headers, scope)
            if scope_id or scope == "global":
                if scope == "global":
                    scope_id = "global"
                rec = self._kv.get((scope, scope_id, str(key)))
                if rec is not None:
                    return TransportResponse(200, self._serialize_kv_record(rec))
        return TransportResponse(
            404, {"error": "not_found", "message": "Memory key not found in any scope"}
        )

    def _handle_delete(
        self, payload: Dict[str, Any], headers: Optional[Dict[str, str]]
    ) -> TransportResponse:
        self._api_calls["delete"] += 1
        key = payload.get("key")
        if not key:
            return TransportResponse(
                400, {"error": "invalid_request", "message": "key is required"}
            )
        scope, scope_id = self._resolve_scope(headers, payload.get("scope"))
        composite = (scope, scope_id, str(key))
        prev = self._kv.pop(composite, None)
        if prev is None:
            return TransportResponse(
                404, {"error": "not_found", "message": "Memory key not found"}
            )
        self._events.append(
            MemoryEventRecord(
                action="delete",
                scope=scope,
                scope_id=scope_id,
                key=str(key),
                data_json=None,
                previous_data_json=prev.data_json,
                timestamp=self._now_iso(),
                actor_id=self._header(headers, "X-Actor-ID"),
                workflow_id=self._header(headers, "X-Workflow-ID"),
                agent_id=self._header(headers, "X-Agent-Node-ID"),
            )
        )
        return TransportResponse(204, None)

    def _handle_list(
        self, params: Dict[str, Any], headers: Optional[Dict[str, str]]
    ) -> TransportResponse:
        self._api_calls["list"] += 1
        scope_param = params.get("scope")
        if not scope_param:
            return TransportResponse(
                400,
                {
                    "error": "missing_scope",
                    "message": "Scope parameter is required for listing memory",
                },
            )
        scope, scope_id = self._resolve_scope(headers, str(scope_param))
        matches = [
            self._serialize_kv_record(rec)
            for (sc, sid, _), rec in sorted(self._kv.items())
            if sc == scope and sid == scope_id
        ]
        return TransportResponse(200, matches)

    def _handle_set_vector(
        self, payload: Dict[str, Any], headers: Optional[Dict[str, str]]
    ) -> TransportResponse:
        self._api_calls["set_vector"] += 1
        key = payload.get("key")
        embedding = payload.get("embedding")
        if not key or not embedding:
            return TransportResponse(
                400,
                {
                    "error": "invalid_request",
                    "message": "key and non-empty embedding are required",
                },
            )
        scope, scope_id = self._resolve_scope(headers, payload.get("scope"))
        vec = [float(x) for x in embedding]
        meta = copy.deepcopy(payload.get("metadata") or {})
        rec = StoredVectorRecord(
            scope=scope,
            scope_id=scope_id,
            key=str(key),
            embedding=vec,
            metadata=meta,
            updated_at=self._now_iso(),
        )
        self._vectors[(scope, scope_id, str(key))] = rec
        return TransportResponse(
            200,
            {
                "key": rec.key,
                "scope": rec.scope,
                "scope_id": rec.scope_id,
                "metadata": rec.metadata,
            },
        )

    def _handle_delete_vector(
        self,
        key: str,
        params: Dict[str, Any],
        headers: Optional[Dict[str, str]],
    ) -> TransportResponse:
        self._api_calls["delete_vector"] += 1
        scope_param = params.get("scope")
        scope, scope_id = self._resolve_scope(
            headers, str(scope_param) if scope_param else None
        )
        self._vectors.pop((scope, scope_id, key), None)
        return TransportResponse(204, None)

    def _handle_vector_search(
        self, payload: Dict[str, Any], headers: Optional[Dict[str, str]]
    ) -> TransportResponse:
        self._api_calls["similarity_search"] += 1
        query_emb = payload.get("query_embedding")
        if not query_emb:
            return TransportResponse(
                400,
                {
                    "error": "invalid_request",
                    "message": "query_embedding cannot be empty",
                },
            )
        top_k = int(payload.get("top_k") or 10)
        if top_k <= 0:
            top_k = 10
        filters = payload.get("filters") or {}
        scope, scope_id = self._resolve_scope(headers, payload.get("scope"))
        q_vec = [float(x) for x in query_emb]

        hits: List[Dict[str, Any]] = []
        for (sc, sid, _), rec in self._vectors.items():
            if sc != scope or sid != scope_id:
                continue
            if len(rec.embedding) != len(q_vec):
                continue
            if not _metadata_matches(rec.metadata, filters):
                continue
            hits.append(
                {
                    "key": rec.key,
                    "scope": rec.scope,
                    "scope_id": rec.scope_id,
                    "score": cosine_similarity(q_vec, rec.embedding),
                    "metadata": copy.deepcopy(rec.metadata),
                }
            )
        # Deterministic tie-breaking by key for identical cosine scores.
        hits.sort(key=lambda item: (-item["score"], item["key"]))
        return TransportResponse(200, hits[:top_k])


class InMemoryHanzoAgentsClient:
    """
    Lightweight client adapter implementing the `api_base` and `_async_request`
    interface consumed by `hanzo_agents.memory.MemoryClient`.
    """

    def __init__(
        self,
        transport: Optional[ControlPlaneMemoryTransport] = None,
        api_base: str = "http://inmemory.hanzo.local/v1",
    ) -> None:
        self.transport = transport or ControlPlaneMemoryTransport()
        self.api_base = api_base.rstrip("/")

    async def _async_request(
        self, method: str, url: str, **kwargs: Any
    ) -> TransportResponse:
        return await self.transport.handle_request(method, url, **kwargs)
