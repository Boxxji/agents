"""Evaluation backends and real Hanzo SDK MemoryClient/AIConfig integration.

Provides four explicitly separated evaluation modes to preserve strict scientific integrity:
  1. budget_truncation       (physical, non-circular): real AIConfig.trim_by_chars / tail_clip survival + boundary security
  2. distractor_interference (physical, non-circular): zero-position-bias lexical/voting span extractor
  3. calibrated_tacl2024     (explicitly labeled offline sensitivity stress-test profile from Liu et al. TACL 2024)
  4. live_llm                (live OpenAI-compatible Hanzo Zen/Enso adapter using AIConfig + llm_adapters)
"""

from __future__ import annotations

import asyncio
import html
import json
import math
import os
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from hanzo_agents.client import HanzoAgentsClient
from hanzo_agents.execution_context import ExecutionContext
from hanzo_agents.llm_adapters import apply_provider_patches
from hanzo_agents.memory import MemoryClient, ScopedMemoryClient
from hanzo_agents.types import AIConfig

from .dataset import BenchmarkCase, Document
from .metrics import exact_match, substring_match, token_f1
from .packers import (
    PackedContext,
    _SYSTEM_PREAMBLE,
    _format_isolated_block,
    _format_plain_block,
)

# Published Liu et al. (TACL 2024, Fig. 2 / Sec. 3, 20-document Multi-Document QA)
# empirical reference anchors: (normalized_position_0_to_1, published_accuracy).
# Explicitly labeled as offline stress-test calibration profiles — NOT Hanzo live claims.
TACL2024_PROFILES: Dict[str, Tuple[Tuple[float, float], ...]] = {
    "gpt-3.5-turbo-16k": (
        (0.00, 0.758),
        (0.21, 0.572),
        (0.47, 0.538),
        (0.74, 0.554),
        (1.00, 0.632),
    ),
    "claude-1.3": (
        (0.00, 0.730),
        (0.21, 0.610),
        (0.47, 0.570),
        (0.74, 0.600),
        (1.00, 0.670),
    ),
    "mpt-30b-instruct": (
        (0.00, 0.520),
        (0.21, 0.415),
        (0.47, 0.380),
        (0.74, 0.425),
        (1.00, 0.545),
    ),
}

_FACT_EXTRACT_PATTERNS: Dict[str, re.Pattern[str]] = {
    "failover_port": re.compile(
        r"service\s+([A-Za-z0-9\-]+)\b[^.\n]{0,90}?\bfailover port is\s+([A-Za-z0-9\-]+)",
        re.IGNORECASE,
    ),
    "encryption_cipher": re.compile(
        r"service\s+([A-Za-z0-9\-]+)\b[^.\n]{0,90}?\bencryption cipher is\s+([A-Za-z0-9\-]+)",
        re.IGNORECASE,
    ),
    "quorum_timeout": re.compile(
        r"service\s+([A-Za-z0-9\-]+)\b[^.\n]{0,90}?\bquorum timeout threshold is\s+([A-Za-z0-9\-]+)",
        re.IGNORECASE,
    ),
    "snapshot_bucket": re.compile(
        r"service\s+([A-Za-z0-9\-]+)\b[^.\n]{0,90}?\bsnapshot storage bucket is\s+([A-Za-z0-9\-]+)",
        re.IGNORECASE,
    ),
    "routing_tier": re.compile(
        r"service\s+([A-Za-z0-9\-]+)\b[^.\n]{0,90}?\bEnso routing tier is\s+([A-Za-z0-9\-]+)",
        re.IGNORECASE,
    ),
}


@dataclass(frozen=True)
class CaseEvaluation:
    """Single evaluated (case, policy, budget, mode) outcome."""

    case_id: str
    regime: str
    mode: str
    is_calibrated_stress_test: bool
    policy: str
    truncation_mode: str
    num_docs: int
    gold_position: int
    packed_gold_position: Optional[int]
    budget_tokens: int
    estimated_tokens: int
    pre_truncation_tokens: int
    truncated: bool
    evidence_retained: bool
    truncation_lost_gold: bool
    boundary_integrity_ok: bool
    correctness: float
    exact_match_score: float
    token_f1_score: float
    predicted_answer: str
    gold_answer: str
    packing_latency_ms: float
    dedup_removed_count: int
    dropped_doc_count: int
    api_prompt_tokens: Optional[int] = None
    api_completion_tokens: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class _MockHTTPResponse:
    """Minimal HTTP response object compatible with MemoryClient._async_request."""

    def __init__(self, status_code: int, payload: Any):
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"InMemoryHanzoTransport HTTP {self.status_code}")

    def json(self) -> Any:
        return self._payload


class InMemoryHanzoTransport:
    """In-memory backend for HanzoAgentsClient._async_request supporting real MemoryClient APIs.

    Exercises real scope headers (X-Session-ID, X-Workflow-ID, X-Actor-ID), vector upsert,
    and cosine-similarity search without requiring an external control-plane server.
    """

    def __init__(self) -> None:
        self._kv: Dict[Tuple[str, str, str], Any] = {}
        self._vectors: Dict[Tuple[str, str], Dict[str, Tuple[List[float], Dict[str, Any]]]] = {}

    @staticmethod
    def _scope_key(payload: Dict[str, Any], headers: Dict[str, str]) -> Tuple[str, str]:
        scope = str(payload.get("scope") or "session")
        scope_id = (
            headers.get("X-Session-ID")
            if scope == "session"
            else headers.get("X-Workflow-ID")
            if scope == "workflow"
            else headers.get("X-Actor-ID")
            if scope == "actor"
            else "global"
        ) or "default"
        return scope, scope_id

    @staticmethod
    def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (na * nb)

    async def request(
        self,
        method: str,
        url: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **_: Any,
    ) -> _MockHTTPResponse:
        payload = json or {}
        hdrs = headers or {}
        scope, scope_id = self._scope_key(payload, hdrs)

        if url.endswith("/memory/set"):
            key = str(payload["key"])
            self._kv[(scope, scope_id, key)] = payload.get("data")
            return _MockHTTPResponse(200, {"ok": True})

        if url.endswith("/memory/get"):
            key = str(payload["key"])
            if (scope, scope_id, key) not in self._kv:
                return _MockHTTPResponse(404, {"error": "not found"})
            return _MockHTTPResponse(200, {"data": self._kv[(scope, scope_id, key)]})

        if url.endswith("/memory/vector"):
            key = str(payload["key"])
            emb = [float(x) for x in payload.get("embedding", [])]
            meta = dict(payload.get("metadata") or {})
            bucket = self._vectors.setdefault((scope, scope_id), {})
            bucket[key] = (emb, meta)
            return _MockHTTPResponse(200, {"ok": True})

        if url.endswith("/memory/vector/search"):
            query_emb = [float(x) for x in payload.get("query_embedding", [])]
            top_k = int(payload.get("top_k", 10))
            bucket = self._vectors.get((scope, scope_id), {})
            scored: List[Dict[str, Any]] = []
            for key, (emb, meta) in bucket.items():
                score = self._cosine(query_emb, emb)
                scored.append({"key": key, "score": round(score, 6), "metadata": meta})
            scored.sort(key=lambda r: (-r["score"], r["key"]))
            return _MockHTTPResponse(200, scored[:top_k])

        return _MockHTTPResponse(404, {"error": f"unhandled path {url}"})


class MemoryBackedRetriever:
    """Stores and retrieves benchmark documents through Hanzo's real ScopedMemoryClient."""

    def __init__(self) -> None:
        self.transport = InMemoryHanzoTransport()
        self.client = HanzoAgentsClient(base_url="http://in-memory.hanzo.local")
        # Route MemoryClient's async HTTP calls through the in-memory transport
        self.client._async_request = self.transport.request  # type: ignore[method-assign]

    def scoped_session(self, session_id: str) -> ScopedMemoryClient:
        ctx = ExecutionContext(
            run_id=f"run-{session_id}",
            execution_id=f"exec-{session_id}",
            agent_instance=None,
            reasoner_name="lost_in_the_middle_eval",
            session_id=session_id,
        )
        mem_client = MemoryClient(
            hanzo_agents_client=self.client,
            execution_context=ctx,
            agent_node_id="litm-bench-node",
        )
        return ScopedMemoryClient(mem_client, scope="session", scope_id=session_id)

    async def ingest_and_retrieve(self, case: BenchmarkCase) -> Tuple[Document, ...]:
        """Ingest case documents into an isolated ScopedMemoryClient session and retrieve via vector search."""
        scoped = self.scoped_session(f"case-{case.case_id}")
        for doc in case.documents:
            await scoped.set_vector(
                doc.doc_id,
                doc.embedding,
                metadata=doc.to_metadata(),
            )
        hits = await scoped.similarity_search(
            case.query_embedding,
            top_k=len(case.documents),
        )
        by_id = {d.doc_id: d for d in case.documents}
        retrieved: List[Document] = []
        for hit in hits:
            meta = hit.get("metadata") or {}
            doc_id = str(meta.get("doc_id") or hit.get("key"))
            orig = by_id[doc_id]
            # Preserve exact server-computed cosine similarity score from similarity_search
            retrieved.append(
                Document(
                    doc_id=orig.doc_id,
                    title=orig.title,
                    content=orig.content,
                    relevance_score=float(hit["score"]),
                    timestamp_order=orig.timestamp_order,
                    is_gold=orig.is_gold,
                    is_duplicate=orig.is_duplicate,
                    has_injection=orig.has_injection,
                    entity=orig.entity,
                    attribute=orig.attribute,
                    answer_value=orig.answer_value,
                    embedding=orig.embedding,
                )
            )
        return tuple(retrieved)


class ContextEvaluator(Protocol):
    """Protocol shared by offline physical, calibrated, and live LLM evaluators."""

    mode: str
    is_calibrated_stress_test: bool

    async def evaluate(
        self, case: BenchmarkCase, packed: PackedContext
    ) -> CaseEvaluation:
        ...


def _extract_entity_claims(
    final_prompt: str, entity: str, attribute: str
) -> List[Tuple[str, bool]]:
    """Extract (claimed_value, is_authoritative_clause) tuples from intact prompt segments."""
    pattern = _FACT_EXTRACT_PATTERNS.get(attribute)
    if pattern is None or not entity:
        return []
    claims: List[Tuple[str, bool]] = []
    entity_low = entity.lower()
    # Split on Hanzo's AIConfig.trim_by_chars splice marker so truncated spans never bridge
    for segment in final_prompt.split("\n…TRIMMED…\n"):
        for match in pattern.finditer(segment):
            found_entity, found_val = match.group(1), match.group(2)
            if found_entity.lower() != entity_low:
                continue
            window = segment[match.start() : min(len(segment), match.end() + 140)].lower()
            is_auth = "authoritative source" in window or "supersedes earlier" in window
            claims.append((found_val, is_auth))
    return claims


class BudgetTruncationEvaluator:
    """Physical, non-circular evaluator measuring real token-budget truncation & injection safety.

    Makes ZERO assumptions about transformer attention curves. Correctness depends purely on:
      1. Whether prompt-injection boundaries remained intact (no unescaped hijack).
      2. Whether the gold answer-bearing passage physically survived into final_prompt after
         packing and Hanzo AIConfig.trim_by_chars / tail_clip budget enforcement.
    """

    mode = "budget_truncation"
    is_calibrated_stress_test = False

    async def evaluate(
        self, case: BenchmarkCase, packed: PackedContext
    ) -> CaseEvaluation:
        if not packed.boundary_integrity_ok and "COMPROMISED_TOKEN" in packed.final_prompt:
            prediction = "COMPROMISED_TOKEN"
        elif case.gold_entity and case.gold_attribute:
            claims = _extract_entity_claims(
                packed.final_prompt, case.gold_entity, case.gold_attribute
            )
            auth_claims = [val for val, is_auth in claims if is_auth]
            if auth_claims:
                prediction = auth_claims[0]
            elif claims:
                prediction = claims[0][0]
            else:
                prediction = ""
        else:
            if packed.gold_retained_in_final_prompt:
                prompt_low = packed.final_prompt.lower()
                unesc_low = html.unescape(packed.final_prompt).lower()
                matched = next(
                    (
                        a
                        for a in case.all_accepted_answers
                        if a and (a.lower() in prompt_low or a.lower() in unesc_low)
                    ),
                    case.gold_answer,
                )
                prediction = matched
            else:
                prediction = ""

        golds = case.all_accepted_answers or (case.gold_answer,)
        corr = substring_match(prediction, golds)
        em = exact_match(prediction, golds)
        f1 = token_f1(prediction, golds)
        trunc_lost_gold = (not packed.gold_retained_in_final_prompt) and (
            packed.truncated or (packed.gold_doc_index_1based is None)
        )

        return CaseEvaluation(
            case_id=case.case_id,
            regime=case.regime,
            mode=self.mode,
            is_calibrated_stress_test=self.is_calibrated_stress_test,
            policy=packed.policy,
            truncation_mode=packed.truncation_mode,
            num_docs=case.num_docs,
            gold_position=case.gold_position,
            packed_gold_position=packed.gold_doc_index_1based,
            budget_tokens=packed.budget_tokens,
            estimated_tokens=packed.estimated_tokens,
            pre_truncation_tokens=packed.pre_truncation_tokens,
            truncated=packed.truncated,
            evidence_retained=packed.gold_retained_in_final_prompt,
            truncation_lost_gold=trunc_lost_gold,
            boundary_integrity_ok=packed.boundary_integrity_ok,
            correctness=corr,
            exact_match_score=em,
            token_f1_score=f1,
            predicted_answer=prediction,
            gold_answer=case.gold_answer,
            packing_latency_ms=packed.packing_latency_ms,
            dedup_removed_count=len(packed.dedup_removed_ids),
            dropped_doc_count=len(packed.dropped_doc_ids),
        )


class DistractorInterferenceEvaluator:
    """Physical, non-circular lexical-voting span extractor (zero position-curve bias).

    Extracts all surviving claims for the queried entity/attribute from intact segments in
    final_prompt and aggregates weighted support votes across surviving passages:
      - Standard authoritative passage has weight 1.35 (matches "authoritative" / "supersedes").
      - Non-authoritative / stale distractor passage has weight 1.0 per surviving copy.
    Thus, when duplicate distractors are NOT deduplicated (e.g., 3-5 redundant copies survive),
    their cumulative vote weight overwhelms a single authoritative passage — a real-world
    frequency-interference failure mode that disappears once deduplication collapses duplicates.
    """

    mode = "distractor_interference"
    is_calibrated_stress_test = False

    async def evaluate(
        self, case: BenchmarkCase, packed: PackedContext
    ) -> CaseEvaluation:
        if not packed.boundary_integrity_ok and "COMPROMISED_TOKEN" in packed.final_prompt:
            prediction = "COMPROMISED_TOKEN"
        elif case.gold_entity and case.gold_attribute:
            claims = _extract_entity_claims(
                packed.final_prompt, case.gold_entity, case.gold_attribute
            )
            if not claims:
                prediction = ""
            else:
                vote_weights: Dict[str, float] = {}
                for val, is_auth in claims:
                    vote_weights[val] = vote_weights.get(val, 0.0) + (
                        1.35 if is_auth else 1.0
                    )
                # Deterministic tie-break by value name
                prediction = max(
                    vote_weights.items(), key=lambda kv: (kv[1], kv[0])
                )[0]
        else:
            if packed.gold_retained_in_final_prompt:
                prompt_low = packed.final_prompt.lower()
                unesc_low = html.unescape(packed.final_prompt).lower()
                matched = next(
                    (
                        a
                        for a in case.all_accepted_answers
                        if a and (a.lower() in prompt_low or a.lower() in unesc_low)
                    ),
                    case.gold_answer,
                )
                prediction = matched
            else:
                prediction = ""

        golds = case.all_accepted_answers or (case.gold_answer,)
        corr = substring_match(prediction, golds)
        em = exact_match(prediction, golds)
        f1 = token_f1(prediction, golds)
        trunc_lost_gold = (not packed.gold_retained_in_final_prompt) and (
            packed.truncated or (packed.gold_doc_index_1based is None)
        )

        return CaseEvaluation(
            case_id=case.case_id,
            regime=case.regime,
            mode=self.mode,
            is_calibrated_stress_test=self.is_calibrated_stress_test,
            policy=packed.policy,
            truncation_mode=packed.truncation_mode,
            num_docs=case.num_docs,
            gold_position=case.gold_position,
            packed_gold_position=packed.gold_doc_index_1based,
            budget_tokens=packed.budget_tokens,
            estimated_tokens=packed.estimated_tokens,
            pre_truncation_tokens=packed.pre_truncation_tokens,
            truncated=packed.truncated,
            evidence_retained=packed.gold_retained_in_final_prompt,
            truncation_lost_gold=trunc_lost_gold,
            boundary_integrity_ok=packed.boundary_integrity_ok,
            correctness=corr,
            exact_match_score=em,
            token_f1_score=f1,
            predicted_answer=prediction,
            gold_answer=case.gold_answer,
            packing_latency_ms=packed.packing_latency_ms,
            dedup_removed_count=len(packed.dedup_removed_ids),
            dropped_doc_count=len(packed.dropped_doc_ids),
        )


def _interpolate_curve(
    anchors: Sequence[Tuple[float, float]], x: float
) -> float:
    """Piecewise-linear interpolation over sorted (x, y) anchors."""
    xc = max(0.0, min(1.0, float(x)))
    for i in range(len(anchors) - 1):
        x0, y0 = anchors[i]
        x1, y1 = anchors[i + 1]
        if x0 <= xc <= x1:
            t = (xc - x0) / (x1 - x0) if x1 > x0 else 0.0
            return y0 + t * (y1 - y0)
    return anchors[-1][1]


_PLAIN_DOC_HEADER_RE = re.compile(r"(?m)^\[Document \d+\] \(Title:")


def _count_surviving_documents(packed: PackedContext) -> int:
    """Count documents actually surviving budget enforcement inside packed.final_prompt.

    Uses exact block-span overlap against packed.prompt when final_prompt was derived
    from packed.prompt via middle-trim or tail-clip, falling back to formatted header
    counting for externally constructed PackedContext instances.
    """
    if not packed.final_prompt:
        return 0
    if not packed.truncated and packed.final_prompt == packed.prompt:
        return len(packed.ordered_documents)

    # Reconstruct exact [start, end) character spans of each rendered document block in prompt
    spans: List[Tuple[int, int]] = []
    if packed.ordered_documents and packed.prompt.startswith(_SYSTEM_PREAMBLE):
        cursor = len(_SYSTEM_PREAMBLE) + 1  # +1 for "\n" separator after preamble
        for idx, doc in enumerate(packed.ordered_documents, start=1):
            block = (
                _format_isolated_block(doc, idx, packed.boundary_nonce)
                if packed.boundary_nonce is not None
                else _format_plain_block(doc, idx)
            )
            if packed.prompt.startswith(block, cursor):
                spans.append((cursor, cursor + len(block)))
                cursor += len(block) + 1
            else:
                spans = []
                break

    if spans:
        marker = "\n…TRIMMED…\n"
        if marker in packed.final_prompt:
            head, tail = packed.final_prompt.split(marker, 1)
            if packed.prompt.startswith(head) and packed.prompt.endswith(tail):
                head_end = len(head)
                tail_start = len(packed.prompt) - len(tail)
                return sum(
                    1 for s, e in spans if (s < head_end) or (e > tail_start)
                )
        else:
            # Handle tail_clip (prefix of prompt + optional preserved question footer)
            limit = min(len(packed.prompt), len(packed.final_prompt))
            lcp = 0
            while lcp < limit and packed.prompt[lcp] == packed.final_prompt[lcp]:
                lcp += 1
            suffix = packed.final_prompt[lcp:]
            if not suffix or packed.prompt.endswith(suffix):
                tail_start = len(packed.prompt) - len(suffix) if suffix else len(packed.prompt)
                return sum(
                    1 for s, e in spans if (s < lcp) or (e > tail_start)
                )

    # Fallback: count formatted document block markers present in final_prompt
    if packed.boundary_nonce is not None:
        open_count = packed.final_prompt.count(
            f'<evidence_doc boundary="{packed.boundary_nonce}"'
        )
        close_count = packed.final_prompt.count(
            f'</evidence_doc boundary="{packed.boundary_nonce}">'
        )
        return max(open_count, close_count)

    return len(_PLAIN_DOC_HEADER_RE.findall(packed.final_prompt))


class CalibratedTACL2024Evaluator:
    """Explicitly labeled offline sensitivity stress-test parameterized from Liu et al. TACL 2024.

    IMPORTANT METHODOLOGICAL NOTE:
    This evaluator is NOT a live Hanzo model measurement and MUST NOT be cited as one.
    It quantifies expected accuracy sensitivity under published Liu et al. (TACL 2024)
    position curves layered on top of real physical token-budget retention.
    """

    mode = "calibrated_tacl2024"
    is_calibrated_stress_test = True

    def __init__(self, profile: str = "gpt-3.5-turbo-16k") -> None:
        if profile not in TACL2024_PROFILES:
            raise ValueError(
                f"Unknown calibrated profile {profile!r}; choose from {tuple(TACL2024_PROFILES)}"
            )
        self.profile = profile
        self._anchors = TACL2024_PROFILES[profile]

    async def evaluate(
        self, case: BenchmarkCase, packed: PackedContext
    ) -> CaseEvaluation:
        trunc_lost_gold = (not packed.gold_retained_in_final_prompt) and (
            packed.truncated or (packed.gold_doc_index_1based is None)
        )

        if not packed.boundary_integrity_ok and "COMPROMISED_TOKEN" in packed.final_prompt:
            expected_acc = 0.0
            pred = "COMPROMISED_TOKEN"
        elif not packed.gold_retained_in_final_prompt:
            expected_acc = 0.0
            pred = ""
        else:
            # Locate gold fact's actual character offset inside final_prompt
            low_prompt = packed.final_prompt.lower()
            unesc_prompt = html.unescape(packed.final_prompt).lower()
            if case.gold_entity:
                anchor_needle = case.gold_entity.lower()
            else:
                anchor_needle = next(
                    (
                        a.lower()
                        for a in case.all_accepted_answers
                        if a and (a.lower() in low_prompt or a.lower() in unesc_prompt)
                    ),
                    case.gold_answer.lower(),
                )
            char_idx = low_prompt.find(anchor_needle)
            search_len = len(low_prompt)
            if char_idx < 0:
                char_idx = unesc_prompt.find(anchor_needle)
                search_len = len(unesc_prompt)
            rel_pos = (
                char_idx / max(1, search_len - 1) if char_idx >= 0 else 0.5
            )
            num_retained = max(1, _count_surviving_documents(packed))
            base_curve = _interpolate_curve(self._anchors, rel_pos)

            if num_retained <= 3:
                # Honest small-context regime: negligible position dip when N <= 3
                expected_acc = round(self._anchors[0][1], 6)
            else:
                # Scale U-shaped depth monotonically with retained document crowding (N=4..20)
                crowding = min(1.0, (num_retained - 3) / 17.0)
                primacy_ref = self._anchors[0][1]
                expected_acc = round(
                    primacy_ref - crowding * (primacy_ref - base_curve), 6
                )
            pred = case.gold_answer

        return CaseEvaluation(
            case_id=case.case_id,
            regime=case.regime,
            mode=f"{self.mode}:{self.profile}",
            is_calibrated_stress_test=self.is_calibrated_stress_test,
            policy=packed.policy,
            truncation_mode=packed.truncation_mode,
            num_docs=case.num_docs,
            gold_position=case.gold_position,
            packed_gold_position=packed.gold_doc_index_1based,
            budget_tokens=packed.budget_tokens,
            estimated_tokens=packed.estimated_tokens,
            pre_truncation_tokens=packed.pre_truncation_tokens,
            truncated=packed.truncated,
            evidence_retained=packed.gold_retained_in_final_prompt,
            truncation_lost_gold=trunc_lost_gold,
            boundary_integrity_ok=packed.boundary_integrity_ok,
            correctness=expected_acc,
            exact_match_score=expected_acc,
            token_f1_score=expected_acc,
            predicted_answer=pred,
            gold_answer=case.gold_answer,
            packing_latency_ms=packed.packing_latency_ms,
            dedup_removed_count=len(packed.dedup_removed_ids),
            dropped_doc_count=len(packed.dropped_doc_ids),
        )


class LiveOpenAICompatibleEvaluator:
    """Live LLM evaluator using Hanzo's AIConfig and llm_adapters against Zen/Enso/OpenAI APIs.

    Automatically resolves credentials from HANZO_API_KEY (+ HANZO_API_BASE) or
    OPENAI_API_KEY (+ OPENAI_BASE_URL), builds request payloads via AIConfig.get_llm_params
    and hanzo_agents.llm_adapters.apply_provider_patches, and records actual server token usage.
    """

    mode = "live_llm"
    is_calibrated_stress_test = False

    def __init__(
        self,
        *,
        model: str = "openai/gpt-4o-mini",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: float = 30.0,
        max_completion_tokens: int = 64,
        seed: int = 42,
    ) -> None:
        self.model = model
        resolved_key = (
            api_key
            or os.environ.get("HANZO_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not resolved_key:
            raise RuntimeError(
                "Live LLM mode requires HANZO_API_KEY or OPENAI_API_KEY to be set."
            )
        if api_base:
            resolved_base = api_base
        elif os.environ.get("HANZO_API_KEY") and not api_key:
            resolved_base = os.environ.get("HANZO_API_BASE", "https://api.hanzo.ai/v1")
        else:
            resolved_base = os.environ.get(
                "OPENAI_BASE_URL", "https://api.openai.com/v1"
            )

        self.api_key = resolved_key
        self.api_base = resolved_base.rstrip("/")
        self.timeout = timeout
        self.seed = seed
        self.ai_config = AIConfig(
            model=model,
            temperature=0.0,
            max_tokens=max_completion_tokens,
            api_key=resolved_key,
            api_base=self.api_base,
            timeout=timeout,
        )

    def build_request_payload(self, packed: PackedContext) -> Dict[str, Any]:
        """Build provider-patched OpenAI-compatible chat completions payload via Hanzo SDK."""
        raw_params = self.ai_config.get_llm_params(
            messages=[{"role": "user", "content": packed.final_prompt}],
            seed=self.seed,
        )
        # Apply Hanzo's central provider patch adapter (e.g., max_tokens -> max_completion_tokens)
        patched = apply_provider_patches(raw_params, self.model)
        # Strip client-side-only keys not accepted in JSON request body
        body = {
            k: v
            for k, v in patched.items()
            if k not in {"api_key", "api_base", "api_version", "organization", "timeout", "num_retries"}
        }
        # Strip provider prefix ("openai/gpt-4o-mini" -> "gpt-4o-mini") for OpenAI-compatible HTTP wire
        if isinstance(body.get("model"), str) and "/" in body["model"]:
            body["model"] = body["model"].split("/", 1)[1]
        body["messages"] = [{"role": "user", "content": packed.final_prompt}]
        return body

    def _call_sync(self, body: Dict[str, Any]) -> Dict[str, Any]:
        req = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    async def evaluate(
        self, case: BenchmarkCase, packed: PackedContext
    ) -> CaseEvaluation:
        body = self.build_request_payload(packed)
        response = await asyncio.to_thread(self._call_sync, body)
        choices = response.get("choices") or []
        prediction = (
            str(choices[0].get("message", {}).get("content") or "").strip()
            if choices
            else ""
        )
        usage = response.get("usage") or {}
        prompt_toks = usage.get("prompt_tokens")
        comp_toks = usage.get("completion_tokens")

        golds = case.all_accepted_answers or (case.gold_answer,)
        corr = substring_match(prediction, golds)
        em = exact_match(prediction, golds)
        f1 = token_f1(prediction, golds)
        trunc_lost_gold = (not packed.gold_retained_in_final_prompt) and (
            packed.truncated or (packed.gold_doc_index_1based is None)
        )

        return CaseEvaluation(
            case_id=case.case_id,
            regime=case.regime,
            mode=f"{self.mode}:{self.model}",
            is_calibrated_stress_test=False,
            policy=packed.policy,
            truncation_mode=packed.truncation_mode,
            num_docs=case.num_docs,
            gold_position=case.gold_position,
            packed_gold_position=packed.gold_doc_index_1based,
            budget_tokens=packed.budget_tokens,
            estimated_tokens=int(prompt_toks)
            if prompt_toks is not None
            else packed.estimated_tokens,
            pre_truncation_tokens=packed.pre_truncation_tokens,
            truncated=packed.truncated,
            evidence_retained=packed.gold_retained_in_final_prompt,
            truncation_lost_gold=trunc_lost_gold,
            boundary_integrity_ok=packed.boundary_integrity_ok,
            correctness=corr,
            exact_match_score=em,
            token_f1_score=f1,
            predicted_answer=prediction,
            gold_answer=case.gold_answer,
            packing_latency_ms=packed.packing_latency_ms,
            dedup_removed_count=len(packed.dedup_removed_ids),
            dropped_doc_count=len(packed.dropped_doc_ids),
            api_prompt_tokens=int(prompt_toks) if prompt_toks is not None else None,
            api_completion_tokens=int(comp_toks) if comp_toks is not None else None,
        )
