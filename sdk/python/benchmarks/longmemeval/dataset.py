"""
Deterministic LongMemEval-structured dataset fixture for evaluating Hanzo Agents
memory policies across all five LongMemEval dimensions (Wu et al., ICLR 2025):

1. knowledge_updates       -- multi-turn overwrites & cross-scope staleness
2. contradiction_handling  -- conflicting entries across scopes/timestamps/sources
3. temporal_ordering       -- chronological event sequence & recency reconstruction
4. abstention              -- unanswerable, deleted, and cross-actor-only queries
5. cross_session_retrieval -- session isolation vs actor persistence vs global fallback

All scenarios are original, clean-room fixtures (no external dataset downloads or
third-party text copied) generated deterministically from a configurable seed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

ALL_CATEGORIES: tuple[str, ...] = (
    "knowledge_updates",
    "contradiction_handling",
    "temporal_ordering",
    "abstention",
    "cross_session_retrieval",
)


@dataclass(frozen=True)
class MemoryTurn:
    """Single conversational or tool-produced memory event to ingest."""

    turn_id: str
    actor_id: str
    session_id: str
    workflow_id: str
    scope: str  # "workflow" | "session" | "actor" | "global"
    timestamp: str  # ISO-8601 UTC string (lexicographically sortable)
    entity: str
    attribute: str
    value: str
    statement: str
    action: str = "set"  # "set" | "update" | "delete"
    source: str = "user"
    confidence: float = 0.95
    is_event: bool = False  # True for multi-step timeline events


@dataclass(frozen=True)
class MemoryQuery:
    """Evaluation query executed after all scenario turns are ingested."""

    actor_id: str
    session_id: str
    workflow_id: str
    entity: str
    attribute: str
    question: str
    query_time: str
    query_type: str = "fact"  # "fact" | "sequence" | "latest"


@dataclass(frozen=True)
class BenchmarkCase:
    """One self-contained LongMemEval evaluation scenario."""

    case_id: str
    category: str
    description: str
    turns: List[MemoryTurn]
    query: MemoryQuery
    expected_answer: Optional[str] = None
    expected_order: List[str] = field(default_factory=list)
    should_abstain: bool = False
    expect_contradiction: bool = False
    stale_values: List[str] = field(default_factory=list)
    cross_actor_secrets: List[str] = field(default_factory=list)


def _distractor_turns(
    prefix: str,
    actor_id: str,
    session_id: str,
    workflow_id: str,
    rng: random.Random,
    count: int = 3,
) -> List[MemoryTurn]:
    """Generate deterministic distractor memories in the same actor scope."""
    pool = [
        ("office", "wifi_ssid", "hanzo-corp-5g", "The office Wi-Fi network is hanzo-corp-5g."),
        ("laptop", "os_version", "ubuntu-24.04", "Developer workstation runs ubuntu-24.04."),
        ("lunch", "favorite_spot", "ramen-nagi", "Team lunch favorite spot is ramen-nagi."),
        ("calendar", "standup_time", "09:30-utc", "Daily engineering standup is at 09:30-utc."),
        ("ci", "runner_arch", "arm64", "Continuous integration builds execute on arm64 runners."),
        ("docs", "style_guide", "google-md", "Documentation follows the google-md style guide."),
        ("billing", "currency", "usd", "Cloud invoice billing currency is usd."),
        ("monitoring", "dashboard", "grafana-prod", "Primary metrics dashboard is grafana-prod."),
    ]
    chosen = rng.sample(pool, k=min(count, len(pool)))
    turns: List[MemoryTurn] = []
    for idx, (ent, attr, val, stmt) in enumerate(chosen, start=1):
        turns.append(
            MemoryTurn(
                turn_id=f"{prefix}-dist-{idx}",
                actor_id=actor_id,
                session_id=session_id,
                workflow_id=workflow_id,
                scope="actor",
                timestamp=f"2026-01-01T08:0{idx}:00Z",
                entity=ent,
                attribute=attr,
                value=val,
                statement=stmt,
                action="set",
                source="background",
                confidence=0.90,
            )
        )
    return turns


def _build_knowledge_updates(rng: random.Random) -> List[BenchmarkCase]:
    cases: List[BenchmarkCase] = []

    # 6 multi-turn sequential overwrites within the active session/actor
    same_scope_specs = [
        ("ku-01", "user", "city", ["Austin", "Seattle", "Zurich"], "Where does the user currently live?"),
        ("ku-02", "project", "primary_model", ["zen-2", "zen-3-pro", "zen-4-ultra"], "What is the project primary model?"),
        ("ku-03", "service", "oncall_engineer", ["maya", "devon", "priya"], "Who is the current service on-call engineer?"),
        ("ku-04", "cluster", "database_engine", ["sqlite", "postgres-16", "postgres-17"], "Which database engine does the cluster run?"),
        ("ku-05", "release", "api_version", ["v1-beta", "v1", "v2"], "What is the current release API version?"),
        ("ku-06", "team", "sprint_budget", ["12000", "18500", "24000"], "What is the team sprint budget?"),
    ]
    for cid, ent, attr, values, question in same_scope_specs:
        actor = f"actor-{cid}"
        sess = f"sess-{cid}-1"
        wf = f"wf-{cid}-1"
        turns = _distractor_turns(cid, actor, sess, wf, rng, count=3)
        for i, val in enumerate(values, start=1):
            turns.append(
                MemoryTurn(
                    turn_id=f"{cid}-t{i}",
                    actor_id=actor,
                    session_id=sess,
                    workflow_id=wf,
                    scope="actor",
                    timestamp=f"2026-02-0{i}T10:00:00Z",
                    entity=ent,
                    attribute=attr,
                    value=val,
                    statement=f"Update {i}: {ent} {attr} is now {val}.",
                    action="set" if i == 1 else "update",
                )
            )
        cases.append(
            BenchmarkCase(
                case_id=cid,
                category="knowledge_updates",
                description=f"Multi-turn overwrite of {ent}.{attr} across {len(values)} turns.",
                turns=turns,
                query=MemoryQuery(
                    actor_id=actor,
                    session_id=sess,
                    workflow_id=wf,
                    entity=ent,
                    attribute=attr,
                    question=question,
                    query_time="2026-02-10T10:00:00Z",
                ),
                expected_answer=values[-1],
                stale_values=values[:-1],
            )
        )

    # 4 cross-scope update cases exposing hierarchical fallback shadowing:
    # An older value was written to session/workflow scope, then updated later in actor scope.
    shadow_specs = [
        ("ku-07", "deploy", "region", "us-east-1", "eu-central-1", "session", "What is the active deploy region?"),
        ("ku-08", "user", "preferred_ide", "vim", "cursor", "session", "What is the user's preferred IDE?"),
        ("ku-09", "pipeline", "freeze_date", "2026-03-01", "2026-03-15", "workflow", "What is the pipeline freeze date?"),
        ("ku-10", "account", "tier", "starter", "enterprise", "workflow", "What is the current account tier?"),
    ]
    for cid, ent, attr, old_val, new_val, stale_scope, question in shadow_specs:
        actor = f"actor-{cid}"
        sess = f"sess-{cid}-active"
        wf = f"wf-{cid}-active"
        turns = _distractor_turns(cid, actor, sess, wf, rng, count=2)
        turns.extend(
            [
                MemoryTurn(
                    turn_id=f"{cid}-t1",
                    actor_id=actor,
                    session_id=sess,
                    workflow_id=wf,
                    scope=stale_scope,
                    timestamp="2026-02-01T09:00:00Z",
                    entity=ent,
                    attribute=attr,
                    value=old_val,
                    statement=f"Initial {stale_scope} setting: {ent} {attr} is {old_val}.",
                    action="set",
                ),
                MemoryTurn(
                    turn_id=f"{cid}-t2",
                    actor_id=actor,
                    session_id=sess,
                    workflow_id=wf,
                    scope="actor",
                    timestamp="2026-02-05T15:00:00Z",
                    entity=ent,
                    attribute=attr,
                    value=new_val,
                    statement=f"Authoritative actor update: {ent} {attr} changed to {new_val}.",
                    action="update",
                ),
            ]
        )
        cases.append(
            BenchmarkCase(
                case_id=cid,
                category="knowledge_updates",
                description=(
                    f"Cross-scope update where older {stale_scope} entry ({old_val}) "
                    f"shadows newer actor update ({new_val}) under fixed scope precedence."
                ),
                turns=turns,
                query=MemoryQuery(
                    actor_id=actor,
                    session_id=sess,
                    workflow_id=wf,
                    entity=ent,
                    attribute=attr,
                    question=question,
                    query_time="2026-02-06T12:00:00Z",
                ),
                expected_answer=new_val,
                stale_values=[old_val],
            )
        )
    return cases


def _build_contradiction_handling(rng: random.Random) -> List[BenchmarkCase]:
    cases: List[BenchmarkCase] = []

    # 5 cross-scope contradictions (conflicting entries across distinct scopes)
    cross_scope_conflicts = [
        ("ch-01", "env", "target_stage", "session", "staging", "actor", "production", "What is the authoritative target stage?"),
        ("ch-02", "sdk", "primary_lang", "workflow", "python", "actor", "go", "What is the primary implementation language?"),
        ("ch-03", "auth", "mfa_policy", "global", "optional", "actor", "mandatory", "What is the MFA policy for this actor?"),
        ("ch-04", "cache", "backend", "session", "memcached", "actor", "redis-cluster", "Which cache backend should be used?"),
        ("ch-05", "storage", "retention_days", "global", "30", "actor", "365", "What is the storage retention period in days?"),
    ]
    for cid, ent, attr, sc_a, val_a, sc_b, val_b, question in cross_scope_conflicts:
        actor = f"actor-{cid}"
        sess = f"sess-{cid}-1"
        wf = f"wf-{cid}-1"
        turns = _distractor_turns(cid, actor, sess, wf, rng, count=2)
        turns.extend(
            [
                MemoryTurn(
                    turn_id=f"{cid}-t1",
                    actor_id=actor,
                    session_id=sess,
                    workflow_id=wf,
                    scope=sc_a,
                    timestamp="2026-03-01T10:00:00Z",
                    entity=ent,
                    attribute=attr,
                    value=val_a,
                    statement=f"{sc_a} assertion: {ent} {attr} is {val_a}.",
                    source=f"{sc_a}-note",
                    confidence=0.65,
                ),
                MemoryTurn(
                    turn_id=f"{cid}-t2",
                    actor_id=actor,
                    session_id=sess,
                    workflow_id=wf,
                    scope=sc_b,
                    timestamp="2026-03-02T14:00:00Z",
                    entity=ent,
                    attribute=attr,
                    value=val_b,
                    statement=f"Verified {sc_b} record: {ent} {attr} is {val_b}.",
                    source="verified-config",
                    confidence=0.98,
                ),
            ]
        )
        cases.append(
            BenchmarkCase(
                case_id=cid,
                category="contradiction_handling",
                description=f"Cross-scope contradiction between {sc_a}={val_a} and newer {sc_b}={val_b}.",
                turns=turns,
                query=MemoryQuery(
                    actor_id=actor,
                    session_id=sess,
                    workflow_id=wf,
                    entity=ent,
                    attribute=attr,
                    question=question,
                    query_time="2026-03-03T09:00:00Z",
                ),
                expected_answer=val_b,
                expect_contradiction=True,
                stale_values=[val_a],
            )
        )

    # 5 multi-session / multi-source contradictions (low-confidence hypothesis vs verified observation)
    source_conflicts = [
        ("ch-06", "incident", "root_cause", "dns-timeout", "tls-cert-expiry", "What was the confirmed incident root cause?"),
        ("ch-07", "payment", "gateway", "legacy-bank", "stripe-v3", "Which payment gateway is confirmed active?"),
        ("ch-08", "router", "protocol", "grpc", "native-zap", "Which RPC protocol does the control plane use?"),
        ("ch-09", "worker", "max_concurrency", "4", "32", "What is the verified worker max concurrency?"),
        ("ch-10", "compliance", "audit_status", "pending", "passed", "What is the verified compliance audit status?"),
    ]
    for cid, ent, attr, wrong_val, verified_val, question in source_conflicts:
        actor = f"actor-{cid}"
        sess1 = f"sess-{cid}-s1"
        sess2 = f"sess-{cid}-s2"
        wf = f"wf-{cid}-1"
        turns = _distractor_turns(cid, actor, sess1, wf, rng, count=2)
        turns.extend(
            [
                MemoryTurn(
                    turn_id=f"{cid}-t1",
                    actor_id=actor,
                    session_id=sess1,
                    workflow_id=wf,
                    scope="actor",
                    timestamp="2026-03-04T08:00:00Z",
                    entity=ent,
                    attribute=attr,
                    value=verified_val,
                    statement=f"Direct tool observation: {ent} {attr} confirmed as {verified_val}.",
                    source="tool-observation",
                    confidence=0.98,
                ),
                MemoryTurn(
                    turn_id=f"{cid}-t2",
                    actor_id=actor,
                    session_id=sess2,
                    workflow_id=wf,
                    scope="session",
                    timestamp="2026-03-04T09:00:00Z",
                    entity=ent,
                    attribute=attr,
                    value=wrong_val,
                    statement=f"Unverified guess in chat: maybe {ent} {attr} is {wrong_val}.",
                    source="unverified-hypothesis",
                    confidence=0.35,
                ),
            ]
        )
        cases.append(
            BenchmarkCase(
                case_id=cid,
                category="contradiction_handling",
                description=(
                    f"Conflicting verified observation ({verified_val}, conf=0.98) vs "
                    f"later low-confidence hypothesis ({wrong_val}, conf=0.35)."
                ),
                turns=turns,
                query=MemoryQuery(
                    actor_id=actor,
                    session_id=sess2,
                    workflow_id=wf,
                    entity=ent,
                    attribute=attr,
                    question=question,
                    query_time="2026-03-04T12:00:00Z",
                ),
                expected_answer=verified_val,
                expect_contradiction=True,
                stale_values=[wrong_val],
            )
        )
    return cases


def _build_temporal_ordering(rng: random.Random) -> List[BenchmarkCase]:
    specs = [
        (
            "to-01",
            "incident_42",
            "stage",
            [("2026-04-01T01:00:00Z", "detected"), ("2026-04-01T01:15:00Z", "triaged"), ("2026-04-01T01:45:00Z", "mitigated"), ("2026-04-01T03:00:00Z", "postmortem")],
            False,
            "List the chronological stages of incident_42.",
        ),
        (
            "to-02",
            "rollout_v2",
            "region",
            [("2026-04-02T08:00:00Z", "sfo"), ("2026-04-02T10:00:00Z", "iad"), ("2026-04-02T12:00:00Z", "fra"), ("2026-04-02T14:00:00Z", "nrt")],
            False,
            "In what chronological order did rollout_v2 deploy across regions?",
        ),
        (
            "to-03",
            "candidate_a",
            "interview_step",
            [("2026-04-03T09:00:00Z", "screen"), ("2026-04-05T11:00:00Z", "coding"), ("2026-04-08T15:00:00Z", "systems"), ("2026-04-10T16:00:00Z", "offer")],
            True,  # Ingested out of timestamp order (backfill)
            "Reconstruct the chronological interview steps for candidate_a.",
        ),
        (
            "to-04",
            "schema_v4",
            "migration",
            [("2026-04-04T06:00:00Z", "backup"), ("2026-04-04T06:20:00Z", "alter-table"), ("2026-04-04T06:40:00Z", "backfill"), ("2026-04-04T07:00:00Z", "verify")],
            True,
            "What was the chronological order of schema_v4 migration steps?",
        ),
        (
            "to-05",
            "model_run",
            "checkpoint",
            [("2026-04-05T00:00:00Z", "pretrain"), ("2026-04-06T00:00:00Z", "sft"), ("2026-04-07T00:00:00Z", "dpo")],
            False,
            "Order the training checkpoints for model_run chronologically.",
        ),
        (
            "to-06",
            "trip_eu",
            "stop",
            [("2026-05-01T08:00:00Z", "london"), ("2026-05-03T09:00:00Z", "paris"), ("2026-05-06T10:00:00Z", "berlin")],
            True,
            "What is the chronological itinerary of stops for trip_eu?",
        ),
        (
            "to-07",
            "bug_901",
            "status",
            [("2026-05-10T09:00:00Z", "opened"), ("2026-05-10T11:00:00Z", "reproduced"), ("2026-05-11T14:00:00Z", "patched"), ("2026-05-12T10:00:00Z", "verified")],
            False,
            "List the chronological status transitions of bug_901.",
        ),
        (
            "to-08",
            "vendor_eval",
            "milestone",
            [("2026-05-15T10:00:00Z", "rfp"), ("2026-05-20T10:00:00Z", "poc"), ("2026-05-25T10:00:00Z", "security-review"), ("2026-05-29T10:00:00Z", "signed")],
            True,
            "Order the vendor_eval milestones chronologically.",
        ),
        (
            "to-09",
            "node_7",
            "lifecycle",
            [("2026-06-01T02:00:00Z", "provisioned"), ("2026-06-01T02:05:00Z", "joined"), ("2026-06-01T02:30:00Z", "drained")],
            False,
            "What is the chronological lifecycle of node_7?",
        ),
        (
            "to-10",
            "quarterly_okr",
            "phase",
            [("2026-06-02T09:00:00Z", "draft"), ("2026-06-05T09:00:00Z", "aligned"), ("2026-06-20T09:00:00Z", "midterm"), ("2026-06-30T09:00:00Z", "scored")],
            True,
            "List the chronological phases for quarterly_okr.",
        ),
    ]

    cases: List[BenchmarkCase] = []
    for cid, ent, attr, timeline, shuffle_ingest, question in specs:
        actor = f"actor-{cid}"
        sess = f"sess-{cid}-main"
        wf = f"wf-{cid}-1"
        turns = _distractor_turns(cid, actor, sess, wf, rng, count=3)

        ordered_events = list(timeline)
        ingest_events = list(timeline)
        if shuffle_ingest:
            # Deterministic non-chronological ingestion order (backfill / delayed log arrival)
            ingest_events = [ingest_events[2], ingest_events[0], *ingest_events[3:], ingest_events[1]]

        for idx, (ts, val) in enumerate(ingest_events, start=1):
            turns.append(
                MemoryTurn(
                    turn_id=f"{cid}-ev{idx}",
                    actor_id=actor,
                    session_id=f"sess-{cid}-s{idx}",
                    workflow_id=wf,
                    scope="actor",
                    timestamp=ts,
                    entity=ent,
                    attribute=attr,
                    value=val,
                    statement=f"At {ts}, {ent} {attr} reached {val}.",
                    action="set",
                    is_event=True,
                )
            )

        expected_order = [val for _, val in ordered_events]
        cases.append(
            BenchmarkCase(
                case_id=cid,
                category="temporal_ordering",
                description=f"Chronological reconstruction of {len(expected_order)} events for {ent}.{attr} (out_of_order_ingest={shuffle_ingest}).",
                turns=turns,
                query=MemoryQuery(
                    actor_id=actor,
                    session_id=sess,
                    workflow_id=wf,
                    entity=ent,
                    attribute=attr,
                    question=question,
                    query_time="2026-07-01T00:00:00Z",
                    query_type="sequence",
                ),
                expected_answer=" -> ".join(expected_order),
                expected_order=expected_order,
            )
        )
    return cases


def _build_abstention(rng: random.Random) -> List[BenchmarkCase]:
    cases: List[BenchmarkCase] = []

    # 5 never-stored facts with rich in-scope distractors
    never_stored = [
        ("ab-01", "user", "passport_number", "What is the user's passport number?"),
        ("ab-02", "project", "hardware_serial", "What is the project hardware serial number?"),
        ("ab-03", "server", "rack_location", "Which datacenter rack hosts the server?"),
        ("ab-04", "customer", "credit_limit", "What is the customer's credit limit?"),
        ("ab-05", "flight", "seat_assignment", "What is the flight seat assignment?"),
    ]
    for cid, ent, attr, question in never_stored:
        actor = f"actor-{cid}"
        sess = f"sess-{cid}-1"
        wf = f"wf-{cid}-1"
        turns = _distractor_turns(cid, actor, sess, wf, rng, count=4)
        cases.append(
            BenchmarkCase(
                case_id=cid,
                category="abstention",
                description=f"Unanswerable query for never-stored {ent}.{attr} with 4 distractors.",
                turns=turns,
                query=MemoryQuery(
                    actor_id=actor,
                    session_id=sess,
                    workflow_id=wf,
                    entity=ent,
                    attribute=attr,
                    question=question,
                    query_time="2026-07-10T10:00:00Z",
                ),
                should_abstain=True,
            )
        )

    # 3 explicitly deleted facts (must abstain and must NOT resurface deleted value)
    deleted_specs = [
        ("ab-06", "session_token", "temp_code", "tok-99182", "What is the temporary session code?"),
        ("ab-07", "meeting", "dial_in_pin", "pin-44210", "What is the cancelled meeting dial-in PIN?"),
        ("ab-08", "user", "marketing_opt_in", "subscribed", "What is the user's marketing subscription status?"),
    ]
    for cid, ent, attr, old_val, question in deleted_specs:
        actor = f"actor-{cid}"
        sess = f"sess-{cid}-1"
        wf = f"wf-{cid}-1"
        turns = _distractor_turns(cid, actor, sess, wf, rng, count=3)
        turns.extend(
            [
                MemoryTurn(
                    turn_id=f"{cid}-t1",
                    actor_id=actor,
                    session_id=sess,
                    workflow_id=wf,
                    scope="actor",
                    timestamp="2026-07-01T10:00:00Z",
                    entity=ent,
                    attribute=attr,
                    value=old_val,
                    statement=f"Stored {ent} {attr} as {old_val}.",
                    action="set",
                ),
                MemoryTurn(
                    turn_id=f"{cid}-t2",
                    actor_id=actor,
                    session_id=sess,
                    workflow_id=wf,
                    scope="actor",
                    timestamp="2026-07-02T10:00:00Z",
                    entity=ent,
                    attribute=attr,
                    value="",
                    statement=f"Deleted {ent} {attr}.",
                    action="delete",
                ),
            ]
        )
        cases.append(
            BenchmarkCase(
                case_id=cid,
                category="abstention",
                description=f"Explicitly deleted fact {ent}.{attr} ({old_val}) must trigger abstention.",
                turns=turns,
                query=MemoryQuery(
                    actor_id=actor,
                    session_id=sess,
                    workflow_id=wf,
                    entity=ent,
                    attribute=attr,
                    question=question,
                    query_time="2026-07-03T10:00:00Z",
                ),
                should_abstain=True,
                stale_values=[old_val],
            )
        )

    # 2 cross-actor isolation abstention cases (fact exists ONLY under a different actor)
    isolation_specs = [
        ("ab-09", "wallet", "recovery_hint", "secret-alpha-hint", "What is the wallet recovery hint?"),
        ("ab-10", "hr", "salary_band", "L8-confidential", "What is the HR salary band?"),
    ]
    for cid, ent, attr, secret_val, question in isolation_specs:
        actor_a = f"actor-{cid}-alice"
        actor_b = f"actor-{cid}-bob"
        sess_a = f"sess-{cid}-alice"
        sess_b = f"sess-{cid}-bob"
        wf_a = f"wf-{cid}-alice"
        wf_b = f"wf-{cid}-bob"
        turns = _distractor_turns(cid, actor_b, sess_b, wf_b, rng, count=3)
        turns.append(
            MemoryTurn(
                turn_id=f"{cid}-secret",
                actor_id=actor_a,
                session_id=sess_a,
                workflow_id=wf_a,
                scope="actor",
                timestamp="2026-07-05T10:00:00Z",
                entity=ent,
                attribute=attr,
                value=secret_val,
                statement=f"Private {actor_a} record: {ent} {attr} is {secret_val}.",
                action="set",
            )
        )
        cases.append(
            BenchmarkCase(
                case_id=cid,
                category="abstention",
                description=f"Cross-actor tenant isolation: {ent}.{attr} exists only for {actor_a}, queried by {actor_b}.",
                turns=turns,
                query=MemoryQuery(
                    actor_id=actor_b,
                    session_id=sess_b,
                    workflow_id=wf_b,
                    entity=ent,
                    attribute=attr,
                    question=question,
                    query_time="2026-07-06T10:00:00Z",
                ),
                should_abstain=True,
                cross_actor_secrets=[secret_val],
            )
        )
    return cases


def _build_cross_session_retrieval(rng: random.Random) -> List[BenchmarkCase]:
    cases: List[BenchmarkCase] = []

    # 8 actor-scoped facts established in session 1 (optionally refined in session 2), queried in session 3
    actor_cross_specs = [
        ("cs-01", "user", "timezone", ["America/Los_Angeles"], "What timezone is the user in?"),
        ("cs-02", "repo", "default_branch", ["main"], "What is the repository default branch?"),
        ("cs-03", "agent", "alert_channel", ["#ops-alerts", "#sre-critical"], "Which alert channel does the agent notify?"),
        ("cs-04", "build", "docker_registry", ["ghcr.io/hanzoai"], "Which container registry is used for builds?"),
        ("cs-05", "user", "dietary_note", ["vegetarian", "vegan"], "What is the user's dietary preference?"),
        ("cs-06", "infra", "vpn_gateway", ["vpn-eu-west.hanzo.ai"], "What is the infrastructure VPN gateway?"),
        ("cs-07", "security", "key_rotation_days", ["90", "60"], "How frequently are API keys rotated in days?"),
        ("cs-08", "analytics", "warehouse", ["clickhouse-prod"], "Which analytics warehouse stores telemetry?"),
    ]
    for cid, ent, attr, values, question in actor_cross_specs:
        actor = f"actor-{cid}"
        turns = _distractor_turns(cid, actor, f"sess-{cid}-s1", f"wf-{cid}-1", rng, count=3)
        for i, val in enumerate(values, start=1):
            turns.append(
                MemoryTurn(
                    turn_id=f"{cid}-t{i}",
                    actor_id=actor,
                    session_id=f"sess-{cid}-s{i}",
                    workflow_id=f"wf-{cid}-{i}",
                    scope="actor",
                    timestamp=f"2026-08-0{i}T12:00:00Z",
                    entity=ent,
                    attribute=attr,
                    value=val,
                    statement=f"In session {i}, recorded {ent} {attr} as {val}.",
                    action="set" if i == 1 else "update",
                )
            )
        cases.append(
            BenchmarkCase(
                case_id=cid,
                category="cross_session_retrieval",
                description=f"Actor-scoped cross-session recall for {ent}.{attr} queried in fresh session s3.",
                turns=turns,
                query=MemoryQuery(
                    actor_id=actor,
                    session_id=f"sess-{cid}-s3",
                    workflow_id=f"wf-{cid}-3",
                    entity=ent,
                    attribute=attr,
                    question=question,
                    query_time="2026-08-10T12:00:00Z",
                ),
                expected_answer=values[-1],
                stale_values=values[:-1],
            )
        )

    # 2 global-scope fallback cases (org-wide shared knowledge queried from fresh actor/session)
    global_specs = [
        ("cs-09", "org", "sla_uptime", "99.95%", "What is the organization's SLA uptime target?"),
        ("cs-10", "org", "license_policy", "apache-2.0-or-mit", "What is the organization's preferred open-source license policy?"),
    ]
    for cid, ent, attr, val, question in global_specs:
        writer_actor = f"actor-{cid}-admin"
        reader_actor = f"actor-{cid}-engineer"
        turns = _distractor_turns(cid, reader_actor, f"sess-{cid}-reader", f"wf-{cid}-r", rng, count=2)
        turns.append(
            MemoryTurn(
                turn_id=f"{cid}-global",
                actor_id=writer_actor,
                session_id=f"sess-{cid}-admin",
                workflow_id=f"wf-{cid}-admin",
                scope="global",
                timestamp="2026-08-01T08:00:00Z",
                entity=ent,
                attribute=attr,
                value=val,
                statement=f"Global organization policy: {ent} {attr} is {val}.",
                action="set",
            )
        )
        cases.append(
            BenchmarkCase(
                case_id=cid,
                category="cross_session_retrieval",
                description=f"Global scope fallback recall of {ent}.{attr} from a distinct actor/session.",
                turns=turns,
                query=MemoryQuery(
                    actor_id=reader_actor,
                    session_id=f"sess-{cid}-reader",
                    workflow_id=f"wf-{cid}-r",
                    entity=ent,
                    attribute=attr,
                    question=question,
                    query_time="2026-08-12T08:00:00Z",
                ),
                expected_answer=val,
            )
        )
    return cases


def build_longmemeval_dataset(
    seed: int = 1347,
    categories: Optional[Sequence[str]] = None,
) -> List[BenchmarkCase]:
    """
    Build the deterministic 50-case LongMemEval-structured dataset (10 cases per
    category) for benchmarking Hanzo Agents memory policies.
    """
    rng = random.Random(seed)
    builders = {
        "knowledge_updates": _build_knowledge_updates,
        "contradiction_handling": _build_contradiction_handling,
        "temporal_ordering": _build_temporal_ordering,
        "abstention": _build_abstention,
        "cross_session_retrieval": _build_cross_session_retrieval,
    }
    selected = list(categories) if categories else list(ALL_CATEGORIES)
    cases: List[BenchmarkCase] = []
    for cat in selected:
        if cat not in builders:
            raise ValueError(f"Unknown category: {cat!r}. Expected one of {ALL_CATEGORIES}")
        cases.extend(builders[cat](rng))
    return cases
