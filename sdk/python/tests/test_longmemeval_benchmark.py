"""
Correctness, concurrency, security (tenant/actor isolation), and regression tests
for the LongMemEval temporal memory policy benchmark (`hanzoai/agents#75`, HIP-1347).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from benchmarks.longmemeval import (
    ALL_CATEGORIES,
    ControlPlaneMemoryTransport,
    DeterministicHashEmbedder,
    InMemoryHanzoAgentsClient,
    build_default_policies,
    build_longmemeval_dataset,
    build_memory_interface,
    cosine_similarity,
    evaluate_policy,
    main as cli_main,
    run_benchmark,
    run_threshold_sweep,
)


@pytest.fixture(autouse=True)
def _mute_debug_logs(monkeypatch):
    monkeypatch.setattr("hanzo_agents.logger.log_debug", lambda *args, **kwargs: None)


# ---------------------------------------------------------------------------
# 1. Correctness: Wire contract & real unmodified Hanzo SDK memory classes
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_real_memory_interface_scope_hierarchy_and_vector_contract():
    """Verify unmodified MemoryClient/ScopedMemoryClient/GlobalMemoryClient/MemoryInterface."""
    transport = ControlPlaneMemoryTransport()
    client = InMemoryHanzoAgentsClient(transport=transport)

    iface = build_memory_interface(
        client,
        actor_id="actor-1",
        session_id="sess-1",
        workflow_id="wf-1",
    )

    # Global -> Actor -> Session -> Workflow precedence
    await iface.global_scope.set("cfg.region", "global-default")
    assert await iface.get("cfg.region") == "global-default"

    await iface.actor("actor-1").set("cfg.region", "actor-eu")
    assert await iface.get("cfg.region") == "actor-eu"

    await iface.session("sess-1").set("cfg.region", "session-us")
    assert await iface.get("cfg.region") == "session-us"

    await iface.workflow("wf-1").set("cfg.region", "workflow-ap")
    assert await iface.get("cfg.region") == "workflow-ap"

    # Explicit scope reads bypass fallback chain
    assert await iface.global_scope.get("cfg.region") == "global-default"
    assert await iface.actor("actor-1").get("cfg.region") == "actor-eu"
    assert await iface.session("sess-1").get("cfg.region") == "session-us"

    # list_keys and delete
    assert await iface.actor("actor-1").list_keys() == ["cfg.region"]
    await iface.workflow("wf-1").delete("cfg.region")
    assert await iface.get("cfg.region") == "session-us"

    # Vector set / filtered search / delete
    emb = DeterministicHashEmbedder(dim=32, seed=42)
    await iface.actor("actor-1").set_vector(
        "v1", emb.embed("deploy region eu-central"), metadata={"tag": "deploy"}
    )
    await iface.actor("actor-1").set_vector(
        "v2", emb.embed("lunch menu ramen"), metadata={"tag": "food"}
    )
    hits = await iface.actor("actor-1").similarity_search(
        emb.embed("deploy region"), top_k=5, filters={"tag": "deploy"}
    )
    assert len(hits) == 1
    assert hits[0]["key"] == "v1"
    assert hits[0]["score"] > 0.5

    await iface.actor("actor-1").delete_vector("v1")
    hits_after = await iface.actor("actor-1").similarity_search(
        emb.embed("deploy region"), top_k=5, filters={"tag": "deploy"}
    )
    assert hits_after == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_documents_existing_memory_client_exists_404_quirk():
    """
    Document real unmodified MemoryClient.exists() behavior discovered during
    benchmark audit: `get()` returns `default=None` on HTTP 404 without raising,
    so `exists()` currently returns True for absent keys.
    """
    client = InMemoryHanzoAgentsClient()
    iface = build_memory_interface(
        client, actor_id="actor-q", session_id="sess-q", workflow_id="wf-q"
    )
    assert await iface.get("never_written", default=None) is None
    # Documented SDK bug (see docs/research/hip-1347-longmemeval/README.md):
    assert await iface.exists("never_written") is True


@pytest.mark.unit
def test_dataset_and_embedder_determinism():
    ds1 = build_longmemeval_dataset(seed=1347)
    ds2 = build_longmemeval_dataset(seed=1347)
    assert ds1 == ds2
    assert len(ds1) == 50

    counts = {cat: sum(1 for c in ds1 if c.category == cat) for cat in ALL_CATEGORIES}
    assert counts == {cat: 10 for cat in ALL_CATEGORIES}

    emb = DeterministicHashEmbedder(dim=128, seed=1347)
    v_same_1 = emb.embed("project primary model zen-4-ultra")
    v_same_2 = emb.embed("What is the project primary model?")
    v_unrelated = emb.embed("office wifi network hanzo-corp-5g")
    assert cosine_similarity(v_same_1, v_same_2) > cosine_similarity(
        v_same_1, v_unrelated
    )
    assert cosine_similarity(v_same_1, v_same_2) > 0.45


# ---------------------------------------------------------------------------
# 2. Falsifiable Hypotheses & Honest Negative Results Verification
# ---------------------------------------------------------------------------


@pytest.mark.functional
@pytest.mark.asyncio
async def test_longmemeval_hypotheses_and_negative_results():
    """Verify all 4 falsifiable hypotheses and documented failure modes."""
    report = await run_benchmark(seed=1347, min_score=0.45, top_k=5)
    by_name = {row["policy_name"]: row for row in report["scores"]}
    assert set(by_name.keys()) == {
        "kv_scoped",
        "kv_fallback_chain",
        "vector_top_k",
        "vector_thresholded",
        "hybrid_scope_policy",
    }

    kv_scoped = by_name["kv_scoped"]
    kv_fallback = by_name["kv_fallback_chain"]
    vec_topk = by_name["vector_top_k"]
    vec_thr = by_name["vector_thresholded"]
    hybrid = by_name["hybrid_scope_policy"]

    # H1 (Negative result): session-only KV fails cross-session retrieval and temporal ordering
    assert kv_scoped["by_category"]["cross_session_retrieval"]["exact_match"] == 0.0
    assert kv_scoped["by_category"]["temporal_ordering"]["exact_match"] == 0.0

    # H2 (Negative result): native hierarchical fallback suffers stale shadowing when
    # older session/workflow records shadow newer actor updates
    assert kv_fallback["stale_shadowing_errors"] >= 4
    assert kv_fallback["by_category"]["knowledge_updates"]["exact_match"] < 1.0
    assert kv_fallback["by_category"]["cross_session_retrieval"]["exact_match"] == 1.0

    # H3 (Negative result): unthresholded vector top-K fails abstention completely
    assert vec_topk["abstention_f1"] == 0.0
    assert vec_topk["false_positive_rate"] == 1.0
    assert vec_topk["recall_at_k"] > vec_topk["exact_match"]

    # Thresholding dramatically improves abstention & overall EM over vector_top_k
    assert vec_thr["abstention_f1"] > vec_topk["abstention_f1"]
    assert vec_thr["exact_match"] > vec_topk["exact_match"]
    # Honest limitation: timestamp-only vector_thresholded still misses contradiction detection
    assert vec_thr["contradiction_detection_rate"] == 0.0
    assert vec_thr["by_category"]["contradiction_handling"]["exact_match"] < 1.0

    # H4: Hybrid policy over existing Hanzo primitives achieves top scores with zero leaks
    assert hybrid["exact_match"] > vec_thr["exact_match"]
    assert hybrid["contradiction_detection_rate"] == 1.0
    assert hybrid["temporal_order_accuracy"] == 1.0
    assert hybrid["abstention_f1"] == 1.0
    assert hybrid["stale_shadowing_errors"] == 0

    # Security invariant: zero cross-actor leaks across every evaluated policy
    for name, row in by_name.items():
        assert row["cross_actor_leaks"] == 0, f"{name} leaked cross-actor data"


@pytest.mark.functional
@pytest.mark.asyncio
async def test_threshold_sweep_exposes_over_abstention_tradeoff():
    """Verify threshold sweep captures both low-threshold FPR and high-threshold over-abstention."""
    sweep = await run_threshold_sweep(seed=1347, thresholds=(0.0, 0.45, 0.75))
    by_thr = {row["min_score"]: row for row in sweep}
    # At 0.0, unanswerable FPR is high
    assert by_thr[0.0]["false_positive_rate"] >= 0.8
    # At 0.45, optimal operating zone
    assert by_thr[0.45]["exact_match"] > by_thr[0.0]["exact_match"]
    assert by_thr[0.45]["abstention_f1"] == 1.0
    # At 0.75, overly strict threshold degrades recall & abstention precision
    assert by_thr[0.75]["exact_match"] < by_thr[0.45]["exact_match"]
    assert by_thr[0.75]["abstention_precision"] < by_thr[0.45]["abstention_precision"]


# ---------------------------------------------------------------------------
# 3. Concurrency Tests (real async MemoryInterface + concurrent policy runs)
# ---------------------------------------------------------------------------


@pytest.mark.functional
@pytest.mark.asyncio
async def test_concurrent_async_memory_reads_writes_and_vector_searches():
    """
    Drive concurrent writes, reads, vector indexing, and similarity searches
    across 20 parallel tasks spanning multiple actors and sessions.
    """
    transport = ControlPlaneMemoryTransport()
    client = InMemoryHanzoAgentsClient(transport=transport)
    embedder = DeterministicHashEmbedder(dim=64, seed=99)

    async def worker(actor_idx: int) -> tuple[str, str, str]:
        actor_id = f"actor-conc-{actor_idx}"
        session_id = f"sess-conc-{actor_idx}"
        workflow_id = f"wf-conc-{actor_idx}"
        iface = build_memory_interface(
            client,
            actor_id=actor_id,
            session_id=session_id,
            workflow_id=workflow_id,
        )
        key = f"item.{actor_idx}"
        val = f"payload-{actor_idx}"
        await iface.actor(actor_id).set(key, {"val": val})
        await iface.session(session_id).set("session.token", f"tok-{actor_idx}")
        await iface.actor(actor_id).set_vector(
            f"vec-{actor_idx}",
            embedder.embed(f"service record {actor_idx} {val}"),
            metadata={"actor": actor_id, "val": val},
        )

        got_actor = await iface.actor(actor_id).get(key)
        got_sess = await iface.session(session_id).get("session.token")
        hits = await iface.actor(actor_id).similarity_search(
            embedder.embed(f"service record {actor_idx}"), top_k=5
        )
        assert len(hits) == 1
        assert hits[0]["metadata"]["actor"] == actor_id
        return got_actor["val"], got_sess, hits[0]["metadata"]["val"]

    results = await asyncio.gather(*(worker(i) for i in range(20)))
    for i, (kv_val, sess_val, vec_val) in enumerate(results):
        assert kv_val == f"payload-{i}"
        assert sess_val == f"tok-{i}"
        assert vec_val == f"payload-{i}"


@pytest.mark.functional
@pytest.mark.asyncio
async def test_concurrent_policy_evaluations_are_isolated():
    """Run all 5 policies concurrently via asyncio.gather and verify consistent results."""
    cases = build_longmemeval_dataset(seed=1347)
    embedder = DeterministicHashEmbedder(dim=128, seed=1347)
    policies = build_default_policies(embedder, min_score=0.45, top_k=5)

    concurrent_scores = await asyncio.gather(
        *(evaluate_policy(pol, cases) for pol in policies)
    )
    by_name = {s.policy_name: s for s in concurrent_scores}
    assert by_name["hybrid_scope_policy"].exact_match == 1.0
    assert by_name["vector_top_k"].abstention_f1 == 0.0
    assert all(s.cross_actor_leaks == 0 for s in concurrent_scores)


# ---------------------------------------------------------------------------
# 4. Security & Tenant Isolation Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_strict_cross_actor_and_cross_session_isolation():
    """
    Verify no KV or vector data leaks across `actor_id` or `session_id`
    boundaries, even for identical keys and identical query embeddings.
    """
    transport = ControlPlaneMemoryTransport()
    client = InMemoryHanzoAgentsClient(transport=transport)
    embedder = DeterministicHashEmbedder(dim=64, seed=7)

    alice = build_memory_interface(
        client, actor_id="actor-alice", session_id="sess-alice-1", workflow_id="wf-a"
    )
    bob = build_memory_interface(
        client, actor_id="actor-bob", session_id="sess-bob-1", workflow_id="wf-b"
    )

    secret_vec = embedder.embed("private api recovery phrase")
    await alice.actor("actor-alice").set("secret.key", "alice-only-secret")
    await alice.session("sess-alice-1").set("ephemeral.note", "alice-session-secret")
    await alice.actor("actor-alice").set_vector(
        "secret-vec", secret_vec, metadata={"owner": "alice"}
    )

    # Bob cannot read Alice's KV via fallback chain, actor scope, or session scope
    assert await bob.get("secret.key", default=None) is None
    assert await bob.actor("actor-bob").get("secret.key", default=None) is None
    assert await bob.session("sess-bob-1").get("ephemeral.note", default=None) is None
    assert await bob.actor("actor-bob").list_keys() == []

    # Bob cannot search Alice's vector even with the exact same embedding vector
    bob_hits = await bob.actor("actor-bob").similarity_search(secret_vec, top_k=10)
    assert bob_hits == []

    # Session isolation for the same actor across two sessions
    alice_s2 = build_memory_interface(
        client, actor_id="actor-alice", session_id="sess-alice-2", workflow_id="wf-a2"
    )
    assert await alice_s2.session("sess-alice-2").get("ephemeral.note") is None
    # Actor memory persists across Alice's own sessions
    assert await alice_s2.get("secret.key") == "alice-only-secret"

    # Global scope is intentionally shared across actors (must NOT hold private tenant data)
    await alice.global_scope.set("org.policy", "public-org-standard")
    assert await bob.get("org.policy") == "public-org-standard"


# ---------------------------------------------------------------------------
# 5. Regression & CLI Reproducibility Tests
# ---------------------------------------------------------------------------


@pytest.mark.functional
@pytest.mark.asyncio
async def test_benchmark_deterministic_regression_reproducibility():
    """Two runs with the same seed must produce identical deterministic scores & outcomes."""
    run1 = await run_benchmark(seed=1347, min_score=0.45, top_k=5)
    run2 = await run_benchmark(seed=1347, min_score=0.45, top_k=5)

    assert len(run1["scores"]) == len(run2["scores"]) == 5
    for s1, s2 in zip(run1["scores"], run2["scores"]):
        for metric_key in (
            "policy_name",
            "total_cases",
            "exact_match",
            "recall_at_k",
            "abstention_precision",
            "abstention_recall",
            "abstention_f1",
            "false_positive_rate",
            "contradiction_detection_rate",
            "temporal_order_accuracy",
            "stale_shadowing_errors",
            "cross_actor_leaks",
            "avg_retrieval_tokens",
            "total_retrieval_tokens",
            "api_calls",
        ):
            assert s1[metric_key] == s2[metric_key], (
                f"Non-deterministic metric {metric_key} for {s1['policy_name']}"
            )
        assert [
            (o["case_id"], o["exact_match"], o["predicted_answer"], o["abstained"])
            for o in s1["outcomes"]
        ] == [
            (o["case_id"], o["exact_match"], o["predicted_answer"], o["abstained"])
            for o in s2["outcomes"]
        ]


@pytest.mark.functional
def test_cli_generates_raw_json_and_markdown(tmp_path: Path):
    out_json = tmp_path / "raw_results.json"
    out_md = tmp_path / "score_table.md"

    rc = cli_main(
        [
            "--seed",
            "1347",
            "--sweep-thresholds",
            "--output",
            str(out_json),
            "--markdown",
            str(out_md),
            "--quiet",
        ]
    )
    assert rc == 0
    assert out_json.exists()
    assert out_md.exists()

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["issue"] == "hanzoai/agents#75"
    assert payload["hip"] == "HIP-1347"
    assert payload["config"]["total_cases"] == 50
    assert len(payload["scores"]) == 5
    assert len(payload["threshold_sweep"]) == 7

    md_text = out_md.read_text(encoding="utf-8")
    assert "| `hybrid_scope_policy` |" in md_text
    assert "| `vector_top_k` |" in md_text


@pytest.mark.unit
@pytest.mark.asyncio
async def test_transport_kv_responses_decode_json_like_go_raw_message():
    """Verify KV set/get/list responses emit decoded JSON matching Go json.RawMessage."""
    transport = ControlPlaneMemoryTransport()
    headers = {
        "X-Actor-ID": "actor-wire",
        "X-Session-ID": "sess-wire",
        "X-Workflow-ID": "wf-wire",
    }

    set_resp = await transport.handle_request(
        "POST",
        "http://inmemory.hanzo.local/v1/memory/set",
        json={"key": "numeric_str", "data": "123", "scope": "actor"},
        headers=headers,
    )
    assert set_resp.status_code == 200
    assert set_resp.json()["data"] == "123"
    assert isinstance(set_resp.json()["data"], str)

    get_explicit = await transport.handle_request(
        "POST",
        "http://inmemory.hanzo.local/v1/memory/get",
        json={"key": "numeric_str", "scope": "actor"},
        headers=headers,
    )
    assert get_explicit.status_code == 200
    assert get_explicit.json()["data"] == "123"

    get_fallback = await transport.handle_request(
        "POST",
        "http://inmemory.hanzo.local/v1/memory/get",
        json={"key": "numeric_str"},
        headers=headers,
    )
    assert get_fallback.status_code == 200
    assert get_fallback.json()["data"] == "123"

    list_resp = await transport.handle_request(
        "GET",
        "http://inmemory.hanzo.local/v1/memory/list",
        params={"scope": "actor"},
        headers=headers,
    )
    assert list_resp.status_code == 200
    assert list_resp.json()[0]["data"] == "123"

    # Dict payloads are also returned directly (not JSON-stringified)
    await transport.handle_request(
        "POST",
        "http://inmemory.hanzo.local/v1/memory/set",
        json={"key": "obj_val", "data": {"nested": "123", "n": 123}, "scope": "actor"},
        headers=headers,
    )
    get_obj = await transport.handle_request(
        "POST",
        "http://inmemory.hanzo.local/v1/memory/get",
        json={"key": "obj_val", "scope": "actor"},
        headers=headers,
    )
    assert get_obj.json()["data"] == {"nested": "123", "n": 123}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hybrid_policy_delete_clears_event_history_for_sequence_recall():
    """Ingesting to-01 then deleting its entity/attribute must abstain without resurrecting events."""
    from benchmarks.longmemeval.dataset import MemoryTurn
    from benchmarks.longmemeval.policies import HybridScopePolicy

    cases = build_longmemeval_dataset(seed=1347, categories=["temporal_ordering"])
    to_01 = next(c for c in cases if c.case_id == "to-01")

    embedder = DeterministicHashEmbedder(dim=128, seed=1347)
    policy = HybridScopePolicy(embedder=embedder, min_score=0.45)
    client = InMemoryHanzoAgentsClient()

    for turn in to_01.turns:
        await policy.ingest(client, turn)

    # Sanity check: before delete, sequence recall succeeds
    before = await policy.recall(client, to_01.query)
    assert not before.abstained
    assert before.ordered_answers == list(to_01.expected_order)

    # Delete the event-backed fact
    delete_turn = MemoryTurn(
        turn_id="to-01-delete",
        actor_id=to_01.query.actor_id,
        session_id=to_01.query.session_id,
        workflow_id=to_01.query.workflow_id,
        scope="actor",
        timestamp="2026-09-01T00:00:00Z",
        entity=to_01.query.entity,
        attribute=to_01.query.attribute,
        value="",
        statement=f"Delete {to_01.query.entity} {to_01.query.attribute}.",
        action="delete",
    )
    await policy.ingest(client, delete_turn)

    after = await policy.recall(client, to_01.query)
    assert after.abstained is True
    assert after.answer is None
    assert after.ordered_answers == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hybrid_policy_global_delete_removes_global_vectors():
    """Deleting a global-scope fact (cs-09) must clear global vectors so fallback cannot resurrect it."""
    from benchmarks.longmemeval.dataset import MemoryTurn
    from benchmarks.longmemeval.policies import HybridScopePolicy

    cases = build_longmemeval_dataset(seed=1347, categories=["cross_session_retrieval"])
    cs_09 = next(c for c in cases if c.case_id == "cs-09")

    embedder = DeterministicHashEmbedder(dim=128, seed=1347)
    policy = HybridScopePolicy(embedder=embedder, min_score=0.45)
    client = InMemoryHanzoAgentsClient()

    for turn in cs_09.turns:
        await policy.ingest(client, turn)

    before = await policy.recall(client, cs_09.query)
    assert not before.abstained
    assert before.answer == cs_09.expected_answer

    global_turn = next(t for t in cs_09.turns if t.scope == "global")
    delete_turn = MemoryTurn(
        turn_id="cs-09-global-delete",
        actor_id=global_turn.actor_id,
        session_id=global_turn.session_id,
        workflow_id=global_turn.workflow_id,
        scope="global",
        timestamp="2026-09-01T00:00:00Z",
        entity=global_turn.entity,
        attribute=global_turn.attribute,
        value="",
        statement=f"Delete global {global_turn.entity} {global_turn.attribute}.",
        action="delete",
    )
    await policy.ingest(client, delete_turn)

    after = await policy.recall(client, cs_09.query)
    assert after.abstained is True
    assert after.answer is None


@pytest.mark.unit
def test_category_filter_preserves_exact_seeded_cases():
    """Filtering --categories with seed 43 must yield the exact same cases as the full corpus slice."""
    full_corpus = build_longmemeval_dataset(seed=43)
    filtered_abstention = build_longmemeval_dataset(seed=43, categories=["abstention"])
    expected_abstention = [c for c in full_corpus if c.category == "abstention"]
    assert filtered_abstention == expected_abstention

    filtered_multi = build_longmemeval_dataset(
        seed=43, categories=["cross_session_retrieval", "temporal_ordering"]
    )
    expected_multi = [
        *[c for c in full_corpus if c.category == "cross_session_retrieval"],
        *[c for c in full_corpus if c.category == "temporal_ordering"],
    ]
    assert filtered_multi == expected_multi


@pytest.mark.functional
def test_cli_filtered_categories_sweep_uses_same_filtered_cases(tmp_path: Path):
    """--categories abstention --sweep-thresholds must sweep the exact filtered cases."""
    out_json = tmp_path / "abstention_sweep.json"

    rc = cli_main(
        [
            "--seed",
            "43",
            "--categories",
            "abstention",
            "--policies",
            "vector_thresholded",
            "--threshold",
            "0.45",
            "--top-k",
            "8",
            "--sweep-thresholds",
            "--output",
            str(out_json),
            "--quiet",
        ]
    )
    assert rc == 0

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["config"]["categories"] == ["abstention"]
    assert payload["config"]["total_cases"] == 10
    baseline = payload["scores"][0]
    sweep_at_45 = next(
        row for row in payload["threshold_sweep"] if row["min_score"] == 0.45
    )
    assert sweep_at_45["exact_match"] == baseline["exact_match"]
    assert sweep_at_45["abstention_f1"] == baseline["abstention_f1"]
    assert sweep_at_45["false_positive_rate"] == baseline["false_positive_rate"]
    assert sweep_at_45["avg_retrieval_tokens"] == baseline["avg_retrieval_tokens"]

    # Confirm the filtered sweep matches categories=["abstention"] and differs from unfiltered 50-case sweep
    direct_filtered = asyncio.run(
        run_threshold_sweep(seed=43, categories=["abstention"], thresholds=(0.0, 0.45))
    )
    full_sweep = asyncio.run(run_threshold_sweep(seed=43, thresholds=(0.0, 0.45)))
    filtered_at_0 = next(
        row for row in payload["threshold_sweep"] if row["min_score"] == 0.0
    )
    direct_at_0 = next(row for row in direct_filtered if row["min_score"] == 0.0)
    full_at_0 = next(row for row in full_sweep if row["min_score"] == 0.0)
    assert filtered_at_0 == direct_at_0
    assert filtered_at_0["exact_match"] < full_at_0["exact_match"]

