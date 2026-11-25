from hanzo_agents.learned_memory_policy import PolicyControlledMemory


def test_learned_policy_hard_overrides_and_abstention():
    pm = PolicyControlledMemory(
        default_scope="session",
        default_scope_id="t1",
        min_evidence_count=1,
        min_answer_confidence=0.65,
    )

    t1 = pm.apply_policy(
        key_slot="deploy_region",
        content="eu-west-1",
        source_identity="src:1",
        confidence=0.9,
        utility_hint=0.8,
        expires_at="2026-12-01T00:00:00Z",
    )
    assert t1.action == "ADD"
    assert pm.record_task_outcome(t1.decision_id, 1.0) is True

    # Update slot
    t2 = pm.apply_policy(
        key_slot="deploy_region",
        content="us-east-1",
        source_identity="src:1",
        confidence=0.92,
        utility_hint=0.85,
        expires_at="2026-12-01T00:00:00Z",
    )
    assert t2.action == "UPDATE"
    assert t2.record is not None
    assert t2.record.version == 2

    # Hard sensitive-data isolation override
    sens = pm.apply_policy(
        key_slot="secret_key",
        content="sk-live-123",
        source_identity="src:2",
        contains_restricted_sensitive_data=True,
    )
    assert sens.action == "NOOP"
    assert sens.overridden_by_hard_policy is True

    # Query succeeds before expiry
    ans = pm.query_with_abstention(
        key_slots=["deploy_region"], now_iso="2026-11-16T00:00:00Z"
    )
    assert ans["status"] == "ANSWER"
    assert len(ans["records"]) == 1

    # Stale forgetting after TTL expiry triggers INSUFFICIENT_EVIDENCE abstention
    forgotten = pm.forget_stale(now_iso="2026-12-02T00:00:00Z")
    assert forgotten == 1

    abstained = pm.query_with_abstention(
        key_slots=["deploy_region"], now_iso="2026-12-02T00:00:00Z"
    )
    assert abstained["status"] == "INSUFFICIENT_EVIDENCE"
    assert len(abstained["records"]) == 0
