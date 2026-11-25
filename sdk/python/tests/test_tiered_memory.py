from hanzo_agents.tiered_memory import TieredMemory


def test_unified_tiered_api_and_compaction():
    backing = {}
    mem = TieredMemory(default_scope="session", default_scope_id="t1", backing_store=backing)

    ep1 = mem.record_episode(
        run_id="run-1",
        step_index=0,
        action="check_db",
        observation="connection reset",
        source_identity="ev:1",
    )
    ep2 = mem.record_episode(
        run_id="run-1",
        step_index=1,
        action="reconnect_pool",
        observation="ok",
        outcome="success",
        source_identity="ev:2",
    )

    proc = mem.extract_procedure_from_run(
        run_id="run-1", name="db-reconnect", trigger_pattern="connection reset"
    )
    assert proc.tier == "procedural"
    assert proc.lineage_ids == [ep1.id, ep2.id]

    sem, compacted_ids = mem.compact_episodic(run_id="run-1", topic="db-recovery")
    assert sem is not None
    assert compacted_ids == [ep1.id, ep2.id]
    assert len(mem.retrieve(tiers=["episodic"], run_id="run-1")) == 0

    # Stateless process restart recovery
    restarted = TieredMemory(default_scope="session", default_scope_id="t1", backing_store=backing)
    recovery = restarted.recover_run_state("run-1", trigger_pattern="connection reset")
    assert recovery["last_step_index"] == 1
    assert len(recovery["steps"]) == 2
    assert len(recovery["applicable_procedures"]) == 1


def test_tenant_isolation_and_index_rebuild():
    backing = {}
    mem = TieredMemory(default_scope="session", default_scope_id="t1", backing_store=backing)
    rec = mem.record_semantic(
        topic="auth",
        statement="Use mTLS",
        source_identity="doc:1",
        acl_roles=["sec"],
        embedding=[1.0, 0.0],
    )

    assert len(mem.retrieve(scope="session", scope_id="t2", caller_roles=["sec"])) == 0
    assert len(mem.retrieve(scope="session", scope_id="t1", caller_roles=["guest"])) == 0
    assert len(mem.retrieve(scope="session", scope_id="t1", caller_roles=["sec"])) == 1

    mem._vectors.clear()
    stats = mem.rebuild_derived_index(embed_fn=lambda _: [1.0, 0.0])
    assert stats["rebuilt_count"] == 1
    assert len(mem._vectors) == 1
    assert rec.id in next(iter(mem._vectors.keys()))
