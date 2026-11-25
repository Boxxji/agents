from hanzo_agents.reflective_memory import ReflectiveMemory


def test_belief_separation_and_contradiction_resolution():
    mem = ReflectiveMemory(default_scope="session", default_scope_id="t1")

    obs1 = mem.retain(
        topic="ci-build",
        statement="the regression test failed",
        source_identity="ci:1",
    )

    # Unsupported belief must be rejected (0% unsupported memory rate)
    rejected = False
    try:
        mem.reflect(
            kind="HYPOTHESIS",
            topic="ci-build",
            statement="we suspect an allocator alias",
            supporting_observation_ids=[],
            source_identity="agent:1",
        )
    except ValueError:
        rejected = True
    assert rejected

    hyp = mem.reflect(
        kind="HYPOTHESIS",
        topic="ci-build",
        statement="we suspect an allocator alias",
        supporting_observation_ids=[obs1.id],
        source_identity="agent:1",
        confidence=0.75,
    )
    assert hyp.status == "ACTIVE"
    assert hyp.supporting_observation_ids == [obs1.id]

    obs2 = mem.retain(
        topic="ci-build",
        statement="allocator alias verified clean; root cause is timeout",
        source_identity="ci:2",
    )
    challenged = mem.challenge_belief(hyp.id, obs2.id)
    assert challenged.status == "REFUTED"

    # Observations remain immutable and ACTIVE
    obs_list = mem.recall(kinds=["OBSERVATION"], topic="ci-build")
    assert len(obs_list) == 2
    assert len(mem.recall(kinds=["HYPOTHESIS"], topic="ci-build")) == 0


def test_tenant_isolation_and_index_rebuild():
    mem = ReflectiveMemory(default_scope="session", default_scope_id="t1")
    obs = mem.retain(
        topic="sec",
        statement="RBAC filter applied pre-retrieval",
        source_identity="ev:1",
        acl_roles=["admin"],
        embedding=[1.0, 0.0],
    )
    assert len(mem.recall(scope="session", scope_id="t2", caller_roles=["admin"])) == 0
    assert len(mem.recall(scope="session", scope_id="t1", caller_roles=["guest"])) == 0
    assert len(mem.recall(scope="session", scope_id="t1", caller_roles=["admin"])) == 1

    mem._vectors.clear()
    stats = mem.rebuild_derived_index(embed_fn=lambda _: [1.0, 0.0])
    assert stats["rebuilt_count"] == 1
    assert obs.id in next(iter(mem._vectors.keys()))
