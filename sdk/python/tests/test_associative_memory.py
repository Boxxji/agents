from hanzo_agents.associative_memory import AssociativeMemory


def test_bounded_consolidation_and_associative_recall():
    amem = AssociativeMemory(
        default_scope="session",
        default_scope_id="t1",
        max_edges_per_node=2,
        min_association_score=0.15,
    )

    ev1 = amem.ingest_event(
        category="incident",
        content="Redis connection pool exhausted under spike",
        tags=["redis", "pool", "outage"],
        source_identity="inc:1",
    )
    amem.ingest_event(
        category="decision",
        content="Cap Redis pool per worker and enable circuit breaker",
        tags=["redis", "pool", "breaker"],
        source_identity="dec:1",
    )
    amem.ingest_event(
        category="code_change",
        content="Updated redis pool max_clients in config",
        tags=["redis", "pool", "config"],
        source_identity="pr:1",
    )
    amem.ingest_event(
        category="lesson",
        content="Bound redis pool size before load test",
        tags=["redis", "pool", "loadtest"],
        source_identity="pm:1",
    )

    stats = amem.consolidate_bounded(max_edges_per_node=2)
    assert stats["processed_events"] == 4
    assert stats["edges_pruned_by_bound"] > 0

    recalled = amem.recall_associative(tags=["outage"], expand_hops=1)
    assert len(recalled) >= 2
    for n in recalled:
        assert len(n.links) <= 2

    # Rebuild derived graph and vector index from immutable raw events
    rebuilt = amem.rebuild_derived_graph_and_index(
        max_edges_per_node=2, embed_fn=lambda _: [1.0, 0.0]
    )
    assert rebuilt["processed_events"] == 4
    assert len(amem._vectors) == 4
    assert f"session:t1:amem_raw_event:{ev1.id}" in amem._raw_events
