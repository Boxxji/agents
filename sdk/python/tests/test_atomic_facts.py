from hanzo_agents.atomic_facts import (
    AtomicFactInput,
    AtomicFactMemory,
)


def test_add_and_noop_idempotency():
    mem = AtomicFactMemory(default_scope="session", default_scope_id="tenant-1")

    res1 = mem.ingest_fact(
        AtomicFactInput(
            subject="User:1",
            predicate="preferred_editor",
            object="Neovim",
            source_identity="event:1",
            confidence=0.95,
        ),
        embedding=[1.0, 0.0, 0.0],
    )
    assert res1.action == "ADD"
    assert res1.record is not None
    assert res1.record.version == 1

    # Re-ingesting identical fact with casing/spacing differences -> NOOP
    res2 = mem.ingest_fact(
        AtomicFactInput(
            subject=" user:1 ",
            predicate="PREFERRED_EDITOR",
            object="neovim",
            source_identity="event:2",
            confidence=0.98,
        ),
        embedding=[1.0, 0.0, 0.0],
    )
    assert res2.action == "NOOP"
    assert res2.target_fact_id == res1.record.id

    metrics = mem.get_metrics()
    assert metrics["events_processed"] == 2
    assert metrics["durable_writes"] == 1
    assert metrics["write_amplification"] == 0.5


def test_contradiction_reconciliation_update_and_lineage():
    mem = AtomicFactMemory(default_scope="session", default_scope_id="tenant-1")

    v1 = mem.ingest_fact(
        AtomicFactInput(
            subject="Cluster:Prod",
            predicate="primary_region",
            object="eu-central-1",
            source_identity="commit:111",
            confidence=0.9,
        ),
        embedding=[0.9, 0.1, 0.0],
    )
    v2 = mem.ingest_fact(
        AtomicFactInput(
            subject="Cluster:Prod",
            predicate="primary_region",
            object="us-west-2",
            source_identity="commit:222",
            confidence=0.95,
        ),
        embedding=[0.1, 0.9, 0.0],
    )

    assert v2.action == "UPDATE"
    assert v2.record is not None
    assert v2.record.version == 2
    assert v2.record.supersedes_id == v1.record.id
    assert v2.superseded_record is not None
    assert v2.superseded_record.lifecycle == "SUPERSEDED"

    active = mem.query_facts(subject="Cluster:Prod", predicate="primary_region")
    assert len(active) == 1
    assert active[0].object == "us-west-2"

    history = mem.query_facts(
        subject="Cluster:Prod",
        predicate="primary_region",
        include_lifecycle=["ACTIVE", "SUPERSEDED"],
    )
    assert len(history) == 2


def test_tenant_isolation_and_rbac_prefiltering():
    mem = AtomicFactMemory(default_scope="session", default_scope_id="tenant-a")

    mem.ingest_fact(
        AtomicFactInput(
            subject="Billing",
            predicate="plan_tier",
            object="enterprise",
            source_identity="doc:a",
            acl_roles=["admin"],
        ),
        scope="session",
        scope_id="tenant-a",
        embedding=[0.0, 1.0, 0.0],
    )
    mem.ingest_fact(
        AtomicFactInput(
            subject="Billing",
            predicate="plan_tier",
            object="hobby",
            source_identity="doc:b",
        ),
        scope="session",
        scope_id="tenant-b",
        embedding=[0.0, 1.0, 0.0],
    )

    # Cross-tenant query never leaks tenant-a data
    b_hits = mem.query_facts(scope="session", scope_id="tenant-b", caller_roles=["admin"])
    assert len(b_hits) == 1
    assert b_hits[0].object == "hobby"

    # Unauthorized role in tenant-a gets 0 hits
    unauth = mem.query_facts(scope="session", scope_id="tenant-a", caller_roles=["viewer"])
    assert len(unauth) == 0

    # Authorized role in tenant-a gets exact hit
    auth = mem.query_facts(scope="session", scope_id="tenant-a", caller_roles=["admin"])
    assert len(auth) == 1
    assert auth[0].object == "enterprise"


def test_derived_index_rebuildability():
    mem = AtomicFactMemory(default_scope="session", default_scope_id="tenant-1")

    added = mem.ingest_fact(
        AtomicFactInput(
            subject="Router",
            predicate="fallback_model",
            object="gpt-5.4",
            source_identity="cfg:1",
        ),
        embedding=[0.6, 0.8, 0.0],
    )
    original_id = added.record.id

    # Purge derived vectors
    mem._vectors.clear()
    assert len(mem.query_facts(query_embedding=[0.6, 0.8, 0.0])) == 0

    # Rebuild derived index from immutable source ledger
    stats = mem.rebuild_derived_index(embed_fn=lambda _: [0.6, 0.8, 0.0])
    assert stats["rebuilt_count"] == 1

    hits = mem.query_facts(query_embedding=[0.6, 0.8, 0.0])
    assert len(hits) == 1
    assert hits[0].id == original_id
    assert hits[0].object == "gpt-5.4"
