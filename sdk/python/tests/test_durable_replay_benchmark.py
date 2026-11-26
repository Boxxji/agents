"""Comprehensive tests for HIP-1348 durable event-driven workflow replay (#78 / #85)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pytest

from benchmarks.durable_replay import (
    CRASH_EXIT_CODE,
    AtomicJsonCheckpointStore,
    BackendKind,
    CrashMode,
    CrashPlan,
    DurableReplayEngine,
    EphemeralCheckpointStore,
    EventType,
    ExternalSideEffectLedger,
    InvalidIdentifierError,
    JournalEvent,
    NonDeterministicReplayError,
    RunStatus,
    SqliteWalCheckpointStore,
    StepSpec,
    TenantIsolationError,
    build_default_workflow,
    compute_idempotency_key,
    invoke_worker_subprocess,
    render_score_table_markdown,
    run_crash_recovery_trial,
    run_full_benchmark,
    run_multi_crash_gauntlet,
    run_nondeterminism_divergence_suite,
    run_trivial_overhead_crossover,
)
from hanzo_agents.agent_workflow import AgentWorkflow


class _CaptureClient:
    """In-memory capture client verifying real AgentWorkflow lifecycle notifications."""

    def __init__(self) -> None:
        self.requests: List[Dict[str, Any]] = []
        self._current_workflow_context = None

    async def _async_request(
        self, method: str, url: str, json: Dict[str, Any]
    ) -> Dict[str, Any]:
        self.requests.append({"method": method, "url": url, "json": json})
        return {"ok": True}


class _StubAgent:
    """Minimal agent host compatible with hanzo_agents.agent_workflow.AgentWorkflow."""

    def __init__(self, node_id: str = "durable-test-agent") -> None:
        self.node_id = node_id
        self.client = _CaptureClient()
        self.hanzo_agents_server = "http://127.0.0.1:0"
        self.dev_mode = False
        self._current_execution_context = None


# ---------------------------------------------------------------------------
# 1. Deterministic Idempotency Keys & Store Persistence
# ---------------------------------------------------------------------------


def test_idempotency_key_matches_control_plane_formula() -> None:
    """Must match Go control-plane/internal/durable/idempotency.go sha256(org:runID:round:stepID)."""
    expected = hashlib.sha256(b"org-alpha:run-42:3:step_commit").hexdigest()
    actual = compute_idempotency_key("org-alpha", "run-42", 3, "step_commit")
    assert actual == expected
    assert len(actual) == 64

    # Different orgs or rounds MUST produce distinct keys.
    assert compute_idempotency_key("org-beta", "run-42", 3, "step_commit") != actual
    assert compute_idempotency_key("org-alpha", "run-42", 4, "step_commit") != actual


@pytest.mark.parametrize(
    "bad_val",
    [
        "",
        "   ",
        "../escape",
        "a/b",
        "a\\b",
        "run\x00id",
        " leadingspace",
        "trailing ",
    ],
)
def test_invalid_identifiers_rejected(bad_val: str) -> None:
    with pytest.raises(InvalidIdentifierError):
        compute_idempotency_key(bad_val, "run-1", 0, "step-1")
    with pytest.raises(InvalidIdentifierError):
        compute_idempotency_key("org-1", bad_val, 0, "step-1")
    with pytest.raises(InvalidIdentifierError):
        compute_idempotency_key("org-1", "run-1", 0, bad_val)

    with pytest.raises(InvalidIdentifierError):
        compute_idempotency_key("org-1", "run-1", -1, "step-1")


def test_sqlite_wal_pragmas_and_restart_persistence(tmp_path: Path) -> None:
    db_path = tmp_path.resolve() / "durable.sqlite"
    ledger_path = tmp_path.resolve() / "effects.sqlite"
    steps = build_default_workflow(4)

    store1 = SqliteWalCheckpointStore(db_path)
    ledger = ExternalSideEffectLedger(ledger_path)
    try:
        pragmas = store1.pragma_state()
        assert pragmas["journal_mode"] == "wal"
        assert pragmas["synchronous"] == "2"  # SQLite FULL

        engine1 = DurableReplayEngine(store1, external_ledger=ledger)
        tel1 = engine1.activate(
            org="hanzo-org",
            workflow_id="wf-persist",
            run_id="run-persist-1",
            actor_id="actor-1",
            steps=steps,
        )
        assert tel1.status == RunStatus.COMPLETED.value
        assert tel1.executed_steps == 4
        assert tel1.replayed_steps == 0
        assert store1.storage_bytes() > 0
    finally:
        store1.close()

    # Re-open from disk in a fresh store handle and verify replay skips all activities.
    store2 = SqliteWalCheckpointStore(db_path)
    try:
        events = store2.load_events("hanzo-org", "run-persist-1")
        event_types = [e.event_type for e in events]
        assert event_types[0] == EventType.WORKFLOW_STARTED
        assert event_types[-1] == EventType.WORKFLOW_COMPLETED
        assert event_types.count(EventType.ACTIVITY_COMPLETED) == 4
        assert len(store2.list_snapshots("hanzo-org", "run-persist-1")) == 4

        engine2 = DurableReplayEngine(store2, external_ledger=ledger)
        tel2 = engine2.activate(
            org="hanzo-org",
            workflow_id="wf-persist",
            run_id="run-persist-1",
            actor_id="actor-1",
            steps=steps,
        )
        assert tel2.status == RunStatus.COMPLETED.value
        assert tel2.executed_steps == 0
        assert tel2.replayed_steps == 4

        counts = ledger.get_applied_counts("hanzo-org", "run-persist-1")
        assert counts == {s.step_id: 1 for s in steps}
    finally:
        store2.close()
        ledger.close()


# ---------------------------------------------------------------------------
# 2. Real Cross-Process Crash-at-Every-Step & Multi-Crash Gauntlet
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "backend",
    [BackendKind.SQLITE_WAL, BackendKind.ATOMIC_JSONL],
)
@pytest.mark.parametrize(
    "crash_mode",
    [
        CrashMode.BEFORE_ACTIVITY_COMMIT,
        CrashMode.AFTER_EFFECT_BEFORE_ACK,
        CrashMode.AFTER_STEP_COMMIT,
    ],
)
def test_real_subprocess_crash_at_every_step_exactly_once(
    tmp_path: Path, backend: BackendKind, crash_mode: CrashMode
) -> None:
    """Acceptance test (#78): a run interrupted via os._exit(99) at EVERY step
    resumes in a brand-new OS process with each activity applied exactly once."""
    steps = build_default_workflow(4)
    work_root = tmp_path.resolve()

    for step_idx in range(len(steps)):
        trial = run_crash_recovery_trial(
            work_dir=work_root,
            backend=backend,
            steps=steps,
            crash_step_index=step_idx,
            crash_mode=crash_mode,
            enable_idempotency_guard=True,
            trial_id=f"t_{backend.value}_{crash_mode.value}_s{step_idx}",
        )
        assert trial["proc1_returncode"] == CRASH_EXIT_CODE
        assert trial["proc1_crashed_hard"] is True
        assert trial["proc2_returncode"] == 0
        assert trial["completed"] is True
        assert trial["exactly_once_all_steps"] is True
        assert trial["duplicate_side_effects"] == 0
        assert trial["applied_by_step"] == {s.step_id: 1 for s in steps}

        if crash_mode == CrashMode.AFTER_EFFECT_BEFORE_ACK:
            # External service MUST have deduplicated the crashed in-flight step.
            assert trial["external_dedup_hits"] == 1
            assert len(trial["distinct_os_pids"]) == 2
        elif step_idx > 0 or crash_mode == CrashMode.AFTER_STEP_COMMIT:
            if step_idx < len(steps) - 1 or crash_mode != CrashMode.AFTER_STEP_COMMIT:
                assert len(trial["distinct_os_pids"]) == 2


@pytest.mark.parametrize(
    "backend",
    [BackendKind.SQLITE_WAL, BackendKind.ATOMIC_JSONL],
)
def test_multi_crash_gauntlet_single_run_across_distinct_pids(
    tmp_path: Path, backend: BackendKind
) -> None:
    """Single run_id crashed at EVERY step 0..4 across 6 OS processes completes exactly-once."""
    steps = build_default_workflow(5)
    result = run_multi_crash_gauntlet(
        work_dir=tmp_path.resolve(),
        backend=backend,
        steps=steps,
        enable_idempotency_guard=True,
    )
    assert result["all_crashes_hard_exit_99"] is True
    assert result["completed"] is True
    assert result["exactly_once_all_steps"] is True
    assert result["duplicate_side_effects"] == 0
    assert result["distinct_os_pids_count"] >= 3
    assert result["applied_by_step"] == {s.step_id: 1 for s in steps}
    assert result["external_dedup_hits"] >= 1


# ---------------------------------------------------------------------------
# 3. Negative Regimes (Crash-After-Effect Window, Ephemeral Baseline, Non-Determinism)
# ---------------------------------------------------------------------------


def test_negative_regime_no_idempotency_guard_suffers_duplicate_side_effect(
    tmp_path: Path,
) -> None:
    """Proves why persisted checkpoints ALONE without activity idempotency_key
    suffer duplicate execution when crashing in the after_effect_before_ack window."""
    steps = build_default_workflow(4)
    trial = run_crash_recovery_trial(
        work_dir=tmp_path.resolve(),
        backend=BackendKind.SQLITE_WAL,
        steps=steps,
        crash_step_index=2,
        crash_mode=CrashMode.AFTER_EFFECT_BEFORE_ACK,
        enable_idempotency_guard=False,
        trial_id="neg_no_idem_s2",
    )
    assert trial["proc1_crashed_hard"] is True
    assert trial["completed"] is True
    assert trial["exactly_once_all_steps"] is False
    assert trial["duplicate_side_effects"] == 1
    assert trial["applied_by_step"][steps[2].step_id] == 2


def test_ephemeral_baseline_loses_state_and_duplicates_prior_steps(
    tmp_path: Path,
) -> None:
    """Control baseline: in-memory dict store loses all state across OS crash."""
    steps = build_default_workflow(4)
    trial = run_crash_recovery_trial(
        work_dir=tmp_path.resolve(),
        backend=BackendKind.EPHEMERAL_BASELINE,
        steps=steps,
        crash_step_index=3,
        crash_mode=CrashMode.BEFORE_ACTIVITY_COMMIT,
        enable_idempotency_guard=False,
        trial_id="ephemeral_ctrl_s3",
    )
    assert trial["proc1_crashed_hard"] is True
    assert trial["completed"] is True
    # Steps 0, 1, 2 were committed in proc1, forgotten on crash, and re-applied in proc2.
    assert trial["duplicate_side_effects"] == 3
    assert trial["exactly_once_all_steps"] is False
    assert trial["store_bytes"] == 0


def test_nondeterministic_replay_detected_fail_closed(tmp_path: Path) -> None:
    """Mutating workflow step order or activity type between crash and replay MUST fail closed."""
    steps = build_default_workflow(4)
    suite = run_nondeterminism_divergence_suite(
        work_dir=tmp_path.resolve(),
        steps=steps,
    )
    assert suite["total_cases"] == 4
    assert suite["detected_count"] == 4
    assert suite["zero_corruption_count"] == 4


def test_nondeterministic_input_payload_mutation_raises_fail_closed(
    tmp_path: Path,
) -> None:
    """Changing a recorded step's input payload on replay triggers NonDeterministicReplayError."""
    root = tmp_path.resolve()
    store = SqliteWalCheckpointStore(root / "nd.sqlite")
    ledger = ExternalSideEffectLedger(root / "nd_effects.sqlite")
    engine = DurableReplayEngine(store, external_ledger=ledger)
    try:
        steps_v1 = [
            StepSpec("s1", "plan", {"amount": 100}),
            StepSpec("s2", "charge", {"amount": 100}, await_signal="approve"),
        ]
        tel = engine.activate(
            org="org-nd",
            workflow_id="wf-nd",
            run_id="run-nd-payload",
            actor_id="actor-1",
            steps=steps_v1,
        )
        assert tel.status == RunStatus.SUSPENDED.value

        steps_v2 = [
            StepSpec("s1", "plan", {"amount": 999}),  # mutated input payload!
            StepSpec("s2", "charge", {"amount": 999}, await_signal="approve"),
        ]
        with pytest.raises(NonDeterministicReplayError):
            engine.activate(
                org="org-nd",
                workflow_id="wf-nd",
                run_id="run-nd-payload",
                actor_id="actor-1",
                steps=steps_v2,
                delivered_signals={"approve": {"approved": True}},
            )
        # External ledger must still only contain step s1 from v1 — zero corrupted writes.
        assert ledger.get_applied_counts("org-nd", "run-nd-payload") == {"s1": 1}
    finally:
        store.close()
        ledger.close()


# ---------------------------------------------------------------------------
# 4. Event-Driven Suspension & Resume (Zero Always-On Daemons)
# ---------------------------------------------------------------------------


def test_event_driven_suspension_and_signal_resume_zero_daemons(
    tmp_path: Path,
) -> None:
    """Workflow suspends cleanly awaiting a signal and resumes in a separate process."""
    root = tmp_path.resolve()
    store_path = root / "sig_store.sqlite"
    ledger_path = root / "sig_effects.sqlite"
    steps = [
        StepSpec("step_1_draft", "zen_draft", {"doc": "v1"}),
        StepSpec("step_2_publish", "s3_publish", {"doc": "v1"}, await_signal="approval"),
        StepSpec("step_3_notify", "cloud_notify", {"doc": "v1"}),
    ]

    out1 = invoke_worker_subprocess(
        work_dir=root,
        backend=BackendKind.SQLITE_WAL,
        store_path=store_path,
        external_ledger_path=ledger_path,
        org="hanzo-org",
        workflow_id="wf-signal",
        run_id="run-sig-1",
        actor_id="actor-sig",
        steps=steps,
        tag="pass1_suspend",
    )
    assert out1.returncode == 0
    assert out1.result is not None
    assert out1.result["telemetry"]["status"] == RunStatus.SUSPENDED.value
    assert out1.result["telemetry"]["awaiting_signal"] == "approval"
    assert out1.result["telemetry"]["executed_steps"] == 1

    # Brand-new OS process wakes up ONLY when the external signal event arrives.
    out2 = invoke_worker_subprocess(
        work_dir=root,
        backend=BackendKind.SQLITE_WAL,
        store_path=store_path,
        external_ledger_path=ledger_path,
        org="hanzo-org",
        workflow_id="wf-signal",
        run_id="run-sig-1",
        actor_id="actor-sig",
        steps=steps,
        delivered_signals={"approval": {"approved_by": "reviewer-1"}},
        tag="pass2_resume",
    )
    assert out2.returncode == 0
    assert out2.result is not None
    assert out2.result["telemetry"]["status"] == RunStatus.COMPLETED.value
    assert out2.result["telemetry"]["replayed_steps"] == 1
    assert out2.result["telemetry"]["executed_steps"] == 2

    ledger = ExternalSideEffectLedger(ledger_path)
    try:
        assert ledger.get_applied_counts("hanzo-org", "run-sig-1") == {
            "step_1_draft": 1,
            "step_2_publish": 1,
            "step_3_notify": 1,
        }
    finally:
        ledger.close()


# ---------------------------------------------------------------------------
# 5. Symlink-Safe Resolved Paths (macOS /var -> /private/var Compatibility)
# ---------------------------------------------------------------------------


def test_symlinked_tmpdir_compatibility(tmp_path: Path) -> None:
    """Both persisted stores and subprocess workers resolve symlinked TMPDIRs cleanly."""
    real_dir = (tmp_path / "real_storage").resolve()
    real_dir.mkdir(parents=True, exist_ok=True)
    symlink_dir = tmp_path / "symlinked_tmp"
    os.symlink(real_dir, symlink_dir)

    steps = build_default_workflow(3)
    for backend in (BackendKind.SQLITE_WAL, BackendKind.ATOMIC_JSONL):
        trial = run_crash_recovery_trial(
            work_dir=symlink_dir,
            backend=backend,
            steps=steps,
            crash_step_index=1,
            crash_mode=CrashMode.AFTER_EFFECT_BEFORE_ACK,
            enable_idempotency_guard=True,
            trial_id=f"symlink_{backend.value}",
        )
        assert trial["completed"] is True
        assert trial["exactly_once_all_steps"] is True
        assert trial["duplicate_side_effects"] == 0


# ---------------------------------------------------------------------------
# 6. Concurrency (Parallel Workflows Sharing WAL / JSONL Stores)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "backend",
    [BackendKind.SQLITE_WAL, BackendKind.ATOMIC_JSONL],
)
def test_concurrent_multi_workflow_execution(
    tmp_path: Path, backend: BackendKind
) -> None:
    """12 concurrent workflows sharing one store produce zero lock failures or corruption."""
    root = tmp_path.resolve()
    store_path = (
        root / "concurrent.sqlite"
        if backend == BackendKind.SQLITE_WAL
        else root / "concurrent_jsonl"
    )
    ledger_path = root / "concurrent_effects.sqlite"
    store = (
        SqliteWalCheckpointStore(store_path)
        if backend == BackendKind.SQLITE_WAL
        else AtomicJsonCheckpointStore(store_path)
    )
    ledger = ExternalSideEffectLedger(ledger_path)
    engine = DurableReplayEngine(store, external_ledger=ledger)
    steps = build_default_workflow(4)

    async def _run_all() -> List[Any]:
        tasks = [
            engine.activate_async(
                org=f"tenant-{i % 3}",
                workflow_id=f"wf-{i}",
                run_id=f"concurrent-run-{i}",
                actor_id=f"actor-{i}",
                steps=steps,
            )
            for i in range(12)
        ]
        return await asyncio.gather(*tasks)

    try:
        results = asyncio.run(_run_all())
        assert len(results) == 12
        for i, tel in enumerate(results):
            assert tel.status == RunStatus.COMPLETED.value
            assert tel.executed_steps == 4
            counts = ledger.get_applied_counts(f"tenant-{i % 3}", f"concurrent-run-{i}")
            assert counts == {s.step_id: 1 for s in steps}
    finally:
        store.close()
        ledger.close()


# ---------------------------------------------------------------------------
# 7. Security, Multi-Tenant Isolation & Torn-Tail Recovery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "backend",
    [BackendKind.SQLITE_WAL, BackendKind.ATOMIC_JSONL],
)
def test_tenant_and_actor_isolation(tmp_path: Path, backend: BackendKind) -> None:
    """Verify strict (org, run_id) tenant separation and fail-closed actor ownership."""
    root = tmp_path.resolve()
    store = (
        SqliteWalCheckpointStore(root / "sec.sqlite")
        if backend == BackendKind.SQLITE_WAL
        else AtomicJsonCheckpointStore(root / "sec_jsonl")
    )
    ledger = ExternalSideEffectLedger(root / "sec_effects.sqlite")
    engine = DurableReplayEngine(store, external_ledger=ledger)
    steps = [
        StepSpec("step_1", "task_a", {"secret": "tenant-data"}),
        StepSpec("step_2", "task_b", {"secret": "tenant-data"}, await_signal="go"),
    ]

    try:
        # Same run_id in two distinct orgs MUST remain completely isolated.
        tel_a = engine.activate(
            org="org-alpha",
            workflow_id="wf-shared",
            run_id="shared-run-id",
            actor_id="actor-alpha",
            steps=steps,
        )
        tel_b = engine.activate(
            org="org-beta",
            workflow_id="wf-shared",
            run_id="shared-run-id",
            actor_id="actor-beta",
            steps=steps,
        )
        assert tel_a.status == RunStatus.SUSPENDED.value
        assert tel_b.status == RunStatus.SUSPENDED.value
        assert ledger.get_applied_counts("org-alpha", "shared-run-id") == {"step_1": 1}
        assert ledger.get_applied_counts("org-beta", "shared-run-id") == {"step_1": 1}

        # Spoofing actor_id on org-alpha's run MUST fail closed.
        with pytest.raises(TenantIsolationError):
            engine.activate(
                org="org-alpha",
                workflow_id="wf-shared",
                run_id="shared-run-id",
                actor_id="spoofed-actor",
                steps=steps,
                delivered_signals={"go": {}},
            )

        # Spoofing workflow_id on org-alpha's run MUST fail closed.
        with pytest.raises(TenantIsolationError):
            engine.activate(
                org="org-alpha",
                workflow_id="different-wf",
                run_id="shared-run-id",
                actor_id="actor-alpha",
                steps=steps,
                delivered_signals={"go": {}},
            )

        # Path traversal identifiers MUST be rejected before touching disk.
        with pytest.raises(InvalidIdentifierError):
            engine.activate(
                org="../evil",
                workflow_id="wf-x",
                run_id="run-x",
                actor_id="actor-x",
                steps=steps,
            )
    finally:
        store.close()
        ledger.close()


def test_atomic_jsonl_recovers_from_torn_tail_line(tmp_path: Path) -> None:
    """If an OS crash leaves an unterminated partial line in journal.jsonl, recovery ignores it."""
    root = tmp_path.resolve()
    store = AtomicJsonCheckpointStore(root / "torn_jsonl")
    ledger = ExternalSideEffectLedger(root / "torn_effects.sqlite")
    engine = DurableReplayEngine(store, external_ledger=ledger)
    steps = [
        StepSpec("step_1", "stage_a", {"k": 1}),
        StepSpec("step_2", "stage_b", {"k": 2}, await_signal="resume"),
    ]
    try:
        tel1 = engine.activate(
            org="org-torn",
            workflow_id="wf-torn",
            run_id="run-torn",
            actor_id="actor-torn",
            steps=steps,
        )
        assert tel1.status == RunStatus.SUSPENDED.value

        # Inject a torn partial write (no trailing newline) directly into journal.jsonl.
        journal_files = list((root / "torn_jsonl").rglob("journal.jsonl"))
        assert len(journal_files) == 1
        with journal_files[0].open("ab") as fh:
            fh.write(b'{"sequence":999,"event_type":"TRUNCAT')

        # Recovery must ignore the torn tail and complete step_2 cleanly.
        tel2 = engine.activate(
            org="org-torn",
            workflow_id="wf-torn",
            run_id="run-torn",
            actor_id="actor-torn",
            steps=steps,
            delivered_signals={"resume": {"ok": True}},
        )
        assert tel2.status == RunStatus.COMPLETED.value
        assert tel2.replayed_steps == 1
        assert tel2.executed_steps == 1
        assert ledger.get_applied_counts("org-torn", "run-torn") == {
            "step_1": 1,
            "step_2": 1,
        }
    finally:
        store.close()
        ledger.close()


# ---------------------------------------------------------------------------
# 8. hanzo_agents.agent_workflow.AgentWorkflow Regression Integration
# ---------------------------------------------------------------------------


def test_agent_workflow_integration_and_replay_dedup(tmp_path: Path) -> None:
    """Verifies real AgentWorkflow lifecycle hooks fire on execution and skip on replay."""
    root = tmp_path.resolve()
    store = SqliteWalCheckpointStore(root / "aw.sqlite")
    ledger = ExternalSideEffectLedger(root / "aw_effects.sqlite")
    agent = _StubAgent(node_id="hanzo-replay-node")
    aw = AgentWorkflow(agent)
    engine = DurableReplayEngine(
        store,
        external_ledger=ledger,
        agent_workflow=aw,
    )
    steps = [
        StepSpec("step_1", "plan_task", {"x": 1}),
        StepSpec("step_2", "apply_patch", {"x": 2}, await_signal="approved"),
        StepSpec("step_3", "publish_result", {"x": 3}),
    ]

    try:
        tel1 = engine.activate(
            org="hanzo-org",
            workflow_id="wf-aw",
            run_id="run-aw-1",
            actor_id="actor-aw",
            steps=steps,
        )
        assert tel1.status == RunStatus.SUSPENDED.value
        assert tel1.executed_steps == 1
        # Step 1 emitted start (running) + complete (succeeded) via AgentWorkflow.
        event_reqs_pass1 = [
            r
            for r in agent.client.requests
            if r["url"].endswith("/v1/workflow/executions/events")
        ]
        assert len(event_reqs_pass1) == 2
        statuses_pass1 = [r["json"]["status"] for r in event_reqs_pass1]
        assert statuses_pass1 == ["running", "succeeded"]
        assert all(r["json"]["run_id"] == "run-aw-1" for r in event_reqs_pass1)

        # Resume after signal: step_1 MUST be replayed (NOT re-invoked via AgentWorkflow).
        agent.client.requests.clear()
        tel2 = engine.activate(
            org="hanzo-org",
            workflow_id="wf-aw",
            run_id="run-aw-1",
            actor_id="actor-aw",
            steps=steps,
            delivered_signals={"approved": {"ok": True}},
        )
        assert tel2.status == RunStatus.COMPLETED.value
        assert tel2.replayed_steps == 1
        assert tel2.executed_steps == 2
        event_reqs_pass2 = [
            r
            for r in agent.client.requests
            if r["url"].endswith("/v1/workflow/executions/events")
        ]
        statuses_pass2 = [r["json"]["status"] for r in event_reqs_pass2]
        assert statuses_pass2 == ["running", "succeeded", "running", "succeeded"]
        assert ledger.get_applied_counts("hanzo-org", "run-aw-1") == {
            "step_1": 1,
            "step_2": 1,
            "step_3": 1,
        }
    finally:
        store.close()
        ledger.close()


# ---------------------------------------------------------------------------
# 9. End-to-End Benchmark Runner & Artifact Serialization
# ---------------------------------------------------------------------------


def test_full_benchmark_runner_and_artifact_generation(tmp_path: Path) -> None:
    """Runs bounded end-to-end benchmark and verifies JSON + Markdown artifacts."""
    out_dir = (tmp_path / "artifacts").resolve()
    report = run_full_benchmark(
        num_steps=3,
        trivial_runs=3,
        work_dir=(tmp_path / "work").resolve(),
        output_dir=out_dir,
    )
    raw_path = out_dir / "raw_results.json"
    md_path = out_dir / "score_table.md"
    assert raw_path.exists()
    assert md_path.exists()

    loaded = json.loads(raw_path.read_text(encoding="utf-8"))
    assert loaded["suite_digest"] == report["suite_digest"]
    assert len(loaded["suite_digest"]) == 64

    summary = loaded["experiment_a_crash_at_every_step"]["summary"]
    assert summary["sqlite_wal_durable"]["exactly_once_rate"] == 1.0
    assert summary["sqlite_wal_durable"]["total_duplicate_side_effects"] == 0
    assert summary["atomic_jsonl_durable"]["exactly_once_rate"] == 1.0
    assert summary["atomic_jsonl_durable"]["total_duplicate_side_effects"] == 0
    assert summary["ephemeral_baseline_no_idempotency"]["total_duplicate_side_effects"] > 0

    neg_b = loaded["experiment_b_negative_no_idempotency_window"]["summary"]
    assert neg_b["sqlite_wal_no_idempotency_after_effect_crash"]["total_duplicate_side_effects"] == 3
    assert neg_b["atomic_jsonl_no_idempotency_after_effect_crash"]["total_duplicate_side_effects"] == 3

    md = render_score_table_markdown(loaded)
    assert "sqlite_wal_durable" in md
    assert "atomic_jsonl_durable" in md
    assert "NonDeterministicReplayError" in md
    # Ensure no absolute user home paths leak into artifacts.
    assert "/Users/" not in md
    assert "/home/" not in md


# ---------------------------------------------------------------------------
# 10. Regression Tests for Codex Review Findings (PR #122)
# ---------------------------------------------------------------------------


def test_atomic_jsonl_truncates_torn_tail_preserving_multi_signal_resume(
    tmp_path: Path,
) -> None:
    """Truncating an unterminated torn tail prevents O_APPEND from corrupting WorkflowResumed."""
    root = tmp_path.resolve()
    store = AtomicJsonCheckpointStore(root / "torn_multi_sig")
    ledger = ExternalSideEffectLedger(root / "torn_multi_sig_effects.sqlite")
    engine = DurableReplayEngine(store, external_ledger=ledger)
    steps = [
        StepSpec("step_1", "stage_a", {"k": 1}),
        StepSpec("step_2", "stage_b", {"k": 2}, await_signal="go"),
        StepSpec("step_3", "stage_c", {"k": 3}, await_signal="next"),
    ]
    try:
        tel1 = engine.activate(
            org="org-torn-sig",
            workflow_id="wf-torn-sig",
            run_id="run-torn-sig",
            actor_id="actor-torn",
            steps=steps,
        )
        assert tel1.status == RunStatus.SUSPENDED.value
        assert tel1.awaiting_signal == "go"

        # Inject unterminated torn JSON tail directly into journal.jsonl.
        journal_files = list((root / "torn_multi_sig").rglob("journal.jsonl"))
        assert len(journal_files) == 1
        journal_path = journal_files[0]
        with journal_path.open("ab") as fh:
            fh.write(b'{"sequence":999,"event_type":"TRUNCAT')

        # Resume with 'go': torn tail must be truncated before WorkflowResumed('go') is appended,
        # step_2 runs, and the workflow suspends waiting for 'next'.
        tel2 = engine.activate(
            org="org-torn-sig",
            workflow_id="wf-torn-sig",
            run_id="run-torn-sig",
            actor_id="actor-torn",
            steps=steps,
            delivered_signals={"go": {"ok": True}},
        )
        assert tel2.status == RunStatus.SUSPENDED.value
        assert tel2.awaiting_signal == "next"
        assert tel2.replayed_steps == 1
        assert tel2.executed_steps == 1

        # Every line in journal.jsonl must now be valid JSON, including WorkflowResumed('go').
        raw_lines = journal_path.read_text(encoding="utf-8").splitlines()
        parsed_events = [json.loads(line) for line in raw_lines if line.strip()]
        resumed_signals = [
            ev["payload"]["signal_name"]
            for ev in parsed_events
            if ev["event_type"] == EventType.WORKFLOW_RESUMED.value
        ]
        assert resumed_signals == ["go"]

        # Next activation with 'next' must NOT re-suspend on 'go' and must complete step_3 cleanly.
        tel3 = engine.activate(
            org="org-torn-sig",
            workflow_id="wf-torn-sig",
            run_id="run-torn-sig",
            actor_id="actor-torn",
            steps=steps,
            delivered_signals={"next": {"ok": True}},
        )
        assert tel3.status == RunStatus.COMPLETED.value
        assert tel3.replayed_steps == 2
        assert tel3.executed_steps == 1
        assert ledger.get_applied_counts("org-torn-sig", "run-torn-sig") == {
            "step_1": 1,
            "step_2": 1,
            "step_3": 1,
        }

        # Direct append_event() without a prior read must also truncate an unterminated tail.
        with journal_path.open("ab") as fh:
            fh.write(b'{"partial_direct_append":')
        events_before = store.load_events("org-torn-sig", "run-torn-sig")
        next_seq = events_before[-1].sequence + 1
        # Re-inject torn tail right before append_event() without calling load_events().
        with journal_path.open("ab") as fh:
            fh.write(b'{"partial_before_append":')
        store.append_event(
            JournalEvent(
                sequence=next_seq,
                previous_sequence=next_seq - 1,
                event_type=EventType.WORKFLOW_RESUMED,
                org="org-torn-sig",
                workflow_id="wf-torn-sig",
                run_id="run-torn-sig",
                execution_id="exec-direct",
                actor_id="actor-torn",
                round_idx=0,
                payload={"signal_name": "post_direct", "signal_payload": {}},
                pid=os.getpid(),
                timestamp="2026-01-01T00:00:00+00:00",
            )
        )
        events_after = store.load_events("org-torn-sig", "run-torn-sig")
        assert events_after[-1].sequence == next_seq
        assert events_after[-1].payload["signal_name"] == "post_direct"
    finally:
        store.close()
        ledger.close()


@pytest.mark.parametrize(
    "backend",
    [BackendKind.SQLITE_WAL, BackendKind.ATOMIC_JSONL],
)
def test_shortened_replay_definition_rejects_unconsumed_pending_and_completed(
    tmp_path: Path, backend: BackendKind
) -> None:
    """Shortened replay definitions leaving pending or completed activities unconsumed fail closed."""
    root = tmp_path.resolve()
    store_path = (
        root / f"short_{backend.value}.sqlite"
        if backend == BackendKind.SQLITE_WAL
        else root / f"short_{backend.value}_jsonl"
    )
    ledger_path = root / f"short_{backend.value}_effects.sqlite"
    full_steps = [
        StepSpec("s1", "stage_1", {"v": 1}),
        StepSpec("s2", "stage_2", {"v": 2}),
    ]

    # Case 1: Crash AFTER_EFFECT_BEFORE_ACK on s2 so s1 is completed and s2 is pending_scheduled.
    crash_out = invoke_worker_subprocess(
        work_dir=root / f"crash_{backend.value}",
        backend=backend,
        store_path=store_path,
        external_ledger_path=ledger_path,
        org="org-short",
        workflow_id="wf-short-pending",
        run_id="run-short-pending",
        actor_id="actor-short",
        steps=full_steps,
        crash_plan=CrashPlan(
            mode=CrashMode.AFTER_EFFECT_BEFORE_ACK,
            target_step_index=1,
            enable_idempotency_guard=True,
        ),
        tag="crash_s2",
    )
    assert crash_out.crashed_as_expected is True

    store = (
        SqliteWalCheckpointStore(store_path)
        if backend == BackendKind.SQLITE_WAL
        else AtomicJsonCheckpointStore(store_path)
    )
    ledger = ExternalSideEffectLedger(ledger_path)
    engine = DurableReplayEngine(store, external_ledger=ledger)
    try:
        # Resuming with only [s1] leaves pending_scheduled=s2 unconsumed -> MUST fail closed.
        with pytest.raises(NonDeterministicReplayError, match="pending scheduled activity"):
            engine.activate(
                org="org-short",
                workflow_id="wf-short-pending",
                run_id="run-short-pending",
                actor_id="actor-short",
                steps=[full_steps[0]],
            )
        pending_events = store.load_events("org-short", "run-short-pending")
        assert EventType.WORKFLOW_COMPLETED not in [e.event_type for e in pending_events]
        assert pending_events[-1].event_type == EventType.WORKFLOW_FAILED

        # Case 2: Complete s1 and s2, suspend on s3, then resume with shortened [s1] only.
        three_steps = [
            StepSpec("s1", "stage_1", {"v": 1}),
            StepSpec("s2", "stage_2", {"v": 2}),
            StepSpec("s3", "stage_3", {"v": 3}, await_signal="go"),
        ]
        tel = engine.activate(
            org="org-short",
            workflow_id="wf-short-completed",
            run_id="run-short-completed",
            actor_id="actor-short",
            steps=three_steps,
        )
        assert tel.status == RunStatus.SUSPENDED.value

        with pytest.raises(NonDeterministicReplayError, match="completed activities"):
            engine.activate(
                org="org-short",
                workflow_id="wf-short-completed",
                run_id="run-short-completed",
                actor_id="actor-short",
                steps=[three_steps[0]],
                delivered_signals={"go": {"ok": True}},
            )
        completed_events = store.load_events("org-short", "run-short-completed")
        assert EventType.WORKFLOW_COMPLETED not in [e.event_type for e in completed_events]
        assert completed_events[-1].event_type == EventType.WORKFLOW_FAILED
    finally:
        store.close()
        ledger.close()


def test_repeated_benchmark_and_overhead_in_same_work_dir_stay_isolated(
    tmp_path: Path,
) -> None:
    """Reusing work_dir across repeated benchmark or overhead invocations stays isolated."""
    shared_work = (tmp_path / "reused_work").resolve()

    ov1 = run_trivial_overhead_crossover(
        work_dir=shared_work, trivial_runs=3, six_step_runs=2
    )
    ov2 = run_trivial_overhead_crossover(
        work_dir=shared_work, trivial_runs=3, six_step_runs=2
    )
    for ov in (ov1, ov2):
        for regime in ("trivial_1_step", "standard_6_step"):
            for backend_name in (
                BackendKind.SQLITE_WAL.value,
                BackendKind.ATOMIC_JSONL.value,
            ):
                stats = ov[regime][backend_name]
                assert stats["checkpoint_write_latency"]["count"] > 0
                assert stats["total_store_bytes"] > 0

    rep1 = run_full_benchmark(num_steps=3, trivial_runs=2, work_dir=shared_work)
    rep2 = run_full_benchmark(num_steps=3, trivial_runs=2, work_dir=shared_work)
    for rep in (rep1, rep2):
        summary_a = rep["experiment_a_crash_at_every_step"]["summary"]
        for cohort_name, cohort in summary_a.items():
            assert cohort["all_crashed_hard_exit_99"] is True, cohort_name
        assert summary_a["sqlite_wal_durable"]["exactly_once_rate"] == 1.0
        assert summary_a["atomic_jsonl_durable"]["exactly_once_rate"] == 1.0

        ov_d = rep["experiment_d_negative_overhead_crossover"]["trivial_1_step"]
        assert ov_d["sqlite_wal"]["checkpoint_write_latency"]["count"] > 0
        assert ov_d["atomic_jsonl"]["checkpoint_write_latency"]["count"] > 0

