"""Comprehensive test suite for the OSWorld computer-workflow Visor benchmark (HIP-1348)."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
from pathlib import Path

import pytest

from benchmarks.osworld_visor import (
    AdversarialOrphanLeakerAgent,
    AdversarialSandboxEscapeAgent,
    CostTracker,
    NoOpBaselineAgent,
    OSWorldVisorRunner,
    OracleWorkflowAgent,
    PartialCapabilityAgent,
    SandboxProfile,
    TaskStatus,
    VisorSandboxManager,
    build_osworld_subset,
    compute_manifest_digest,
    format_score_table,
    load_external_osworld_dir,
    probe_desktop_tooling,
)
from benchmarks.osworld_visor.__main__ import main as cli_main
from hanzo_agents.types import AIConfig


@pytest.mark.functional
def test_oracle_all_10_tasks_succeed_with_real_os_metrics_on_both_profiles(tmp_path: Path) -> None:
    """All 10 real multi-step tasks pass on cold_sandbox and warm_cas_visor with real OS telemetry."""
    with OSWorldVisorRunner(
        workspace_scale="medium",
        max_concurrency=1,
        base_dir=tmp_path / "bench",
    ) as runner:
        report = asyncio.run(
            runner.run_suite(
                agents=[OracleWorkflowAgent()],
                profiles=[SandboxProfile.COLD_SANDBOX, SandboxProfile.WARM_CAS_VISOR],
                warm_visor_cas=True,
            )
        )

    assert len(report.cohorts) == 2
    assert len(report.results) == 20

    cold_cohort = next(c for c in report.cohorts if c.profile == SandboxProfile.COLD_SANDBOX)
    warm_cohort = next(c for c in report.cohorts if c.profile == SandboxProfile.WARM_CAS_VISOR)

    assert cold_cohort.total_tasks == 10
    assert cold_cohort.succeeded_tasks == 10
    assert cold_cohort.success_rate == 100.0
    assert cold_cohort.cleanup_clean_tasks == 10
    assert cold_cohort.escape_blocks == 0

    assert warm_cohort.total_tasks == 10
    assert warm_cohort.succeeded_tasks == 10
    assert warm_cohort.success_rate == 100.0
    assert warm_cohort.cleanup_clean_tasks == 10
    assert warm_cohort.escape_blocks == 0

    # Verify 100% real positive OS measurements (zero hardcoded constants).
    cold_starts = []
    for res in report.results:
        assert res.status == TaskStatus.SUCCESS, f"{res.task_id} failed: {res.evaluator_details}"
        assert res.success is True
        assert res.score == 1.0
        assert res.cleanup.cleanup_clean is True
        assert res.cleanup.workspace_removed is True
        assert res.cleanup.cas_template_intact is True
        assert res.security.escaped is False
        assert res.telemetry.cold_start_ms > 0.0
        assert res.telemetry.subprocess_boot_ms > 0.0
        assert res.telemetry.wall_ms > 0.0
        assert res.telemetry.peak_rss_kib > 1000
        assert res.telemetry.rss_source in {"os.wait4", "os.wait4+proc_vmhwm", "proc_vmhwm"}
        assert res.telemetry.subprocess_launches >= 2
        cold_starts.append(res.telemetry.cold_start_ms)

    # Real timing measurements vary across runs rather than repeating a synthetic constant.
    assert len(set(round(v, 3) for v in cold_starts)) > 5

    # Warm CAS Visor must drastically cut copied workspace bytes via read-only hardlinks.
    assert warm_cohort.total_hardlinked_files > 0
    assert cold_cohort.total_hardlinked_files == 0
    assert warm_cohort.total_bytes_copied < cold_cohort.total_bytes_copied * 0.15
    assert all(
        r.telemetry.cas_cache_hit is True
        for r in report.results
        if r.profile == SandboxProfile.WARM_CAS_VISOR
    )

    if report.hardware_info.get("proc_io_available"):
        assert cold_cohort.total_wchar_bytes > 0
        assert warm_cohort.total_wchar_bytes > 0
        assert warm_cohort.total_wchar_bytes < cold_cohort.total_wchar_bytes


@pytest.mark.functional
def test_partial_and_noop_baselines_show_profile_parity(tmp_path: Path) -> None:
    """partial_agent achieves 6/10 parity and noop_agent achieves 0/10 across both profiles."""
    with OSWorldVisorRunner(
        workspace_scale="tiny",
        max_concurrency=1,
        base_dir=tmp_path / "baselines",
    ) as runner:
        report = asyncio.run(
            runner.run_suite(
                agents=[PartialCapabilityAgent(), NoOpBaselineAgent()],
                profiles=[SandboxProfile.COLD_SANDBOX, SandboxProfile.WARM_CAS_VISOR],
                warm_visor_cas=True,
            )
        )

    by_key = {(c.agent_name, c.profile): c for c in report.cohorts}
    assert by_key[("partial_agent", SandboxProfile.COLD_SANDBOX)].succeeded_tasks == 6
    assert by_key[("partial_agent", SandboxProfile.WARM_CAS_VISOR)].succeeded_tasks == 6
    assert by_key[("noop_agent", SandboxProfile.COLD_SANDBOX)].succeeded_tasks == 0
    assert by_key[("noop_agent", SandboxProfile.WARM_CAS_VISOR)].succeeded_tasks == 0
    for cohort in report.cohorts:
        assert cohort.cleanup_clean_tasks == 10
        assert cohort.mean_peak_rss_kib > 1000


@pytest.mark.functional
def test_cold_cache_tiny_workspace_negative_crossover_regime(tmp_path: Path) -> None:
    """Honest negative regime: cold CAS build on tiny tasks writes more bytes than cold_sandbox."""
    subset = build_osworld_subset()[:4]
    with OSWorldVisorRunner(
        tasks=subset,
        workspace_scale="tiny",
        max_concurrency=1,
        base_dir=tmp_path / "crossover",
    ) as runner:
        report = asyncio.run(
            runner.run_suite(
                agents=[OracleWorkflowAgent()],
                profiles=[SandboxProfile.COLD_SANDBOX, SandboxProfile.WARM_CAS_VISOR],
                warm_visor_cas=False,  # Deliberately cold CAS cache!
            )
        )

    cold = next(c for c in report.cohorts if c.profile == SandboxProfile.COLD_SANDBOX)
    warm_cold_miss = next(c for c in report.cohorts if c.profile == SandboxProfile.WARM_CAS_VISOR)
    assert cold.succeeded_tasks == 4
    assert warm_cold_miss.succeeded_tasks == 4
    # On a cold CAS miss, warm_cas_visor builds the template AND copies mutable seed files,
    # resulting in strictly more bytes copied than cold_sandbox on tiny single-shot tasks.
    assert warm_cold_miss.total_bytes_copied > cold.total_bytes_copied
    assert all(
        r.telemetry.cas_cache_hit is False
        for r in report.results
        if r.profile == SandboxProfile.WARM_CAS_VISOR
    )


@pytest.mark.unit
def test_detached_writable_copies_never_corrupt_cas_template(tmp_path: Path) -> None:
    """Mutating seeded mutable files in one warm_cas_visor rollout never corrupts future runs."""
    task = build_osworld_subset()[0]  # osw_01_filesystem_batch_ops
    with VisorSandboxManager(base_dir=tmp_path / "cas_iso", workspace_scale="tiny") as mgr:
        mgr.warm_cas_pool([task])

        sb1 = mgr.provision(task, SandboxProfile.WARM_CAS_VISOR)
        target = sb1.workspace / "inbox" / "readme.txt"
        original_text = target.read_text(encoding="utf-8")
        # Mutate and delete seeded files inside sandbox 1.
        target.write_text("CORRUPTED-BY-RUN-1\n", encoding="utf-8")
        (sb1.workspace / "inbox" / "app.log").unlink()
        cleanup1, _ = mgr.audit_and_teardown(sb1)
        assert cleanup1.cleanup_clean is True
        assert cleanup1.cas_template_intact is True

        # Provision sandbox 2 from the same warm CAS pool -> seed files must be pristine!
        sb2 = mgr.provision(task, SandboxProfile.WARM_CAS_VISOR)
        assert sb2.cas_cache_hit is True
        assert (sb2.workspace / "inbox" / "readme.txt").read_text(encoding="utf-8") == original_text
        assert (sb2.workspace / "inbox" / "app.log").is_file()
        cleanup2, _ = mgr.audit_and_teardown(sb2)
        assert cleanup2.cleanup_clean is True


@pytest.mark.unit
def test_cas_template_mutation_detected_fail_closed(tmp_path: Path) -> None:
    """Direct tampering of an immutable CAS ro_layer file is flagged by CleanupAuditor."""
    task = build_osworld_subset()[0]
    with VisorSandboxManager(base_dir=tmp_path / "cas_tamper", workspace_scale="tiny") as mgr:
        mgr.warm_cas_pool([task])
        sb = mgr.provision(task, SandboxProfile.WARM_CAS_VISOR)
        assert sb.cas_template_dir is not None
        first_ro_rel = next(iter(sb.cas_ro_manifest))
        ro_file = sb.cas_template_dir / "ro_layer" / first_ro_rel
        os.chmod(ro_file, 0o644)
        ro_file.write_bytes(b"tampered-cas-inode")
        cleanup, _ = mgr.audit_and_teardown(sb)
        assert cleanup.cas_template_intact is False
        assert cleanup.cleanup_clean is False
        assert any(v.startswith("cas_template_mutated:") for v in cleanup.violations)


@pytest.mark.functional
def test_cleanup_auditor_detects_and_reaps_orphan_pids_and_temp_leaks(tmp_path: Path) -> None:
    """AdversarialOrphanLeakerAgent is caught, its orphan PID reaped, and run failed closed."""
    subset = build_osworld_subset()[:2]
    with OSWorldVisorRunner(
        tasks=subset,
        workspace_scale="tiny",
        max_concurrency=1,
        base_dir=tmp_path / "orphan_test",
    ) as runner:
        report = asyncio.run(
            runner.run_suite(
                agents=[AdversarialOrphanLeakerAgent()],
                profiles=[SandboxProfile.COLD_SANDBOX, SandboxProfile.WARM_CAS_VISOR],
                warm_visor_cas=True,
            )
        )

    assert len(report.results) == 4
    for res in report.results:
        assert res.status == TaskStatus.CLEANUP_VIOLATION
        assert res.success is False
        assert res.score == 0.0
        assert res.cleanup.cleanup_clean is False
        assert len(res.cleanup.orphan_pids_detected) >= 1
        assert res.cleanup.orphan_pids_reaped == res.cleanup.orphan_pids_detected
        assert "tmp/leaked_artifact.tmp" in res.cleanup.leaked_temp_files
        assert res.cleanup.workspace_removed is True
        # Confirm every reaped orphan PID is genuinely terminated on the OS!
        for pid in res.cleanup.orphan_pids_reaped:
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)


@pytest.mark.functional
def test_security_auditor_blocks_traversal_and_symlink_escape_fail_closed(tmp_path: Path) -> None:
    """AdversarialSandboxEscapeAgent is rejected fail-closed even when it solves the task."""
    subset = build_osworld_subset()[:2]
    with OSWorldVisorRunner(
        tasks=subset,
        workspace_scale="tiny",
        max_concurrency=1,
        base_dir=tmp_path / "escape_test",
    ) as runner:
        report = asyncio.run(
            runner.run_suite(
                agents=[AdversarialSandboxEscapeAgent()],
                profiles=[SandboxProfile.COLD_SANDBOX, SandboxProfile.WARM_CAS_VISOR],
                warm_visor_cas=True,
            )
        )

    assert len(report.results) == 4
    for res in report.results:
        assert res.status == TaskStatus.SANDBOX_ESCAPE_BLOCKED
        assert res.success is False
        assert res.score == 0.0
        assert res.security.escaped is True
        assert any("escape_write_in_sandbox_root:escaped_outside.txt" in v for v in res.security.violations)
        assert any("symlink_escape:escape_canary_link" in v for v in res.security.violations)
        assert res.cleanup.workspace_removed is True


@pytest.mark.unit
def test_working_subdir_path_traversal_rejected_before_launch(tmp_path: Path) -> None:
    """Attempting to execute a command with working_subdir='../host_guard' fails closed."""
    task = build_osworld_subset()[0]
    with VisorSandboxManager(base_dir=tmp_path / "subdir_escape", workspace_scale="tiny") as mgr:
        sb = mgr.provision(task, SandboxProfile.COLD_SANDBOX)
        res = mgr.exec_in_sandbox(
            sb,
            [sys.executable, "-c", "print('should never run')"],
            working_subdir="../host_guard",
        )
        assert res.escape_rejected is True
        assert res.returncode == 126
        assert "should never run" not in res.stdout
        sec = mgr.audit_security(sb)
        assert sec.escaped is True
        assert any("path_traversal_blocked" in v for v in sec.violations)
        mgr.audit_and_teardown(sb)


@pytest.mark.functional
def test_symlinked_tmpdir_compatibility(tmp_path: Path) -> None:
    """Symlinked base_dir (matching macOS /var/folders -> /private/var/folders) passes cleanly."""
    real_dir = tmp_path / "real_storage"
    real_dir.mkdir()
    symlink_dir = tmp_path / "symlinked_tmp"
    os.symlink(real_dir, symlink_dir)

    subset = build_osworld_subset()[:3]
    with OSWorldVisorRunner(
        tasks=subset,
        workspace_scale="tiny",
        max_concurrency=1,
        base_dir=symlink_dir,
    ) as runner:
        report = asyncio.run(
            runner.run_suite(
                agents=[OracleWorkflowAgent()],
                profiles=[SandboxProfile.COLD_SANDBOX, SandboxProfile.WARM_CAS_VISOR],
                warm_visor_cas=True,
            )
        )

    assert len(report.results) == 6
    for res in report.results:
        assert res.success is True
        assert res.security.escaped is False
        assert res.cleanup.cleanup_clean is True


@pytest.mark.unit
def test_cleanup_auditor_detects_lingering_loopback_socket(tmp_path: Path) -> None:
    """A task leaving an open loopback TCP port recorded in probe_report.json is flagged."""
    task = build_osworld_subset()[6]  # osw_07_local_http_service_check
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    bound_port = sock.getsockname()[1]
    try:
        with VisorSandboxManager(base_dir=tmp_path / "sock_test", workspace_scale="tiny") as mgr:
            sb = mgr.provision(task, SandboxProfile.COLD_SANDBOX)
            (sb.workspace / "web" / "probe_report.json").write_text(
                json.dumps({"port": bound_port, "routes": {}, "server_stopped": False}),
                encoding="utf-8",
            )
            cleanup, _ = mgr.audit_and_teardown(sb)
            assert cleanup.cleanup_clean is False
            assert f"tcp:127.0.0.1:{bound_port}" in cleanup.lingering_sockets
    finally:
        sock.close()


@pytest.mark.functional
def test_env_scrubbing_strips_host_secrets_from_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Host credentials and PYTHONPATH are scrubbed from isolated sandbox subprocesses."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-must-not-leak")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret-must-not-leak")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-must-not-leak")
    monkeypatch.setenv("PYTHONPATH", "/tmp/injected_site")

    task = build_osworld_subset()[0]
    with VisorSandboxManager(base_dir=tmp_path / "env_test", workspace_scale="tiny") as mgr:
        sb = mgr.provision(task, SandboxProfile.COLD_SANDBOX)
        res = mgr.exec_in_sandbox(
            sb,
            [sys.executable, "-c", "import json, os; print(json.dumps(dict(os.environ)))"],
        )
        assert res.returncode == 0
        child_env = json.loads(res.stdout)
        assert "OPENAI_API_KEY" not in child_env
        assert "AWS_SECRET_ACCESS_KEY" not in child_env
        assert "GITHUB_TOKEN" not in child_env
        assert "PYTHONPATH" not in child_env
        assert child_env["HOME"] == str(sb.home_dir)
        assert child_env["TMPDIR"] == str(sb.tmp_dir)
        assert child_env["HANZO_SANDBOX_ID"] == sb.sandbox_id
        mgr.audit_and_teardown(sb)


@pytest.mark.functional
def test_parallel_asyncio_gather_concurrency_zero_contamination(tmp_path: Path) -> None:
    """Running 6 tasks in parallel (concurrency=4) preserves isolation and 100% success."""
    subset = build_osworld_subset()[:6]
    with OSWorldVisorRunner(
        tasks=subset,
        workspace_scale="tiny",
        max_concurrency=4,
        base_dir=tmp_path / "parallel",
    ) as runner:
        report = asyncio.run(
            runner.run_suite(
                agents=[OracleWorkflowAgent()],
                profiles=[SandboxProfile.COLD_SANDBOX, SandboxProfile.WARM_CAS_VISOR],
                warm_visor_cas=True,
            )
        )

    assert len(report.results) == 12
    assert all(r.success is True for r in report.results)
    assert all(r.cleanup.cleanup_clean is True for r in report.results)
    assert all(r.security.escaped is False for r in report.results)
    assert list((tmp_path / "parallel" / "runs").iterdir()) == []


@pytest.mark.unit
def test_external_osworld_json_loader_and_manifest_digest(tmp_path: Path) -> None:
    """External upstream OSWorld Apache-2.0 JSON loader validates schemas & updates digest."""
    ext_dir = tmp_path / "osworld_examples"
    (ext_dir / "os").mkdir(parents=True)
    (ext_dir / "libreoffice_calc").mkdir(parents=True)

    # Index file should be skipped gracefully.
    (ext_dir / "test_all.json").write_text('{"os": ["task-os-1"]}\n', encoding="utf-8")

    (ext_dir / "os" / "task-os-1.json").write_text(
        json.dumps(
            {
                "id": "94d95f96-9699-4208-98ba-3c3119edf9c2",
                "snapshot": "os",
                "instruction": "Find all .log files larger than 1MB and archive them.",
                "related_apps": ["os"],
                "config": [{"type": "execute", "parameters": {"command": ["mkdir", "-p", "/tmp/x"]}}],
                "evaluator": {"func": "check_include_exclude", "result": {"type": "vm_file"}},
            }
        ),
        encoding="utf-8",
    )
    (ext_dir / "libreoffice_calc" / "task-calc-1.json").write_text(
        json.dumps(
            {
                "id": "a1b2c3d4-0000-1111-2222-333344445555",
                "snapshot": "libreoffice_calc",
                "instruction": "Compute SUM column in Calc spreadsheet and save as xlsx.",
                "related_apps": ["libreoffice_calc"],
                "config": [],
                "evaluator": {"func": "compare_table"},
            }
        ),
        encoding="utf-8",
    )

    tooling = probe_desktop_tooling()
    loaded = load_external_osworld_dir(ext_dir, tooling=tooling)
    assert len(loaded) == 2

    by_snap = {t.snapshot: t for t in loaded}
    assert by_snap["os"].requires_gui is False
    assert by_snap["os"].compatible_with_host is True
    assert by_snap["os"].raw_config_steps == 1
    assert by_snap["libreoffice_calc"].requires_gui is True

    # Canonical manifest digest is deterministic and changes when external tasks are added.
    builtin = build_osworld_subset()
    d1 = compute_manifest_digest(builtin)
    d2 = compute_manifest_digest(builtin)
    d3 = compute_manifest_digest(builtin, external_tasks=loaded)
    assert d1 == d2
    assert len(d1) == 64
    assert d3 != d1

    # Invalid JSON missing required field must fail fast.
    bad_dir = tmp_path / "bad_osworld"
    bad_dir.mkdir()
    (bad_dir / "broken.json").write_text('{"id": "x", "instruction": "missing evaluator"}', encoding="utf-8")
    with pytest.raises(ValueError, match="missing required key"):
        load_external_osworld_dir(bad_dir)


@pytest.mark.unit
def test_hanzo_aiconfig_cost_tracker_strips_api_key_and_checks_budget() -> None:
    """CostTracker routes via Hanzo AIConfig/llm_adapters, strips secrets, and checks budget."""
    cfg = AIConfig(
        model="openai/gpt-4o",
        api_key="sk-secret-do-not-serialize",
        temperature=0.0,
        max_tokens=2048,
        max_cost_per_call=0.005,
    )
    tracker = CostTracker(ai_config=cfg)
    small = tracker.record(prompt_tokens=1000, completion_tokens=100, cached_tokens=500, calls=1)
    assert small.provider == "openai"
    assert small.cost_usd > 0.0
    assert small.budget_exceeded is False
    assert "api_key" not in small.llm_params_snapshot

    large = tracker.record(prompt_tokens=5000, completion_tokens=2000, cached_tokens=0, calls=2)
    assert large.cost_usd > 0.005
    assert large.budget_exceeded is True


@pytest.mark.functional
def test_cli_entrypoint_writes_valid_json_and_markdown_score_table(tmp_path: Path) -> None:
    """CLI writes reproducible raw_results.json and score_table.md."""
    out_dir = tmp_path / "cli_out"
    rc = cli_main(
        [
            "--profiles",
            "cold_sandbox",
            "warm_cas_visor",
            "--agents",
            "oracle",
            "noop",
            "--workspace-scale",
            "tiny",
            "--warm-visor-cas",
            "--concurrency",
            "1",
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    raw_path = out_dir / "raw_results.json"
    table_path = out_dir / "score_table.md"
    assert raw_path.is_file()
    assert table_path.is_file()

    data = json.loads(raw_path.read_text(encoding="utf-8"))
    assert data["workspace_scale"] == "tiny"
    assert data["warm_visor_cas"] is True
    assert len(data["cohorts"]) == 4
    assert len(data["results"]) == 40
    assert "desktop_tooling" in data
    assert data["desktop_tooling"]["execution_backend"] == "isolated_subprocess_posix"

    md = table_path.read_text(encoding="utf-8")
    assert "| Agent | Profile |" in md
    assert "`oracle_agent`" in md
    assert "`noop_agent`" in md
