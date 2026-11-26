"""Comprehensive test suite for the OSWorld computer-workflow Visor benchmark (HIP-1348)."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

import pytest

from benchmarks.osworld_visor import (
    AdversarialOrphanLeakerAgent,
    AdversarialSandboxEscapeAgent,
    CostTracker,
    NoOpBaselineAgent,
    OSWorldTaskSpec,
    OSWorldVisorRunner,
    OracleWorkflowAgent,
    PartialCapabilityAgent,
    SandboxProfile,
    TaskStatus,
    VisorSandboxManager,
    WorkflowAgent,
    WorkflowStep,
    build_osworld_subset,
    compute_manifest_digest,
    evaluate_task,
    format_score_table,
    load_external_osworld_dir,
    probe_desktop_tooling,
)
from benchmarks.osworld_visor import runner as runner_mod
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
    # 1. Default runner (enforces Linux mount-namespace confinement when supported).
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
        assert any("symlink_escape:escape_canary_link" in v for v in res.security.violations)
        if not res.security.fs_confinement_enforced:
            assert any(
                "escape_write_in_sandbox_root:escaped_outside.txt" in v
                for v in res.security.violations
            )
            assert any(
                "escape_write_in_runs_root:escaped_runs_root.txt" in v
                for v in res.security.violations
            )
        assert res.cleanup.workspace_removed is True
    assert list((tmp_path / "escape_test" / "runs").iterdir()) == []

    # 2. Explicit unenforced POSIX mode: post-hoc auditor catches both ../ and ../../ writes
    #    and explicitly reports fs_confinement_enforced=False.
    with OSWorldVisorRunner(
        tasks=subset[:1],
        workspace_scale="tiny",
        max_concurrency=1,
        base_dir=tmp_path / "escape_unenforced",
        enable_fs_confinement=False,
    ) as runner:
        report_u = asyncio.run(
            runner.run_suite(
                agents=[AdversarialSandboxEscapeAgent()],
                profiles=[SandboxProfile.COLD_SANDBOX],
                warm_visor_cas=False,
            )
        )
    res_u = report_u.results[0]
    assert res_u.status == TaskStatus.SANDBOX_ESCAPE_BLOCKED
    assert res_u.security.fs_confinement_enforced is False
    assert res_u.security.fs_confinement == "unenforced_posix_audit_only"
    assert any("escape_write_in_sandbox_root:escaped_outside.txt" in v for v in res_u.security.violations)
    assert any("escape_write_in_runs_root:escaped_runs_root.txt" in v for v in res_u.security.violations)
    assert any("symlink_escape:escape_canary_link" in v for v in res_u.security.violations)
    assert list((tmp_path / "escape_unenforced" / "runs").iterdir()) == []


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


# ---------------------------------------------------------------------------
# Regression Tests for Codex Review Findings (PR #123)
# ---------------------------------------------------------------------------


@pytest.mark.functional
def test_evaluator_scrubs_host_secrets_for_agent_controlled_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding 1 (P1): Evaluators executing agent-controlled calc.py or validating task 10 scrub host secrets."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-eval-secret-must-not-leak")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-eval-secret-must-not-leak")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-eval-secret-must-not-leak")

    subset = build_osworld_subset()
    task4 = next(t for t in subset if t.task_id == "osw_04_cli_git_automation")
    task10 = next(t for t in subset if t.task_id == "osw_10_json_yaml_pipeline_patch")

    with VisorSandboxManager(base_dir=tmp_path / "eval_env_sec", workspace_scale="tiny") as mgr:
        # Task 4: Agent-controlled calc.py tries to exfiltrate host secrets on import.
        sb4 = mgr.provision(task4, SandboxProfile.COLD_SANDBOX)
        (sb4.workspace / "repo" / "calc.py").write_text(
            "import json, os, pathlib\n"
            "pathlib.Path('leaked_env.json').write_text(json.dumps(dict(os.environ)))\n"
            "def compute_ratio(a, b):\n"
            "    return 0.0 if b == 0 else round(a / b, 4)\n",
            encoding="utf-8",
        )
        evaluate_task(task4, sb4.workspace, env=mgr.build_sandbox_env(sb4))
        leaked4 = json.loads((sb4.workspace / "repo" / "leaked_env.json").read_text(encoding="utf-8"))
        assert "OPENAI_API_KEY" not in leaked4
        assert "AWS_SECRET_ACCESS_KEY" not in leaked4
        assert "GITHUB_TOKEN" not in leaked4

        # Direct call without explicit env= must also use scrubbed env policy.
        (sb4.workspace / "repo" / "leaked_env.json").unlink()
        evaluate_task(task4, sb4.workspace)
        leaked4_direct = json.loads((sb4.workspace / "repo" / "leaked_env.json").read_text(encoding="utf-8"))
        assert "OPENAI_API_KEY" not in leaked4_direct
        assert "AWS_SECRET_ACCESS_KEY" not in leaked4_direct
        mgr.audit_and_teardown(sb4)

        # Task 10: Agent replaces ci/validator.py with credential exfiltration probe;
        # trusted evaluator never executes agent-controlled validator.py and scrubs env.
        sb10 = mgr.provision(task10, SandboxProfile.COLD_SANDBOX)
        (sb10.workspace / "ci" / "validator.py").write_text(
            "import os, pathlib\n"
            "pathlib.Path('ci/leaked_secret.txt').write_text(os.environ.get('OPENAI_API_KEY', 'MISSING'))\n"
            "print('validator-ok')\n",
            encoding="utf-8",
        )
        evaluate_task(task10, sb10.workspace, env=mgr.build_sandbox_env(sb10))
        assert not (sb10.workspace / "ci" / "leaked_secret.txt").exists()
        mgr.audit_and_teardown(sb10)


@pytest.mark.functional
def test_deep_traversal_write_to_runs_root_contained_and_blocked(tmp_path: Path) -> None:
    """Finding 2 (P1): ../../escaped.txt is kernel-blocked on Linux and audited/rejected when unenforced."""

    class _DeepTraversalAgent(WorkflowAgent):
        name = "deep_traversal_agent"

        def plan_steps(self, task: OSWorldTaskSpec) -> Tuple[List[WorkflowStep], int, int, int]:
            escape_step = WorkflowStep(
                name="write_runs_root",
                description="Write ../../escaped.txt into runs_root",
                command=[
                    sys.executable,
                    "-c",
                    "import pathlib; pathlib.Path('../../escaped.txt').write_text('escaped\\n', encoding='utf-8')",
                ],
                timeout_s=10.0,
            )
            return list(task.oracle_steps) + [escape_step], 500, 100, 0

    task1 = build_osworld_subset()[0]

    # 1. With confinement enabled (Linux mount namespace): write is blocked by kernel (EROFS),
    #    escaped.txt never lands in runs_root, and fs_confinement_enforced=True is reported.
    base_confined = tmp_path / "confined"
    with OSWorldVisorRunner(
        tasks=[task1],
        workspace_scale="tiny",
        max_concurrency=1,
        base_dir=base_confined,
        enable_fs_confinement=True,
    ) as runner:
        if runner.sandbox_manager.fs_confinement_enforced:
            report_c = asyncio.run(
                runner.run_suite(
                    agents=[_DeepTraversalAgent()],
                    profiles=[SandboxProfile.COLD_SANDBOX],
                    warm_visor_cas=False,
                )
            )
            res_c = report_c.results[0]
            assert res_c.security.fs_confinement_enforced is True
            assert res_c.security.fs_confinement == "linux_mount_namespace"
            # The escape write step failed with non-zero exit (EROFS) so steps_succeeded < step_count.
            assert res_c.steps_succeeded < res_c.step_count
            assert not (base_confined / "runs" / "escaped.txt").exists()
            assert list((base_confined / "runs").iterdir()) == []

    # 2. With confinement disabled (unenforced POSIX host): rollout MUST fail closed as
    #    SANDBOX_ESCAPE_BLOCKED, explicitly report fs_confinement_enforced=False, and clean up.
    base_unenforced = tmp_path / "unenforced"
    with OSWorldVisorRunner(
        tasks=[task1],
        workspace_scale="tiny",
        max_concurrency=1,
        base_dir=base_unenforced,
        enable_fs_confinement=False,
    ) as runner:
        report_u = asyncio.run(
            runner.run_suite(
                agents=[_DeepTraversalAgent()],
                profiles=[SandboxProfile.COLD_SANDBOX],
                warm_visor_cas=False,
            )
        )
    res_u = report_u.results[0]
    assert res_u.status == TaskStatus.SANDBOX_ESCAPE_BLOCKED
    assert res_u.success is False
    assert res_u.score == 0.0
    assert res_u.security.escaped is True
    assert res_u.security.fs_confinement_enforced is False
    assert any("escape_write_in_runs_root:escaped.txt" in v for v in res_u.security.violations)
    assert not (base_unenforced / "runs" / "escaped.txt").exists()
    assert list((base_unenforced / "runs").iterdir()) == []


@pytest.mark.unit
def test_reap_orphan_pids_never_steals_concurrent_sandbox_child_exit_status(
    tmp_path: Path,
) -> None:
    """Finding 3 (P1): Reaping one sandbox never steals another concurrent sandbox's zombie child."""
    task1 = build_osworld_subset()[0]
    with VisorSandboxManager(
        base_dir=tmp_path / "reap_scope",
        workspace_scale="tiny",
        enable_fs_confinement=False,
    ) as mgr:
        sb_a = mgr.provision(task1, SandboxProfile.COLD_SANDBOX)
        sb_b = mgr.provision(task1, SandboxProfile.COLD_SANDBOX)

        # Spawn a direct child owned by sb_b that exits quickly with exit code 7.
        child_b = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.exit(7)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        with mgr._state_lock:
            mgr._active_pids[child_b.pid] = sb_b.sandbox_id
        try:
            # Wait briefly so child_b exits and sits as an unreaped zombie ready for os.wait4.
            time.sleep(0.15)
            # Reap sb_a while child_b is an unreaped zombie owned by sb_b.
            mgr._reap_orphan_pids(sb_a, [])
            # sb_b's waiter MUST still reap child_b with exact exit code 7 and valid rusage!
            waited_pid, status, rusage = os.wait4(child_b.pid, 0)
            assert waited_pid == child_b.pid
            assert os.waitstatus_to_exitcode(status) == 7
            assert rusage is not None
            child_b.returncode = 7
        finally:
            with mgr._state_lock:
                mgr._active_pids.pop(child_b.pid, None)
            if child_b.stdout:
                child_b.stdout.close()
            if child_b.stderr:
                child_b.stderr.close()
            mgr.audit_and_teardown(sb_a)
            mgr.audit_and_teardown(sb_b)


@pytest.mark.functional
def test_malformed_evaluator_artifacts_and_exceptions_do_not_abort_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding 4 (P2): Malformed agent outputs (e.g. manifest.json=[]) or evaluator exceptions fail per-task."""

    class _MalformedManifestAgent(WorkflowAgent):
        name = "malformed_manifest_agent"

        def plan_steps(self, task: OSWorldTaskSpec) -> Tuple[List[WorkflowStep], int, int, int]:
            if task.task_id == "osw_01_filesystem_batch_ops":
                bad_step = WorkflowStep(
                    name="write_list_manifest",
                    description="Write manifest.json as JSON array instead of object",
                    command=[
                        sys.executable,
                        "-c",
                        "import pathlib; pathlib.Path('manifest.json').write_text('[]', encoding='utf-8')",
                    ],
                    timeout_s=10.0,
                )
                return list(task.oracle_steps) + [bad_step], 400, 100, 0
            return list(task.oracle_steps), 400, 100, 0

    subset = build_osworld_subset()[:2]
    with OSWorldVisorRunner(
        tasks=subset,
        workspace_scale="tiny",
        max_concurrency=2,
        base_dir=tmp_path / "malformed_suite",
    ) as runner:
        report = asyncio.run(
            runner.run_suite(
                agents=[_MalformedManifestAgent()],
                profiles=[SandboxProfile.COLD_SANDBOX],
                warm_visor_cas=False,
            )
        )
    by_id = {r.task_id: r for r in report.results}
    assert by_id["osw_01_filesystem_batch_ops"].status == TaskStatus.FAILED
    assert by_id["osw_01_filesystem_batch_ops"].success is False
    assert by_id["osw_01_filesystem_batch_ops"].evaluator_checks["manifest_verified"] is False
    assert by_id["osw_02_spreadsheet_csv_transform"].status == TaskStatus.SUCCESS

    # Even if an evaluator raises an unexpected exception, the rollout boundary records
    # TaskStatus.ERROR without aborting asyncio.gather or discarding sibling tasks.
    real_eval = runner_mod.evaluate_task

    def _flaky_eval(task: OSWorldTaskSpec, ws: Path, *, env=None):
        if task.task_id == "osw_01_filesystem_batch_ops":
            raise RuntimeError("simulated-evaluator-crash")
        return real_eval(task, ws, env=env)

    monkeypatch.setattr(runner_mod, "evaluate_task", _flaky_eval)
    with OSWorldVisorRunner(
        tasks=subset,
        workspace_scale="tiny",
        max_concurrency=2,
        base_dir=tmp_path / "eval_exc_suite",
    ) as runner:
        report2 = asyncio.run(
            runner.run_suite(
                agents=[OracleWorkflowAgent()],
                profiles=[SandboxProfile.COLD_SANDBOX],
                warm_visor_cas=False,
            )
        )
    by_id2 = {r.task_id: r for r in report2.results}
    assert by_id2["osw_01_filesystem_batch_ops"].status == TaskStatus.ERROR
    assert by_id2["osw_01_filesystem_batch_ops"].success is False
    assert by_id2["osw_01_filesystem_batch_ops"].score == 0.0
    assert "evaluator_error:RuntimeError:simulated-evaluator-crash" in by_id2["osw_01_filesystem_batch_ops"].evaluator_details
    assert by_id2["osw_01_filesystem_batch_ops"].cleanup.workspace_removed is True
    assert by_id2["osw_02_spreadsheet_csv_transform"].status == TaskStatus.SUCCESS


@pytest.mark.functional
def test_task10_tampered_validator_cannot_bypass_independent_checks(tmp_path: Path) -> None:
    """Finding 5 (P2): Replacing ci/validator.py with print('validator-ok') cannot pass task 10."""

    class _CheatTask10Agent(WorkflowAgent):
        name = "cheat_task10_agent"

        def plan_steps(self, task: OSWorldTaskSpec) -> Tuple[List[WorkflowStep], int, int, int]:
            cheat_code = (
                "import json, pathlib\n"
                "# Replace writable validator.py with a no-op print\n"
                "pathlib.Path('ci/validator.py').write_text(\"print('validator-ok')\\n\", encoding='utf-8')\n"
                "# Only patch lint image + timeouts; leave test/package and all env files legacy!\n"
                "wf_p = pathlib.Path('ci/workflow.json')\n"
                "wf = json.loads(wf_p.read_text(encoding='utf-8'))\n"
                "wf['jobs']['lint']['image'] = 'hanzo/runner:v2-hardened'\n"
                "wf['jobs']['lint']['timeout_minutes'] = 15\n"
                "wf['jobs']['package']['timeout_minutes'] = 15\n"
                "wf_p.write_text(json.dumps(wf, indent=2), encoding='utf-8')\n"
                "diff = {\n"
                "    'ci/workflow.json': ['jobs.lint.image'],\n"
                "    'ci/environments/dev.json': ['image'],\n"
                "    'ci/environments/staging.json': ['image'],\n"
                "    'ci/environments/prod.json': ['image', 'read_only_root'],\n"
                "}\n"
                "pathlib.Path('ci/migration_diff.json').write_text(json.dumps(diff, indent=2), encoding='utf-8')\n"
            )
            step = WorkflowStep(
                name="tamper_validator",
                description="Tamper validator.py while leaving legacy configs untouched",
                command=[sys.executable, "-c", cheat_code],
                timeout_s=10.0,
            )
            return [step], 400, 120, 0

    task10 = next(t for t in build_osworld_subset() if t.task_id == "osw_10_json_yaml_pipeline_patch")
    with OSWorldVisorRunner(
        tasks=[task10],
        workspace_scale="tiny",
        max_concurrency=1,
        base_dir=tmp_path / "cheat_task10",
    ) as runner:
        report = asyncio.run(
            runner.run_suite(
                agents=[_CheatTask10Agent()],
                profiles=[SandboxProfile.COLD_SANDBOX],
                warm_visor_cas=False,
            )
        )
    res = report.results[0]
    assert res.status == TaskStatus.FAILED
    assert res.success is False
    assert res.score < 1.0
    assert res.evaluator_checks["validator_passes"] is False
    assert res.evaluator_checks["workflow_timeouts_and_tags"] is False
    assert res.evaluator_checks["environments_migrated"] is False


@pytest.mark.functional
def test_non_procfs_fallback_detects_and_reaps_recorded_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding 6 (P2): Disabling /proc still detects and kills lingering recorded process groups."""
    task1 = build_osworld_subset()[0]
    spawn_code = (
        "import subprocess, sys\n"
        "p = subprocess.Popen(\n"
        "    [sys.executable, '-c', 'import time; time.sleep(30)'],\n"
        "    stdin=subprocess.DEVNULL,\n"
        "    stdout=subprocess.DEVNULL,\n"
        "    stderr=subprocess.DEVNULL,\n"
        ")\n"
        "print(p.pid)\n"
    )

    with VisorSandboxManager(
        base_dir=tmp_path / "no_procfs_ps",
        workspace_scale="tiny",
        enable_fs_confinement=False,
    ) as mgr:
        mgr._proc_dir = tmp_path / "nonexistent_proc"

        # Case 1: procfs unavailable, POSIX ps + killpg fallback available.
        sb1 = mgr.provision(task1, SandboxProfile.COLD_SANDBOX)
        res1 = mgr.exec_in_sandbox(sb1, [sys.executable, "-c", spawn_code])
        sleeper1 = int(res1.stdout.strip())
        os.kill(sleeper1, 0)  # Alive before teardown
        cleanup1, _ = mgr.audit_and_teardown(sb1)
        assert cleanup1.cleanup_clean is False
        assert cleanup1.workspace_removed is True
        with pytest.raises(ProcessLookupError):
            os.kill(sleeper1, 0)

        # Case 2: BOTH procfs AND ps unavailable — os.killpg liveness fallback still reaps!
        sb2 = mgr.provision(task1, SandboxProfile.COLD_SANDBOX)
        res2 = mgr.exec_in_sandbox(sb2, [sys.executable, "-c", spawn_code])
        sleeper2 = int(res2.stdout.strip())
        os.kill(sleeper2, 0)  # Alive before teardown
        monkeypatch.setattr(shutil, "which", lambda *args, **kwargs: None)
        cleanup2, _ = mgr.audit_and_teardown(sb2)
        assert cleanup2.cleanup_clean is False
        assert any("orphan_pgids_detected" in v for v in cleanup2.violations)
        assert cleanup2.workspace_removed is True
        with pytest.raises(ProcessLookupError):
            os.kill(sleeper2, 0)

