"""Comprehensive tests for the SWE-bench Pro V2 Visor subset runner (HIP-1348).

Covers:
- End-to-end real ``git apply`` + ``pytest`` execution (oracle, partial, noop, broken diff)
- Red-team anti-leakage tests (git history/reflog/objects purge, env & credential scope blocks)
- Red-team anti-tampering tests (test-file edits, symlink escapes, sitecustomize/pytest shims)
- Real OS filesystem I/O telemetry & plain NVMe vs Visor CoW comparison
- Hanzo SDK ``AIConfig`` & ``llm_adapters`` cost tracking and budget enforcement
- External SWE-bench Pro V2 manifest validation (rejects legacy V1)
- Concurrent ``asyncio.gather`` runs with zero cross-task workspace contamination
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Dict

import pytest

from benchmarks.swebench_pro import (
    AdversarialGitLeakAgent,
    AdversarialShimInjectorAgent,
    AdversarialSymlinkEscapeAgent,
    AdversarialTestEditorAgent,
    AgentInvocationResult,
    CodingAgent,
    CostTracker,
    NoOpBaselineAgent,
    OraclePatchAgent,
    PartialCapabilityAgent,
    StorageProfile,
    SWEBenchProRunner,
    SWEBenchTask,
    TaskStatus,
    ViolationType,
    build_cleanroom_v2_subset,
    format_score_table,
    load_external_manifest,
    read_proc_self_io,
    select_subset,
)
from hanzo_agents.types import AIConfig


class _MalformedDiffAgent(CodingAgent):
    name = "broken_diff_agent"

    async def solve(
        self,
        task: SWEBenchTask,
        workspace: Path,
        env: Dict[str, str],
        ai_config: AIConfig,
    ) -> AgentInvocationResult:
        bad_diff = (
            "diff --git a/nonexistent.py b/nonexistent.py\n"
            "--- a/nonexistent.py\n"
            "+++ b/nonexistent.py\n"
            "@@ -1,2 +1,2 @@\n"
            "-does_not_exist()\n"
            "+still_broken()\n"
        )
        return AgentInvocationResult(patch_diff=bad_diff, prompt_tokens=100, completion_tokens=40)


class _TestDeleterAgent(CodingAgent):
    name = "redteam_test_deleter"

    async def solve(
        self,
        task: SWEBenchTask,
        workspace: Path,
        env: Dict[str, str],
        ai_config: AIConfig,
    ) -> AgentInvocationResult:
        for rel in task.protected_test_paths:
            p = workspace / rel
            if p.exists():
                try:
                    os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)
                except OSError:
                    pass
                p.unlink()
        return AgentInvocationResult(patch_diff="", prompt_tokens=50, completion_tokens=10)


class _CanaryIsolationAgent(CodingAgent):
    """Writes a unique canary file and verifies no sibling task canary exists."""

    name = "canary_isolation_agent"

    def __init__(self) -> None:
        self.oracle = OraclePatchAgent()
        self.cross_task_leaks: list[str] = []

    async def solve(
        self,
        task: SWEBenchTask,
        workspace: Path,
        env: Dict[str, str],
        ai_config: AIConfig,
    ) -> AgentInvocationResult:
        existing_canaries = list(workspace.glob(".canary_*"))
        if existing_canaries:
            self.cross_task_leaks.extend(str(p) for p in existing_canaries)
        canary = workspace / f".canary_{task.instance_id}"
        canary.write_text(task.instance_id, encoding="utf-8")
        await asyncio.sleep(0.02)
        others = [p.name for p in workspace.glob(".canary_*") if p != canary]
        if others:
            self.cross_task_leaks.extend(others)
        return await self.oracle.solve(task, workspace, env, ai_config)


# ---------------------------------------------------------------------------
# 1. End-to-End Task Execution & Resolution Correctness
# ---------------------------------------------------------------------------


@pytest.mark.functional
@pytest.mark.asyncio
async def test_oracle_resolves_all_tasks_on_both_profiles() -> None:
    tasks = build_cleanroom_v2_subset()
    assert len(tasks) == 9

    with SWEBenchProRunner(repo_scale="tiny", max_concurrency=2) as runner:
        report = await runner.run_suite(
            tasks=tasks,
            agents=[OraclePatchAgent()],
            profiles=(StorageProfile.PLAIN_NVME, StorageProfile.VISOR_ISOLATED),
        )

    assert len(report.results) == 18
    for r in report.results:
        assert r.resolved is True, f"{r.instance_id} failed on {r.storage_profile}: {r.test_output_tail}"
        assert r.status == TaskStatus.RESOLVED
        assert r.tampered is False
        assert r.leakage_blocked is False
        assert r.fail_to_pass_passed == r.fail_to_pass_total > 0
        assert r.pass_to_pass_passed == r.pass_to_pass_total > 0
        assert r.integrity.clean is True
        assert r.wall_time_ms > 0.0


@pytest.mark.functional
@pytest.mark.asyncio
async def test_noop_fails_all_f2p_and_preserves_p2p() -> None:
    tasks = build_cleanroom_v2_subset()
    with SWEBenchProRunner(repo_scale="tiny", max_concurrency=2) as runner:
        report = await runner.run_suite(
            tasks=tasks,
            agents=[NoOpBaselineAgent()],
            profiles=(StorageProfile.PLAIN_NVME,),
        )

    for r in report.results:
        assert r.resolved is False
        assert r.status == TaskStatus.UNRESOLVED
        assert r.fail_to_pass_passed < r.fail_to_pass_total
        assert r.pass_to_pass_passed == r.pass_to_pass_total


@pytest.mark.functional
@pytest.mark.asyncio
async def test_partial_agent_resolves_subset_and_fails_hard_tasks() -> None:
    tasks = build_cleanroom_v2_subset()
    with SWEBenchProRunner(repo_scale="tiny", max_concurrency=2) as runner:
        report = await runner.run_suite(
            tasks=tasks,
            agents=[PartialCapabilityAgent()],
            profiles=(StorageProfile.VISOR_ISOLATED,),
        )

    resolved_count = sum(1 for r in report.results if r.resolved)
    assert resolved_count == 6
    assert len(report.results) - resolved_count == 3
    for r in report.results:
        assert r.tampered is False
        assert r.cost.cost_usd > 0.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_malformed_patch_diff_rejected_cleanly() -> None:
    task = build_cleanroom_v2_subset()[0]
    with SWEBenchProRunner(repo_scale="tiny") as runner:
        res = await runner.run_task(task, _MalformedDiffAgent(), StorageProfile.PLAIN_NVME)

    assert res.resolved is False
    assert res.tampered is False
    assert res.status == TaskStatus.UNRESOLVED
    assert res.error and "git apply --check failed" in res.error


# ---------------------------------------------------------------------------
# 2. Red-Team Anti-Leakage Tests
# ---------------------------------------------------------------------------


@pytest.mark.functional
@pytest.mark.asyncio
async def test_git_history_reflog_remote_and_objects_physically_purged() -> None:
    task = build_cleanroom_v2_subset()[0]
    with SWEBenchProRunner(repo_scale="tiny") as runner:
        sb = runner.sandbox_manager.provision(
            task, StorageProfile.PLAIN_NVME, scale=runner.scale
        )
        try:
            assert sb.integrity_report.pre_audit_clean is True
            assert "origin" in sb.integrity_report.stripped_remotes
            assert "v-gold-fix" in sb.integrity_report.stripped_tags
            assert sb.integrity_report.stripped_reflogs is True
            assert sb.integrity_report.pruned_objects is True
            assert "gold_patch.diff" in sb.integrity_report.removed_artifacts

            # Verify gold_commit_sha is physically gone from .git/objects
            cat = subprocess.run(
                ["git", "cat-file", "-e", sb.gold_commit_sha],
                cwd=str(sb.workspace),
                capture_output=True,
                text=True,
            )
            assert cat.returncode != 0
            assert not (sb.workspace / "gold_patch.diff").exists()
        finally:
            sb.cleanup()


@pytest.mark.functional
@pytest.mark.asyncio
async def test_redteam_git_leak_agent_finds_zero_secrets_on_both_profiles() -> None:
    tasks = build_cleanroom_v2_subset()[:3]
    with SWEBenchProRunner(repo_scale="tiny", max_concurrency=1) as runner:
        report = await runner.run_suite(
            tasks=tasks,
            agents=[AdversarialGitLeakAgent()],
            profiles=(StorageProfile.PLAIN_NVME, StorageProfile.VISOR_ISOLATED),
        )

    assert len(report.results) == 6
    for r in report.results:
        assert r.resolved is False
        assert r.leakage_blocked is False
        assert r.integrity.pre_audit_clean is True
        assert "leak_findings=0" in r.test_output_tail or r.status == TaskStatus.UNRESOLVED


@pytest.mark.functional
@pytest.mark.asyncio
async def test_unscrubbed_workspace_fails_closed_with_leakage_blocked() -> None:
    task = build_cleanroom_v2_subset()[0]
    with SWEBenchProRunner(repo_scale="tiny") as runner:
        res = await runner.run_task(
            task,
            OraclePatchAgent(),
            StorageProfile.PLAIN_NVME,
            skip_scrub=True,
        )

    assert res.resolved is False
    assert res.leakage_blocked is True
    assert res.status == TaskStatus.LEAKAGE_BLOCKED
    codes = {v.code for v in res.integrity.violations}
    assert ViolationType.GIT_REMOTE_PRESENT in codes
    assert ViolationType.GIT_TAG_PRESENT in codes
    assert ViolationType.GIT_GOLD_OBJECT_LEAK in codes
    assert ViolationType.LEAK_ARTIFACT_PRESENT in codes


@pytest.mark.functional
@pytest.mark.asyncio
async def test_leaked_env_var_and_disallowed_credential_scope_blocked() -> None:
    task = build_cleanroom_v2_subset()[0]
    with SWEBenchProRunner(repo_scale="tiny") as runner:
        res_env = await runner.run_task(
            task,
            OraclePatchAgent(),
            StorageProfile.VISOR_ISOLATED,
            extra_env={"GOLD_PATCH": "leaked", "SAFE_VAR": task.gold_marker},
        )
        res_scope = await runner.run_task(
            task,
            OraclePatchAgent(),
            StorageProfile.VISOR_ISOLATED,
            token_scopes=("ai:inference", "s3:admin"),
        )

    assert res_env.leakage_blocked is True
    assert res_env.resolved is False
    assert ViolationType.ENV_LEAK_VAR in {v.code for v in res_env.integrity.violations}

    assert res_scope.leakage_blocked is True
    assert res_scope.resolved is False
    assert ViolationType.CREDENTIAL_SCOPE_VIOLATION in {
        v.code for v in res_scope.integrity.violations
    }


# ---------------------------------------------------------------------------
# 3. Red-Team Anti-Tampering Tests
# ---------------------------------------------------------------------------


@pytest.mark.functional
@pytest.mark.asyncio
async def test_redteam_test_editor_detected_and_rejected() -> None:
    task = build_cleanroom_v2_subset()[0]
    with SWEBenchProRunner(repo_scale="tiny") as runner:
        for profile in (StorageProfile.PLAIN_NVME, StorageProfile.VISOR_ISOLATED):
            res = await runner.run_task(task, AdversarialTestEditorAgent(), profile)
            assert res.tampered is True
            assert res.resolved is False
            assert res.status == TaskStatus.TAMPERED
            codes = {v.code for v in res.integrity.violations}
            assert ViolationType.TEST_FILE_MODIFIED in codes
            assert ViolationType.PROTECTED_PATH_IN_DIFF in codes


@pytest.mark.functional
@pytest.mark.asyncio
async def test_redteam_symlink_escape_rejected_even_with_passing_patch() -> None:
    task = build_cleanroom_v2_subset()[1]
    with SWEBenchProRunner(repo_scale="tiny") as runner:
        res = await runner.run_task(
            task, AdversarialSymlinkEscapeAgent(), StorageProfile.VISOR_ISOLATED
        )

    assert res.tampered is True
    assert res.resolved is False
    assert res.status == TaskStatus.TAMPERED
    assert ViolationType.SYMLINK_ESCAPE in {v.code for v in res.integrity.violations}


@pytest.mark.functional
@pytest.mark.asyncio
async def test_redteam_shim_injector_detected_and_rejected() -> None:
    task = build_cleanroom_v2_subset()[2]
    with SWEBenchProRunner(repo_scale="tiny") as runner:
        res = await runner.run_task(
            task, AdversarialShimInjectorAgent(), StorageProfile.PLAIN_NVME
        )

    assert res.tampered is True
    assert res.resolved is False
    assert res.status == TaskStatus.TAMPERED
    codes = {v.code for v in res.integrity.violations}
    assert ViolationType.SITECUSTOMIZE_INJECTION in codes
    assert ViolationType.TEST_RUNNER_SHIM in codes
    assert ViolationType.ENV_TAMPER_PYTHONPATH in codes


@pytest.mark.functional
@pytest.mark.asyncio
async def test_deleted_protected_test_file_detected() -> None:
    task = build_cleanroom_v2_subset()[0]
    with SWEBenchProRunner(repo_scale="tiny") as runner:
        res = await runner.run_task(task, _TestDeleterAgent(), StorageProfile.VISOR_ISOLATED)

    assert res.tampered is True
    assert res.resolved is False
    assert ViolationType.TEST_FILE_MISSING in {v.code for v in res.integrity.violations}


@pytest.mark.functional
def test_visor_isolated_protects_tests_readonly_and_isolates_cas_image() -> None:
    task = build_cleanroom_v2_subset()[0]
    with SWEBenchProRunner(repo_scale="tiny") as runner:
        sb1 = runner.sandbox_manager.provision(
            task, StorageProfile.VISOR_ISOLATED, scale=runner.scale
        )
        try:
            # Protected test file must be read-only (no write bits set)
            test_path = sb1.workspace / task.protected_test_paths[0]
            mode = stat.S_IMODE(test_path.stat().st_mode)
            assert (mode & stat.S_IWUSR) == 0

            # Mutating a mutable source file in sb1 must NOT mutate CAS or sb2
            src_rel = task.expected_touched_files[0]
            (sb1.workspace / src_rel).write_text("# mutated in sb1\n", encoding="utf-8")

            sb2 = runner.sandbox_manager.provision(
                task, StorageProfile.VISOR_ISOLATED, scale=runner.scale
            )
            try:
                assert sb2.cas_cache_hit is True
                assert "# mutated in sb1" not in (sb2.workspace / src_rel).read_text(
                    encoding="utf-8"
                )
            finally:
                sb2.cleanup()
        finally:
            sb1.cleanup()


# ---------------------------------------------------------------------------
# 4. Real Filesystem I/O, Timing, and Hanzo AIConfig Cost Telemetry
# ---------------------------------------------------------------------------


@pytest.mark.functional
@pytest.mark.asyncio
async def test_real_filesystem_io_telemetry_and_visor_cow_savings() -> None:
    tasks = build_cleanroom_v2_subset()[:3]
    with SWEBenchProRunner(repo_scale="medium", max_concurrency=1) as runner:
        report = await runner.run_suite(
            tasks=tasks,
            agents=[OraclePatchAgent()],
            profiles=(StorageProfile.PLAIN_NVME, StorageProfile.VISOR_ISOLATED),
            warm_visor_cas=True,
        )

    by_profile = {a.storage_profile: a for a in report.aggregates}
    nvme = by_profile[StorageProfile.PLAIN_NVME]
    visor = by_profile[StorageProfile.VISOR_ISOLATED]

    assert nvme.resolved_tasks == 3
    assert visor.resolved_tasks == 3
    assert visor.total_hardlinked_files >= 3 * 240
    assert nvme.total_hardlinked_files == 0
    # Warm Visor CoW must copy substantially fewer bytes than plain NVMe
    assert visor.total_workspace_bytes_copied * 5 < nvme.total_workspace_bytes_copied
    assert visor.mean_setup_ms < nvme.mean_setup_ms

    if read_proc_self_io() is not None:
        assert nvme.total_rchar_bytes > 0
        assert nvme.total_wchar_bytes > 0
        assert nvme.total_syscr_ops > 0
        assert nvme.total_syscw_ops > 0
        assert visor.total_wchar_bytes < nvme.total_wchar_bytes


@pytest.mark.unit
def test_hanzo_aiconfig_and_llm_adapters_cost_tracking() -> None:
    cfg = AIConfig(
        model="openai/gpt-4o",
        temperature=0.0,
        max_tokens=2048,
        api_key="sk-test-secret-should-not-leak",
        max_cost_per_call=0.01,
    )
    tracker = CostTracker(ai_config=cfg)
    params = tracker.build_llm_params()

    # apply_provider_patches renames OpenAI max_tokens -> max_completion_tokens
    assert params["max_completion_tokens"] == 2048
    assert "max_tokens" not in params
    assert "api_key" not in params

    cheap = tracker.record(prompt_tokens=1000, completion_tokens=200, cached_tokens=500, calls=1)
    assert cheap.provider == "openai"
    assert cheap.cost_usd > 0.0
    assert cheap.budget_exceeded is False

    expensive = tracker.record(prompt_tokens=10000, completion_tokens=5000, calls=2)
    assert expensive.cost_usd > 0.01
    assert expensive.budget_exceeded is True


# ---------------------------------------------------------------------------
# 5. External SWE-bench Pro V2 Manifest & Determinism
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_external_manifest_enforces_v2_and_rejects_v1(tmp_path: Path) -> None:
    v2_file = tmp_path / "v2.jsonl"
    v2_file.write_text(
        json.dumps(
            {
                "instance_id": "repo__issue-1",
                "repo": "org/repo",
                "base_commit": "0123456789abcdef",
                "version": "v2",
                "revision": "sep-2026",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = load_external_manifest(v2_file, require_v2=True)
    assert len(loaded) == 1
    assert loaded[0]["instance_id"] == "repo__issue-1"

    v1_file = tmp_path / "v1.jsonl"
    v1_file.write_text(
        json.dumps(
            {
                "instance_id": "repo__old-1",
                "repo": "org/repo",
                "base_commit": "0123456789abcdef",
                "version": "v1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="legacy version"):
        load_external_manifest(v1_file, require_v2=True)


@pytest.mark.unit
def test_subset_selection_deterministic_digest() -> None:
    tasks = build_cleanroom_v2_subset()
    s1, d1 = select_subset(tasks, max_tasks=5, seed=1348)
    s2, d2 = select_subset(tasks, max_tasks=5, seed=1348)
    _, d3 = select_subset(tasks, max_tasks=5, seed=9999)
    assert [t.instance_id for t in s1] == [t.instance_id for t in s2]
    assert d1 == d2
    assert d1 != d3


# ---------------------------------------------------------------------------
# 6. Concurrency & Cross-Task Workspace Isolation
# ---------------------------------------------------------------------------


@pytest.mark.functional
@pytest.mark.asyncio
async def test_concurrent_asyncio_gather_zero_cross_task_contamination() -> None:
    tasks = build_cleanroom_v2_subset()[:6]
    canary_agent = _CanaryIsolationAgent()

    with SWEBenchProRunner(repo_scale="small", max_concurrency=4) as runner:
        results = await asyncio.gather(
            *[
                runner.run_task(task, canary_agent, StorageProfile.VISOR_ISOLATED)
                for task in tasks
            ]
        )

    assert len(results) == 6
    assert canary_agent.cross_task_leaks == []
    assert all(r.resolved and not r.tampered and not r.leakage_blocked for r in results)


# ---------------------------------------------------------------------------
# 7. Serialization & Report Formatting
# ---------------------------------------------------------------------------


@pytest.mark.functional
@pytest.mark.asyncio
async def test_score_table_and_json_report_roundtrip() -> None:
    tasks = build_cleanroom_v2_subset()[:2]
    with SWEBenchProRunner(repo_scale="tiny", max_concurrency=1) as runner:
        report = await runner.run_suite(
            tasks=tasks,
            agents=[OraclePatchAgent(), AdversarialTestEditorAgent()],
            profiles=(StorageProfile.PLAIN_NVME, StorageProfile.VISOR_ISOLATED),
        )

    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["benchmark_revision"] == "sep-2026"
    assert payload["benchmark_version"] == "v2"
    assert len(payload["results"]) == 8

    table = format_score_table(report)
    assert "| Agent | Profile |" in table
    assert "`oracle_patch`" in table
    assert "`redteam_test_editor`" in table
