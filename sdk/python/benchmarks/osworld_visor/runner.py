"""Async benchmark runner, built-in real workflow agents, and Markdown table formatter."""

from __future__ import annotations

import asyncio
import math
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from hanzo_agents.types import AIConfig

from .profiler import (
    CostTracker,
    collect_hardware_info,
    compute_io_delta,
    probe_desktop_tooling,
    take_io_snapshot,
)
from .sandbox import SandboxInstance, SubprocessExecResult, VisorSandboxManager
from .schemas import (
    BenchmarkReport,
    CleanupAuditResult,
    CohortSummary,
    ExternalOSWorldTask,
    OSWorldTaskSpec,
    SandboxProfile,
    SandboxResourceTelemetry,
    SecurityAuditResult,
    TaskRunResult,
    TaskStatus,
    WorkflowStep,
)
from .tasks import (
    build_osworld_subset,
    compute_manifest_digest,
    evaluate_task,
    load_external_osworld_dir,
)


class WorkflowAgent(ABC):
    """Abstract interface for computer-workflow agents evaluated on Visor."""

    name: str = "base_agent"

    @abstractmethod
    def plan_steps(self, task: OSWorldTaskSpec) -> Tuple[List[WorkflowStep], int, int, int]:
        """Return ``(steps, prompt_tokens, completion_tokens, cached_tokens)``."""


class OracleWorkflowAgent(WorkflowAgent):
    """Executes all reference multi-step commands cleanly inside the sandbox."""

    name = "oracle_agent"

    def plan_steps(self, task: OSWorldTaskSpec) -> Tuple[List[WorkflowStep], int, int, int]:
        prompt_tokens = 1200 + len(task.instruction) * 2 + len(task.seed_files) * 140
        completion_tokens = 420 * max(1, len(task.oracle_steps))
        cached_tokens = 600
        return list(task.oracle_steps), prompt_tokens, completion_tokens, cached_tokens


class PartialCapabilityAgent(WorkflowAgent):
    """Solves 6/10 tasks completely and executes incomplete steps on the remaining 4/10."""

    name = "partial_agent"

    SOLVED_IDS = frozenset(
        {
            "osw_01_filesystem_batch_ops",
            "osw_02_spreadsheet_csv_transform",
            "osw_04_cli_git_automation",
            "osw_06_document_markdown_report",
            "osw_08_permissions_and_cleanup_audit",
            "osw_10_json_yaml_pipeline_patch",
        }
    )

    def plan_steps(self, task: OSWorldTaskSpec) -> Tuple[List[WorkflowStep], int, int, int]:
        prompt_tokens = 1350 + len(task.instruction) * 2 + len(task.seed_files) * 140
        if task.task_id in self.SOLVED_IDS:
            return list(task.oracle_steps), prompt_tokens, 460 * len(task.oracle_steps), 500
        # Perform partial work (first step only if multi-step, or incomplete read-only probe)
        if len(task.oracle_steps) > 1:
            return [task.oracle_steps[0]], prompt_tokens, 320, 400
        partial_step = WorkflowStep(
            name="partial_probe",
            description="Inspect workspace without completing required output artifacts",
            command=[sys.executable, "-c", "import os; print(len(os.listdir('.')))"],
            timeout_s=10.0,
        )
        return [partial_step], prompt_tokens, 240, 400


class NoOpBaselineAgent(WorkflowAgent):
    """Launches a real isolated subprocess read-only check without modifying workspace."""

    name = "noop_agent"

    def plan_steps(self, task: OSWorldTaskSpec) -> Tuple[List[WorkflowStep], int, int, int]:
        noop_step = WorkflowStep(
            name="noop_inspect",
            description="Read-only workspace listing inside isolated sandbox",
            command=[sys.executable, "-c", "import os; print(sorted(os.listdir('.')))"],
            timeout_s=10.0,
        )
        return [noop_step], 480, 64, 200


class AdversarialOrphanLeakerAgent(WorkflowAgent):
    """Red-team buggy/adversarial agent that leaks a background orphan PID & TMPDIR file."""

    name = "redteam_orphan_leaker"

    def plan_steps(self, task: OSWorldTaskSpec) -> Tuple[List[WorkflowStep], int, int, int]:
        leak_code = (
            "import os, pathlib, subprocess, sys\n"
            "tmp = pathlib.Path(os.environ['TMPDIR'])\n"
            "(tmp / 'leaked_artifact.tmp').write_text('uncleaned temp data\\n', encoding='utf-8')\n"
            "subprocess.Popen(\n"
            "    [sys.executable, '-c', 'import time; time.sleep(30)'],\n"
            "    stdin=subprocess.DEVNULL,\n"
            "    stdout=subprocess.DEVNULL,\n"
            "    stderr=subprocess.DEVNULL,\n"
            ")\n"
        )
        leak_step = WorkflowStep(
            name="spawn_orphan_and_leak_tmp",
            description="Spawn detached background orphan sleeper and leak temp file in TMPDIR",
            command=[sys.executable, "-c", leak_code],
            timeout_s=10.0,
        )
        # Even if it also solves the task, cleanup failure must reject fail-closed!
        steps = list(task.oracle_steps) + [leak_step]
        return steps, 1280, 520, 400


class AdversarialSandboxEscapeAgent(WorkflowAgent):
    """Red-team agent attempting relative `../`, `../../` writes and symlink escape outside workspace."""

    name = "redteam_sandbox_escape"

    def plan_steps(self, task: OSWorldTaskSpec) -> Tuple[List[WorkflowStep], int, int, int]:
        escape_code = (
            "import os, pathlib\n"
            "for target in ('../escaped_outside.txt', '../../escaped_runs_root.txt'):\n"
            "    try:\n"
            "        pathlib.Path(target).write_text('escape-probe\\n', encoding='utf-8')\n"
            "    except OSError:\n"
            "        pass\n"
            "link = pathlib.Path('escape_canary_link')\n"
            "if not link.exists() and not link.is_symlink():\n"
            "    os.symlink('../host_guard/secret_canary.txt', link)\n"
        )
        escape_step = WorkflowStep(
            name="attempt_sandbox_escape",
            description="Attempt relative traversal writes and symlink pointing outside workspace",
            command=[sys.executable, "-c", escape_code],
            timeout_s=10.0,
        )
        steps = list(task.oracle_steps) + [escape_step]
        return steps, 1300, 510, 400


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil((pct / 100.0) * len(ordered)) - 1))
    return ordered[idx]


class OSWorldVisorRunner:
    """Runs reproducible OSWorld computer-workflow suites on cold vs warm-CAS Visor."""

    def __init__(
        self,
        *,
        tasks: Optional[Sequence[OSWorldTaskSpec]] = None,
        external_osworld_dir: Optional[Path] = None,
        ai_config: Optional[AIConfig] = None,
        workspace_scale: str = "medium",
        max_concurrency: int = 1,
        base_dir: Optional[Path] = None,
        enable_rlimits: bool = True,
        enable_fs_confinement: bool = True,
    ) -> None:
        self.tasks: List[OSWorldTaskSpec] = (
            list(tasks) if tasks is not None else build_osworld_subset()
        )
        self.workspace_scale = workspace_scale
        self.max_concurrency = max(1, int(max_concurrency))
        self.cost_tracker = CostTracker(ai_config=ai_config)
        self.hardware_info = collect_hardware_info()
        self.desktop_tooling = probe_desktop_tooling()
        self.external_tasks: List[ExternalOSWorldTask] = []
        if external_osworld_dir is not None:
            self.external_tasks = load_external_osworld_dir(
                Path(external_osworld_dir), tooling=self.desktop_tooling
            )
        self.sandbox_manager = VisorSandboxManager(
            base_dir=base_dir,
            workspace_scale=workspace_scale,
            enable_rlimits=enable_rlimits,
            enable_fs_confinement=enable_fs_confinement,
        )

    def close(self) -> None:
        self.sandbox_manager.close()

    def __enter__(self) -> "OSWorldVisorRunner":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _run_single_sync(
        self,
        task: OSWorldTaskSpec,
        agent: WorkflowAgent,
        profile: SandboxProfile,
    ) -> TaskRunResult:
        io_start = take_io_snapshot()
        telemetry = SandboxResourceTelemetry(profile=profile)
        cleanup_res = CleanupAuditResult()
        security_res = SecurityAuditResult()

        steps, prompt_tok, comp_tok, cached_tok = agent.plan_steps(task)
        cost = self.cost_tracker.record(
            prompt_tokens=prompt_tok,
            completion_tokens=comp_tok,
            cached_tokens=cached_tok,
            calls=max(1, len(steps)),
        )

        sandbox: Optional[SandboxInstance] = None
        steps_succeeded = 0
        any_timeout = False
        eval_passed = False
        eval_score = 0.0
        eval_checks: Dict[str, bool] = {}
        eval_details = ""
        eval_error = ""
        peak_rss = 0
        rss_sources: List[str] = []
        child_cpu_user_ms = 0.0
        child_cpu_sys_ms = 0.0
        child_inblock = 0
        child_oublock = 0

        def _accumulate_exec(res: SubprocessExecResult) -> None:
            nonlocal peak_rss, child_cpu_user_ms, child_cpu_sys_ms, child_inblock, child_oublock
            telemetry.subprocess_launches += 1
            if res.peak_rss_kib > peak_rss:
                peak_rss = res.peak_rss_kib
            if res.rss_source:
                rss_sources.append(res.rss_source)
            child_cpu_user_ms += res.cpu_user_ms
            child_cpu_sys_ms += res.cpu_system_ms
            child_inblock += res.inblock
            child_oublock += res.oublock

        try:
            sandbox = self.sandbox_manager.provision(task, profile)
            telemetry.provision_ms = round(sandbox.provision_ms, 4)
            telemetry.bytes_copied = sandbox.bytes_copied
            telemetry.hardlinked_files = sandbox.hardlinked_files
            telemetry.cas_cache_hit = sandbox.cas_cache_hit

            # Measure genuine isolated subprocess cold-start readiness.
            probe = self.sandbox_manager.measure_cold_start_probe(sandbox)
            _accumulate_exec(probe)
            telemetry.subprocess_boot_ms = round(probe.wall_ms, 4)
            telemetry.cold_start_ms = round(
                sandbox.provision_ms + probe.wall_ms, 4
            )

            # Execute real multi-step computer workflow inside isolated subprocesses.
            exec_t0 = time.perf_counter_ns()
            for step in steps:
                res = self.sandbox_manager.exec_in_sandbox(
                    sandbox,
                    step.command,
                    working_subdir=step.working_subdir,
                    timeout_s=step.timeout_s,
                )
                _accumulate_exec(res)
                if res.timed_out:
                    any_timeout = True
                    break
                if res.escape_rejected:
                    break
                if res.returncode == 0:
                    steps_succeeded += 1
            telemetry.execution_ms = round(
                (time.perf_counter_ns() - exec_t0) / 1_000_000.0, 4
            )

            # Security audit runs BEFORE accepting any evaluation result.
            security_res = self.sandbox_manager.audit_security(sandbox)

            # Independent evaluation of workspace outputs under the scrubbed env policy.
            eval_t0 = time.perf_counter_ns()
            try:
                eval_passed, eval_score, eval_checks, eval_details = evaluate_task(
                    task,
                    sandbox.workspace,
                    env=self.sandbox_manager.build_sandbox_env(sandbox),
                )
            except Exception as exc:
                eval_error = f"evaluator_error:{type(exc).__name__}:{exc}"
                eval_passed = False
                eval_score = 0.0
                eval_checks = {"evaluator_exception": False}
                eval_details = eval_error
            telemetry.evaluation_ms = round(
                (time.perf_counter_ns() - eval_t0) / 1_000_000.0, 4
            )
        finally:
            if sandbox is not None:
                cleanup_res, cleanup_ms = self.sandbox_manager.audit_and_teardown(
                    sandbox
                )
                telemetry.cleanup_ms = round(cleanup_ms, 4)

        io_end = take_io_snapshot()
        delta = compute_io_delta(io_start, io_end)
        telemetry.wall_ms = round(delta["wall_ms"], 4)
        telemetry.peak_rss_kib = peak_rss
        telemetry.rss_source = rss_sources[-1] if rss_sources else "unmeasured"
        telemetry.cpu_user_ms = round(
            child_cpu_user_ms + delta["self_cpu_user_ms"], 4
        )
        telemetry.cpu_system_ms = round(
            child_cpu_sys_ms + delta["self_cpu_system_ms"], 4
        )
        telemetry.proc_io_available = delta["proc_io_available"]
        telemetry.rchar_bytes = delta["rchar_bytes"]
        telemetry.wchar_bytes = delta["wchar_bytes"]
        telemetry.syscr_ops = delta["syscr_ops"]
        telemetry.syscw_ops = delta["syscw_ops"]
        telemetry.read_bytes_storage = delta["read_bytes_storage"]
        telemetry.write_bytes_storage = delta["write_bytes_storage"]
        telemetry.rusage_inblock = child_inblock + delta["self_inblock"]
        telemetry.rusage_oublock = child_oublock + delta["self_oublock"]

        # Fail-closed verdict ordering: security escape > cleanup violation > timeout > eval error > eval.
        if security_res.escaped:
            status = TaskStatus.SANDBOX_ESCAPE_BLOCKED
            success = False
            score = 0.0
            eval_details = f"escape_blocked:{security_res.violations}"
        elif not cleanup_res.cleanup_clean:
            status = TaskStatus.CLEANUP_VIOLATION
            success = False
            score = 0.0
            eval_details = f"cleanup_violation:{cleanup_res.violations}"
        elif any_timeout:
            status = TaskStatus.TIMEOUT
            success = False
            score = 0.0
            eval_details = "step_timeout"
        elif eval_error:
            status = TaskStatus.ERROR
            success = False
            score = 0.0
            eval_details = eval_error
        elif eval_passed:
            status = TaskStatus.SUCCESS
            success = True
            score = eval_score
        else:
            status = TaskStatus.FAILED
            success = False
            score = eval_score

        return TaskRunResult(
            task_id=task.task_id,
            title=task.title,
            category=task.category,
            osworld_domain=task.osworld_domain,
            agent_name=agent.name,
            profile=profile,
            status=status,
            success=success,
            score=score,
            evaluator_checks=eval_checks,
            evaluator_details=eval_details,
            step_count=len(steps),
            steps_succeeded=steps_succeeded,
            telemetry=telemetry,
            cleanup=cleanup_res,
            security=security_res,
            cost=cost,
        )

    async def run_suite(
        self,
        *,
        agents: Sequence[WorkflowAgent],
        profiles: Sequence[SandboxProfile] = (
            SandboxProfile.COLD_SANDBOX,
            SandboxProfile.WARM_CAS_VISOR,
        ),
        warm_visor_cas: bool = False,
    ) -> BenchmarkReport:
        """Execute all (agent, profile, task) rollouts with bounded async concurrency."""
        if warm_visor_cas and SandboxProfile.WARM_CAS_VISOR in profiles:
            self.sandbox_manager.warm_cas_pool(self.tasks)

        sem = asyncio.Semaphore(self.max_concurrency)
        all_results: List[TaskRunResult] = []
        cohorts: List[CohortSummary] = []

        async def _guarded(
            task: OSWorldTaskSpec, agent: WorkflowAgent, profile: SandboxProfile
        ) -> TaskRunResult:
            async with sem:
                return await asyncio.to_thread(
                    self._run_single_sync, task, agent, profile
                )

        for agent in agents:
            for profile in profiles:
                cohort_results = list(
                    await asyncio.gather(
                        *[_guarded(t, agent, profile) for t in self.tasks]
                    )
                )
                all_results.extend(cohort_results)
                cohorts.append(_summarize_cohort(agent.name, profile, cohort_results))

        return BenchmarkReport(
            benchmark_name="OSWorld Computer-Workflow Visor Subset",
            hip_reference="HIP-1348 (hanzoai/agents#80, #89)",
            osworld_license="Apache-2.0 (xlang-ai/OSWorld reference; clean-room runner)",
            manifest_digest=compute_manifest_digest(self.tasks, self.external_tasks),
            workspace_scale=self.workspace_scale,
            warm_visor_cas=warm_visor_cas,
            concurrency=self.max_concurrency,
            hardware_info=self.hardware_info,
            desktop_tooling=self.desktop_tooling,
            external_tasks_loaded=len(self.external_tasks),
            external_tasks=list(self.external_tasks),
            cohorts=cohorts,
            results=all_results,
        )


def _summarize_cohort(
    agent_name: str,
    profile: SandboxProfile,
    results: Sequence[TaskRunResult],
) -> CohortSummary:
    n = len(results)
    succeeded = sum(1 for r in results if r.success)
    cleanup_clean = sum(1 for r in results if r.cleanup.cleanup_clean)
    cleanup_violations = sum(1 for r in results if not r.cleanup.cleanup_clean)
    escape_blocks = sum(1 for r in results if r.security.escaped)

    cold_starts = [r.telemetry.cold_start_ms for r in results]
    provisions = [r.telemetry.provision_ms for r in results]
    walls = [r.telemetry.wall_ms for r in results]
    rss_vals = [r.telemetry.peak_rss_kib for r in results]
    cpu_vals = [
        r.telemetry.cpu_user_ms + r.telemetry.cpu_system_ms for r in results
    ]

    total_cost = round(sum(r.cost.cost_usd for r in results), 6)
    cost_per_success = round(total_cost / succeeded, 6) if succeeded > 0 else None

    return CohortSummary(
        agent_name=agent_name,
        profile=profile,
        total_tasks=n,
        succeeded_tasks=succeeded,
        success_rate=round((succeeded / n) * 100.0, 2) if n else 0.0,
        cleanup_clean_tasks=cleanup_clean,
        cleanup_violations=cleanup_violations,
        escape_blocks=escape_blocks,
        mean_cold_start_ms=round(sum(cold_starts) / n, 2) if n else 0.0,
        p95_cold_start_ms=round(_percentile(cold_starts, 95.0), 2) if n else 0.0,
        mean_provision_ms=round(sum(provisions) / n, 2) if n else 0.0,
        mean_wall_ms=round(sum(walls) / n, 2) if n else 0.0,
        mean_peak_rss_kib=round(sum(rss_vals) / n, 1) if n else 0.0,
        max_peak_rss_kib=max(rss_vals) if rss_vals else 0,
        mean_cpu_ms=round(sum(cpu_vals) / n, 2) if n else 0.0,
        total_bytes_copied=sum(r.telemetry.bytes_copied for r in results),
        total_hardlinked_files=sum(r.telemetry.hardlinked_files for r in results),
        total_wchar_bytes=sum(r.telemetry.wchar_bytes for r in results),
        total_rchar_bytes=sum(r.telemetry.rchar_bytes for r in results),
        total_cost_usd=total_cost,
        cost_per_success_usd=cost_per_success,
    )


def format_score_table(report: BenchmarkReport) -> str:
    """Format empirical cohort results into a GitHub-flavored Markdown table."""
    lines = [
        f"# {report.benchmark_name} — Score Table",
        "",
        f"- **HIP Reference**: {report.hip_reference}",
        f"- **Manifest Digest**: `{report.manifest_digest[:16]}...`",
        f"- **Workspace Scale**: `{report.workspace_scale}` | **Warm Visor CAS**: `{report.warm_visor_cas}` | **Concurrency**: `{report.concurrency}`",
        f"- **Host**: `{report.hardware_info.get('os')} {report.hardware_info.get('release')}` ({report.hardware_info.get('machine')}), Python `{report.hardware_info.get('python_version')}`, `/proc/self/io`: `{report.hardware_info.get('proc_io_available')}`",
        "",
        "| Agent | Profile | Success | Rate | Cleanup Clean | Escapes Blocked | Mean Cold-Start (ms) | p95 Cold-Start (ms) | Mean Provision (ms) | Mean Wall (ms) | Mean Peak RSS (KiB) | Bytes Copied (MiB) | Hardlinks | `/proc/self/io` wchar (MiB) | Cost (USD) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for c in report.cohorts:
        copied_mib = c.total_bytes_copied / (1024.0 * 1024.0)
        wchar_mib = c.total_wchar_bytes / (1024.0 * 1024.0)
        lines.append(
            f"| `{c.agent_name}` | `{c.profile.value}` | {c.succeeded_tasks}/{c.total_tasks} | "
            f"{c.success_rate:.1f}% | {c.cleanup_clean_tasks}/{c.total_tasks} | {c.escape_blocks} | "
            f"{c.mean_cold_start_ms:.2f} | {c.p95_cold_start_ms:.2f} | {c.mean_provision_ms:.2f} | "
            f"{c.mean_wall_ms:.2f} | {c.mean_peak_rss_kib:.1f} | {copied_mib:.2f} | "
            f"{c.total_hardlinked_files} | {wchar_mib:.2f} | \\${c.total_cost_usd:.5f} |"
        )
    lines.append("")
    return "\n".join(lines)
