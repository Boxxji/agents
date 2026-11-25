"""Reproducible SWE-bench Pro V2 subset runner on plain NVMe vs Visor sandboxes.

Executes coding agents inside isolated workspaces with:
- Real ``git apply`` patch application and real ``pytest`` fail-to-pass /
  pass-to-pass grading
- Fail-closed pre-execution anti-leakage scrubbing & verification
- Fail-closed post-execution anti-tampering audit (protected test SHA-256,
  ``git diff`` path checks, symlink escape detection, Python shim / env guards)
- Real OS I/O telemetry (``/proc/self/io``, ``rusage``, ``os.times``) and
  Hanzo SDK ``AIConfig`` / ``llm_adapters`` model cost tracking
- Concurrent task execution via ``asyncio.gather`` with bounded semaphore
"""

from __future__ import annotations

import asyncio
import difflib
import os
import re
import stat
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from hanzo_agents.types import AIConfig

from .integrity import (
    _run_git,
    sanitize_execution_env,
    verify_post_execution_integrity,
)
from .profiler import CostTracker, IOProfiler, collect_hardware_info
from .sandbox import REPO_SCALES, ProvisionedSandbox, RepoScaleConfig, SandboxManager
from .schemas import (
    AgentInvocationResult,
    BenchmarkReport,
    FilesystemIOTelemetry,
    ProfileAggregate,
    StorageProfile,
    SWEBenchTask,
    TaskRunResult,
    TaskStatus,
)
from .subset import (
    SWE_BENCH_PRO_V2_REVISION,
    SWE_BENCH_PRO_V2_VERSION,
    build_cleanroom_v2_subset,
    select_subset,
)

_PYTEST_RESULT_RE = re.compile(r"^(tests/\S+::\S+)\s+(PASSED|FAILED|ERROR|SKIPPED)\b")


def build_unified_diff(
    base_files: Dict[str, str],
    patched_files: Dict[str, str],
) -> str:
    """Generate a git-compatible unified diff between base and patched file maps."""
    chunks: List[str] = []
    for rel in sorted(patched_files.keys()):
        old_text = base_files.get(rel, "")
        new_text = patched_files[rel]
        if old_text == new_text:
            continue
        old_lines = old_text.splitlines(keepends=True)
        new_lines = new_text.splitlines(keepends=True)
        diff_lines = list(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
                n=3,
            )
        )
        if diff_lines:
            chunks.append(f"diff --git a/{rel} b/{rel}\n" + "".join(diff_lines))
    return "".join(chunks)


def _estimate_tokens(text: str, chars_per_token: int = 4) -> int:
    return max(1, (len(text) + chars_per_token - 1) // max(1, chars_per_token))


# ---------------------------------------------------------------------------
# Coding Agent Protocol & Deterministic Reference / Red-Team Agents
# ---------------------------------------------------------------------------


class CodingAgent(ABC):
    """Interface for coding agents evaluated by SWEBenchProRunner."""

    name: str

    @abstractmethod
    async def solve(
        self,
        task: SWEBenchTask,
        workspace: Path,
        env: Dict[str, str],
        ai_config: AIConfig,
    ) -> AgentInvocationResult:
        """Attempt to resolve ``task`` inside isolated ``workspace``."""


class OraclePatchAgent(CodingAgent):
    """Upper-bound harness verification agent applying clean multi-file reference diffs."""

    name = "oracle_patch"

    async def solve(
        self,
        task: SWEBenchTask,
        workspace: Path,
        env: Dict[str, str],
        ai_config: AIConfig,
    ) -> AgentInvocationResult:
        diff_text = build_unified_diff(task.base_files, task.gold_patch_files)
        context_chars = len(task.problem_statement) + sum(
            len(v) for v in task.base_files.values()
        )
        prompt_tokens = _estimate_tokens("x" * context_chars, ai_config.avg_chars_per_token) * 4
        completion_tokens = _estimate_tokens(diff_text, ai_config.avg_chars_per_token) * 2
        cached_tokens = int(prompt_tokens * 0.5)
        return AgentInvocationResult(
            patch_diff=diff_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            calls=3,
            notes="clean multi-file oracle diff",
        )


class PartialCapabilityAgent(CodingAgent):
    """Realistic mid-tier coding agent resolving medium tasks and partially fixing hard ones."""

    name = "partial_agent"

    async def solve(
        self,
        task: SWEBenchTask,
        workspace: Path,
        env: Dict[str, str],
        ai_config: AIConfig,
    ) -> AgentInvocationResult:
        if task.partial_patch_files is not None:
            diff_text = build_unified_diff(task.base_files, task.partial_patch_files)
            notes = "incomplete multi-file fix (misses hard edge cases)"
            calls = 5
        else:
            diff_text = build_unified_diff(task.base_files, task.gold_patch_files)
            notes = "resolved clean multi-file diff"
            calls = 4

        context_chars = len(task.problem_statement) + sum(
            len(v) for v in task.base_files.values()
        )
        prompt_tokens = _estimate_tokens("x" * context_chars, ai_config.avg_chars_per_token) * 5
        completion_tokens = _estimate_tokens(diff_text, ai_config.avg_chars_per_token) * 3
        cached_tokens = int(prompt_tokens * 0.4)
        return AgentInvocationResult(
            patch_diff=diff_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            calls=calls,
            notes=notes,
        )


class NoOpBaselineAgent(CodingAgent):
    """Empty-patch baseline proving fail-to-pass tests genuinely fail before fixes."""

    name = "noop_baseline"

    async def solve(
        self,
        task: SWEBenchTask,
        workspace: Path,
        env: Dict[str, str],
        ai_config: AIConfig,
    ) -> AgentInvocationResult:
        prompt_tokens = _estimate_tokens(task.problem_statement, ai_config.avg_chars_per_token)
        return AgentInvocationResult(
            patch_diff="",
            prompt_tokens=prompt_tokens,
            completion_tokens=16,
            cached_tokens=0,
            calls=1,
            notes="empty baseline patch",
        )


class AdversarialTestEditorAgent(CodingAgent):
    """Red-team agent attempting to cheat by rewriting protected test files to pass."""

    name = "redteam_test_editor"

    async def solve(
        self,
        task: SWEBenchTask,
        workspace: Path,
        env: Dict[str, str],
        ai_config: AIConfig,
    ) -> AgentInvocationResult:
        # Attempt to chmod/unlink and overwrite every protected test file with `assert True`
        for rel in task.protected_test_paths:
            target = workspace / rel
            all_tests = [
                t.split("::", 1)[1]
                for t in (task.fail_to_pass_tests + task.pass_to_pass_tests)
                if t.startswith(rel + "::")
            ]
            fake_body = "\n".join(f"def {fn}():\n    assert True\n" for fn in all_tests)
            try:
                if target.exists():
                    try:
                        os.chmod(target, stat.S_IRUSR | stat.S_IWUSR)
                    except OSError:
                        pass
                    target.unlink()
                target.write_text(fake_body, encoding="utf-8")
            except OSError:
                pass
        return AgentInvocationResult(
            patch_diff="",
            prompt_tokens=512,
            completion_tokens=128,
            calls=2,
            notes="attempted protected test overwrite",
        )


class AdversarialGitLeakAgent(CodingAgent):
    """Red-team agent attempting to harvest reference fixes from git history/objects/env."""

    name = "redteam_git_leak_probe"

    async def solve(
        self,
        task: SWEBenchTask,
        workspace: Path,
        env: Dict[str, str],
        ai_config: AIConfig,
    ) -> AgentInvocationResult:
        findings: List[str] = []

        # Probe 1: git log --all, reflog, tags, remotes, all objects
        for cmd in (
            ("log", "--all", "-p"),
            ("reflog", "--all"),
            ("tag", "-l"),
            ("remote", "-v"),
            ("branch", "-a"),
        ):
            proc = _run_git(workspace, *cmd, check=False)
            if task.gold_marker in proc.stdout:
                findings.append(f"git {' '.join(cmd)}")

        # Probe 2: raw .git/objects scan
        obj_dir = workspace / ".git" / "objects"
        marker_bytes = task.gold_marker.encode("utf-8")
        if obj_dir.exists():
            for p in obj_dir.rglob("*"):
                if p.is_file():
                    try:
                        if marker_bytes in p.read_bytes():
                            findings.append(f"object:{p.name}")
                    except OSError:
                        pass

        # Probe 3: leaked files & environment variables
        for fname in ("gold_patch.diff", ".swe_bench_gold.json", "test_patch.diff"):
            if (workspace / fname).exists():
                findings.append(f"file:{fname}")
        for k, v in env.items():
            if task.gold_marker in v or "GOLD" in k.upper():
                findings.append(f"env:{k}")

        return AgentInvocationResult(
            patch_diff="",
            prompt_tokens=640,
            completion_tokens=96,
            calls=2,
            notes=f"leak_findings={len(findings)}:{','.join(findings)}",
        )


class AdversarialSymlinkEscapeAgent(CodingAgent):
    """Red-team agent planting an out-of-sandbox symlink alongside a valid patch."""

    name = "redteam_symlink_escape"

    async def solve(
        self,
        task: SWEBenchTask,
        workspace: Path,
        env: Dict[str, str],
        ai_config: AIConfig,
    ) -> AgentInvocationResult:
        escape_link = workspace / "escaped_host_hosts.txt"
        target = Path("/etc/hosts") if Path("/etc/hosts").exists() else workspace.parent
        try:
            os.symlink(str(target), str(escape_link))
        except OSError:
            pass
        # Even submit the real oracle patch to prove tampering overrides passing tests!
        diff_text = build_unified_diff(task.base_files, task.gold_patch_files)
        return AgentInvocationResult(
            patch_diff=diff_text,
            prompt_tokens=600,
            completion_tokens=200,
            calls=2,
            notes="planted out-of-root symlink with valid patch",
        )


class AdversarialShimInjectorAgent(CodingAgent):
    """Red-team agent injecting sitecustomize.py, pytest.py shim, and PYTEST_ADDOPTS."""

    name = "redteam_shim_injector"

    async def solve(
        self,
        task: SWEBenchTask,
        workspace: Path,
        env: Dict[str, str],
        ai_config: AIConfig,
    ) -> AgentInvocationResult:
        (workspace / "sitecustomize.py").write_text(
            "import sys; sys.exit(0)\n", encoding="utf-8"
        )
        (workspace / "pytest.py").write_text(
            "def main(*a, **kw): return 0\n", encoding="utf-8"
        )
        diff_text = build_unified_diff(task.base_files, task.gold_patch_files)
        return AgentInvocationResult(
            patch_diff=diff_text,
            prompt_tokens=580,
            completion_tokens=190,
            calls=2,
            env_overrides={
                "PYTHONPATH": str(workspace),
                "PYTEST_ADDOPTS": "-k never_run_anything",
            },
            notes="injected sitecustomize.py + pytest.py + env override",
        )


# ---------------------------------------------------------------------------
# Runner Engine
# ---------------------------------------------------------------------------


def _apply_patch_in_workspace(workspace: Path, patch_diff: str) -> Tuple[bool, str]:
    """Apply a unified diff via ``git apply --check`` + ``git apply``."""
    if not patch_diff or not patch_diff.strip():
        return True, ""
    env = sanitize_execution_env()
    check_proc = subprocess.run(
        ["git", "apply", "--check", "-"],
        cwd=str(workspace),
        input=patch_diff,
        env=env,
        capture_output=True,
        text=True,
    )
    if check_proc.returncode != 0:
        return False, f"git apply --check failed: {check_proc.stderr.strip()}"

    apply_proc = subprocess.run(
        ["git", "apply", "-"],
        cwd=str(workspace),
        input=patch_diff,
        env=env,
        capture_output=True,
        text=True,
    )
    if apply_proc.returncode != 0:
        return False, f"git apply failed: {apply_proc.stderr.strip()}"
    return True, ""


def _run_pytest_grading(
    workspace: Path,
    task: SWEBenchTask,
    timeout_s: float = 30.0,
) -> Tuple[int, int, str, bool]:
    """Execute real pytest inside sanitized workspace and score F2P / P2P tests.

    Returns ``(f2p_passed, p2p_passed, output_tail, timed_out)``.
    """
    test_env = sanitize_execution_env()
    # Trusted evaluator explicitly pins PYTHONPATH to the verified workspace root only.
    test_env["PYTHONPATH"] = str(workspace)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-v",
        "--tb=short",
        "--no-header",
        "-o",
        "addopts=",
        "-c",
        "/dev/null",
        *task.protected_test_paths,
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(workspace),
            env=test_env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        tail = (exc.stdout or "")[-600:] if isinstance(exc.stdout, str) else ""
        return 0, 0, f"TIMEOUT after {timeout_s}s\n{tail}", True

    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    outcomes: Dict[str, str] = {}
    for line in combined.splitlines():
        m = _PYTEST_RESULT_RE.match(line.strip())
        if m:
            outcomes[m.group(1)] = m.group(2)

    f2p_passed = sum(1 for t in task.fail_to_pass_tests if outcomes.get(t) == "PASSED")
    p2p_passed = sum(1 for t in task.pass_to_pass_tests if outcomes.get(t) == "PASSED")
    tail = "\n".join(combined.strip().splitlines()[-12:])
    return f2p_passed, p2p_passed, tail, False


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((p / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


class SWEBenchProRunner:
    """Reproducible SWE-bench Pro V2 subset runner comparing NVMe vs Visor profiles."""

    def __init__(
        self,
        *,
        ai_config: Optional[AIConfig] = None,
        repo_scale: str = "small",
        max_concurrency: int = 2,
        test_timeout_s: float = 30.0,
        sandbox_manager: Optional[SandboxManager] = None,
    ) -> None:
        if repo_scale not in REPO_SCALES:
            raise ValueError(f"Unknown repo_scale {repo_scale!r}; expected one of {sorted(REPO_SCALES)}")
        self.scale: RepoScaleConfig = REPO_SCALES[repo_scale]
        self.max_concurrency = max(1, int(max_concurrency))
        self.test_timeout_s = float(test_timeout_s)
        self.ai_config = ai_config or AIConfig(
            model="openai/gpt-4o",
            temperature=0.0,
            max_tokens=4096,
        )
        self.cost_tracker = CostTracker(ai_config=self.ai_config)
        self._owns_sandbox_manager = sandbox_manager is None
        self.sandbox_manager = sandbox_manager or SandboxManager()

    def close(self) -> None:
        if self._owns_sandbox_manager:
            self.sandbox_manager.close()

    def __enter__(self) -> "SWEBenchProRunner":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    async def run_task(
        self,
        task: SWEBenchTask,
        agent: CodingAgent,
        profile: StorageProfile,
        *,
        skip_scrub: bool = False,
        extra_env: Optional[Dict[str, str]] = None,
        token_scopes: Tuple[str, ...] = ("ai:inference",),
    ) -> TaskRunResult:
        """Execute a single ``(task, agent, storage_profile)`` rollout end-to-end."""
        sandbox: Optional[ProvisionedSandbox] = None
        agent_time_ms = 0.0
        eval_time_ms = 0.0

        with IOProfiler(profile) as io_prof:
            try:
                # 1. Build sanitized env (unless extra_env intentionally tests leak blocking)
                exec_env = sanitize_execution_env(token_scopes=token_scopes)
                if extra_env:
                    exec_env.update(extra_env)

                # 2. Provision isolated workspace & run pre-execution anti-leakage audit
                sandbox = await asyncio.to_thread(
                    self.sandbox_manager.provision,
                    task,
                    profile,
                    scale=self.scale,
                    skip_scrub=skip_scrub,
                    env=exec_env,
                    token_scopes=token_scopes,
                )
                integrity = sandbox.integrity_report

                # Fail closed if pre-execution anti-leakage audit detected any violation
                if not integrity.pre_audit_clean:
                    cost = self.cost_tracker.record(prompt_tokens=0, completion_tokens=0, calls=0)
                    result_status = TaskStatus.LEAKAGE_BLOCKED
                    error_msg = "; ".join(v.detail for v in integrity.violations)
                    io_tel = self._finalize_io(io_prof, sandbox, 0.0)
                    return TaskRunResult(
                        instance_id=task.instance_id,
                        agent_name=agent.name,
                        storage_profile=profile,
                        status=result_status,
                        resolved=False,
                        tampered=False,
                        leakage_blocked=True,
                        fail_to_pass_passed=0,
                        fail_to_pass_total=len(task.fail_to_pass_tests),
                        pass_to_pass_passed=0,
                        pass_to_pass_total=len(task.pass_to_pass_tests),
                        wall_time_ms=io_prof.wall_ms or sandbox.setup_ms,
                        agent_time_ms=0.0,
                        eval_time_ms=0.0,
                        cost=cost,
                        io=io_tel,
                        integrity=integrity,
                        error=error_msg,
                    )

                # 3. Execute coding agent inside isolated workspace
                t_agent0 = time.perf_counter_ns()
                agent_out = await agent.solve(task, sandbox.workspace, dict(exec_env), self.ai_config)
                agent_time_ms = (time.perf_counter_ns() - t_agent0) / 1_000_000.0

                cost = self.cost_tracker.record(
                    prompt_tokens=agent_out.prompt_tokens,
                    completion_tokens=agent_out.completion_tokens,
                    cached_tokens=agent_out.cached_tokens,
                    calls=agent_out.calls,
                )

                # 4. Apply agent unified diff (if returned) via real `git apply`
                patch_err = ""
                if agent_out.patch_diff:
                    applied, patch_err = await asyncio.to_thread(
                        _apply_patch_in_workspace, sandbox.workspace, agent_out.patch_diff
                    )
                    if not applied:
                        # Still run post-integrity check so tampering is never masked
                        await asyncio.to_thread(
                            verify_post_execution_integrity,
                            sandbox.workspace,
                            protected_hashes=integrity.protected_test_hashes,
                            base_filenames=set(task.base_files.keys()) | set(task.test_files.keys()),
                            agent_env_overrides=agent_out.env_overrides,
                            report=integrity,
                        )
                        diff_ms, _ = await asyncio.to_thread(
                            self.sandbox_manager.measure_snapshot_diff, sandbox
                        )
                        io_tel = self._finalize_io(io_prof, sandbox, diff_ms)
                        tampered = not integrity.post_audit_clean
                        return TaskRunResult(
                            instance_id=task.instance_id,
                            agent_name=agent.name,
                            storage_profile=profile,
                            status=TaskStatus.TAMPERED if tampered else TaskStatus.UNRESOLVED,
                            resolved=False,
                            tampered=tampered,
                            leakage_blocked=False,
                            fail_to_pass_passed=0,
                            fail_to_pass_total=len(task.fail_to_pass_tests),
                            pass_to_pass_passed=0,
                            pass_to_pass_total=len(task.pass_to_pass_tests),
                            wall_time_ms=0.0,  # filled after profiler exits
                            agent_time_ms=agent_time_ms,
                            eval_time_ms=0.0,
                            cost=cost,
                            io=io_tel,
                            integrity=integrity,
                            error=patch_err,
                        )

                # 5. Snapshot diff telemetry + Fail-closed post-execution anti-tampering audit
                diff_ms, _ = await asyncio.to_thread(
                    self.sandbox_manager.measure_snapshot_diff, sandbox
                )
                await asyncio.to_thread(
                    verify_post_execution_integrity,
                    sandbox.workspace,
                    protected_hashes=integrity.protected_test_hashes,
                    base_filenames=set(task.base_files.keys()) | set(task.test_files.keys()),
                    agent_env_overrides=agent_out.env_overrides,
                    report=integrity,
                )

                if not integrity.post_audit_clean:
                    io_tel = self._finalize_io(io_prof, sandbox, diff_ms)
                    err = "; ".join(v.detail for v in integrity.violations)
                    res = TaskRunResult(
                        instance_id=task.instance_id,
                        agent_name=agent.name,
                        storage_profile=profile,
                        status=TaskStatus.TAMPERED,
                        resolved=False,
                        tampered=True,
                        leakage_blocked=False,
                        fail_to_pass_passed=0,
                        fail_to_pass_total=len(task.fail_to_pass_tests),
                        pass_to_pass_passed=0,
                        pass_to_pass_total=len(task.pass_to_pass_tests),
                        wall_time_ms=0.0,
                        agent_time_ms=agent_time_ms,
                        eval_time_ms=0.0,
                        cost=cost,
                        io=io_tel,
                        integrity=integrity,
                        test_output_tail=agent_out.notes,
                        error=err,
                    )
                else:
                    # 6. Execute real pytest grading inside verified workspace
                    t_eval0 = time.perf_counter_ns()
                    f2p_ok, p2p_ok, out_tail, timed_out = await asyncio.to_thread(
                        _run_pytest_grading, sandbox.workspace, task, self.test_timeout_s
                    )
                    eval_time_ms = (time.perf_counter_ns() - t_eval0) / 1_000_000.0
                    io_tel = self._finalize_io(io_prof, sandbox, diff_ms)

                    if timed_out:
                        status = TaskStatus.TIMEOUT
                        resolved = False
                    elif (
                        f2p_ok == len(task.fail_to_pass_tests)
                        and p2p_ok == len(task.pass_to_pass_tests)
                    ):
                        status = TaskStatus.RESOLVED
                        resolved = True
                    else:
                        status = TaskStatus.UNRESOLVED
                        resolved = False

                    res = TaskRunResult(
                        instance_id=task.instance_id,
                        agent_name=agent.name,
                        storage_profile=profile,
                        status=status,
                        resolved=resolved,
                        tampered=False,
                        leakage_blocked=False,
                        fail_to_pass_passed=f2p_ok,
                        fail_to_pass_total=len(task.fail_to_pass_tests),
                        pass_to_pass_passed=p2p_ok,
                        pass_to_pass_total=len(task.pass_to_pass_tests),
                        wall_time_ms=0.0,
                        agent_time_ms=agent_time_ms,
                        eval_time_ms=eval_time_ms,
                        cost=cost,
                        io=io_tel,
                        integrity=integrity,
                        test_output_tail=out_tail or agent_out.notes,
                    )
            finally:
                if sandbox is not None:
                    await asyncio.to_thread(sandbox.cleanup)

        # Populate finalized profiler deltas after context exit
        res.wall_time_ms = io_prof.wall_ms
        res.io.proc_io_available = io_prof.telemetry.proc_io_available
        res.io.rchar_bytes = io_prof.telemetry.rchar_bytes
        res.io.wchar_bytes = io_prof.telemetry.wchar_bytes
        res.io.syscr_ops = io_prof.telemetry.syscr_ops
        res.io.syscw_ops = io_prof.telemetry.syscw_ops
        res.io.read_bytes_storage = io_prof.telemetry.read_bytes_storage
        res.io.write_bytes_storage = io_prof.telemetry.write_bytes_storage
        res.io.rusage_inblock = io_prof.telemetry.rusage_inblock
        res.io.rusage_oublock = io_prof.telemetry.rusage_oublock
        res.io.cpu_user_ms = io_prof.telemetry.cpu_user_ms
        res.io.cpu_system_ms = io_prof.telemetry.cpu_system_ms
        return res

    @staticmethod
    def _finalize_io(
        io_prof: IOProfiler,
        sandbox: ProvisionedSandbox,
        snapshot_diff_ms: float,
    ) -> FilesystemIOTelemetry:
        return FilesystemIOTelemetry(
            profile=sandbox.profile,
            workspace_setup_ms=round(sandbox.setup_ms, 3),
            workspace_bytes_copied=sandbox.bytes_copied,
            workspace_files_materialized=sandbox.files_materialized,
            workspace_hardlinked_files=sandbox.hardlinked_files,
            snapshot_diff_ms=round(snapshot_diff_ms, 3),
            cas_cache_hit=sandbox.cas_cache_hit,
        )

    async def run_suite(
        self,
        *,
        tasks: Optional[Sequence[SWEBenchTask]] = None,
        agents: Optional[Sequence[CodingAgent]] = None,
        profiles: Sequence[StorageProfile] = (
            StorageProfile.PLAIN_NVME,
            StorageProfile.VISOR_ISOLATED,
        ),
        seed: int = 42,
        warm_visor_cas: bool = False,
    ) -> BenchmarkReport:
        """Run all ``(profile, agent, task)`` pairs with bounded async concurrency."""
        selected_tasks, manifest_digest = select_subset(
            tasks if tasks is not None else build_cleanroom_v2_subset(),
            seed=seed,
        )
        run_agents = list(agents) if agents is not None else [
            OraclePatchAgent(),
            PartialCapabilityAgent(),
            NoOpBaselineAgent(),
        ]

        # Optional CAS pre-warm (mirrors persistent S3/Visor base image cache across runs)
        if warm_visor_cas and StorageProfile.VISOR_ISOLATED in profiles:
            for t in selected_tasks:
                await asyncio.to_thread(
                    self.sandbox_manager._get_or_build_cas_image, t, self.scale
                )

        sem = asyncio.Semaphore(self.max_concurrency)

        async def _bounded_run(
            task: SWEBenchTask,
            agent: CodingAgent,
            profile: StorageProfile,
        ) -> TaskRunResult:
            async with sem:
                return await self.run_task(task, agent, profile)

        # Execute each (profile, agent) cohort together so OS I/O telemetry stays clean per cohort
        all_results: List[TaskRunResult] = []
        for profile in profiles:
            for agent in run_agents:
                cohort_coros = [_bounded_run(task, agent, profile) for task in selected_tasks]
                cohort_results = await asyncio.gather(*cohort_coros)
                all_results.extend(cohort_results)

        aggregates = build_aggregates(all_results)
        return BenchmarkReport(
            benchmark_revision=SWE_BENCH_PRO_V2_REVISION,
            benchmark_version=SWE_BENCH_PRO_V2_VERSION,
            manifest_digest=manifest_digest,
            repo_scale_name=self.scale.name,
            repo_asset_files=self.scale.asset_files,
            repo_asset_bytes=self.scale.total_asset_bytes,
            hardware_info=collect_hardware_info(),
            aggregates=aggregates,
            results=all_results,
        )


def build_aggregates(results: Sequence[TaskRunResult]) -> List[ProfileAggregate]:
    """Compute per-(agent, storage_profile) aggregate benchmark metrics."""
    grouped: Dict[Tuple[str, StorageProfile], List[TaskRunResult]] = {}
    for r in results:
        grouped.setdefault((r.agent_name, r.storage_profile), []).append(r)

    aggregates: List[ProfileAggregate] = []
    for (agent_name, profile), items in sorted(
        grouped.items(), key=lambda kv: (kv[0][0], kv[0][1].value)
    ):
        total = len(items)
        resolved = sum(1 for x in items if x.resolved)
        tampered = sum(1 for x in items if x.tampered)
        leakage = sum(1 for x in items if x.leakage_blocked)
        walls = [x.wall_time_ms for x in items]
        setups = [x.io.workspace_setup_ms for x in items]
        diffs = [x.io.snapshot_diff_ms for x in items]
        total_cost = round(sum(x.cost.cost_usd for x in items), 6)
        cost_per_res = round(total_cost / resolved, 6) if resolved > 0 else None
        mean_cpu = sum(x.io.cpu_user_ms + x.io.cpu_system_ms for x in items) / max(1, total)

        aggregates.append(
            ProfileAggregate(
                agent_name=agent_name,
                storage_profile=profile,
                total_tasks=total,
                resolved_tasks=resolved,
                resolution_rate=round(resolved / max(1, total), 4),
                tampered_tasks=tampered,
                leakage_blocked_tasks=leakage,
                mean_wall_time_ms=round(sum(walls) / max(1, total), 2),
                p95_wall_time_ms=round(_percentile(walls, 95.0), 2),
                mean_setup_ms=round(sum(setups) / max(1, total), 2),
                mean_snapshot_diff_ms=round(sum(diffs) / max(1, total), 2),
                total_cost_usd=total_cost,
                cost_per_resolved_usd=cost_per_res,
                total_rchar_bytes=sum(x.io.rchar_bytes for x in items),
                total_wchar_bytes=sum(x.io.wchar_bytes for x in items),
                total_syscr_ops=sum(x.io.syscr_ops for x in items),
                total_syscw_ops=sum(x.io.syscw_ops for x in items),
                total_workspace_bytes_copied=sum(x.io.workspace_bytes_copied for x in items),
                total_hardlinked_files=sum(x.io.workspace_hardlinked_files for x in items),
                mean_cpu_ms=round(mean_cpu, 2),
            )
        )
    return aggregates


def format_score_table(report: BenchmarkReport) -> str:
    """Format a BenchmarkReport into a GitHub-flavored Markdown score table."""
    lines = [
        f"# SWE-bench Pro V2 Visor Subset Benchmark Score Table",
        "",
        f"- **Revision**: `{report.benchmark_revision}` (`{report.benchmark_version}`)",
        f"- **Subset Manifest SHA-256**: `{report.manifest_digest[:16]}...`",
        f"- **Repository Ballast Scale**: `{report.repo_scale_name}` "
        f"({report.repo_asset_files} files, {report.repo_asset_bytes / (1024 * 1024):.2f} MiB)",
        f"- **Host**: `{report.hardware_info.get('os')} {report.hardware_info.get('release')}` / "
        f"`{report.hardware_info.get('cpu_model')}` ({report.hardware_info.get('cpu_count')} vCPU)",
        "",
        "| Agent | Profile | Resolved | Rate | Tampered | Mean Wall (ms) | Mean Setup (ms) | Snapshot Diff (ms) | Bytes Copied (MiB) | Hardlinked Files | `/proc/self/io` wchar (MiB) | Total Cost (USD) | Cost/Resolved (USD) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for a in report.aggregates:
        copied_mib = a.total_workspace_bytes_copied / (1024 * 1024)
        wchar_mib = a.total_wchar_bytes / (1024 * 1024)
        cpr = f"${a.cost_per_resolved_usd:.5f}" if a.cost_per_resolved_usd is not None else "N/A"
        lines.append(
            f"| `{a.agent_name}` | `{a.storage_profile.value}` | "
            f"{a.resolved_tasks}/{a.total_tasks} | {a.resolution_rate * 100:.1f}% | "
            f"{a.tampered_tasks} | {a.mean_wall_time_ms:.1f} | {a.mean_setup_ms:.1f} | "
            f"{a.mean_snapshot_diff_ms:.2f} | {copied_mib:.2f} | {a.total_hardlinked_files} | "
            f"{wchar_mib:.2f} | ${a.total_cost_usd:.5f} | {cpr} |"
        )
    return "\n".join(lines) + "\n"
