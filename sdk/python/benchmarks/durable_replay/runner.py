"""Reproducible cross-process crash-replay benchmark runner for HIP-1348 (#78/#85).

Launches real OS subprocesses (``subprocess.run([sys.executable, "-m",
"benchmarks.durable_replay.worker", ...])``) that hard-crash via ``os._exit(99)``
at every workflow step and every crash window, then spawns fresh OS processes to
resume from persisted disk checkpoints and verifies exact side-effect counts in
an independent persisted SQLite WAL ledger.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .profiler import collect_host_metadata, summarize_latencies
from .schemas import (
    CRASH_EXIT_CODE,
    BackendKind,
    CrashMode,
    CrashPlan,
    RunStatus,
    StepSpec,
    hash_payload,
)
from .side_effect_ledger import ExternalSideEffectLedger
from .stores import create_checkpoint_store


SDK_PYTHON_ROOT = Path(__file__).resolve().parents[2]


def build_default_workflow(num_steps: int = 6) -> List[StepSpec]:
    """Build a deterministic multi-step agent workflow matching Hanzo task stages."""
    canonical_stages = [
        ("step_1_plan", "kai_plan_task", {"stage": "plan", "priority": "high"}),
        ("step_2_fetch", "vfs_fetch_context", {"stage": "fetch", "files": 12}),
        ("step_3_patch", "zen_generate_patch", {"stage": "patch", "max_tokens": 2048}),
        ("step_4_verify", "visor_run_tests", {"stage": "verify", "suite": "unit"}),
        ("step_5_commit", "s3_commit_artifact", {"stage": "commit", "immutable": True}),
        ("step_6_notify", "cloud_publish_summary", {"stage": "publish", "billing": True}),
    ]
    if num_steps <= len(canonical_stages):
        selected = canonical_stages[:num_steps]
    else:
        selected = list(canonical_stages)
        for idx in range(len(canonical_stages) + 1, num_steps + 1):
            selected.append(
                (f"step_{idx}_extra", "generic_activity", {"stage": f"extra_{idx}"})
            )
    return [
        StepSpec(step_id=sid, activity_type=atype, payload=payload, round_idx=0)
        for sid, atype, payload in selected
    ]


@dataclass
class SubprocessOutcome:
    """Result from invoking one OS-isolated worker subprocess."""

    returncode: int
    wall_ms: float
    crashed_as_expected: bool
    result: Optional[Dict[str, Any]]
    stderr: str


def invoke_worker_subprocess(
    *,
    work_dir: Path,
    backend: BackendKind,
    store_path: Optional[Path],
    external_ledger_path: Path,
    org: str,
    workflow_id: str,
    run_id: str,
    actor_id: str,
    steps: Sequence[StepSpec],
    crash_plan: Optional[CrashPlan] = None,
    delivered_signals: Optional[Dict[str, Dict[str, Any]]] = None,
    enable_agent_workflow: bool = False,
    tag: str = "attempt",
) -> SubprocessOutcome:
    """Spawn a real OS subprocess executing one stateless workflow activation."""
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = work_dir / f"{run_id}_{tag}_config.json"
    res_path = work_dir / f"{run_id}_{tag}_result.json"

    cfg_payload = {
        "backend": backend.value,
        "store_path": str(store_path.resolve()) if store_path is not None else None,
        "external_ledger_path": str(external_ledger_path.resolve()),
        "result_path": str(res_path.resolve()),
        "org": org,
        "workflow_id": workflow_id,
        "run_id": run_id,
        "actor_id": actor_id,
        "steps": [s.to_dict() for s in steps],
        "crash_plan": (crash_plan or CrashPlan()).to_dict(),
        "delivered_signals": delivered_signals or {},
        "enable_agent_workflow": enable_agent_workflow,
    }
    cfg_path.write_text(json.dumps(cfg_payload, indent=2), encoding="utf-8")

    env = dict(os.environ)
    existing_pp = env.get("PYTHONPATH", "")
    sdk_str = str(SDK_PYTHON_ROOT)
    env["PYTHONPATH"] = f"{sdk_str}{os.pathsep}{existing_pp}" if existing_pp else sdk_str

    t0 = time.perf_counter()
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.durable_replay.worker",
            "--config",
            str(cfg_path),
        ],
        cwd=str(SDK_PYTHON_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    wall_ms = (time.perf_counter() - t0) * 1000.0

    parsed_result: Optional[Dict[str, Any]] = None
    if res_path.exists():
        parsed_result = json.loads(res_path.read_text(encoding="utf-8"))

    return SubprocessOutcome(
        returncode=proc.returncode,
        wall_ms=wall_ms,
        crashed_as_expected=(proc.returncode == CRASH_EXIT_CODE),
        result=parsed_result,
        stderr=proc.stderr.strip(),
    )


def _store_path_for(base_dir: Path, backend: BackendKind, label: str) -> Optional[Path]:
    if backend == BackendKind.SQLITE_WAL:
        return (base_dir / f"{label}_checkpoints.sqlite").resolve()
    if backend == BackendKind.ATOMIC_JSONL:
        return (base_dir / f"{label}_jsonl_store").resolve()
    return None


def run_crash_recovery_trial(
    *,
    work_dir: Path,
    backend: BackendKind,
    steps: Sequence[StepSpec],
    crash_step_index: int,
    crash_mode: CrashMode,
    enable_idempotency_guard: bool,
    trial_id: str,
    org: str = "hanzo-eval",
    actor_id: str = "agent-worker-1",
    workflow_id: str = "durable-agent-wf",
) -> Dict[str, Any]:
    """Run one 2-process crash-and-resume trial and verify external side-effect counts."""
    trial_dir = (work_dir / trial_id).resolve()
    trial_dir.mkdir(parents=True, exist_ok=True)
    store_path = _store_path_for(trial_dir, backend, trial_id)
    ledger_path = (trial_dir / f"{trial_id}_external_effects.sqlite").resolve()

    # Process 1: runs until injected os._exit(99) at (crash_step_index, crash_mode).
    crash_out = invoke_worker_subprocess(
        work_dir=trial_dir,
        backend=backend,
        store_path=store_path,
        external_ledger_path=ledger_path,
        org=org,
        workflow_id=workflow_id,
        run_id=trial_id,
        actor_id=actor_id,
        steps=steps,
        crash_plan=CrashPlan(
            target_step_index=crash_step_index,
            mode=crash_mode,
            enable_idempotency_guard=enable_idempotency_guard,
        ),
        tag="proc1_crash",
    )

    # Process 2: brand-new OS process resumes from disk and runs to completion.
    resume_out = invoke_worker_subprocess(
        work_dir=trial_dir,
        backend=backend,
        store_path=store_path,
        external_ledger_path=ledger_path,
        org=org,
        workflow_id=workflow_id,
        run_id=trial_id,
        actor_id=actor_id,
        steps=steps,
        crash_plan=CrashPlan(
            target_step_index=-1,
            mode=CrashMode.NONE,
            enable_idempotency_guard=enable_idempotency_guard,
        ),
        tag="proc2_resume",
    )

    ledger = ExternalSideEffectLedger(ledger_path)
    try:
        stats = ledger.get_invocation_stats(org, trial_id)
    finally:
        ledger.close()

    resume_tel = (resume_out.result or {}).get("telemetry", {})
    completed = (
        resume_out.returncode == 0
        and resume_tel.get("status") == RunStatus.COMPLETED.value
    )
    expected_steps = [s.step_id for s in steps]
    applied_by_step = stats["applied_by_step"]
    exactly_once_all_steps = completed and all(
        applied_by_step.get(sid, 0) == 1 for sid in expected_steps
    )

    return {
        "trial_id": trial_id,
        "backend": backend.value,
        "crash_step_index": crash_step_index,
        "crash_step_id": steps[crash_step_index].step_id,
        "crash_mode": crash_mode.value,
        "enable_idempotency_guard": enable_idempotency_guard,
        "proc1_returncode": crash_out.returncode,
        "proc1_crashed_hard": crash_out.crashed_as_expected,
        "proc1_wall_ms": round(crash_out.wall_ms, 4),
        "proc2_returncode": resume_out.returncode,
        "proc2_wall_ms": round(resume_out.wall_ms, 4),
        "completed": completed,
        "exactly_once_all_steps": exactly_once_all_steps,
        "duplicate_side_effects": stats["duplicate_side_effects"],
        "total_applied_effects": stats["total_applied"],
        "total_invocations": stats["total_invocations"],
        "external_dedup_hits": stats["total_deduplicated"],
        "distinct_os_pids": stats["distinct_pids"],
        "applied_by_step": applied_by_step,
        "resume_replayed_steps": int(resume_tel.get("replayed_steps", 0)),
        "resume_executed_steps": int(resume_tel.get("executed_steps", 0)),
        "resume_replay_load_ms": float(resume_tel.get("replay_load_ms", 0.0)),
        "resume_checkpoint_write_ms": list(
            resume_tel.get("checkpoint_write_ms", [])
        ),
        "resume_proc_io_wchar": int(resume_tel.get("proc_io_wchar", 0)),
        "resume_proc_io_write_bytes": int(resume_tel.get("proc_io_write_bytes", 0)),
        "store_bytes": int((resume_out.result or {}).get("store_bytes", 0)),
    }


def run_multi_crash_gauntlet(
    *,
    work_dir: Path,
    backend: BackendKind,
    steps: Sequence[StepSpec],
    enable_idempotency_guard: bool = True,
    org: str = "hanzo-eval",
    actor_id: str = "agent-gauntlet",
    workflow_id: str = "durable-gauntlet-wf",
) -> Dict[str, Any]:
    """Crash a SINGLE workflow run at EVERY step 0..M-1 across M+1 distinct OS PIDs.

    Alternates crash modes across steps so the same run survives
    ``before_activity_commit``, ``after_effect_before_ack``, and ``after_step_commit``
    before completing on attempt M+1 with every step applied exactly once.
    """
    run_id = f"gauntlet_{backend.value}"
    gdir = (work_dir / run_id).resolve()
    gdir.mkdir(parents=True, exist_ok=True)
    store_path = _store_path_for(gdir, backend, run_id)
    ledger_path = (gdir / f"{run_id}_external_effects.sqlite").resolve()

    mode_cycle = [
        CrashMode.BEFORE_ACTIVITY_COMMIT,
        CrashMode.AFTER_EFFECT_BEFORE_ACK,
        CrashMode.AFTER_STEP_COMMIT,
    ]
    attempts: List[Dict[str, Any]] = []

    for idx in range(len(steps)):
        mode = mode_cycle[idx % len(mode_cycle)]
        out = invoke_worker_subprocess(
            work_dir=gdir,
            backend=backend,
            store_path=store_path,
            external_ledger_path=ledger_path,
            org=org,
            workflow_id=workflow_id,
            run_id=run_id,
            actor_id=actor_id,
            steps=steps,
            crash_plan=CrashPlan(
                target_step_index=idx,
                mode=mode,
                enable_idempotency_guard=enable_idempotency_guard,
            ),
            tag=f"crash_step_{idx}",
        )
        attempts.append(
            {
                "attempt": idx + 1,
                "target_step_index": idx,
                "crash_mode": mode.value,
                "returncode": out.returncode,
                "crashed_hard": out.crashed_as_expected,
                "wall_ms": round(out.wall_ms, 4),
            }
        )

    final_out = invoke_worker_subprocess(
        work_dir=gdir,
        backend=backend,
        store_path=store_path,
        external_ledger_path=ledger_path,
        org=org,
        workflow_id=workflow_id,
        run_id=run_id,
        actor_id=actor_id,
        steps=steps,
        crash_plan=CrashPlan(
            target_step_index=-1,
            mode=CrashMode.NONE,
            enable_idempotency_guard=enable_idempotency_guard,
        ),
        tag="final_resume",
    )

    ledger = ExternalSideEffectLedger(ledger_path)
    try:
        stats = ledger.get_invocation_stats(org, run_id)
    finally:
        ledger.close()

    final_tel = (final_out.result or {}).get("telemetry", {})
    expected_steps = [s.step_id for s in steps]
    completed = (
        final_out.returncode == 0
        and final_tel.get("status") == RunStatus.COMPLETED.value
    )
    exactly_once = completed and all(
        stats["applied_by_step"].get(sid, 0) == 1 for sid in expected_steps
    )
    return {
        "run_id": run_id,
        "backend": backend.value,
        "step_count": len(steps),
        "total_crashes_injected": len(steps),
        "all_crashes_hard_exit_99": all(a["crashed_hard"] for a in attempts),
        "completed": completed,
        "exactly_once_all_steps": exactly_once,
        "duplicate_side_effects": stats["duplicate_side_effects"],
        "distinct_os_pids_count": len(stats["distinct_pids"]),
        "distinct_os_pids": stats["distinct_pids"],
        "applied_by_step": stats["applied_by_step"],
        "external_dedup_hits": stats["total_deduplicated"],
        "final_replayed_steps": int(final_tel.get("replayed_steps", 0)),
        "store_bytes": int((final_out.result or {}).get("store_bytes", 0)),
        "attempts": attempts,
    }


def run_nondeterminism_divergence_suite(
    *,
    work_dir: Path,
    steps: Sequence[StepSpec],
) -> Dict[str, Any]:
    """Negative Regime #2: verify fail-closed NonDeterministicReplayError on code mutation."""
    cases: List[Dict[str, Any]] = []
    if len(steps) < 3:
        raise ValueError("Non-determinism suite requires at least 3 workflow steps")

    reordered = [steps[1], steps[0], *steps[2:]]
    mutated_type = list(steps)
    mutated_type[1] = StepSpec(
        step_id=steps[1].step_id,
        activity_type="mutated_incompatible_activity",
        payload=steps[1].payload,
        round_idx=steps[1].round_idx,
    )

    mutations = [
        ("reorder_steps", reordered),
        ("change_activity_type", mutated_type),
    ]

    for backend in (BackendKind.SQLITE_WAL, BackendKind.ATOMIC_JSONL):
        for mut_name, mutated_steps in mutations:
            run_id = f"nondet_{backend.value}_{mut_name}"
            cdir = (work_dir / run_id).resolve()
            cdir.mkdir(parents=True, exist_ok=True)
            store_path = _store_path_for(cdir, backend, run_id)
            ledger_path = (cdir / f"{run_id}_external_effects.sqlite").resolve()

            # Crash after step 2 (index 2) commits so steps 0..2 are in journal.
            crash_out = invoke_worker_subprocess(
                work_dir=cdir,
                backend=backend,
                store_path=store_path,
                external_ledger_path=ledger_path,
                org="hanzo-eval",
                workflow_id="nondet-wf",
                run_id=run_id,
                actor_id="agent-nondet",
                steps=steps,
                crash_plan=CrashPlan(
                    target_step_index=2,
                    mode=CrashMode.AFTER_STEP_COMMIT,
                    enable_idempotency_guard=True,
                ),
                tag="proc1_crash",
            )

            ledger = ExternalSideEffectLedger(ledger_path)
            try:
                before_invocations = ledger.get_invocation_stats("hanzo-eval", run_id)[
                    "total_invocations"
                ]
            finally:
                ledger.close()

            # Replay with mutated workflow definition -> MUST fail closed!
            replay_out = invoke_worker_subprocess(
                work_dir=cdir,
                backend=backend,
                store_path=store_path,
                external_ledger_path=ledger_path,
                org="hanzo-eval",
                workflow_id="nondet-wf",
                run_id=run_id,
                actor_id="agent-nondet",
                steps=mutated_steps,
                crash_plan=CrashPlan(mode=CrashMode.NONE),
                tag="proc2_mutated",
            )

            ledger = ExternalSideEffectLedger(ledger_path)
            try:
                after_invocations = ledger.get_invocation_stats("hanzo-eval", run_id)[
                    "total_invocations"
                ]
            finally:
                ledger.close()

            detected = (
                replay_out.returncode == 2
                and (replay_out.result or {}).get("error_type")
                == "NonDeterministicReplayError"
            )
            zero_corrupted_writes = after_invocations == before_invocations
            cases.append(
                {
                    "backend": backend.value,
                    "mutation": mut_name,
                    "proc1_crashed_hard": crash_out.crashed_as_expected,
                    "detected_fail_closed": detected,
                    "error_type": (replay_out.result or {}).get("error_type", ""),
                    "zero_corrupted_writes": zero_corrupted_writes,
                }
            )

    return {
        "total_cases": len(cases),
        "detected_count": sum(1 for c in cases if c["detected_fail_closed"]),
        "zero_corruption_count": sum(1 for c in cases if c["zero_corrupted_writes"]),
        "cases": cases,
    }


def run_trivial_overhead_crossover(
    *,
    work_dir: Path,
    trivial_runs: int = 15,
    six_step_runs: int = 5,
) -> Dict[str, Any]:
    """Negative Regime #3: measure persistence fsync overhead on no-crash 1-step vs 6-step workflows."""
    from .engine import DurableReplayEngine

    one_step = build_default_workflow(1)
    six_step = build_default_workflow(6)
    results: Dict[str, Any] = {}

    for label, steps_list, n_runs in (
        ("trivial_1_step", one_step, trivial_runs),
        ("standard_6_step", six_step, six_step_runs),
    ):
        cohort_stats: Dict[str, Any] = {}
        for backend in (
            BackendKind.EPHEMERAL_BASELINE,
            BackendKind.SQLITE_WAL,
            BackendKind.ATOMIC_JSONL,
        ):
            bdir = (work_dir / f"overhead_{label}_{backend.value}").resolve()
            bdir.mkdir(parents=True, exist_ok=True)
            store_path = _store_path_for(bdir, backend, f"{label}_{backend.value}")
            ledger_path = (bdir / f"{label}_{backend.value}_effects.sqlite").resolve()

            store = create_checkpoint_store(backend, store_path)
            ledger = ExternalSideEffectLedger(ledger_path)
            engine = DurableReplayEngine(store, external_ledger=ledger)

            wall_samples: List[float] = []
            write_samples: List[float] = []
            wchar_total = 0
            write_bytes_total = 0
            try:
                for i in range(n_runs):
                    tel = engine.activate(
                        org="hanzo-overhead",
                        workflow_id=f"wf-{label}",
                        run_id=f"run-{label}-{i}",
                        actor_id="actor-bench",
                        steps=steps_list,
                    )
                    wall_samples.append(tel.wall_ms)
                    write_samples.extend(tel.checkpoint_write_ms)
                    wchar_total += tel.proc_io_wchar
                    write_bytes_total += tel.proc_io_write_bytes
                disk_bytes = store.storage_bytes()
            finally:
                store.close()
                ledger.close()

            cohort_stats[backend.value] = {
                "runs": n_runs,
                "steps_per_run": len(steps_list),
                "wall_latency": summarize_latencies(wall_samples),
                "checkpoint_write_latency": summarize_latencies(write_samples),
                "mean_proc_io_wchar": round(wchar_total / max(1, n_runs), 1),
                "mean_proc_io_write_bytes": round(write_bytes_total / max(1, n_runs), 1),
                "total_store_bytes": disk_bytes,
                "bytes_per_run": round(disk_bytes / max(1, n_runs), 1),
            }

        base_mean = cohort_stats[BackendKind.EPHEMERAL_BASELINE.value]["wall_latency"][
            "mean_ms"
        ]
        for bname in (BackendKind.SQLITE_WAL.value, BackendKind.ATOMIC_JSONL.value):
            bmean = cohort_stats[bname]["wall_latency"]["mean_ms"]
            cohort_stats[bname]["overhead_vs_ephemeral_ms"] = round(
                bmean - base_mean, 4
            )
            cohort_stats[bname]["overhead_ratio_vs_ephemeral"] = round(
                bmean / base_mean, 2
            ) if base_mean > 0 else 0.0

        results[label] = cohort_stats

    return results


def _aggregate_cohort(trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(trials)
    completed = sum(1 for t in trials if t["completed"])
    exactly_once = sum(1 for t in trials if t["exactly_once_all_steps"])
    total_dups = sum(int(t["duplicate_side_effects"]) for t in trials)
    total_dedup_hits = sum(int(t["external_dedup_hits"]) for t in trials)
    all_crashed_hard = all(t["proc1_crashed_hard"] for t in trials)
    proc2_walls = [float(t["proc2_wall_ms"]) for t in trials]
    replay_loads = [float(t["resume_replay_load_ms"]) for t in trials]
    ckpt_writes: List[float] = []
    for t in trials:
        ckpt_writes.extend(float(x) for x in t["resume_checkpoint_write_ms"])
    store_bytes = [int(t["store_bytes"]) for t in trials]
    io_wchars = [int(t["resume_proc_io_wchar"]) for t in trials]

    return {
        "trials": total,
        "all_crashed_hard_exit_99": all_crashed_hard,
        "resumption_completed": completed,
        "resumption_completion_rate": round(completed / max(1, total), 4),
        "exactly_once_trials": exactly_once,
        "exactly_once_rate": round(exactly_once / max(1, total), 4),
        "total_duplicate_side_effects": total_dups,
        "mean_duplicate_side_effects_per_trial": round(total_dups / max(1, total), 4),
        "total_external_dedup_hits": total_dedup_hits,
        "resume_subprocess_wall_ms": summarize_latencies(proc2_walls),
        "replay_load_ms": summarize_latencies(replay_loads),
        "checkpoint_write_ms": summarize_latencies(ckpt_writes),
        "mean_store_bytes": round(sum(store_bytes) / max(1, total), 1),
        "mean_resume_proc_io_wchar": round(sum(io_wchars) / max(1, total), 1),
    }


def run_full_benchmark(
    *,
    num_steps: int = 6,
    trivial_runs: int = 15,
    work_dir: Optional[Path | str] = None,
    output_dir: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Execute all HIP-1348 durable replay experiments and optionally write artifacts."""
    steps = build_default_workflow(num_steps)
    suite_digest = hash_payload(
        {
            "benchmark": "hip-1348-durable-replay",
            "num_steps": num_steps,
            "trivial_runs": trivial_runs,
            "steps": [s.to_dict() for s in steps],
        }
    )

    tmp_ctx: Optional[tempfile.TemporaryDirectory[str]] = None
    if work_dir is None:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="hanzo_durable_replay_")
        root = Path(tmp_ctx.name).resolve()
    else:
        root = Path(work_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)

    try:
        # Experiment A: Primary crash-at-every-step matrix (real os._exit(99) subprocesses).
        crash_modes = [
            CrashMode.BEFORE_ACTIVITY_COMMIT,
            CrashMode.AFTER_EFFECT_BEFORE_ACK,
            CrashMode.AFTER_STEP_COMMIT,
        ]
        primary_configs = [
            ("ephemeral_baseline_no_idempotency", BackendKind.EPHEMERAL_BASELINE, False),
            ("ephemeral_baseline_with_idempotency", BackendKind.EPHEMERAL_BASELINE, True),
            ("sqlite_wal_durable", BackendKind.SQLITE_WAL, True),
            ("atomic_jsonl_durable", BackendKind.ATOMIC_JSONL, True),
        ]

        primary_trials: Dict[str, List[Dict[str, Any]]] = {}
        primary_summary: Dict[str, Dict[str, Any]] = {}
        for cohort_name, backend, idem_guard in primary_configs:
            trials: List[Dict[str, Any]] = []
            for mode in crash_modes:
                for step_idx in range(len(steps)):
                    trial_id = f"{cohort_name}_{mode.value}_s{step_idx}"
                    trials.append(
                        run_crash_recovery_trial(
                            work_dir=root / "exp_a",
                            backend=backend,
                            steps=steps,
                            crash_step_index=step_idx,
                            crash_mode=mode,
                            enable_idempotency_guard=idem_guard,
                            trial_id=trial_id,
                        )
                    )
            primary_trials[cohort_name] = trials
            primary_summary[cohort_name] = _aggregate_cohort(trials)

        # Experiment A2: Multi-crash gauntlet on a SINGLE run across M+1 OS PIDs.
        gauntlet = {
            BackendKind.SQLITE_WAL.value: run_multi_crash_gauntlet(
                work_dir=root / "exp_a2",
                backend=BackendKind.SQLITE_WAL,
                steps=steps,
                enable_idempotency_guard=True,
            ),
            BackendKind.ATOMIC_JSONL.value: run_multi_crash_gauntlet(
                work_dir=root / "exp_a2",
                backend=BackendKind.ATOMIC_JSONL,
                steps=steps,
                enable_idempotency_guard=True,
            ),
        }

        # Experiment B (Negative Regime 1): Persisted checkpoints WITHOUT idempotency_key
        # under AFTER_EFFECT_BEFORE_ACK crashes -> proves duplicate side effects occur!
        neg_no_idem_trials: Dict[str, List[Dict[str, Any]]] = {}
        neg_no_idem_summary: Dict[str, Dict[str, Any]] = {}
        for backend in (BackendKind.SQLITE_WAL, BackendKind.ATOMIC_JSONL):
            cohort_key = f"{backend.value}_no_idempotency_after_effect_crash"
            trials = [
                run_crash_recovery_trial(
                    work_dir=root / "exp_b",
                    backend=backend,
                    steps=steps,
                    crash_step_index=step_idx,
                    crash_mode=CrashMode.AFTER_EFFECT_BEFORE_ACK,
                    enable_idempotency_guard=False,
                    trial_id=f"{cohort_key}_s{step_idx}",
                )
                for step_idx in range(len(steps))
            ]
            neg_no_idem_trials[cohort_key] = trials
            neg_no_idem_summary[cohort_key] = _aggregate_cohort(trials)

        # Experiment C (Negative Regime 2): Non-deterministic replay divergence detection.
        nondet_suite = run_nondeterminism_divergence_suite(
            work_dir=root / "exp_c",
            steps=steps,
        )

        # Experiment D (Negative Regime 3): Overhead crossover on trivial 1-step workflows.
        overhead_suite = run_trivial_overhead_crossover(
            work_dir=root / "exp_d",
            trivial_runs=trivial_runs,
            six_step_runs=max(5, trivial_runs // 3),
        )

        report: Dict[str, Any] = {
            "benchmark": "hip-1348-durable-workflow-replay",
            "issues": ["hanzoai/agents#78", "hanzoai/agents#85"],
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "suite_digest": suite_digest,
            "host": collect_host_metadata(),
            "workflow_steps": [s.to_dict() for s in steps],
            "experiment_a_crash_at_every_step": {
                "summary": primary_summary,
                "trials": primary_trials,
            },
            "experiment_a2_multi_crash_gauntlet": gauntlet,
            "experiment_b_negative_no_idempotency_window": {
                "summary": neg_no_idem_summary,
                "trials": neg_no_idem_trials,
            },
            "experiment_c_negative_nondeterminism": nondet_suite,
            "experiment_d_negative_overhead_crossover": overhead_suite,
        }

        if output_dir is not None:
            out_path = Path(output_dir).resolve()
            out_path.mkdir(parents=True, exist_ok=True)
            (out_path / "raw_results.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (out_path / "score_table.md").write_text(
                render_score_table_markdown(report),
                encoding="utf-8",
            )

        return report
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


def render_score_table_markdown(report: Dict[str, Any]) -> str:
    """Render publication-grade Markdown tables directly from measured benchmark results."""
    lines: List[str] = [
        "# Durable Workflow Replay Benchmark — Score Table (HIP-1348, `#78` / `#85`)",
        "",
        f"- **Generated (UTC)**: `{report['generated_at_utc']}`",
        f"- **Suite Digest**: `{report['suite_digest'][:16]}`",
        (
            f"- **Host**: `{report['host']['system']} {report['host']['release']}` "
            f"(`{report['host']['machine']}`, `{report['host']['cpu_model']}`, "
            f"`{report['host']['cpu_count']} vCPU`, Python `{report['host']['python_version']}`, "
            f"`/proc/self/io={report['host']['proc_self_io_available']}`)"
        ),
        "",
        "## 1. Primary Cross-Process Crash-at-Every-Step Matrix (`os._exit(99)` across 6 steps x 3 crash modes = 18 trials/cohort)",
        "",
        "| Cohort | Trials | Hard Crashes (`exit 99`) | Resumption Rate | Exactly-Once Rate | Duplicate Side Effects | External Dedup Hits | Resume Replay Load p50/p95 (ms) | Ckpt Write p50/p95 (ms) | Mean Store Footprint (KiB) | Mean Resume `/proc/self/io` wchar (KiB) |",
        "| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for name, s in report["experiment_a_crash_at_every_step"]["summary"].items():
        lines.append(
            f"| `{name}` | {s['trials']} | "
            f"{'PASS' if s['all_crashed_hard_exit_99'] else 'FAIL'} | "
            f"{s['resumption_completion_rate'] * 100:.1f}% | "
            f"{s['exactly_once_rate'] * 100:.1f}% | "
            f"{s['total_duplicate_side_effects']} | "
            f"{s['total_external_dedup_hits']} | "
            f"{s['replay_load_ms']['p50_ms']:.3f} / {s['replay_load_ms']['p95_ms']:.3f} | "
            f"{s['checkpoint_write_ms']['p50_ms']:.3f} / {s['checkpoint_write_ms']['p95_ms']:.3f} | "
            f"{s['mean_store_bytes'] / 1024.0:.2f} | "
            f"{s['mean_resume_proc_io_wchar'] / 1024.0:.2f} |"
        )

    lines.extend(
        [
            "",
            "## 2. Single-Run Multi-Crash Gauntlet (Crash at EVERY Step `1..6` on ONE `run_id` across 7 OS PIDs)",
            "",
            "| Backend | Crashes Injected | Distinct OS PIDs | Completed | Exactly-Once (`applied==1` all steps) | Duplicate Side Effects | External Dedup Hits | Final Replayed Steps | Store Footprint (KiB) |",
            "| --- | ---: | ---: | :---: | :---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for bname, g in report["experiment_a2_multi_crash_gauntlet"].items():
        lines.append(
            f"| `{bname}` | {g['total_crashes_injected']} | "
            f"{g['distinct_os_pids_count']} | "
            f"{'YES' if g['completed'] else 'NO'} | "
            f"{'YES' if g['exactly_once_all_steps'] else 'NO'} | "
            f"{g['duplicate_side_effects']} | "
            f"{g['external_dedup_hits']} | "
            f"{g['final_replayed_steps']} | "
            f"{g['store_bytes'] / 1024.0:.2f} |"
        )

    lines.extend(
        [
            "",
            "## 3. Negative Regime #1 — Persisted Checkpoints WITHOUT Activity `idempotency_key` (`after_effect_before_ack` crashes)",
            "",
            "| Cohort | Trials | Resumption Rate | Exactly-Once Rate | Duplicate Side Effects | Mean Duplicates / Trial |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for cname, s in report["experiment_b_negative_no_idempotency_window"]["summary"].items():
        lines.append(
            f"| `{cname}` | {s['trials']} | "
            f"{s['resumption_completion_rate'] * 100:.1f}% | "
            f"{s['exactly_once_rate'] * 100:.1f}% | "
            f"{s['total_duplicate_side_effects']} | "
            f"{s['mean_duplicate_side_effects_per_trial']:.2f} |"
        )

    nd = report["experiment_c_negative_nondeterminism"]
    lines.extend(
        [
            "",
            "## 4. Negative Regime #2 — Non-Deterministic Replay Divergence Detection (Fail-Closed)",
            "",
            f"- **Detected Fail-Closed**: `{nd['detected_count']}/{nd['total_cases']}`",
            f"- **Zero Corrupted External Writes**: `{nd['zero_corruption_count']}/{nd['total_cases']}`",
            "",
            "| Backend | Mutation | Hard Crash (`exit 99`) | Fail-Closed Error | Zero Corrupted Writes |",
            "| --- | --- | :---: | --- | :---: |",
        ]
    )
    for c in nd["cases"]:
        lines.append(
            f"| `{c['backend']}` | `{c['mutation']}` | "
            f"{'YES' if c['proc1_crashed_hard'] else 'NO'} | "
            f"`{c['error_type']}` | "
            f"{'YES' if c['zero_corrupted_writes'] else 'NO'} |"
        )

    lines.extend(
        [
            "",
            "## 5. Negative Regime #3 — Persistence Latency Overhead Crossover (No-Crash Happy Path)",
            "",
            "| Workflow Regime | Backend | Runs | Mean Wall (ms) | Wall p50/p95 (ms) | Ckpt Write p50/p95 (ms) | Overhead vs Ephemeral (ms) | Ratio vs Ephemeral | Mean `/proc/self/io` wchar (KiB) | Store Footprint / Run (KiB) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for regime_name, backends in report["experiment_d_negative_overhead_crossover"].items():
        for bname, stats in backends.items():
            ov_ms = stats.get("overhead_vs_ephemeral_ms", 0.0)
            ov_ratio = stats.get("overhead_ratio_vs_ephemeral", 1.0)
            lines.append(
                f"| `{regime_name}` | `{bname}` | {stats['runs']} | "
                f"{stats['wall_latency']['mean_ms']:.3f} | "
                f"{stats['wall_latency']['p50_ms']:.3f} / {stats['wall_latency']['p95_ms']:.3f} | "
                f"{stats['checkpoint_write_latency']['p50_ms']:.3f} / {stats['checkpoint_write_latency']['p95_ms']:.3f} | "
                f"{ov_ms:+.3f} | "
                f"{ov_ratio:.2f}x | "
                f"{stats['mean_proc_io_wchar'] / 1024.0:.2f} | "
                f"{stats['bytes_per_run'] / 1024.0:.2f} |"
            )

    lines.append("")
    return "\n".join(lines)
