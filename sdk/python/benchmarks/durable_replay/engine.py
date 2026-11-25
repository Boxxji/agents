"""Stateless event-triggered durable workflow replay engine.

Reproduces Temporal Go SDK durable-execution semantics inside Hanzo's
event-triggered lifecycle (``hanzo_agents.agent_workflow.AgentWorkflow`` +
``hanzo_agents.execution_context.ExecutionContext``) without any always-on
worker daemon or polling process.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from hanzo_agents.agent_workflow import AgentWorkflow
from hanzo_agents.execution_context import (
    ExecutionContext,
    reset_execution_context,
    set_execution_context,
)

from .profiler import diff_proc_io, read_proc_io
from .schemas import (
    CRASH_EXIT_CODE,
    ActivationTelemetry,
    CrashMode,
    CrashPlan,
    EventType,
    JournalEvent,
    LedgerRecord,
    NonDeterministicReplayError,
    RunStatus,
    StepSnapshot,
    StepSpec,
    TenantIsolationError,
    WorkflowSuspendedSignal,
    compute_idempotency_key,
    hash_payload,
    validate_identifier,
)
from .side_effect_ledger import ExternalSideEffectLedger
from .stores import CheckpointStore


ActivityExecutor = Callable[
    [StepSpec, str, ExecutionContext],
    Tuple[Dict[str, Any], bool],
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_execution_id() -> str:
    return f"exec_{uuid.uuid4().hex[:16]}"


class DurableReplayEngine:
    """Stateless event-triggered durable replay engine aligned with Hanzo SDK lifecycle.

    Each call to :meth:`activate` performs exactly one event-triggered pass:
    replays already-completed steps from persisted checkpoints, executes pending
    steps with deterministic ``idempotency_key`` deduplication, persists step
    checkpoints + snapshots atomically, and exits cleanly (or suspends awaiting
    an external signal). Zero background threads or polling loops are created.
    """

    def __init__(
        self,
        store: CheckpointStore,
        *,
        external_ledger: Optional[ExternalSideEffectLedger] = None,
        agent_workflow: Optional[AgentWorkflow] = None,
    ) -> None:
        self.store = store
        self.external_ledger = external_ledger
        self.agent_workflow = agent_workflow

    def _default_activity_executor(
        self,
        *,
        org: str,
        run_id: str,
        enforce_idempotency: bool,
    ) -> ActivityExecutor:
        if self.external_ledger is None:
            raise ValueError(
                "Either external_ledger or a custom activity_executor must be provided"
            )
        ledger = self.external_ledger

        def _exec(
            step: StepSpec,
            idempotency_key: str,
            exec_ctx: ExecutionContext,
        ) -> Tuple[Dict[str, Any], bool]:
            payload = dict(step.payload)
            payload["_execution_id"] = exec_ctx.execution_id
            payload["_workflow_id"] = exec_ctx.workflow_id
            # Hash excludes volatile per-attempt execution_id so input_hash remains deterministic.
            return ledger.apply_side_effect(
                org=org,
                run_id=run_id,
                step_id=step.step_id,
                activity_type=step.activity_type,
                idempotency_key=idempotency_key,
                payload=step.payload,
                enforce_idempotency=enforce_idempotency,
            )

        return _exec

    def _append_timed(
        self,
        event: JournalEvent,
        checkpoint_write_ms: List[float],
        *,
        snapshot: Optional[StepSnapshot] = None,
        ledger_record: Optional[LedgerRecord] = None,
    ) -> None:
        t0 = time.perf_counter()
        self.store.append_event(
            event,
            snapshot=snapshot,
            ledger_record=ledger_record,
        )
        checkpoint_write_ms.append((time.perf_counter() - t0) * 1000.0)

    def _invoke_with_agent_workflow(
        self,
        executor: ActivityExecutor,
        step: StepSpec,
        idempotency_key: str,
        exec_ctx: ExecutionContext,
    ) -> Tuple[Dict[str, Any], bool]:
        if self.agent_workflow is None:
            return executor(step, idempotency_key, exec_ctx)

        aw = self.agent_workflow

        def _tracked_step(
            step_id: str,
            activity_type: str,
            idem_key: str,
        ) -> Tuple[Dict[str, Any], bool]:
            return executor(step, idem_key, exec_ctx)

        _tracked_step.__name__ = f"{step.activity_type}__{step.step_id}"

        async def _run_tracked() -> Tuple[Dict[str, Any], bool]:
            token = set_execution_context(exec_ctx)
            prev = getattr(aw.agent, "_current_execution_context", None)
            aw.agent._current_execution_context = exec_ctx
            try:
                return await aw.execute_with_tracking(
                    _tracked_step,
                    (step.step_id, step.activity_type, idempotency_key),
                    {},
                )
            finally:
                aw.agent._current_execution_context = prev
                reset_execution_context(token)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # Already inside an async caller: execute step synchronously while
            # still propagating ExecutionContext so nested async loops never clash.
            token = set_execution_context(exec_ctx)
            try:
                return executor(step, idempotency_key, exec_ctx)
            finally:
                reset_execution_context(token)

        res = asyncio.run(_run_tracked())
        if inspect.isawaitable(res):  # pragma: no cover - defensive guard
            return asyncio.run(res)
        return res

    async def activate_async(
        self,
        *,
        org: str,
        workflow_id: str,
        run_id: str,
        actor_id: str,
        steps: Sequence[StepSpec],
        delivered_signals: Optional[Dict[str, Dict[str, Any]]] = None,
        crash_plan: Optional[CrashPlan] = None,
        activity_executor: Optional[ActivityExecutor] = None,
    ) -> ActivationTelemetry:
        """Async-friendly event-triggered activation (integrates cleanly with ``AgentWorkflow``)."""
        return await asyncio.to_thread(
            self.activate,
            org=org,
            workflow_id=workflow_id,
            run_id=run_id,
            actor_id=actor_id,
            steps=steps,
            delivered_signals=delivered_signals,
            crash_plan=crash_plan,
            activity_executor=activity_executor,
        )

    def activate(
        self,
        *,
        org: str,
        workflow_id: str,
        run_id: str,
        actor_id: str,
        steps: Sequence[StepSpec],
        delivered_signals: Optional[Dict[str, Dict[str, Any]]] = None,
        crash_plan: Optional[CrashPlan] = None,
        activity_executor: Optional[ActivityExecutor] = None,
    ) -> ActivationTelemetry:
        """Run one deterministic stateless workflow activation pass.

        Raises:
            InvalidIdentifierError: If any tenant/run/step identifier is invalid.
            TenantIsolationError: If ``actor_id`` or ``workflow_id`` conflicts
                with persisted ownership metadata for ``(org, run_id)``.
            NonDeterministicReplayError: Fail-closed if the workflow definition
                diverged from persisted ``ActivityScheduled`` / ``ActivityCompleted``
                journal history.
        """
        org = validate_identifier("org", org)
        workflow_id = validate_identifier("workflow_id", workflow_id)
        run_id = validate_identifier("run_id", run_id)
        actor_id = validate_identifier("actor_id", actor_id)

        plan = crash_plan or CrashPlan()
        pid = os.getpid()
        execution_id = _make_execution_id()
        signals_in = dict(delivered_signals or {})

        wall_start = time.perf_counter()
        io_start = read_proc_io()
        checkpoint_write_ms: List[float] = []

        load_start = time.perf_counter()
        meta = self.store.get_run_metadata(org, run_id)
        history = self.store.load_events(org, run_id)
        replay_load_ms = (time.perf_counter() - load_start) * 1000.0

        # Tenant & actor ownership enforcement before any replay or write.
        if meta is not None:
            if meta.get("actor_id") != actor_id:
                raise TenantIsolationError(
                    f"Run {run_id!r} in org {org!r} is owned by actor "
                    f"{meta.get('actor_id')!r}, not {actor_id!r}"
                )
            if meta.get("workflow_id") != workflow_id:
                raise TenantIsolationError(
                    f"Run {run_id!r} in org {org!r} belongs to workflow "
                    f"{meta.get('workflow_id')!r}, not {workflow_id!r}"
                )

        seq = history[-1].sequence if history else 0
        outputs: Dict[str, Dict[str, Any]] = {}
        replayed_steps = 0
        executed_steps = 0
        deduplicated_hits = 0

        # Reconstruct delivered signals from persisted history + incoming event signals.
        persisted_signals: Dict[str, Dict[str, Any]] = {}
        for ev in history:
            if ev.event_type == EventType.WORKFLOW_RESUMED:
                sig_name = str(ev.payload.get("signal_name", ""))
                if sig_name:
                    persisted_signals[sig_name] = dict(ev.payload.get("signal_payload") or {})

        # Fast-path if already completed.
        if meta is not None and meta.get("status") == RunStatus.COMPLETED.value:
            for ev in history:
                if ev.event_type == EventType.ACTIVITY_COMPLETED:
                    outputs[ev.step_id] = dict(ev.payload.get("output") or {})
                    replayed_steps += 1
            io_delta = diff_proc_io(io_start, read_proc_io())
            return ActivationTelemetry(
                org=org,
                workflow_id=workflow_id,
                run_id=run_id,
                execution_id=execution_id,
                actor_id=actor_id,
                backend=self.store.backend_kind.value,
                pid=pid,
                status=RunStatus.COMPLETED.value,
                executed_steps=0,
                replayed_steps=replayed_steps,
                deduplicated_hits=replayed_steps,
                wall_ms=(time.perf_counter() - wall_start) * 1000.0,
                replay_load_ms=replay_load_ms,
                checkpoint_write_ms=checkpoint_write_ms,
                proc_io_rchar=io_delta.rchar,
                proc_io_wchar=io_delta.wchar,
                proc_io_write_bytes=io_delta.write_bytes,
                proc_io_syscw=io_delta.syscw,
                outputs=outputs,
            )

        # Record WorkflowStarted on first activation.
        if seq == 0:
            seq += 1
            self._append_timed(
                JournalEvent(
                    sequence=seq,
                    previous_sequence=0,
                    event_type=EventType.WORKFLOW_STARTED,
                    org=org,
                    workflow_id=workflow_id,
                    run_id=run_id,
                    execution_id=execution_id,
                    actor_id=actor_id,
                    round_idx=0,
                    payload={"step_count": len(steps)},
                    pid=pid,
                    timestamp=_utc_now(),
                ),
                checkpoint_write_ms,
            )

        # Record any newly delivered external signals (event-triggered resume).
        for sig_name, sig_payload in sorted(signals_in.items()):
            validate_identifier("signal_name", sig_name)
            if sig_name not in persisted_signals:
                seq += 1
                self._append_timed(
                    JournalEvent(
                        sequence=seq,
                        previous_sequence=seq - 1,
                        event_type=EventType.WORKFLOW_RESUMED,
                        org=org,
                        workflow_id=workflow_id,
                        run_id=run_id,
                        execution_id=execution_id,
                        actor_id=actor_id,
                        round_idx=0,
                        payload={
                            "signal_name": sig_name,
                            "signal_payload": sig_payload,
                        },
                        pid=pid,
                        timestamp=_utc_now(),
                    ),
                    checkpoint_write_ms,
                )
                persisted_signals[sig_name] = dict(sig_payload)

        # Extract ordered completed activities and any trailing in-flight scheduled activity.
        completed_history: List[JournalEvent] = []
        completed_keys: set[str] = set()
        pending_scheduled: Optional[JournalEvent] = None

        for ev in history:
            if ev.event_type == EventType.ACTIVITY_COMPLETED:
                completed_history.append(ev)
                completed_keys.add(ev.idempotency_key)
                if (
                    pending_scheduled is not None
                    and pending_scheduled.idempotency_key == ev.idempotency_key
                ):
                    pending_scheduled = None
            elif ev.event_type == EventType.ACTIVITY_SCHEDULED:
                if ev.idempotency_key not in completed_keys:
                    pending_scheduled = ev

        # Build ExecutionContext aligned with Hanzo SDK conventions.
        root_ctx = ExecutionContext(
            run_id=run_id,
            execution_id=execution_id,
            agent_instance=getattr(self.agent_workflow, "agent", None),
            reasoner_name=workflow_id,
            actor_id=actor_id,
            workflow_id=workflow_id,
            root_workflow_id=workflow_id,
            registered=True,
        )

        executor = activity_executor or self._default_activity_executor(
            org=org,
            run_id=run_id,
            enforce_idempotency=plan.enable_idempotency_guard,
        )

        status_out = RunStatus.RUNNING.value
        awaiting_signal = ""
        error_text = ""

        try:
            for step_idx, step in enumerate(steps):
                validate_identifier("step_id", step.step_id)
                validate_identifier("activity_type", step.activity_type)

                # Handle optional event-driven suspension gate before step execution.
                if step.await_signal:
                    validate_identifier("await_signal", step.await_signal)
                    if step.await_signal not in persisted_signals:
                        seq += 1
                        self._append_timed(
                            JournalEvent(
                                sequence=seq,
                                previous_sequence=seq - 1,
                                event_type=EventType.WORKFLOW_SUSPENDED,
                                org=org,
                                workflow_id=workflow_id,
                                run_id=run_id,
                                execution_id=execution_id,
                                actor_id=actor_id,
                                round_idx=step.round_idx,
                                step_id=step.step_id,
                                payload={"awaiting_signal": step.await_signal},
                                pid=pid,
                                timestamp=_utc_now(),
                            ),
                            checkpoint_write_ms,
                            snapshot=StepSnapshot(
                                org=org,
                                run_id=run_id,
                                round_idx=step.round_idx,
                                step_id=f"await_{step.await_signal}",
                                sequence=seq,
                                state={
                                    "awaiting_signal": step.await_signal,
                                    "completed_steps": sorted(outputs.keys()),
                                },
                                created_at=_utc_now(),
                            ),
                        )
                        raise WorkflowSuspendedSignal(step.await_signal, step.step_id)

                idem_key = compute_idempotency_key(
                    org, run_id, step.round_idx, step.step_id
                )
                in_hash = hash_payload(step.payload)

                # 1. Deterministic replay against persisted ActivityCompleted history.
                if step_idx < len(completed_history):
                    recorded = completed_history[step_idx]
                    if (
                        recorded.step_id != step.step_id
                        or recorded.activity_type != step.activity_type
                        or recorded.idempotency_key != idem_key
                        or recorded.input_hash != in_hash
                    ):
                        raise NonDeterministicReplayError(
                            f"History divergence at step index {step_idx}: "
                            f"recorded=(step_id={recorded.step_id!r}, "
                            f"activity_type={recorded.activity_type!r}, "
                            f"input_hash={recorded.input_hash[:12]}) vs "
                            f"current=(step_id={step.step_id!r}, "
                            f"activity_type={step.activity_type!r}, "
                            f"input_hash={in_hash[:12]})"
                        )
                    outputs[step.step_id] = dict(recorded.payload.get("output") or {})
                    replayed_steps += 1
                    continue

                # 2. Check for trailing in-flight ActivityScheduled from a crashed process.
                already_scheduled = False
                if pending_scheduled is not None and step_idx == len(completed_history):
                    if (
                        pending_scheduled.step_id != step.step_id
                        or pending_scheduled.activity_type != step.activity_type
                        or pending_scheduled.idempotency_key != idem_key
                        or pending_scheduled.input_hash != in_hash
                    ):
                        raise NonDeterministicReplayError(
                            f"Pending scheduled-activity divergence at step index {step_idx}: "
                            f"recorded=(step_id={pending_scheduled.step_id!r}, "
                            f"activity_type={pending_scheduled.activity_type!r}) vs "
                            f"current=(step_id={step.step_id!r}, "
                            f"activity_type={step.activity_type!r})"
                        )
                    already_scheduled = True
                    pending_scheduled = None

                # 3. Check transactional checkpoint ledger before re-running activity.
                ledger_hit = self.store.get_ledger_record(org, idem_key)
                if ledger_hit is not None and ledger_hit.status == "completed":
                    outputs[step.step_id] = dict(ledger_hit.output)
                    replayed_steps += 1
                    deduplicated_hits += 1
                    seq += 1
                    self._append_timed(
                        JournalEvent(
                            sequence=seq,
                            previous_sequence=seq - 1,
                            event_type=EventType.ACTIVITY_COMPLETED,
                            org=org,
                            workflow_id=workflow_id,
                            run_id=run_id,
                            execution_id=execution_id,
                            actor_id=actor_id,
                            round_idx=step.round_idx,
                            step_id=step.step_id,
                            activity_type=step.activity_type,
                            idempotency_key=idem_key,
                            input_hash=in_hash,
                            payload={"output": ledger_hit.output, "deduplicated": True},
                            pid=pid,
                            timestamp=_utc_now(),
                        ),
                        checkpoint_write_ms,
                    )
                    continue

                # 4. Schedule activity durably before external invocation.
                if not already_scheduled:
                    seq += 1
                    self._append_timed(
                        JournalEvent(
                            sequence=seq,
                            previous_sequence=seq - 1,
                            event_type=EventType.ACTIVITY_SCHEDULED,
                            org=org,
                            workflow_id=workflow_id,
                            run_id=run_id,
                            execution_id=execution_id,
                            actor_id=actor_id,
                            round_idx=step.round_idx,
                            step_id=step.step_id,
                            activity_type=step.activity_type,
                            idempotency_key=idem_key,
                            input_hash=in_hash,
                            payload={"input": step.payload},
                            pid=pid,
                            timestamp=_utc_now(),
                        ),
                        checkpoint_write_ms,
                    )

                # Hard OS crash injection point 1: after ActivityScheduled, before external effect.
                if (
                    plan.mode == CrashMode.BEFORE_ACTIVITY_COMMIT
                    and plan.target_step_index == step_idx
                ):
                    os._exit(CRASH_EXIT_CODE)

                # 5. Execute the potentially side-effecting activity.
                step_ctx = root_ctx.child_context()
                step_ctx.reasoner_name = f"{step.activity_type}:{step.step_id}"
                output, deduped = self._invoke_with_agent_workflow(
                    executor, step, idem_key, step_ctx
                )
                executed_steps += 1
                if deduped:
                    deduplicated_hits += 1

                # Hard OS crash injection point 2: AFTER external side-effect committed,
                # BEFORE workflow checkpoint acknowledges ActivityCompleted.
                if (
                    plan.mode == CrashMode.AFTER_EFFECT_BEFORE_ACK
                    and plan.target_step_index == step_idx
                ):
                    os._exit(CRASH_EXIT_CODE)

                # 6. Persist ActivityCompleted + LedgerRecord + StepSnapshot atomically.
                outputs[step.step_id] = dict(output)
                now_ts = _utc_now()
                seq += 1
                self._append_timed(
                    JournalEvent(
                        sequence=seq,
                        previous_sequence=seq - 1,
                        event_type=EventType.ACTIVITY_COMPLETED,
                        org=org,
                        workflow_id=workflow_id,
                        run_id=run_id,
                        execution_id=step_ctx.execution_id,
                        actor_id=actor_id,
                        round_idx=step.round_idx,
                        step_id=step.step_id,
                        activity_type=step.activity_type,
                        idempotency_key=idem_key,
                        input_hash=in_hash,
                        payload={"output": output, "deduplicated": deduped},
                        pid=pid,
                        timestamp=now_ts,
                    ),
                    checkpoint_write_ms,
                    snapshot=StepSnapshot(
                        org=org,
                        run_id=run_id,
                        round_idx=step.round_idx,
                        step_id=step.step_id,
                        sequence=seq,
                        state={
                            "step_index": step_idx,
                            "step_id": step.step_id,
                            "completed_steps": sorted(outputs.keys()),
                        },
                        created_at=now_ts,
                    ),
                    ledger_record=LedgerRecord(
                        idempotency_key=idem_key,
                        org=org,
                        run_id=run_id,
                        round_idx=step.round_idx,
                        step_id=step.step_id,
                        activity_type=step.activity_type,
                        status="completed",
                        input_hash=in_hash,
                        output=output,
                        attempt=2 if already_scheduled else 1,
                        pid=pid,
                        completed_at=now_ts,
                    ),
                )

                # Hard OS crash injection point 3: AFTER durable step commit, mid-workflow.
                if (
                    plan.mode == CrashMode.AFTER_STEP_COMMIT
                    and plan.target_step_index == step_idx
                ):
                    os._exit(CRASH_EXIT_CODE)

            # All steps completed — persist WorkflowCompleted.
            seq += 1
            self._append_timed(
                JournalEvent(
                    sequence=seq,
                    previous_sequence=seq - 1,
                    event_type=EventType.WORKFLOW_COMPLETED,
                    org=org,
                    workflow_id=workflow_id,
                    run_id=run_id,
                    execution_id=execution_id,
                    actor_id=actor_id,
                    round_idx=steps[-1].round_idx if steps else 0,
                    payload={"completed_step_count": len(steps)},
                    pid=pid,
                    timestamp=_utc_now(),
                ),
                checkpoint_write_ms,
            )
            status_out = RunStatus.COMPLETED.value

        except WorkflowSuspendedSignal as sig:
            status_out = RunStatus.SUSPENDED.value
            awaiting_signal = sig.signal_name

        except NonDeterministicReplayError as exc:
            # Fail closed: record WorkflowFailed in journal and re-raise without executing mutated steps.
            error_text = str(exc)
            seq += 1
            try:
                self._append_timed(
                    JournalEvent(
                        sequence=seq,
                        previous_sequence=seq - 1,
                        event_type=EventType.WORKFLOW_FAILED,
                        org=org,
                        workflow_id=workflow_id,
                        run_id=run_id,
                        execution_id=execution_id,
                        actor_id=actor_id,
                        round_idx=0,
                        error=error_text,
                        pid=pid,
                        timestamp=_utc_now(),
                    ),
                    checkpoint_write_ms,
                )
            except Exception:
                pass
            raise

        except Exception as exc:
            error_text = str(exc)
            seq += 1
            try:
                self._append_timed(
                    JournalEvent(
                        sequence=seq,
                        previous_sequence=seq - 1,
                        event_type=EventType.WORKFLOW_FAILED,
                        org=org,
                        workflow_id=workflow_id,
                        run_id=run_id,
                        execution_id=execution_id,
                        actor_id=actor_id,
                        round_idx=0,
                        error=error_text,
                        pid=pid,
                        timestamp=_utc_now(),
                    ),
                    checkpoint_write_ms,
                )
            except Exception:
                pass
            raise

        io_delta = diff_proc_io(io_start, read_proc_io())
        return ActivationTelemetry(
            org=org,
            workflow_id=workflow_id,
            run_id=run_id,
            execution_id=execution_id,
            actor_id=actor_id,
            backend=self.store.backend_kind.value,
            pid=pid,
            status=status_out,
            executed_steps=executed_steps,
            replayed_steps=replayed_steps,
            deduplicated_hits=deduplicated_hits,
            awaiting_signal=awaiting_signal,
            error=error_text,
            wall_ms=(time.perf_counter() - wall_start) * 1000.0,
            replay_load_ms=replay_load_ms,
            checkpoint_write_ms=checkpoint_write_ms,
            proc_io_rchar=io_delta.rchar,
            proc_io_wchar=io_delta.wchar,
            proc_io_write_bytes=io_delta.write_bytes,
            proc_io_syscw=io_delta.syscw,
            outputs=outputs,
        )
