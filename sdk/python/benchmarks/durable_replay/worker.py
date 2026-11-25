"""Subprocess worker executing a single OS-isolated workflow activation.

Invoked via ``python3 -m benchmarks.durable_replay.worker --config <path>`` so
crash-recovery experiments kill real OS processes with ``os._exit(99)`` and
verify durable resumption across distinct PIDs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from hanzo_agents.agent_workflow import AgentWorkflow

from .engine import DurableReplayEngine
from .schemas import (
    BackendKind,
    CrashPlan,
    DurableReplayError,
    NonDeterministicReplayError,
    StepSpec,
    TenantIsolationError,
)
from .side_effect_ledger import ExternalSideEffectLedger
from .stores import create_checkpoint_store


class _RecordingClient:
    """Minimal in-process client stub capturing ``AgentWorkflow`` event notifications."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self._current_workflow_context = None

    async def _async_request(self, method: str, url: str, json: Dict[str, Any]) -> Dict[str, Any]:
        self.events.append({"method": method, "url": url, "payload": json})
        return {"ok": True}


class _MinimalAgent:
    """Minimal agent host compatible with ``hanzo_agents.agent_workflow.AgentWorkflow``."""

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self.client = _RecordingClient()
        self.hanzo_agents_server = "http://127.0.0.1:0"
        self.dev_mode = False
        self._current_execution_context = None


def _write_result_atomic(result_path: Path, payload: Dict[str, Any]) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".res-", dir=str(result_path.parent))
    try:
        try:
            os.write(fd, json.dumps(payload, sort_keys=True, indent=2).encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp_name, str(result_path))
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def run_worker_from_config(config: Dict[str, Any]) -> int:
    """Execute one activation described by ``config`` and write telemetry to ``result_path``."""
    backend = BackendKind(config["backend"])
    store_path = config.get("store_path")
    external_ledger_path = Path(config["external_ledger_path"]).resolve()
    result_path = Path(config["result_path"]).resolve()

    steps = [StepSpec.from_dict(s) for s in config["steps"]]
    crash_plan = CrashPlan.from_dict(config.get("crash_plan"))
    delivered_signals = config.get("delivered_signals") or {}
    enable_agent_workflow = bool(config.get("enable_agent_workflow", False))

    store = create_checkpoint_store(backend, store_path)
    ledger = ExternalSideEffectLedger(external_ledger_path)

    agent_workflow = None
    agent_host = None
    if enable_agent_workflow:
        agent_host = _MinimalAgent(node_id=f"worker-{os.getpid()}")
        agent_workflow = AgentWorkflow(agent_host)

    engine = DurableReplayEngine(
        store,
        external_ledger=ledger,
        agent_workflow=agent_workflow,
    )

    try:
        telemetry = engine.activate(
            org=str(config["org"]),
            workflow_id=str(config["workflow_id"]),
            run_id=str(config["run_id"]),
            actor_id=str(config["actor_id"]),
            steps=steps,
            delivered_signals=delivered_signals,
            crash_plan=crash_plan,
        )
        out = {
            "ok": True,
            "error_type": "",
            "error": "",
            "telemetry": telemetry.to_dict(),
            "store_bytes": store.storage_bytes(),
            "agent_workflow_events": (
                len(agent_host.client.events) if agent_host is not None else 0
            ),
        }
        _write_result_atomic(result_path, out)
        return 0
    except (NonDeterministicReplayError, TenantIsolationError, DurableReplayError) as exc:
        out = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "pid": os.getpid(),
            "store_bytes": store.storage_bytes(),
        }
        _write_result_atomic(result_path, out)
        return 2
    finally:
        store.close()
        ledger.close()


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Execute one OS-isolated durable workflow activation."
    )
    parser.add_argument("--config", required=True, help="Path to worker JSON config")
    args = parser.parse_args(argv)
    cfg_path = Path(args.config).resolve()
    config = json.loads(cfg_path.read_text(encoding="utf-8"))
    return run_worker_from_config(config)


if __name__ == "__main__":
    sys.exit(main())
