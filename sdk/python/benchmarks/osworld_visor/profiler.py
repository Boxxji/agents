"""Real OS process/IO/memory telemetry and Hanzo AIConfig cost tracking.

Every timing, memory, CPU, and I/O value comes directly from the host OS:
- Monotonic latency via ``time.perf_counter_ns()``
- Per-child peak RSS (KiB) and CPU times via ``os.wait4(pid, 0)`` (Linux kernel
  ``mm->hiwater_rss`` in KiB; macOS ``ru_maxrss`` bytes normalized to KiB) and
  live ``/proc/<pid>/status`` ``VmHWM`` sampling when available
- Logical and storage I/O via Linux ``/proc/self/io``
- Model routing & budget tracking via ``hanzo_agents.types.AIConfig`` and
  ``hanzo_agents.llm_adapters``
"""

from __future__ import annotations

import os
import platform
import resource
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from hanzo_agents.llm_adapters import (
    apply_provider_patches,
    filter_none_values,
    get_provider_from_model,
)
from hanzo_agents.types import AIConfig

from .schemas import DesktopToolingReport, ModelCostTelemetry

# Transparent per-million-token pricing table (USD):
# (input_per_1M, output_per_1M, cached_input_per_1M).
MODEL_PRICING_PER_1M: Dict[str, Tuple[float, float, float]] = {
    "openai/gpt-4o": (2.50, 10.00, 1.25),
    "openai/gpt-5.4": (5.00, 15.00, 1.25),
    "anthropic/claude-sonnet-4": (3.00, 15.00, 0.30),
    "anthropic/claude-opus-4-6": (15.00, 75.00, 1.50),
    "google/gemini-3.1-pro": (2.00, 12.00, 0.50),
    "zen/zen-coder": (0.80, 3.20, 0.20),
    "gpt-4o": (2.50, 10.00, 1.25),
}

DEFAULT_PRICING_PER_1M: Tuple[float, float, float] = (3.00, 15.00, 0.30)


def read_proc_self_io() -> Optional[Dict[str, int]]:
    """Read Linux ``/proc/self/io`` counters when available."""
    try:
        with open("/proc/self/io", "r", encoding="utf-8") as fh:
            counters: Dict[str, int] = {}
            for line in fh:
                if ":" not in line:
                    continue
                key, val = line.split(":", 1)
                counters[key.strip()] = int(val.strip())
            return counters
    except (OSError, ValueError):
        return None


def read_proc_self_cmdline() -> Optional[bytes]:
    """Read Linux ``/proc/self/cmdline`` once to detect post-execve child transitions."""
    try:
        with open("/proc/self/cmdline", "rb") as fh:
            return fh.read()
    except OSError:
        return None


def read_proc_pid_vmhwm_kib(
    pid: int, *, parent_cmdline: Optional[bytes] = None
) -> Optional[int]:
    """Read real post-execve peak RSS (``VmHWM`` in KiB) from ``/proc/<pid>/status``.

    When ``parent_cmdline`` is supplied, skips sampling while the forked child
    still carries the parent process's pre-``execve`` address space so the
    runner's own RSS does not pollute per-task measurements.
    """
    try:
        if parent_cmdline is not None:
            with open(f"/proc/{pid}/cmdline", "rb") as cmd_fh:
                if cmd_fh.read() == parent_cmdline:
                    return None
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1])
    except (OSError, ValueError):
        return None
    return None


def rusage_maxrss_to_kib(ru_maxrss: int) -> int:
    """Normalize ``struct_rusage.ru_maxrss`` to KiB across Linux and macOS."""
    if ru_maxrss <= 0:
        return 0
    # On macOS (Darwin), ru_maxrss is reported in bytes; on Linux/BSD, KiB.
    if sys.platform == "darwin":
        return max(1, ru_maxrss // 1024)
    return int(ru_maxrss)


def collect_hardware_info() -> Dict[str, Any]:
    """Capture host platform metadata for reproducible HIP-1348 reports."""
    cpu_model = platform.processor() or "unknown"
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass

    mem_total_kib: Optional[int] = None
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        mem_total_kib = int(parts[1])
                    break
    except (OSError, ValueError):
        pass

    return {
        "os": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count() or 1,
        "cpu_model": cpu_model,
        "mem_total_kib": mem_total_kib,
        "proc_io_available": read_proc_self_io() is not None,
    }


def probe_desktop_tooling() -> DesktopToolingReport:
    """Inspect real host desktop/container/sandbox tooling availability."""
    tool_names = [
        "python3",
        "git",
        "sqlite3",
        "tar",
        "gzip",
        "xvfb-run",
        "playwright",
        "docker",
        "podman",
        "bwrap",
        "libreoffice",
        "chromium",
        "google-chrome",
    ]
    binaries: Dict[str, Optional[str]] = {
        name: shutil.which(name) for name in tool_names
    }
    display_env = os.environ.get("DISPLAY")
    has_xvfb = binaries["xvfb-run"] is not None
    has_browser = (
        binaries["playwright"] is not None
        or binaries["chromium"] is not None
        or binaries["google-chrome"] is not None
    )
    has_container = binaries["docker"] is not None or binaries["podman"] is not None
    gui_stack_ready = bool(display_env or has_xvfb) and has_browser

    notes: List[str] = []
    if has_xvfb and not display_env:
        notes.append(
            "xvfb-run binary present without active DISPLAY; headless GUI launch possible when desktop apps are installed."
        )
    if not binaries["libreoffice"]:
        notes.append(
            "libreoffice not installed; upstream GUI office tasks require VM/container image while headless POSIX tasks run natively."
        )
    if has_container:
        notes.append(
            "Container CLI detected on host; this benchmark executes inside isolated POSIX subprocess sandboxes without requiring privileged Docker daemon access."
        )
    else:
        notes.append(
            "Running in headless isolated POSIX subprocess sandbox mode (no container daemon required)."
        )

    return DesktopToolingReport(
        execution_backend="isolated_subprocess_posix",
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        display_env=display_env,
        gui_stack_ready=gui_stack_ready,
        binaries=binaries,
        notes=notes,
    )


@dataclass
class _IOSnapshot:
    wall_ns: int
    proc_io: Optional[Dict[str, int]]
    ru_self: resource.struct_rusage


def take_io_snapshot() -> _IOSnapshot:
    """Capture a lightweight monotonic + procfs + rusage snapshot."""
    return _IOSnapshot(
        wall_ns=time.perf_counter_ns(),
        proc_io=read_proc_self_io(),
        ru_self=resource.getrusage(resource.RUSAGE_SELF),
    )


def compute_io_delta(start: _IOSnapshot, end: _IOSnapshot) -> Dict[str, Any]:
    """Compute non-negative deltas between two host snapshots."""
    wall_ms = max(0.0, (end.wall_ns - start.wall_ns) / 1_000_000.0)
    result: Dict[str, Any] = {
        "wall_ms": wall_ms,
        "proc_io_available": False,
        "rchar_bytes": 0,
        "wchar_bytes": 0,
        "syscr_ops": 0,
        "syscw_ops": 0,
        "read_bytes_storage": 0,
        "write_bytes_storage": 0,
        "self_inblock": max(0, end.ru_self.ru_inblock - start.ru_self.ru_inblock),
        "self_oublock": max(0, end.ru_self.ru_oublock - start.ru_self.ru_oublock),
        "self_cpu_user_ms": max(
            0.0, (end.ru_self.ru_utime - start.ru_self.ru_utime) * 1000.0
        ),
        "self_cpu_system_ms": max(
            0.0, (end.ru_self.ru_stime - start.ru_self.ru_stime) * 1000.0
        ),
    }
    if start.proc_io is not None and end.proc_io is not None:
        result["proc_io_available"] = True
        result["rchar_bytes"] = max(
            0, end.proc_io.get("rchar", 0) - start.proc_io.get("rchar", 0)
        )
        result["wchar_bytes"] = max(
            0, end.proc_io.get("wchar", 0) - start.proc_io.get("wchar", 0)
        )
        result["syscr_ops"] = max(
            0, end.proc_io.get("syscr", 0) - start.proc_io.get("syscr", 0)
        )
        result["syscw_ops"] = max(
            0, end.proc_io.get("syscw", 0) - start.proc_io.get("syscw", 0)
        )
        result["read_bytes_storage"] = max(
            0, end.proc_io.get("read_bytes", 0) - start.proc_io.get("read_bytes", 0)
        )
        result["write_bytes_storage"] = max(
            0, end.proc_io.get("write_bytes", 0) - start.proc_io.get("write_bytes", 0)
        )
    return result


class CostTracker:
    """Computes token & USD cost via Hanzo ``AIConfig`` and ``llm_adapters``."""

    def __init__(
        self,
        ai_config: Optional[AIConfig] = None,
        pricing_table: Optional[Dict[str, Tuple[float, float, float]]] = None,
    ) -> None:
        self.ai_config = ai_config or AIConfig(
            model="openai/gpt-4o",
            temperature=0.0,
            max_tokens=4096,
        )
        self.pricing_table = pricing_table or MODEL_PRICING_PER_1M

    def build_llm_params(self) -> Dict[str, Any]:
        """Resolve provider-patched LLM parameters using Hanzo SDK hooks."""
        raw_params = self.ai_config.get_llm_params()
        cleaned = filter_none_values(raw_params)
        # Never persist secrets in benchmark telemetry artifacts.
        cleaned.pop("api_key", None)
        return apply_provider_patches(cleaned, self.ai_config.model)

    def record(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
        calls: int = 1,
    ) -> ModelCostTelemetry:
        """Compute USD cost and enforce optional AIConfig budget ceilings."""
        model = self.ai_config.model
        provider = get_provider_from_model(model)
        in_price, out_price, cache_price = self.pricing_table.get(
            model, DEFAULT_PRICING_PER_1M
        )

        billable_prompt = max(0, prompt_tokens - cached_tokens)
        cost_usd = (
            (billable_prompt / 1_000_000.0) * in_price
            + (cached_tokens / 1_000_000.0) * cache_price
            + (completion_tokens / 1_000_000.0) * out_price
        )
        cost_usd = round(cost_usd, 6)

        budget_exceeded = False
        if (
            self.ai_config.max_cost_per_call is not None
            and cost_usd > self.ai_config.max_cost_per_call
        ):
            budget_exceeded = True
        if (
            self.ai_config.daily_budget is not None
            and cost_usd > self.ai_config.daily_budget
        ):
            budget_exceeded = True

        return ModelCostTelemetry(
            model=model,
            provider=provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            cost_usd=cost_usd,
            calls=calls,
            budget_exceeded=budget_exceeded,
            llm_params_snapshot=self.build_llm_params(),
        )
