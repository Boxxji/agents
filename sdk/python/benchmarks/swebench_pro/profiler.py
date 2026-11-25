"""Real OS filesystem I/O profiling and Hanzo AIConfig model cost accounting.

Measures:
- Logical & storage I/O via ``/proc/self/io`` (Linux kernel procfs)
- Block I/O and CPU times via ``resource.getrusage`` (self + children) and ``os.times``
- Wall-clock monotonic latency via ``time.perf_counter_ns``
- Token & USD cost via ``hanzo_agents.types.AIConfig`` and ``hanzo_agents.llm_adapters``
"""

from __future__ import annotations

import os
import platform
import resource
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from hanzo_agents.llm_adapters import (
    apply_provider_patches,
    filter_none_values,
    get_provider_from_model,
)
from hanzo_agents.types import AIConfig

from .schemas import FilesystemIOTelemetry, ModelCostTelemetry, StorageProfile

# Transparent per-million-token pricing table (USD) used for deterministic cost modeling.
# Prices are (input_per_1M, output_per_1M, cached_input_per_1M).
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

    return {
        "os": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count() or 1,
        "cpu_model": cpu_model,
        "proc_io_available": read_proc_self_io() is not None,
    }


@dataclass
class _IOSnapshot:
    wall_ns: int
    proc_io: Optional[Dict[str, int]]
    ru_self: resource.struct_rusage
    ru_children: resource.struct_rusage
    os_times: os.times_result


def _take_io_snapshot() -> _IOSnapshot:
    return _IOSnapshot(
        wall_ns=time.perf_counter_ns(),
        proc_io=read_proc_self_io(),
        ru_self=resource.getrusage(resource.RUSAGE_SELF),
        ru_children=resource.getrusage(resource.RUSAGE_CHILDREN),
        os_times=os.times(),
    )


class IOProfiler:
    """Context manager measuring real OS I/O, CPU, and wall-clock deltas."""

    def __init__(self, profile: StorageProfile) -> None:
        self.profile = profile
        self._start: Optional[_IOSnapshot] = None
        self.wall_ms: float = 0.0
        self.telemetry = FilesystemIOTelemetry(profile=profile)

    def __enter__(self) -> "IOProfiler":
        self._start = _take_io_snapshot()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        end = _take_io_snapshot()
        assert self._start is not None
        start = self._start

        self.wall_ms = (end.wall_ns - start.wall_ns) / 1_000_000.0

        if start.proc_io is not None and end.proc_io is not None:
            self.telemetry.proc_io_available = True
            self.telemetry.rchar_bytes = max(
                0, end.proc_io.get("rchar", 0) - start.proc_io.get("rchar", 0)
            )
            self.telemetry.wchar_bytes = max(
                0, end.proc_io.get("wchar", 0) - start.proc_io.get("wchar", 0)
            )
            self.telemetry.syscr_ops = max(
                0, end.proc_io.get("syscr", 0) - start.proc_io.get("syscr", 0)
            )
            self.telemetry.syscw_ops = max(
                0, end.proc_io.get("syscw", 0) - start.proc_io.get("syscw", 0)
            )
            self.telemetry.read_bytes_storage = max(
                0, end.proc_io.get("read_bytes", 0) - start.proc_io.get("read_bytes", 0)
            )
            self.telemetry.write_bytes_storage = max(
                0,
                end.proc_io.get("write_bytes", 0) - start.proc_io.get("write_bytes", 0),
            )

        self.telemetry.rusage_inblock = max(
            0,
            (end.ru_self.ru_inblock - start.ru_self.ru_inblock)
            + (end.ru_children.ru_inblock - start.ru_children.ru_inblock),
        )
        self.telemetry.rusage_oublock = max(
            0,
            (end.ru_self.ru_oublock - start.ru_self.ru_oublock)
            + (end.ru_children.ru_oublock - start.ru_children.ru_oublock),
        )

        # Combine self + child process user/system CPU times from os.times()
        user_delta = (end.os_times.user - start.os_times.user) + (
            end.os_times.children_user - start.os_times.children_user
        )
        sys_delta = (end.os_times.system - start.os_times.system) + (
            end.os_times.children_system - start.os_times.children_system
        )
        self.telemetry.cpu_user_ms = max(0.0, user_delta * 1000.0)
        self.telemetry.cpu_system_ms = max(0.0, sys_delta * 1000.0)


class CostTracker:
    """Computes token & USD cost through Hanzo ``AIConfig`` and ``llm_adapters``."""

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
        # Never serialize raw API keys into telemetry snapshots.
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
