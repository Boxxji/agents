"""Real OS-level telemetry and percentile utilities for durable workflow replay."""

from __future__ import annotations

import math
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class ProcIOSnapshot:
    """Point-in-time Linux ``/proc/self/io`` counters (zeros if unavailable)."""

    available: bool
    rchar: int = 0
    wchar: int = 0
    syscr: int = 0
    syscw: int = 0
    read_bytes: int = 0
    write_bytes: int = 0


def read_proc_io() -> ProcIOSnapshot:
    """Read real kernel I/O counters from ``/proc/self/io`` when running on Linux."""
    proc_path = Path("/proc/self/io")
    if not proc_path.exists():
        return ProcIOSnapshot(available=False)
    try:
        raw = proc_path.read_text(encoding="utf-8")
    except OSError:
        return ProcIOSnapshot(available=False)

    values: Dict[str, int] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        try:
            values[key.strip()] = int(val.strip())
        except ValueError:
            continue
    return ProcIOSnapshot(
        available=True,
        rchar=values.get("rchar", 0),
        wchar=values.get("wchar", 0),
        syscr=values.get("syscr", 0),
        syscw=values.get("syscw", 0),
        read_bytes=values.get("read_bytes", 0),
        write_bytes=values.get("write_bytes", 0),
    )


def diff_proc_io(start: ProcIOSnapshot, end: ProcIOSnapshot) -> ProcIOSnapshot:
    """Compute non-negative I/O delta between two snapshots."""
    if not (start.available and end.available):
        return ProcIOSnapshot(available=False)
    return ProcIOSnapshot(
        available=True,
        rchar=max(0, end.rchar - start.rchar),
        wchar=max(0, end.wchar - start.wchar),
        syscr=max(0, end.syscr - start.syscr),
        syscw=max(0, end.syscw - start.syscw),
        read_bytes=max(0, end.read_bytes - start.read_bytes),
        write_bytes=max(0, end.write_bytes - start.write_bytes),
    )


def percentile(values: Iterable[float], pct: float) -> float:
    """Compute linear-interpolated percentile (0..100) without external libraries."""
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    clamped = max(0.0, min(100.0, pct))
    rank = (clamped / 100.0) * (len(ordered) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def summarize_latencies(values: Iterable[float]) -> Dict[str, float]:
    """Return count, mean, p50, p95, and max latency summary in milliseconds."""
    items: List[float] = [float(v) for v in values]
    if not items:
        return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    return {
        "count": len(items),
        "mean_ms": round(sum(items) / len(items), 4),
        "p50_ms": round(percentile(items, 50.0), 4),
        "p95_ms": round(percentile(items, 95.0), 4),
        "max_ms": round(max(items), 4),
    }


def collect_host_metadata() -> Dict[str, Any]:
    """Collect reproducible host hardware/runtime metadata without local paths."""
    cpu_model: Optional[str] = None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        try:
            for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.lower().startswith("model name"):
                    _, val = line.split(":", 1)
                    cpu_model = val.strip()
                    break
        except OSError:
            cpu_model = None

    io_probe = read_proc_io()
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "cpu_model": cpu_model or platform.processor() or "unknown",
        "cpu_count": os.cpu_count() or 1,
        "python_version": sys.version.split()[0],
        "proc_self_io_available": io_probe.available,
    }
