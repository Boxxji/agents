"""Clean-room SWE-bench Pro V2 subset definitions and external manifest loader.

SWE-bench Pro V2 (Scale Labs, released 2026-09-22, revision ``sep-2026``)
contains 642 validated public tasks across 11 strong-copyleft (GPL/AGPL)
repositories (Go: 256, Python: 237, JS: 145, TS: 4) plus a 51-task HARD split.
Because ``hanzoai/agents`` is Apache-2.0, this module intentionally ships ZERO
GPL repository code or benchmark test patches. Instead it provides:

1. ``load_external_manifest(path)``: validates external SWE-bench Pro V2 /
   Harbor JSONL manifests at runtime, enforcing ``v2`` / ``sep-2026`` metadata
   (rejecting stale V1 manifests unless explicitly opted in).
2. ``build_cleanroom_v2_subset()``: 8 deterministic, Apache-2.0 multi-file
   Python engineering tasks (each spanning 3-4 source modules + fail-to-pass
   and pass-to-pass test suites) for reproducible offline benchmarking and
   anti-leakage / anti-tampering verification.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .schemas import SWEBenchTask

SWE_BENCH_PRO_V2_REVISION = "sep-2026"
SWE_BENCH_PRO_V2_VERSION = "v2"
SWE_BENCH_PRO_V2_RELEASE_DATE = "2026-09-22"
SWE_BENCH_PRO_V2_PUBLIC_TASK_COUNT = 642
SWE_BENCH_PRO_V2_HARD_TASK_COUNT = 51
SWE_BENCH_PRO_V2_LANGUAGE_COUNTS: Dict[str, int] = {
    "go": 256,
    "python": 237,
    "javascript": 145,
    "typescript": 4,
}


def compute_manifest_digest(tasks: Sequence[SWEBenchTask], seed: int = 42) -> str:
    """Compute a deterministic SHA-256 content digest over all task metadata and contents."""
    h = hashlib.sha256()
    header = {
        "benchmark_revision": SWE_BENCH_PRO_V2_REVISION,
        "benchmark_version": SWE_BENCH_PRO_V2_VERSION,
        "seed": seed,
    }
    h.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    h.update(b"\n")
    for task in tasks:
        canonical_task = {
            "instance_id": task.instance_id,
            "repo": task.repo,
            "language": task.language,
            "difficulty": task.difficulty,
            "revision": task.revision,
            "benchmark_version": task.benchmark_version,
            "problem_statement": task.problem_statement,
            "gold_marker": task.gold_marker,
            "base_files": sorted(task.base_files.items()),
            "test_files": sorted(task.test_files.items()),
            "gold_patch_files": sorted(task.gold_patch_files.items()),
            "partial_patch_files": (
                sorted(task.partial_patch_files.items())
                if task.partial_patch_files is not None
                else None
            ),
            "fail_to_pass_tests": list(task.fail_to_pass_tests),
            "pass_to_pass_tests": list(task.pass_to_pass_tests),
            "protected_test_paths": list(task.protected_test_paths),
            "expected_touched_files": list(task.expected_touched_files),
        }
        h.update(
            json.dumps(canonical_task, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        h.update(b"\n")
    return h.hexdigest()


def select_subset(
    tasks: Sequence[SWEBenchTask],
    *,
    max_tasks: Optional[int] = None,
    seed: int = 42,
    difficulties: Optional[Sequence[str]] = None,
) -> Tuple[List[SWEBenchTask], str]:
    """Deterministically filter and sample a reproducible subset."""
    filtered = list(tasks)
    if difficulties:
        allowed = {d.lower() for d in difficulties}
        filtered = [t for t in filtered if t.difficulty.lower() in allowed]
    filtered.sort(key=lambda t: t.instance_id)

    if max_tasks is not None and max_tasks < len(filtered):
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(len(filtered)), max_tasks))
        filtered = [filtered[i] for i in indices]

    digest = compute_manifest_digest(filtered, seed=seed)
    return filtered, digest


def load_external_manifest(
    manifest_path: Path,
    *,
    require_v2: bool = True,
) -> List[Dict[str, Any]]:
    """Load and validate an external SWE-bench Pro V2 / Harbor JSONL manifest.

    Validates required task identifiers and enforces the current V2 revision
    (rejecting older V1 manifests when ``require_v2=True``).
    """
    records: List[Dict[str, Any]] = []
    with open(manifest_path, "r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"Line {line_no}: expected JSON object")
            for required in ("instance_id", "repo", "base_commit"):
                if not item.get(required):
                    raise ValueError(f"Line {line_no}: missing required field {required!r}")
            version = str(item.get("version", item.get("benchmark_version", ""))).lower()
            revision = str(item.get("revision", "")).lower()
            if require_v2:
                if version not in {"v2", "2", "pro-v2"}:
                    raise ValueError(
                        f"Line {line_no} ({item['instance_id']}): legacy version {version!r}; "
                        f"SWE-bench Pro V2 requires version 'v2'"
                    )
                if revision and revision != SWE_BENCH_PRO_V2_REVISION:
                    raise ValueError(
                        f"Line {line_no} ({item['instance_id']}): stale revision {revision!r}; "
                        f"expected {SWE_BENCH_PRO_V2_REVISION!r}"
                    )
            records.append(item)
    return records


# ---------------------------------------------------------------------------
# Clean-room Apache-2.0 multi-file SWE-bench Pro V2 task suite (8 tasks)
# ---------------------------------------------------------------------------


def _task_vfs_cache_invalidation() -> SWEBenchTask:
    base_files = {
        "vfscache/__init__.py": (
            "from .paths import normalize_vfs_path, is_subpath\n"
            "from .store import VFSCacheStore\n"
            "__all__ = ['normalize_vfs_path', 'is_subpath', 'VFSCacheStore']\n"
        ),
        "vfscache/paths.py": (
            "import posixpath\n\n"
            "def normalize_vfs_path(path: str) -> str:\n"
            "    cleaned = posixpath.normpath('/' + path.strip().lstrip('/'))\n"
            "    return cleaned\n\n"
            "def is_subpath(candidate: str, prefix: str) -> bool:\n"
            "    # BUG: raw string prefix matches '/mnt/app2' for prefix '/mnt/app'\n"
            "    return normalize_vfs_path(candidate).startswith(normalize_vfs_path(prefix))\n"
        ),
        "vfscache/store.py": (
            "from typing import Dict, Optional\n"
            "from .paths import is_subpath, normalize_vfs_path\n\n"
            "class VFSCacheStore:\n"
            "    def __init__(self) -> None:\n"
            "        self._entries: Dict[str, bytes] = {}\n"
            "        self._versions: Dict[str, int] = {}\n\n"
            "    def put(self, path: str, payload: bytes) -> int:\n"
            "        norm = normalize_vfs_path(path)\n"
            "        self._entries[norm] = payload\n"
            "        ver = self._versions.get(norm, 0) + 1\n"
            "        self._versions[norm] = ver\n"
            "        return ver\n\n"
            "    def get(self, path: str) -> Optional[bytes]:\n"
            "        return self._entries.get(normalize_vfs_path(path))\n\n"
            "    def invalidate_prefix(self, prefix: str) -> int:\n"
            "        removed = [k for k in self._entries if is_subpath(k, prefix)]\n"
            "        for k in removed:\n"
            "            del self._entries[k]\n"
            "        return len(removed)\n"
        ),
    }
    gold_files = {
        "vfscache/paths.py": (
            "import posixpath\n\n"
            "def normalize_vfs_path(path: str) -> str:\n"
            "    if '\\x00' in path:\n"
            "        raise ValueError('null byte in VFS path')\n"
            "    raw = path.replace('\\\\', '/').strip()\n"
            "    parts = [p for p in raw.split('/') if p and p != '.']\n"
            "    stack = []\n"
            "    for part in parts:\n"
            "        if part == '..':\n"
            "            if not stack:\n"
            "                raise ValueError('path escapes VFS root')\n"
            "            stack.pop()\n"
            "        else:\n"
            "            stack.append(part)\n"
            "    return '/' + '/'.join(stack)\n\n"
            "def is_subpath(candidate: str, prefix: str) -> bool:\n"
            "    c = normalize_vfs_path(candidate)\n"
            "    p = normalize_vfs_path(prefix)\n"
            "    if p == '/':\n"
            "        return True\n"
            "    return c == p or c.startswith(p + '/')\n"
        ),
        "vfscache/store.py": (
            "from typing import Dict, Optional\n"
            "from .paths import is_subpath, normalize_vfs_path\n\n"
            "class VFSCacheStore:\n"
            "    def __init__(self) -> None:\n"
            "        self._entries: Dict[str, bytes] = {}\n"
            "        self._versions: Dict[str, int] = {}\n\n"
            "    def put(self, path: str, payload: bytes) -> int:\n"
            "        norm = normalize_vfs_path(path)\n"
            "        self._entries[norm] = payload\n"
            "        ver = self._versions.get(norm, 0) + 1\n"
            "        self._versions[norm] = ver\n"
            "        return ver\n\n"
            "    def get(self, path: str) -> Optional[bytes]:\n"
            "        return self._entries.get(normalize_vfs_path(path))\n\n"
            "    def invalidate_prefix(self, prefix: str) -> int:\n"
            "        norm_prefix = normalize_vfs_path(prefix)\n"
            "        removed = [k for k in self._entries if is_subpath(k, norm_prefix)]\n"
            "        for k in removed:\n"
            "            del self._entries[k]\n"
            "            self._versions[k] = self._versions.get(k, 0) + 1\n"
            "        return len(removed)\n"
        ),
    }
    test_files = {
        "tests/test_vfs_cache.py": (
            "import pytest\n"
            "from vfscache import VFSCacheStore, is_subpath, normalize_vfs_path\n\n"
            "def test_put_and_get_basic():\n"
            "    store = VFSCacheStore()\n"
            "    assert store.put('/mnt/app/config.json', b'{}') == 1\n"
            "    assert store.get('/mnt/app/config.json') == b'{}'\n\n"
            "def test_invalidate_exact_directory():\n"
            "    store = VFSCacheStore()\n"
            "    store.put('/mnt/app/a.py', b'1')\n"
            "    store.put('/mnt/app/sub/b.py', b'2')\n"
            "    assert store.invalidate_prefix('/mnt/app') == 2\n"
            "    assert store.get('/mnt/app/a.py') is None\n\n"
            "def test_invalidate_does_not_clobber_sibling_mount():\n"
            "    store = VFSCacheStore()\n"
            "    store.put('/mnt/app/a.py', b'1')\n"
            "    store.put('/mnt/app2/b.py', b'2')\n"
            "    count = store.invalidate_prefix('/mnt/app')\n"
            "    assert count == 1\n"
            "    assert store.get('/mnt/app2/b.py') == b'2'\n\n"
            "def test_path_traversal_above_root_rejected():\n"
            "    with pytest.raises(ValueError):\n"
            "        normalize_vfs_path('/mnt/../../etc/passwd')\n"
            "    assert is_subpath('/mnt/app/x', '/mnt/app')\n"
            "    assert not is_subpath('/mnt/app-staging/x', '/mnt/app')\n"
        ),
    }
    return SWEBenchTask(
        instance_id="swebench_pro_v2__vfs_cache_invalidation",
        repo="hanzo-cleanroom/vfscache",
        language="python",
        difficulty="medium",
        problem_statement=(
            "Fix prefix invalidation in VFSCacheStore so invalidating '/mnt/app' "
            "never evicts sibling mount '/mnt/app2', and reject path traversal "
            "attempts above the VFS root in normalize_vfs_path."
        ),
        base_files=base_files,
        test_files=test_files,
        gold_patch_files=gold_files,
        gold_marker="GOLD_SENTINEL_VFS_CACHE_9F2A",
        fail_to_pass_tests=[
            "tests/test_vfs_cache.py::test_invalidate_does_not_clobber_sibling_mount",
            "tests/test_vfs_cache.py::test_path_traversal_above_root_rejected",
        ],
        pass_to_pass_tests=[
            "tests/test_vfs_cache.py::test_put_and_get_basic",
            "tests/test_vfs_cache.py::test_invalidate_exact_directory",
        ],
        protected_test_paths=["tests/test_vfs_cache.py"],
        expected_touched_files=["vfscache/paths.py", "vfscache/store.py"],
    )


def _task_dag_scheduler_cycle_priority() -> SWEBenchTask:
    base_files = {
        "dagsched/__init__.py": (
            "from .graph import DAGGraph, CycleError\n"
            "from .scheduler import PriorityDAGScheduler\n"
            "__all__ = ['DAGGraph', 'CycleError', 'PriorityDAGScheduler']\n"
        ),
        "dagsched/graph.py": (
            "from typing import Dict, List, Set\n\n"
            "class CycleError(ValueError):\n"
            "    pass\n\n"
            "class DAGGraph:\n"
            "    def __init__(self) -> None:\n"
            "        self.deps: Dict[str, Set[str]] = {}\n"
            "        self.priorities: Dict[str, int] = {}\n\n"
            "    def add_node(self, node_id: str, priority: int = 0, deps: List[str] | None = None) -> None:\n"
            "        self.deps.setdefault(node_id, set()).update(deps or [])\n"
            "        self.priorities[node_id] = priority\n"
            "        for dep in deps or []:\n"
            "            self.deps.setdefault(dep, set())\n"
            "            self.priorities.setdefault(dep, 0)\n"
        ),
        "dagsched/scheduler.py": (
            "from typing import List, Set\n"
            "from .graph import CycleError, DAGGraph\n\n"
            "class PriorityDAGScheduler:\n"
            "    def __init__(self, graph: DAGGraph) -> None:\n"
            "        self.graph = graph\n\n"
            "    def plan(self, failed: Set[str] | None = None) -> List[str]:\n"
            "        # BUG: ignores priority ordering and does not cancel transitive dependents of failed nodes\n"
            "        failed_set = set(failed or ())\n"
            "        order: List[str] = []\n"
            "        remaining = {n: set(d) for n, d in self.graph.deps.items() if n not in failed_set}\n"
            "        while remaining:\n"
            "            ready = sorted([n for n, d in remaining.items() if not (d - failed_set)])\n"
            "            if not ready:\n"
            "                raise CycleError('cycle detected')\n"
            "            nxt = ready[0]\n"
            "            order.append(nxt)\n"
            "            del remaining[nxt]\n"
            "            for d in remaining.values():\n"
            "                d.discard(nxt)\n"
            "        return order\n"
        ),
    }
    gold_files = {
        "dagsched/graph.py": (
            "from typing import Dict, List, Set\n\n"
            "class CycleError(ValueError):\n"
            "    def __init__(self, message: str, cycle_nodes: List[str] | None = None) -> None:\n"
            "        super().__init__(message)\n"
            "        self.cycle_nodes = list(cycle_nodes or [])\n\n"
            "class DAGGraph:\n"
            "    def __init__(self) -> None:\n"
            "        self.deps: Dict[str, Set[str]] = {}\n"
            "        self.priorities: Dict[str, int] = {}\n\n"
            "    def add_node(self, node_id: str, priority: int = 0, deps: List[str] | None = None) -> None:\n"
            "        self.deps.setdefault(node_id, set()).update(deps or [])\n"
            "        self.priorities[node_id] = priority\n"
            "        for dep in deps or []:\n"
            "            self.deps.setdefault(dep, set())\n"
            "            self.priorities.setdefault(dep, 0)\n\n"
            "    def transitive_dependents(self, roots: Set[str]) -> Set[str]:\n"
            "        cancelled = set(roots)\n"
            "        changed = True\n"
            "        while changed:\n"
            "            changed = False\n"
            "            for node, deps in self.deps.items():\n"
            "                if node not in cancelled and (deps & cancelled):\n"
            "                    cancelled.add(node)\n"
            "                    changed = True\n"
            "        return cancelled\n"
        ),
        "dagsched/scheduler.py": (
            "from typing import List, Set\n"
            "from .graph import CycleError, DAGGraph\n\n"
            "class PriorityDAGScheduler:\n"
            "    def __init__(self, graph: DAGGraph) -> None:\n"
            "        self.graph = graph\n\n"
            "    def plan(self, failed: Set[str] | None = None) -> List[str]:\n"
            "        cancelled = self.graph.transitive_dependents(set(failed or ()))\n"
            "        remaining = {\n"
            "            n: set(d)\n"
            "            for n, d in self.graph.deps.items()\n"
            "            if n not in cancelled\n"
            "        }\n"
            "        order: List[str] = []\n"
            "        while remaining:\n"
            "            ready = [n for n, d in remaining.items() if not d]\n"
            "            if not ready:\n"
            "                stuck = sorted(remaining.keys())\n"
            "                raise CycleError(f'cycle involving {stuck}', cycle_nodes=stuck)\n"
            "            ready.sort(key=lambda n: (-self.graph.priorities.get(n, 0), n))\n"
            "            nxt = ready[0]\n"
            "            order.append(nxt)\n"
            "            del remaining[nxt]\n"
            "            for d in remaining.values():\n"
            "                d.discard(nxt)\n"
            "        return order\n"
        ),
    }
    test_files = {
        "tests/test_dag_scheduler.py": (
            "import pytest\n"
            "from dagsched import CycleError, DAGGraph, PriorityDAGScheduler\n\n"
            "def test_linear_dependency_order():\n"
            "    g = DAGGraph()\n"
            "    g.add_node('build')\n"
            "    g.add_node('test', deps=['build'])\n"
            "    g.add_node('deploy', deps=['test'])\n"
            "    assert PriorityDAGScheduler(g).plan() == ['build', 'test', 'deploy']\n\n"
            "def test_priority_tie_breaking_highest_first():\n"
            "    g = DAGGraph()\n"
            "    g.add_node('low', priority=1)\n"
            "    g.add_node('high', priority=10)\n"
            "    g.add_node('mid_b', priority=5)\n"
            "    g.add_node('mid_a', priority=5)\n"
            "    assert PriorityDAGScheduler(g).plan() == ['high', 'mid_a', 'mid_b', 'low']\n\n"
            "def test_transitive_cancellation_on_failure():\n"
            "    g = DAGGraph()\n"
            "    g.add_node('a')\n"
            "    g.add_node('b', deps=['a'])\n"
            "    g.add_node('c', deps=['b'])\n"
            "    g.add_node('independent', priority=2)\n"
            "    assert PriorityDAGScheduler(g).plan(failed={'a'}) == ['independent']\n\n"
            "def test_cycle_reports_nodes():\n"
            "    g = DAGGraph()\n"
            "    g.add_node('x', deps=['y'])\n"
            "    g.add_node('y', deps=['x'])\n"
            "    with pytest.raises(CycleError) as exc:\n"
            "        PriorityDAGScheduler(g).plan()\n"
            "    assert set(exc.value.cycle_nodes) == {'x', 'y'}\n"
        ),
    }
    return SWEBenchTask(
        instance_id="swebench_pro_v2__dag_scheduler_cycle_priority",
        repo="hanzo-cleanroom/dagsched",
        language="python",
        difficulty="medium",
        problem_statement=(
            "Fix PriorityDAGScheduler to dispatch ready nodes by descending priority "
            "(lexicographical tie-break), transitively cancel dependents of failed nodes, "
            "and attach cycle_nodes to CycleError."
        ),
        base_files=base_files,
        test_files=test_files,
        gold_patch_files=gold_files,
        gold_marker="GOLD_SENTINEL_DAG_SCHED_4C1B",
        fail_to_pass_tests=[
            "tests/test_dag_scheduler.py::test_priority_tie_breaking_highest_first",
            "tests/test_dag_scheduler.py::test_transitive_cancellation_on_failure",
            "tests/test_dag_scheduler.py::test_cycle_reports_nodes",
        ],
        pass_to_pass_tests=[
            "tests/test_dag_scheduler.py::test_linear_dependency_order",
        ],
        protected_test_paths=["tests/test_dag_scheduler.py"],
        expected_touched_files=["dagsched/graph.py", "dagsched/scheduler.py"],
    )


def _task_token_bucket_burst_refill() -> SWEBenchTask:
    base_files = {
        "quotas/__init__.py": (
            "from .bucket import TokenBucket\n"
            "from .router import HierarchicalQuotaRouter\n"
            "__all__ = ['TokenBucket', 'HierarchicalQuotaRouter']\n"
        ),
        "quotas/bucket.py": (
            "class TokenBucket:\n"
            "    def __init__(self, capacity: float, refill_per_sec: float, now: float = 0.0) -> None:\n"
            "        self.capacity = float(capacity)\n"
            "        self.refill_per_sec = float(refill_per_sec)\n"
            "        self.tokens = float(capacity)\n"
            "        self.last_ts = float(now)\n\n"
            "    def allow(self, cost: float, now: float) -> bool:\n"
            "        # BUG: does not clamp negative clock skew and does not cap burst at capacity\n"
            "        elapsed = now - self.last_ts\n"
            "        self.tokens += elapsed * self.refill_per_sec\n"
            "        self.last_ts = now\n"
            "        if self.tokens >= cost:\n"
            "            self.tokens -= cost\n"
            "            return True\n"
            "        return False\n"
        ),
        "quotas/router.py": (
            "from typing import Dict\n"
            "from .bucket import TokenBucket\n\n"
            "class HierarchicalQuotaRouter:\n"
            "    def __init__(self, global_bucket: TokenBucket) -> None:\n"
            "        self.global_bucket = global_bucket\n"
            "        self.tenants: Dict[str, TokenBucket] = {}\n\n"
            "    def register_tenant(self, tenant: str, bucket: TokenBucket) -> None:\n"
            "        self.tenants[tenant] = bucket\n\n"
            "    def acquire(self, tenant: str, cost: float, now: float) -> bool:\n"
            "        # BUG: debits global bucket even if tenant bucket rejects (no rollback)\n"
            "        if not self.global_bucket.allow(cost, now):\n"
            "            return False\n"
            "        tb = self.tenants.get(tenant)\n"
            "        if tb is None:\n"
            "            return True\n"
            "        return tb.allow(cost, now)\n"
        ),
    }
    gold_files = {
        "quotas/bucket.py": (
            "class TokenBucket:\n"
            "    def __init__(self, capacity: float, refill_per_sec: float, now: float = 0.0) -> None:\n"
            "        self.capacity = float(capacity)\n"
            "        self.refill_per_sec = float(refill_per_sec)\n"
            "        self.tokens = float(capacity)\n"
            "        self.last_ts = float(now)\n\n"
            "    def _refill(self, now: float) -> None:\n"
            "        if now <= self.last_ts:\n"
            "            return\n"
            "        elapsed = now - self.last_ts\n"
            "        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_sec)\n"
            "        self.last_ts = now\n\n"
            "    def can_consume(self, cost: float, now: float) -> bool:\n"
            "        self._refill(now)\n"
            "        return self.tokens >= cost\n\n"
            "    def allow(self, cost: float, now: float) -> bool:\n"
            "        self._refill(now)\n"
            "        if self.tokens >= cost:\n"
            "            self.tokens -= cost\n"
            "            return True\n"
            "        return False\n"
        ),
        "quotas/router.py": (
            "from typing import Dict\n"
            "from .bucket import TokenBucket\n\n"
            "class HierarchicalQuotaRouter:\n"
            "    def __init__(self, global_bucket: TokenBucket) -> None:\n"
            "        self.global_bucket = global_bucket\n"
            "        self.tenants: Dict[str, TokenBucket] = {}\n\n"
            "    def register_tenant(self, tenant: str, bucket: TokenBucket) -> None:\n"
            "        self.tenants[tenant] = bucket\n\n"
            "    def acquire(self, tenant: str, cost: float, now: float) -> bool:\n"
            "        if not self.global_bucket.can_consume(cost, now):\n"
            "            return False\n"
            "        tb = self.tenants.get(tenant)\n"
            "        if tb is not None and not tb.can_consume(cost, now):\n"
            "            return False\n"
            "        self.global_bucket.allow(cost, now)\n"
            "        if tb is not None:\n"
            "            tb.allow(cost, now)\n"
            "        return True\n"
        ),
    }
    test_files = {
        "tests/test_quotas.py": (
            "from quotas import HierarchicalQuotaRouter, TokenBucket\n\n"
            "def test_basic_bucket_allow_and_exhaust():\n"
            "    b = TokenBucket(capacity=10, refill_per_sec=1, now=0.0)\n"
            "    assert b.allow(6, now=0.0)\n"
            "    assert not b.allow(6, now=0.0)\n"
            "    assert b.allow(6, now=3.0)\n\n"
            "def test_burst_capped_at_capacity():\n"
            "    b = TokenBucket(capacity=10, refill_per_sec=5, now=0.0)\n"
            "    assert not b.allow(15, now=100.0)\n"
            "    assert b.allow(10, now=100.0)\n\n"
            "def test_negative_clock_skew_does_not_drain_tokens():\n"
            "    b = TokenBucket(capacity=10, refill_per_sec=2, now=10.0)\n"
            "    assert b.allow(5, now=5.0)\n\n"
            "def test_hierarchical_rejection_does_not_debit_global_quota():\n"
            "    g = TokenBucket(capacity=10, refill_per_sec=0, now=0.0)\n"
            "    router = HierarchicalQuotaRouter(g)\n"
            "    router.register_tenant('t1', TokenBucket(capacity=2, refill_per_sec=0, now=0.0))\n"
            "    router.register_tenant('t2', TokenBucket(capacity=10, refill_per_sec=0, now=0.0))\n"
            "    assert not router.acquire('t1', cost=5, now=0.0)\n"
            "    assert router.acquire('t2', cost=10, now=0.0)\n"
        ),
    }
    return SWEBenchTask(
        instance_id="swebench_pro_v2__token_bucket_burst_refill",
        repo="hanzo-cleanroom/quotas",
        language="python",
        difficulty="medium",
        problem_statement=(
            "Cap TokenBucket refill at capacity, ignore backward clock skew, and make "
            "HierarchicalQuotaRouter.acquire atomic so tenant rejections never debit "
            "the global quota bucket."
        ),
        base_files=base_files,
        test_files=test_files,
        gold_patch_files=gold_files,
        gold_marker="GOLD_SENTINEL_QUOTAS_7E3D",
        fail_to_pass_tests=[
            "tests/test_quotas.py::test_burst_capped_at_capacity",
            "tests/test_quotas.py::test_negative_clock_skew_does_not_drain_tokens",
            "tests/test_quotas.py::test_hierarchical_rejection_does_not_debit_global_quota",
        ],
        pass_to_pass_tests=[
            "tests/test_quotas.py::test_basic_bucket_allow_and_exhaust",
        ],
        protected_test_paths=["tests/test_quotas.py"],
        expected_touched_files=["quotas/bucket.py", "quotas/router.py"],
    )


def _task_stream_jsonl_chunk_assembler() -> SWEBenchTask:
    base_files = {
        "streamrpc/__init__.py": (
            "from .buffer import FrameBuffer, FrameTooLargeError\n"
            "from .decoder import JSONLStreamDecoder\n"
            "__all__ = ['FrameBuffer', 'FrameTooLargeError', 'JSONLStreamDecoder']\n"
        ),
        "streamrpc/buffer.py": (
            "from typing import List\n\n"
            "class FrameTooLargeError(ValueError):\n"
            "    pass\n\n"
            "class FrameBuffer:\n"
            "    def __init__(self, max_frame_bytes: int = 65536) -> None:\n"
            "        self.max_frame_bytes = max_frame_bytes\n"
            "        self._buf = ''\n\n"
            "    def feed(self, chunk: bytes) -> List[bytes]:\n"
            "        # BUG: decodes each chunk as UTF-8 immediately (breaks multi-byte boundary)\n"
            "        # and does not enforce max_frame_bytes\n"
            "        self._buf += chunk.decode('utf-8')\n"
            "        lines = self._buf.split('\\n')\n"
            "        self._buf = lines.pop()\n"
            "        return [l.encode('utf-8') for l in lines if l.strip()]\n"
        ),
        "streamrpc/decoder.py": (
            "import json\n"
            "from typing import Any, List\n"
            "from .buffer import FrameBuffer\n\n"
            "class JSONLStreamDecoder:\n"
            "    def __init__(self, max_frame_bytes: int = 65536) -> None:\n"
            "        self.buffer = FrameBuffer(max_frame_bytes=max_frame_bytes)\n\n"
            "    def feed(self, chunk: bytes) -> List[Any]:\n"
            "        frames = self.buffer.feed(chunk)\n"
            "        return [json.loads(f.decode('utf-8')) for f in frames]\n"
        ),
    }
    gold_files = {
        "streamrpc/buffer.py": (
            "from typing import List\n\n"
            "class FrameTooLargeError(ValueError):\n"
            "    pass\n\n"
            "class FrameBuffer:\n"
            "    def __init__(self, max_frame_bytes: int = 65536) -> None:\n"
            "        self.max_frame_bytes = max_frame_bytes\n"
            "        self._buf = bytearray()\n\n"
            "    def feed(self, chunk: bytes) -> List[bytes]:\n"
            "        self._buf.extend(chunk)\n"
            "        frames: List[bytes] = []\n"
            "        while True:\n"
            "            nl = self._buf.find(b'\\n')\n"
            "            if nl == -1:\n"
            "                if len(self._buf) > self.max_frame_bytes:\n"
            "                    self._buf.clear()\n"
            "                    raise FrameTooLargeError('buffered frame exceeds max_frame_bytes')\n"
            "                break\n"
            "            raw = bytes(self._buf[:nl]).rstrip(b'\\r')\n"
            "            del self._buf[: nl + 1]\n"
            "            if len(raw) > self.max_frame_bytes:\n"
            "                raise FrameTooLargeError('frame exceeds max_frame_bytes')\n"
            "            if raw.strip():\n"
            "                frames.append(raw)\n"
            "        return frames\n\n"
            "    def flush(self) -> List[bytes]:\n"
            "        raw = bytes(self._buf).rstrip(b'\\r')\n"
            "        self._buf.clear()\n"
            "        if len(raw) > self.max_frame_bytes:\n"
            "            raise FrameTooLargeError('trailing frame exceeds max_frame_bytes')\n"
            "        return [raw] if raw.strip() else []\n"
        ),
        "streamrpc/decoder.py": (
            "import json\n"
            "from typing import Any, List\n"
            "from .buffer import FrameBuffer\n\n"
            "class JSONLStreamDecoder:\n"
            "    def __init__(self, max_frame_bytes: int = 65536) -> None:\n"
            "        self.buffer = FrameBuffer(max_frame_bytes=max_frame_bytes)\n\n"
            "    def feed(self, chunk: bytes) -> List[Any]:\n"
            "        frames = self.buffer.feed(chunk)\n"
            "        return [json.loads(f.decode('utf-8')) for f in frames]\n\n"
            "    def flush(self) -> List[Any]:\n"
            "        frames = self.buffer.flush()\n"
            "        return [json.loads(f.decode('utf-8')) for f in frames]\n"
        ),
    }
    test_files = {
        "tests/test_stream_rpc.py": (
            "import pytest\n"
            "from streamrpc import FrameTooLargeError, JSONLStreamDecoder\n\n"
            "def test_basic_jsonl_chunks():\n"
            "    dec = JSONLStreamDecoder()\n"
            "    assert dec.feed(b'{\"a\": 1}\\n{\"b\": 2}\\n') == [{'a': 1}, {'b': 2}]\n\n"
            "def test_multibyte_utf8_split_across_chunks():\n"
            "    dec = JSONLStreamDecoder()\n"
            "    payload = '{\"msg\": \"héllo 🚀\"}\\n'.encode('utf-8')\n"
            "    # Split right inside the 4-byte emoji sequence\n"
            "    split = payload.index(b'\\xf0') + 2\n"
            "    assert dec.feed(payload[:split]) == []\n"
            "    assert dec.feed(payload[split:]) == [{'msg': 'héllo 🚀'}]\n\n"
            "def test_max_frame_bytes_enforced():\n"
            "    dec = JSONLStreamDecoder(max_frame_bytes=16)\n"
            "    with pytest.raises(FrameTooLargeError):\n"
            "        dec.feed(b'{\"payload\": \"0123456789abcdef\"}\\n')\n\n"
            "def test_flush_trailing_record_without_newline():\n"
            "    dec = JSONLStreamDecoder()\n"
            "    assert dec.feed(b'{\"final\": true}') == []\n"
            "    assert dec.flush() == [{'final': True}]\n"
        ),
    }
    return SWEBenchTask(
        instance_id="swebench_pro_v2__stream_jsonl_chunk_assembler",
        repo="hanzo-cleanroom/streamrpc",
        language="python",
        difficulty="medium",
        problem_statement=(
            "Buffer raw bytes in FrameBuffer until newline boundaries so split multi-byte "
            "UTF-8 sequences decode intact, enforce max_frame_bytes, and add flush()."
        ),
        base_files=base_files,
        test_files=test_files,
        gold_patch_files=gold_files,
        gold_marker="GOLD_SENTINEL_STREAM_RPC_2B8F",
        fail_to_pass_tests=[
            "tests/test_stream_rpc.py::test_multibyte_utf8_split_across_chunks",
            "tests/test_stream_rpc.py::test_max_frame_bytes_enforced",
            "tests/test_stream_rpc.py::test_flush_trailing_record_without_newline",
        ],
        pass_to_pass_tests=[
            "tests/test_stream_rpc.py::test_basic_jsonl_chunks",
        ],
        protected_test_paths=["tests/test_stream_rpc.py"],
        expected_touched_files=["streamrpc/buffer.py", "streamrpc/decoder.py"],
    )


def _task_semver_constraint_solver() -> SWEBenchTask:
    base_files = {
        "semversolver/__init__.py": (
            "from .version import SemVer\n"
            "from .constraints import matches_constraint, resolve_highest\n"
            "__all__ = ['SemVer', 'matches_constraint', 'resolve_highest']\n"
        ),
        "semversolver/version.py": (
            "from dataclasses import dataclass\n"
            "from typing import Optional\n\n"
            "@dataclass(frozen=True, order=True)\n"
            "class SemVer:\n"
            "    major: int\n"
            "    minor: int\n"
            "    patch: int\n"
            "    prerelease: Optional[str] = None\n\n"
            "    @classmethod\n"
            "    def parse(cls, text: str) -> 'SemVer':\n"
            "        core, _, pre = text.strip().lstrip('v').partition('-')\n"
            "        maj, min_, pat = [int(x) for x in core.split('.')]\n"
            "        return cls(maj, min_, pat, pre or None)\n"
        ),
        "semversolver/constraints.py": (
            "from typing import Iterable, Optional\n"
            "from .version import SemVer\n\n"
            "def matches_constraint(ver: SemVer, expr: str) -> bool:\n"
            "    expr = expr.strip()\n"
            "    if expr.startswith('^'):\n"
            "        base = SemVer.parse(expr[1:])\n"
            "        # BUG: ignores ^0.x minor pinning and prerelease exclusion\n"
            "        return ver.major == base.major and ver >= base\n"
            "    if expr.startswith('>='):\n"
            "        return ver >= SemVer.parse(expr[2:])\n"
            "    return ver == SemVer.parse(expr)\n\n"
            "def resolve_highest(versions: Iterable[str], expr: str) -> Optional[str]:\n"
            "    parsed = [SemVer.parse(v) for v in versions]\n"
            "    matched = [v for v in parsed if matches_constraint(v, expr)]\n"
            "    if not matched:\n"
            "        return None\n"
            "    best = max(matched)\n"
            "    suffix = f'-{best.prerelease}' if best.prerelease else ''\n"
            "    return f'{best.major}.{best.minor}.{best.patch}{suffix}'\n"
        ),
    }
    gold_files = {
        "semversolver/version.py": (
            "from dataclasses import dataclass\n"
            "from typing import Optional, Tuple, Union\n\n"
            "@dataclass(frozen=True)\n"
            "class SemVer:\n"
            "    major: int\n"
            "    minor: int\n"
            "    patch: int\n"
            "    prerelease: Optional[str] = None\n\n"
            "    @classmethod\n"
            "    def parse(cls, text: str) -> 'SemVer':\n"
            "        core, _, pre = text.strip().lstrip('v').partition('-')\n"
            "        maj, min_, pat = [int(x) for x in core.split('.')]\n"
            "        return cls(maj, min_, pat, pre or None)\n\n"
            "    def _pre_key(self) -> Tuple[int, Tuple[Tuple[int, Union[int, str]], ...]]:\n"
            "        if self.prerelease is None:\n"
            "            return (1, ())\n"
            "        parts = []\n"
            "        for token in self.prerelease.split('.'):\n"
            "            if token.isdigit():\n"
            "                parts.append((0, int(token)))\n"
            "            else:\n"
            "                parts.append((1, token))\n"
            "        return (0, tuple(parts))\n\n"
            "    def _sort_key(self):\n"
            "        return (self.major, self.minor, self.patch, self._pre_key())\n\n"
            "    def __lt__(self, other: 'SemVer') -> bool:\n"
            "        return self._sort_key() < other._sort_key()\n\n"
            "    def __le__(self, other: 'SemVer') -> bool:\n"
            "        return self._sort_key() <= other._sort_key()\n\n"
            "    def __gt__(self, other: 'SemVer') -> bool:\n"
            "        return self._sort_key() > other._sort_key()\n\n"
            "    def __ge__(self, other: 'SemVer') -> bool:\n"
            "        return self._sort_key() >= other._sort_key()\n"
        ),
        "semversolver/constraints.py": (
            "from typing import Iterable, Optional\n"
            "from .version import SemVer\n\n"
            "def _caret_upper(base: SemVer) -> SemVer:\n"
            "    if base.major > 0:\n"
            "        return SemVer(base.major + 1, 0, 0)\n"
            "    if base.minor > 0:\n"
            "        return SemVer(0, base.minor + 1, 0)\n"
            "    return SemVer(0, 0, base.patch + 1)\n\n"
            "def matches_constraint(ver: SemVer, expr: str) -> bool:\n"
            "    expr = expr.strip()\n"
            "    if expr.startswith('^'):\n"
            "        base = SemVer.parse(expr[1:])\n"
            "        if ver.prerelease is not None and base.prerelease is None:\n"
            "            return False\n"
            "        upper = _caret_upper(base)\n"
            "        return base <= ver < upper\n"
            "    if expr.startswith('>='):\n"
            "        base = SemVer.parse(expr[2:])\n"
            "        if ver.prerelease is not None and base.prerelease is None:\n"
            "            return False\n"
            "        return ver >= base\n"
            "    return ver == SemVer.parse(expr)\n\n"
            "def resolve_highest(versions: Iterable[str], expr: str) -> Optional[str]:\n"
            "    parsed = [SemVer.parse(v) for v in versions]\n"
            "    matched = [v for v in parsed if matches_constraint(v, expr)]\n"
            "    if not matched:\n"
            "        return None\n"
            "    best = max(matched)\n"
            "    suffix = f'-{best.prerelease}' if best.prerelease else ''\n"
            "    return f'{best.major}.{best.minor}.{best.patch}{suffix}'\n"
        ),
    }
    test_files = {
        "tests/test_semver_solver.py": (
            "from semversolver import SemVer, resolve_highest\n\n"
            "def test_caret_major_resolution():\n"
            "    assert resolve_highest(['1.0.0', '1.4.2', '2.0.0'], '^1.2.0') == '1.4.2'\n\n"
            "def test_prerelease_precedence_lower_than_release():\n"
            "    assert SemVer.parse('1.0.0-alpha.2') < SemVer.parse('1.0.0-alpha.10')\n"
            "    assert SemVer.parse('1.0.0-rc.1') < SemVer.parse('1.0.0')\n\n"
            "def test_caret_zero_major_pins_minor():\n"
            "    assert resolve_highest(['0.2.1', '0.2.9', '0.3.0'], '^0.2.0') == '0.2.9'\n"
            "    assert resolve_highest(['0.0.3', '0.0.4'], '^0.0.3') == '0.0.3'\n\n"
            "def test_caret_excludes_prereleases_by_default():\n"
            "    assert resolve_highest(['1.2.0', '1.3.0-beta.1'], '^1.2.0') == '1.2.0'\n"
        ),
    }
    partial_files = {
        # Fixes caret 0.x but leaves SemVer dataclass ordering broken for prereleases
        "semversolver/constraints.py": gold_files["semversolver/constraints.py"],
    }
    return SWEBenchTask(
        instance_id="swebench_pro_v2__semver_constraint_solver",
        repo="hanzo-cleanroom/semversolver",
        language="python",
        difficulty="hard",
        problem_statement=(
            "Implement SemVer 2.0.0 prerelease comparison (numeric vs lexical tokens, "
            "prerelease < release), caret ^0.x minor/patch pinning, and exclude prereleases "
            "from non-prerelease constraints."
        ),
        base_files=base_files,
        test_files=test_files,
        gold_patch_files=gold_files,
        partial_patch_files=partial_files,
        gold_marker="GOLD_SENTINEL_SEMVER_8A5C",
        fail_to_pass_tests=[
            "tests/test_semver_solver.py::test_prerelease_precedence_lower_than_release",
            "tests/test_semver_solver.py::test_caret_zero_major_pins_minor",
            "tests/test_semver_solver.py::test_caret_excludes_prereleases_by_default",
        ],
        pass_to_pass_tests=[
            "tests/test_semver_solver.py::test_caret_major_resolution",
        ],
        protected_test_paths=["tests/test_semver_solver.py"],
        expected_touched_files=["semversolver/version.py", "semversolver/constraints.py"],
    )


def _task_merkle_snapshot_diff() -> SWEBenchTask:
    base_files = {
        "merklediff/__init__.py": (
            "from .hasher import hash_tree\n"
            "from .diff import diff_snapshots\n"
            "__all__ = ['hash_tree', 'diff_snapshots']\n"
        ),
        "merklediff/hasher.py": (
            "import hashlib\n"
            "from typing import Dict\n\n"
            "def hash_tree(files: Dict[str, bytes]) -> str:\n"
            "    # BUG: depends on dict insertion order and omits path separator framing\n"
            "    h = hashlib.sha256()\n"
            "    for k, v in files.items():\n"
            "        h.update(k.encode('utf-8') + v)\n"
            "    return h.hexdigest()\n"
        ),
        "merklediff/diff.py": (
            "import hashlib\n"
            "from typing import Dict, List, Tuple\n\n"
            "def diff_snapshots(old: Dict[str, bytes], new: Dict[str, bytes]) -> Dict[str, List]:\n"
            "    # BUG: does not detect exact-content renames\n"
            "    added = sorted(set(new) - set(old))\n"
            "    removed = sorted(set(old) - set(new))\n"
            "    modified = sorted(k for k in (set(old) & set(new)) if old[k] != new[k])\n"
            "    return {'added': added, 'removed': removed, 'modified': modified, 'renamed': []}\n"
        ),
    }
    gold_files = {
        "merklediff/hasher.py": (
            "import hashlib\n"
            "from typing import Dict\n\n"
            "def hash_tree(files: Dict[str, bytes]) -> str:\n"
            "    h = hashlib.sha256()\n"
            "    for path in sorted(files.keys()):\n"
            "        blob_digest = hashlib.sha256(files[path]).hexdigest()\n"
            "        entry = f'{len(path)}:{path}:{len(files[path])}:{blob_digest}\\n'\n"
            "        h.update(entry.encode('utf-8'))\n"
            "    return h.hexdigest()\n"
        ),
        "merklediff/diff.py": (
            "import hashlib\n"
            "from typing import Dict, List, Tuple\n\n"
            "def _digest(data: bytes) -> str:\n"
            "    return hashlib.sha256(data).hexdigest()\n\n"
            "def diff_snapshots(old: Dict[str, bytes], new: Dict[str, bytes]) -> Dict[str, List]:\n"
            "    raw_added = sorted(set(new) - set(old))\n"
            "    raw_removed = sorted(set(old) - set(new))\n"
            "    modified = sorted(k for k in (set(old) & set(new)) if old[k] != new[k])\n\n"
            "    removed_by_hash: Dict[str, List[str]] = {}\n"
            "    for path in raw_removed:\n"
            "        removed_by_hash.setdefault(_digest(old[path]), []).append(path)\n\n"
            "    renamed: List[Tuple[str, str]] = []\n"
            "    added: List[str] = []\n"
            "    matched_removed = set()\n"
            "    for path in raw_added:\n"
            "        d = _digest(new[path])\n"
            "        candidates = removed_by_hash.get(d)\n"
            "        if candidates:\n"
            "            src = candidates.pop(0)\n"
            "            matched_removed.add(src)\n"
            "            renamed.append((src, path))\n"
            "        else:\n"
            "            added.append(path)\n"
            "    removed = [p for p in raw_removed if p not in matched_removed]\n"
            "    return {'added': added, 'removed': removed, 'modified': modified, 'renamed': renamed}\n"
        ),
    }
    test_files = {
        "tests/test_merkle_diff.py": (
            "from merklediff import diff_snapshots, hash_tree\n\n"
            "def test_modified_file_detected():\n"
            "    d = diff_snapshots({'a.py': b'1'}, {'a.py': b'2'})\n"
            "    assert d['modified'] == ['a.py']\n\n"
            "def test_hash_tree_order_independent_and_collision_safe():\n"
            "    h1 = hash_tree({'a': b'bc', 'd': b'e'})\n"
            "    h2 = hash_tree({'d': b'e', 'a': b'bc'})\n"
            "    h3 = hash_tree({'ab': b'c', 'd': b'e'})\n"
            "    assert h1 == h2\n"
            "    assert h1 != h3\n\n"
            "def test_exact_content_rename_detected():\n"
            "    d = diff_snapshots({'old.py': b'same-blob', 'del.py': b'gone'}, {'new.py': b'same-blob', 'add.py': b'fresh'})\n"
            "    assert d['renamed'] == [('old.py', 'new.py')]\n"
            "    assert d['added'] == ['add.py']\n"
            "    assert d['removed'] == ['del.py']\n"
        ),
    }
    return SWEBenchTask(
        instance_id="swebench_pro_v2__merkle_snapshot_diff",
        repo="hanzo-cleanroom/merklediff",
        language="python",
        difficulty="medium",
        problem_statement=(
            "Make hash_tree order-independent and framing-safe against path/content "
            "concatenation collisions, and detect exact-content file renames in diff_snapshots."
        ),
        base_files=base_files,
        test_files=test_files,
        gold_patch_files=gold_files,
        gold_marker="GOLD_SENTINEL_MERKLE_3D9E",
        fail_to_pass_tests=[
            "tests/test_merkle_diff.py::test_hash_tree_order_independent_and_collision_safe",
            "tests/test_merkle_diff.py::test_exact_content_rename_detected",
        ],
        pass_to_pass_tests=[
            "tests/test_merkle_diff.py::test_modified_file_detected",
        ],
        protected_test_paths=["tests/test_merkle_diff.py"],
        expected_touched_files=["merklediff/hasher.py", "merklediff/diff.py"],
    )


def _task_retry_circuit_breaker_fsm() -> SWEBenchTask:
    base_files = {
        "resilience/__init__.py": (
            "from .breaker import CircuitBreaker, BreakerState\n"
            "from .policy import should_retry_status, compute_backoff\n"
            "__all__ = ['CircuitBreaker', 'BreakerState', 'should_retry_status', 'compute_backoff']\n"
        ),
        "resilience/policy.py": (
            "def should_retry_status(status_code: int) -> bool:\n"
            "    # BUG: retries 400/401/403 client errors!\n"
            "    return status_code >= 400\n\n"
            "def compute_backoff(attempt: int, base: float = 0.5, cap: float = 10.0) -> float:\n"
            "    return min(cap, base * (2 ** attempt))\n"
        ),
        "resilience/breaker.py": (
            "from enum import Enum\n\n"
            "class BreakerState(str, Enum):\n"
            "    CLOSED = 'closed'\n"
            "    OPEN = 'open'\n"
            "    HALF_OPEN = 'half_open'\n\n"
            "class CircuitBreaker:\n"
            "    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 10.0) -> None:\n"
            "        self.failure_threshold = failure_threshold\n"
            "        self.recovery_timeout = recovery_timeout\n"
            "        self.state = BreakerState.CLOSED\n"
            "        self.failures = 0\n"
            "        self.opened_at = 0.0\n\n"
            "    def allow_request(self, now: float) -> bool:\n"
            "        if self.state == BreakerState.OPEN:\n"
            "            if now - self.opened_at >= self.recovery_timeout:\n"
            "                self.state = BreakerState.HALF_OPEN\n"
            "                return True\n"
            "            return False\n"
            "        return True\n\n"
            "    def record_success(self) -> None:\n"
            "        self.failures = 0\n"
            "        self.state = BreakerState.CLOSED\n\n"
            "    def record_failure(self, now: float) -> None:\n"
            "        self.failures += 1\n"
            "        if self.failures >= self.failure_threshold:\n"
            "            self.state = BreakerState.OPEN\n"
            "            self.opened_at = now\n"
        ),
    }
    gold_files = {
        "resilience/policy.py": (
            "RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}\n\n"
            "def should_retry_status(status_code: int) -> bool:\n"
            "    return status_code in RETRYABLE_STATUS_CODES\n\n"
            "def compute_backoff(attempt: int, base: float = 0.5, cap: float = 10.0) -> float:\n"
            "    if attempt < 0:\n"
            "        raise ValueError('attempt must be non-negative')\n"
            "    return min(cap, base * (2 ** attempt))\n"
        ),
        "resilience/breaker.py": (
            "from enum import Enum\n\n"
            "class BreakerState(str, Enum):\n"
            "    CLOSED = 'closed'\n"
            "    OPEN = 'open'\n"
            "    HALF_OPEN = 'half_open'\n\n"
            "class CircuitBreaker:\n"
            "    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 10.0, half_open_max_probes: int = 1) -> None:\n"
            "        self.failure_threshold = failure_threshold\n"
            "        self.recovery_timeout = recovery_timeout\n"
            "        self.half_open_max_probes = half_open_max_probes\n"
            "        self.state = BreakerState.CLOSED\n"
            "        self.failures = 0\n"
            "        self.opened_at = 0.0\n"
            "        self._half_open_inflight = 0\n\n"
            "    def allow_request(self, now: float) -> bool:\n"
            "        if self.state == BreakerState.OPEN:\n"
            "            if now - self.opened_at >= self.recovery_timeout:\n"
            "                self.state = BreakerState.HALF_OPEN\n"
            "                self._half_open_inflight = 1\n"
            "                return True\n"
            "            return False\n"
            "        if self.state == BreakerState.HALF_OPEN:\n"
            "            if self._half_open_inflight >= self.half_open_max_probes:\n"
            "                return False\n"
            "            self._half_open_inflight += 1\n"
            "            return True\n"
            "        return True\n\n"
            "    def record_success(self) -> None:\n"
            "        self.failures = 0\n"
            "        self._half_open_inflight = 0\n"
            "        self.state = BreakerState.CLOSED\n\n"
            "    def record_failure(self, now: float) -> None:\n"
            "        if self.state == BreakerState.HALF_OPEN:\n"
            "            self.state = BreakerState.OPEN\n"
            "            self.opened_at = now\n"
            "            self._half_open_inflight = 0\n"
            "            return\n"
            "        self.failures += 1\n"
            "        if self.failures >= self.failure_threshold:\n"
            "            self.state = BreakerState.OPEN\n"
            "            self.opened_at = now\n"
        ),
    }
    test_files = {
        "tests/test_resilience.py": (
            "from resilience import BreakerState, CircuitBreaker, compute_backoff, should_retry_status\n\n"
            "def test_breaker_opens_after_threshold():\n"
            "    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=5.0)\n"
            "    cb.record_failure(now=1.0)\n"
            "    assert cb.state == BreakerState.CLOSED\n"
            "    cb.record_failure(now=2.0)\n"
            "    assert cb.state == BreakerState.OPEN\n"
            "    assert not cb.allow_request(now=3.0)\n"
            "    assert compute_backoff(2, base=0.5, cap=10.0) == 2.0\n\n"
            "def test_half_open_limits_concurrent_probes_and_reopens_on_probe_failure():\n"
            "    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=5.0)\n"
            "    cb.record_failure(now=0.0)\n"
            "    assert cb.allow_request(now=6.0)\n"
            "    assert cb.state == BreakerState.HALF_OPEN\n"
            "    assert not cb.allow_request(now=6.1)\n"
            "    cb.record_failure(now=6.2)\n"
            "    assert cb.state == BreakerState.OPEN\n\n"
            "def test_retry_policy_rejects_client_4xx_except_408_429():\n"
            "    assert should_retry_status(429)\n"
            "    assert should_retry_status(503)\n"
            "    assert not should_retry_status(400)\n"
            "    assert not should_retry_status(401)\n"
            "    assert not should_retry_status(404)\n"
        ),
    }
    partial_files = {
        # Fixes retry policy only, leaves half-open probe storm unfixed
        "resilience/policy.py": gold_files["resilience/policy.py"],
    }
    return SWEBenchTask(
        instance_id="swebench_pro_v2__retry_circuit_breaker_fsm",
        repo="hanzo-cleanroom/resilience",
        language="python",
        difficulty="hard",
        problem_statement=(
            "Restrict should_retry_status to transient codes (408/429/5xx) and cap "
            "HALF_OPEN concurrent probe traffic in CircuitBreaker while immediately "
            "re-opening on probe failure."
        ),
        base_files=base_files,
        test_files=test_files,
        gold_patch_files=gold_files,
        partial_patch_files=partial_files,
        gold_marker="GOLD_SENTINEL_BREAKER_6F1A",
        fail_to_pass_tests=[
            "tests/test_resilience.py::test_half_open_limits_concurrent_probes_and_reopens_on_probe_failure",
            "tests/test_resilience.py::test_retry_policy_rejects_client_4xx_except_408_429",
        ],
        pass_to_pass_tests=[
            "tests/test_resilience.py::test_breaker_opens_after_threshold",
        ],
        protected_test_paths=["tests/test_resilience.py"],
        expected_touched_files=["resilience/policy.py", "resilience/breaker.py"],
    )


def _task_sql_filter_ast_compiler() -> SWEBenchTask:
    base_files = {
        "querydsl/__init__.py": (
            "from .lexer import tokenize\n"
            "from .compiler import compile_filter, UnsafeIdentifierError\n"
            "__all__ = ['tokenize', 'compile_filter', 'UnsafeIdentifierError']\n"
        ),
        "querydsl/lexer.py": (
            "from typing import List\n\n"
            "def tokenize(expr: str) -> List[str]:\n"
            "    return expr.replace('(', ' ( ').replace(')', ' ) ').split()\n"
        ),
        "querydsl/compiler.py": (
            "from typing import Any, List, Set, Tuple\n"
            "from .lexer import tokenize\n\n"
            "class UnsafeIdentifierError(ValueError):\n"
            "    pass\n\n"
            "def compile_filter(expr: str, allowed_columns: Set[str]) -> Tuple[str, List[Any]]:\n"
            "    # BUG: interpolates raw values directly (SQLi), ignores allowed_columns, breaks on IN ()\n"
            "    tokens = tokenize(expr)\n"
            "    if len(tokens) == 3 and tokens[1] == '=':\n"
            "        return f'{tokens[0]} = {tokens[2]}', []\n"
            "    return expr, []\n"
        ),
    }
    gold_files = {
        "querydsl/lexer.py": (
            "import re\n"
            "from typing import List\n\n"
            "_TOKEN_RE = re.compile(r\"\\s*(>=|<=|!=|=|\\(|\\)|,|'[^']*'|[A-Za-z0-9_.-]+)\")\n\n"
            "def tokenize(expr: str) -> List[str]:\n"
            "    pos = 0\n"
            "    tokens: List[str] = []\n"
            "    while pos < len(expr):\n"
            "        if expr[pos].isspace():\n"
            "            pos += 1\n"
            "            continue\n"
            "        m = _TOKEN_RE.match(expr, pos)\n"
            "        if not m:\n"
            "            raise ValueError(f'unexpected token near: {expr[pos:pos+12]!r}')\n"
            "        tokens.append(m.group(1))\n"
            "        pos = m.end()\n"
            "    return tokens\n"
        ),
        "querydsl/compiler.py": (
            "from typing import Any, List, Set, Tuple\n"
            "from .lexer import tokenize\n\n"
            "class UnsafeIdentifierError(ValueError):\n"
            "    pass\n\n"
            "def _parse_literal(tok: str) -> Any:\n"
            "    if tok.startswith(\"'\") and tok.endswith(\"'\"):\n"
            "        return tok[1:-1]\n"
            "    if tok.isdigit():\n"
            "        return int(tok)\n"
            "    return tok\n\n"
            "def compile_filter(expr: str, allowed_columns: Set[str]) -> Tuple[str, List[Any]]:\n"
            "    tokens = tokenize(expr)\n"
            "    params: List[Any] = []\n"
            "    idx = 0\n\n"
            "    def parse_or() -> str:\n"
            "        nonlocal idx\n"
            "        left = parse_and()\n"
            "        while idx < len(tokens) and tokens[idx].upper() == 'OR':\n"
            "            idx += 1\n"
            "            right = parse_and()\n"
            "            left = f'({left} OR {right})'\n"
            "        return left\n\n"
            "    def parse_and() -> str:\n"
            "        nonlocal idx\n"
            "        left = parse_atom()\n"
            "        while idx < len(tokens) and tokens[idx].upper() == 'AND':\n"
            "            idx += 1\n"
            "            right = parse_atom()\n"
            "            left = f'({left} AND {right})'\n"
            "        return left\n\n"
            "    def parse_atom() -> str:\n"
            "        nonlocal idx\n"
            "        if tokens[idx] == '(':\n"
            "            idx += 1\n"
            "            inner = parse_or()\n"
            "            if idx >= len(tokens) or tokens[idx] != ')':\n"
            "                raise ValueError('unclosed parenthesis')\n"
            "            idx += 1\n"
            "            return inner\n"
            "        col = tokens[idx]\n"
            "        if col not in allowed_columns:\n"
            "            raise UnsafeIdentifierError(f'column {col!r} not allowed')\n"
            "        idx += 1\n"
            "        op = tokens[idx].upper()\n"
            "        idx += 1\n"
            "        if op == 'IN':\n"
            "            if tokens[idx] != '(':\n"
            "                raise ValueError('IN expects (')\n"
            "            idx += 1\n"
            "            vals = []\n"
            "            while idx < len(tokens) and tokens[idx] != ')':\n"
            "                if tokens[idx] != ',':\n"
            "                    vals.append(_parse_literal(tokens[idx]))\n"
            "                idx += 1\n"
            "            if idx >= len(tokens) or tokens[idx] != ')':\n"
            "                raise ValueError('unclosed IN list')\n"
            "            idx += 1\n"
            "            if not vals:\n"
            "                return '1 = 0'\n"
            "            placeholders = ', '.join(['?'] * len(vals))\n"
            "            params.extend(vals)\n"
            "            return f'{col} IN ({placeholders})'\n"
            "        if op not in {'=', '!=', '>=', '<='}:\n"
            "            raise ValueError(f'unsupported operator {op!r}')\n"
            "        val = _parse_literal(tokens[idx])\n"
            "        idx += 1\n"
            "        params.append(val)\n"
            "        return f'{col} {op} ?'\n\n"
            "    sql = parse_or()\n"
            "    if idx != len(tokens):\n"
            "        raise ValueError(f'unexpected trailing token {tokens[idx]!r}')\n"
            "    return sql, params\n"
        ),
    }
    test_files = {
        "tests/test_query_dsl.py": (
            "import pytest\n"
            "from querydsl import UnsafeIdentifierError, compile_filter, tokenize\n\n"
            "def test_tokenize_splits_parentheses():\n"
            "    toks = tokenize('(a = 1)')\n"
            "    assert toks[0] == '(' and toks[-1] == ')'\n\n"
            "def test_basic_equality_compiles_parameterized():\n"
            "    sql, params = compile_filter(\"status = 'open'\", {'status', 'priority'})\n"
            "    assert '?' in sql and params == ['open']\n\n"
            "def test_and_binds_tighter_than_or():\n"
            "    sql, params = compile_filter(\"a = 1 OR b = 2 AND c = 3\", {'a', 'b', 'c'})\n"
            "    assert sql == '(a = ? OR (b = ? AND c = ?))'\n"
            "    assert params == [1, 2, 3]\n\n"
            "def test_rejects_disallowed_column_identifier():\n"
            "    with pytest.raises(UnsafeIdentifierError):\n"
            "        compile_filter(\"secret = 'x'\", {'status'})\n\n"
            "def test_empty_in_list_compiles_to_false_predicate():\n"
            "    sql, params = compile_filter('status IN ()', {'status'})\n"
            "    assert sql == '1 = 0'\n"
            "    assert params == []\n"
        ),
    }
    partial_files = {
        # Fixes lexer only; compiler remains unparameterized
        "querydsl/lexer.py": gold_files["querydsl/lexer.py"],
    }
    return SWEBenchTask(
        instance_id="swebench_pro_v2__sql_filter_ast_compiler",
        repo="hanzo-cleanroom/querydsl",
        language="python",
        difficulty="long-horizon",
        problem_statement=(
            "Implement a proper lexer and recursive-descent filter compiler with "
            "parameterized placeholders, AND>OR precedence, allowed_columns whitelist "
            "enforcement, and safe empty IN () compilation."
        ),
        base_files=base_files,
        test_files=test_files,
        gold_patch_files=gold_files,
        partial_patch_files=partial_files,
        gold_marker="GOLD_SENTINEL_QUERYDSL_5E0C",
        fail_to_pass_tests=[
            "tests/test_query_dsl.py::test_basic_equality_compiles_parameterized",
            "tests/test_query_dsl.py::test_and_binds_tighter_than_or",
            "tests/test_query_dsl.py::test_rejects_disallowed_column_identifier",
            "tests/test_query_dsl.py::test_empty_in_list_compiles_to_false_predicate",
        ],
        pass_to_pass_tests=[
            "tests/test_query_dsl.py::test_tokenize_splits_parentheses",
        ],
        protected_test_paths=["tests/test_query_dsl.py"],
        expected_touched_files=["querydsl/lexer.py", "querydsl/compiler.py"],
    )


def _task_checkpoint_lease_reconciler() -> SWEBenchTask:
    base_files = {
        "leases/__init__.py": (
            "from .store import LeaseRecord, LeaseStore\n"
            "from .reconciler import reconcile_expired\n"
            "__all__ = ['LeaseRecord', 'LeaseStore', 'reconcile_expired']\n"
        ),
        "leases/store.py": (
            "from dataclasses import dataclass\n"
            "from typing import Dict, Optional\n\n"
            "@dataclass\n"
            "class LeaseRecord:\n"
            "    resource_id: str\n"
            "    holder: str\n"
            "    fencing_token: int\n"
            "    expires_at: float\n\n"
            "class LeaseStore:\n"
            "    def __init__(self) -> None:\n"
            "        self.records: Dict[str, LeaseRecord] = {}\n"
            "        self._seq = 0\n\n"
            "    def acquire(self, resource_id: str, holder: str, now: float, ttl: float) -> Optional[LeaseRecord]:\n"
            "        # BUG: allows stealing non-expired lease and resets fencing token\n"
            "        rec = LeaseRecord(resource_id, holder, 1, now + ttl)\n"
            "        self.records[resource_id] = rec\n"
            "        return rec\n"
        ),
        "leases/reconciler.py": (
            "from typing import List\n"
            "from .store import LeaseStore\n\n"
            "def reconcile_expired(store: LeaseStore, now: float) -> List[str]:\n"
            "    # BUG: uses strict < instead of <=, leaving exact-boundary expired leases active\n"
            "    expired = [k for k, v in store.records.items() if v.expires_at < now]\n"
            "    for k in expired:\n"
            "        del store.records[k]\n"
            "    return sorted(expired)\n"
        ),
    }
    gold_files = {
        "leases/store.py": (
            "from dataclasses import dataclass\n"
            "from typing import Dict, Optional\n\n"
            "@dataclass\n"
            "class LeaseRecord:\n"
            "    resource_id: str\n"
            "    holder: str\n"
            "    fencing_token: int\n"
            "    expires_at: float\n\n"
            "class LeaseStore:\n"
            "    def __init__(self) -> None:\n"
            "        self.records: Dict[str, LeaseRecord] = {}\n"
            "        self._seq = 0\n\n"
            "    def acquire(self, resource_id: str, holder: str, now: float, ttl: float) -> Optional[LeaseRecord]:\n"
            "        existing = self.records.get(resource_id)\n"
            "        if existing is not None and existing.expires_at > now and existing.holder != holder:\n"
            "            return None\n"
            "        self._seq += 1\n"
            "        rec = LeaseRecord(resource_id, holder, self._seq, now + ttl)\n"
            "        self.records[resource_id] = rec\n"
            "        return rec\n\n"
            "    def validate_write(self, resource_id: str, fencing_token: int, now: float) -> bool:\n"
            "        rec = self.records.get(resource_id)\n"
            "        if rec is None or rec.expires_at <= now:\n"
            "            return False\n"
            "        return rec.fencing_token == fencing_token\n"
        ),
        "leases/reconciler.py": (
            "from typing import List\n"
            "from .store import LeaseStore\n\n"
            "def reconcile_expired(store: LeaseStore, now: float) -> List[str]:\n"
            "    expired = [k for k, v in store.records.items() if v.expires_at <= now]\n"
            "    for k in expired:\n"
            "        del store.records[k]\n"
            "    return sorted(expired)\n"
        ),
    }
    test_files = {
        "tests/test_leases.py": (
            "from leases import LeaseStore, reconcile_expired\n\n"
            "def test_acquire_fresh_lease():\n"
            "    s = LeaseStore()\n"
            "    rec = s.acquire('sandbox-1', 'worker-a', now=0.0, ttl=10.0)\n"
            "    assert rec is not None and rec.holder == 'worker-a'\n\n"
            "def test_active_lease_cannot_be_stolen_and_fencing_token_monotonic():\n"
            "    s = LeaseStore()\n"
            "    r1 = s.acquire('sandbox-1', 'worker-a', now=0.0, ttl=10.0)\n"
            "    assert s.acquire('sandbox-1', 'worker-b', now=5.0, ttl=10.0) is None\n"
            "    r2 = s.acquire('sandbox-1', 'worker-b', now=10.0, ttl=10.0)\n"
            "    assert r2 is not None and r2.fencing_token > r1.fencing_token\n"
            "    assert not s.validate_write('sandbox-1', r1.fencing_token, now=11.0)\n"
            "    assert s.validate_write('sandbox-1', r2.fencing_token, now=11.0)\n\n"
            "def test_reconciler_evicts_exact_expiry_boundary():\n"
            "    s = LeaseStore()\n"
            "    s.acquire('res-a', 'w1', now=0.0, ttl=5.0)\n"
            "    assert reconcile_expired(s, now=5.0) == ['res-a']\n"
        ),
    }
    return SWEBenchTask(
        instance_id="swebench_pro_v2__checkpoint_lease_reconciler",
        repo="hanzo-cleanroom/leases",
        language="python",
        difficulty="long-horizon",
        problem_statement=(
            "Enforce exclusive non-expired lease ownership with strictly monotonic "
            "fencing tokens, add validate_write, and evict leases at exact expiry boundary."
        ),
        base_files=base_files,
        test_files=test_files,
        gold_patch_files=gold_files,
        gold_marker="GOLD_SENTINEL_LEASES_1D7B",
        fail_to_pass_tests=[
            "tests/test_leases.py::test_active_lease_cannot_be_stolen_and_fencing_token_monotonic",
            "tests/test_leases.py::test_reconciler_evicts_exact_expiry_boundary",
        ],
        pass_to_pass_tests=[
            "tests/test_leases.py::test_acquire_fresh_lease",
        ],
        protected_test_paths=["tests/test_leases.py"],
        expected_touched_files=["leases/store.py", "leases/reconciler.py"],
    )


def build_cleanroom_v2_subset() -> List[SWEBenchTask]:
    """Return the complete 9-task clean-room Apache-2.0 SWE-bench Pro V2 subset."""
    return [
        _task_vfs_cache_invalidation(),
        _task_dag_scheduler_cycle_priority(),
        _task_token_bucket_burst_refill(),
        _task_stream_jsonl_chunk_assembler(),
        _task_semver_constraint_solver(),
        _task_merkle_snapshot_diff(),
        _task_retry_circuit_breaker_fsm(),
        _task_sql_filter_ast_compiler(),
        _task_checkpoint_lease_reconciler(),
    ]
