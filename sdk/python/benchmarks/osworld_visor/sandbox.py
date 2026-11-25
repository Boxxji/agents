"""Isolated POSIX subprocess sandbox manager, Visor CAS pool, security & cleanup auditor.

Models two real local sandbox execution profiles aligned with Hanzo's canonical
``vfs -> Hanzo S3 -> Visor`` storage architecture:

1. ``cold_sandbox``: provisions a fresh workspace, isolated ``HOME``/``TMPDIR``/
   ``XDG_*`` directories, and copies all desktop assets and mutable task seed
   files from scratch before launching a fresh isolated subprocess.
2. ``warm_cas_visor``: caches immutable desktop asset pools once per content
   digest inside a read-only content-addressed store (``0o444``), mounts those
   immutable assets via zero-copy hardlinks, and always materializes mutable
   task seed files as detached writable per-task copies so task mutations never
   corrupt shared CAS templates.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import resource
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .profiler import (
    read_proc_pid_vmhwm_kib,
    read_proc_self_cmdline,
    rusage_maxrss_to_kib,
)
from .schemas import (
    CleanupAuditResult,
    OSWorldTaskSpec,
    SandboxProfile,
    SecurityAuditResult,
)

# Read-only desktop asset ballast scales: (num_files, bytes_per_file).
# - "tiny": ~2 KiB -> demonstrates honest cold-cache Visor CAS crossover regime.
# - "medium": ~3.75 MiB -> realistic desktop asset/template pool per sandbox.
# - "large": ~12.5 MiB -> heavy desktop/documentation asset pool.
WORKSPACE_SCALES: Dict[str, Tuple[int, int]] = {
    "tiny": (4, 512),
    "medium": (160, 24_576),
    "large": (400, 32_768),
}

# Environment variables stripped from every sandboxed subprocess.
FORBIDDEN_ENV_PREFIXES: Tuple[str, ...] = (
    "AWS_",
    "OPENAI_",
    "ANTHROPIC_",
    "GOOGLE_API",
    "GEMINI_",
    "GITHUB_",
    "GH_",
    "HANZO_API",
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "DYLD_",
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "PYTEST_",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _safe_rel_path(rel: str) -> Path:
    """Validate and return a clean relative path without stripping leading dots."""
    if not rel or os.path.isabs(rel):
        raise ValueError(f"Path must be relative and non-empty: {rel!r}")
    normalized = Path(rel)
    if any(part == ".." for part in normalized.parts):
        raise ValueError(f"Path traversal forbidden in relative path: {rel!r}")
    return normalized


def _is_within_dir(candidate: Path, root: Path) -> bool:
    """Return True iff realpath(candidate) is equal to or inside realpath(root)."""
    real_root = os.path.realpath(str(root))
    real_cand = os.path.realpath(str(candidate))
    try:
        return os.path.commonpath([real_root, real_cand]) == real_root
    except ValueError:
        return False


def _deterministic_ballast_bytes(index: int, size: int) -> bytes:
    """Generate deterministic non-empty desktop asset ballast (templates/docs)."""
    header = f"# desktop-asset-{index:04d}\n".encode("ascii")
    if size <= len(header):
        return header[:size]
    pattern = f"asset_block_{index:04d}_visor_cas_template_line\n".encode("ascii")
    reps = ((size - len(header)) // len(pattern)) + 1
    body = (pattern * reps)[: size - len(header)]
    return header + body


@dataclass
class SubprocessExecResult:
    """Real OS-measured result of a single isolated subprocess invocation."""

    returncode: int
    stdout: str
    stderr: str
    wall_ms: float
    peak_rss_kib: int
    rss_source: str
    cpu_user_ms: float
    cpu_system_ms: float
    inblock: int
    oublock: int
    timed_out: bool = False
    pgid: int = 0
    escape_rejected: bool = False
    escape_reason: str = ""


@dataclass
class SandboxInstance:
    """Live isolated sandbox handle for one task rollout."""

    sandbox_id: str
    task_id: str
    profile: SandboxProfile
    sandbox_root: Path
    workspace: Path
    home_dir: Path
    tmp_dir: Path
    xdg_config_dir: Path
    xdg_cache_dir: Path
    external_canary_dir: Path
    external_canary_path: Path
    external_canary_sha256: str
    provision_ms: float
    bytes_copied: int
    hardlinked_files: int
    cas_cache_hit: bool
    cas_template_dir: Optional[Path] = None
    cas_ro_manifest: Dict[str, str] = field(default_factory=dict)
    spawned_pgids: Set[int] = field(default_factory=set)
    escape_events: List[str] = field(default_factory=list)
    monitored_ports: Set[int] = field(default_factory=set)
    initial_root_entries: Set[str] = field(default_factory=set)


@dataclass
class _CASTemplateEntry:
    cache_key: str
    template_dir: Path
    ro_layer: Path
    seed_layer: Path
    ro_manifest: Dict[str, str]


class VisorSandboxManager:
    """Manages isolated cold and warm-CAS Visor sandboxes with real OS audits."""

    def __init__(
        self,
        *,
        base_dir: Optional[Path] = None,
        workspace_scale: str = "medium",
        enable_rlimits: bool = True,
    ) -> None:
        if workspace_scale not in WORKSPACE_SCALES:
            raise ValueError(
                f"Unknown workspace_scale {workspace_scale!r}; choose from {sorted(WORKSPACE_SCALES)}"
            )
        self.workspace_scale = workspace_scale
        self.enable_rlimits = enable_rlimits
        self._owns_base_dir = base_dir is None
        if base_dir is None:
            raw_base = Path(tempfile.mkdtemp(prefix="osworld_visor_"))
        else:
            raw_base = Path(base_dir)
            raw_base.mkdir(parents=True, exist_ok=True)
        # Resolve symlinked TMPDIRs (e.g. macOS /var/folders -> /private/var/folders).
        self.base_dir = Path(os.path.realpath(str(raw_base)))
        self.cas_root = self.base_dir / "cas_pool"
        self.runs_root = self.base_dir / "runs"
        self.cas_root.mkdir(parents=True, exist_ok=True)
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self._cas_lock = threading.Lock()
        self._cas_entries: Dict[str, _CASTemplateEntry] = {}

    def close(self) -> None:
        """Remove temporary manager storage when owned by this instance."""
        if self._owns_base_dir and self.base_dir.exists():
            self._force_rmtree(self.base_dir, allow_chmod_files=True)

    def __enter__(self) -> "VisorSandboxManager":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _compute_cas_key(self, task: OSWorldTaskSpec) -> str:
        payload = {
            "task_id": task.task_id,
            "workspace_scale": self.workspace_scale,
            "seed_files": task.seed_files,
            "readonly_assets": task.readonly_assets,
            "initial_modes": task.initial_modes,
            "setup_git_repo": task.setup_git_repo,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(canonical).hexdigest()[:24]

    def _init_git_repo(self, target_dir: Path, spec: Dict[str, Any]) -> None:
        """Initialize a clean deterministic local git repo for git workflow tasks."""
        repo_dir = target_dir / _safe_rel_path(spec["subdir"])
        repo_dir.mkdir(parents=True, exist_ok=True)
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(target_dir),
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "Benchmark Bot",
            "GIT_AUTHOR_EMAIL": "bench@hanzo.local",
            "GIT_COMMITTER_NAME": "Benchmark Bot",
            "GIT_COMMITTER_EMAIL": "bench@hanzo.local",
        }

        def _git(*args: str) -> None:
            subprocess.run(
                ["git", *args],
                cwd=str(repo_dir),
                env=env,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        _git("init", "-b", "main")
        for rel, content in spec["base_files"].items():
            fp = repo_dir / _safe_rel_path(rel)
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(content, encoding="utf-8")
        _git("add", "-A")
        _git("commit", "-m", spec.get("base_commit_msg", "chore: initial release"))

        feature_branch = spec.get("feature_branch")
        if feature_branch:
            _git("checkout", "-b", feature_branch)
            for rel, content in spec.get("feature_files", {}).items():
                fp = repo_dir / _safe_rel_path(rel)
                fp.parent.mkdir(parents=True, exist_ok=True)
                fp.write_text(content, encoding="utf-8")
            _git("add", "-A")
            _git("commit", "-m", spec.get("feature_commit_msg", "fix: apply feature patch"))
            _git("checkout", "main")

        for rel, content in spec.get("untracked_files", {}).items():
            fp = repo_dir / _safe_rel_path(rel)
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(content, encoding="utf-8")

    def _write_ballast_files(self, ro_root: Path) -> Tuple[int, Dict[str, str]]:
        """Write deterministic read-only desktop ballast assets."""
        num_files, file_size = WORKSPACE_SCALES[self.workspace_scale]
        pool_dir = ro_root / "assets" / "desktop_pool"
        pool_dir.mkdir(parents=True, exist_ok=True)
        bytes_written = 0
        manifest: Dict[str, str] = {}
        for idx in range(num_files):
            subdir = pool_dir / f"pack_{idx % 8:02d}"
            subdir.mkdir(parents=True, exist_ok=True)
            fpath = subdir / f"template_{idx:04d}.dat"
            data = _deterministic_ballast_bytes(idx, file_size)
            fpath.write_bytes(data)
            bytes_written += len(data)
            rel = str(fpath.relative_to(ro_root).as_posix())
            manifest[rel] = _sha256_bytes(data)
        return bytes_written, manifest

    def _get_or_build_cas_template(
        self, task: OSWorldTaskSpec
    ) -> Tuple[_CASTemplateEntry, bool, int]:
        """Return (cas_entry, cache_hit, build_bytes_written)."""
        cache_key = self._compute_cas_key(task)
        with self._cas_lock:
            existing = self._cas_entries.get(cache_key)
            if existing is not None and existing.template_dir.exists():
                return existing, True, 0

            template_dir = self.cas_root / f"{task.task_id}_{cache_key}"
            if template_dir.exists():
                self._force_rmtree(template_dir, allow_chmod_files=True)
            ro_layer = template_dir / "ro_layer"
            seed_layer = template_dir / "seed_layer"
            ro_layer.mkdir(parents=True, exist_ok=True)
            seed_layer.mkdir(parents=True, exist_ok=True)

            build_bytes = 0
            ballast_bytes, ro_manifest = self._write_ballast_files(ro_layer)
            build_bytes += ballast_bytes

            for rel, content in task.readonly_assets.items():
                rel_p = _safe_rel_path(rel)
                target = ro_layer / rel_p
                target.parent.mkdir(parents=True, exist_ok=True)
                raw = content.encode("utf-8")
                target.write_bytes(raw)
                build_bytes += len(raw)
                ro_manifest[rel_p.as_posix()] = _sha256_bytes(raw)

            # Lock immutable CAS layer files to read-only 0o444.
            for rel_str in ro_manifest:
                os.chmod(ro_layer / rel_str, 0o444)

            for rel, content in task.seed_files.items():
                rel_p = _safe_rel_path(rel)
                target = seed_layer / rel_p
                target.parent.mkdir(parents=True, exist_ok=True)
                raw = content.encode("utf-8")
                target.write_bytes(raw)
                build_bytes += len(raw)

            if task.setup_git_repo:
                self._init_git_repo(seed_layer, task.setup_git_repo)
                for p in seed_layer.rglob("*"):
                    if p.is_file():
                        build_bytes += p.stat().st_size

            entry = _CASTemplateEntry(
                cache_key=cache_key,
                template_dir=template_dir,
                ro_layer=ro_layer,
                seed_layer=seed_layer,
                ro_manifest=ro_manifest,
            )
            self._cas_entries[cache_key] = entry
            return entry, False, build_bytes

    def warm_cas_pool(self, tasks: List[OSWorldTaskSpec]) -> int:
        """Pre-populate immutable Visor CAS base snapshots for all tasks."""
        built = 0
        for task in tasks:
            _, hit, _ = self._get_or_build_cas_template(task)
            if not hit:
                built += 1
        return built

    def provision(
        self, task: OSWorldTaskSpec, profile: SandboxProfile
    ) -> SandboxInstance:
        """Provision an isolated sandbox directory tree for a single rollout."""
        t0 = time.perf_counter_ns()
        sandbox_id = f"{task.task_id}_{profile.value}_{uuid.uuid4().hex[:10]}"
        sandbox_root = Path(os.path.realpath(str(self.runs_root / sandbox_id)))
        workspace = sandbox_root / "workspace"
        home_dir = sandbox_root / "home"
        tmp_dir = sandbox_root / "tmp"
        xdg_config_dir = sandbox_root / "xdg_config"
        xdg_cache_dir = sandbox_root / "xdg_cache"
        external_canary_dir = sandbox_root / "host_guard"

        for d in (
            workspace,
            home_dir,
            tmp_dir,
            xdg_config_dir,
            xdg_cache_dir,
            external_canary_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

        external_canary_path = external_canary_dir / "secret_canary.txt"
        canary_data = f"canary-{sandbox_id}-do-not-touch\n".encode("utf-8")
        external_canary_path.write_bytes(canary_data)
        external_canary_sha = _sha256_bytes(canary_data)

        bytes_copied = 0
        hardlinked_files = 0
        cas_cache_hit = False
        cas_template_dir: Optional[Path] = None
        cas_ro_manifest: Dict[str, str] = {}

        if profile == SandboxProfile.COLD_SANDBOX:
            ballast_bytes, _ = self._write_ballast_files(workspace)
            bytes_copied += ballast_bytes

            for rel, content in task.readonly_assets.items():
                rel_p = _safe_rel_path(rel)
                target = workspace / rel_p
                target.parent.mkdir(parents=True, exist_ok=True)
                raw = content.encode("utf-8")
                target.write_bytes(raw)
                os.chmod(target, 0o444)
                bytes_copied += len(raw)

            for rel, content in task.seed_files.items():
                rel_p = _safe_rel_path(rel)
                target = workspace / rel_p
                target.parent.mkdir(parents=True, exist_ok=True)
                raw = content.encode("utf-8")
                target.write_bytes(raw)
                bytes_copied += len(raw)

            if task.setup_git_repo:
                self._init_git_repo(workspace, task.setup_git_repo)
                git_sub = workspace / _safe_rel_path(task.setup_git_repo["subdir"])
                for p in git_sub.rglob("*"):
                    if p.is_file():
                        bytes_copied += p.stat().st_size

        elif profile == SandboxProfile.WARM_CAS_VISOR:
            entry, cas_cache_hit, build_bytes = self._get_or_build_cas_template(task)
            cas_template_dir = entry.template_dir
            cas_ro_manifest = dict(entry.ro_manifest)
            if not cas_cache_hit:
                # Account honestly for cold CAS materialization writes!
                bytes_copied += build_bytes

            # 1. Hardlink immutable read-only desktop assets (zero-copy CoW).
            for rel_str in entry.ro_manifest:
                src = entry.ro_layer / rel_str
                dst = workspace / rel_str
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.link(src, dst)
                hardlinked_files += 1

            # 2. ALWAYS detach mutable seed files as private writable copies.
            for root, dirs, files in os.walk(entry.seed_layer):
                rel_root = Path(root).relative_to(entry.seed_layer)
                dst_dir = workspace / rel_root
                dst_dir.mkdir(parents=True, exist_ok=True)
                for dname in dirs:
                    (dst_dir / dname).mkdir(parents=True, exist_ok=True)
                for fname in files:
                    src_f = Path(root) / fname
                    dst_f = dst_dir / fname
                    shutil.copy2(src_f, dst_f)
                    bytes_copied += dst_f.stat().st_size
        else:
            raise ValueError(f"Unsupported profile: {profile}")

        # Apply task-specific initial POSIX modes on detached workspace files.
        for rel, mode in task.initial_modes.items():
            target = workspace / _safe_rel_path(rel)
            if target.exists():
                os.chmod(target, mode)

        provision_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        initial_root_entries = {p.name for p in sandbox_root.iterdir()}

        return SandboxInstance(
            sandbox_id=sandbox_id,
            task_id=task.task_id,
            profile=profile,
            sandbox_root=sandbox_root,
            workspace=workspace,
            home_dir=home_dir,
            tmp_dir=tmp_dir,
            xdg_config_dir=xdg_config_dir,
            xdg_cache_dir=xdg_cache_dir,
            external_canary_dir=external_canary_dir,
            external_canary_path=external_canary_path,
            external_canary_sha256=external_canary_sha,
            provision_ms=provision_ms,
            bytes_copied=bytes_copied,
            hardlinked_files=hardlinked_files,
            cas_cache_hit=cas_cache_hit,
            cas_template_dir=cas_template_dir,
            cas_ro_manifest=cas_ro_manifest,
            initial_root_entries=initial_root_entries,
        )

    def build_sandbox_env(self, sandbox: SandboxInstance) -> Dict[str, str]:
        """Construct an environment-scrubbed POSIX env isolated to the sandbox."""
        py_bin_dir = str(Path(sys.executable).resolve().parent)
        safe_path_parts = [py_bin_dir, "/usr/local/bin", "/usr/bin", "/bin"]
        # Preserve unique existing directories in order.
        seen: Set[str] = set()
        clean_path: List[str] = []
        for part in safe_path_parts:
            if part and part not in seen:
                seen.add(part)
                clean_path.append(part)

        return {
            "PATH": os.pathsep.join(clean_path),
            "HOME": str(sandbox.home_dir),
            "TMPDIR": str(sandbox.tmp_dir),
            "TEMP": str(sandbox.tmp_dir),
            "TMP": str(sandbox.tmp_dir),
            "XDG_CONFIG_HOME": str(sandbox.xdg_config_dir),
            "XDG_CACHE_HOME": str(sandbox.xdg_cache_dir),
            "XDG_DATA_HOME": str(sandbox.home_dir / ".local" / "share"),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "Sandbox Agent",
            "GIT_AUTHOR_EMAIL": "agent@hanzo.local",
            "GIT_COMMITTER_NAME": "Sandbox Agent",
            "GIT_COMMITTER_EMAIL": "agent@hanzo.local",
            "HANZO_SANDBOX_ID": sandbox.sandbox_id,
            "HANZO_WORKSPACE": str(sandbox.workspace),
        }

    def safe_resolve_workspace_path(
        self, sandbox: SandboxInstance, rel_path: str
    ) -> Path:
        """Resolve a path inside ``sandbox.workspace`` or fail closed on escape."""
        candidate = sandbox.workspace / rel_path
        if not _is_within_dir(candidate, sandbox.workspace):
            msg = f"path_traversal_blocked:{rel_path}"
            sandbox.escape_events.append(msg)
            raise PermissionError(msg)
        return Path(os.path.realpath(str(candidate)))

    def _make_preexec_rlimits(self):
        if not self.enable_rlimits:
            return None

        def _apply() -> None:
            # Cap single file write size at 64 MiB to bound runaway disk writes.
            try:
                soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
                limit = 64 * 1024 * 1024
                if hard == resource.RLIM_INFINITY or hard >= limit:
                    new_soft = limit if soft == resource.RLIM_INFINITY else min(soft, limit)
                    resource.setrlimit(resource.RLIMIT_FSIZE, (new_soft, hard))
            except (ValueError, OSError):
                pass

        return _apply

    def exec_in_sandbox(
        self,
        sandbox: SandboxInstance,
        command: List[str],
        *,
        working_subdir: str = "",
        timeout_s: float = 20.0,
    ) -> SubprocessExecResult:
        """Execute a real command inside an isolated subprocess and measure via os.wait4."""
        if working_subdir:
            try:
                cwd_path = self.safe_resolve_workspace_path(sandbox, working_subdir)
            except PermissionError as exc:
                return SubprocessExecResult(
                    returncode=126,
                    stdout="",
                    stderr=str(exc),
                    wall_ms=0.0,
                    peak_rss_kib=0,
                    rss_source="rejected_before_launch",
                    cpu_user_ms=0.0,
                    cpu_system_ms=0.0,
                    inblock=0,
                    oublock=0,
                    escape_rejected=True,
                    escape_reason=str(exc),
                )
        else:
            cwd_path = sandbox.workspace

        env = self.build_sandbox_env(sandbox)
        t0 = time.perf_counter_ns()
        proc = subprocess.Popen(
            command,
            cwd=str(cwd_path),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            preexec_fn=self._make_preexec_rlimits(),
        )
        pgid = proc.pid
        sandbox.spawned_pgids.add(pgid)

        stdout_chunks: List[bytes] = []
        stderr_chunks: List[bytes] = []
        sampled_vmhwm_kib: int = 0

        def _read_stream(stream: Any, dest: List[bytes]) -> None:
            try:
                dest.append(stream.read())
            finally:
                try:
                    stream.close()
                except OSError:
                    pass

        t_out = threading.Thread(target=_read_stream, args=(proc.stdout, stdout_chunks), daemon=True)
        t_err = threading.Thread(target=_read_stream, args=(proc.stderr, stderr_chunks), daemon=True)
        t_out.start()
        t_err.start()

        timed_out = False
        rusage: Optional[resource.struct_rusage] = None
        exit_status = 0

        parent_cmdline = read_proc_self_cmdline()
        deadline = time.monotonic() + max(0.1, timeout_s)
        while True:
            live_hwm = read_proc_pid_vmhwm_kib(
                proc.pid, parent_cmdline=parent_cmdline
            )
            if live_hwm is not None and live_hwm > sampled_vmhwm_kib:
                sampled_vmhwm_kib = live_hwm

            try:
                waited_pid, status, ru = os.wait4(proc.pid, os.WNOHANG)
            except ChildProcessError:
                break

            if waited_pid == proc.pid:
                exit_status = status
                rusage = ru
                break

            if time.monotonic() >= deadline:
                timed_out = True
                self._terminate_pgid(pgid)
                try:
                    _, exit_status, rusage = os.wait4(proc.pid, 0)
                except ChildProcessError:
                    pass
                break

            time.sleep(0.001)

        wall_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        t_out.join(timeout=2.0)
        t_err.join(timeout=2.0)

        if timed_out:
            returncode = -9
        else:
            returncode = os.waitstatus_to_exitcode(exit_status)
        proc.returncode = returncode

        ru_rss_kib = rusage_maxrss_to_kib(rusage.ru_maxrss) if rusage is not None else 0
        # Prefer post-execve /proc/<pid>/status VmHWM on Linux to avoid fork()
        # parent-RSS inheritance inflating os.wait4().ru_maxrss; fall back to
        # os.wait4() on non-procfs hosts (e.g. macOS) or sub-millisecond exits.
        if sampled_vmhwm_kib > 0:
            peak_rss_kib = sampled_vmhwm_kib
            rss_source = "proc_vmhwm"
        elif ru_rss_kib > 0:
            peak_rss_kib = ru_rss_kib
            rss_source = "os.wait4"
        else:
            peak_rss_kib = 0
            rss_source = "unavailable"

        cpu_user_ms = (rusage.ru_utime * 1000.0) if rusage is not None else 0.0
        cpu_system_ms = (rusage.ru_stime * 1000.0) if rusage is not None else 0.0
        inblock = int(rusage.ru_inblock) if rusage is not None else 0
        oublock = int(rusage.ru_oublock) if rusage is not None else 0

        return SubprocessExecResult(
            returncode=returncode,
            stdout=b"".join(stdout_chunks).decode("utf-8", errors="replace"),
            stderr=b"".join(stderr_chunks).decode("utf-8", errors="replace"),
            wall_ms=wall_ms,
            peak_rss_kib=peak_rss_kib,
            rss_source=rss_source,
            cpu_user_ms=cpu_user_ms,
            cpu_system_ms=cpu_system_ms,
            inblock=inblock,
            oublock=oublock,
            timed_out=timed_out,
            pgid=pgid,
        )

    def measure_cold_start_probe(self, sandbox: SandboxInstance) -> SubprocessExecResult:
        """Run a real minimal isolated subprocess probe to measure cold-start boot latency."""
        return self.exec_in_sandbox(
            sandbox,
            [sys.executable, "-c", "import os; print(os.getpid())"],
            timeout_s=10.0,
        )

    def audit_security(self, sandbox: SandboxInstance) -> SecurityAuditResult:
        """Fail-closed check for path traversal, symlink escape, and external canary integrity."""
        violations: List[str] = list(sandbox.escape_events)

        # 1. Verify external canary outside workspace was not modified or deleted.
        if not sandbox.external_canary_path.exists():
            violations.append("external_canary_deleted")
        elif _sha256_file(sandbox.external_canary_path) != sandbox.external_canary_sha256:
            violations.append("external_canary_modified")

        # 2. Verify no unexpected files were created in sandbox_root outside workspace
        #    (catches relative `../` traversal writes from workspace!).
        if sandbox.sandbox_root.exists():
            current_entries = {p.name for p in sandbox.sandbox_root.iterdir()}
            extra_root = sorted(current_entries - sandbox.initial_root_entries)
            for name in extra_root:
                violations.append(f"escape_write_in_sandbox_root:{name}")

        # 3. Verify no unexpected files appeared in external_canary_dir.
        if sandbox.external_canary_dir.exists():
            for child in sorted(sandbox.external_canary_dir.iterdir()):
                if child.name != sandbox.external_canary_path.name:
                    violations.append(f"escape_write_in_host_guard:{child.name}")

        # 4. Walk workspace without following symlinks and reject any symlink whose
        #    resolved target escapes realpath(workspace).
        real_ws = os.path.realpath(str(sandbox.workspace))
        for dirpath, dirnames, filenames in os.walk(sandbox.workspace, followlinks=False):
            for entry_name in list(dirnames) + list(filenames):
                full_path = os.path.join(dirpath, entry_name)
                if os.path.islink(full_path):
                    raw_target = os.readlink(full_path)
                    if os.path.isabs(raw_target):
                        resolved = os.path.realpath(raw_target)
                    else:
                        resolved = os.path.realpath(os.path.join(dirpath, raw_target))
                    try:
                        inside = os.path.commonpath([real_ws, resolved]) == real_ws
                    except ValueError:
                        inside = False
                    if not inside:
                        rel_link = os.path.relpath(full_path, real_ws)
                        violations.append(
                            f"symlink_escape:{rel_link}->{raw_target}"
                        )

        return SecurityAuditResult(
            escaped=bool(violations),
            violations=violations,
        )

    def _find_orphan_pids(self, sandbox: SandboxInstance) -> List[int]:
        """Detect live orphan processes belonging to this sandbox via Linux /proc."""
        orphans: Set[int] = set()
        my_pid = os.getpid()
        marker = f"HANZO_SANDBOX_ID={sandbox.sandbox_id}".encode("utf-8")
        real_root = os.path.realpath(str(sandbox.sandbox_root))

        proc_dir = Path("/proc")
        if proc_dir.is_dir():
            for entry in proc_dir.iterdir():
                if not entry.name.isdigit():
                    continue
                pid = int(entry.name)
                if pid == my_pid:
                    continue

                # Check 1: /proc/<pid>/stat pgrp matches any sandbox-spawned PGID.
                matched = False
                try:
                    stat_txt = (entry / "stat").read_text(encoding="utf-8", errors="ignore")
                    rparen = stat_txt.rfind(")")
                    if rparen != -1:
                        fields = stat_txt[rparen + 2 :].split()
                        # fields[0]=state, fields[1]=ppid, fields[2]=pgrp
                        if len(fields) >= 3:
                            state = fields[0]
                            pgrp = int(fields[2])
                            if state != "Z" and pgrp in sandbox.spawned_pgids:
                                matched = True
                except (OSError, ValueError):
                    pass

                # Check 2: /proc/<pid>/environ contains our unique sandbox ID
                #          (catches processes that called setsid()!).
                if not matched:
                    try:
                        env_bytes = (entry / "environ").read_bytes()
                        if marker in env_bytes:
                            matched = True
                    except OSError:
                        pass

                # Check 3: /proc/<pid>/cwd points inside sandbox_root.
                if not matched:
                    try:
                        cwd_target = os.path.realpath(os.readlink(str(entry / "cwd")))
                        if os.path.commonpath([real_root, cwd_target]) == real_root:
                            matched = True
                    except (OSError, ValueError):
                        pass

                if matched:
                    orphans.add(pid)

        return sorted(orphans)

    def _terminate_pgid(self, pgid: int) -> None:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                return
            except PermissionError:
                return
            if sig == signal.SIGTERM:
                time.sleep(0.05)

    def _reap_orphan_pids(
        self, sandbox: SandboxInstance, orphan_pids: List[int]
    ) -> List[int]:
        """Terminate and confirm death of all detected orphan PIDs."""
        reaped: List[int] = []
        # First signal all spawned process groups.
        for pgid in sandbox.spawned_pgids:
            try:
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        for pid in orphan_pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass

        deadline = time.monotonic() + 0.3
        remaining = set(orphan_pids)
        while remaining and time.monotonic() < deadline:
            for pid in list(remaining):
                if not self._pid_alive(pid):
                    remaining.discard(pid)
                    reaped.append(pid)
            if remaining:
                time.sleep(0.01)

        # Escalate to SIGKILL for stubborn orphans.
        if remaining:
            for pgid in sandbox.spawned_pgids:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            for pid in list(remaining):
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            time.sleep(0.05)
            for pid in list(remaining):
                if not self._pid_alive(pid):
                    remaining.discard(pid)
                    reaped.append(pid)

        # Drain any direct zombie children of this runner process.
        while True:
            try:
                done_pid, _ = os.waitpid(-1, os.WNOHANG)
                if done_pid == 0:
                    break
            except ChildProcessError:
                break

        return sorted(set(reaped))

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        # Check if zombie on Linux.
        try:
            stat_txt = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="ignore")
            rparen = stat_txt.rfind(")")
            if rparen != -1 and stat_txt[rparen + 2 :].startswith("Z"):
                return False
        except OSError:
            pass
        return True

    def _find_leaked_temp_and_sockets(
        self, sandbox: SandboxInstance
    ) -> Tuple[List[str], List[str]]:
        """Inspect isolated TMPDIR, sandbox dirs, and bound loopback ports."""
        leaked_temp: List[str] = []
        lingering_sockets: List[str] = []

        if sandbox.tmp_dir.exists():
            for root, _, files in os.walk(sandbox.tmp_dir, followlinks=False):
                for fname in files:
                    full = Path(root) / fname
                    rel = full.relative_to(sandbox.tmp_dir).as_posix()
                    try:
                        st = os.lstat(full)
                        if stat.S_ISSOCK(st.st_mode):
                            lingering_sockets.append(f"unix:tmp/{rel}")
                        else:
                            leaked_temp.append(f"tmp/{rel}")
                    except OSError:
                        leaked_temp.append(f"tmp/{rel}")

        # Check for lingering unix sockets or .lock files anywhere in workspace.
        if sandbox.workspace.exists():
            for root, _, files in os.walk(sandbox.workspace, followlinks=False):
                for fname in files:
                    full = Path(root) / fname
                    rel = full.relative_to(sandbox.workspace).as_posix()
                    try:
                        st = os.lstat(full)
                        if stat.S_ISSOCK(st.st_mode):
                            lingering_sockets.append(f"unix:workspace/{rel}")
                    except OSError:
                        pass

        # Check any TCP loopback ports reported by the task (e.g. HTTP probe report).
        probe_report = sandbox.workspace / "web" / "probe_report.json"
        if probe_report.is_file():
            try:
                data = json.loads(probe_report.read_text(encoding="utf-8"))
                port = data.get("port")
                if isinstance(port, int) and 1 <= port <= 65535:
                    sandbox.monitored_ports.add(port)
            except (OSError, ValueError, TypeError):
                pass

        for port in sorted(sandbox.monitored_ports):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.05):
                    lingering_sockets.append(f"tcp:127.0.0.1:{port}")
            except OSError:
                pass

        return sorted(leaked_temp), sorted(lingering_sockets)

    def _verify_cas_template_intact(self, sandbox: SandboxInstance) -> Tuple[bool, List[str]]:
        """Verify zero mutations occurred to immutable Visor CAS template files."""
        if not sandbox.cas_template_dir or not sandbox.cas_ro_manifest:
            return True, []
        ro_layer = sandbox.cas_template_dir / "ro_layer"
        violations: List[str] = []
        for rel_str, expected_sha in sandbox.cas_ro_manifest.items():
            target = ro_layer / rel_str
            if not target.is_file():
                violations.append(f"cas_template_missing:{rel_str}")
            elif _sha256_file(target) != expected_sha:
                violations.append(f"cas_template_mutated:{rel_str}")
        return not violations, violations

    @staticmethod
    def _force_rmtree(path: Path, *, allow_chmod_files: bool = False) -> None:
        """Delete a sandbox tree without ever chmodding hardlinked CAS files."""
        if not path.exists():
            return

        def _on_error(func: Any, fpath: str, exc_info: Any) -> None:
            parent = os.path.dirname(fpath)
            try:
                if os.path.isdir(parent):
                    os.chmod(parent, 0o755)
                if os.path.isdir(fpath):
                    os.chmod(fpath, 0o755)
                elif allow_chmod_files:
                    os.chmod(fpath, 0o644)
                func(fpath)
            except OSError:
                pass

        shutil.rmtree(path, onerror=_on_error)

    def audit_and_teardown(self, sandbox: SandboxInstance) -> Tuple[CleanupAuditResult, float]:
        """Audit orphan PIDs, leaked temp files/sockets, CAS integrity, and remove sandbox."""
        t0 = time.perf_counter_ns()
        violations: List[str] = []

        orphan_pids = self._find_orphan_pids(sandbox)
        reaped_pids: List[int] = []
        if orphan_pids:
            violations.append(f"orphan_pids_detected:{orphan_pids}")
            reaped_pids = self._reap_orphan_pids(sandbox, orphan_pids)
            still_alive = [p for p in orphan_pids if p not in reaped_pids]
            if still_alive:
                violations.append(f"orphan_pids_unreaped:{still_alive}")

        leaked_temp, lingering_sockets = self._find_leaked_temp_and_sockets(sandbox)
        if leaked_temp:
            violations.append(f"leaked_temp_files:{len(leaked_temp)}")
        if lingering_sockets:
            violations.append(f"lingering_sockets:{lingering_sockets}")

        cas_intact, cas_violations = self._verify_cas_template_intact(sandbox)
        violations.extend(cas_violations)

        # Remove sandbox directory tree without touching CAS hardlink inode permissions.
        self._force_rmtree(sandbox.sandbox_root, allow_chmod_files=False)
        workspace_removed = not sandbox.sandbox_root.exists()
        if not workspace_removed:
            violations.append("workspace_not_removed")

        cleanup_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        result = CleanupAuditResult(
            cleanup_clean=not violations,
            leaked_temp_files=leaked_temp,
            orphan_pids_detected=orphan_pids,
            orphan_pids_reaped=reaped_pids,
            lingering_sockets=lingering_sockets,
            cas_template_intact=cas_intact,
            workspace_removed=workspace_removed,
            violations=violations,
        )
        return result, cleanup_ms
