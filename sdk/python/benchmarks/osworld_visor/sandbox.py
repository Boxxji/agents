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

import ctypes
import errno
import hashlib
import json
import os
import platform
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
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

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

# Linux kernel constants for unprivileged user+mount namespace confinement.
_CLONE_NEWNS = 0x00020000
_CLONE_NEWUSER = 0x10000000
_MS_RDONLY = 1
_MS_REMOUNT = 32
_MS_BIND = 4096
_MS_REC = 16384
_MS_PRIVATE = 1 << 18
_MNT_DETACH = 2
_PR_CAPBSET_DROP = 24
_PR_SET_NO_NEW_PRIVS = 38
_SYS_PIVOT_ROOT_BY_ARCH: Dict[str, int] = {
    "x86_64": 155,
    "amd64": 155,
    "aarch64": 41,
    "arm64": 41,
}
_RO_SYSTEM_DIRS: Tuple[bytes, ...] = (
    b"bin",
    b"etc",
    b"lib",
    b"lib64",
    b"nix",
    b"opt",
    b"proc",
    b"sbin",
    b"usr",
)
_DEV_NODES: Tuple[bytes, ...] = (
    b"null",
    b"zero",
    b"full",
    b"random",
    b"urandom",
    b"tty",
)


def _load_linux_libc() -> Optional[Tuple[Any, int]]:
    """Return ``(libc, sys_pivot_root)`` on supported Linux hosts, else None."""
    if sys.platform != "linux":
        return None
    sys_pivot_root = _SYS_PIVOT_ROOT_BY_ARCH.get(platform.machine().lower())
    if sys_pivot_root is None:
        return None
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.unshare.argtypes = [ctypes.c_int]
        libc.unshare.restype = ctypes.c_int
        libc.mount.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_ulong,
            ctypes.c_char_p,
        ]
        libc.mount.restype = ctypes.c_int
        libc.umount2.argtypes = [ctypes.c_char_p, ctypes.c_int]
        libc.umount2.restype = ctypes.c_int
        libc.prctl.argtypes = [
            ctypes.c_int,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
        ]
        libc.prctl.restype = ctypes.c_int
        libc.syscall.restype = ctypes.c_long
        return libc, sys_pivot_root
    except OSError:
        return None


def _build_linux_mount_ns_preexec(
    libc: Any,
    sys_pivot_root: int,
    *,
    nsroot_b: bytes,
    workdir_b: bytes,
    rw_dirs_b: Tuple[bytes, ...],
    uid: int,
    gid: int,
    enable_rlimits: bool,
) -> Callable[[], None]:
    """Build a post-fork preexec_fn enforcing a read-only pivot_root mount namespace."""
    uid_map = f"0 {uid} 1\n".encode("ascii")
    gid_map = f"0 {gid} 1\n".encode("ascii")

    def _confine() -> None:
        if libc.unshare(_CLONE_NEWUSER | _CLONE_NEWNS) != 0:
            raise OSError(ctypes.get_errno(), "unshare(CLONE_NEWUSER|CLONE_NEWNS) failed")
        with open("/proc/self/setgroups", "wb") as fh:
            fh.write(b"deny\n")
        with open("/proc/self/uid_map", "wb") as fh:
            fh.write(uid_map)
        with open("/proc/self/gid_map", "wb") as fh:
            fh.write(gid_map)
        if libc.mount(None, b"/", None, _MS_REC | _MS_PRIVATE, None) != 0:
            raise OSError(ctypes.get_errno(), "mount(MS_PRIVATE) failed")
        os.makedirs(nsroot_b, exist_ok=True)
        if libc.mount(b"tmpfs", nsroot_b, b"tmpfs", 0, b"size=16m") != 0:
            raise OSError(ctypes.get_errno(), "mount(tmpfs) failed")
        for d in _RO_SYSTEM_DIRS:
            src = b"/" + d
            dst = nsroot_b + b"/" + d
            if os.path.islink(src):
                os.symlink(os.readlink(src), dst)
            elif os.path.isdir(src):
                os.mkdir(dst)
                if libc.mount(src, dst, None, _MS_BIND | _MS_REC, None) == 0:
                    libc.mount(None, dst, None, _MS_REMOUNT | _MS_BIND | _MS_RDONLY, None)
        dev_dst = nsroot_b + b"/dev"
        os.mkdir(dev_dst)
        for df in _DEV_NODES:
            src = b"/dev/" + df
            dst = dev_dst + b"/" + df
            if os.path.exists(src):
                open(dst, "wb").close()
                libc.mount(src, dst, None, _MS_BIND, None)
        os.symlink(b"/proc/self/fd", dev_dst + b"/fd")
        os.symlink(b"/proc/self/fd/0", dev_dst + b"/stdin")
        os.symlink(b"/proc/self/fd/1", dev_dst + b"/stdout")
        os.symlink(b"/proc/self/fd/2", dev_dst + b"/stderr")
        for rw in rw_dirs_b:
            dst = nsroot_b + rw
            os.makedirs(dst, exist_ok=True)
            if libc.mount(rw, dst, None, _MS_BIND, None) != 0:
                raise OSError(ctypes.get_errno(), f"bind mount failed for {rw!r}")
        oldroot = nsroot_b + b"/.oldroot"
        os.mkdir(oldroot)
        os.chdir(nsroot_b)
        if (
            libc.syscall(
                ctypes.c_long(sys_pivot_root),
                ctypes.c_char_p(b"."),
                ctypes.c_char_p(b".oldroot"),
            )
            != 0
        ):
            raise OSError(ctypes.get_errno(), "pivot_root failed")
        os.chdir(b"/")
        if libc.umount2(b"/.oldroot", _MNT_DETACH) != 0:
            raise OSError(ctypes.get_errno(), "umount2(/.oldroot) failed")
        os.rmdir(b"/.oldroot")
        if libc.mount(b"tmpfs", b"/", b"tmpfs", _MS_REMOUNT | _MS_RDONLY, None) != 0:
            raise OSError(ctypes.get_errno(), "remount ro / failed")
        os.chdir(workdir_b)
        # Drop all bounding capabilities and forbid privilege escalation so
        # execve() starts with zero capabilities (enforcing 0o444 CAS immutability).
        for cap in range(64):
            libc.prctl(_PR_CAPBSET_DROP, cap, 0, 0, 0)
        libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)
        if enable_rlimits:
            try:
                soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
                limit = 64 * 1024 * 1024
                if hard == resource.RLIM_INFINITY or hard >= limit:
                    new_soft = limit if soft == resource.RLIM_INFINITY else min(soft, limit)
                    resource.setrlimit(resource.RLIMIT_FSIZE, (new_soft, hard))
            except (ValueError, OSError):
                pass

    return _confine


def build_scrubbed_env(
    workspace: Path,
    *,
    home_dir: Optional[Path] = None,
    tmp_dir: Optional[Path] = None,
    xdg_config_dir: Optional[Path] = None,
    xdg_cache_dir: Optional[Path] = None,
    sandbox_id: str = "eval",
) -> Dict[str, str]:
    """Construct an environment-scrubbed POSIX env allowlist with zero host secrets."""
    ws = Path(os.path.realpath(str(workspace)))
    resolved_home = Path(os.path.realpath(str(home_dir))) if home_dir is not None else ws
    resolved_tmp = Path(os.path.realpath(str(tmp_dir))) if tmp_dir is not None else ws
    resolved_xdg_cfg = (
        Path(os.path.realpath(str(xdg_config_dir)))
        if xdg_config_dir is not None
        else resolved_home / ".config"
    )
    resolved_xdg_cache = (
        Path(os.path.realpath(str(xdg_cache_dir)))
        if xdg_cache_dir is not None
        else resolved_home / ".cache"
    )

    py_bin_dir = str(Path(sys.executable).resolve().parent)
    safe_path_parts = [py_bin_dir, "/usr/local/bin", "/usr/bin", "/bin"]
    seen: Set[str] = set()
    clean_path: List[str] = []
    for part in safe_path_parts:
        if part and part not in seen:
            seen.add(part)
            clean_path.append(part)

    return {
        "PATH": os.pathsep.join(clean_path),
        "HOME": str(resolved_home),
        "TMPDIR": str(resolved_tmp),
        "TEMP": str(resolved_tmp),
        "TMP": str(resolved_tmp),
        "XDG_CONFIG_HOME": str(resolved_xdg_cfg),
        "XDG_CACHE_HOME": str(resolved_xdg_cache),
        "XDG_DATA_HOME": str(resolved_home / ".local" / "share"),
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
        "HANZO_SANDBOX_ID": sandbox_id,
        "HANZO_WORKSPACE": str(ws),
    }


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
    spawned_pids: Set[int] = field(default_factory=set)
    escape_events: List[str] = field(default_factory=list)
    monitored_ports: Set[int] = field(default_factory=set)
    initial_root_entries: Set[str] = field(default_factory=set)
    nsroot_dir: Optional[Path] = None


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
        enable_fs_confinement: bool = True,
    ) -> None:
        if workspace_scale not in WORKSPACE_SCALES:
            raise ValueError(
                f"Unknown workspace_scale {workspace_scale!r}; choose from {sorted(WORKSPACE_SCALES)}"
            )
        self.workspace_scale = workspace_scale
        self.enable_rlimits = enable_rlimits
        self.enable_fs_confinement = enable_fs_confinement
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
        self._proc_dir = Path("/proc")
        self._state_lock = threading.Lock()
        self._active_pids: Dict[int, str] = {}
        self._known_run_dirs: Set[str] = set()
        self._linux_libc: Optional[Any] = None
        self._sys_pivot_root: int = 0
        self.fs_confinement: str = "unenforced_posix_audit_only"
        self.fs_confinement_enforced: bool = False
        if self.enable_fs_confinement:
            self._probe_linux_mount_ns_confinement()

    def _probe_linux_mount_ns_confinement(self) -> None:
        """Probe unprivileged Linux user+mount namespace confinement once at startup."""
        loaded = _load_linux_libc()
        if loaded is None:
            return
        libc, sys_pivot_root = loaded
        probe_root = Path(tempfile.mkdtemp(prefix=".ns_probe_", dir=str(self.runs_root)))
        try:
            p_ws = probe_root / "workspace"
            p_home = probe_root / "home"
            p_tmp = probe_root / "tmp"
            p_xcfg = probe_root / "xdg_config"
            p_xcache = probe_root / "xdg_cache"
            p_nsroot = probe_root / ".nsroot"
            for d in (p_ws, p_home, p_tmp, p_xcfg, p_xcache, p_nsroot):
                d.mkdir(parents=True, exist_ok=True)
            preexec = _build_linux_mount_ns_preexec(
                libc,
                sys_pivot_root,
                nsroot_b=os.fsencode(str(p_nsroot)),
                workdir_b=os.fsencode(str(p_ws)),
                rw_dirs_b=(
                    os.fsencode(str(p_ws)),
                    os.fsencode(str(p_home)),
                    os.fsencode(str(p_tmp)),
                    os.fsencode(str(p_xcfg)),
                    os.fsencode(str(p_xcache)),
                ),
                uid=os.getuid(),
                gid=os.getgid(),
                enable_rlimits=False,
            )
            probe_py = (
                "import pathlib, sys\n"
                "pathlib.Path('inside.txt').write_text('ok', encoding='utf-8')\n"
                "try:\n"
                "    pathlib.Path('../../probe_escape.txt').write_text('bad', encoding='utf-8')\n"
                "    sys.exit(2)\n"
                "except OSError:\n"
                "    sys.exit(0)\n"
            )
            res = subprocess.run(
                [sys.executable, "-c", probe_py],
                cwd=str(p_ws),
                env=build_scrubbed_env(
                    p_ws,
                    home_dir=p_home,
                    tmp_dir=p_tmp,
                    xdg_config_dir=p_xcfg,
                    xdg_cache_dir=p_xcache,
                    sandbox_id="confinement_probe",
                ),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=preexec,
                timeout=10.0,
            )
            if (
                res.returncode == 0
                and (p_ws / "inside.txt").is_file()
                and not (self.runs_root / "probe_escape.txt").exists()
            ):
                self._linux_libc = libc
                self._sys_pivot_root = sys_pivot_root
                self.fs_confinement = "linux_mount_namespace"
                self.fs_confinement_enforced = True
        except (OSError, subprocess.SubprocessError):
            pass
        finally:
            self._force_rmtree(probe_root, allow_chmod_files=True)
            escaped_probe = self.runs_root / "probe_escape.txt"
            if escaped_probe.exists():
                try:
                    escaped_probe.unlink()
                except OSError:
                    pass

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
        with self._state_lock:
            self._known_run_dirs.add(sandbox_root.name)
        workspace = sandbox_root / "workspace"
        home_dir = sandbox_root / "home"
        tmp_dir = sandbox_root / "tmp"
        xdg_config_dir = sandbox_root / "xdg_config"
        xdg_cache_dir = sandbox_root / "xdg_cache"
        external_canary_dir = sandbox_root / "host_guard"
        nsroot_dir = sandbox_root / ".nsroot"

        for d in (
            workspace,
            home_dir,
            tmp_dir,
            xdg_config_dir,
            xdg_cache_dir,
            external_canary_dir,
            nsroot_dir,
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
            nsroot_dir=nsroot_dir,
        )

    def build_sandbox_env(self, sandbox: SandboxInstance) -> Dict[str, str]:
        """Construct an environment-scrubbed POSIX env isolated to the sandbox."""
        return build_scrubbed_env(
            sandbox.workspace,
            home_dir=sandbox.home_dir,
            tmp_dir=sandbox.tmp_dir,
            xdg_config_dir=sandbox.xdg_config_dir,
            xdg_cache_dir=sandbox.xdg_cache_dir,
            sandbox_id=sandbox.sandbox_id,
        )

    def _make_preexec(
        self, sandbox: SandboxInstance, cwd_path: Path
    ) -> Optional[Callable[[], None]]:
        """Build preexec_fn applying Linux mount-namespace confinement and/or rlimits."""
        if self.fs_confinement_enforced and self._linux_libc is not None:
            nsroot = sandbox.nsroot_dir or (sandbox.sandbox_root / ".nsroot")
            nsroot.mkdir(parents=True, exist_ok=True)
            return _build_linux_mount_ns_preexec(
                self._linux_libc,
                self._sys_pivot_root,
                nsroot_b=os.fsencode(str(nsroot)),
                workdir_b=os.fsencode(str(cwd_path)),
                rw_dirs_b=(
                    os.fsencode(str(sandbox.workspace)),
                    os.fsencode(str(sandbox.home_dir)),
                    os.fsencode(str(sandbox.tmp_dir)),
                    os.fsencode(str(sandbox.xdg_config_dir)),
                    os.fsencode(str(sandbox.xdg_cache_dir)),
                ),
                uid=os.getuid(),
                gid=os.getgid(),
                enable_rlimits=self.enable_rlimits,
            )
        return self._make_preexec_rlimits()

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
            preexec_fn=self._make_preexec(sandbox, cwd_path),
        )
        pgid = proc.pid
        sandbox.spawned_pgids.add(pgid)
        sandbox.spawned_pids.add(proc.pid)
        with self._state_lock:
            self._active_pids[proc.pid] = sandbox.sandbox_id

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
        try:
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
        finally:
            with self._state_lock:
                self._active_pids.pop(proc.pid, None)

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

        # 3. Verify no unexpected files appeared in runs_root or base_dir
        #    (catches deeper `../../` and `../../../` traversal writes on unenforced hosts).
        if self.runs_root.exists():
            with self._state_lock:
                known_runs = set(self._known_run_dirs)
            for child in sorted(self.runs_root.iterdir(), key=lambda p: p.name):
                if child.name not in known_runs:
                    violations.append(f"escape_write_in_runs_root:{child.name}")

        if self.base_dir.exists():
            for child in sorted(self.base_dir.iterdir(), key=lambda p: p.name):
                if child.name not in {"cas_pool", "runs"}:
                    violations.append(f"escape_write_in_base_dir:{child.name}")

        # 4. Verify no unexpected files appeared in external_canary_dir.
        if sandbox.external_canary_dir.exists():
            for child in sorted(sandbox.external_canary_dir.iterdir()):
                if child.name != sandbox.external_canary_path.name:
                    violations.append(f"escape_write_in_host_guard:{child.name}")

        # 5. Walk workspace without following symlinks and reject any symlink whose
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
            fs_confinement=self.fs_confinement,
            fs_confinement_enforced=self.fs_confinement_enforced,
        )

    @staticmethod
    def _pgid_alive(pgid: int) -> bool:
        """Return True iff at least one process in process group ``pgid`` exists."""
        if pgid <= 0:
            return False
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def _find_pids_via_ps(self, target_pgids: Set[int], my_pid: int) -> Set[int]:
        """POSIX non-procfs fallback (e.g. macOS/BSD) discovering live PIDs in recorded PGIDs."""
        if not target_pgids:
            return set()
        ps_bin = shutil.which("ps")
        if not ps_bin:
            return set()
        found: Set[int] = set()
        try:
            res = subprocess.run(
                [ps_bin, "-axo", "pid=,pgid=,stat="],
                capture_output=True,
                text=True,
                timeout=3.0,
            )
            if res.returncode != 0:
                res = subprocess.run(
                    [ps_bin, "-eo", "pid=,pgid=,stat="],
                    capture_output=True,
                    text=True,
                    timeout=3.0,
                )
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    try:
                        pid = int(parts[0])
                        pgid = int(parts[1])
                    except ValueError:
                        continue
                    state = parts[2] if len(parts) >= 3 else ""
                    if pid != my_pid and pgid in target_pgids and not state.startswith("Z"):
                        found.add(pid)
        except (OSError, subprocess.SubprocessError):
            pass
        return found

    def _find_live_pgids(self, sandbox: SandboxInstance) -> List[int]:
        """Return recorded sandbox PGIDs that still have live processes."""
        live: List[int] = []
        for pgid in sorted(sandbox.spawned_pgids):
            if self._pgid_alive(pgid):
                live.append(pgid)
        return live

    def _find_orphan_pids(self, sandbox: SandboxInstance) -> List[int]:
        """Detect live orphan processes belonging to this sandbox (procfs + POSIX fallback)."""
        orphans: Set[int] = set()
        my_pid = os.getpid()
        marker = f"HANZO_SANDBOX_ID={sandbox.sandbox_id}".encode("utf-8")
        real_root = os.path.realpath(str(sandbox.sandbox_root))

        proc_dir = self._proc_dir
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
        else:
            # Non-procfs POSIX fallback (e.g. macOS/BSD): inspect recorded PGIDs via ps.
            live_pgids = set(self._find_live_pgids(sandbox))
            if live_pgids:
                orphans.update(self._find_pids_via_ps(live_pgids, my_pid))

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

    def _other_sandbox_active_pids(self, sandbox: SandboxInstance) -> Set[int]:
        with self._state_lock:
            return {
                pid
                for pid, owner in self._active_pids.items()
                if owner != sandbox.sandbox_id
            }

    def _reap_sandbox_owned_children(
        self, sandbox: SandboxInstance, orphan_pids: List[int]
    ) -> None:
        """Reap ONLY direct child PIDs/PGIDs owned by this sandbox (never waitpid(-1))."""
        other_active = self._other_sandbox_active_pids(sandbox)
        candidate_pids = (set(orphan_pids) | set(sandbox.spawned_pids)) - other_active
        for pid in sorted(candidate_pids):
            if pid <= 0:
                continue
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass

        for pgid in sorted(sandbox.spawned_pgids):
            if pgid <= 0 or pgid in other_active:
                continue
            while True:
                try:
                    done_pid, _ = os.waitpid(-pgid, os.WNOHANG)
                    if done_pid == 0:
                        break
                except ChildProcessError:
                    break

    def _reap_orphan_pids(
        self,
        sandbox: SandboxInstance,
        orphan_pids: List[int],
        *,
        live_pgids: Optional[List[int]] = None,
    ) -> Tuple[List[int], List[int]]:
        """Terminate and confirm death of all detected orphan PIDs and live sandbox PGIDs."""
        reaped: List[int] = []
        other_active = self._other_sandbox_active_pids(sandbox)
        target_pgids = sorted(
            (set(sandbox.spawned_pgids) | set(live_pgids or ())) - other_active
        )
        safe_orphan_pids = [p for p in orphan_pids if p not in other_active]

        # First signal all sandbox-owned process groups and discovered orphan PIDs.
        for pgid in target_pgids:
            try:
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        for pid in safe_orphan_pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass

        deadline = time.monotonic() + 0.3
        remaining_pids = set(safe_orphan_pids)
        remaining_pgids = {pgid for pgid in target_pgids if self._pgid_alive(pgid)}
        while (remaining_pids or remaining_pgids) and time.monotonic() < deadline:
            self._reap_sandbox_owned_children(sandbox, list(remaining_pids))
            for pid in list(remaining_pids):
                if not self._pid_alive(pid):
                    remaining_pids.discard(pid)
                    reaped.append(pid)
            for pgid in list(remaining_pgids):
                if not self._pgid_alive(pgid):
                    remaining_pgids.discard(pgid)
            if remaining_pids or remaining_pgids:
                time.sleep(0.01)

        # Escalate to SIGKILL for stubborn orphans or surviving process groups.
        if remaining_pids or remaining_pgids:
            for pgid in sorted(remaining_pgids | set(target_pgids)):
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            for pid in list(remaining_pids):
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            kill_deadline = time.monotonic() + 0.2
            while (remaining_pids or remaining_pgids) and time.monotonic() < kill_deadline:
                self._reap_sandbox_owned_children(sandbox, list(remaining_pids))
                for pid in list(remaining_pids):
                    if not self._pid_alive(pid):
                        remaining_pids.discard(pid)
                        reaped.append(pid)
                for pgid in list(remaining_pgids):
                    if not self._pgid_alive(pgid):
                        remaining_pgids.discard(pgid)
                if remaining_pids or remaining_pgids:
                    time.sleep(0.01)

        # Final sandbox-scoped zombie drain (strictly scoped — never waitpid(-1)).
        self._reap_sandbox_owned_children(sandbox, safe_orphan_pids)
        still_alive_pgids = sorted(pgid for pgid in remaining_pgids if self._pgid_alive(pgid))
        return sorted(set(reaped)), still_alive_pgids

    def _pid_alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        # Check if zombie when procfs is available.
        if self._proc_dir.is_dir():
            try:
                stat_txt = (self._proc_dir / str(pid) / "stat").read_text(
                    encoding="utf-8", errors="ignore"
                )
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
                port = data.get("port") if isinstance(data, dict) else None
                if isinstance(port, int) and 1 <= port <= 65535:
                    sandbox.monitored_ports.add(port)
            except (OSError, ValueError, TypeError, AttributeError):
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
        live_pgids = self._find_live_pgids(sandbox)
        reaped_pids: List[int] = []
        if orphan_pids or live_pgids:
            if orphan_pids:
                violations.append(f"orphan_pids_detected:{orphan_pids}")
            else:
                violations.append(f"orphan_pgids_detected:{live_pgids}")
            reaped_pids, unreaped_pgids = self._reap_orphan_pids(
                sandbox, orphan_pids, live_pgids=live_pgids
            )
            still_alive = [p for p in orphan_pids if p not in reaped_pids]
            if still_alive:
                violations.append(f"orphan_pids_unreaped:{still_alive}")
            if unreaped_pgids:
                violations.append(f"orphan_pgids_unreaped:{unreaped_pgids}")

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
        if workspace_removed:
            with self._state_lock:
                self._known_run_dirs.discard(sandbox.sandbox_root.name)
        else:
            violations.append("workspace_not_removed")

        # On unenforced hosts, also remove any stray escape files left in runs_root
        # (already flagged by audit_security before teardown).
        if not self.fs_confinement_enforced and self.runs_root.exists():
            with self._state_lock:
                known_runs = set(self._known_run_dirs)
            for child in list(self.runs_root.iterdir()):
                if child.name not in known_runs:
                    try:
                        if child.is_dir() and not child.is_symlink():
                            self._force_rmtree(child, allow_chmod_files=True)
                        else:
                            child.unlink()
                    except OSError:
                        pass

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
