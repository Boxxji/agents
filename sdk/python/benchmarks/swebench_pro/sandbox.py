"""Isolated sandbox workspace provisioning: plain NVMe vs Visor CoW snapshot.

Models two real local storage profiles on Linux without requiring root mounts:
- ``plain_nvme``: Fresh full-tree materialization + git repo build + anti-leakage
  scrub per task execution (every surrounding repo file physically written).
- ``visor_isolated``: Content-addressed base image store (emulating
  ``vfs -> Hanzo S3 -> Visor``) built and scrubbed once per ``(instance_id, scale)``;
  each task sandbox mounts immutable repo ballast and protected test files via
  read-only ``0o444`` hardlinks (zero physical copy) while isolating mutable
  package files and ``.git`` metadata in a per-task writable upper layer.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from .integrity import (
    _run_git,
    restore_writable_tree,
    scrub_git_workspace,
    sha256_file,
    snapshot_protected_hashes,
    verify_no_leakage,
)
from .schemas import IntegrityReport, StorageProfile, SWEBenchTask


@dataclass(frozen=True)
class RepoScaleConfig:
    """Surrounding repository ballast sizing to benchmark NVMe vs Visor CoW."""

    name: str
    asset_files: int
    asset_bytes_per_file: int

    @property
    def total_asset_bytes(self) -> int:
        return self.asset_files * self.asset_bytes_per_file


REPO_SCALES: Dict[str, RepoScaleConfig] = {
    "tiny": RepoScaleConfig(name="tiny", asset_files=8, asset_bytes_per_file=2048),
    "small": RepoScaleConfig(name="small", asset_files=40, asset_bytes_per_file=4096),
    "medium": RepoScaleConfig(name="medium", asset_files=240, asset_bytes_per_file=16384),
    "large": RepoScaleConfig(name="large", asset_files=480, asset_bytes_per_file=32768),
}


@dataclass
class ProvisionedSandbox:
    """Handle to a provisioned, isolated task workspace."""

    workspace: Path
    root_tempdir: Path
    profile: StorageProfile
    base_commit: str
    gold_commit_sha: str
    setup_ms: float
    bytes_copied: int
    files_materialized: int
    hardlinked_files: int
    cas_cache_hit: bool
    base_manifest: Dict[str, str]
    integrity_report: IntegrityReport

    def cleanup(self) -> None:
        restore_writable_tree(self.root_tempdir)
        shutil.rmtree(self.root_tempdir, ignore_errors=True)


@dataclass
class _CASImage:
    image_dir: Path
    base_commit: str
    gold_commit_sha: str
    base_manifest: Dict[str, str]
    scrub_report: IntegrityReport


def _generate_deterministic_asset(task_id: str, idx: int, size_bytes: int) -> bytes:
    """Deterministic compressible+incompressible source-like ballast blob."""
    seed = hashlib.sha256(f"{task_id}:asset:{idx}".encode("utf-8")).hexdigest()
    header = f"# Auto-generated repository fixture {idx} ({seed[:16]})\n".encode("utf-8")
    line = f"RECORD_{idx:04d} = '{seed}'\n".encode("utf-8")
    repeats = max(1, (size_bytes - len(header)) // len(line) + 1)
    payload = header + (line * repeats)
    return payload[:size_bytes]


def _write_file_bytes(root: Path, rel: str, data: bytes) -> int:
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return len(data)


def _init_unscrubbed_upstream_repo(
    repo_dir: Path,
    task: SWEBenchTask,
    scale: RepoScaleConfig,
) -> Tuple[str, str, int, int]:
    """Initialize a realistic upstream git repo containing base + leaked gold fix.

    Returns ``(base_commit_sha, gold_commit_sha, bytes_written, files_written)``.
    """
    bytes_written = 0
    files_written = 0

    # Write base source files
    for rel, content in sorted(task.base_files.items()):
        bytes_written += _write_file_bytes(repo_dir, rel, content.encode("utf-8"))
        files_written += 1

    # Write evaluation test files
    for rel, content in sorted(task.test_files.items()):
        bytes_written += _write_file_bytes(repo_dir, rel, content.encode("utf-8"))
        files_written += 1

    # Write surrounding repository ballast files
    for idx in range(scale.asset_files):
        rel = f"repo_assets/module_{idx // 25:02d}/asset_{idx:04d}.py"
        data = _generate_deterministic_asset(task.instance_id, idx, scale.asset_bytes_per_file)
        bytes_written += _write_file_bytes(repo_dir, rel, data)
        files_written += 1

    _run_git(repo_dir, "init", "-q")
    _run_git(repo_dir, "checkout", "-q", "-b", "main")
    _run_git(repo_dir, "add", "-A")
    _run_git(repo_dir, "commit", "-q", "-m", f"Base commit for {task.instance_id}")
    base_commit = _run_git(repo_dir, "rev-parse", "HEAD").stdout.strip()

    # Simulate an upstream repo that ALSO has the future gold fix commit, tag,
    # branch, remote, reflog, and leaked artifact before scrubbing!
    _run_git(repo_dir, "checkout", "-q", "-b", "upstream-gold-fix")
    for rel, content in sorted(task.gold_patch_files.items()):
        gold_annotated = f"# {task.gold_marker}\n" + content
        _write_file_bytes(repo_dir, rel, gold_annotated.encode("utf-8"))
    _run_git(repo_dir, "add", "-A")
    _run_git(
        repo_dir,
        "commit",
        "-q",
        "-m",
        f"Reference gold fix {task.gold_marker} for {task.instance_id}",
    )
    gold_commit_sha = _run_git(repo_dir, "rev-parse", "HEAD").stdout.strip()
    _run_git(repo_dir, "tag", "v-gold-fix", gold_commit_sha)
    _run_git(repo_dir, "remote", "add", "origin", "https://upstream.invalid/leak-test.git")

    # Drop a simulated leaked artifact file to verify scrubber removal
    _write_file_bytes(
        repo_dir,
        "gold_patch.diff",
        f"# leaked {task.gold_marker}\n".encode("utf-8"),
    )

    return base_commit, gold_commit_sha, bytes_written, files_written


def _build_file_manifest(workspace: Path) -> Dict[str, str]:
    manifest: Dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(workspace, followlinks=False):
        if ".git" in dirnames:
            dirnames.remove(".git")
        for fname in filenames:
            fpath = Path(dirpath) / fname
            if fpath.is_file() and not fpath.is_symlink():
                rel = str(fpath.relative_to(workspace))
                manifest[rel] = sha256_file(fpath)
    return manifest


def _task_cas_fingerprint(task: SWEBenchTask, scale: RepoScaleConfig) -> str:
    """Compute a content-addressed fingerprint covering task files, tests, patches, and ballast scale."""
    h = hashlib.sha256()
    for field_name, val in (
        ("instance_id", task.instance_id),
        ("revision", task.revision),
        ("gold_marker", task.gold_marker),
        ("scale_name", scale.name),
        ("asset_files", str(scale.asset_files)),
        ("asset_bytes_per_file", str(scale.asset_bytes_per_file)),
    ):
        h.update(f"{field_name}:{val}\n".encode("utf-8"))
    for section, mapping in (
        ("base_files", task.base_files),
        ("test_files", task.test_files),
        ("gold_patch_files", task.gold_patch_files),
    ):
        h.update(f"[{section}]\n".encode("utf-8"))
        for rel in sorted(mapping.keys()):
            content_bytes = mapping[rel].encode("utf-8")
            h.update(f"{rel}:{len(content_bytes)}:".encode("utf-8"))
            h.update(hashlib.sha256(content_bytes).digest())
            h.update(b"\n")
    h.update(b"[protected_test_paths]\n")
    for rel in sorted(task.protected_test_paths):
        h.update(f"{rel}\n".encode("utf-8"))
    return h.hexdigest()


class SandboxManager:
    """Manages isolated task sandboxes for both ``plain_nvme`` and ``visor_isolated``."""

    def __init__(self, cas_root: Optional[Path] = None) -> None:
        self._owns_cas_root = cas_root is None
        raw_cas = cas_root or Path(tempfile.mkdtemp(prefix="visor_cas_store_"))
        self.cas_root = raw_cas.resolve()
        self._cas_lock = threading.Lock()
        self._cas_images: Dict[Tuple[str, str, str], _CASImage] = {}

    def close(self) -> None:
        if self._owns_cas_root and self.cas_root.exists():
            restore_writable_tree(self.cas_root)
            shutil.rmtree(self.cas_root, ignore_errors=True)

    def _get_or_build_cas_image(
        self,
        task: SWEBenchTask,
        scale: RepoScaleConfig,
    ) -> Tuple[_CASImage, bool, int]:
        """Build/fetch scrubbed read-only CAS image for ``visor_isolated``."""
        fingerprint = _task_cas_fingerprint(task, scale)
        key = (task.instance_id, scale.name, fingerprint)
        with self._cas_lock:
            existing = self._cas_images.get(key)
            if existing is not None and existing.image_dir.exists():
                return existing, True, 0

            image_dir = self.cas_root / f"{task.instance_id}__{scale.name}__{fingerprint[:12]}"
            if image_dir.exists():
                restore_writable_tree(image_dir)
                shutil.rmtree(image_dir, ignore_errors=True)
            image_dir.mkdir(parents=True, exist_ok=True)

            base_commit, gold_commit_sha, bytes_written, _ = _init_unscrubbed_upstream_repo(
                image_dir, task, scale
            )
            scrub_report = scrub_git_workspace(image_dir, base_commit)
            verify_no_leakage(
                image_dir,
                base_commit=base_commit,
                gold_commit_sha=gold_commit_sha,
                gold_marker=task.gold_marker,
                report=scrub_report,
            )
            base_manifest = _build_file_manifest(image_dir)

            # Harden immutable asset & protected test files inside CAS as read-only
            for rel in list(task.protected_test_paths):
                p = image_dir / rel
                if p.is_file():
                    os.chmod(p, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            assets_dir = image_dir / "repo_assets"
            if assets_dir.exists():
                for p in assets_dir.rglob("*"):
                    if p.is_file():
                        os.chmod(p, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

            image = _CASImage(
                image_dir=image_dir,
                base_commit=base_commit,
                gold_commit_sha=gold_commit_sha,
                base_manifest=base_manifest,
                scrub_report=scrub_report,
            )
            self._cas_images[key] = image
            return image, False, bytes_written

    def provision(
        self,
        task: SWEBenchTask,
        profile: StorageProfile,
        *,
        scale: RepoScaleConfig,
        skip_scrub: bool = False,
        env: Optional[Dict[str, str]] = None,
        token_scopes: Optional[Tuple[str, ...]] = ("ai:inference",),
    ) -> ProvisionedSandbox:
        """Provision a fresh isolated workspace under ``plain_nvme`` or ``visor_isolated``."""
        t0 = time.perf_counter_ns()
        root_tempdir = Path(
            tempfile.mkdtemp(prefix=f"swebench_{profile.value}_{task.instance_id[:24]}_")
        ).resolve()
        workspace = root_tempdir / "repo"
        workspace.mkdir(parents=True, exist_ok=True)

        bytes_copied = 0
        files_materialized = 0
        hardlinked_files = 0
        cas_cache_hit = False

        if profile == StorageProfile.PLAIN_NVME or skip_scrub:
            base_commit, gold_commit_sha, bytes_copied, files_materialized = (
                _init_unscrubbed_upstream_repo(workspace, task, scale)
            )
            report = IntegrityReport()
            if not skip_scrub:
                scrub_git_workspace(workspace, base_commit, report=report)
            verify_no_leakage(
                workspace,
                base_commit=base_commit,
                gold_commit_sha=gold_commit_sha,
                gold_marker=task.gold_marker,
                env=env,
                token_scopes=token_scopes,
                report=report,
            )
            report.protected_test_hashes = snapshot_protected_hashes(
                workspace, task.protected_test_paths, make_readonly=False
            )
            base_manifest = _build_file_manifest(workspace)
        else:
            # visor_isolated: mount immutable CAS blobs via CoW hardlinks + isolated upper layer
            image, cas_cache_hit, cold_bytes = self._get_or_build_cas_image(task, scale)
            base_commit = image.base_commit
            gold_commit_sha = image.gold_commit_sha
            bytes_copied += cold_bytes

            # 1. Copy isolated .git metadata (small) so each task can run git diff/apply safely
            src_git = image.image_dir / ".git"
            dst_git = workspace / ".git"
            shutil.copytree(src_git, dst_git)
            for p in dst_git.rglob("*"):
                if p.is_file():
                    bytes_copied += p.stat().st_size

            # 2. Hardlink immutable repo_assets ballast (zero copy) while creating
            #    private detached copies for mutable source files AND protected test files
            #    so a rollout running `chmod +w` + edit can never mutate a shared CAS inode.
            protected_set = set(task.protected_test_paths)
            for dirpath, dirnames, filenames in os.walk(image.image_dir, followlinks=False):
                if ".git" in dirnames:
                    dirnames.remove(".git")
                rel_dir = Path(dirpath).relative_to(image.image_dir)
                dst_dir = workspace / rel_dir
                dst_dir.mkdir(parents=True, exist_ok=True)
                for fname in filenames:
                    src_file = Path(dirpath) / fname
                    rel_file = src_file.relative_to(image.image_dir).as_posix()
                    dst_file = workspace / rel_file
                    files_materialized += 1
                    if rel_file.startswith("repo_assets/") and rel_file not in protected_set:
                        os.link(src_file, dst_file)
                        hardlinked_files += 1
                    else:
                        # Private detached copy for mutable package files and protected tests
                        data = src_file.read_bytes()
                        dst_file.write_bytes(data)
                        bytes_copied += len(data)

            report = IntegrityReport(
                stripped_remotes=list(image.scrub_report.stripped_remotes),
                stripped_tags=list(image.scrub_report.stripped_tags),
                stripped_branches=list(image.scrub_report.stripped_branches),
                stripped_reflogs=image.scrub_report.stripped_reflogs,
                pruned_objects=image.scrub_report.pruned_objects,
                removed_artifacts=list(image.scrub_report.removed_artifacts),
            )
            verify_no_leakage(
                workspace,
                base_commit=base_commit,
                gold_commit_sha=gold_commit_sha,
                gold_marker=task.gold_marker,
                env=env,
                token_scopes=token_scopes,
                report=report,
            )
            report.protected_test_hashes = snapshot_protected_hashes(
                workspace, task.protected_test_paths, make_readonly=True
            )
            base_manifest = dict(image.base_manifest)

        setup_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        return ProvisionedSandbox(
            workspace=workspace,
            root_tempdir=root_tempdir,
            profile=profile,
            base_commit=base_commit,
            gold_commit_sha=gold_commit_sha,
            setup_ms=setup_ms,
            bytes_copied=bytes_copied,
            files_materialized=files_materialized,
            hardlinked_files=hardlinked_files,
            cas_cache_hit=cas_cache_hit,
            base_manifest=base_manifest,
            integrity_report=report,
        )

    @staticmethod
    def measure_snapshot_diff(sandbox: ProvisionedSandbox) -> Tuple[float, Dict[str, str]]:
        """Measure snapshot delta extraction time (overlay/working-tree vs base manifest)."""
        t0 = time.perf_counter_ns()
        changed: Dict[str, str] = {}
        workspace = sandbox.workspace

        if sandbox.profile == StorageProfile.VISOR_ISOLATED:
            # Visor CoW fast-path: skip files whose inode link count > 1 (still hardlinked to CAS!)
            for dirpath, dirnames, filenames in os.walk(workspace, followlinks=False):
                if ".git" in dirnames:
                    dirnames.remove(".git")
                for fname in filenames:
                    fpath = Path(dirpath) / fname
                    if fpath.is_symlink() or not fpath.is_file():
                        continue
                    rel = str(fpath.relative_to(workspace))
                    st = fpath.stat()
                    if st.st_nlink > 1 and rel in sandbox.base_manifest:
                        # Untouched CoW inode from CAS — no re-hash required
                        continue
                    digest = sha256_file(fpath)
                    if sandbox.base_manifest.get(rel) != digest:
                        changed[rel] = digest
        else:
            # Plain NVMe: must scan & hash working tree files to compute full snapshot delta
            for dirpath, dirnames, filenames in os.walk(workspace, followlinks=False):
                if ".git" in dirnames:
                    dirnames.remove(".git")
                for fname in filenames:
                    fpath = Path(dirpath) / fname
                    if fpath.is_symlink() or not fpath.is_file():
                        continue
                    rel = str(fpath.relative_to(workspace))
                    digest = sha256_file(fpath)
                    if sandbox.base_manifest.get(rel) != digest:
                        changed[rel] = digest

        diff_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        return diff_ms, changed
