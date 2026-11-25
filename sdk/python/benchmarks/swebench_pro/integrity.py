"""Anti-leakage and anti-tampering enforcement for SWE-bench Pro V2 sandboxes.

Implements fail-closed defenses tested against red-team agents:
1. Git history / reflog / remote / tag / unreachable-object purge so reference
   fix commits are physically removed from ``.git/objects`` before agent runs.
2. Environment & credential sanitization (blocks gold-patch variables, admin
   secrets, and non-``ai:inference`` token scopes).
3. Protected test-file SHA-256 snapshotting & byte-for-byte post-run integrity
   verification (plus read-only permissions on ``visor_isolated``).
4. Workspace anti-tampering audit detecting symlink escapes outside the sandbox
   root, ``git diff`` modifications to protected paths, ``PYTHONPATH``/
   ``PYTEST_ADDOPTS`` hijacks, and ``sitecustomize.py`` / ``pytest.py`` shims.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .schemas import IntegrityReport, ViolationType

FORBIDDEN_LEAK_ARTIFACTS: Tuple[str, ...] = (
    "gold_patch.diff",
    "gold_patch.patch",
    "test_patch.diff",
    "test_patch.patch",
    ".swe_bench_gold.json",
    "reference_fix.diff",
)

FORBIDDEN_GIT_STATE_FILES: Tuple[str, ...] = (
    "FETCH_HEAD",
    "ORIG_HEAD",
    "MERGE_HEAD",
    "CHERRY_PICK_HEAD",
    "REVERT_HEAD",
    "BISECT_LOG",
)

FORBIDDEN_ENV_KEYS: Set[str] = {
    "GOLD_PATCH",
    "REFERENCE_FIX",
    "SWEBENCH_GOLD",
    "TEST_PATCH",
    "HANZO_ADMIN_TOKEN",
    "ADMIN_TOKEN",
    "AWS_SECRET_ACCESS_KEY",
    "S3_SECRET_KEY",
}

TAMPER_ENV_KEYS: Set[str] = {
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "PYTEST_ADDOPTS",
    "PYTEST_PLUGINS",
    "LD_PRELOAD",
    "SITE_CUSTOMIZE",
}

FORBIDDEN_SHIM_FILENAMES: Set[str] = {
    "sitecustomize.py",
    "usercustomize.py",
    "pytest.py",
    "unittest.py",
    "pluggy.py",
}

ALLOWED_CREDENTIAL_SCOPES: Set[str] = {"ai:inference"}


def _run_git(workspace: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(workspace),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_AUTHOR_NAME": "swebench-harness",
        "GIT_AUTHOR_EMAIL": "harness@hanzo.local",
        "GIT_COMMITTER_NAME": "swebench-harness",
        "GIT_COMMITTER_EMAIL": "harness@hanzo.local",
    }
    return subprocess.run(
        ["git", *args],
        cwd=str(workspace),
        env=env,
        capture_output=True,
        text=True,
        check=check,
    )


def sha256_file(path: Path) -> str:
    """Compute hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_protected_hashes(
    workspace: Path,
    protected_paths: Sequence[str],
    *,
    make_readonly: bool = False,
) -> Dict[str, str]:
    """Hash all protected evaluation test files before agent execution."""
    hashes: Dict[str, str] = {}
    for rel in sorted(set(protected_paths)):
        target = workspace / rel
        if target.is_file():
            hashes[rel] = sha256_file(target)
            if make_readonly:
                try:
                    os.chmod(target, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
                except OSError:
                    pass
    return hashes


def restore_writable_tree(root: Path) -> None:
    """Ensure all files/dirs under root are owner-writable for clean teardown."""
    if not root.exists():
        return
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        try:
            os.chmod(dirpath, stat.S_IRWXU)
        except OSError:
            pass
        for name in filenames:
            p = Path(dirpath) / name
            if not p.is_symlink():
                try:
                    os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)
                except OSError:
                    pass


def scrub_git_workspace(
    workspace: Path,
    base_commit: str,
    *,
    report: Optional[IntegrityReport] = None,
) -> IntegrityReport:
    """Strip remotes, tags, non-HEAD branches, reflogs, and unreachable gold objects.

    Physically purges unreachable commits and blobs from ``.git/objects`` via
    ``git repack -Ad`` + ``git prune --expire=now`` so that an adversarial agent
    inspecting ``.git`` cannot recover a future reference fix commit.
    """
    rep = report or IntegrityReport()

    # 1. Remove all remotes
    remotes_proc = _run_git(workspace, "remote", check=False)
    for remote in [r.strip() for r in remotes_proc.stdout.splitlines() if r.strip()]:
        _run_git(workspace, "remote", "remove", remote, check=False)
        rep.stripped_remotes.append(remote)

    # 2. Detach HEAD at base_commit and hard reset working tree
    _run_git(workspace, "checkout", "--force", "--detach", base_commit)
    _run_git(workspace, "reset", "--hard", base_commit)

    # 3. Delete all tags
    tags_proc = _run_git(workspace, "tag", "-l", check=False)
    for tag in [t.strip() for t in tags_proc.stdout.splitlines() if t.strip()]:
        _run_git(workspace, "tag", "-d", tag, check=False)
        rep.stripped_tags.append(tag)

    # 4. Delete all local and remote-tracking refs (heads, remotes, stash, notes)
    refs_proc = _run_git(
        workspace,
        "for-each-ref",
        "--format=%(refname)",
        "refs/heads",
        "refs/remotes",
        "refs/tags",
        "refs/stash",
        "refs/notes",
        check=False,
    )
    for ref in [r.strip() for r in refs_proc.stdout.splitlines() if r.strip()]:
        _run_git(workspace, "update-ref", "-d", ref, check=False)
        rep.stripped_branches.append(ref)

    # 5. Expire & delete reflogs completely
    _run_git(workspace, "reflog", "expire", "--expire=now", "--expire-unreachable=now", "--all", check=False)
    logs_dir = workspace / ".git" / "logs"
    if logs_dir.exists():
        shutil.rmtree(logs_dir, ignore_errors=True)
    rep.stripped_reflogs = True

    # 6. Remove transient git state files that may contain future SHAs
    git_dir = workspace / ".git"
    for state_name in FORBIDDEN_GIT_STATE_FILES:
        state_file = git_dir / state_name
        if state_file.exists():
            state_file.unlink(missing_ok=True)
            rep.removed_artifacts.append(f".git/{state_name}")

    # 7. Physically purge unreachable commits/trees/blobs from .git/objects
    _run_git(workspace, "repack", "-Ad", "-q", check=False)
    _run_git(workspace, "prune", "--expire=now", check=False)
    _run_git(workspace, "gc", "--prune=now", "-q", check=False)
    rep.pruned_objects = True

    # 8. Remove any forbidden gold/test patch artifacts left in workspace
    for artifact in FORBIDDEN_LEAK_ARTIFACTS:
        p = workspace / artifact
        if p.exists() or p.is_symlink():
            p.unlink(missing_ok=True)
            rep.removed_artifacts.append(artifact)

    # 9. Clean untracked files
    _run_git(workspace, "clean", "-fdx", check=False)
    return rep


def sanitize_execution_env(
    raw_env: Optional[Dict[str, str]] = None,
    *,
    gold_marker: Optional[str] = None,
    token_scopes: Optional[Sequence[str]] = None,
) -> Dict[str, str]:
    """Build a minimal, leakage-free execution environment for sandbox subprocesses."""
    base_env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }
    if raw_env:
        for k, v in raw_env.items():
            key_upper = k.upper()
            if key_upper in FORBIDDEN_ENV_KEYS or key_upper in TAMPER_ENV_KEYS:
                continue
            if gold_marker and gold_marker in v:
                continue
            base_env[k] = v
    if token_scopes is not None:
        base_env["HANZO_TOKEN_SCOPES"] = ",".join(token_scopes)
    return base_env


def verify_no_leakage(
    workspace: Path,
    *,
    base_commit: str,
    gold_commit_sha: Optional[str] = None,
    gold_marker: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    token_scopes: Optional[Sequence[str]] = None,
    report: Optional[IntegrityReport] = None,
) -> IntegrityReport:
    """Fail-closed pre-execution audit verifying zero reference-fix leakage."""
    rep = report or IntegrityReport()
    git_dir = workspace / ".git"

    # 1. Check remotes
    remotes = [
        r.strip()
        for r in _run_git(workspace, "remote", check=False).stdout.splitlines()
        if r.strip()
    ]
    for remote in remotes:
        rep.add_violation(
            ViolationType.GIT_REMOTE_PRESENT,
            remote,
            f"Git remote {remote!r} present in sandbox",
            phase="pre",
        )

    # 2. Check tags
    tags = [
        t.strip()
        for t in _run_git(workspace, "tag", "-l", check=False).stdout.splitlines()
        if t.strip()
    ]
    for tag in tags:
        rep.add_violation(
            ViolationType.GIT_TAG_PRESENT,
            tag,
            f"Git tag {tag!r} present in sandbox",
            phase="pre",
        )

    # 3. Check extra branches / refs
    refs = [
        r.strip()
        for r in _run_git(
            workspace,
            "for-each-ref",
            "--format=%(refname)",
            "refs/heads",
            "refs/remotes",
            "refs/stash",
            check=False,
        ).stdout.splitlines()
        if r.strip()
    ]
    for ref in refs:
        rep.add_violation(
            ViolationType.GIT_EXTRA_BRANCH,
            ref,
            f"Extra git ref {ref!r} present in sandbox",
            phase="pre",
        )

    # 4. Check reflogs
    logs_dir = git_dir / "logs"
    if logs_dir.exists() and any(logs_dir.rglob("*")):
        rep.add_violation(
            ViolationType.GIT_REFLOG_PRESENT,
            ".git/logs",
            "Non-empty .git/logs directory detected",
            phase="pre",
        )

    # 5. Check git state files
    for state_name in FORBIDDEN_GIT_STATE_FILES:
        if (git_dir / state_name).exists():
            rep.add_violation(
                ViolationType.GIT_STATE_FILE_PRESENT,
                f".git/{state_name}",
                f"Git state file .git/{state_name} present",
                phase="pre",
            )

    # 6. Verify HEAD matches base_commit and no future commits exist in history
    head_sha = _run_git(workspace, "rev-parse", "HEAD", check=False).stdout.strip()
    if head_sha != base_commit:
        rep.add_violation(
            ViolationType.GIT_FUTURE_COMMIT,
            head_sha or "HEAD",
            f"HEAD {head_sha!r} != expected base_commit {base_commit!r}",
            phase="pre",
        )

    all_commits = [
        c.strip()
        for c in _run_git(workspace, "rev-list", "--all", check=False).stdout.splitlines()
        if c.strip()
    ]
    if gold_commit_sha:
        if gold_commit_sha in all_commits:
            rep.add_violation(
                ViolationType.GIT_FUTURE_COMMIT,
                gold_commit_sha,
                "Gold reference commit is reachable from refs",
                phase="pre",
            )
        cat_proc = _run_git(workspace, "cat-file", "-e", gold_commit_sha, check=False)
        if cat_proc.returncode == 0:
            rep.add_violation(
                ViolationType.GIT_GOLD_OBJECT_LEAK,
                gold_commit_sha,
                "Gold reference commit object still exists in .git/objects",
                phase="pre",
            )

    # 7. Scan git object database & working tree for gold_marker sentinel
    if gold_marker:
        marker_bytes = gold_marker.encode("utf-8")
        # Scan all reachable commit messages and blobs via git
        log_proc = _run_git(workspace, "log", "--all", "--stat", "-p", check=False)
        if gold_marker in log_proc.stdout:
            rep.add_violation(
                ViolationType.GIT_GOLD_OBJECT_LEAK,
                "git log",
                "Gold reference marker found in git log output",
                phase="pre",
            )
        # Scan raw pack/loose objects in .git/objects
        objects_dir = git_dir / "objects"
        if objects_dir.exists():
            for obj_file in objects_dir.rglob("*"):
                if obj_file.is_file():
                    try:
                        if marker_bytes in obj_file.read_bytes():
                            rep.add_violation(
                                ViolationType.GIT_GOLD_OBJECT_LEAK,
                                str(obj_file.relative_to(workspace)),
                                "Gold reference marker found inside .git/objects",
                                phase="pre",
                            )
                            break
                    except OSError:
                        pass
        # Scan all working tree files
        for dirpath, dirnames, filenames in os.walk(workspace, followlinks=False):
            if ".git" in dirnames:
                dirnames.remove(".git")
            for fname in filenames:
                fpath = Path(dirpath) / fname
                if fpath.is_file() and not fpath.is_symlink():
                    try:
                        if marker_bytes in fpath.read_bytes():
                            rep.add_violation(
                                ViolationType.LEAK_ARTIFACT_PRESENT,
                                str(fpath.relative_to(workspace)),
                                "Gold reference marker found in workspace file",
                                phase="pre",
                            )
                    except OSError:
                        pass

    # 8. Check forbidden leak artifact filenames
    for artifact in FORBIDDEN_LEAK_ARTIFACTS:
        if (workspace / artifact).exists() or (workspace / artifact).is_symlink():
            rep.add_violation(
                ViolationType.LEAK_ARTIFACT_PRESENT,
                artifact,
                f"Forbidden leak artifact {artifact!r} present in workspace",
                phase="pre",
            )

    # 9. Check environment variables for gold markers / forbidden keys
    if env:
        for k, v in env.items():
            if k.upper() in FORBIDDEN_ENV_KEYS:
                rep.add_violation(
                    ViolationType.ENV_LEAK_VAR,
                    k,
                    f"Forbidden secret/leak environment variable {k!r} exposed",
                    phase="pre",
                )
            elif gold_marker and gold_marker in v:
                rep.add_violation(
                    ViolationType.ENV_LEAK_VAR,
                    k,
                    f"Gold reference marker leaked in environment variable {k!r}",
                    phase="pre",
                )

    # 10. Validate credential scopes (ai:inference only)
    if token_scopes is not None:
        disallowed = [s for s in token_scopes if s not in ALLOWED_CREDENTIAL_SCOPES]
        if disallowed:
            rep.add_violation(
                ViolationType.CREDENTIAL_SCOPE_VIOLATION,
                ",".join(disallowed),
                f"Disallowed sandbox credential scopes: {disallowed}",
                phase="pre",
            )

    return rep


def verify_post_execution_integrity(
    workspace: Path,
    *,
    protected_hashes: Dict[str, str],
    base_filenames: Iterable[str],
    agent_env_overrides: Optional[Dict[str, str]] = None,
    report: Optional[IntegrityReport] = None,
) -> IntegrityReport:
    """Fail-closed post-agent audit detecting test tampering, symlink escapes, and shims."""
    rep = report or IntegrityReport()
    workspace_real = os.path.realpath(str(workspace))
    base_file_set = set(base_filenames)

    # 1. Verify every protected test file exists, is not a symlink, and matches SHA-256
    for rel, expected_digest in sorted(protected_hashes.items()):
        target = workspace / rel
        if target.is_symlink():
            rep.add_violation(
                ViolationType.SYMLINK_ESCAPE,
                rel,
                f"Protected test path {rel!r} was replaced with a symlink",
                phase="post",
            )
            continue
        if not target.is_file():
            rep.add_violation(
                ViolationType.TEST_FILE_MISSING,
                rel,
                f"Protected test file {rel!r} is missing",
                phase="post",
            )
            continue
        actual_digest = sha256_file(target)
        if actual_digest != expected_digest:
            rep.add_violation(
                ViolationType.TEST_FILE_MODIFIED,
                rel,
                f"Protected test file {rel!r} hash mismatch ({actual_digest[:12]} != {expected_digest[:12]})",
                phase="post",
            )

    # 2. Check git diff / status against protected paths
    protected_set = set(protected_hashes.keys())
    diff_proc = _run_git(workspace, "diff", "--name-only", check=False)
    for changed in [line.strip() for line in diff_proc.stdout.splitlines() if line.strip()]:
        if changed in protected_set:
            rep.add_violation(
                ViolationType.PROTECTED_PATH_IN_DIFF,
                changed,
                f"Workspace diff touches protected test path {changed!r}",
                phase="post",
            )

    # 3. Walk entire workspace (excluding .git) for symlink escapes & runner shims
    for dirpath, dirnames, filenames in os.walk(workspace, followlinks=False):
        if ".git" in dirnames:
            dirnames.remove(".git")
        current_dir = Path(dirpath)

        # Check directory symlinks too
        for dname in list(dirnames):
            dpath = current_dir / dname
            if dpath.is_symlink():
                resolved = os.path.realpath(str(dpath))
                if not _is_within_root(resolved, workspace_real):
                    rep.add_violation(
                        ViolationType.SYMLINK_ESCAPE,
                        str(dpath.relative_to(workspace)),
                        f"Directory symlink escapes workspace root -> {resolved}",
                        phase="post",
                    )
                dirnames.remove(dname)
            elif dname == "_pytest":
                rep.add_violation(
                    ViolationType.TEST_RUNNER_SHIM,
                    str(dpath.relative_to(workspace)),
                    "Injected _pytest package directory detected",
                    phase="post",
                )

        for fname in filenames:
            fpath = current_dir / fname
            rel_path = str(fpath.relative_to(workspace))

            if fpath.is_symlink():
                resolved = os.path.realpath(str(fpath))
                if not _is_within_root(resolved, workspace_real):
                    rep.add_violation(
                        ViolationType.SYMLINK_ESCAPE,
                        rel_path,
                        f"Symlink {rel_path!r} resolves outside sandbox root -> {resolved}",
                        phase="post",
                    )
                elif resolved.startswith(os.path.join(workspace_real, ".git")):
                    rep.add_violation(
                        ViolationType.SYMLINK_ESCAPE,
                        rel_path,
                        f"Symlink {rel_path!r} targets internal .git directory",
                        phase="post",
                    )

            # Check for unauthorized Python / pytest shim injection
            if fname in {"sitecustomize.py", "usercustomize.py"} and rel_path not in base_file_set:
                rep.add_violation(
                    ViolationType.SITECUSTOMIZE_INJECTION,
                    rel_path,
                    f"Unauthorized Python startup hook {rel_path!r} injected",
                    phase="post",
                )
            elif fname in {"pytest.py", "unittest.py", "pluggy.py"} and rel_path not in base_file_set:
                rep.add_violation(
                    ViolationType.TEST_RUNNER_SHIM,
                    rel_path,
                    f"Unauthorized test runner shadowing module {rel_path!r} injected",
                    phase="post",
                )
            elif fname == "conftest.py" and rel_path not in base_file_set:
                rep.add_violation(
                    ViolationType.TEST_RUNNER_SHIM,
                    rel_path,
                    f"Unauthorized pytest conftest.py injected at {rel_path!r}",
                    phase="post",
                )

    # 4. Check for unauthorized environment overrides (PYTHONPATH, PYTEST_ADDOPTS, etc.)
    if agent_env_overrides:
        for k, v in agent_env_overrides.items():
            if k.upper() in TAMPER_ENV_KEYS:
                rep.add_violation(
                    ViolationType.ENV_TAMPER_PYTHONPATH,
                    k,
                    f"Agent attempted to inject tampered environment variable {k}={v!r}",
                    phase="post",
                )

    return rep


def _is_within_root(resolved_path: str, root_real: str) -> bool:
    try:
        common = os.path.commonpath([resolved_path, root_real])
    except ValueError:
        return False
    return common == root_real
