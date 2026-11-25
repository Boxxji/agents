"""Schemas and telemetry models for the SWE-bench Pro V2 Visor subset runner.

All types are clean-room Apache-2.0 data structures aligned with Hanzo's
canonical architecture (vfs -> Hanzo S3 -> Visor) and SWE-bench Pro V2
(revision ``sep-2026``, 642 validated tasks across 4 languages).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class StorageProfile(str, Enum):
    """Sandbox workspace storage profile under evaluation."""

    PLAIN_NVME = "plain_nvme"
    VISOR_ISOLATED = "visor_isolated"


class TaskStatus(str, Enum):
    """Terminal execution classification for a SWE-bench Pro V2 task."""

    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    LEAKAGE_BLOCKED = "leakage_blocked"
    TAMPERED = "tampered"
    TIMEOUT = "timeout"
    ERROR = "error"


class ViolationType(str, Enum):
    """Enumerated anti-leakage and anti-tampering violation codes."""

    GIT_REMOTE_PRESENT = "git_remote_present"
    GIT_TAG_PRESENT = "git_tag_present"
    GIT_EXTRA_BRANCH = "git_extra_branch"
    GIT_REFLOG_PRESENT = "git_reflog_present"
    GIT_STATE_FILE_PRESENT = "git_state_file_present"
    GIT_FUTURE_COMMIT = "git_future_commit"
    GIT_GOLD_OBJECT_LEAK = "git_gold_object_leak"
    LEAK_ARTIFACT_PRESENT = "leak_artifact_present"
    ENV_LEAK_VAR = "env_leak_var"
    CREDENTIAL_SCOPE_VIOLATION = "credential_scope_violation"
    TEST_FILE_MODIFIED = "test_file_modified"
    TEST_FILE_MISSING = "test_file_missing"
    PROTECTED_PATH_IN_DIFF = "protected_path_in_diff"
    SYMLINK_ESCAPE = "symlink_escape"
    ENV_TAMPER_PYTHONPATH = "env_tamper_pythonpath"
    SITECUSTOMIZE_INJECTION = "sitecustomize_injection"
    TEST_RUNNER_SHIM = "test_runner_shim"


@dataclass
class IntegrityViolation:
    """Single anti-leakage or anti-tampering audit finding."""

    code: ViolationType
    target: str
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code.value,
            "target": self.target,
            "detail": self.detail,
        }


@dataclass
class IntegrityReport:
    """Pre-execution anti-leakage and post-execution anti-tampering audit report."""

    clean: bool = True
    pre_audit_clean: bool = True
    post_audit_clean: bool = True
    stripped_remotes: List[str] = field(default_factory=list)
    stripped_tags: List[str] = field(default_factory=list)
    stripped_branches: List[str] = field(default_factory=list)
    stripped_reflogs: bool = False
    pruned_objects: bool = False
    removed_artifacts: List[str] = field(default_factory=list)
    protected_test_hashes: Dict[str, str] = field(default_factory=dict)
    violations: List[IntegrityViolation] = field(default_factory=list)

    def add_violation(
        self,
        code: ViolationType,
        target: str,
        detail: str,
        *,
        phase: str = "post",
    ) -> None:
        self.violations.append(IntegrityViolation(code=code, target=target, detail=detail))
        self.clean = False
        if phase == "pre":
            self.pre_audit_clean = False
        else:
            self.post_audit_clean = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clean": self.clean,
            "pre_audit_clean": self.pre_audit_clean,
            "post_audit_clean": self.post_audit_clean,
            "stripped_remotes": list(self.stripped_remotes),
            "stripped_tags": list(self.stripped_tags),
            "stripped_branches": list(self.stripped_branches),
            "stripped_reflogs": self.stripped_reflogs,
            "pruned_objects": self.pruned_objects,
            "removed_artifacts": list(self.removed_artifacts),
            "protected_test_hashes": dict(self.protected_test_hashes),
            "violations": [v.to_dict() for v in self.violations],
        }


@dataclass
class ModelCostTelemetry:
    """Token accounting and USD cost tracked through Hanzo AIConfig hooks."""

    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0
    budget_exceeded: bool = False
    llm_params_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FilesystemIOTelemetry:
    """Real OS filesystem I/O and workspace materialization metrics."""

    profile: StorageProfile
    proc_io_available: bool = False
    rchar_bytes: int = 0
    wchar_bytes: int = 0
    syscr_ops: int = 0
    syscw_ops: int = 0
    read_bytes_storage: int = 0
    write_bytes_storage: int = 0
    rusage_inblock: int = 0
    rusage_oublock: int = 0
    cpu_user_ms: float = 0.0
    cpu_system_ms: float = 0.0
    workspace_setup_ms: float = 0.0
    workspace_bytes_copied: int = 0
    workspace_files_materialized: int = 0
    workspace_hardlinked_files: int = 0
    snapshot_diff_ms: float = 0.0
    cas_cache_hit: bool = False

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["profile"] = self.profile.value
        return data


@dataclass
class SWEBenchTask:
    """Self-contained multi-file coding task aligned with SWE-bench Pro V2."""

    instance_id: str
    repo: str
    language: str
    difficulty: str
    problem_statement: str
    base_files: Dict[str, str]
    test_files: Dict[str, str]
    gold_patch_files: Dict[str, str]
    gold_marker: str
    fail_to_pass_tests: List[str]
    pass_to_pass_tests: List[str]
    protected_test_paths: List[str]
    expected_touched_files: List[str]
    revision: str = "sep-2026"
    benchmark_version: str = "v2"
    partial_patch_files: Optional[Dict[str, str]] = None


@dataclass
class AgentInvocationResult:
    """Output returned by a coding agent after attempting a SWEBenchTask."""

    patch_diff: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    calls: int = 1
    env_overrides: Dict[str, str] = field(default_factory=dict)
    notes: str = ""


@dataclass
class TaskRunResult:
    """Complete verified telemetry and scoring outcome for one task run."""

    instance_id: str
    agent_name: str
    storage_profile: StorageProfile
    status: TaskStatus
    resolved: bool
    tampered: bool
    leakage_blocked: bool
    fail_to_pass_passed: int
    fail_to_pass_total: int
    pass_to_pass_passed: int
    pass_to_pass_total: int
    wall_time_ms: float
    agent_time_ms: float
    eval_time_ms: float
    cost: ModelCostTelemetry
    io: FilesystemIOTelemetry
    integrity: IntegrityReport
    test_output_tail: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "agent_name": self.agent_name,
            "storage_profile": self.storage_profile.value,
            "status": self.status.value,
            "resolved": self.resolved,
            "tampered": self.tampered,
            "leakage_blocked": self.leakage_blocked,
            "fail_to_pass_passed": self.fail_to_pass_passed,
            "fail_to_pass_total": self.fail_to_pass_total,
            "pass_to_pass_passed": self.pass_to_pass_passed,
            "pass_to_pass_total": self.pass_to_pass_total,
            "wall_time_ms": round(self.wall_time_ms, 3),
            "agent_time_ms": round(self.agent_time_ms, 3),
            "eval_time_ms": round(self.eval_time_ms, 3),
            "cost": self.cost.to_dict(),
            "io": self.io.to_dict(),
            "integrity": self.integrity.to_dict(),
            "test_output_tail": self.test_output_tail,
            "error": self.error,
        }


@dataclass
class ProfileAggregate:
    """Aggregated resolution, latency, cost, and I/O summary for (agent, profile)."""

    agent_name: str
    storage_profile: StorageProfile
    total_tasks: int
    resolved_tasks: int
    resolution_rate: float
    tampered_tasks: int
    leakage_blocked_tasks: int
    mean_wall_time_ms: float
    p95_wall_time_ms: float
    mean_setup_ms: float
    mean_snapshot_diff_ms: float
    total_cost_usd: float
    cost_per_resolved_usd: Optional[float]
    total_rchar_bytes: int
    total_wchar_bytes: int
    total_syscr_ops: int
    total_syscw_ops: int
    total_workspace_bytes_copied: int
    total_hardlinked_files: int
    mean_cpu_ms: float

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["storage_profile"] = self.storage_profile.value
        return data


@dataclass
class BenchmarkReport:
    """Full reproducible SWE-bench Pro V2 subset evaluation report."""

    benchmark_revision: str
    benchmark_version: str
    manifest_digest: str
    repo_scale_name: str
    repo_asset_files: int
    repo_asset_bytes: int
    hardware_info: Dict[str, Any]
    aggregates: List[ProfileAggregate]
    results: List[TaskRunResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_revision": self.benchmark_revision,
            "benchmark_version": self.benchmark_version,
            "manifest_digest": self.manifest_digest,
            "repo_scale_name": self.repo_scale_name,
            "repo_asset_files": self.repo_asset_files,
            "repo_asset_bytes": self.repo_asset_bytes,
            "hardware_info": dict(self.hardware_info),
            "aggregates": [a.to_dict() for a in self.aggregates],
            "results": [r.to_dict() for r in self.results],
        }
