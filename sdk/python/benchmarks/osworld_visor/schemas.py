"""Typed schemas for the reproducible OSWorld computer-workflow Visor benchmark.

All telemetry fields record real OS-level measurements (monotonic wall-clock
nanoseconds, kernel ``os.wait4`` per-child peak RSS and CPU time, Linux
``/proc/self/io`` logical/storage I/O counters, physical workspace copy bytes,
and post-teardown process/filesystem/socket cleanup audits). No synthetic or
hardcoded latency/memory constants are used anywhere in this package.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SandboxProfile(str, Enum):
    """Sandbox provisioning and isolation profile."""

    COLD_SANDBOX = "cold_sandbox"
    WARM_CAS_VISOR = "warm_cas_visor"


class TaskCategory(str, Enum):
    """Computer-workflow task category aligned with OSWorld taxonomy."""

    OS_FILESYSTEM = "os_filesystem"
    SPREADSHEET_DATA = "spreadsheet_data"
    DATABASE_PIPELINE = "database_pipeline"
    DEV_CLI_GIT = "dev_cli_git"
    ARCHIVE_CONFIG = "archive_config"
    DOCUMENT_OFFICE = "document_office"
    WEB_LOCAL_HTTP = "web_local_http"
    SYSTEM_SECURITY_CLEANUP = "system_security_cleanup"


class TaskStatus(str, Enum):
    """Terminal status for a single task rollout."""

    SUCCESS = "success"
    FAILED = "failed"
    SANDBOX_ESCAPE_BLOCKED = "sandbox_escape_blocked"
    CLEANUP_VIOLATION = "cleanup_violation"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass(frozen=True)
class WorkflowStep:
    """A single real executable workflow step run inside an isolated subprocess."""

    name: str
    description: str
    command: List[str]
    timeout_s: float = 20.0
    working_subdir: str = ""


@dataclass(frozen=True)
class OSWorldTaskSpec:
    """Reproducible clean-room computer-workflow task modeled on OSWorld."""

    task_id: str
    title: str
    category: TaskCategory
    osworld_domain: str
    instruction: str
    seed_files: Dict[str, str]
    oracle_steps: List[WorkflowStep]
    evaluator_kind: str
    evaluator_spec: Dict[str, Any] = field(default_factory=dict)
    readonly_assets: Dict[str, str] = field(default_factory=dict)
    initial_modes: Dict[str, int] = field(default_factory=dict)
    setup_git_repo: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExternalOSWorldTask:
    """Parsed external upstream xlang-ai/OSWorld Apache-2.0 JSON task spec."""

    task_id: str
    source_path: str
    snapshot: str
    instruction: str
    related_apps: List[str]
    evaluator_func: str
    requires_gui: bool
    compatible_with_host: bool
    compatibility_reason: str
    raw_config_steps: int


@dataclass
class DesktopToolingReport:
    """Transparent host capability probe for GUI/sandbox execution backends."""

    execution_backend: str
    platform: str
    display_env: Optional[str]
    gui_stack_ready: bool
    binaries: Dict[str, Optional[str]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SandboxResourceTelemetry:
    """100% empirically measured OS timing, memory, CPU, and I/O telemetry."""

    profile: SandboxProfile
    cold_start_ms: float = 0.0
    provision_ms: float = 0.0
    subprocess_boot_ms: float = 0.0
    execution_ms: float = 0.0
    evaluation_ms: float = 0.0
    cleanup_ms: float = 0.0
    wall_ms: float = 0.0
    peak_rss_kib: int = 0
    rss_source: str = "unmeasured"
    cpu_user_ms: float = 0.0
    cpu_system_ms: float = 0.0
    proc_io_available: bool = False
    rchar_bytes: int = 0
    wchar_bytes: int = 0
    syscr_ops: int = 0
    syscw_ops: int = 0
    read_bytes_storage: int = 0
    write_bytes_storage: int = 0
    rusage_inblock: int = 0
    rusage_oublock: int = 0
    bytes_copied: int = 0
    hardlinked_files: int = 0
    cas_cache_hit: bool = False
    subprocess_launches: int = 0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["profile"] = self.profile.value
        return data


@dataclass
class CleanupAuditResult:
    """Real post-execution resource cleanup verification."""

    cleanup_clean: bool = True
    leaked_temp_files: List[str] = field(default_factory=list)
    orphan_pids_detected: List[int] = field(default_factory=list)
    orphan_pids_reaped: List[int] = field(default_factory=list)
    lingering_sockets: List[str] = field(default_factory=list)
    cas_template_intact: bool = True
    workspace_removed: bool = True
    violations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SecurityAuditResult:
    """Fail-closed sandbox containment and escape-attempt verification."""

    escaped: bool = False
    violations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelCostTelemetry:
    """Token and USD cost telemetry routed via Hanzo AIConfig + llm_adapters."""

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
class TaskRunResult:
    """Full reproducible outcome for a single (task, agent, profile) rollout."""

    task_id: str
    title: str
    category: TaskCategory
    osworld_domain: str
    agent_name: str
    profile: SandboxProfile
    status: TaskStatus
    success: bool
    score: float
    evaluator_checks: Dict[str, bool]
    evaluator_details: str
    step_count: int
    steps_succeeded: int
    telemetry: SandboxResourceTelemetry
    cleanup: CleanupAuditResult
    security: SecurityAuditResult
    cost: ModelCostTelemetry

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "category": self.category.value,
            "osworld_domain": self.osworld_domain,
            "agent_name": self.agent_name,
            "profile": self.profile.value,
            "status": self.status.value,
            "success": self.success,
            "score": round(self.score, 4),
            "evaluator_checks": dict(self.evaluator_checks),
            "evaluator_details": self.evaluator_details,
            "step_count": self.step_count,
            "steps_succeeded": self.steps_succeeded,
            "telemetry": self.telemetry.to_dict(),
            "cleanup": self.cleanup.to_dict(),
            "security": self.security.to_dict(),
            "cost": self.cost.to_dict(),
        }


@dataclass
class CohortSummary:
    """Aggregated empirical metrics for one (agent, sandbox_profile) cohort."""

    agent_name: str
    profile: SandboxProfile
    total_tasks: int
    succeeded_tasks: int
    success_rate: float
    cleanup_clean_tasks: int
    cleanup_violations: int
    escape_blocks: int
    mean_cold_start_ms: float
    p95_cold_start_ms: float
    mean_provision_ms: float
    mean_wall_ms: float
    mean_peak_rss_kib: float
    max_peak_rss_kib: int
    mean_cpu_ms: float
    total_bytes_copied: int
    total_hardlinked_files: int
    total_wchar_bytes: int
    total_rchar_bytes: int
    total_cost_usd: float
    cost_per_success_usd: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["profile"] = self.profile.value
        return data


@dataclass
class BenchmarkReport:
    """Complete HIP-1348 OSWorld Visor benchmark run report."""

    benchmark_name: str
    hip_reference: str
    osworld_license: str
    manifest_digest: str
    workspace_scale: str
    warm_visor_cas: bool
    concurrency: int
    hardware_info: Dict[str, Any]
    desktop_tooling: DesktopToolingReport
    external_tasks_loaded: int
    external_tasks: List[ExternalOSWorldTask]
    cohorts: List[CohortSummary]
    results: List[TaskRunResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "hip_reference": self.hip_reference,
            "osworld_license": self.osworld_license,
            "manifest_digest": self.manifest_digest,
            "workspace_scale": self.workspace_scale,
            "warm_visor_cas": self.warm_visor_cas,
            "concurrency": self.concurrency,
            "hardware_info": dict(self.hardware_info),
            "desktop_tooling": self.desktop_tooling.to_dict(),
            "external_tasks_loaded": self.external_tasks_loaded,
            "external_tasks": [asdict(t) for t in self.external_tasks],
            "cohorts": [c.to_dict() for c in self.cohorts],
            "results": [r.to_dict() for r in self.results],
        }
