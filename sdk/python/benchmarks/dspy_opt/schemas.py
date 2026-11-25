"""Typed schemas for the HIP-1348 DSPy-style optimization benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

TaskCategory = Literal[
    "json_extraction",
    "tool_routing",
    "workflow_synthesis",
    "injection_defense",
]
SplitName = Literal["train", "dev", "test"]
ComplexityLevel = Literal["simple", "complex"]


class IncidentExtraction(BaseModel):
    """Structured incident record extracted from operational reports."""

    incident_id: str = Field(description="Canonical incident identifier, e.g. INC-4021")
    severity: Literal["critical", "high", "medium", "low"] = Field(
        description="Normalized severity level"
    )
    service: str = Field(description="Affected canonical service name")
    owner_team: str = Field(description="Escalation team identifier")
    impact_usd: int = Field(ge=0, description="Estimated USD impact as integer")
    requires_rollback: bool = Field(
        description="Whether an immediate rollback is required"
    )


class ToolRoutePlan(BaseModel):
    """Multi-step tool routing classification over canonical Hanzo surfaces."""

    primary_tool: Literal[
        "vfs_read",
        "vfs_write",
        "s3_artifact_store",
        "visor_sandbox_exec",
        "enso_model_route",
        "cloud_billing_check",
        "kai_policy_gate",
    ]
    secondary_tool: Optional[
        Literal[
            "vfs_read",
            "vfs_write",
            "s3_artifact_store",
            "visor_sandbox_exec",
            "enso_model_route",
            "cloud_billing_check",
            "kai_policy_gate",
        ]
    ] = None
    requires_approval: bool
    target_env: Literal["dev", "staging", "prod"]


class WorkflowStep(BaseModel):
    """Single step in a synthesized workflow DAG."""

    step_id: str
    tool: Literal[
        "vfs_read",
        "vfs_write",
        "s3_artifact_store",
        "visor_sandbox_exec",
        "enso_model_route",
        "cloud_billing_check",
        "kai_policy_gate",
    ]
    depends_on: List[str] = Field(default_factory=list)
    est_latency_ms: int = Field(gt=0)
    est_cost_millicents: int = Field(ge=0)


class WorkflowDAG(BaseModel):
    """Constraint-verified workflow DAG executed through AgentWorkflow."""

    workflow_name: str
    steps: List[WorkflowStep]
    has_checkpoint_gate: bool


@dataclass(frozen=True)
class TaskSpec:
    """Single benchmark task with ground-truth executable verification criteria."""

    task_id: str
    split: SplitName
    category: TaskCategory
    complexity: ComplexityLevel
    prompt_input: str
    expected: Dict[str, Any]
    constraints: Dict[str, Any] = field(default_factory=dict)
    is_adversarial: bool = False
    injection_target: Optional[str] = None
    content_fingerprint: str = ""


@dataclass(frozen=True)
class Demonstration:
    """Passing execution trace selected as a few-shot demonstration."""

    task_id: str
    category: TaskCategory
    prompt_input: str
    expected_output_json: str
    rationale: str = ""


@dataclass(frozen=True)
class CompiledProgram:
    """Compiled prompt + demonstration + runtime parameter program."""

    strategy: str
    category: TaskCategory
    instruction: str
    demonstrations: tuple[Demonstration, ...] = ()
    model: str = "openai/zen-coder-480b"
    temperature: float = 0.0
    max_tokens: int = 384
    metadata: Dict[str, Any] = field(default_factory=dict)

    def render_system_prompt(self) -> str:
        """Render the complete system prompt including any selected demonstrations."""
        parts = [self.instruction.strip()]
        if self.demonstrations:
            demo_lines = ["", "### Verified Demonstrations"]
            for idx, demo in enumerate(self.demonstrations, start=1):
                demo_lines.append(f"Example {idx} Input:\n{demo.prompt_input}")
                if demo.rationale:
                    demo_lines.append(f"Example {idx} Trace:\n{demo.rationale}")
                demo_lines.append(
                    f"Example {idx} Output:\n{demo.expected_output_json}"
                )
            parts.append("\n".join(demo_lines))
        return "\n".join(parts)


@dataclass
class TaskResult:
    """Measured execution result for a single task invocation."""

    task_id: str
    split: SplitName
    category: TaskCategory
    complexity: ComplexityLevel
    strategy: str
    passed: bool
    schema_valid: bool
    injection_defended: Optional[bool]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: float
    workflow_steps_executed: int = 1
    error: Optional[str] = None
    output: Optional[Dict[str, Any]] = None
