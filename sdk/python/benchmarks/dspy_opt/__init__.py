"""Clean-room DSPy-style prompt and workflow optimization benchmark for Hanzo Agents (HIP-1348)."""

from .dataset import (
    CATEGORIES,
    BenchmarkDataset,
    assert_no_split_leakage,
    build_dataset,
    evaluate_task_output,
)
from .metrics import (
    CategoryMetrics,
    StrategyMetrics,
    aggregate_metrics,
    render_markdown_table,
)
from .optimizers import (
    BootstrapFewShotOptimizer,
    InstructionSearchOptimizer,
    ParetoCostAwareOptimizer,
    ZeroShotBaselineOptimizer,
    validate_security_boundary,
)
from .runner import BenchmarkReport, run_benchmark
from .runtime import HanzoProgramExecutor, PricingModel, SimulatedEnsoZenProvider
from .schemas import (
    CompiledProgram,
    Demonstration,
    IncidentExtraction,
    TaskResult,
    TaskSpec,
    ToolRoutePlan,
    WorkflowDAG,
    WorkflowStep,
)

__all__ = [
    "BenchmarkDataset",
    "BenchmarkReport",
    "BootstrapFewShotOptimizer",
    "CATEGORIES",
    "CategoryMetrics",
    "CompiledProgram",
    "Demonstration",
    "HanzoProgramExecutor",
    "IncidentExtraction",
    "InstructionSearchOptimizer",
    "ParetoCostAwareOptimizer",
    "PricingModel",
    "SimulatedEnsoZenProvider",
    "StrategyMetrics",
    "TaskResult",
    "TaskSpec",
    "ToolRoutePlan",
    "WorkflowDAG",
    "WorkflowStep",
    "ZeroShotBaselineOptimizer",
    "aggregate_metrics",
    "assert_no_split_leakage",
    "build_dataset",
    "evaluate_task_output",
    "render_markdown_table",
    "run_benchmark",
    "validate_security_boundary",
]
