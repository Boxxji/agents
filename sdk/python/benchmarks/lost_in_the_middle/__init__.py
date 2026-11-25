"""Lost-in-the-Middle evidence-ordering & context-packing benchmark for Hanzo Agents.

Evaluates context packing policies across gold-document positions 1..N (N=5, 10, 20),
strict token budgets (512, 1024, 2048, 4096), and explicit failure/negative regimes
over Hanzo's MemoryClient, ScopedMemoryClient, AIConfig, and llm_adapters.
"""

from .dataset import (
    BenchmarkCase,
    Document,
    generate_negative_regime_suite,
    generate_position_sweep_suite,
    load_jsonl_cases,
)
from .evaluator import (
    BudgetTruncationEvaluator,
    CalibratedTACL2024Evaluator,
    CaseEvaluation,
    ContextEvaluator,
    DistractorInterferenceEvaluator,
    LiveOpenAICompatibleEvaluator,
    MemoryBackedRetriever,
    TACL2024_PROFILES,
)
from .metrics import (
    BootstrapCI,
    PolicyAggregate,
    aggregate_evaluations,
    exact_match,
    normalize_answer,
    paired_bootstrap_ci,
    substring_match,
    token_f1,
)
from .packers import (
    POLICY_NAMES,
    PackedContext,
    audit_boundary_integrity,
    estimate_tokens,
    jaccard_similarity,
    pack_context,
)
from .runner import BenchmarkConfig, render_score_table_markdown, run_benchmark

__all__ = [
    "BenchmarkCase",
    "BenchmarkConfig",
    "BootstrapCI",
    "BudgetTruncationEvaluator",
    "CalibratedTACL2024Evaluator",
    "CaseEvaluation",
    "ContextEvaluator",
    "DistractorInterferenceEvaluator",
    "Document",
    "LiveOpenAICompatibleEvaluator",
    "MemoryBackedRetriever",
    "POLICY_NAMES",
    "PackedContext",
    "PolicyAggregate",
    "TACL2024_PROFILES",
    "aggregate_evaluations",
    "audit_boundary_integrity",
    "estimate_tokens",
    "exact_match",
    "generate_negative_regime_suite",
    "generate_position_sweep_suite",
    "jaccard_similarity",
    "load_jsonl_cases",
    "normalize_answer",
    "pack_context",
    "paired_bootstrap_ci",
    "render_score_table_markdown",
    "run_benchmark",
    "substring_match",
    "token_f1",
]
