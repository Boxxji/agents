"""Quantitative metric aggregation and markdown score-table rendering."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

from .schemas import TaskCategory, TaskResult


def percentile(values: Sequence[float], pct: float) -> float:
    """Compute linear-interpolated percentile (0-100) deterministically."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(float(ordered[0]), 2)
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return round(float(ordered[lo]), 2)
    frac = rank - lo
    return round(float(ordered[lo] + frac * (ordered[hi] - ordered[lo])), 2)


@dataclass
class CategoryMetrics:
    """Held-out test metrics broken down by a single task category."""

    category: str
    tasks: int
    passed: int
    pass_rate_pct: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    cost_per_passing_task_usd: Optional[float]
    injection_defense_rate_pct: Optional[float]
    p50_latency_ms: float
    p95_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StrategyMetrics:
    """Aggregate metrics for an optimization strategy across train, dev, and held-out test splits."""

    strategy: str
    train_pass_rate_pct: float
    dev_pass_rate_pct: float
    test_pass_rate_pct: float
    train_vs_test_overfitting_gap_pp: float
    dev_vs_test_overfitting_gap_pp: float
    test_tasks: int
    test_passed: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    cost_per_passing_task_usd: Optional[float]
    injection_defense_rate_pct: float
    p50_latency_ms: float
    p95_latency_ms: float
    simple_pass_rate_pct: float
    simple_total_tokens: int
    simple_cost_usd: float
    complex_pass_rate_pct: float
    complex_total_tokens: int
    complex_cost_usd: float
    by_category: Dict[str, CategoryMetrics]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["by_category"] = {k: v.to_dict() for k, v in self.by_category.items()}
        return data


def _pass_rate_pct(results: Sequence[TaskResult]) -> float:
    if not results:
        return 0.0
    return round(100.0 * sum(1 for r in results if r.passed) / len(results), 2)


def aggregate_metrics(
    strategy: str,
    train_results: Sequence[TaskResult],
    dev_results: Sequence[TaskResult],
    test_results: Sequence[TaskResult],
) -> StrategyMetrics:
    """Compute complete metrics for a strategy from train/dev/test execution results."""
    train_pr = _pass_rate_pct(train_results)
    dev_pr = _pass_rate_pct(dev_results)
    test_pr = _pass_rate_pct(test_results)

    test_passed = sum(1 for r in test_results if r.passed)
    prompt_tokens = sum(r.prompt_tokens for r in test_results)
    completion_tokens = sum(r.completion_tokens for r in test_results)
    total_tokens = sum(r.total_tokens for r in test_results)
    total_cost = sum(r.cost_usd for r in test_results)
    cost_per_pass = (
        round(total_cost / test_passed, 6) if test_passed > 0 else None
    )

    adv_results = [r for r in test_results if r.injection_defended is not None]
    inj_rate = (
        round(
            100.0
            * sum(1 for r in adv_results if r.injection_defended)
            / len(adv_results),
            2,
        )
        if adv_results
        else 0.0
    )

    latencies = [r.latency_ms for r in test_results]
    simple_res = [r for r in test_results if r.complexity == "simple"]
    complex_res = [r for r in test_results if r.complexity == "complex"]

    categories: List[TaskCategory] = [
        "json_extraction",
        "tool_routing",
        "workflow_synthesis",
        "injection_defense",
    ]
    by_cat: Dict[str, CategoryMetrics] = {}
    for cat in categories:
        cat_res = [r for r in test_results if r.category == cat]
        if not cat_res:
            continue
        c_pass = sum(1 for r in cat_res if r.passed)
        c_cost = sum(r.cost_usd for r in cat_res)
        c_adv = [r for r in cat_res if r.injection_defended is not None]
        c_inj = (
            round(
                100.0
                * sum(1 for r in c_adv if r.injection_defended)
                / len(c_adv),
                2,
            )
            if c_adv
            else None
        )
        by_cat[cat] = CategoryMetrics(
            category=cat,
            tasks=len(cat_res),
            passed=c_pass,
            pass_rate_pct=_pass_rate_pct(cat_res),
            prompt_tokens=sum(r.prompt_tokens for r in cat_res),
            completion_tokens=sum(r.completion_tokens for r in cat_res),
            total_tokens=sum(r.total_tokens for r in cat_res),
            estimated_cost_usd=round(c_cost, 6),
            cost_per_passing_task_usd=round(c_cost / c_pass, 6) if c_pass > 0 else None,
            injection_defense_rate_pct=c_inj,
            p50_latency_ms=percentile([r.latency_ms for r in cat_res], 50),
            p95_latency_ms=percentile([r.latency_ms for r in cat_res], 95),
        )

    return StrategyMetrics(
        strategy=strategy,
        train_pass_rate_pct=train_pr,
        dev_pass_rate_pct=dev_pr,
        test_pass_rate_pct=test_pr,
        train_vs_test_overfitting_gap_pp=round(train_pr - test_pr, 2),
        dev_vs_test_overfitting_gap_pp=round(dev_pr - test_pr, 2),
        test_tasks=len(test_results),
        test_passed=test_passed,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=round(total_cost, 6),
        cost_per_passing_task_usd=cost_per_pass,
        injection_defense_rate_pct=inj_rate,
        p50_latency_ms=percentile(latencies, 50),
        p95_latency_ms=percentile(latencies, 95),
        simple_pass_rate_pct=_pass_rate_pct(simple_res),
        simple_total_tokens=sum(r.total_tokens for r in simple_res),
        simple_cost_usd=round(sum(r.cost_usd for r in simple_res), 6),
        complex_pass_rate_pct=_pass_rate_pct(complex_res),
        complex_total_tokens=sum(r.total_tokens for r in complex_res),
        complex_cost_usd=round(sum(r.cost_usd for r in complex_res), 6),
        by_category=by_cat,
    )


def render_markdown_table(
    strategies: Sequence[StrategyMetrics],
    ablations: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> str:
    """Render reproducible held-out score tables in GitHub-flavored Markdown."""
    lines: List[str] = [
        "# HIP-1348 DSPy-Style Prompt & Workflow Optimization — Held-Out Score Table",
        "",
        f"- **Seed**: `{seed}`",
        "- **Held-out test set**: 48 executable tasks (12 per category; strictly disjoint from 32 train and 24 dev tasks)",
        "- **Runtime**: Hanzo Python SDK (`AIConfig`, `llm_adapters`, `AgentAI`, `AgentWorkflow`) over deterministic Enso/Zen simulator",
        "",
        "## Primary Held-Out Results (4 Strategies)",
        "",
        "| Strategy | Held-Out Pass Rate (%) | Train-vs-Test Gap (pp) | Total Tokens | Est. Cost (USD) | Cost / Passing Task (USD) | Injection Defense (%) | p50 Latency (ms) | p95 Latency (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m in strategies:
        cpp = (
            f"${m.cost_per_passing_task_usd:.6f}"
            if m.cost_per_passing_task_usd is not None
            else "N/A"
        )
        lines.append(
            f"| `{m.strategy}` | {m.test_pass_rate_pct:.2f}% ({m.test_passed}/{m.test_tasks}) "
            f"| {m.train_vs_test_overfitting_gap_pp:+.2f} | {m.total_tokens:,} "
            f"| ${m.estimated_cost_usd:.6f} | {cpp} "
            f"| {m.injection_defense_rate_pct:.2f}% | {m.p50_latency_ms:.2f} | {m.p95_latency_ms:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Per-Category Held-Out Pass Rate (%) & Tokens",
            "",
            "| Strategy | JSON Extraction | Tool Routing | Workflow Synthesis | Injection Defense |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for m in strategies:
        cells = []
        for cat in (
            "json_extraction",
            "tool_routing",
            "workflow_synthesis",
            "injection_defense",
        ):
            cm = m.by_category.get(cat)
            if cm:
                cells.append(f"{cm.pass_rate_pct:.1f}% ({cm.total_tokens:,} tok)")
            else:
                cells.append("—")
        lines.append(f"| `{m.strategy}` | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Negative Regimes & Failure-Mode Ablations",
            "",
            "### 1. Over-Prompting Bloat on Simple Tasks (Held-Out Simple Subset)",
            "",
            "| Strategy | Simple Pass Rate (%) | Simple Total Tokens | Simple Est. Cost (USD) | Token Overhead vs Baseline |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    base_simple_tokens = strategies[0].simple_total_tokens if strategies else 1
    for m in strategies:
        delta_pct = (
            100.0 * (m.simple_total_tokens - base_simple_tokens) / base_simple_tokens
            if base_simple_tokens
            else 0.0
        )
        lines.append(
            f"| `{m.strategy}` | {m.simple_pass_rate_pct:.2f}% "
            f"| {m.simple_total_tokens:,} | ${m.simple_cost_usd:.6f} | {delta_pct:+.1f}% |"
        )

    if ablations:
        small_dev = ablations.get("small_dev_overfitting")
        if small_dev:
            lines.extend(
                [
                    "",
                    "### 2. Small-Dev Overfitting Regime (`dev_size=4`, 1 task/category)",
                    "",
                    "| Configuration | Dev Pass Rate (%) | Held-Out Test Pass Rate (%) | Dev-vs-Test Overfitting Gap (pp) |",
                    "| --- | ---: | ---: | ---: |",
                    f"| `instruction_search` (full dev=24) | {small_dev['full_dev']['dev_pass_rate_pct']:.2f}% | {small_dev['full_dev']['test_pass_rate_pct']:.2f}% | {small_dev['full_dev']['dev_vs_test_gap_pp']:+.2f} |",
                    f"| `instruction_search` (tiny dev=4) | {small_dev['tiny_dev']['dev_pass_rate_pct']:.2f}% | {small_dev['tiny_dev']['test_pass_rate_pct']:.2f}% | {small_dev['tiny_dev']['dev_vs_test_gap_pp']:+.2f} |",
                ]
            )

        sec_abl = ablations.get("unguarded_compression_security")
        if sec_abl:
            lines.extend(
                [
                    "",
                    "### 3. Security Boundary Erosion Without Guardrail (Injection Defense Held-Out)",
                    "",
                    "| Optimizer Mode | Candidate Selected | Injection Defense Rate (%) | Held-Out Pass Rate (%) |",
                    "| --- | --- | ---: | ---: |",
                    f"| `instruction_search` (security guard ON) | `{sec_abl['guarded']['candidate_id']}` | {sec_abl['guarded']['injection_defense_rate_pct']:.2f}% | {sec_abl['guarded']['pass_rate_pct']:.2f}% |",
                    f"| `pareto_cost_aware` (security guard OFF, strict 615-tok cap) | `{sec_abl['unguarded_compressed']['candidate_id']}` | {sec_abl['unguarded_compressed']['injection_defense_rate_pct']:.2f}% | {sec_abl['unguarded_compressed']['pass_rate_pct']:.2f}% |",
                ]
            )

    lines.append("")
    return "\n".join(lines)
