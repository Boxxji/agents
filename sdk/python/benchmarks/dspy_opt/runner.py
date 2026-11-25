"""Reproducible end-to-end benchmark runner for HIP-1348 DSPy-style optimization."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .dataset import CATEGORIES, BenchmarkDataset, assert_no_split_leakage, build_dataset
from .metrics import StrategyMetrics, aggregate_metrics, render_markdown_table
from .optimizers import (
    BootstrapFewShotOptimizer,
    BudgetInfeasibleError,
    InstructionSearchOptimizer,
    ParetoBudget,
    ParetoCostAwareOptimizer,
    ZeroShotBaselineOptimizer,
)
from .runtime import HanzoProgramExecutor
from .schemas import CompiledProgram, TaskCategory, TaskResult


@dataclass
class BenchmarkReport:
    """Complete reproducible benchmark report with raw metrics and negative-regime ablations."""

    seed: int
    dataset_sizes: Dict[str, int]
    strategies: List[StrategyMetrics]
    compiled_programs: Dict[str, Dict[str, Dict[str, Any]]]
    ablations: Dict[str, Any]
    raw_test_results: Dict[str, List[Dict[str, Any]]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": "hip-1348-dspy-prompt-workflow-optimization",
            "seed": self.seed,
            "dataset_sizes": self.dataset_sizes,
            "strategies": [s.to_dict() for s in self.strategies],
            "compiled_programs": self.compiled_programs,
            "ablations": self.ablations,
            "raw_test_results": self.raw_test_results,
        }

    def to_markdown(self) -> str:
        return render_markdown_table(
            self.strategies, ablations=self.ablations, seed=self.seed
        )


async def _run_strategy(
    optimizer: Any,
    dataset: BenchmarkDataset,
    executor: HanzoProgramExecutor,
) -> tuple[StrategyMetrics, Dict[str, Dict[str, Any]], List[TaskResult]]:
    """Compile per-category programs on train/dev and evaluate on train/dev/held-out test."""
    programs: Dict[TaskCategory, CompiledProgram] = {}

    # Compile categories concurrently to verify concurrency safety
    async def _compile_cat(cat: TaskCategory) -> tuple[TaskCategory, CompiledProgram]:
        prog = await optimizer.compile(
            category=cat,
            train_tasks=dataset.by_category("train", cat),
            dev_tasks=dataset.by_category("dev", cat),
            executor=executor,
        )
        return cat, prog

    compiled_pairs = await asyncio.gather(*[_compile_cat(cat) for cat in CATEGORIES])
    for cat, prog in compiled_pairs:
        programs[cat] = prog

    # Evaluate on train, dev, and held-out test splits
    async def _eval_split(split_name: str) -> List[TaskResult]:
        coros = [
            executor.evaluate_program(
                programs[cat], list(dataset.by_category(split_name, cat))  # type: ignore[arg-type]
            )
            for cat in CATEGORIES
        ]
        nested = await asyncio.gather(*coros)
        flat: List[TaskResult] = []
        for chunk in nested:
            flat.extend(chunk)
        flat.sort(key=lambda r: r.task_id)
        return flat

    train_res, dev_res, test_res = await asyncio.gather(
        _eval_split("train"),
        _eval_split("dev"),
        _eval_split("test"),
    )

    metrics = aggregate_metrics(optimizer.name, train_res, dev_res, test_res)
    program_summaries = {
        cat: {
            "model": prog.model,
            "demo_count": len(prog.demonstrations),
            "demo_task_ids": [d.task_id for d in prog.demonstrations],
            "instruction": prog.instruction,
            "metadata": prog.metadata,
        }
        for cat, prog in programs.items()
    }
    return metrics, program_summaries, test_res


async def _run_ablations(
    dataset: BenchmarkDataset,
    executor: HanzoProgramExecutor,
    full_instruction_metrics: StrategyMetrics,
    full_instruction_programs: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Run negative-regime ablations: small-dev overfitting & unguarded compression security loss."""
    # 1. Small-dev overfitting ablation: pick a tiny 1-task-per-category dev slice
    #    where brittle short heuristics score 100% on dev but degrade on held-out test.
    tiny_dev_by_cat: Dict[TaskCategory, List[Any]] = {}
    for cat in CATEGORIES:
        cat_dev = list(dataset.by_category("dev", cat))
        # Pick 1 task where narrow heuristics look artificially strong
        if cat == "json_extraction":
            chosen = [
                t
                for t in cat_dev
                if t.expected["severity"] == "high" and t.expected["requires_rollback"]
            ][:1] or cat_dev[:1]
        elif cat == "tool_routing":
            chosen = [t for t in cat_dev if t.expected["requires_approval"]][:1] or cat_dev[:1]
        elif cat == "workflow_synthesis":
            chosen = [t for t in cat_dev if t.complexity == "simple"][:1] or cat_dev[:1]
        else:
            chosen = cat_dev[:1]
        tiny_dev_by_cat[cat] = chosen

    tiny_opt = InstructionSearchOptimizer(
        enforce_security_guard=True, prefer_shorter_on_tie=True
    )
    tiny_dev_results: List[TaskResult] = []
    tiny_test_results: List[TaskResult] = []
    tiny_selected: Dict[str, str] = {}
    for cat in CATEGORIES:
        prog = await tiny_opt.compile(
            category=cat,
            train_tasks=dataset.by_category("train", cat),
            dev_tasks=tiny_dev_by_cat[cat],
            executor=executor,
        )
        tiny_selected[cat] = str(prog.metadata.get("candidate_id", ""))
        tiny_dev_results.extend(
            await executor.evaluate_program(prog, tiny_dev_by_cat[cat])
        )
        tiny_test_results.extend(
            await executor.evaluate_program(prog, list(dataset.by_category("test", cat)))
        )

    tiny_dev_pr = round(
        100.0 * sum(1 for r in tiny_dev_results if r.passed) / len(tiny_dev_results), 2
    )
    tiny_test_pr = round(
        100.0 * sum(1 for r in tiny_test_results if r.passed) / len(tiny_test_results),
        2,
    )

    # 2. Unguarded compression security ablation on injection_defense:
    #    Show what happens when a cost-capped optimizer disables the security boundary guard
    #    under an aggressive 615-token cap (terse/baseline prompts fit, hardened boundary exceeds cap)
    #    versus keeping the security guard ON (explicitly reports BudgetInfeasibleError instead of
    #    silently returning an over-budget candidate).
    tight_budget = ParetoBudget(
        max_avg_tokens_per_task=615,
        max_cost_per_task_usd=0.0020,
        max_accuracy_drop_from_best=0.0,
    )
    guarded_tight_opt = ParetoCostAwareOptimizer(
        budget=tight_budget,
        enforce_security_guard=True,
        demo_counts=(0,),
    )
    try:
        guarded_tight_prog = await guarded_tight_opt.compile(
            category="injection_defense",
            train_tasks=dataset.by_category("train", "injection_defense"),
            dev_tasks=dataset.by_category("dev", "injection_defense"),
            executor=executor,
        )
        guarded_tight_info: Dict[str, Any] = {
            "compiled": True,
            "infeasible": False,
            "candidate_id": guarded_tight_prog.metadata.get("candidate_id"),
            "dev_avg_tokens": guarded_tight_prog.metadata.get("dev_avg_tokens"),
            "budget_max_avg_tokens": tight_budget.max_avg_tokens_per_task,
        }
    except BudgetInfeasibleError as exc:
        guarded_tight_info = {
            "compiled": False,
            "infeasible": True,
            "reason": str(exc),
            "min_avg_tokens": round(exc.min_avg_tokens, 2)
            if exc.min_avg_tokens is not None
            else None,
            "budget_max_avg_tokens": tight_budget.max_avg_tokens_per_task,
            "rejected_unsafe_count": len(exc.rejected_unsafe_candidates),
        }

    unguarded_opt = ParetoCostAwareOptimizer(
        budget=tight_budget,
        enforce_security_guard=False,
        demo_counts=(0,),
    )
    unguarded_prog = await unguarded_opt.compile(
        category="injection_defense",
        train_tasks=dataset.by_category("train", "injection_defense"),
        dev_tasks=dataset.by_category("dev", "injection_defense"),
        executor=executor,
    )
    unguarded_test = await executor.evaluate_program(
        unguarded_prog, list(dataset.by_category("test", "injection_defense"))
    )
    unguarded_inj_rate = round(
        100.0
        * sum(1 for r in unguarded_test if r.injection_defended)
        / len(unguarded_test),
        2,
    )
    unguarded_pr = round(
        100.0 * sum(1 for r in unguarded_test if r.passed) / len(unguarded_test), 2
    )

    guarded_cat = full_instruction_metrics.by_category["injection_defense"]
    return {
        "small_dev_overfitting": {
            "full_dev": {
                "dev_size": len(dataset.dev),
                "dev_pass_rate_pct": full_instruction_metrics.dev_pass_rate_pct,
                "test_pass_rate_pct": full_instruction_metrics.test_pass_rate_pct,
                "dev_vs_test_gap_pp": full_instruction_metrics.dev_vs_test_overfitting_gap_pp,
            },
            "tiny_dev": {
                "dev_size": sum(len(v) for v in tiny_dev_by_cat.values()),
                "selected_candidates": tiny_selected,
                "dev_pass_rate_pct": tiny_dev_pr,
                "test_pass_rate_pct": tiny_test_pr,
                "dev_vs_test_gap_pp": round(tiny_dev_pr - tiny_test_pr, 2),
            },
        },
        "unguarded_compression_security": {
            "guarded": {
                "candidate_id": full_instruction_programs["injection_defense"][
                    "metadata"
                ].get("candidate_id"),
                "injection_defense_rate_pct": guarded_cat.injection_defense_rate_pct,
                "pass_rate_pct": guarded_cat.pass_rate_pct,
            },
            "guarded_tight_cap": guarded_tight_info,
            "unguarded_compressed": {
                "candidate_id": unguarded_prog.metadata.get("candidate_id"),
                "injection_defense_rate_pct": unguarded_inj_rate,
                "pass_rate_pct": unguarded_pr,
            },
        },
    }


async def run_benchmark(
    seed: int = 42,
    max_concurrency: int = 8,
    output_dir: Optional[Path] = None,
) -> BenchmarkReport:
    """Run the full 4-strategy benchmark + negative-regime ablations deterministically."""
    dataset = build_dataset(seed=seed)
    assert_no_split_leakage(dataset)

    optimizers = [
        ZeroShotBaselineOptimizer(),
        BootstrapFewShotOptimizer(max_demos=5),
        InstructionSearchOptimizer(enforce_security_guard=True),
        ParetoCostAwareOptimizer(
            budget=ParetoBudget(
                max_avg_tokens_per_task=900,
                max_cost_per_task_usd=0.0045,
                max_accuracy_drop_from_best=0.02,
            ),
            enforce_security_guard=True,
        ),
    ]

    strategies: List[StrategyMetrics] = []
    compiled_programs: Dict[str, Dict[str, Dict[str, Any]]] = {}
    raw_test_results: Dict[str, List[Dict[str, Any]]] = {}

    for opt in optimizers:
        # Fresh executor per strategy ensures identical RNG/provider state isolation
        executor = HanzoProgramExecutor(seed=seed, max_concurrency=max_concurrency)
        metrics, prog_map, test_res = await _run_strategy(opt, dataset, executor)
        strategies.append(metrics)
        compiled_programs[opt.name] = prog_map
        raw_test_results[opt.name] = [
            {
                "task_id": r.task_id,
                "category": r.category,
                "complexity": r.complexity,
                "passed": r.passed,
                "schema_valid": r.schema_valid,
                "injection_defended": r.injection_defended,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "total_tokens": r.total_tokens,
                "cost_usd": round(r.cost_usd, 6),
                "latency_ms": r.latency_ms,
                "workflow_steps_executed": r.workflow_steps_executed,
                "error": r.error,
            }
            for r in test_res
        ]

    ablation_executor = HanzoProgramExecutor(seed=seed, max_concurrency=max_concurrency)
    instr_metrics = next(s for s in strategies if s.strategy == "instruction_search")
    ablations = await _run_ablations(
        dataset,
        ablation_executor,
        instr_metrics,
        compiled_programs["instruction_search"],
    )

    report = BenchmarkReport(
        seed=seed,
        dataset_sizes={
            "train": len(dataset.train),
            "dev": len(dataset.dev),
            "test": len(dataset.test),
            "total": len(dataset.all_tasks()),
        },
        strategies=strategies,
        compiled_programs=compiled_programs,
        ablations=ablations,
        raw_test_results=raw_test_results,
    )

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "raw_results.json").write_text(
            json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        (out_path / "score_table.md").write_text(
            report.to_markdown(), encoding="utf-8"
        )

    return report
