"""Async benchmark runner and Markdown score-table generator for Lost-in-the-Middle."""

from __future__ import annotations

import asyncio
import json
import platform
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .dataset import (
    BenchmarkCase,
    generate_negative_regime_suite,
    generate_position_sweep_suite,
    load_jsonl_cases,
)
from .evaluator import (
    BudgetTruncationEvaluator,
    CalibratedTACL2024Evaluator,
    ContextEvaluator,
    DistractorInterferenceEvaluator,
    LiveOpenAICompatibleEvaluator,
    MemoryBackedRetriever,
)
from .metrics import aggregate_evaluations, paired_bootstrap_ci
from .packers import POLICY_NAMES, pack_context


@dataclass
class BenchmarkConfig:
    """Configuration for a reproducible Lost-in-the-Middle benchmark run."""

    mode: str = "all"  # budget_truncation | distractor_interference | calibrated_tacl2024 | live_llm | all
    truncation_mode: str = "hanzo_middle_trim"  # hanzo_middle_trim | tail_clip
    doc_counts: Tuple[int, ...] = (5, 10, 20)
    budgets: Tuple[int, ...] = (512, 1024, 2048, 4096)
    cases_per_position: int = 2
    negative_cases_per_regime: int = 10
    policies: Tuple[str, ...] = POLICY_NAMES
    seed: int = 42
    concurrency: int = 8
    model: str = "zen/zen-coder"
    calibrated_profile: str = "gpt-3.5-turbo-16k"
    external_jsonl: Optional[Path] = None
    external_limit: Optional[int] = None
    include_raw_records: bool = True
    api_key: Optional[str] = None
    api_base: Optional[str] = None


def _build_mode_runs(
    config: BenchmarkConfig,
) -> List[Tuple[str, str, ContextEvaluator]]:
    """Return list of (run_label, truncation_mode, evaluator) tuples."""
    if config.mode == "all":
        return [
            (
                "budget_truncation__hanzo_middle_trim",
                "hanzo_middle_trim",
                BudgetTruncationEvaluator(),
            ),
            (
                "budget_truncation__tail_clip",
                "tail_clip",
                BudgetTruncationEvaluator(),
            ),
            (
                "distractor_interference__hanzo_middle_trim",
                "hanzo_middle_trim",
                DistractorInterferenceEvaluator(),
            ),
            (
                f"calibrated_tacl2024__{config.calibrated_profile}",
                "hanzo_middle_trim",
                CalibratedTACL2024Evaluator(profile=config.calibrated_profile),
            ),
        ]
    if config.mode == "budget_truncation":
        return [
            (
                f"budget_truncation__{config.truncation_mode}",
                config.truncation_mode,
                BudgetTruncationEvaluator(),
            )
        ]
    if config.mode == "distractor_interference":
        return [
            (
                f"distractor_interference__{config.truncation_mode}",
                config.truncation_mode,
                DistractorInterferenceEvaluator(),
            )
        ]
    if config.mode == "calibrated_tacl2024":
        return [
            (
                f"calibrated_tacl2024__{config.calibrated_profile}",
                config.truncation_mode,
                CalibratedTACL2024Evaluator(profile=config.calibrated_profile),
            )
        ]
    if config.mode == "live_llm":
        return [
            (
                f"live_llm__{config.model}",
                config.truncation_mode,
                LiveOpenAICompatibleEvaluator(
                    model=config.model,
                    api_key=config.api_key,
                    api_base=config.api_base,
                    seed=config.seed,
                ),
            )
        ]
    raise ValueError(f"Unsupported mode: {config.mode!r}")


async def _evaluate_single_case_all_policies(
    *,
    case: BenchmarkCase,
    retriever: MemoryBackedRetriever,
    evaluator: ContextEvaluator,
    policies: Sequence[str],
    budgets: Sequence[int],
    truncation_mode: str,
    model: str,
    semaphore: asyncio.Semaphore,
) -> List[Dict[str, Any]]:
    """Retrieve case docs via ScopedMemoryClient and evaluate every (budget, policy) pair."""
    async with semaphore:
        retrieved_docs = await retriever.ingest_and_retrieve(case)
        records: List[Dict[str, Any]] = []
        for budget in budgets:
            for policy in policies:
                packed = pack_context(
                    case,
                    policy=policy,
                    budget_tokens=budget,
                    truncation_mode=truncation_mode,
                    model=model,
                    documents_override=retrieved_docs,
                )
                ev = await evaluator.evaluate(case, packed)
                records.append(ev.to_dict())
        return records


def _summarize_run(
    *,
    run_label: str,
    truncation_mode: str,
    is_calibrated: bool,
    records: List[Dict[str, Any]],
    policies: Sequence[str],
    budgets: Sequence[int],
    doc_counts: Sequence[int],
    seed: int,
) -> Dict[str, Any]:
    """Build structured aggregates, position curves, regime tables, and paired bootstrap CIs."""
    standard_recs = [r for r in records if r["regime"] in ("standard", "external_jsonl")]
    regimes = sorted({r["regime"] for r in records})

    # 1. Overall summary on standard position sweep
    overall_by_policy: Dict[str, Any] = {}
    for policy in policies:
        p_recs = [r for r in standard_recs if r["policy"] == policy]
        overall_by_policy[policy] = aggregate_evaluations(policy, p_recs).to_dict()

    # 2. Per-budget summary on standard position sweep
    by_budget: Dict[str, Dict[str, Any]] = {}
    for budget in budgets:
        b_key = str(budget)
        by_budget[b_key] = {}
        for policy in policies:
            p_recs = [
                r
                for r in standard_recs
                if r["policy"] == policy and int(r["budget_tokens"]) == budget
            ]
            by_budget[b_key][policy] = aggregate_evaluations(policy, p_recs).to_dict()

    # 3. Per-N summary and full gold-position curve (1..N)
    by_doc_count: Dict[str, Dict[str, Any]] = {}
    position_curves: Dict[str, Dict[str, Dict[str, float]]] = {}
    for n_docs in doc_counts:
        n_key = str(n_docs)
        by_doc_count[n_key] = {}
        position_curves[n_key] = {}
        for policy in policies:
            p_recs = [
                r
                for r in standard_recs
                if r["policy"] == policy and int(r["num_docs"]) == n_docs
            ]
            by_doc_count[n_key][policy] = aggregate_evaluations(policy, p_recs).to_dict()
            pos_map: Dict[str, float] = {}
            for pos in range(1, n_docs + 1):
                pos_recs = [r for r in p_recs if int(r["gold_position"]) == pos]
                if pos_recs:
                    pos_map[str(pos)] = round(
                        sum(float(r["correctness"]) for r in pos_recs) / len(pos_recs),
                        4,
                    )
            position_curves[n_key][policy] = pos_map

    # 4. Per-regime summary (including small_context, misranked_retriever, duplicate_heavy, prompt_injection)
    by_regime: Dict[str, Dict[str, Any]] = {}
    for regime in regimes:
        by_regime[regime] = {}
        for policy in policies:
            p_recs = [
                r
                for r in records
                if r["regime"] == regime and r["policy"] == policy
            ]
            by_regime[regime][policy] = aggregate_evaluations(policy, p_recs).to_dict()

    # 5. Paired bootstrap 95% CIs on standard sweep (aligned by case_id, budget_tokens)
    def _aligned_scores(
        subset: Sequence[Dict[str, Any]], pol_a: str, pol_b: str
    ) -> Tuple[List[float], List[float]]:
        map_a = {
            (r["case_id"], int(r["budget_tokens"])): float(r["correctness"])
            for r in subset
            if r["policy"] == pol_a
        }
        map_b = {
            (r["case_id"], int(r["budget_tokens"])): float(r["correctness"])
            for r in subset
            if r["policy"] == pol_b
        }
        keys = sorted(set(map_a) & set(map_b))
        return [map_a[k] for k in keys], [map_b[k] for k in keys]

    paired_comparisons: Dict[str, Any] = {}
    pairs = (
        ("dedup_budgeted_bookend", "chronological"),
        ("dedup_budgeted_bookend", "score_descending"),
        ("dedup_budgeted_bookend", "bookend_primacy_recency"),
        ("bookend_primacy_recency", "chronological"),
        ("dedup_budgeted_bookend", "middle_buried_adversarial"),
    )
    for pol_a, pol_b in pairs:
        if pol_a in policies and pol_b in policies:
            sa, sb = _aligned_scores(standard_recs, pol_a, pol_b)
            paired_comparisons[f"{pol_a}_vs_{pol_b}"] = paired_bootstrap_ci(
                sa, sb, n_resamples=1000, seed=seed
            ).to_dict()

    # Also compute paired CIs for negative regimes (honestly quantifying regressions!)
    negative_regime_cis: Dict[str, Any] = {}
    for neg_regime in ("small_context", "misranked_retriever", "duplicate_heavy", "prompt_injection"):
        sub = [r for r in records if r["regime"] == neg_regime]
        if not sub:
            continue
        negative_regime_cis[neg_regime] = {}
        for pol_a, pol_b in (
            ("dedup_budgeted_bookend", "chronological"),
            ("dedup_budgeted_bookend", "bookend_primacy_recency"),
        ):
            if pol_a in policies and pol_b in policies:
                sa, sb = _aligned_scores(sub, pol_a, pol_b)
                negative_regime_cis[neg_regime][f"{pol_a}_vs_{pol_b}"] = paired_bootstrap_ci(
                    sa, sb, n_resamples=1000, seed=seed
                ).to_dict()

    return {
        "run_label": run_label,
        "truncation_mode": truncation_mode,
        "is_calibrated_stress_test": is_calibrated,
        "disclaimer": (
            "CALIBRATED OFFLINE SENSITIVITY STRESS-TEST (Liu et al. TACL 2024 published curve) — "
            "NOT a live Hanzo model measurement."
            if is_calibrated
            else "Physical / non-circular evaluation executed over Hanzo SDK MemoryClient and AIConfig."
        ),
        "total_evaluations": len(records),
        "overall_standard": overall_by_policy,
        "by_budget": by_budget,
        "by_doc_count": by_doc_count,
        "position_curves": position_curves,
        "by_regime": by_regime,
        "paired_bootstrap_ci_standard": paired_comparisons,
        "paired_bootstrap_ci_regimes": negative_regime_cis,
    }


async def run_benchmark(config: BenchmarkConfig) -> Dict[str, Any]:
    """Execute the full Lost-in-the-Middle benchmark suite asynchronously."""
    if config.external_jsonl is not None:
        cases = load_jsonl_cases(config.external_jsonl, limit=config.external_limit)
    else:
        cases = generate_position_sweep_suite(
            doc_counts=config.doc_counts,
            cases_per_position=config.cases_per_position,
            seed=config.seed,
        )
        if config.negative_cases_per_regime > 0:
            cases.extend(
                generate_negative_regime_suite(
                    cases_per_regime=config.negative_cases_per_regime,
                    seed=config.seed + 1305,
                )
            )

    mode_runs = _build_mode_runs(config)
    semaphore = asyncio.Semaphore(max(1, config.concurrency))
    summaries: Dict[str, Any] = {}
    raw_by_run: Dict[str, List[Dict[str, Any]]] = {}

    for run_label, trunc_mode, evaluator in mode_runs:
        retriever = MemoryBackedRetriever()
        tasks = [
            _evaluate_single_case_all_policies(
                case=case,
                retriever=retriever,
                evaluator=evaluator,
                policies=config.policies,
                budgets=config.budgets,
                truncation_mode=trunc_mode,
                model=config.model,
                semaphore=semaphore,
            )
            for case in cases
        ]
        nested = await asyncio.gather(*tasks)
        records = [rec for batch in nested for rec in batch]
        summaries[run_label] = _summarize_run(
            run_label=run_label,
            truncation_mode=trunc_mode,
            is_calibrated=evaluator.is_calibrated_stress_test,
            records=records,
            policies=config.policies,
            budgets=config.budgets,
            doc_counts=config.doc_counts,
            seed=config.seed,
        )
        if config.include_raw_records:
            raw_by_run[run_label] = records

    result: Dict[str, Any] = {
        "benchmark": "hip-1347-lost-in-the-middle",
        "reference_paper": "Liu et al. (TACL 2024), https://aclanthology.org/2024.tacl-1.9/",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "config": {
            "mode": config.mode,
            "truncation_mode": config.truncation_mode,
            "doc_counts": list(config.doc_counts),
            "budgets": list(config.budgets),
            "cases_per_position": config.cases_per_position,
            "negative_cases_per_regime": config.negative_cases_per_regime,
            "policies": list(config.policies),
            "seed": config.seed,
            "concurrency": config.concurrency,
            "model": config.model,
            "calibrated_profile": config.calibrated_profile,
            "total_cases": len(cases),
        },
        "runs": summaries,
    }
    if config.include_raw_records:
        result["raw_records"] = raw_by_run
    return result


def _fmt_pct(val: Optional[float]) -> str:
    if val is None:
        return "—"
    return f"{val * 100:.1f}%"


def render_score_table_markdown(results: Dict[str, Any]) -> str:
    """Render deterministic Markdown tables directly from measured benchmark results."""
    lines: List[str] = [
        "# HIP-1347 Lost-in-the-Middle Context Packing — Measured Score Tables",
        "",
        f"- **Generated (UTC)**: `{results['generated_at_utc']}`",
        f"- **Python / Platform**: `{results['environment']['python_version']}` / `{results['environment']['platform']}`",
        f"- **Seed**: `{results['config']['seed']}` | **Total Cases**: `{results['config']['total_cases']}` | "
        f"**Doc Counts (N)**: `{results['config']['doc_counts']}` | **Token Budgets**: `{results['config']['budgets']}`",
        "",
        "> **Scientific Integrity Notice**: Physical evaluation modes (`budget_truncation`, `distractor_interference`) "
        "contain **zero assumed position-attention curves** and measure real Hanzo SDK `AIConfig.trim_by_chars`, "
        "`tail_clip`, lexical interference, and prompt-injection delimiter isolation. "
        "`calibrated_tacl2024` is explicitly labeled as an offline sensitivity stress-test profile parameterized "
        "from Liu et al. (TACL 2024) published measurements and is **not** a live Hanzo model claim.",
        "",
    ]

    policies = results["config"]["policies"]
    budgets = results["config"]["budgets"]

    for run_label, run_data in results["runs"].items():
        tag = (
            "CALIBRATED STRESS-TEST (OFFLINE PROFILE)"
            if run_data["is_calibrated_stress_test"]
            else "PHYSICAL / NON-CIRCULAR"
        )
        lines.append(f"## Run: `{run_label}` [{tag}]")
        lines.append("")
        lines.append(f"*{run_data['disclaimer']}*")
        lines.append("")

        # Table 1: Overall Standard Sweep
        lines.append("### Overall Standard Position Sweep (`N ∈ {5, 10, 20}`, All Positions `1..N`, Budgets `512..4096`)")
        lines.append("")
        lines.append(
            "| Policy | Answer Correctness | Evidence Retention | Truncation Loss | Mean Tokens | "
            "Efficiency (Acc/1K tok) | Boundary Integrity | Primacy (P1) | Middle (N/2) | Recency (PN) | "
            "p50 Lat (ms) | p95 Lat (ms) |"
        )
        lines.append(
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        )
        overall = run_data["overall_standard"]
        for pol in policies:
            m = overall[pol]
            lines.append(
                f"| `{pol}` | {_fmt_pct(m['answer_correctness'])} | {_fmt_pct(m['evidence_retention_rate'])} | "
                f"{_fmt_pct(m['truncation_loss_rate'])} | {m['mean_tokens']:.1f} | "
                f"{m['token_efficiency_per_1k']:.3f} | {_fmt_pct(m['boundary_integrity_rate'])} | "
                f"{_fmt_pct(m['primacy_correctness'])} | {_fmt_pct(m['middle_correctness'])} | "
                f"{_fmt_pct(m['recency_correctness'])} | {m['p50_latency_ms']:.3f} | {m['p95_latency_ms']:.3f} |"
            )
        lines.append("")

        # Table 2: Per-Budget Breakdown
        lines.append("### Answer Correctness & Token Efficiency by Strict Token Budget")
        lines.append("")
        header = "| Policy | " + " | ".join(f"Budget {b}" for b in budgets) + " |"
        sep = "|---|" + "|".join("---:" for _ in budgets) + "|"
        lines.append(header)
        lines.append(sep)
        by_budget = run_data["by_budget"]
        for pol in policies:
            cells = []
            for b in budgets:
                bm = by_budget[str(b)][pol]
                cells.append(
                    f"{_fmt_pct(bm['answer_correctness'])} ({bm['mean_tokens']:.0f} tok, eff={bm['token_efficiency_per_1k']:.2f})"
                )
            lines.append(f"| `{pol}` | " + " | ".join(cells) + " |")
        lines.append("")

        # Table 3: Paired Bootstrap 95% CIs
        lines.append("### Paired Bootstrap 95% Confidence Intervals (Standard Sweep)")
        lines.append("")
        lines.append("| Comparison | Mean Delta (A - B) | 95% CI Low | 95% CI High | Pairs |")
        lines.append("|---|---:|---:|---:|---:|")
        for comp_name, ci in run_data["paired_bootstrap_ci_standard"].items():
            lines.append(
                f"| `{comp_name}` | {ci['mean_diff']:+.4f} | {ci['ci_low']:+.4f} | "
                f"{ci['ci_high']:+.4f} | {ci['n_pairs']} |"
            )
        lines.append("")

        # Table 4: Honest Negative & Security Regimes
        neg_regimes = [
            r
            for r in ("small_context", "misranked_retriever", "duplicate_heavy", "prompt_injection")
            if r in run_data["by_regime"]
        ]
        if neg_regimes:
            lines.append("### Negative & Adversarial Failure Regimes")
            lines.append("")
            lines.append(
                "| Regime | Policy | Correctness | Retention | Trunc Loss | Mean Tokens | Boundary Integrity |"
            )
            lines.append("|---|---|---:|---:|---:|---:|---:|")
            for reg in neg_regimes:
                for pol in policies:
                    rm = run_data["by_regime"][reg][pol]
                    lines.append(
                        f"| `{reg}` | `{pol}` | {_fmt_pct(rm['answer_correctness'])} | "
                        f"{_fmt_pct(rm['evidence_retention_rate'])} | {_fmt_pct(rm['truncation_loss_rate'])} | "
                        f"{rm['mean_tokens']:.1f} | {_fmt_pct(rm['boundary_integrity_rate'])} |"
                    )
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"
