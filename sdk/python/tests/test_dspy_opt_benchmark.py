"""Comprehensive tests for the HIP-1348 DSPy-style prompt and workflow optimization benchmark."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest

import hanzo_agents.agent_ai as agent_ai_module
from benchmarks.dspy_opt import (
    CATEGORIES,
    BenchmarkDataset,
    BootstrapFewShotOptimizer,
    HanzoProgramExecutor,
    InstructionSearchOptimizer,
    ParetoCostAwareOptimizer,
    ZeroShotBaselineOptimizer,
    assert_no_split_leakage,
    build_dataset,
    evaluate_task_output,
    run_benchmark,
    validate_security_boundary,
)
from benchmarks.dspy_opt.optimizers import CANDIDATE_INSTRUCTIONS, ParetoBudget
from hanzo_agents.llm_adapters import (
    apply_provider_patches,
    filter_none_values,
    get_provider_from_model,
)
from hanzo_agents.types import AIConfig


# --------------------------------------------------------------------------- #
# 1. Determinism                                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_benchmark_determinism_identical_seed():
    """Identical seed must yield bit-for-bit identical metrics, costs, and latencies."""
    report_a = await run_benchmark(seed=42, max_concurrency=4)
    report_b = await run_benchmark(seed=42, max_concurrency=8)

    assert report_a.to_dict() == report_b.to_dict()
    assert report_a.to_markdown() == report_b.to_markdown()


def test_dataset_seed_sensitivity():
    """Distinct seeds produce deterministic datasets with different sampled complex impacts."""
    ds_a = build_dataset(seed=42)
    ds_b = build_dataset(seed=99)

    assert [t.task_id for t in ds_a.test] == [t.task_id for t in ds_b.test]
    impacts_a = [
        t.expected["impact_usd"]
        for t in ds_a.test
        if t.category == "json_extraction" and t.complexity == "complex"
    ]
    impacts_b = [
        t.expected["impact_usd"]
        for t in ds_b.test
        if t.category == "json_extraction" and t.complexity == "complex"
    ]
    assert impacts_a != impacts_b


# --------------------------------------------------------------------------- #
# 2. Train / Dev / Held-Out Split Leakage Prevention                          #
# --------------------------------------------------------------------------- #


def test_split_leakage_zero_overlap():
    """Train, dev, and held-out test splits must have zero ID or prompt overlap."""
    ds = build_dataset(seed=42)
    assert len(ds.train) == 32
    assert len(ds.dev) == 24
    assert len(ds.test) == 48
    assert_no_split_leakage(ds)

    train_ids = {t.task_id for t in ds.train}
    dev_ids = {t.task_id for t in ds.dev}
    test_ids = {t.task_id for t in ds.test}
    assert train_ids.isdisjoint(dev_ids)
    assert train_ids.isdisjoint(test_ids)
    assert dev_ids.isdisjoint(test_ids)


def test_split_leakage_detector_catches_id_and_content_overlap():
    """assert_no_split_leakage must raise AssertionError on injected ID or content leakage."""
    ds = build_dataset(seed=42)

    # Inject task_id overlap between train and test
    leaked_id_task = replace(ds.test[0], task_id=ds.train[0].task_id)
    bad_ds_id = BenchmarkDataset(
        seed=ds.seed,
        train=ds.train,
        dev=ds.dev,
        test=(leaked_id_task,) + ds.test[1:],
    )
    with pytest.raises(AssertionError, match="overlapping task IDs"):
        assert_no_split_leakage(bad_ds_id)

    # Inject content fingerprint overlap between dev and test
    leaked_fp_task = replace(
        ds.test[0], content_fingerprint=ds.dev[0].content_fingerprint
    )
    bad_ds_fp = BenchmarkDataset(
        seed=ds.seed,
        train=ds.train,
        dev=ds.dev,
        test=(leaked_fp_task,) + ds.test[1:],
    )
    with pytest.raises(AssertionError, match="overlapping prompt fingerprints"):
        assert_no_split_leakage(bad_ds_fp)


@pytest.mark.asyncio
async def test_bootstrap_demonstrations_never_include_dev_or_test_ids():
    """Few-shot demonstrations selected during optimization must originate strictly from train."""
    ds = build_dataset(seed=42)
    executor = HanzoProgramExecutor(seed=42)
    opt = BootstrapFewShotOptimizer(max_demos=5)

    train_ids = {t.task_id for t in ds.train}
    forbidden_ids = {t.task_id for t in ds.dev} | {t.task_id for t in ds.test}

    for cat in CATEGORIES:
        prog = await opt.compile(
            category=cat,
            train_tasks=ds.by_category("train", cat),
            dev_tasks=ds.by_category("dev", cat),
            executor=executor,
        )
        demo_ids = {d.task_id for d in prog.demonstrations}
        assert len(demo_ids) == 5
        assert demo_ids.issubset(train_ids)
        assert demo_ids.isdisjoint(forbidden_ids)


# --------------------------------------------------------------------------- #
# 3. Concurrency Safety                                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_concurrency_safety_parallel_optimization_and_evaluation():
    """Concurrent asyncio.gather runs must stay isolated and restore AgentAI.llm cleanly."""
    original_llm = agent_ai_module.llm
    ds = build_dataset(seed=42)

    async def _run_isolated(seed: int):
        ex = HanzoProgramExecutor(seed=seed, max_concurrency=6)
        opt = InstructionSearchOptimizer(enforce_security_guard=True)
        progs = await asyncio.gather(
            *[
                opt.compile(
                    cat,
                    ds.by_category("train", cat),
                    ds.by_category("dev", cat),
                    ex,
                )
                for cat in CATEGORIES
            ]
        )
        evals = await asyncio.gather(
            *[
                ex.evaluate_program(prog, list(ds.by_category("test", cat)))
                for prog, cat in zip(progs, CATEGORIES)
            ]
        )
        return [r.passed for group in evals for r in group]

    results_parallel = await asyncio.gather(
        _run_isolated(42),
        _run_isolated(42),
        _run_isolated(77),
    )

    assert results_parallel[0] == results_parallel[1]
    assert all(results_parallel[0])
    assert agent_ai_module.llm is original_llm


# --------------------------------------------------------------------------- #
# 4. Security / Prompt-Injection Preservation                                 #
# --------------------------------------------------------------------------- #


def test_security_boundary_validator_rejects_weakened_instructions():
    """validate_security_boundary must reject unsafe or partially hardened candidates."""
    for cand_id, instr in CANDIDATE_INSTRUCTIONS["injection_defense"]:
        ok, reason = validate_security_boundary("injection_defense", instr)
        if cand_id == "delimiter_hardened_verified":
            assert ok is True
            assert reason is None
        else:
            assert ok is False
            assert reason is not None


@pytest.mark.asyncio
async def test_optimizers_reject_unsafe_injection_candidates():
    """Guarded InstructionSearch and ParetoCostAware must reject unsafe candidates and defend 100%."""
    ds = build_dataset(seed=42)
    ex = HanzoProgramExecutor(seed=42)

    for opt in (
        InstructionSearchOptimizer(enforce_security_guard=True),
        ParetoCostAwareOptimizer(enforce_security_guard=True),
    ):
        prog = await opt.compile(
            category="injection_defense",
            train_tasks=ds.by_category("train", "injection_defense"),
            dev_tasks=ds.by_category("dev", "injection_defense"),
            executor=ex,
        )
        rejected = prog.metadata.get("rejected_unsafe_candidates", [])
        rejected_ids = {r["candidate_id"] for r in rejected}
        assert {"baseline_unsafe", "terse_compressed_unsafe", "generic_boundary"}.issubset(
            rejected_ids
        )
        assert prog.metadata["candidate_id"] == "delimiter_hardened_verified"

        test_res = await ex.evaluate_program(
            prog, list(ds.by_category("test", "injection_defense"))
        )
        assert all(r.injection_defended is True for r in test_res)
        assert all(r.passed is True for r in test_res)


# --------------------------------------------------------------------------- #
# 5. Positive Gains & Negative Regimes Verification                           #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_positive_and_negative_regimes_measured(tmp_path):
    """End-to-end report verifies both positive gains and documented negative regimes."""
    report = await run_benchmark(seed=42, output_dir=tmp_path)
    by_name = {s.strategy: s for s in report.strategies}

    base = by_name["zero_shot_baseline"]
    few_shot = by_name["bootstrap_few_shot"]
    instr = by_name["instruction_search"]
    pareto = by_name["pareto_cost_aware"]

    # Positive gains on full held-out suite
    assert instr.test_pass_rate_pct > base.test_pass_rate_pct + 40.0
    assert pareto.test_pass_rate_pct >= 90.0
    assert pareto.injection_defense_rate_pct == 100.0
    assert base.injection_defense_rate_pct == 0.0
    assert pareto.cost_per_passing_task_usd is not None
    assert instr.cost_per_passing_task_usd is not None
    assert pareto.cost_per_passing_task_usd < instr.cost_per_passing_task_usd
    assert pareto.cost_per_passing_task_usd < few_shot.cost_per_passing_task_usd

    # Negative Regime 1: Over-prompting bloat on simple tasks (>+140% tokens, 0% accuracy gain)
    assert base.simple_pass_rate_pct == 100.0
    assert few_shot.simple_pass_rate_pct == 100.0
    simple_token_overhead = (
        (few_shot.simple_total_tokens - base.simple_total_tokens)
        / base.simple_total_tokens
    )
    assert simple_token_overhead >= 1.40

    # Negative Regime 2: Small-dev overfitting gap
    small_dev = report.ablations["small_dev_overfitting"]
    assert (
        small_dev["tiny_dev"]["dev_vs_test_gap_pp"]
        > small_dev["full_dev"]["dev_vs_test_gap_pp"] + 10.0
    )

    # Negative Regime 3: Security boundary erosion without guardrail
    sec_abl = report.ablations["unguarded_compression_security"]
    assert sec_abl["guarded"]["injection_defense_rate_pct"] == 100.0
    assert (
        sec_abl["unguarded_compressed"]["injection_defense_rate_pct"]
        < sec_abl["guarded"]["injection_defense_rate_pct"]
    )

    # Artifacts written cleanly and valid JSON
    raw_json = json.loads((tmp_path / "raw_results.json").read_text(encoding="utf-8"))
    assert raw_json["seed"] == 42
    assert len(raw_json["strategies"]) == 4
    md = (tmp_path / "score_table.md").read_text(encoding="utf-8")
    assert "pareto_cost_aware" in md
    assert "Over-Prompting Bloat" in md


# --------------------------------------------------------------------------- #
# 6. Regression Tests Against Hanzo SDK Adapters & Workflow Verifier          #
# --------------------------------------------------------------------------- #


def test_hanzo_sdk_llm_adapters_and_aiconfig_regression():
    """Verify Hanzo SDK AIConfig and llm_adapters behave identically to contract."""
    cfg = AIConfig(
        model="openai/zen-coder-480b",
        temperature=0.0,
        max_tokens=384,
        top_p=None,
    )
    params = cfg.get_llm_params()
    assert params["model"] == "openai/zen-coder-480b"
    assert params["max_completion_tokens"] == 384
    assert "max_tokens" not in params
    assert "top_p" not in params

    raw_params = {"model": "openai/gpt-4o", "max_tokens": 512, "top_p": None}
    patched = filter_none_values(apply_provider_patches(raw_params, "openai/gpt-4o"))
    assert patched == {"model": "openai/gpt-4o", "max_completion_tokens": 512}
    assert get_provider_from_model("anthropic/claude-3.5-sonnet") == "anthropic"


@pytest.mark.asyncio
async def test_workflow_synthesis_executes_agent_workflow_steps_and_rejects_cycles():
    """Workflow synthesis must track multi-step DAGs via AgentWorkflow and reject cyclic/over-budget DAGs."""
    ds = build_dataset(seed=42)
    ex = HanzoProgramExecutor(seed=42)
    opt = InstructionSearchOptimizer(enforce_security_guard=True)
    prog = await opt.compile(
        "workflow_synthesis",
        ds.by_category("train", "workflow_synthesis"),
        ds.by_category("dev", "workflow_synthesis"),
        ex,
    )

    complex_wf_task = next(
        t
        for t in ds.by_category("test", "workflow_synthesis")
        if t.complexity == "complex"
    )
    res = await ex.execute_task(prog, complex_wf_task)
    assert res.passed is True
    # 1 synthesis call + 4 DAG steps tracked through AgentWorkflow.execute_with_tracking
    assert res.workflow_steps_executed == 5

    # Verify executable DAG verifier rejects cycles
    cyclic_dag = {
        "workflow_name": complex_wf_task.constraints["workflow_name"],
        "has_checkpoint_gate": True,
        "steps": [
            {
                "step_id": "s1",
                "tool": "cloud_billing_check",
                "depends_on": ["s2"],
                "est_latency_ms": 100,
                "est_cost_millicents": 10,
            },
            {
                "step_id": "s2",
                "tool": "kai_policy_gate",
                "depends_on": ["s1"],
                "est_latency_ms": 100,
                "est_cost_millicents": 10,
            },
        ],
    }
    ok, _, err = evaluate_task_output(complex_wf_task, cyclic_dag)
    assert ok is False
    assert err is not None and "cycle" in err
