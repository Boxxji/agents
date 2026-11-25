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


def test_workflow_precedence_requires_transitive_depends_on_edges():
    """Concurrent ungated nodes (depends_on=[]) must fail precedence even if listed in order."""
    ds = build_dataset(seed=42)
    complex_wf_task = next(
        t
        for t in ds.by_category("test", "workflow_synthesis")
        if t.complexity == "complex"
        and any(
            pair == ["kai_policy_gate", "visor_sandbox_exec"]
            or tuple(pair) == ("kai_policy_gate", "visor_sandbox_exec")
            for pair in t.constraints.get("precedence", [])
        )
    )
    req_tools = list(complex_wf_task.constraints["required_tools"])

    # Put kai_policy_gate first in steps list, but leave all depends_on=[] (concurrent/ungated)
    ordered_tools = ["kai_policy_gate"] + [t for t in req_tools if t != "kai_policy_gate"]
    ungated_dag = {
        "workflow_name": complex_wf_task.constraints["workflow_name"],
        "has_checkpoint_gate": True,
        "steps": [
            {
                "step_id": f"s{i + 1}",
                "tool": tool,
                "depends_on": [],
                "est_latency_ms": 50,
                "est_cost_millicents": 5,
            }
            for i, tool in enumerate(ordered_tools)
        ],
    }
    ok, _, err = evaluate_task_output(complex_wf_task, ungated_dag)
    assert ok is False
    assert err is not None and "precedence violated" in err

    # Verify transitive chain (s1 -> s2 -> s3 -> ...) satisfies all precedence pairs
    # Build an explicit valid linear chain respecting every precedence constraint
    precedence = [tuple(p) for p in complex_wf_task.constraints.get("precedence", [])]
    # Topological sort tools using precedence constraints
    remaining = list(req_tools)
    chain: list[str] = []
    while remaining:
        progressed = False
        for tool in list(remaining):
            prereqs = [b for b, a in precedence if a == tool and b in remaining]
            if not prereqs:
                chain.append(tool)
                remaining.remove(tool)
                progressed = True
        assert progressed, "Test constraints should be acyclic"

    transitive_dag = {
        "workflow_name": complex_wf_task.constraints["workflow_name"],
        "has_checkpoint_gate": True,
        "steps": [
            {
                "step_id": f"s{i + 1}",
                "tool": tool,
                "depends_on": [] if i == 0 else [f"s{i}"],
                "est_latency_ms": 50,
                "est_cost_millicents": 5,
            }
            for i, tool in enumerate(chain)
        ],
    }
    ok_valid, _, err_valid = evaluate_task_output(complex_wf_task, transitive_dag)
    assert ok_valid is True, f"Expected valid transitive chain to pass, got: {err_valid}"


@pytest.mark.asyncio
async def test_overlapping_executors_preserve_and_restore_global_llm_hook():
    """A-acquire / B-acquire / A-release / B-release across two executors must not leak or prematurely restore."""
    original_llm = agent_ai_module.llm
    exec_a = HanzoProgramExecutor(seed=11)
    exec_b = HanzoProgramExecutor(seed=22)

    await exec_a._acquire_llm_hook()
    proxy_after_a = agent_ai_module.llm
    assert proxy_after_a is not original_llm

    await exec_b._acquire_llm_hook()
    assert agent_ai_module.llm is proxy_after_a

    # Releasing A while B is still active must keep the proxy installed
    await exec_a._release_llm_hook()
    assert agent_ai_module.llm is proxy_after_a

    # Releasing B must restore the exact original LLM object (not a stale proxy)
    await exec_b._release_llm_hook()
    assert agent_ai_module.llm is original_llm


@pytest.mark.asyncio
async def test_pareto_optimizer_raises_budget_infeasible_error_on_tight_guarded_cap():
    """Guarded ParetoCostAwareOptimizer must raise BudgetInfeasibleError when no candidate fits the hard cap."""
    from benchmarks.dspy_opt import BudgetInfeasibleError

    ds = build_dataset(seed=42)
    ex = HanzoProgramExecutor(seed=42)
    tight_opt = ParetoCostAwareOptimizer(
        budget=ParetoBudget(
            max_avg_tokens_per_task=615,
            max_cost_per_task_usd=0.0020,
            max_accuracy_drop_from_best=0.0,
        ),
        enforce_security_guard=True,
        demo_counts=(0,),
    )

    with pytest.raises(BudgetInfeasibleError) as exc_info:
        await tight_opt.compile(
            category="injection_defense",
            train_tasks=ds.by_category("train", "injection_defense"),
            dev_tasks=ds.by_category("dev", "injection_defense"),
            executor=ex,
        )

    err = exc_info.value
    assert err.category == "injection_defense"
    assert err.min_avg_tokens is not None and err.min_avg_tokens > 615
    assert len(err.rejected_unsafe_candidates) >= 3

    # Also verify full benchmark runner records the guarded tight-cap infeasibility cleanly
    report = await run_benchmark(seed=42)
    sec_abl = report.ablations["unguarded_compression_security"]
    assert sec_abl["guarded_tight_cap"]["compiled"] is False
    assert sec_abl["guarded_tight_cap"]["infeasible"] is True
    assert sec_abl["guarded_tight_cap"]["min_avg_tokens"] > 615


@pytest.mark.asyncio
async def test_concurrent_non_benchmark_call_delegates_to_original_backend(
    monkeypatch: pytest.MonkeyPatch,
):
    """
    While a benchmark call holds the module-level `agent_ai_module.llm` proxy,
    an ordinary concurrent `AgentAI.ai()` call with no benchmark ContextVar bound
    must delegate seamlessly to the saved original LLM backend instead of raising.
    """
    from types import SimpleNamespace
    from benchmarks.dspy_opt.runtime import _BenchmarkAgent

    backend_calls: list[dict] = []
    bench_entered = asyncio.Event()
    allow_bench_finish = asyncio.Event()

    class _SentinelBackend:
        class utils:
            @staticmethod
            def token_counter(model: str, messages: list[dict]) -> int:
                return 17

            @staticmethod
            def trim_messages(messages: list[dict], model: str, max_tokens: int) -> list[dict]:
                return messages

        custom_marker = "original-backend-marker"

        async def acompletion(self, **kwargs):
            backend_calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="ordinary-backend-response")
                    )
                ]
            )

    sentinel_backend = _SentinelBackend()
    monkeypatch.setattr(agent_ai_module, "llm", sentinel_backend)

    ds = build_dataset(seed=42)
    task = ds.by_category("dev", "json_extraction")[0]
    ex = HanzoProgramExecutor(seed=42)
    orig_provider_acompletion = ex.provider.acompletion

    async def _slow_provider_acompletion(**kwargs):
        bench_entered.set()
        await allow_bench_finish.wait()
        return await orig_provider_acompletion(**kwargs)

    monkeypatch.setattr(ex.provider, "acompletion", _slow_provider_acompletion)
    program = await ZeroShotBaselineOptimizer().compile(
        "json_extraction",
        ds.by_category("train", "json_extraction"),
        ds.by_category("dev", "json_extraction"),
        ex,
    )

    async def _run_benchmark_call():
        return await ex.execute_task(program, task)

    async def _run_ordinary_call():
        await bench_entered.wait()
        try:
            # Confirm proxy is installed while benchmark call is in flight
            assert agent_ai_module.llm is not sentinel_backend
            assert agent_ai_module.llm.custom_marker == "original-backend-marker"
            ordinary_agent = _BenchmarkAgent(
                ai_config=AIConfig(model="openai/gpt-4o-mini", enable_rate_limit_retry=False)
            )
            ordinary_ai = agent_ai_module.AgentAI(ordinary_agent)
            return await ordinary_ai.ai(system="You are helpful.", user="Ping outside benchmark")
        finally:
            allow_bench_finish.set()

    bench_res, ordinary_resp = await asyncio.gather(
        _run_benchmark_call(), _run_ordinary_call()
    )

    assert bench_res.task_id == task.task_id
    assert len(backend_calls) == 1
    assert getattr(ordinary_resp, "text", ordinary_resp) == "ordinary-backend-response"
    # Original backend restored cleanly once benchmark call finishes
    assert agent_ai_module.llm is sentinel_backend


@pytest.mark.asyncio
async def test_bootstrap_few_shot_max_demos_zero_returns_zero_demos():
    """BootstrapFewShotOptimizer(max_demos=0) on a simple-only passing slice must return 0 demos."""
    ds = build_dataset(seed=42)
    ex = HanzoProgramExecutor(seed=42)
    simple_train = [
        t
        for t in ds.by_category("train", "json_extraction")
        if t.complexity == "simple"
    ]
    assert simple_train  # Sanity check: simple tasks exist and pass under zero-shot baseline

    baseline = await ZeroShotBaselineOptimizer().compile(
        "json_extraction", simple_train, simple_train, ex
    )
    baseline_results = await ex.evaluate_program(baseline, simple_train)
    assert any(r.passed for r in baseline_results)

    opt = BootstrapFewShotOptimizer(max_demos=0)
    program = await opt.compile(
        category="json_extraction",
        train_tasks=simple_train,
        dev_tasks=simple_train,
        executor=ex,
    )
    assert len(program.demonstrations) == 0
    assert program.metadata["total_demos"] == 0
    assert program.metadata["bootstrapped_traces"] == 0
    assert program.metadata["labeled_fallback_traces"] == 0


