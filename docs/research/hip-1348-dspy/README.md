# HIP-1348 Research Submission: Automatic Prompt & Workflow Optimization (DSPy)

**Issue:** `hanzoai/agents#79` — Automatic prompt and workflow optimization (DSPy)  
**Track:** HIP-1348 (Research Bounty)  
**Benchmark Package:** [`sdk/python/benchmarks/dspy_opt/`](../../../sdk/python/benchmarks/dspy_opt/)  
**Raw Artifacts:** [`raw_results.json`](./raw_results.json) | [`score_table.md`](./score_table.md)

---

## 1. Primary Reference, Summary & License Audit

- **Primary Repository:** [stanfordnlp/dspy](https://github.com/stanfordnlp/dspy)
- **Primary Papers:**
  - Khattab et al., *"DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines"*, ICLR 2024 (arXiv:2310.03714)
  - Opsahl-Ong et al., *"Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs"* (MIPROv2, EMNLP 2024, arXiv:2406.11695)
- **Summary:** DSPy frames LLM calls as typed declarative signatures composed into executable programs, separating program structure from prompt strings and model weights. Optimizers (teleprompters) such as `BootstrapFewShot` and `MIPRO` compile a program against an executable task metric using training/validation traces rather than manual prompt hand-tuning.
- **License & Dependency Audit:**
  - **Upstream DSPy License:** MIT License (`stanfordnlp/dspy/LICENSE`) — permissive and compatible with Hanzo's Apache-2.0 Python SDK.
  - **Dependency Surface Warning:** Installing upstream `dspy-ai` pulls a heavy transitive dependency graph (`litellm`, `optuna`, `diskcache`, `magicattr`, `datasets`, `numpy`, `scipy`, etc.), introducing supply-chain surface, cold-start latency, and a parallel LLM router (`dspy.LM` / `litellm`) that conflicts with Hanzo's canonical `Enso -> Zen` model routing and `AgentAI` / `AIConfig` interfaces.
  - **Clean-Room Implementation Status:** This submission implements a **100% clean-room Apache-2.0** DSPy-style optimization benchmark (`sdk/python/benchmarks/dspy_opt/`) using **zero copied DSPy source code** and **zero new third-party dependencies** (Python stdlib + existing `pydantic` and `hanzo_agents` SDK modules only). All benchmark tasks are synthetically generated with deterministic seeds (no external dataset license restrictions, no patent-encumbered components).

---

## 2. Reproducible Baseline & Hanzo SDK Harness

All strategies execute over the real Hanzo Python SDK interfaces rather than a standalone harness:

- **`hanzo_agents.types.AIConfig`** — parameter resolution (`get_llm_params`), token approximation (`avg_chars_per_token`), and context-window settings.
- **`hanzo_agents.llm_adapters`** — `apply_provider_patches`, `filter_none_values`, and `get_provider_from_model`.
- **`hanzo_agents.agent_ai.AgentAI`** — `ai(system=..., user=..., schema=...)` structured Pydantic schema injection, token accounting, and JSON validation.
- **`hanzo_agents.agent_workflow.AgentWorkflow`** — `execute_with_tracking(...)` wrapping both synthesis calls and multi-step `WorkflowDAG` step execution.

### Dataset Splits & Leakage Controls

- **Total Executable Tasks:** 104 deterministic tasks across 4 categories (`json_extraction`, `tool_routing`, `workflow_synthesis`, `injection_defense`):
  - **Train split:** 32 tasks (8/category) — trace bootstrapping & demonstration selection only.
  - **Dev split:** 24 tasks (6/category) — candidate instruction & Pareto selection only.
  - **Held-out Test split:** 48 tasks (12/category) — touched strictly once for final reporting.
- **Anti-Leakage Invariant:** Disjoint service/team entity pools per split plus automated `assert_no_split_leakage()` verifying zero task-ID overlap and zero SHA-256 prompt-fingerprint overlap across `(train, dev, test)`.

### Reproduction Command

```bash
cd sdk/python
python3 -m benchmarks.dspy_opt --seed 42 --output-dir ../../docs/research/hip-1348-dspy
```

> **Note on Published Numbers:** Published DSPy paper numbers belong to the original papers and are not Hanzo claims. All metrics below are measured directly from the Hanzo SDK benchmark harness using a deterministic offline Enso/Zen pricing and prompt-sensitivity simulator (`SimulatedEnsoZenProvider`) for zero-network CI reproducibility; Section 6 documents the live-model validation step required before production promotion.

---

## 3. Falsifiable Hypotheses & Expected Failure Modes

- **H1 (Executable Quality Lift):** Automated instruction search (`instruction_search`) and trace bootstrapping (`bootstrap_few_shot`) increase held-out executable pass rate by at least +40 percentage points over `zero_shot_baseline` across complex multi-constraint and structured extraction tasks.
- **H2 (Pareto Cost/Token Efficiency):** Jointly optimizing crisp verified instructions with budget-capped model-tier selection (`pareto_cost_aware`, routing across `openai/zen-mini-8b` and `openai/zen-coder-480b` within a 900-token / \$0.0045 per-task cap) reduces **Cost per Passing Task** by >5x relative to `instruction_search` and >10x relative to `bootstrap_few_shot` while staying within 2.5 pp of peak held-out pass rate.
- **H3 (Negative Regime — Over-Prompting Bloat on Simple Tasks):** On low-complexity tasks where `zero_shot_baseline` already achieves 100% pass rate, 5-shot `bootstrap_few_shot` inflates input tokens by >+140% and cost by >+120% with 0.00 pp pass-rate gain.
- **H4 (Negative Regime — Small-Dev Overfitting):** Reducing the dev split from 24 tasks to 4 tasks (`dev_size=4`, 1/category) allows brittle heuristic prompts to score 100% on dev while degrading held-out test pass rate by >=15 pp (large positive Dev-vs-Test overfitting gap).
- **H5 (Negative Regime — Security Erosion Under Naive Compression):** Disabling the untrusted-input boundary guard (`enforce_security_guard=False`) under an aggressive token cap causes the optimizer to drop delimiter-breakout protections to save tokens, reducing Prompt-Injection Defense Rate below 100%.

---

## 4. Bounded Implementation Scope

Located in `sdk/python/benchmarks/dspy_opt/` (no parallel agent framework introduced):

| File | Responsibility |
| --- | --- |
| `schemas.py` | Pydantic schemas (`IncidentExtraction`, `ToolRoutePlan`, `WorkflowStep`, `WorkflowDAG`) and compiled program dataclasses |
| `dataset.py` | Seeded train/dev/test generator, `assert_no_split_leakage()`, and executable ground-truth verifiers (including Kahn topological DAG cycle/precedence/budget verifier) |
| `runtime.py` | `HanzoProgramExecutor` bridging `AIConfig`, `llm_adapters`, `AgentAI`, `AgentWorkflow`, and context-isolated `SimulatedEnsoZenProvider` |
| `optimizers.py` | 4 strategies (`zero_shot_baseline`, `bootstrap_few_shot`, `instruction_search`, `pareto_cost_aware`) + `validate_security_boundary()` guard |
| `metrics.py` | Pass rate, Train/Dev-vs-Test overfitting gap, tokens, USD cost, cost/passing task, injection defense rate, p50/p95 latency, markdown exporter |
| `runner.py` & `__main__.py` | Async concurrent benchmark runner (`asyncio.gather`), negative-regime ablations, CLI outputting `raw_results.json` & `score_table.md` |

---

## 5. Correctness, Concurrency, Security & Regression Evidence

Verified via `sdk/python/tests/test_dspy_opt_benchmark.py` (**11/11 passed**) and the full `sdk/python/tests/` suite (`pytest --ignore=tests/test_web3_mpc_client.py`: **528 passed, 4 skipped, 0 failed**; optional `[web3]` extra excluded as `eth_account` is not installed):

- **Determinism:** Two independent runs with `seed=42` at different concurrency levels (`max_concurrency=4` vs `8`) produce bit-for-bit identical `raw_results.json` and `score_table.md`.
- **Leakage Prevention:** Automated tests verify 0 overlapping task IDs and 0 overlapping prompt fingerprints across train/dev/test, confirm `assert_no_split_leakage()` raises on injected leakage, and confirm `bootstrap_few_shot` demonstrations come strictly from `train`.
- **Concurrency Safety:** Parallel `asyncio.gather` compilation + evaluation across 3 concurrent isolated executors verifies zero cross-task state leakage via `contextvars` and exact restoration of `hanzo_agents.agent_ai.llm`.
- **Security Invariant Preservation:** Both `InstructionSearchOptimizer` and `ParetoCostAwareOptimizer` reject `baseline_unsafe`, `terse_compressed_unsafe`, and `generic_boundary` via `validate_security_boundary()`, maintaining **100.00%** held-out Prompt-Injection Defense Rate.
- **SDK Regressions:** Existing `AIConfig`, `llm_adapters`, `AgentAI`, and `AgentWorkflow` unit tests plus the full Python SDK suite pass with zero regressions.

---

## 6. Before/After Measurements, Negative Results & Canonical Integration Recommendation

### Primary Held-Out Results (`seed=42`, 48 Held-Out Tasks)

| Strategy | Held-Out Pass Rate (%) | Train-vs-Test Gap (pp) | Total Tokens | Est. Cost (USD) | Cost / Passing Task (USD) | Injection Defense (%) | p50 Latency (ms) | p95 Latency (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `zero_shot_baseline` | 37.50% (18/48) | +0.00 | 28,918 | \$0.090355 | \$0.005020 | 0.00% | 277.08 | 474.33 |
| `bootstrap_few_shot` | 87.50% (42/48) | +0.00 | 73,833 | \$0.202725 | \$0.004827 | 50.00% | 413.86 | 638.00 |
| `instruction_search` | 100.00% (48/48) | +0.00 | 32,459 | \$0.099305 | \$0.002069 | 100.00% | 288.56 | 490.73 |
| `pareto_cost_aware` | 97.92% (47/48) | +2.08 | 32,459 | \$0.013903 | \$0.000296 | 100.00% | 126.33 | 230.05 |

### Documented Negative Regimes & Limitations

1. **Over-Prompting Bloat on Simple Tasks (Confirmed H3):** On the held-out benign simple subset (18 tasks), `zero_shot_baseline` already achieves **100.00%** pass rate using **10,460 tokens** (\$0.032218). Adding 5-shot `bootstrap_few_shot` traces balloons usage to **26,462 tokens (+153.0%)** and **\$0.072222 (+124.2% cost)** for **+0.00 pp** accuracy gain. **Takeaway:** Never apply unconditional few-shot compilation globally; gate optimization behind complexity routing or baseline pass-rate thresholds.
2. **Small-Dev Overfitting (Confirmed H4):** With `dev_size=4` (1 task/category), `instruction_search` achieves **100.00% dev pass rate** by selecting brittle short rules (`brittle_overfit_short`) that collapse to **75.00% on held-out test** — a **+25.00 pp overfitting gap** versus **+0.00 pp** with `dev_size=24`. **Takeaway:** Require minimum stratified dev sets (>=20 tasks across complexity buckets) and fail CI promotion if `dev - test` gap exceeds tolerance.
3. **Security Boundary Erosion Without Guardrail (Confirmed H5):** Under a strict 615-token cap with `enforce_security_guard=False`, `pareto_cost_aware` drops `delimiter_hardened_verified` in favor of `generic_boundary`, reducing held-out Prompt-Injection Defense Rate from **100.00% -> 75.00%** (vulnerable to `</untrusted_report>` tag-breakout payloads). **Takeaway:** Security invariants must be hard constraints outside the cost objective, never soft penalty weights.
4. **Offline Simulator Limitation:** The CI harness uses `SimulatedEnsoZenProvider` for deterministic, zero-network verification. Before promoting any compiled prompt artifact to production, the identical `HanzoProgramExecutor` interface must be re-run against live Zen/Enso endpoints across at least 3 seeds and >=100 held-out production traces to measure real model variance.

### Canonical Hanzo Integration Recommendation (No Parallel Stack)

Do **not** import upstream `dspy-ai` into `hanzo_agents` core or build a parallel runtime. Instead, integrate prompt/workflow compilation directly as an **offline build/eval pass** over Hanzo's fixed canonical architecture:

1. **Zen (Generation):** Compile versioned prompt artifacts (`instruction` + optional `0..2` verified traces + Pydantic schema contract) offline against executable eval suites and store immutable compiled artifacts in `vfs -> Hanzo S3` (content-addressed by SHA-256).
2. **Enso (Model Routing):** Consume the `pareto_cost_aware` output table so Enso routes tasks with high-confidence crisp instructions to smaller/faster Zen tiers (`zen-mini-8b`) and escalates high-complexity/low-confidence tasks to flagship tiers (`zen-coder-480b`), capturing the observed ~7x cost/pass reduction without altering SDK call sites.
3. **Kai (Policy & Decision Gate):** Enforce `validate_security_boundary()` and minimum dev-set / held-out-test promotion gates inside Kai before any optimized instruction can be activated in production, and run Visor sandbox execution (`visor_sandbox_exec`) strictly behind Kai policy gates.
4. **Cloud (Lifecycle & Billing):** Track compilation-time token spend alongside serving-time savings so optimization ROI is measured net of compile cost (amortized over expected production call volume).
