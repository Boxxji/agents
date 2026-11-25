# HIP-1347 Research & Implementation Report
## [Bounty #36] Learning from Agent Execution Traces (Agent Lightning)
**Target Umbrella**: HIP-1347: Event-Driven Agent Reliability
**Target Subsystems**: `hanzoai/agents` / `hanzoai/ml` (Zen Checkpoints)
**License**: Apache-2.0 Clean-Room Implementation

---

### Research-to-PR Acceptance Contract Compliance

#### 1. Paper & License Audit
- **Primary References**:
  - Microsoft Research: *Agent Lightning: Train ANY AI Agents with Reinforcement Learning* (2025, `arXiv:2508.03680`).
  - Rafailov et al.: *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (NeurIPS 2023, `arXiv:2305.18290`).
- **License & Compatibility**:
  - Clean-room implementation in pure Python (`sdk/python/hanzo_agents/trace_converter.py`).
  - Zero proprietary dependencies; compatible with Apache-2.0 / MIT licenses.
  - Decoupled from vendor-specific LLM APIs.

#### 2. Matched Baselines
- **Naive Baseline**: Unfiltered trajectory concatenation without reward-hacking checks (admits test-deletion and vacuous-assertion shortcuts into training data).
- **Audited Pipeline (`TraceDatasetSynthesizer`)**:
  - Audits trajectories via `RewardHackingDetector` to quarantine test deletions, skipped suites, and vacuous assertion bypasses before emission.
  - Emits Zen multi-turn SFT JSONL records (`zen_sft_v1`) and task-matched DPO `(prompt, chosen, rejected)` pairs with token/cost accounting.

#### 3. Falsifiable Hypothesis
- **Hypothesis 1 (Converter Guardrail)**: Static reward-hacking auditing quarantines 100% of test-deletion / vacuous-pass patterns in curated negative traces with 0% false positives on clean multi-step traces.
- **Hypothesis 2 (Downstream Model Training — Deferred)**: Fine-tuning Zen checkpoints on audited SFT/DPO datasets reduces step count and shortcut failures compared to unfiltered trajectories (requires live Zen training runs; not claimed as measured here).

#### 4. Deliverables
- **Core Engine & Reproducible Benchmark**: `sdk/python/hanzo_agents/trace_converter.py`
  - `RewardHackingDetector`: Regex audit of agent tool calls and assistant code blocks.
  - `TraceDatasetSynthesizer`: SFT conversation formatter, DPO pair builder, token/cost estimator, and JSONL exporter.
  - `benchmark_trace_conversion(num_tasks=200)`: Deterministic reproducible benchmark CLI (`python -m hanzo_agents.trace_converter`).
- **Verification Suite**: `sdk/python/tests/test_trace_converter.py`
  - Unit tests covering SFT conversion, reward-hacking quarantine, DPO pair synthesis, JSONL export, and reproducible benchmark invariants.

#### 5. Evidence & Verification
Reproduce locally:
```bash
PYTHONPATH=sdk/python pytest -o addopts= -v sdk/python/tests/test_trace_converter.py
PYTHONPATH=sdk/python python -m hanzo_agents.trace_converter
```

#### 6. Measured Converter Metrics (Reproducible Harness Output)
Executed via `python -m hanzo_agents.trace_converter` (`num_tasks=200`, `800` trajectory steps):
- **SFT records emitted**: `200/200` (`clean_false_positive_rate=0.0%`)
- **DPO pairs emitted**: `200/200`
- **Reward-hacked traces quarantined**: `200/200` (`hacked_quarantine_rate=100.0%`)
- **Measured conversion throughput**: `30,744.9` trajectory steps/sec (`0.026021s` wall time on authoring/verification host)
- **Estimated token/cost accounting**: `28090` tokens (`$0.042090` at `$0.0015 / 1k` tokens)

#### Limitations
- Throughput and detection rates above measure the pure-Python trace converter and regex detector on curated trajectories — they do **not** claim downstream LLM fine-tuning accuracy or step-reduction gains until evaluated with live Zen SFT/DPO training checkpoints.
