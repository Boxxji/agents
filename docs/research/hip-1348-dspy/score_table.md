# HIP-1348 DSPy-Style Prompt & Workflow Optimization — Held-Out Score Table

- **Seed**: `42`
- **Held-out test set**: 48 executable tasks (12 per category; strictly disjoint from 32 train and 24 dev tasks)
- **Runtime**: Hanzo Python SDK (`AIConfig`, `llm_adapters`, `AgentAI`, `AgentWorkflow`) over deterministic Enso/Zen simulator

## Primary Held-Out Results (4 Strategies)

| Strategy | Held-Out Pass Rate (%) | Train-vs-Test Gap (pp) | Total Tokens | Est. Cost (USD) | Cost / Passing Task (USD) | Injection Defense (%) | p50 Latency (ms) | p95 Latency (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `zero_shot_baseline` | 37.50% (18/48) | +0.00 | 28,918 | $0.090355 | $0.005020 | 0.00% | 277.08 | 474.33 |
| `bootstrap_few_shot` | 87.50% (42/48) | +0.00 | 73,833 | $0.202725 | $0.004827 | 50.00% | 413.86 | 638.00 |
| `instruction_search` | 100.00% (48/48) | +0.00 | 32,459 | $0.099305 | $0.002069 | 100.00% | 288.56 | 490.73 |
| `pareto_cost_aware` | 97.92% (47/48) | +2.08 | 32,459 | $0.013903 | $0.000296 | 100.00% | 126.33 | 230.05 |

## Per-Category Held-Out Pass Rate (%) & Tokens

| Strategy | JSON Extraction | Tool Routing | Workflow Synthesis | Injection Defense |
| --- | ---: | ---: | ---: | ---: |
| `zero_shot_baseline` | 50.0% (6,703 tok) | 50.0% (6,188 tok) | 50.0% (9,018 tok) | 0.0% (7,009 tok) |
| `bootstrap_few_shot` | 100.0% (16,771 tok) | 100.0% (14,250 tok) | 100.0% (22,889 tok) | 50.0% (19,923 tok) |
| `instruction_search` | 100.0% (7,471 tok) | 100.0% (7,350 tok) | 100.0% (9,785 tok) | 100.0% (7,853 tok) |
| `pareto_cost_aware` | 91.7% (7,471 tok) | 100.0% (7,350 tok) | 100.0% (9,785 tok) | 100.0% (7,853 tok) |

## Negative Regimes & Failure-Mode Ablations

### 1. Over-Prompting Bloat on Simple Tasks (Held-Out Simple Subset)

| Strategy | Simple Pass Rate (%) | Simple Total Tokens | Simple Est. Cost (USD) | Token Overhead vs Baseline |
| --- | ---: | ---: | ---: | ---: |
| `zero_shot_baseline` | 100.00% | 10,460 | $0.032218 | +0.0% |
| `bootstrap_few_shot` | 100.00% | 26,462 | $0.072222 | +153.0% |
| `instruction_search` | 100.00% | 11,810 | $0.035592 | +12.9% |
| `pareto_cost_aware` | 100.00% | 11,810 | $0.004983 | +12.9% |

### 2. Small-Dev Overfitting Regime (`dev_size=4`, 1 task/category)

| Configuration | Dev Pass Rate (%) | Held-Out Test Pass Rate (%) | Dev-vs-Test Overfitting Gap (pp) |
| --- | ---: | ---: | ---: |
| `instruction_search` (full dev=24) | 100.00% | 100.00% | +0.00 |
| `instruction_search` (tiny dev=4) | 100.00% | 75.00% | +25.00 |

### 3. Security Boundary Erosion Without Guardrail (Injection Defense Held-Out)

| Optimizer Mode | Candidate Selected | Injection Defense Rate (%) | Held-Out Pass Rate (%) |
| --- | --- | ---: | ---: |
| `instruction_search` (security guard ON) | `delimiter_hardened_verified` | 100.00% | 100.00% |
| `pareto_cost_aware` (security guard OFF, strict 615-tok cap) | `generic_boundary` | 75.00% | 75.00% |
