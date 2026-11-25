# HIP-1347 Lost-in-the-Middle Context Packing — Measured Score Tables

- **Generated (UTC)**: `2026-11-25T10:18:14.549082+00:00`
- **Python / Platform**: `3.12.16` / `Linux-6.18.57-x86_64-with-glibc2.39`
- **Seed**: `42` | **Total Cases**: `110` | **Doc Counts (N)**: `[5, 10, 20]` | **Token Budgets**: `[512, 1024, 2048, 4096]`

> **Scientific Integrity Notice**: Physical evaluation modes (`budget_truncation`, `distractor_interference`) contain **zero assumed position-attention curves** and measure real Hanzo SDK `AIConfig.trim_by_chars`, `tail_clip`, lexical interference, and prompt-injection delimiter isolation. `calibrated_tacl2024` is explicitly labeled as an offline sensitivity stress-test profile parameterized from Liu et al. (TACL 2024) published measurements and is **not** a live Hanzo model claim.

## Run: `budget_truncation__hanzo_middle_trim` [PHYSICAL / NON-CIRCULAR]

*Physical / non-circular evaluation executed over Hanzo SDK MemoryClient and AIConfig.*

### Overall Standard Position Sweep (`N ∈ {5, 10, 20}`, All Positions `1..N`, Budgets `512..4096`)

| Policy | Answer Correctness | Evidence Retention | Truncation Loss | Mean Tokens | Efficiency (Acc/1K tok) | Boundary Integrity | Primacy (P1) | Middle (N/2) | Recency (PN) | p50 Lat (ms) | p95 Lat (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `chronological` | 60.7% | 60.7% | 39.3% | 1475.6 | 0.411 | 100.0% | 100.0% | 50.0% | 100.0% | 0.081 | 0.205 |
| `score_descending` | 100.0% | 100.0% | 0.0% | 1475.6 | 0.678 | 100.0% | 100.0% | 100.0% | 100.0% | 0.080 | 0.151 |
| `bookend_primacy_recency` | 100.0% | 100.0% | 0.0% | 1475.6 | 0.678 | 100.0% | 100.0% | 100.0% | 100.0% | 0.082 | 0.154 |
| `dedup_budgeted_bookend` | 100.0% | 100.0% | 0.0% | 1378.9 | 0.725 | 100.0% | 100.0% | 100.0% | 100.0% | 5.741 | 8.124 |
| `middle_buried_adversarial` | 39.3% | 39.3% | 60.7% | 1475.6 | 0.266 | 100.0% | 50.0% | 50.0% | 50.0% | 0.100 | 0.177 |

### Answer Correctness & Token Efficiency by Strict Token Budget

| Policy | Budget 512 | Budget 1024 | Budget 2048 | Budget 4096 |
|---|---:|---:|---:|---:|
| `chronological` | 25.7% (512 tok, eff=0.50) | 42.9% (1009 tok, eff=0.42) | 74.3% (1801 tok, eff=0.41) | 100.0% (2581 tok, eff=0.39) |
| `score_descending` | 100.0% (512 tok, eff=1.95) | 100.0% (1009 tok, eff=0.99) | 100.0% (1801 tok, eff=0.56) | 100.0% (2581 tok, eff=0.39) |
| `bookend_primacy_recency` | 100.0% (512 tok, eff=1.95) | 100.0% (1009 tok, eff=0.99) | 100.0% (1801 tok, eff=0.56) | 100.0% (2581 tok, eff=0.39) |
| `dedup_budgeted_bookend` | 100.0% (462 tok, eff=2.16) | 100.0% (891 tok, eff=1.12) | 100.0% (1793 tok, eff=0.56) | 100.0% (2370 tok, eff=0.42) |
| `middle_buried_adversarial` | 0.0% (512 tok, eff=0.00) | 14.3% (1009 tok, eff=0.14) | 42.9% (1801 tok, eff=0.24) | 100.0% (2581 tok, eff=0.39) |

### Paired Bootstrap 95% Confidence Intervals (Standard Sweep)

| Comparison | Mean Delta (A - B) | 95% CI Low | 95% CI High | Pairs |
|---|---:|---:|---:|---:|
| `dedup_budgeted_bookend_vs_chronological` | +0.3929 | +0.3321 | +0.4536 | 280 |
| `dedup_budgeted_bookend_vs_score_descending` | +0.0000 | +0.0000 | +0.0000 | 280 |
| `dedup_budgeted_bookend_vs_bookend_primacy_recency` | +0.0000 | +0.0000 | +0.0000 | 280 |
| `bookend_primacy_recency_vs_chronological` | +0.3929 | +0.3321 | +0.4536 | 280 |
| `dedup_budgeted_bookend_vs_middle_buried_adversarial` | +0.6071 | +0.5463 | +0.6643 | 280 |

### Negative & Adversarial Failure Regimes

| Regime | Policy | Correctness | Retention | Trunc Loss | Mean Tokens | Boundary Integrity |
|---|---|---:|---:|---:|---:|---:|
| `small_context` | `chronological` | 100.0% | 100.0% | 0.0% | 565.1 | 100.0% |
| `small_context` | `score_descending` | 100.0% | 100.0% | 0.0% | 565.1 | 100.0% |
| `small_context` | `bookend_primacy_recency` | 100.0% | 100.0% | 0.0% | 565.1 | 100.0% |
| `small_context` | `dedup_budgeted_bookend` | 100.0% | 100.0% | 0.0% | 606.1 | 100.0% |
| `small_context` | `middle_buried_adversarial` | 100.0% | 100.0% | 0.0% | 565.1 | 100.0% |
| `misranked_retriever` | `chronological` | 100.0% | 100.0% | 0.0% | 1287.1 | 100.0% |
| `misranked_retriever` | `score_descending` | 97.5% | 97.5% | 2.5% | 1287.1 | 100.0% |
| `misranked_retriever` | `bookend_primacy_recency` | 52.5% | 52.5% | 47.5% | 1287.1 | 100.0% |
| `misranked_retriever` | `dedup_budgeted_bookend` | 42.5% | 42.5% | 57.5% | 1303.8 | 100.0% |
| `misranked_retriever` | `middle_buried_adversarial` | 50.0% | 50.0% | 50.0% | 1287.1 | 100.0% |
| `duplicate_heavy` | `chronological` | 70.0% | 70.0% | 30.0% | 1291.5 | 100.0% |
| `duplicate_heavy` | `score_descending` | 50.0% | 50.0% | 50.0% | 1291.5 | 100.0% |
| `duplicate_heavy` | `bookend_primacy_recency` | 50.0% | 50.0% | 50.0% | 1291.5 | 100.0% |
| `duplicate_heavy` | `dedup_budgeted_bookend` | 75.0% | 75.0% | 25.0% | 1156.2 | 100.0% |
| `duplicate_heavy` | `middle_buried_adversarial` | 50.0% | 50.0% | 50.0% | 1291.5 | 100.0% |
| `prompt_injection` | `chronological` | 20.0% | 70.0% | 30.0% | 1297.4 | 50.0% |
| `prompt_injection` | `score_descending` | 47.5% | 100.0% | 0.0% | 1297.4 | 47.5% |
| `prompt_injection` | `bookend_primacy_recency` | 0.0% | 100.0% | 0.0% | 1297.4 | 0.0% |
| `prompt_injection` | `dedup_budgeted_bookend` | 100.0% | 100.0% | 0.0% | 1320.5 | 100.0% |
| `prompt_injection` | `middle_buried_adversarial` | 0.0% | 50.0% | 50.0% | 1297.4 | 50.0% |

## Run: `budget_truncation__tail_clip` [PHYSICAL / NON-CIRCULAR]

*Physical / non-circular evaluation executed over Hanzo SDK MemoryClient and AIConfig.*

### Overall Standard Position Sweep (`N ∈ {5, 10, 20}`, All Positions `1..N`, Budgets `512..4096`)

| Policy | Answer Correctness | Evidence Retention | Truncation Loss | Mean Tokens | Efficiency (Acc/1K tok) | Boundary Integrity | Primacy (P1) | Middle (N/2) | Recency (PN) | p50 Lat (ms) | p95 Lat (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `chronological` | 62.9% | 62.9% | 37.1% | 1475.6 | 0.426 | 100.0% | 100.0% | 75.0% | 50.0% | 0.078 | 0.124 |
| `score_descending` | 100.0% | 100.0% | 0.0% | 1475.6 | 0.678 | 100.0% | 100.0% | 100.0% | 100.0% | 0.075 | 0.134 |
| `bookend_primacy_recency` | 100.0% | 100.0% | 0.0% | 1475.6 | 0.678 | 100.0% | 100.0% | 100.0% | 100.0% | 0.078 | 0.134 |
| `dedup_budgeted_bookend` | 100.0% | 100.0% | 0.0% | 1378.9 | 0.725 | 100.0% | 100.0% | 100.0% | 100.0% | 5.634 | 7.262 |
| `middle_buried_adversarial` | 64.3% | 64.3% | 35.7% | 1475.6 | 0.436 | 100.0% | 75.0% | 75.0% | 75.0% | 0.095 | 0.150 |

### Answer Correctness & Token Efficiency by Strict Token Budget

| Policy | Budget 512 | Budget 1024 | Budget 2048 | Budget 4096 |
|---|---:|---:|---:|---:|
| `chronological` | 25.7% (512 tok, eff=0.50) | 48.6% (1009 tok, eff=0.48) | 77.1% (1801 tok, eff=0.43) | 100.0% (2581 tok, eff=0.39) |
| `score_descending` | 100.0% (512 tok, eff=1.95) | 100.0% (1009 tok, eff=0.99) | 100.0% (1801 tok, eff=0.56) | 100.0% (2581 tok, eff=0.39) |
| `bookend_primacy_recency` | 100.0% (512 tok, eff=1.95) | 100.0% (1009 tok, eff=0.99) | 100.0% (1801 tok, eff=0.56) | 100.0% (2581 tok, eff=0.39) |
| `dedup_budgeted_bookend` | 100.0% (462 tok, eff=2.16) | 100.0% (891 tok, eff=1.12) | 100.0% (1793 tok, eff=0.56) | 100.0% (2370 tok, eff=0.42) |
| `middle_buried_adversarial` | 14.3% (512 tok, eff=0.28) | 42.9% (1009 tok, eff=0.42) | 100.0% (1801 tok, eff=0.56) | 100.0% (2581 tok, eff=0.39) |

### Paired Bootstrap 95% Confidence Intervals (Standard Sweep)

| Comparison | Mean Delta (A - B) | 95% CI Low | 95% CI High | Pairs |
|---|---:|---:|---:|---:|
| `dedup_budgeted_bookend_vs_chronological` | +0.3714 | +0.3179 | +0.4286 | 280 |
| `dedup_budgeted_bookend_vs_score_descending` | +0.0000 | +0.0000 | +0.0000 | 280 |
| `dedup_budgeted_bookend_vs_bookend_primacy_recency` | +0.0000 | +0.0000 | +0.0000 | 280 |
| `bookend_primacy_recency_vs_chronological` | +0.3714 | +0.3179 | +0.4286 | 280 |
| `dedup_budgeted_bookend_vs_middle_buried_adversarial` | +0.3571 | +0.3035 | +0.4143 | 280 |

### Negative & Adversarial Failure Regimes

| Regime | Policy | Correctness | Retention | Trunc Loss | Mean Tokens | Boundary Integrity |
|---|---|---:|---:|---:|---:|---:|
| `small_context` | `chronological` | 100.0% | 100.0% | 0.0% | 565.1 | 100.0% |
| `small_context` | `score_descending` | 100.0% | 100.0% | 0.0% | 565.1 | 100.0% |
| `small_context` | `bookend_primacy_recency` | 100.0% | 100.0% | 0.0% | 565.1 | 100.0% |
| `small_context` | `dedup_budgeted_bookend` | 100.0% | 100.0% | 0.0% | 606.1 | 100.0% |
| `small_context` | `middle_buried_adversarial` | 100.0% | 100.0% | 0.0% | 565.1 | 100.0% |
| `misranked_retriever` | `chronological` | 75.0% | 75.0% | 25.0% | 1287.1 | 100.0% |
| `misranked_retriever` | `score_descending` | 50.0% | 50.0% | 50.0% | 1287.1 | 100.0% |
| `misranked_retriever` | `bookend_primacy_recency` | 72.5% | 72.5% | 27.5% | 1287.1 | 100.0% |
| `misranked_retriever` | `dedup_budgeted_bookend` | 42.5% | 42.5% | 57.5% | 1303.8 | 100.0% |
| `misranked_retriever` | `middle_buried_adversarial` | 75.0% | 75.0% | 25.0% | 1287.1 | 100.0% |
| `duplicate_heavy` | `chronological` | 72.5% | 72.5% | 27.5% | 1291.5 | 100.0% |
| `duplicate_heavy` | `score_descending` | 75.0% | 75.0% | 25.0% | 1291.5 | 100.0% |
| `duplicate_heavy` | `bookend_primacy_recency` | 100.0% | 100.0% | 0.0% | 1291.5 | 100.0% |
| `duplicate_heavy` | `dedup_budgeted_bookend` | 75.0% | 75.0% | 25.0% | 1156.2 | 100.0% |
| `duplicate_heavy` | `middle_buried_adversarial` | 72.5% | 75.0% | 25.0% | 1291.5 | 100.0% |
| `prompt_injection` | `chronological` | 0.0% | 72.5% | 27.5% | 1297.4 | 0.0% |
| `prompt_injection` | `score_descending` | 0.0% | 100.0% | 0.0% | 1297.4 | 0.0% |
| `prompt_injection` | `bookend_primacy_recency` | 2.5% | 100.0% | 0.0% | 1297.4 | 2.5% |
| `prompt_injection` | `dedup_budgeted_bookend` | 100.0% | 100.0% | 0.0% | 1320.5 | 100.0% |
| `prompt_injection` | `middle_buried_adversarial` | 0.0% | 67.5% | 32.5% | 1297.4 | 0.0% |

## Run: `distractor_interference__hanzo_middle_trim` [PHYSICAL / NON-CIRCULAR]

*Physical / non-circular evaluation executed over Hanzo SDK MemoryClient and AIConfig.*

### Overall Standard Position Sweep (`N ∈ {5, 10, 20}`, All Positions `1..N`, Budgets `512..4096`)

| Policy | Answer Correctness | Evidence Retention | Truncation Loss | Mean Tokens | Efficiency (Acc/1K tok) | Boundary Integrity | Primacy (P1) | Middle (N/2) | Recency (PN) | p50 Lat (ms) | p95 Lat (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `chronological` | 60.7% | 60.7% | 39.3% | 1475.6 | 0.411 | 100.0% | 100.0% | 50.0% | 100.0% | 0.083 | 0.130 |
| `score_descending` | 100.0% | 100.0% | 0.0% | 1475.6 | 0.678 | 100.0% | 100.0% | 100.0% | 100.0% | 0.080 | 0.122 |
| `bookend_primacy_recency` | 100.0% | 100.0% | 0.0% | 1475.6 | 0.678 | 100.0% | 100.0% | 100.0% | 100.0% | 0.080 | 0.132 |
| `dedup_budgeted_bookend` | 100.0% | 100.0% | 0.0% | 1378.9 | 0.725 | 100.0% | 100.0% | 100.0% | 100.0% | 5.539 | 7.013 |
| `middle_buried_adversarial` | 39.3% | 39.3% | 60.7% | 1475.6 | 0.266 | 100.0% | 50.0% | 50.0% | 50.0% | 0.099 | 0.145 |

### Answer Correctness & Token Efficiency by Strict Token Budget

| Policy | Budget 512 | Budget 1024 | Budget 2048 | Budget 4096 |
|---|---:|---:|---:|---:|
| `chronological` | 25.7% (512 tok, eff=0.50) | 42.9% (1009 tok, eff=0.42) | 74.3% (1801 tok, eff=0.41) | 100.0% (2581 tok, eff=0.39) |
| `score_descending` | 100.0% (512 tok, eff=1.95) | 100.0% (1009 tok, eff=0.99) | 100.0% (1801 tok, eff=0.56) | 100.0% (2581 tok, eff=0.39) |
| `bookend_primacy_recency` | 100.0% (512 tok, eff=1.95) | 100.0% (1009 tok, eff=0.99) | 100.0% (1801 tok, eff=0.56) | 100.0% (2581 tok, eff=0.39) |
| `dedup_budgeted_bookend` | 100.0% (462 tok, eff=2.16) | 100.0% (891 tok, eff=1.12) | 100.0% (1793 tok, eff=0.56) | 100.0% (2370 tok, eff=0.42) |
| `middle_buried_adversarial` | 0.0% (512 tok, eff=0.00) | 14.3% (1009 tok, eff=0.14) | 42.9% (1801 tok, eff=0.24) | 100.0% (2581 tok, eff=0.39) |

### Paired Bootstrap 95% Confidence Intervals (Standard Sweep)

| Comparison | Mean Delta (A - B) | 95% CI Low | 95% CI High | Pairs |
|---|---:|---:|---:|---:|
| `dedup_budgeted_bookend_vs_chronological` | +0.3929 | +0.3321 | +0.4536 | 280 |
| `dedup_budgeted_bookend_vs_score_descending` | +0.0000 | +0.0000 | +0.0000 | 280 |
| `dedup_budgeted_bookend_vs_bookend_primacy_recency` | +0.0000 | +0.0000 | +0.0000 | 280 |
| `bookend_primacy_recency_vs_chronological` | +0.3929 | +0.3321 | +0.4536 | 280 |
| `dedup_budgeted_bookend_vs_middle_buried_adversarial` | +0.6071 | +0.5463 | +0.6643 | 280 |

### Negative & Adversarial Failure Regimes

| Regime | Policy | Correctness | Retention | Trunc Loss | Mean Tokens | Boundary Integrity |
|---|---|---:|---:|---:|---:|---:|
| `small_context` | `chronological` | 100.0% | 100.0% | 0.0% | 565.1 | 100.0% |
| `small_context` | `score_descending` | 100.0% | 100.0% | 0.0% | 565.1 | 100.0% |
| `small_context` | `bookend_primacy_recency` | 100.0% | 100.0% | 0.0% | 565.1 | 100.0% |
| `small_context` | `dedup_budgeted_bookend` | 100.0% | 100.0% | 0.0% | 606.1 | 100.0% |
| `small_context` | `middle_buried_adversarial` | 100.0% | 100.0% | 0.0% | 565.1 | 100.0% |
| `misranked_retriever` | `chronological` | 100.0% | 100.0% | 0.0% | 1287.1 | 100.0% |
| `misranked_retriever` | `score_descending` | 97.5% | 97.5% | 2.5% | 1287.1 | 100.0% |
| `misranked_retriever` | `bookend_primacy_recency` | 52.5% | 52.5% | 47.5% | 1287.1 | 100.0% |
| `misranked_retriever` | `dedup_budgeted_bookend` | 42.5% | 42.5% | 57.5% | 1303.8 | 100.0% |
| `misranked_retriever` | `middle_buried_adversarial` | 50.0% | 50.0% | 50.0% | 1287.1 | 100.0% |
| `duplicate_heavy` | `chronological` | 20.0% | 70.0% | 30.0% | 1291.5 | 100.0% |
| `duplicate_heavy` | `score_descending` | 0.0% | 50.0% | 50.0% | 1291.5 | 100.0% |
| `duplicate_heavy` | `bookend_primacy_recency` | 0.0% | 50.0% | 50.0% | 1291.5 | 100.0% |
| `duplicate_heavy` | `dedup_budgeted_bookend` | 75.0% | 75.0% | 25.0% | 1156.2 | 100.0% |
| `duplicate_heavy` | `middle_buried_adversarial` | 0.0% | 50.0% | 50.0% | 1291.5 | 100.0% |
| `prompt_injection` | `chronological` | 20.0% | 70.0% | 30.0% | 1297.4 | 50.0% |
| `prompt_injection` | `score_descending` | 47.5% | 100.0% | 0.0% | 1297.4 | 47.5% |
| `prompt_injection` | `bookend_primacy_recency` | 0.0% | 100.0% | 0.0% | 1297.4 | 0.0% |
| `prompt_injection` | `dedup_budgeted_bookend` | 100.0% | 100.0% | 0.0% | 1320.5 | 100.0% |
| `prompt_injection` | `middle_buried_adversarial` | 0.0% | 50.0% | 50.0% | 1297.4 | 50.0% |

## Run: `calibrated_tacl2024__gpt-3.5-turbo-16k` [CALIBRATED STRESS-TEST (OFFLINE PROFILE)]

*CALIBRATED OFFLINE SENSITIVITY STRESS-TEST (Liu et al. TACL 2024 published curve) — NOT a live Hanzo model measurement.*

### Overall Standard Position Sweep (`N ∈ {5, 10, 20}`, All Positions `1..N`, Budgets `512..4096`)

| Policy | Answer Correctness | Evidence Retention | Truncation Loss | Mean Tokens | Efficiency (Acc/1K tok) | Boundary Integrity | Primacy (P1) | Middle (N/2) | Recency (PN) | p50 Lat (ms) | p95 Lat (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `chronological` | 41.3% | 60.7% | 39.3% | 1475.6 | 0.280 | 100.0% | 75.0% | 34.0% | 71.3% | 0.081 | 0.151 |
| `score_descending` | 75.0% | 100.0% | 0.0% | 1475.6 | 0.508 | 100.0% | 75.0% | 75.0% | 75.0% | 0.079 | 0.158 |
| `bookend_primacy_recency` | 75.0% | 100.0% | 0.0% | 1475.6 | 0.508 | 100.0% | 75.0% | 75.0% | 75.0% | 0.079 | 0.140 |
| `dedup_budgeted_bookend` | 75.1% | 100.0% | 0.0% | 1378.9 | 0.545 | 100.0% | 75.2% | 75.2% | 75.2% | 5.694 | 7.786 |
| `middle_buried_adversarial` | 25.1% | 39.3% | 60.7% | 1475.6 | 0.170 | 100.0% | 34.0% | 34.0% | 34.0% | 0.098 | 0.157 |

### Answer Correctness & Token Efficiency by Strict Token Budget

| Policy | Budget 512 | Budget 1024 | Budget 2048 | Budget 4096 |
|---|---:|---:|---:|---:|
| `chronological` | 19.2% (512 tok, eff=0.38) | 31.3% (1009 tok, eff=0.31) | 50.8% (1801 tok, eff=0.28) | 63.6% (2581 tok, eff=0.25) |
| `score_descending` | 75.3% (512 tok, eff=1.47) | 75.1% (1009 tok, eff=0.74) | 74.7% (1801 tok, eff=0.41) | 74.7% (2581 tok, eff=0.29) |
| `bookend_primacy_recency` | 75.3% (512 tok, eff=1.47) | 75.1% (1009 tok, eff=0.74) | 74.7% (1801 tok, eff=0.41) | 74.7% (2581 tok, eff=0.29) |
| `dedup_budgeted_bookend` | 75.8% (462 tok, eff=1.64) | 75.4% (891 tok, eff=0.85) | 74.8% (1793 tok, eff=0.42) | 74.6% (2370 tok, eff=0.31) |
| `middle_buried_adversarial` | 0.0% (512 tok, eff=0.00) | 10.5% (1009 tok, eff=0.10) | 29.6% (1801 tok, eff=0.16) | 60.4% (2581 tok, eff=0.23) |

### Paired Bootstrap 95% Confidence Intervals (Standard Sweep)

| Comparison | Mean Delta (A - B) | 95% CI Low | 95% CI High | Pairs |
|---|---:|---:|---:|---:|
| `dedup_budgeted_bookend_vs_chronological` | +0.3388 | +0.2967 | +0.3797 | 280 |
| `dedup_budgeted_bookend_vs_score_descending` | +0.0017 | +0.0015 | +0.0020 | 280 |
| `dedup_budgeted_bookend_vs_bookend_primacy_recency` | +0.0017 | +0.0015 | +0.0020 | 280 |
| `bookend_primacy_recency_vs_chronological` | +0.3371 | +0.2950 | +0.3777 | 280 |
| `dedup_budgeted_bookend_vs_middle_buried_adversarial` | +0.5002 | +0.4598 | +0.5372 | 280 |

### Negative & Adversarial Failure Regimes

| Regime | Policy | Correctness | Retention | Trunc Loss | Mean Tokens | Boundary Integrity |
|---|---|---:|---:|---:|---:|---:|
| `small_context` | `chronological` | 75.8% | 100.0% | 0.0% | 565.1 | 100.0% |
| `small_context` | `score_descending` | 75.8% | 100.0% | 0.0% | 565.1 | 100.0% |
| `small_context` | `bookend_primacy_recency` | 75.8% | 100.0% | 0.0% | 565.1 | 100.0% |
| `small_context` | `dedup_budgeted_bookend` | 75.8% | 100.0% | 0.0% | 606.1 | 100.0% |
| `small_context` | `middle_buried_adversarial` | 75.8% | 100.0% | 0.0% | 565.1 | 100.0% |
| `misranked_retriever` | `chronological` | 75.0% | 100.0% | 0.0% | 1287.1 | 100.0% |
| `misranked_retriever` | `score_descending` | 73.1% | 97.5% | 2.5% | 1287.1 | 100.0% |
| `misranked_retriever` | `bookend_primacy_recency` | 39.3% | 52.5% | 47.5% | 1287.1 | 100.0% |
| `misranked_retriever` | `dedup_budgeted_bookend` | 31.7% | 42.5% | 57.5% | 1303.8 | 100.0% |
| `misranked_retriever` | `middle_buried_adversarial` | 37.4% | 50.0% | 50.0% | 1287.1 | 100.0% |
| `duplicate_heavy` | `chronological` | 52.4% | 70.0% | 30.0% | 1291.5 | 100.0% |
| `duplicate_heavy` | `score_descending` | 37.4% | 50.0% | 50.0% | 1291.5 | 100.0% |
| `duplicate_heavy` | `bookend_primacy_recency` | 37.4% | 50.0% | 50.0% | 1291.5 | 100.0% |
| `duplicate_heavy` | `dedup_budgeted_bookend` | 56.3% | 75.0% | 25.0% | 1156.2 | 100.0% |
| `duplicate_heavy` | `middle_buried_adversarial` | 37.4% | 50.0% | 50.0% | 1291.5 | 100.0% |
| `prompt_injection` | `chronological` | 14.7% | 70.0% | 30.0% | 1297.4 | 50.0% |
| `prompt_injection` | `score_descending` | 35.7% | 100.0% | 0.0% | 1297.4 | 47.5% |
| `prompt_injection` | `bookend_primacy_recency` | 0.0% | 100.0% | 0.0% | 1297.4 | 0.0% |
| `prompt_injection` | `dedup_budgeted_bookend` | 75.2% | 100.0% | 0.0% | 1320.5 | 100.0% |
| `prompt_injection` | `middle_buried_adversarial` | 0.0% | 50.0% | 50.0% | 1297.4 | 50.0% |
