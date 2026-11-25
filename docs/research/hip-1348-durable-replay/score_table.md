# Durable Workflow Replay Benchmark — Score Table (HIP-1348, `#78` / `#85`)

- **Generated (UTC)**: `2026-11-25T11:12:23.351858+00:00`
- **Suite Digest**: `029a0dbf8d64cc2c`
- **Host**: `Linux 6.18.57` (`x86_64`, `Intel(R) Xeon(R) CPU @ 2.30GHz`, `2 vCPU`, Python `3.12.16`, `/proc/self/io=True`)

## 1. Primary Cross-Process Crash-at-Every-Step Matrix (`os._exit(99)` across 6 steps x 3 crash modes = 18 trials/cohort)

| Cohort | Trials | Hard Crashes (`exit 99`) | Resumption Rate | Exactly-Once Rate | Duplicate Side Effects | External Dedup Hits | Resume Replay Load p50/p95 (ms) | Ckpt Write p50/p95 (ms) | Mean Store Footprint (KiB) | Mean Resume `/proc/self/io` wchar (KiB) |
| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ephemeral_baseline_no_idempotency` | 18 | PASS | 100.0% | 5.6% | 57 | 0 | 0.012 / 0.060 | 0.017 / 0.086 | 0.00 | 165.63 |
| `ephemeral_baseline_with_idempotency` | 18 | PASS | 100.0% | 100.0% | 0 | 57 | 0.012 / 0.015 | 0.017 / 0.064 | 0.00 | 118.02 |
| `sqlite_wal_durable` | 18 | PASS | 100.0% | 100.0% | 0 | 6 | 0.450 / 0.682 | 7.349 / 11.594 | 353.88 | 215.25 |
| `atomic_jsonl_durable` | 18 | PASS | 100.0% | 100.0% | 0 | 6 | 2.051 / 2.853 | 28.571 / 58.881 | 16.45 | 86.13 |

## 2. Single-Run Multi-Crash Gauntlet (Crash at EVERY Step `1..6` on ONE `run_id` across 7 OS PIDs)

| Backend | Crashes Injected | Distinct OS PIDs | Completed | Exactly-Once (`applied==1` all steps) | Duplicate Side Effects | External Dedup Hits | Final Replayed Steps | Store Footprint (KiB) |
| --- | ---: | ---: | :---: | :---: | ---: | ---: | ---: | ---: |
| `sqlite_wal` | 6 | 4 | YES | YES | 0 | 2 | 6 | 353.88 |
| `atomic_jsonl` | 6 | 4 | YES | YES | 0 | 2 | 6 | 15.86 |

## 3. Negative Regime #1 — Persisted Checkpoints WITHOUT Activity `idempotency_key` (`after_effect_before_ack` crashes)

| Cohort | Trials | Resumption Rate | Exactly-Once Rate | Duplicate Side Effects | Mean Duplicates / Trial |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sqlite_wal_no_idempotency_after_effect_crash` | 6 | 100.0% | 0.0% | 6 | 1.00 |
| `atomic_jsonl_no_idempotency_after_effect_crash` | 6 | 100.0% | 0.0% | 6 | 1.00 |

## 4. Negative Regime #2 — Non-Deterministic Replay Divergence Detection (Fail-Closed)

- **Detected Fail-Closed**: `4/4`
- **Zero Corrupted External Writes**: `4/4`

| Backend | Mutation | Hard Crash (`exit 99`) | Fail-Closed Error | Zero Corrupted Writes |
| --- | --- | :---: | --- | :---: |
| `sqlite_wal` | `reorder_steps` | YES | `NonDeterministicReplayError` | YES |
| `sqlite_wal` | `change_activity_type` | YES | `NonDeterministicReplayError` | YES |
| `atomic_jsonl` | `reorder_steps` | YES | `NonDeterministicReplayError` | YES |
| `atomic_jsonl` | `change_activity_type` | YES | `NonDeterministicReplayError` | YES |

## 5. Negative Regime #3 — Persistence Latency Overhead Crossover (No-Crash Happy Path)

| Workflow Regime | Backend | Runs | Mean Wall (ms) | Wall p50/p95 (ms) | Ckpt Write p50/p95 (ms) | Overhead vs Ephemeral (ms) | Ratio vs Ephemeral | Mean `/proc/self/io` wchar (KiB) | Store Footprint / Run (KiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `trivial_1_step` | `ephemeral_baseline` | 15 | 8.509 | 8.503 / 9.727 | 0.018 / 0.115 | +0.000 | 1.00x | 26.29 | 0.00 |
| `trivial_1_step` | `sqlite_wal` | 15 | 43.003 | 42.004 / 48.797 | 7.995 / 10.321 | +34.494 | 5.05x | 98.98 | 77.51 |
| `trivial_1_step` | `atomic_jsonl` | 15 | 118.666 | 121.403 / 138.523 | 21.533 / 50.769 | +110.157 | 13.95x | 30.28 | 3.41 |
| `standard_6_step` | `ephemeral_baseline` | 5 | 49.755 | 48.933 / 56.287 | 0.019 / 0.043 | +0.000 | 1.00x | 158.52 | 0.00 |
| `standard_6_step` | `sqlite_wal` | 5 | 138.448 | 138.276 / 144.821 | 6.672 / 8.019 | +88.693 | 2.78x | 450.62 | 306.55 |
| `standard_6_step` | `atomic_jsonl` | 5 | 476.341 | 467.448 / 504.427 | 22.244 / 50.472 | +426.586 | 9.57x | 177.00 | 15.91 |
