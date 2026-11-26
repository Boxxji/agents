# HIP-1347 Durable Event-Driven Workflow Replay — Measured Score Tables

- **Generated (UTC)**: `2026-11-26T20:55:02Z`
- **Host**: `linux/amd64`, `go1.26.5`, `2` vCPUs
- **Seed**: `42` | **Checkpoint Interval (`K`)**: `8`
- **Zero-Daemon Verification**: baseline goroutines=`1`, post-run=`1` (delta=`0`), idle active locks=`0`, passed=`true`

> **Measurement Attribution**: All latency, storage, and I/O measurements are real local-disk wall-clock measurements on the authoring host running control-plane/internal/replay with FileCheckpointStore. No external LLM/Temporal cluster is invoked; figures must not be extrapolated to production cloud hardware without reproducing on target Hanzo infrastructure.

## 1. Multi-Step Coding-Agent Crash Recovery (`fsync=true`, Crash at 75% of Workflow)

| Steps (`N`) | Crash Step | Mode | Time-to-Resume p50 / p95 / p99 (ms) | Full Recovery p50 (ms) | Events Loaded | Bytes Read on Load | Dup Side Effects | Storage (B/step) | Ckpt Restored |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 10 | 7 | `stateless_retry` | 0.000 / 0.123 / 0.134 | 0.000 | 0.0 | 0 | 21 / 30 (70.00%) | 0.0 | 0% |
| 10 | 7 | `full_history_replay` | 1.068 / 3.129 / 3.312 | 44.707 | 16.0 | 10863 | 0 / 30 (0.00%) | 2043.1 | 0% |
| 10 | 7 | `checkpointed_replay` | 0.000 / 0.575 / 0.626 | 48.004 | 16.0 | 10863 | 0 / 30 (0.00%) | 2043.1 | 0% |
| 40 | 30 | `stateless_retry` | 0.000 / 0.115 / 0.125 | 0.000 | 0.0 | 0 | 90 / 120 (75.00%) | 0.0 | 0% |
| 40 | 30 | `full_history_replay` | 0.740 / 1.499 / 1.566 | 144.011 | 62.0 | 42601 | 0 / 120 (0.00%) | 1923.8 | 0% |
| 40 | 30 | `checkpointed_replay` | 0.472 / 0.483 / 0.484 | 155.771 | 13.0 | 9704 | 0 / 120 (0.00%) | 1974.8 | 100% |
| 100 | 75 | `stateless_retry` | 0.531 / 1.905 / 2.027 | 0.745 | 0.0 | 0 | 225 / 300 (75.00%) | 0.0 | 0% |
| 100 | 75 | `full_history_replay` | 4.770 / 9.144 / 9.533 | 353.380 | 152.0 | 104911 | 0 / 300 (0.00%) | 1905.3 | 0% |
| 100 | 75 | `checkpointed_replay` | 0.326 / 0.574 / 0.596 | 379.308 | 7.0 | 5575 | 0 / 300 (0.00%) | 1955.8 | 100% |

## 2. Deep-History Recovery Scaling (`fsync=false` Page-Cache Isolated, Crash at 90%)

| Steps (`N`) | Crash Step | Mode | Time-to-Resume p50 / p95 / p99 (ms) | Full Recovery p50 (ms) | Events Loaded | Bytes Read on Load | Dup Side Effects | Storage (B/step) |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 50 | 45 | `stateless_retry` | 0.356 / 0.673 / 1.171 | 0.396 | 0.0 | 0 | 675 / 750 (90.00%) | 0.0 |
| 50 | 45 | `full_history_replay` | 2.218 / 3.173 / 3.439 | 7.164 | 92.0 | 63323 | 0 / 750 (0.00%) | 1916.7 |
| 50 | 45 | `checkpointed_replay` | 0.433 / 0.882 / 1.483 | 4.889 | 11.0 | 8327 | 0 / 750 (0.00%) | 1965.3 |
| 200 | 180 | `stateless_retry` | 1.505 / 4.000 / 4.000 | 1.827 | 0.0 | 0 | 2700 / 3000 (90.00%) | 0.0 |
| 200 | 180 | `full_history_replay` | 8.495 / 11.976 / 12.733 | 26.824 | 362.0 | 250692 | 0 / 3000 (0.00%) | 1901.5 |
| 200 | 180 | `checkpointed_replay` | 0.409 / 0.852 / 0.929 | 19.212 | 9.0 | 6996 | 0 / 3000 (0.00%) | 1949.3 |
| 500 | 450 | `stateless_retry` | 3.646 / 8.272 / 9.350 | 4.000 | 0.0 | 0 | 6750 / 7500 (90.00%) | 0.0 |
| 500 | 450 | `full_history_replay` | 20.901 / 24.885 / 25.937 | 66.262 | 902.0 | 625565 | 0 / 7500 (0.00%) | 1898.6 |
| 500 | 450 | `checkpointed_replay` | 0.273 / 0.468 / 0.506 | 53.943 | 5.0 | 4215 | 0 / 7500 (0.00%) | 1946.5 |

## 3. Concurrent Multi-Tenant Duplicate Webhook Storm (`fsync=true`, 8 Tenants × 5 Deliveries)

| Mode | Activation p50 / p95 (ms) | Expected Side Effects | Observed Side Effects | Duplicate Rate | Cross-Tenant Violations | Idle Locks |
|---|---:|---:|---:|---:|---:|---:|
| `stateless_retry` | 0.056 / 0.579 | 80 | 400 | 400.00% | 0 | 0 |
| `full_history_replay` | 1226.836 / 1232.989 | 80 | 80 | 0.00% | 0 | 0 |
| `checkpointed_replay` | 1190.544 / 1199.836 | 80 | 80 | 0.00% | 0 | 0 |

## 4. Honest Negative Regimes

### NR1: Trivial 1-Step Workflow (`N=1`, 0 Crashes, `fsync=true`)

| Mode | Trials | Happy-Path p50 / p95 / p99 (ms) | Storage (B/step) |
|---|---:|---:|---:|
| `stateless_retry` | 20 | 0.004 / 0.008 / 0.030 | 0.0 |
| `checkpointed_replay` | 20 | 36.003 / 39.815 / 40.261 | 3346.1 |

### NR2: Large Binary Step Payloads (`N=10`, 256 KiB Blob/Step, `fsync=true`)

| Payload Strategy | Happy-Path p50 / p95 (ms) | Recovery p50 / p95 (ms) | Storage (B/step) | Bytes Read on Recovery |
|---|---:|---:|---:|---:|
| `inline_256kb_blob` | 400.301 / 410.650 | 157.536 / 167.096 | 1085154.5 | 2451053 |
| `vfs_content_hash_ref` | 194.403 / 200.234 | 51.107 / 52.095 | 1856.2 | 4889 |

### NR3: Over-Frequent Checkpointing Tradeoff (`N=40`, `fsync=true`)

| Checkpoint Interval (`K`) | Checkpoints Saved | Happy-Path p50 / p95 (ms) | Crash Recovery p50 / p95 (ms) | Storage (B/step) |
|---:|---:|---:|---:|---:|
| 1 | 40 | 1123.389 / 1159.330 | 136.010 / 137.347 | 2084.7 |
| 8 | 5 | 656.020 / 773.843 | 78.104 / 83.384 | 1776.7 |
| 20 | 2 | 554.525 / 583.615 | 76.683 / 81.738 | 1753.6 |
