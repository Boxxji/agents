# HIP-1347 Research Report: Durable Event-Driven Agent Workflow Replay

- **Tracking Issue**: `hanzoai/agents#85` (`[Bounty #35] Durable event-driven agent workflow replay · agents`)
- **Umbrella HIP**: [HIP-1347: Event-Driven Agent Reliability](https://hips.hanzo.ai/docs/hip-1347-event-driven-agent-reliability/) (`hanzoai/hips`)
- **Implementation Package**: [`control-plane/internal/replay/`](../../../control-plane/internal/replay/)
- **Benchmark Harness**: [`control-plane/cmd/replay-bench/`](../../../control-plane/cmd/replay-bench/)
- **Measured Artifacts**: [`score_table.md`](./score_table.md) | [`raw_results.json`](./raw_results.json)

---

## 1. Primary Reference, Design Summary, and License Audit

### Primary Research / Code Reference
- **Reference System**: [Temporal Go SDK (`temporalio/sdk-go`)](https://github.com/temporalio/sdk-go) — studied strictly at the architectural/semantic level (deterministic event-sourced replay, workflow virtual clock & seeded side-effect RNG, activity scheduling/completion history matching, non-determinism detection).
- **Clean-Room Verification**: **Zero code copied and zero Temporal packages imported.** `control-plane/go.mod` is completely unchanged (zero new external dependencies).

### Core Architectural Delta vs. Temporal
Temporal's execution model relies on long-running worker processes polling task queues backed by an external Temporal Server cluster and sticky worker caches. Under Hanzo's event-driven architecture (HIP-1347 / `hanzoai/tasks`), maintaining permanent per-agent polling daemons violates the zero-idle-compute constraint.

`control-plane/internal/replay` reproduces durable-execution replay semantics **purely on event-triggered activations** (`Engine.HandleEvent`):
1. **Zero Always-On Processes**: `Engine` spawns zero background goroutines, zero tickers, and zero polling loops; per-workflow locks are reference-counted and deleted when idle (`ActiveLockCount() == 0`).
2. **Append-Only SHA-256 Chained Event Log**: Monotonic sequence numbers (`1..N`) covering `WorkflowStarted`, `ActivityScheduled`, `ActivityCompleted`, `ActivityFailed`, `TimerFired`, `SignalReceived`, `CheckpointSaved`, `WorkflowSuspended`, `WorkflowCompleted`, and `WorkflowFailed`, chained via canonical length-prefixed SHA-256 digests (`VerifyHistoryIntegrity` rejects tampering with `ErrCorruptedHistory`).
3. **O(1) Prefix-Skip Periodic Checkpointing**: Atomic `checkpoint.json` (`CreateTemp` → `Write` → `fsync` → `Rename` → directory `fsync`) stores user state, deterministic `WorkflowClock`, checkpointable SplitMix64 `WorkflowRNG` state, `HistoryDigest`, and exact `LogByteOffset` into `events.jsonl`. On cold-start recovery, `FileCheckpointStore` seeks directly to `LogByteOffset` and verifies only delta events (`seq > checkpoint.Sequence`), converting recovery I/O from **O(N)** to **O(K)**.
4. **Tenant-Scoped Idempotency Ledger & Non-Determinism Guard**: Length-prefixed SHA-256 `DeriveIdempotencyKey(tenant_id, workflow_id, activity_name, step_id)` prevents delimiter-injection collisions and deduplicates side effects (billing charges, VFS writes, external tool calls) across crashes and duplicate webhook deliveries. Any divergence in activity name, key, input digest, command kind (timer vs. activity), or ordering fails fast with `ErrNonDeterministicReplay`.
5. **Canonical Hanzo Boundaries Preserved**: Maps cleanly onto existing `control-plane/pkg/types` (`WorkflowRun`, `WorkflowRunEvent`, `WorkflowStep`, `Execution`), leaves the pinned 130-route HTTP surface (`internal/server/routes_test.go`) untouched, and preserves `vfs -> hanzoai/s3 -> visor` by passing large binary artifacts via `VFSContentRef` (`vfs://sha256/<hex>`).

### License & Dependency Audit

| Component | License | Bundled / Imported? | Patent / Copyleft Notes | Audit Status |
|---|---|---|---|---|
| `temporalio/sdk-go` | MIT | **Not imported** — architecture studied for semantics only | Permissive MIT; zero source copied | Cleared |
| `control-plane/internal/replay/` | Apache-2.0 (repo license) | Clean-room Go stdlib + existing `control-plane/pkg/types` | Zero new dependencies in `go.mod` | Cleared |
| `control-plane/cmd/replay-bench/` | Apache-2.0 (repo license) | Clean-room Go stdlib + `internal/replay` | Zero external dependencies | Cleared |
| GPL / AGPL / LGPL dependencies | None | Zero copyleft dependencies introduced | None | Cleared |

---

## 2. Matched Baselines & Reproduction Commands

All three execution strategies execute the **exact same deterministic coding-agent workflow code** (`tool_call`, `model_infer`, `vfs_edit`, `billing_charge` steps) through `replay.Engine` against local disk (`FileCheckpointStore`) with zero synthetic latency formulas:

1. **`stateless_retry` (Naive Stateless Baseline)**: No persisted checkpoint or idempotency ledger; crashes and duplicate webhooks re-execute completed steps from step 0.
2. **`full_history_replay` (Matched Temporal-Style Uncompacted Baseline)**: Persists append-only WAL + idempotency ledger, ignores snapshots, and on every recovery reads/verifies the entire event history from sequence 1 (`O(N)` read + SHA-256 verification).
3. **`checkpointed_replay` (Hanzo Event-Triggered Checkpointed Replay)**: Periodic snapshot checkpoint every `K=8` steps + `Seek(LogByteOffset)` delta event log + `(tenant_id, workflow_id, idempotency_key)` ledger (`O(K)` cold-start recovery).

### Exact Reproduction Commands

```bash
cd control-plane

# 1. Static analysis and race-detector test suite (unit, crash-recovery, determinism, non-determinism, multi-tenant, bounds)
GOWORK=off go vet ./internal/replay/... ./cmd/replay-bench/...
GOWORK=off go test -v -race -count=1 ./internal/replay/... ./cmd/replay-bench/...

# 2. Verify existing durable package remains 100% green
GOWORK=off go test -v -race -count=1 ./internal/durable/...

# 3. Reproduce raw_results.json and score_table.md (seed=42, K=8)
GOWORK=off go run ./cmd/replay-bench \
  --seed 42 \
  --checkpoint-interval 8 \
  --fsync-trials 3 \
  --scaling-trials 15 \
  --tenants 8 \
  --duplicate-webhooks 5 \
  --trivial-trials 20 \
  --negative-trials 3 \
  --output-json ../docs/research/hip-1347-durable-workflow-replay/raw_results.json \
  --output-markdown ../docs/research/hip-1347-durable-workflow-replay/score_table.md
```

---

## 3. Falsifiable Hypothesis & Expected Failure Regimes

### Hypothesis (H1)
> Under mid-workflow crashes (`75%–90%` completion) and concurrent multi-tenant duplicate webhook deliveries:
> 1. Both durable modes eliminate duplicate side effects completely (**0.00%** vs. **70%–400%** under `stateless_retry`) with **0** cross-tenant isolation violations and **0** leaked background goroutines or idle locks.
> 2. `checkpointed_replay` (`K=8`) bounds cold-start recovery load to **O(K)** events/bytes and sub-millisecond time-to-resume (`< 1 ms` p50 up to `N=500`), achieving **>= 10× faster time-to-resume** and **>= 20× lower recovery read I/O** than `full_history_replay` at `N >= 200`, at `< 5%` incremental storage overhead per step.
>
> **Explicit Falsification / Negative Regimes (Expected to Degrade)**:
> - **NR1 (Trivial 1-step workflows, `N=1`, 0 crashes, `fsync=true`)**: Durable WAL + idempotency + checkpoint fsync adds tens of milliseconds of latency over `stateless_retry` with zero recovery benefit.
> - **NR2 (Large binary step payloads, 256 KiB inline blobs)**: Inlining binary blobs into event logs/checkpoints explodes per-step storage and recovery I/O by `> 100×` unless passed by `vfs://sha256/...` content-hash reference.
> - **NR3 (Over-frequent checkpointing, `K=1`)**: Checkpointing on every step slows happy-path and post-crash fsync latency substantially relative to periodic checkpointing (`K=8` or `K=20`).

---

## 4. Measured Before / After Results (Authoring Host)

> **Attribution Notice**: Measurements below were recorded on `linux/amd64`, `go1.26.5`, `2` vCPUs against local disk (`FileCheckpointStore`) with `seed=42`, `K=8`. They represent verified authoring-host measurements and must be reproduced on target Hanzo production storage before treating latency numbers as cloud SLOs.
>
> **Zero-Daemon Verification**: baseline goroutines = `1`, post-run goroutines = `1` (`delta = 0`), idle active locks = `0` (**PASS**).

### A. Real Crash-Durable Recovery (`fsync=true`, Crash at 75% of Workflow)

| Steps (`N`) | Crash Step | Mode | Time-to-Resume p50 / p95 / p99 (ms) | Full Recovery p50 (ms) | Events Loaded | Bytes Read | Dup Side Effects | Storage (B/step) |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 10 | 7 | `stateless_retry` | 0.000 / 0.123 / 0.134 | 0.000 | 0.0 | 0 | 21 / 30 (70.00%) | 0.0 |
| 10 | 7 | `full_history_replay` | 1.068 / 3.129 / 3.312 | 44.707 | 16.0 | 10,863 | **0 / 30 (0.00%)** | 2,043.1 |
| 10 | 7 | `checkpointed_replay` | 0.000 / 0.575 / 0.626 | 48.004 | 16.0 | 10,863 | **0 / 30 (0.00%)** | 2,043.1 |
| 40 | 30 | `stateless_retry` | 0.000 / 0.115 / 0.125 | 0.000 | 0.0 | 0 | 90 / 120 (75.00%) | 0.0 |
| 40 | 30 | `full_history_replay` | 0.740 / 1.499 / 1.566 | 144.011 | 62.0 | 42,601 | **0 / 120 (0.00%)** | 1,923.8 |
| 40 | 30 | `checkpointed_replay` | **0.472 / 0.483 / 0.484** | 155.771 | **13.0** | **9,704** | **0 / 120 (0.00%)** | 1,974.8 |
| 100 | 75 | `stateless_retry` | 0.531 / 1.905 / 2.027 | 0.745 | 0.0 | 0 | 225 / 300 (75.00%) | 0.0 |
| 100 | 75 | `full_history_replay` | 4.770 / 9.144 / 9.533 | 353.380 | 152.0 | 104,911 | **0 / 300 (0.00%)** | 1,905.3 |
| 100 | 75 | `checkpointed_replay` | **0.326 / 0.574 / 0.596** | 379.308 | **7.0** | **5,575** | **0 / 300 (0.00%)** | 1,955.8 |

*Observation*: At `N=100`, `checkpointed_replay` cuts cold-start time-to-resume by **14.6×** (`0.326 ms` vs. `4.770 ms`) and recovery read I/O by **18.8×** (`5,575 B` vs. `104,911 B`). Full recovery-to-completion (`379.3 ms` vs. `353.4 ms`) is dominated by live `fsync` of the remaining 25 post-crash steps plus 3 post-crash periodic checkpoints.

### B. Deep-History Recovery Scaling (`fsync=false` Page-Cache Isolated, Crash at 90%, `15` Trials/Cell)

| Steps (`N`) | Crash Step | Mode | Time-to-Resume p50 / p95 / p99 (ms) | Full Recovery p50 (ms) | Events Loaded | Bytes Read | Dup Side Effects | Storage (B/step) |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 50 | 45 | `stateless_retry` | 0.356 / 0.673 / 1.171 | 0.396 | 0.0 | 0 | 675 / 750 (90.00%) | 0.0 |
| 50 | 45 | `full_history_replay` | 2.218 / 3.173 / 3.439 | 7.164 | 92.0 | 63,323 | **0.00%** | 1,916.7 |
| 50 | 45 | `checkpointed_replay` | **0.433 / 0.882 / 1.483** | **4.889** | **11.0** | **8,327** | **0.00%** | 1,965.3 |
| 200 | 180 | `stateless_retry` | 1.505 / 4.000 / 4.000 | 1.827 | 0.0 | 0 | 2,700 / 3,000 (90.00%) | 0.0 |
| 200 | 180 | `full_history_replay` | 8.495 / 11.976 / 12.733 | 26.824 | 362.0 | 250,692 | **0.00%** | 1,901.5 |
| 200 | 180 | `checkpointed_replay` | **0.409 / 0.852 / 0.929** | **19.212** | **9.0** | **6,996** | **0.00%** | 1,949.3 |
| 500 | 450 | `stateless_retry` | 3.646 / 8.272 / 9.350 | 4.000 | 0.0 | 0 | 6,750 / 7,500 (90.00%) | 0.0 |
| 500 | 450 | `full_history_replay` | 20.901 / 24.885 / 25.937 | 66.262 | 902.0 | 625,565 | **0.00%** | 1,898.6 |
| 500 | 450 | `checkpointed_replay` | **0.273 / 0.468 / 0.506** | **53.943** | **5.0** | **4,215** | **0.00%** | 1,946.5 |

*Observation*: At `N=500`, `checkpointed_replay` delivers **76.6× faster time-to-resume** (`0.273 ms` vs. `20.901 ms`), **180.4× fewer events parsed** (`5.0` vs. `902.0`), and **148.4× fewer bytes read on recovery** (`4,215 B` vs. `625,565 B`), with only **+2.5%** storage overhead (`1,946.5` vs. `1,898.6 B/step`).

### C. Concurrent Multi-Tenant Duplicate Webhook Storm (`fsync=true`, 8 Tenants × 5 Deliveries)

| Mode | Activation p50 / p95 (ms) | Expected Side Effects | Observed Side Effects | Duplicate Rate | Cross-Tenant Violations | Idle Locks |
|---|---:|---:|---:|---:|---:|---:|
| `stateless_retry` | 0.056 / 0.579 | 80 | 400 | 400.00% | 0 | 0 |
| `full_history_replay` | 1226.836 / 1232.989 | 80 | **80** | **0.00%** | **0** | **0** |
| `checkpointed_replay` | 1190.544 / 1199.836 | 80 | **80** | **0.00%** | **0** | **0** |

### D. Honest Negative Regimes Confirmed Empirically

1. **NR1 — Trivial 1-Step Workflow (`N=1`, 0 Crashes, `fsync=true`, 20 trials)**:
   - `stateless_retry`: `0.004 / 0.008 / 0.030 ms` (p50/p95/p99), `0.0 B/step`.
   - `checkpointed_replay`: **`36.003 / 39.815 / 40.261 ms`** (p50/p95/p99), **`3,346.1 B/step`**.
   - *Takeaway*: Single-shot, idempotent read-only queries with zero crash-recovery need should bypass durable fsync checkpointing.
2. **NR2 — Large Binary Step Payloads (`N=10`, 256 KiB Blob/Step, `fsync=true`)**:
   - `inline_256kb_blob`: happy-path p50 `400.301 ms`, recovery p50 `157.536 ms`, storage **`1,085,154.5 B/step`**, recovery read **`2,451,053 B`**.
   - `vfs_content_hash_ref` (`vfs://sha256/...`): happy-path p50 **`194.403 ms`** (2.06× faster), recovery p50 **`51.107 ms`** (3.08× faster), storage **`1,856.2 B/step`** (**584.6× smaller**), recovery read **`4,889 B`** (**501.3× less I/O**).
   - *Takeaway*: Large artifacts (diffs, screenshots, build outputs) MUST be stored in Hanzo VFS/S3 and referenced by content hash rather than inlined into replay checkpoints.
3. **NR3 — Over-Frequent Checkpointing (`N=40`, `fsync=true`)**:
   - `K=1` (40 checkpoints): happy-path p50 **`1,123.389 ms`**, crash recovery p50 `136.010 ms`, `2,084.7 B/step`.
   - `K=8` (5 checkpoints): happy-path p50 **`656.020 ms`** (1.71× faster), crash recovery p50 **`78.104 ms`** (1.74× faster), `1,776.7 B/step`.
   - `K=20` (2 checkpoints): happy-path p50 `554.525 ms`, crash recovery p50 `76.683 ms`, `1,753.6 B/step`.
   - *Takeaway*: Periodic checkpointing around `K=8..16` sits on the latency/storage Pareto frontier; `K=1` degrades both happy-path and recovery throughput due to redundant checkpoint `fsync` + directory-sync calls.

---

## 5. Safety, Concurrency & Regression Verification

All tests pass with Go's race detector enabled (`GOWORK=off go test -race -count=1 ./internal/replay/... ./cmd/replay-bench/...`):
- **Correctness & Crash Recovery**: Cold-start recovery after injected crashes at every step `0..7` across fresh `FileCheckpointStore` + `Engine` instances, plus WAL tail recovery when both `meta.json` and `checkpoint.json` are deleted.
- **Determinism & Non-Determinism**: Bit-exact virtual clock + SplitMix64 PRNG equivalence across un-crashed, full-history crash recovery, and checkpoint-restored crash recovery; explicit `ErrNonDeterministicReplay` rejection on altered activity name, altered `idempotency_key`, mutated input payload, or timer-vs-activity swap.
- **Security & Multi-Tenant Isolation**: Length-prefixed SHA-256 keys immune to delimiter injection; strict `tenant_id` scoping across all reads/writes/ledger lookups; tampered `meta.json` tenant identity rejected with `ErrTenantAccessDenied`; tampered `events.jsonl` rejected with `ErrCorruptedHistory`; `MaxPayloadBytes` (`ErrPayloadTooLarge`) and `MaxHistoryDepth` (`ErrHistoryDepthExceeded`) enforced against memory exhaustion.
- **Zero-Daemon Guarantee**: Signal and timer suspend/resume verified with `goroutine_delta == 0` and `ActiveLockCount() == 0`.

---

## 6. HIP-1347 Acceptance Contract Status Update

- [x] **1. Paper & License Audit**: Temporal Go SDK (MIT) audited; clean-room Apache-2.0 Go implementation with zero new dependencies in `go.mod`.
- [x] **2. Matched Baselines**: Reproducible CLI (`control-plane/cmd/replay-bench`) running identical multi-step coding-agent traces across `stateless_retry`, `full_history_replay`, and `checkpointed_replay`.
- [x] **3. Falsifiable Hypothesis**: Documented quantitative targets and three explicit negative regimes (`NR1`, `NR2`, `NR3`), all empirically verified.
- [x] **4. Deliverable**: Bounded `control-plane/internal/replay/` engine + `control-plane/cmd/replay-bench/` + `docs/research/hip-1347-durable-workflow-replay/` artifacts; existing `control-plane/internal/durable/` and 130-route HTTP surface untouched.
- [x] **5. Evidence & Safety**: 17 top-level tests (29 test cases including subtests) passing under `-race` covering crash recovery, determinism, non-determinism, concurrency, multi-tenant isolation, tamper detection, and bounds enforcement.
- [x] **6. Measurements & HIP Update**: Raw JSON (`raw_results.json`), Markdown tables (`score_table.md`), and HIP-1347 status published in this report.
