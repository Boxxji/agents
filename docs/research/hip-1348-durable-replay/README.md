# HIP-1348 Research Report: Durable Event-Driven Agent Workflow Replay

- **Issues**: [`hanzoai/agents#78`](https://github.com/hanzoai/agents/issues/78) & [`hanzoai/agents#85`](https://github.com/hanzoai/agents/issues/85)
- **HIP**: HIP-1348 (draft) / HIP-1347 — Event-Driven Agent Reliability
- **Primary Reference**: Temporal Go SDK ([`temporalio/sdk-go`](https://github.com/temporalio/sdk-go), MIT License)
- **Control-Plane Parity Reference**: `control-plane/internal/durable/` (Go stateless journal + `sha256(org:runID:round:stepID)` idempotency ledger) & `sdk/python/hanzo_agents/agent_workflow.py` (`AgentWorkflow` + `ExecutionContext`)
- **Raw Data & Tables**: [`raw_results.json`](./raw_results.json) | [`score_table.md`](./score_table.md)

---

## 1. Primary Reference Summary & License Audit

### 1.1 Temporal Go SDK Durable Execution Semantics
Temporal's durable execution model (`temporalio/sdk-go`) achieves crash-safe workflow resumption without requiring sticky in-memory state on worker hosts through three coupled invariants:

1. **Append-Only Event History**: Every workflow lifecycle transition (`WorkflowStarted`, `ActivityScheduled`, `ActivityCompleted`, `ActivityFailed`, `WorkflowSuspended`, `WorkflowResumed`, `WorkflowCompleted`, `WorkflowFailed`) is durably persisted in strict sequence order.
2. **Deterministic Replay Verification**: When a workflow resumes after interruption, the engine replays the workflow definition against persisted event history, returning cached activity outputs for previously completed steps and failing closed (`NonDeterministicReplayError`) whenever workflow code diverges from recorded history (step reordering, activity type changes, or input payload mutation).
3. **Activity-Level Idempotency Across the Two-Phase Crash Window**: Because an external side effect (billing charge, S3 commit, git push, webhook) and the workflow journal commit live in separate failure domains, a hard process crash *after* external commit and *before* journal acknowledgement cannot be prevented by checkpoints alone; retrying with a deterministic `idempotency_key` (`sha256(org:run_id:round:step_id)`) closes the window to guarantee **exactly-once** external application.

### 1.2 License & Patent Audit (Clean-Room Clearance)

| Artifact | License | Vendored in `hanzoai/agents`? | Audit Decision |
| --- | --- | --- | --- |
| Temporal Go SDK (`github.com/temporalio/sdk-go`) | **MIT** | **No** (semantics studied and reproduced clean-room; zero upstream files copied) | Cleared — MIT compatible with Apache-2.0; no copyleft or patent restrictions |
| `control-plane/internal/durable/` (existing Go engine on `main`) | **Apache-2.0** | Yes (pre-existing on `main`) | Reused canonical `sha256(org:runID:round:stepID)` key formula and event schema |
| `sdk/python/benchmarks/durable_replay/` & tests | **Apache-2.0** (original clean-room Python + stdlib + `hanzo_agents`) | Yes | Cleared for Apache-2.0 merge; zero new runtime dependencies |

No Temporal server, Cassandra/MySQL cluster, or always-on background worker daemon is introduced. All replay activations are stateless and event-triggered.

---

## 2. Canonical Architecture Alignment & Matched Baselines

This experiment aligns strictly with Hanzo's canonical boundaries (`vfs -> Hanzo S3 -> Visor`, Cloud owning lifecycle/billing, Enso routing models, Zen generating, Kai deciding, `hanzoai/tasks` driving event-triggered execution) and directly exercises `hanzo_agents.agent_workflow.AgentWorkflow` + `ExecutionContext`:

- **Matched Real Cross-Process Harness**: Every trial launches real OS subprocesses (`subprocess.run([sys.executable, "-m", "benchmarks.durable_replay.worker", ...])`) and injects unbuffered hard OS crashes via `os._exit(99)` (bypassing Python `finally` blocks, context managers, and `atexit` handlers) at **every step** `k = 0..5` across **three crash windows**:
  1. `before_activity_commit`: crash after `ActivityScheduled` is persisted, before external execution.
  2. `after_effect_before_ack`: crash *after* the independent external SQLite WAL service (`ExternalSideEffectLedger`) commits the side effect, *before* the workflow checkpoint store persists `ActivityCompleted`.
  3. `after_step_commit`: crash *after* `ActivityCompleted`, ledger entry, and step snapshot are durably fsynced, before step `k+1`.
- **Three Compared Persistence Backends** (identical 6-step agent workflow: `kai_plan_task -> vfs_fetch_context -> zen_generate_patch -> visor_run_tests -> s3_commit_artifact -> cloud_publish_summary`):
  1. `ephemeral_baseline`: unpersisted in-memory Python `dict` (control baseline representing naive non-durable runners; tested both without and with downstream `idempotency_key` deduplication).
  2. `sqlite_wal`: real SQLite (`PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;`) persisting `workflow_events`, `workflow_runs`, `activity_ledger`, and `step_snapshots` inside a single atomic `BEGIN IMMEDIATE ... COMMIT` transaction.
  3. `atomic_jsonl`: append+`os.fsync` JSON-lines journal + atomic `tempfile + os.fsync + os.replace` state/ledger/snapshot files with SHA-256 directory sharding and `fcntl.flock` serialization (Python parity mirror of Go `control-plane/internal/durable/journal.go`).

---

## 3. Falsifiable Hypotheses & Expected Failure Modes

- **H1 (Cross-Process Exactly-Once Resumption)**: Under hard `os._exit(99)` crashes at every workflow step `k = 0..5` and all 3 crash windows (`18` trials/backend), plus a single-run 6-crash gauntlet across distinct OS PIDs, both `sqlite_wal_durable` and `atomic_jsonl_durable` achieve **100.0% resumption completion**, **100.0% exactly-once application (`applied_count == 1` for every step)**, **0 duplicate side effects**, and **>= 85% fewer redundant downstream invocations** than `ephemeral_baseline_with_idempotency`.
  - *Falsification criterion*: Any step across any crash point records `applied_count != 1` or fails to resume cleanly.
- **H2 (Negative Regime #1 — Two-Phase Crash Window Necessity)**: Persisted checkpoints *alone* (`sqlite_wal` or `atomic_jsonl`) with `enable_idempotency_guard=False` will suffer **100% duplicate-execution failure** (`applied_count == 2` on the crashed step) under `after_effect_before_ack` crashes, proving mathematically and empirically that both persisted checkpoints and activity-level `idempotency_key` deduplication are jointly required.
- **H3 (Negative Regime #2 — Fail-Closed Non-Determinism Detection)**: Mutating workflow step order or activity type between hard crash and restart will be detected **100% fail-closed** (`NonDeterministicReplayError`) with **0 corrupted external writes**.
- **H4 (Negative Regime #3 — Persistence Latency Crossover on Trivial Workflows)**: On no-crash 1-step workflows (`< 10 ms` baseline), durable `synchronous=FULL` / `os.fsync` persistence imposes a measurable multi-x wall-clock overhead (`>= 3x` for SQLite WAL, `>= 10x` for per-file atomic JSONL), whereas SQLite WAL outperforms atomic JSONL on checkpoint write latency by `>= 3x` while trading larger fixed WAL page footprint.

---

## 4. Reproducible Measurements

### 4.1 Environment & Reproduction Command

- **Host Hardware**: `Linux 6.18.57` (`x86_64`), `Intel(R) Xeon(R) CPU @ 2.30GHz` (`2 vCPU`), Python `3.12.16`, `/proc/self/io` enabled.
- **Suite Digest**: `029a0dbf8d64cc2c` (full 64-hex SHA-256 in [`raw_results.json`](./raw_results.json)).

Reproduction command (from `sdk/python/`):

```bash
python3 -m benchmarks.durable_replay \
  --steps 6 \
  --trivial-runs 15 \
  --output-dir ../../docs/research/hip-1348-durable-replay
```

### 4.2 Primary Cross-Process Crash-at-Every-Step Matrix (18 Real `os._exit(99)` Trials / Cohort)

| Cohort | Trials | Hard Crashes (`exit 99`) | Resumption Rate | Exactly-Once Rate | Duplicate Side Effects | External Dedup Hits | Replay Load p50/p95 (ms) | Ckpt Write p50/p95 (ms) | Mean Store Footprint (KiB) | Mean Resume `/proc/self/io` wchar (KiB) |
| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ephemeral_baseline_no_idempotency` | 18 | PASS | 100.0% | **5.6%** | **57** | 0 | 0.012 / 0.060 | 0.017 / 0.086 | 0.00 | 165.63 |
| `ephemeral_baseline_with_idempotency` | 18 | PASS | 100.0% | 100.0% | 0 | **57** | 0.012 / 0.015 | 0.017 / 0.064 | 0.00 | 118.02 |
| `sqlite_wal_durable` | 18 | PASS | **100.0%** | **100.0%** | **0** | **6** | **0.450 / 0.682** | **7.349 / 11.594** | 353.88 | 215.25 |
| `atomic_jsonl_durable` | 18 | PASS | **100.0%** | **100.0%** | **0** | **6** | 2.051 / 2.853 | 28.571 / 58.881 | **16.45** | **86.13** |

#### Single-Run Multi-Crash Gauntlet (6 Consecutive Hard Crashes on ONE `run_id` Across 7 OS Process Invocations)

| Backend | Crashes Injected | Distinct External-Call PIDs | Completed | Exactly-Once (`applied==1` all steps) | Duplicate Side Effects | External Dedup Hits | Final Replayed Steps | Store Footprint (KiB) |
| --- | ---: | ---: | :---: | :---: | ---: | ---: | ---: | ---: |
| `sqlite_wal` | 6 | 4 | **YES** | **YES** | **0** | 2 | 6 | 353.88 |
| `atomic_jsonl` | 6 | 4 | **YES** | **YES** | **0** | 2 | 6 | **15.86** |

*(Note: 7 OS processes ran sequentially on the same `run_id`; 4 distinct PIDs appear in `effect_invocations` because 2 crashes fired in `before_activity_commit` before reaching the external ledger and attempt #7 replayed all 6 steps from disk with zero external calls.)*

**H1 Confirmed**:
- `ephemeral_baseline_no_idempotency` suffered **57 duplicate external side effects** (`5.6%` exactly-once rate — only step 0 `before_activity_commit` avoided duplication).
- `ephemeral_baseline_with_idempotency` avoided duplicates only by forcing **57 redundant downstream calls** (re-running every completed pre-crash step).
- Both `sqlite_wal_durable` and `atomic_jsonl_durable` achieved **100.0% resumption**, **0 duplicate side effects**, and only **6 external dedup hits** (strictly the 6 `after_effect_before_ack` crashes — **-89.5% redundant downstream traffic**).
- Comparing persisted stores: `sqlite_wal` is **4.56x faster on replay load** (`0.450 ms` vs `2.051 ms` p50) and **3.89x faster on checkpoint writes** (`7.349 ms` vs `28.571 ms` p50), while `atomic_jsonl` has a **21.5x smaller on-disk footprint** (`16.45 KiB` vs `353.88 KiB` WAL+SHM pages) and **2.50x lower resume `/proc/self/io` `wchar`** (`86.13 KiB` vs `215.25 KiB`).

### 4.3 Negative Regime #1 — Persisted Checkpoints Without `idempotency_key` (`after_effect_before_ack`)

| Cohort | Trials | Resumption Rate | Exactly-Once Rate | Duplicate Side Effects | Mean Duplicates / Trial |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sqlite_wal_no_idempotency_after_effect_crash` | 6 | 100.0% | **0.0%** | **6** | **1.00** |
| `atomic_jsonl_no_idempotency_after_effect_crash` | 6 | 100.0% | **0.0%** | **6** | **1.00** |

**H2 Confirmed (Honest Failure Mode)**: Even with durable `synchronous=FULL` SQLite WAL or `os.fsync` JSONL checkpoints, crashing in the window between external commit and journal write yields **0.0% exactly-once compliance** (`applied_count == 2` at every crashed step) when downstream activities lack `idempotency_key` deduplication. Persisted checkpoints are necessary to skip already-acknowledged steps, and `idempotency_key` is necessary to protect the in-flight commit window.

### 4.4 Negative Regime #2 — Non-Deterministic Replay Divergence (Fail-Closed)

| Backend | Mutation | Hard Crash (`exit 99`) | Fail-Closed Error | Zero Corrupted Writes |
| --- | --- | :---: | --- | :---: |
| `sqlite_wal` | `reorder_steps` | YES | `NonDeterministicReplayError` | **YES** |
| `sqlite_wal` | `change_activity_type` | YES | `NonDeterministicReplayError` | **YES** |
| `atomic_jsonl` | `reorder_steps` | YES | `NonDeterministicReplayError` | **YES** |
| `atomic_jsonl` | `change_activity_type` | YES | `NonDeterministicReplayError` | **YES** |

**H3 Confirmed**: **4/4** code-mutation replays failed closed with `NonDeterministicReplayError` and **0** corrupted external side effects.

### 4.5 Negative Regime #3 — Persistence Latency Overhead Crossover (No-Crash Happy Path)

| Workflow Regime | Backend | Runs | Mean Wall (ms) | Wall p50/p95 (ms) | Ckpt Write p50/p95 (ms) | Overhead vs Ephemeral (ms) | Ratio vs Ephemeral | Mean `/proc/self/io` wchar (KiB) | Store Footprint / Run (KiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `trivial_1_step` | `ephemeral_baseline` | 15 | **8.509** | 8.503 / 9.727 | 0.018 / 0.115 | +0.000 | 1.00x | 26.29 | 0.00 |
| `trivial_1_step` | `sqlite_wal` | 15 | 43.003 | 42.004 / 48.797 | 7.995 / 10.321 | **+34.494** | **5.05x** | 98.98 | 77.51 |
| `trivial_1_step` | `atomic_jsonl` | 15 | 118.666 | 121.403 / 138.523 | 21.533 / 50.769 | **+110.157** | **13.95x** | 30.28 | 3.41 |
| `standard_6_step` | `ephemeral_baseline` | 5 | **49.755** | 48.933 / 56.287 | 0.019 / 0.043 | +0.000 | 1.00x | 158.52 | 0.00 |
| `standard_6_step` | `sqlite_wal` | 5 | 138.448 | 138.276 / 144.821 | 6.672 / 8.019 | **+88.693** | **2.78x** | 450.62 | 306.55 |
| `standard_6_step` | `atomic_jsonl` | 5 | 476.341 | 467.448 / 504.427 | 22.244 / 50.472 | **+426.586** | **9.57x** | 177.00 | 15.91 |

**H4 Confirmed (Honest Overhead Crossover)**: On trivial 1-step local calls (`8.5 ms`), synchronous durability adds `+34.5 ms` (**5.05x**) for SQLite WAL and `+110.2 ms` (**13.95x**) for per-file atomic JSONL (caused by multiple `fsync` barriers per step across ledger, snapshot, journal, state, and parent directory). On realistic multi-step agent workflows (6 steps, where real LLM + Visor tool calls take 5,000–30,000 ms), SQLite WAL's total `+88.7 ms` durability cost is `< 1.8%` of total wall time while completely eliminating duplicate side effects and redundant LLM calls on crash.

---

## 5. Correctness, Concurrency, Security & Regression Evidence

All 31 automated verification tests live in [`sdk/python/tests/test_durable_replay_benchmark.py`](../../../sdk/python/tests/test_durable_replay_benchmark.py):

- **Idempotency & Store Integrity (11 tests)**: Exact `sha256(org:runID:round:stepID)` parity with Go `control-plane/internal/durable/idempotency.go`, invalid/traversal identifier rejection (8 parametrized edge cases), SQLite `WAL` + `synchronous=FULL` PRAGMA verification, and cross-restart replay persistence.
- **Real Cross-Process Crash-at-Every-Step & Multi-Crash Gauntlet (8 tests)**: `os._exit(99)` injected at every step across all 3 crash windows on both `sqlite_wal` and `atomic_jsonl`, plus 5-step multi-crash gauntlet across distinct OS PIDs confirming `applied_count == 1` and `duplicate_side_effects == 0`.
- **Negative Regimes (4 tests)**: No-idempotency-guard `after_effect_before_ack` duplicate proof (`applied_count == 2`), ephemeral baseline duplicate proof, fail-closed `NonDeterministicReplayError` on step reorder / activity type mutation / input payload mutation.
- **Event-Driven Suspension & Symlink Safety (2 tests)**: Multi-process `WorkflowSuspended` -> external signal wake-up -> `WorkflowCompleted` with zero background daemons, plus explicit symlinked `TMPDIR` resolution test (macOS `/var -> /private/var` compatibility).
- **Concurrency, Multi-Tenant Security & Torn-Tail Recovery (5 tests)**: 12 parallel workflows via `asyncio.gather` sharing one SQLite WAL or atomic JSONL store with zero lock errors or cross-tenant bleed, strict `(org, run_id)` tenant isolation + fail-closed `actor_id` / `workflow_id` spoofing (`TenantIsolationError`), and `AtomicJsonCheckpointStore` recovery from a simulated mid-write torn JSONL tail line.
- **Hanzo SDK Regression & Artifact Roundtrip (2 tests)**: Real `hanzo_agents.agent_workflow.AgentWorkflow` lifecycle event integration (`running`/`succeeded` events emitted to `/v1/workflow/executions/events` on execution and skipped on replay) and full benchmark JSON + Markdown artifact generation with zero local path leakage.

---

## 6. Limitations & Canonical Integration Recommendation

### Limitations
1. **Single-node local disk vs S3 replication latency**: Benchmarks measure local POSIX `fsync` latency on Linux NVMe/block storage. Production `vfs -> Hanzo S3 -> Visor` deployments will additionally observe S3 `PUT`/`GET` tail latency (`p95`/`p99`) when hydrating or flushing checkpoints across ephemeral Visor nodes.
2. **Cooperative idempotency requirement on external tools**: Exactly-once guarantees across the `after_effect_before_ack` crash window require external tools/APIs (Cloud billing, S3 object upload, git commit webhooks) to accept and deduplicate on `idempotency_key` (`sha256(org:run_id:round:step_id)`). Non-idempotent legacy tools must be wrapped in an idempotent proxy or compensation step.
3. **Workflow versioning on live in-flight upgrades**: Strict deterministic replay fails closed when workflow code changes mid-run; long-running multi-day workflows require versioned workflow identifiers or `checkpoint` boundary migration rather than in-place step reordering.

### Integration Recommendation (`hanzoai/tasks` + `vfs -> Hanzo S3 -> Visor` + `control-plane/internal/durable`)
Adopt **hybrid SQLite WAL hot checkpoints + Hanzo S3 cold snapshots** triggered strictly by `hanzoai/tasks` events (zero always-on bot processes):

1. **Hot Path (Visor / Worker Node)**: Use **SQLite WAL (`synchronous=FULL`)** for active multi-step workflow execution — **3.89x faster checkpoint writes** (`7.35 ms` vs `28.57 ms` p50) and **4.56x faster replay loads** (`0.45 ms` vs `2.05 ms` p50) than per-file atomic JSONL, with single-transaction atomicity across events, snapshots, and the `idempotency_key` ledger.
2. **Cold / Cross-Node Durability (`vfs -> Hanzo S3`)**: Flush compact JSONL/snapshot state (~`16 KiB`/run) to tenant-scoped Hanzo S3 (`orgs/<hash>/runs/<hash>/`) on `WorkflowSuspended` and `WorkflowCompleted` so any fresh stateless Visor worker can hydrate and resume on the next event trigger.
3. **Selective Durability Bypass for Trivial Read-Only 1-Step Calls**: Do not force per-step `synchronous=FULL` disk checkpoints on stateless single-step read-only inference calls (`+34.5 ms` / `5.05x` overhead); enable durable replay automatically for multi-step or side-effecting workflows carrying `idempotency_key`.
4. **Cross-Language Schema Parity**: Keep Python SDK `hanzo_agents.agent_workflow` and Go `control-plane/internal/durable` locked to the identical `sha256(org:run_id:round:step_id)` key derivation and fail-closed `NonDeterministicReplayError` contract.
