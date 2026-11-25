# HIP-1348 Research Report: Long-Horizon Coding-Agent Evaluation on Visor (SWE-bench Pro V2)

- **Issue**: [`hanzoai/agents#77`](https://github.com/hanzoai/agents/issues/77)
- **HIP**: HIP-1348 (draft) — Reproducible Coding-Agent Evaluation on Visor
- **Primary Reference**: SWE-bench Pro V2 — Scale Labs Leaderboard ([`swe_bench_pro_public_v2`](https://labs.scale.com/leaderboard/swe_bench_pro_public_v2?tab=full)), Release Blog ([2026-09-22](https://labs.scale.com/blog/swe-bench-pro-v2)), Paper ([Deng et al., arXiv:2509.16941v2, CC BY 4.0](https://arxiv.org/abs/2509.16941v2)), Reference Harness ([`scaleapi/SWE-bench_Pro-os`](https://github.com/scaleapi/SWE-bench_Pro-os))
- **Revision Evaluated**: `sep-2026` (`v2`, released 2026-09-22; legacy V1 manifests rejected by default)
- **Raw Data & Tables**: [`raw_results.json`](./raw_results.json) | [`score_table.md`](./score_table.md)

---

## 1. Primary Reference Summary & License Audit

### 1.1 SWE-bench Pro V2 Overview
SWE-bench Pro targets realistic, long-horizon, multi-file software engineering tasks (averaging 107.4 changed lines across 4.1 files per reference solution in the paper, vs single-file patches in earlier SWE-bench splits). On **2026-09-22**, Scale Labs shipped **SWE-bench Pro V2** (`revision: sep-2026`), refreshing the public set from the original 731 V1 instances down to **642 validated tasks** in Harbor format across 11 repositories and 4 languages:

| Language | V2 Public Tasks | Share |
| --- | ---: | ---: |
| Go | 256 | 39.9% |
| Python | 237 | 36.9% |
| JavaScript | 145 | 22.6% |
| TypeScript | 4 | 0.6% |
| **Total V2 Public** | **642** | **100.0%** |
| **V2 HARD Subset** | **51** | **7.9%** |

The full benchmark additionally maintains private held-out (858 tasks) and commercial startup (276 tasks) splits. Published leaderboard scores are Scale Labs' own measurements and are **not** Hanzo claims.

### 1.2 License Audit (Clean-Room Clearance)

| Artifact | License | Vendored in `hanzoai/agents`? | Audit Decision |
| --- | --- | --- | --- |
| SWE-bench Pro paper (`arXiv:2509.16941v2`) | CC BY 4.0 | No (cited only) | Safe to cite methodology and schema |
| SWE-bench Pro V2 public task repositories (11 OSS repos) | **Strong copyleft (GPL / AGPL)** by anti-contamination design | **No (zero GPL code copied)** | **Must NOT be vendored** into Apache-2.0 `hanzoai/agents`; loaded only externally at runtime via `load_external_manifest()` |
| SWE-bench Pro commercial/held-out tasks | Proprietary / restricted | **No** | Never accessed or copied |
| `sdk/python/benchmarks/swebench_pro/` runner & 9-task clean-room suite | **Apache-2.0** (original clean-room Python + stdlib + existing `hanzo_agents` deps) | Yes | Cleared for Apache-2.0 merge |

No third-party GPL code, proprietary repository snippets, or upstream gold/test patches are included in this repository. External SWE-bench Pro V2 Harbor JSONL manifests are ingested at runtime via `load_external_manifest(path, require_v2=True)`, which enforces `version == "v2"` and `revision == "sep-2026"` so stale V1 results cannot silently contaminate evaluations.

---

## 2. Canonical Architecture Alignment & Experimental Design

This runner stays strictly within Hanzo's canonical architecture without introducing any parallel runtime or orchestrator:

- **Storage path (`vfs -> Hanzo S3 -> Visor`)**: `SandboxManager` models two real local Linux storage profiles on identical tasks:
  1. `plain_nvme`: full per-task repository materialization + git initialization + anti-leakage scrub on local NVMe-style storage (zero inode sharing).
  2. `visor_isolated`: content-addressed read-only base image store (emulating `vfs -> Hanzo S3 -> Visor` immutable layers) built and scrubbed once per `(instance_id, repo_scale)`, with per-task sandboxes mounting immutable repository ballast and protected evaluation tests via read-only `0o444` hardlinks (CoW zero-copy) and isolating mutable package sources and `.git` metadata in a per-task writable upper layer.
- **Model routing & cost (`Enso` / `Zen` / Hanzo `AIConfig` + `llm_adapters`)**: `CostTracker` routes LLM parameters through `hanzo_agents.types.AIConfig.get_llm_params()` and `hanzo_agents.llm_adapters.apply_provider_patches()`, strips API keys from telemetry snapshots, and enforces `max_cost_per_call` / `daily_budget`.
- **Control-plane alignment**: Mirrors `control-plane/internal/eval/swebench/` (`sep-2026` / `v2` revision, `ai:inference`-only credential scope enforcement, and git workspace sanitization).

### 2.1 Anti-Leakage & Anti-Tampering Controls (Tested Fail-Closed)

Before every task rollout, the harness intentionally constructs a realistic unscrubbed upstream repository containing a future reference fix commit, `v-gold-fix` tag, `upstream-gold-fix` branch, `origin` remote, reflogs, and a `gold_patch.diff` artifact, then executes:

1. **Git history / reflog / remote / object purge (`scrub_git_workspace`)**: detaches `HEAD` at `base_commit`, deletes all remotes/tags/branches/reflogs/transient state files, and runs `git repack -Ad` + `git prune --expire=now` + `git gc --prune=now` so unreachable gold commits are physically purged from `.git/objects`.
2. **Pre-execution leakage audit (`verify_no_leakage`)**: fails closed (`status=leakage_blocked`, `resolved=False`) if any remote, tag, extra branch, reflog, future commit, gold object (`git cat-file -e`), sentinel marker in `.git/objects` or workspace, leaked env var (`GOLD_PATCH`, `AWS_SECRET_ACCESS_KEY`, etc.), or non-`ai:inference` token scope is present.
3. **Protected test integrity (`snapshot_protected_hashes`)**: hashes all evaluation test files (`SHA-256`) pre-run and enforces `0o444` read-only permissions under `visor_isolated`.
4. **Post-execution anti-tampering audit (`verify_post_execution_integrity`)**: runs **before** `pytest` grading and fails closed (`tampered=True`, `resolved=False`) upon any protected test hash mismatch/deletion, `git diff` touching protected paths, symlink escaping `os.path.realpath(workspace)` or targeting `.git`, unauthorized `sitecustomize.py`/`usercustomize.py`/`pytest.py`/`conftest.py`/`_pytest/` injection, or tampered env overrides (`PYTHONPATH`, `PYTEST_ADDOPTS`, `LD_PRELOAD`).

---

## 3. Falsifiable Hypotheses & Expected Failure Modes

- **H1 (Resolution Parity)**: Moving execution from `plain_nvme` to `visor_isolated` changes task resolution rate by **0.0 percentage points** across all benign agents (`oracle_patch`, `partial_agent`, `noop_baseline`).
  - *Falsification criterion*: Any task passing on `plain_nvme` fails on `visor_isolated` (or vice versa) due to CoW/read-only permission artifacts.
- **H2 (I/O & Setup Speedup at Realistic Repo Scale)**: On multi-megabyte repositories (`medium`: 240 files / 3.75 MiB per repo) with warm Visor CAS base images, `visor_isolated` reduces workspace bytes copied by `>= 90%`, setup latency by `>= 5x`, snapshot diff latency by `>= 1.5x`, and Linux `/proc/self/io` `wchar` by `>= 40%`.
  - *Falsification criterion*: Warm `visor_isolated` fails to reduce copied bytes or `/proc/self/io` write volume relative to `plain_nvme`.
- **H3 (Honest Negative / Crossover Regime)**: On tiny single-shot repositories (`tiny`: 8 files / 0.02 MiB) with a **cold** Visor CAS cache, `visor_isolated` will perform **worse** than `plain_nvme` on setup latency and bytes written due to CAS manifest indexing + `.git` metadata duplication overhead.
  - *Expected failure mode*: Cold-start single-rollout evaluations on tiny repos pay fixed CAS materialization cost without amortization.
- **H4 (Fail-Closed Security Enforcement)**: All four red-team exploit classes (git history/object harvesting, protected test rewriting, out-of-sandbox symlink escape even alongside a valid patch, and `sitecustomize.py`/`pytest.py`/`PYTEST_ADDOPTS` shim injection) achieve **0.0% resolution** and **100% tamper/leak containment**.

---

## 4. Reproducible Measurements

### 4.1 Environment & Commands

- **Host Hardware**: Linux `6.18.57` (`x86_64`), Intel Xeon CPU @ 2.30 GHz (2 vCPU), Python `3.12.16`, `/proc/self/io` kernel telemetry enabled.
- **Subset**: 9 clean-room multi-file Python tasks (`manifest_digest: ff801c884b86b343...`), each with explicit `fail_to_pass` and `pass_to_pass` pytest suites.
- **Model Cost Hook**: `AIConfig(model="openai/gpt-4o", temperature=0.0, max_tokens=4096)` routed via `apply_provider_patches`.

Reproduction commands (from `sdk/python/`):

```bash
# Primary experiment (medium repo scale, warm Visor CAS, sequential for clean /proc/self/io)
python3 -m benchmarks.swebench_pro \
  --repo-scale medium \
  --agents oracle partial noop redteam \
  --warm-visor-cas \
  --concurrency 1 \
  --output-dir ../../docs/research/hip-1348-swebench-pro-v2

# Cold-start tiny-repo negative-regime experiment
python3 -m benchmarks.swebench_pro \
  --repo-scale tiny \
  --agents oracle \
  --concurrency 1
```

### 4.2 Primary Results (`medium` Scale: 240 Files, 3.75 MiB/Repo, Warm Visor CAS)

| Agent | Profile | Resolved | Rate | Tampered | Mean Wall (ms) | Mean Setup (ms) | Snapshot Diff (ms) | Bytes Copied (MiB) | Hardlinked Files | `/proc/self/io` wchar (MiB) | Total Cost (USD) | Cost/Resolved (USD) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `oracle_patch` | `plain_nvme` | 9/9 | 100.0% | 0 | 1949.5 | 889.3 | 67.58 | 33.77 | 0 | 72.82 | \$0.12241 | \$0.01360 |
| `oracle_patch` | `visor_isolated` | 9/9 | 100.0% | 0 | **1040.8** | **122.6** | **30.50** | **1.03** | **2169** | **36.66** | \$0.12241 | \$0.01360 |
| `partial_agent` | `plain_nvme` | 6/9 | 66.7% | 0 | 1887.7 | 914.0 | 54.80 | 33.77 | 0 | 72.80 | \$0.13429 | \$0.02238 |
| `partial_agent` | `visor_isolated` | 6/9 | 66.7% | 0 | **1085.1** | **127.0** | **33.90** | **1.03** | **2169** | **36.65** | \$0.13429 | \$0.02238 |
| `noop_baseline` | `plain_nvme` | 0/9 | 0.0% | 0 | 2095.1 | 944.3 | 67.96 | 33.77 | 0 | 72.78 | \$0.00240 | N/A |
| `noop_baseline` | `visor_isolated` | 0/9 | 0.0% | 0 | **1094.3** | **122.4** | **30.86** | **1.03** | **2169** | **36.62** | \$0.00240 | N/A |
| `redteam_git_leak_probe` | `plain_nvme` | 0/9 | 0.0% | 0 | 1961.7 | 853.0 | 61.38 | 33.77 | 0 | 107.45 | \$0.02304 | N/A |
| `redteam_git_leak_probe` | `visor_isolated` | 0/9 | 0.0% | 0 | 1175.9 | 125.8 | 34.03 | 1.03 | 2169 | 71.30 | \$0.02304 | N/A |
| `redteam_test_editor` | `plain_nvme` | 0/9 | 0.0% | **9** | 1303.1 | 1167.8 | 63.84 | 33.77 | 0 | 72.25 | \$0.02304 | N/A |
| `redteam_test_editor` | `visor_isolated` | 0/9 | 0.0% | **9** | **229.8** | **119.8** | **30.38** | **1.03** | **2169** | **36.10** | \$0.02304 | N/A |
| `redteam_symlink_escape` | `plain_nvme` | 0/9 | 0.0% | **9** | 977.4 | 839.9 | 55.61 | 33.77 | 0 | 72.30 | \$0.03150 | N/A |
| `redteam_symlink_escape` | `visor_isolated` | 0/9 | 0.0% | **9** | **240.8** | **118.7** | **31.15** | **1.03** | **2169** | **36.15** | \$0.03150 | N/A |
| `redteam_shim_injector` | `plain_nvme` | 0/9 | 0.0% | **9** | 987.2 | 853.3 | 55.35 | 33.77 | 0 | 72.30 | \$0.03015 | N/A |
| `redteam_shim_injector` | `visor_isolated` | 0/9 | 0.0% | **9** | **252.7** | **129.2** | **33.29** | **1.03** | **2169** | **36.15** | \$0.03015 | N/A |

#### Key Quantitative Deltas (`oracle_patch`, `medium` scale)
- **Resolution parity (H1 confirmed)**: `9/9 (100.0%)` vs `9/9 (100.0%)`; `partial_agent` `6/9 (66.7%)` vs `6/9 (66.7%)`.
- **Workspace setup latency (H2 confirmed)**: `889.3 ms -> 122.6 ms` (**7.25x faster**, `-86.2%`).
- **End-to-end wall-clock time**: `1949.5 ms -> 1040.8 ms` (**1.87x faster**, `-46.6%`).
- **Snapshot diff extraction**: `67.58 ms -> 30.50 ms` (**2.22x faster**, `-54.9%`) via CoW hardlink inode skip (`st_nlink > 1`).
- **Physical workspace bytes copied (9 tasks)**: `33.77 MiB -> 1.03 MiB` (**32.8x reduction**, `-96.95%`).
- **Kernel `/proc/self/io` `wchar` (9 tasks)**: `72.82 MiB -> 36.66 MiB` (**-49.7%** logical write traffic; remaining ~36 MiB is `.git` metadata + pytest subprocess I/O).
- **Red-team containment (H4 confirmed)**: `27/27` tamper attempts (`redteam_test_editor`, `redteam_symlink_escape`, `redteam_shim_injector`) rejected fail-closed (`tampered=True, resolved=False`), and `0/9` leaks discovered by `redteam_git_leak_probe`.

### 4.3 Negative / Crossover Regime (`tiny` Scale: 8 Files, 0.02 MiB/Repo, Cold Visor CAS)

| Agent | Profile | Resolved | Mean Wall (ms) | Mean Setup (ms) | Snapshot Diff (ms) | Bytes Copied (MiB) | `/proc/self/io` wchar (MiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `oracle_patch` | `plain_nvme` | 9/9 | 1111.3 | **237.9** | 2.59 | **0.16** | **1.40** |
| `oracle_patch` | `visor_isolated` (cold CAS) | 9/9 | 1098.1 | 267.5 (**+12.4%**) | 2.00 | 0.46 (**+187.5%**) | 1.90 (**+35.7%**) |

**H3 confirmed (honest limitation)**: When running a single cold evaluation on a tiny repository (`< 100 KiB`), building the immutable CAS snapshot + duplicating `.git` metadata costs **+12.4% setup latency** and **+35.7% kernel write bytes** compared to plain NVMe. Visor CoW wins decisively only once repositories exceed ~1 MiB or when multiple agents/rollouts share a warm CAS base image (the standard SWE-bench Pro V2 regime, where real repositories are 50 MiB–2 GiB and multiple models/seeds run against the same base commits).

---

## 5. Correctness, Concurrency, Security & Regression Evidence

All verification tests live in [`sdk/python/tests/test_swebench_pro_runner.py`](../../../sdk/python/tests/test_swebench_pro_runner.py):

- **Correctness (4 tests)**: Full 9-task oracle resolution on both profiles, `noop_baseline` confirming all `fail_to_pass` fail and all `pass_to_pass` pass at base commit, `partial_agent` 6/9 split, and malformed diff rejection via `git apply --check`.
- **Anti-leakage red-team (4 tests)**: Physical `.git/objects` purge verification (`git cat-file -e` fails on `gold_commit_sha`), `AdversarialGitLeakAgent` finding zero secrets, fail-closed `LEAKAGE_BLOCKED` on unscrubbed workspaces, and `ENV_LEAK_VAR` / `CREDENTIAL_SCOPE_VIOLATION` enforcement.
- **Anti-tampering red-team (5 tests)**: Test-file modification detection, symlink escape rejection even alongside a valid passing patch, `sitecustomize.py` + `pytest.py` + `PYTEST_ADDOPTS` shim detection, deleted test detection, and `0o444` read-only + CAS immutability verification.
- **Telemetry, Hanzo SDK hooks & V2 validation (4 tests)**: Real `/proc/self/io` + CoW savings assertions, `AIConfig` + `apply_provider_patches` cost & budget verification, external V2 manifest loader rejecting legacy V1, and deterministic `manifest_digest`.
- **Concurrency & serialization (2 tests)**: Parallel `asyncio.gather` execution across 6 isolated Visor workspaces with canary files confirming zero cross-task contamination, plus JSON/Markdown report roundtrip.

---

## 6. Limitations & Integration Recommendation

### Limitations
1. **Userspace CoW emulation vs kernel OverlayFS/FUSE**: Local Linux tests without root emulate Visor's VFS->S3->CoW layer via read-only `0o444` hardlinks + writable upper-layer copy-up. Production Visor deployments on real VFS/S3 will additionally measure network/S3 cold-fetch tail latency (`p99`) on cache miss.
2. **Clean-room synthetic tasks vs full 642-task GPL suite**: Offline tests evaluate 9 clean-room Apache-2.0 Python tasks (2–3 files each); full leaderboard comparability requires mounting external SWE-bench Pro V2 Harbor images via `load_external_manifest()` on dedicated Visor worker hosts.
3. **Single-process `/proc/self/io` attribution**: Linux `/proc/self/io` aggregates counters for the runner process (and requires `--concurrency 1` for strict per-cohort attribution; high-concurrency runs should additionally read per-cgroup `io.stat` inside Visor).

### Integration Recommendation
**Adopt as the standard SWE-bench Pro V2 evaluation harness for Visor**, wired directly into the existing Go control-plane contract (`control-plane/internal/eval/swebench/`):
1. Pre-warm Visor's S3/VFS content-addressed base snapshots per `base_commit` once per V2 release (`sep-2026`) so every model rollout hits the warm CoW fast-path (`-86%` setup time, `-97%` workspace copy bytes).
2. Keep the fail-closed anti-leakage (`scrub_git_workspace` + `verify_no_leakage`) and pre-grading anti-tampering (`verify_post_execution_integrity`) gates mandatory before invoking any test runner.
3. Record per-cgroup `io.stat` alongside `/proc/self/io` when graduating from single-node SDK benchmarking to multi-tenant Visor clusters.
