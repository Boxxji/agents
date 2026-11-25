# HIP-1348 Research Report: Computer-Workflow Benchmark on Visor (OSWorld)

- **Issues**: [`hanzoai/agents#80`](https://github.com/hanzoai/agents/issues/80), [`hanzoai/agents#89`](https://github.com/hanzoai/agents/issues/89)
- **HIP**: HIP-1348 (draft) / HIP-1347 umbrella — Reproducible Computer-Workflow Evaluation on Visor
- **Primary Reference**: OSWorld — Xie et al., *"OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments"* (NeurIPS 2024 Datasets & Benchmarks, [arXiv:2404.07972](https://arxiv.org/abs/2404.07972), [os-world.github.io](https://os-world.github.io/), Reference Harness [`xlang-ai/OSWorld`](https://github.com/xlang-ai/OSWorld))
- **Raw Data & Tables**: [`raw_results.json`](./raw_results.json) | [`score_table.md`](./score_table.md)

---

## 1. Primary Reference Summary & License Audit

### 1.1 OSWorld Overview
OSWorld defines 369 real computer tasks spanning Ubuntu desktop applications and system workflows (`os`, `libreoffice_calc`, `libreoffice_writer`, `libreoffice_impress`, `chrome`, `vs_code`, `gimp`, `thunderbird`, `vlc`, and `multi_apps`), evaluated via execution-based post-task state checkers rather than string matching. Published leaderboard scores from OSWorld and subsequent papers are strictly the original authors' measurements in their VMware/Docker VM harness and are **not** Hanzo claims.

### 1.2 License & Dependency Audit (Clean-Room Clearance)

| Artifact | License | Vendored in `hanzoai/agents`? | Audit Decision |
| --- | --- | --- | --- |
| OSWorld paper (`arXiv:2404.07972`) | arXiv / academic citation | No (cited only) | Safe to reference taxonomy & evaluation methodology |
| Upstream harness & task JSON specs (`xlang-ai/OSWorld`) | **Apache-2.0** | **No** (loaded externally at runtime via `--external-osworld-dir`) | Compatible with Apache-2.0; external loader validates schemas and classifies GUI vs headless host compatibility transparently |
| Desktop VM base images (Ubuntu + third-party GUI apps such as Chrome/VLC/LibreOffice) | Mixed (GPL/LGPL/MPL/proprietary browser binaries) | **No (zero binaries or GPL assets vendored)** | Must remain external to Apache-2.0 `hanzoai/agents`; provisioned via Visor container/VM images when GUI stack is required |
| `sdk/python/benchmarks/osworld_visor/` runner & 10-task clean-room suite | **Apache-2.0** (original Python stdlib + existing `hanzo_agents` SDK hooks) | Yes | Cleared for Apache-2.0 merge; zero new runtime dependencies |

---

## 2. Canonical Architecture Alignment & Matched Baselines

This benchmark stays strictly within Hanzo's canonical stack (`vfs -> Hanzo S3 -> Visor`, Cloud owning lifecycle & billing, Enso routing models, Zen generating, Kai deciding) without introducing parallel runtimes:

1. **Matched Sandbox Profiles (`VisorSandboxManager`)**:
   - **`cold_sandbox` (baseline)**: Provisions a brand-new sandbox root (`workspace/`, `home/`, `tmp/`, `xdg_config/`, `xdg_cache/`), writes all desktop assets and task seed files from scratch (zero inode sharing), scrubs host secrets (`AWS_*`, `OPENAI_*`, `ANTHROPIC_*`, `GITHUB_*`, `PYTHONPATH`, `LD_PRELOAD`), and launches isolated subprocesses (`start_new_session=True`, optional `RLIMIT_FSIZE`).
   - **`warm_cas_visor` (Visor CoW model)**: Materializes immutable read-only desktop asset layers (`0o444`) once per canonical content digest inside a content-addressed store (emulating `vfs -> Hanzo S3 -> Visor` immutable layers), mounts immutable assets via zero-copy hardlinks, and **always detaches mutable task seed files as private writable per-task copies** so mutations never corrupt shared CAS templates.
2. **10 Real Multi-Step Computer-Workflow Tasks (`build_osworld_subset()`)**:
   - `osw_01_filesystem_batch_ops` (`os`): multi-directory organization, `.txt` whitespace normalization, `.tmp`/`.bak` deletion, dotfile preservation, SHA-256 manifest verification.
   - `osw_02_spreadsheet_csv_transform` (`libreoffice_calc`): CSV deduplication by timestamp, currency parsing, department tax rollups.
   - `osw_03_sqlite_data_pipeline` (`multi_apps`): SQLite schema creation, JSONL ingestion, malformed-row dead-letter quarantine, index & `service_sla` view export.
   - `osw_04_cli_git_automation` (`vs_code`): real `git` branch creation, cherry-pick of bugfix commit, `VERSION`/`CHANGELOG.md` bump, clean working tree, tag `v1.2.0`.
   - `osw_05_config_and_archive_ops` (`os`): INI config merge, secret redaction, gzipped tarball creation, SHA-256 verification.
   - `osw_06_document_markdown_report` (`libreoffice_writer`): multi-file meeting note extraction, KPI delta calculation, executive Markdown + JSON synthesis.
   - `osw_07_local_http_service_check` (`chrome`): ephemeral loopback `127.0.0.1:0` HTTP server launch, 3 live GET probes with SHA-256 + latency verification, and guaranteed server shutdown.
   - `osw_08_permissions_and_cleanup_audit` (`os`): POSIX permission hardening (`0o600`/`0o755`), script execution, stale cache purge, mode audit JSON.
   - `osw_09_log_triage_incident_report` (`multi_apps`): rotated log parsing, chronological error extraction, 5xx service & retry correlation.
   - `osw_10_json_yaml_pipeline_patch` (`vs_code`): multi-environment CI/config JSON migration + independent `validator.py` execution.
3. **100% Real OS Measurement Discipline (Zero Synthetic Offsets)**:
   - **Cold-start latency**: `provision_ms + subprocess_boot_ms` via `time.perf_counter_ns()` from directory creation through a real isolated Python subprocess readiness probe.
   - **Peak RSS (KiB)**: Real post-`execve` `/proc/<pid>/status` `VmHWM` guarded by `/proc/<pid>/cmdline != /proc/self/cmdline` so the runner's own pre-`exec` forked memory never inflates per-task child RSS (with automatic `os.wait4()` `ru_maxrss` fallback on macOS/non-procfs hosts).
   - **CPU & I/O**: Per-child `os.wait4()` `ru_utime`/`ru_stime`/`ru_inblock`/`ru_oublock` plus Linux `/proc/self/io` (`rchar`, `wchar`, `syscr`, `syscw`, `read_bytes`, `write_bytes`).
   - **Hanzo Model Accounting**: Routed via `hanzo_agents.types.AIConfig.get_llm_params()` and `hanzo_agents.llm_adapters.apply_provider_patches()` with API keys stripped and budget ceilings checked.
   - **Teardown & Security Audits**: Post-run verification of (a) zero orphan child PIDs (via PGID, `/proc/<pid>/environ`, and `/proc/<pid>/cwd`, with `SIGTERM` -> `SIGKILL` reaping), (b) zero leaked `TMPDIR` artifacts or lingering Unix/TCP sockets, (c) SHA-256 CAS template immutability, (d) complete workspace deletion, and (e) fail-closed rejection of `../` path traversal and out-of-sandbox symlinks.

---

## 3. Falsifiable Hypotheses & Expected Failure Modes

- **H1 (Success & Footprint Parity)**: Switching from `cold_sandbox` to `warm_cas_visor` preserves exact task success rates (`0.0 pp` delta across `oracle_agent`, `partial_agent`, `noop_agent`) and introduces `< 1%` peak RSS overhead.
  - *Falsification criterion*: Any task passing on `cold_sandbox` fails on `warm_cas_visor` due to read-only hardlink permissions or path resolution bugs, or peak RSS diverges by `>= 5%`.
- **H2 (Cold-Start & Write-Amplification Reduction at Realistic Asset Scale)**: With a warm Visor CAS pool at `medium` scale (160 read-only desktop assets, ~3.75 MiB per sandbox), `warm_cas_visor` cuts workspace copy bytes by `>= 95%`, `/proc/self/io` `wchar` by `>= 90%`, provisioning time by `>= 3x`, and total cold-start latency by `>= 1.8x`.
  - *Falsification criterion*: Warm CAS fails to beat `cold_sandbox` on provisioning latency or `/proc/self/io` `wchar`.
- **H3 (Honest Negative / Crossover Regime)**: On `tiny` single-shot tasks (4 assets / ~2 KiB) with a **cold** Visor CAS cache (`warm_visor_cas=False`), `warm_cas_visor` performs **worse** than `cold_sandbox` on provisioning latency, cold-start latency, copied bytes, and `/proc/self/io` `wchar` due to fixed CAS digesting and dual-layer materialization overhead.
  - *Expected failure mode*: Single-use tiny sandboxes do not amortize CAS snapshot creation.
- **H4 (Fail-Closed Cleanup & Escape Containment)**: Adversarial/buggy agents that leak orphan background PIDs + temp files (`redteam_orphan_leaker`) or attempt `../` writes + out-of-sandbox symlinks (`redteam_sandbox_escape`) achieve **0/10 (0.0%)** success, with **100%** detection and **100%** orphan PID reaping.

---

## 4. Reproducible Empirical Measurements

### 4.1 Host Environment & Reproduction Commands

- **Host Hardware**: Linux `6.18.57` (`x86_64`), Intel Xeon CPU @ 2.30 GHz (2 vCPU), Python `3.12.16`, `/proc/self/io` enabled.
- **Task Manifest Digest**: `99f92c4852d9e59f...` (canonical SHA-256 across all 10 task specs).
- **Detected Host Tooling**: `python3`, `git`, `sqlite3`, `tar`, `gzip`, `xvfb-run`, `playwright`, `docker` present; `libreoffice` absent -> headless isolated POSIX subprocess backend (`execution_backend="isolated_subprocess_posix"`).

Run from `sdk/python/`:

```bash
# Primary experiment: medium desktop asset scale (3.75 MiB/sandbox), warm Visor CAS
python3 -m benchmarks.osworld_visor \
  --workspace-scale medium \
  --agents oracle partial noop redteam \
  --warm-visor-cas \
  --concurrency 1 \
  --output-dir ../../docs/research/hip-1348-osworld-visor

# Negative / crossover regime: tiny single-shot tasks with cold Visor CAS
python3 -m benchmarks.osworld_visor \
  --workspace-scale tiny \
  --agents oracle \
  --concurrency 1
```

### 4.2 Primary Results (`medium` Scale: 160 Assets / 3.75 MiB per Sandbox, Warm Visor CAS)

| Agent | Profile | Success | Rate | Cleanup Clean | Escapes Blocked | Mean Cold-Start (ms) | p95 Cold-Start (ms) | Mean Provision (ms) | Mean Wall (ms) | Mean Peak RSS (KiB) | Bytes Copied (MiB) | Hardlinks | `/proc/self/io` wchar (MiB) | Cost (USD) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `oracle_agent` | `cold_sandbox` | 10/10 | 100.0% | 10/10 | 0 | 101.79 | 149.87 | 70.11 | 324.16 | 11473.6 | 37.53 | 0 | 37.59 | \$0.11034 |
| `oracle_agent` | `warm_cas_visor` | 10/10 | 100.0% | 10/10 | 0 | **49.47** | **71.69** | **15.87** | **294.72** | 11471.2 | **0.03** | **1600** | **0.08** | \$0.11034 |
| `partial_agent` | `cold_sandbox` | 6/10 | 60.0% | 10/10 | 0 | 109.91 | 185.61 | 74.16 | 262.11 | 9714.0 | 37.53 | 0 | 37.58 | \$0.10544 |
| `partial_agent` | `warm_cas_visor` | 6/10 | 60.0% | 10/10 | 0 | **51.05** | **77.71** | **17.57** | **227.54** | 9717.6 | **0.03** | **1600** | **0.08** | \$0.10544 |
| `noop_agent` | `cold_sandbox` | 0/10 | 0.0% | 10/10 | 0 | 149.16 | 229.38 | 101.98 | 265.21 | 7284.0 | 37.53 | 0 | 37.54 | \$0.01590 |
| `noop_agent` | `warm_cas_visor` | 0/10 | 0.0% | 10/10 | 0 | **52.52** | **92.28** | **18.67** | **165.68** | 7283.6 | **0.03** | **1600** | **0.04** | \$0.01590 |
| `redteam_orphan_leaker` | `cold_sandbox` | 0/10 | 0.0% | **0/10** | 0 | 96.53 | 152.18 | 65.93 | 387.71 | 11651.6 | 37.53 | 0 | 37.59 | \$0.07900 |
| `redteam_orphan_leaker` | `warm_cas_visor` | 0/10 | 0.0% | **0/10** | 0 | **51.37** | **78.51** | **15.86** | 402.25 | 11656.0 | **0.03** | **1600** | **0.09** | \$0.07900 |
| `redteam_sandbox_escape` | `cold_sandbox` | 0/10 | 0.0% | 10/10 | **10** | 102.17 | 150.86 | 71.10 | 389.02 | 11468.0 | 37.53 | 0 | 37.59 | \$0.07850 |
| `redteam_sandbox_escape` | `warm_cas_visor` | 0/10 | 0.0% | 10/10 | **10** | **47.98** | **57.76** | **16.55** | **346.37** | 11471.2 | **0.03** | **1600** | **0.08** | \$0.07850 |

#### Quantitative Summary (`oracle_agent`, `medium` scale)
- **H1 confirmed (Success & RSS parity)**: `10/10 (100.0%)` vs `10/10 (100.0%)`; `partial_agent` `6/10` vs `6/10`; `noop_agent` `0/10` vs `0/10`. Mean peak RSS (`proc_vmhwm`): `11,473.6 KiB` vs `11,471.2 KiB` (`-0.02%`, zero memory regression).
- **H2 confirmed (Cold-start & write-amplification wins)**:
  - Provisioning latency: `70.11 ms -> 15.87 ms` (**4.42x faster**, `-77.4%`).
  - Mean cold-start latency (provision + isolated subprocess readiness boot): `101.79 ms -> 49.47 ms` (**2.06x faster**, `-51.4%`).
  - p95 cold-start latency: `149.87 ms -> 71.69 ms` (**2.09x faster**, `-52.2%`).
  - Physical bytes copied (10 tasks): `37.53 MiB -> 0.03 MiB` (**-99.92%** via 1,600 zero-copy `0o444` hardlinks).
  - Linux `/proc/self/io` `wchar` (10 tasks): `37.59 MiB -> 0.08 MiB` (**-99.79%** logical write traffic).
- **H4 confirmed (Fail-closed cleanup & escape enforcement)**:
  - `redteam_orphan_leaker`: `20/20` rollouts across both profiles caught the lingering background sleeper PID and `tmp/leaked_artifact.tmp`, reaped `20/20` orphan PIDs cleanly, removed `20/20` sandboxes, and failed closed (`status=cleanup_violation`, `success=False`).
  - `redteam_sandbox_escape`: `20/20` rollouts caught both `../escaped_outside.txt` and `escape_canary_link -> ../host_guard/secret_canary.txt` fail-closed (`status=sandbox_escape_blocked`, `success=False`) even though the agent also executed the full oracle steps.

### 4.3 Honest Negative / Crossover Regime (`tiny` Scale: 4 Assets / 2 KiB per Sandbox, Cold Visor CAS)

| Agent | Profile | Success | Mean Cold-Start (ms) | p95 Cold-Start (ms) | Mean Provision (ms) | Mean Wall (ms) | Mean Peak RSS (KiB) | Bytes Copied (MiB) | `/proc/self/io` wchar (MiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `oracle_agent` | `cold_sandbox` | 10/10 | **40.67** | **91.23** | **8.70** | **238.33** | 11468.0 | **0.05** | **0.11** |
| `oracle_agent` | `warm_cas_visor` (cold CAS miss) | 10/10 | 43.92 (**+8.0%**) | 102.86 (**+12.7%**) | 11.86 (**+36.3%**) | 242.49 (**+1.7%**) | 11467.6 | 0.09 (**+80.0%**) | 0.14 (**+27.3%**) |

**H3 confirmed**: When a sandbox carries only ~2 KiB of assets and hits a **cold** Visor CAS cache on a single-shot run, building the CAS snapshot + detached seed layer adds **+36.3% provisioning latency**, **+8.0% cold-start latency**, and **+27.3% `/proc/self/io` write traffic** compared to `cold_sandbox`. Visor CAS should therefore be enabled for multi-megabyte desktop/repo images or multi-rollout benchmark sweeps where base layers are pre-warmed or reused across runs, and bypassed for ephemeral sub-100 KiB one-shot tasks.

---

## 5. Correctness, Concurrency, Security & Regression Evidence

All 15 automated tests live in [`sdk/python/tests/test_osworld_visor_benchmark.py`](../../../sdk/python/tests/test_osworld_visor_benchmark.py):

- **Correctness & Real Telemetry (3 tests)**: Full 10/10 oracle success on both profiles with non-zero real OS `cold_start_ms`, `subprocess_boot_ms`, `proc_vmhwm` RSS, and `/proc/self/io` assertions; 6/10 `partial_agent` and 0/10 `noop_agent` parity; cold-cache tiny-workspace negative crossover regime verification.
- **CAS Isolation & Integrity (2 tests)**: Detached writable copies proving run-1 file mutations/deletions never corrupt run-2 seed files; direct CAS `ro_layer` tamper detection failing closed (`cas_template_intact=False`).
- **Cleanup & Teardown Correctness (2 tests)**: `AdversarialOrphanLeakerAgent` orphan PID detection, `SIGTERM`/`SIGKILL` reaping verified dead via `os.kill(pid, 0)`, and `TMPDIR` leak detection; lingering loopback TCP socket detection in `CleanupAuditor`.
- **Security & Containment (3 tests)**: `AdversarialSandboxEscapeAgent` fail-closed rejection of `../` writes and out-of-sandbox symlinks even alongside passing oracle steps; pre-launch rejection of `working_subdir="../host_guard"`; host credential scrubbing (`OPENAI_API_KEY`, `AWS_SECRET_ACCESS_KEY`, `GITHUB_TOKEN`, `PYTHONPATH`).
- **Portability & Concurrency (2 tests)**: Symlinked `TMPDIR`/`base_dir` parity (matching macOS `/var/folders -> /private/var/folders`); 4-way parallel `asyncio.gather` execution across 12 rollouts verifying zero cross-sandbox contamination and zero leftover run directories.
- **External Loader, Hanzo SDK Hooks & CLI (3 tests)**: Upstream `xlang-ai/OSWorld` Apache-2.0 JSON loader (schema validation, GUI vs headless classification, canonical SHA-256 `manifest_digest`); `AIConfig` + `apply_provider_patches` secret stripping & budget ceiling enforcement; end-to-end CLI JSON + Markdown artifact roundtrip.

---

## 6. Limitations & Integration Recommendation

### Limitations
1. **Headless POSIX subprocess sandbox vs full X11/VM desktop GUI**: This harness runs reproducible OS/file/CSV/SQLite/git/HTTP/config/security workflows in isolated POSIX subprocesses without requiring privileged VMware/KVM/Docker daemon access. Upstream OSWorld tasks targeting graphical desktop applications (`libreoffice_calc`, `gimp`, `vlc`, interactive `chrome`) are explicitly flagged `requires_gui=True` by `load_external_osworld_dir()` and require a Visor GUI container/VM image with Xvfb/desktop apps installed.
2. **Userspace hardlink CoW vs kernel OverlayFS/FUSE/cgroup isolation**: Unprivileged Linux hosts emulate Visor's immutable VFS->S3 base layer via `0o444` hardlinks and detached writable copies. Production Visor workers on Linux namespaces + OverlayFS/FUSE + cgroups v2 enforce hard kernel-level UID/mount isolation (preventing same-UID `chmod` on lower layers) and report per-cgroup `memory.peak` and `io.stat` under high concurrency.
3. **Single-process `/proc/self/io` attribution**: Runner-side `/proc/self/io` aggregates counters across the Python runner process; strict per-cohort write-byte attribution uses `--concurrency 1`, while multi-tenant Visor clusters should additionally scrape per-sandbox cgroup `io.stat`.

### Integration Recommendation
**Adopt `sdk/python/benchmarks/osworld_visor/` as Hanzo's reproducible computer-workflow evaluation harness for Visor (HIP-1348)**:
1. **Pre-warm Visor's `vfs -> Hanzo S3 -> Visor` CAS layers** for multi-megabyte desktop/app snapshots (`-51.4%` cold-start latency, `-77.4%` provisioning latency, `-99.8%` write amplification at 3.75 MiB scale, scaling linearly with real multi-gigabyte desktop images), while routing sub-100 KiB single-shot tasks directly to ephemeral local scratch.
2. **Enforce mandatory `CleanupAuditor` + `SecurityAuditor` gates** in Visor's teardown lifecycle so orphan process groups, lingering loopback listeners, leaked `TMPDIR` files, and symlink/traversal escapes are always detected, reaped, and failed closed.
3. **Wire `--external-osworld-dir`** into scheduled Visor GUI worker runs to evaluate upstream Apache-2.0 `xlang-ai/OSWorld` JSON tasks alongside this fast, zero-dependency 10-task headless regression gate in CI.
