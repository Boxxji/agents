# HIP-1347 SWE-bench Pro V2 Score Table (`hanzoai/agents#84`)

- **Revision**: `sep-2026` (`v2`)
- **Manifest SHA-256**: `7c343817b2378b94...`
- **Model**: `zen/zen-coder` | **Seed**: `42` | **Go**: `go1.26.5` (`linux/amd64`, 2 vCPU)

## 1. Main Benchmark (`medium`: 240 files / 3.75 MiB per repo, Warm VFS CAS vs Plain NVMe)

| Policy | Backend | Resolved | Pass@1 | Tampered | Wall p50 / p95 / p99 (ms) | Setup+Scrub (ms) | Test (ms) | Diff (ms) | Written (MiB) | Hardlinked | `/proc/self/io` wchar (MiB) | Total Cost (USD) | Cost/Resolved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `noop_baseline` | `nvme` | 0/8 | 0.0% | 0 | 1168 / 1535 / 1535 | 732.5 | 395.9 | 21.52 | 30.03 | 0 | 67.39 | \$0.000640 | N/A |
| `noop_baseline` | `vfs` | 0/8 | 0.0% | 0 | 548 / 637 / 637 | 145.3 | 380.6 | 3.70 | 0.87 | 1920 | 36.24 | \$0.000640 | N/A |
| `single_shot_baseline` | `nvme` | 4/8 | 50.0% | 0 | 1188 / 1247 / 1247 | 747.2 | 374.3 | 21.97 | 30.04 | 0 | 67.45 | \$0.018086 | \$0.004522 |
| `single_shot_baseline` | `vfs` | 4/8 | 50.0% | 0 | 595 / 636 / 636 | 152.6 | 393.3 | 2.19 | 0.88 | 1920 | 36.28 | \$0.018086 | \$0.004522 |
| `iterative_kai_gated` | `nvme` | 8/8 | 100.0% | 0 | 1763 / 2856 / 2856 | 941.5 | 489.5 | 36.18 | 30.04 | 0 | 69.27 | \$0.030730 | \$0.003841 |
| `iterative_kai_gated` | `vfs` | 8/8 | 100.0% | 0 | 1327 / 2234 / 2234 | 223.4 | 572.0 | 4.19 | 0.88 | 1920 | 38.10 | \$0.030730 | \$0.003841 |
| `budget_capped_iterative` | `nvme` | 4/8 | 50.0% | 0 | 1915 / 3087 / 3087 | 984.9 | 440.5 | 37.38 | 30.04 | 0 | 67.59 | \$0.030730 | \$0.007682 |
| `budget_capped_iterative` | `vfs` | 4/8 | 50.0% | 0 | 1142 / 1253 / 1253 | 163.7 | 403.3 | 2.99 | 0.88 | 1920 | 36.44 | \$0.030730 | \$0.007682 |
| `oracle_reference` | `nvme` | 8/8 | 100.0% | 0 | 1478 / 1696 / 1696 | 911.5 | 501.2 | 25.15 | 30.04 | 0 | 67.44 | \$0.012076 | \$0.001510 |
| `oracle_reference` | `vfs` | 8/8 | 100.0% | 0 | 725 / 861 / 861 | 171.0 | 532.5 | 2.89 | 0.88 | 1920 | 36.30 | \$0.012076 | \$0.001510 |
| `adversarial_leak_probe` | `nvme` | 0/8 | 0.0% | 0 | 1598 / 1862 / 1862 | 949.0 | 525.2 | 27.75 | 30.03 | 0 | 98.24 | \$0.006554 | N/A |
| `adversarial_leak_probe` | `vfs` | 0/8 | 0.0% | 0 | 894 / 992 / 992 | 187.3 | 579.0 | 3.41 | 0.87 | 1920 | 67.09 | \$0.006554 | N/A |
| `adversarial_tamper_agent` | `nvme` | 0/8 | 0.0% | 8 | 985 / 1105 / 1105 | 913.7 | 0.0 | 25.11 | 30.04 | 0 | 64.10 | \$0.009216 | N/A |
| `adversarial_tamper_agent` | `vfs` | 0/8 | 0.0% | 8 | 195 / 245 / 245 | 156.0 | 0.0 | 2.75 | 0.88 | 1920 | 32.92 | \$0.009216 | N/A |

## 2. Negative Regime A: Cold-Start Tiny Repo (`tiny`: 8 files / 16 KiB, Cold VFS CAS vs Plain NVMe)

| Policy | Backend | Resolved | Pass@1 | Tampered | Wall p50 / p95 / p99 (ms) | Setup+Scrub (ms) | Test (ms) | Diff (ms) | Written (MiB) | Hardlinked | `/proc/self/io` wchar (MiB) | Total Cost (USD) | Cost/Resolved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `oracle_reference` | `nvme` | 8/8 | 100.0% | 0 | 817 / 994 / 994 | 305.1 | 533.8 | 0.41 | 0.17 | 0 | 4.11 | \$0.012076 | \$0.001510 |
| `oracle_reference` | `vfs` | 8/8 | 100.0% | 0 | 869 / 941 / 941 | 361.8 | 492.9 | 0.34 | 0.43 | 64 | 4.82 | \$0.012076 | \$0.001510 |

## 3. Negative Regime B: Oversized Monorepo (`large`: 640 files / 20.0 MiB, Cold NVMe Git-Scrub vs Warm VFS CoW)

| Policy | Backend | Resolved | Pass@1 | Tampered | Wall p50 / p95 / p99 (ms) | Setup+Scrub (ms) | Test (ms) | Diff (ms) | Written (MiB) | Hardlinked | `/proc/self/io` wchar (MiB) | Total Cost (USD) | Cost/Resolved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `oracle_reference` | `nvme` | 4/4 | 100.0% | 0 | 2525 / 3053 / 3053 | 2035.0 | 378.3 | 100.25 | 80.02 | 0 | 168.04 | \$0.006807 | \$0.001702 |
| `oracle_reference` | `vfs` | 4/4 | 100.0% | 0 | 1248 / 1558 / 1558 | 596.9 | 516.1 | 6.79 | 1.09 | 2560 | 86.01 | \$0.006807 | \$0.001702 |

## 4. Adversarial Anti-Leakage & API-Key Scope Gate Verification

| Check | Attack Vector | Expected | Actual | Caught |
| --- | --- | --- | --- | --- |
| `unscrubbed_git_leakage_injection` | future commit + v-reference-fix tag + remote + gold_patch.diff + solution fingerprint | `leakage_blocked` | `leakage_blocked` | **PASS** |
| `excess_admin_scope` | HANZO_API_KEY with [ai:inference, admin:write] | `policy_blocked` | `policy_blocked` | **PASS** |
| `wildcard_scope_rejection` | HANZO_API_KEY with [ai:*] | `policy_blocked` | `policy_blocked` | **PASS** |
| `unscoped_token_rejection` | HANZO_API_KEY with empty scopes | `policy_blocked` | `policy_blocked` | **PASS** |
| `forbidden_secret_env_var` | GITHUB_TOKEN + AWS_SECRET_ACCESS_KEY present in sandbox env | `policy_blocked` | `policy_blocked` | **PASS** |
| `cross_tenant_isolation` | tenant-other attempting execution in tenant-sec sandbox | `policy_blocked` | `policy_blocked` | **PASS** |

