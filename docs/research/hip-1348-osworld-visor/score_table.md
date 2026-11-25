# OSWorld Computer-Workflow Visor Subset — Score Table

- **HIP Reference**: HIP-1348 (hanzoai/agents#80, #89)
- **Manifest Digest**: `99f92c4852d9e59f...`
- **Workspace Scale**: `medium` | **Warm Visor CAS**: `True` | **Concurrency**: `1`
- **Host**: `Linux 6.18.57` (x86_64), Python `3.12.16`, `/proc/self/io`: `True`

| Agent | Profile | Success | Rate | Cleanup Clean | Escapes Blocked | Mean Cold-Start (ms) | p95 Cold-Start (ms) | Mean Provision (ms) | Mean Wall (ms) | Mean Peak RSS (KiB) | Bytes Copied (MiB) | Hardlinks | `/proc/self/io` wchar (MiB) | Cost (USD) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `oracle_agent` | `cold_sandbox` | 10/10 | 100.0% | 10/10 | 0 | 101.79 | 149.87 | 70.11 | 324.16 | 11473.6 | 37.53 | 0 | 37.59 | \$0.11034 |
| `oracle_agent` | `warm_cas_visor` | 10/10 | 100.0% | 10/10 | 0 | 49.47 | 71.69 | 15.87 | 294.72 | 11471.2 | 0.03 | 1600 | 0.08 | \$0.11034 |
| `partial_agent` | `cold_sandbox` | 6/10 | 60.0% | 10/10 | 0 | 109.91 | 185.61 | 74.16 | 262.11 | 9714.0 | 37.53 | 0 | 37.58 | \$0.10544 |
| `partial_agent` | `warm_cas_visor` | 6/10 | 60.0% | 10/10 | 0 | 51.05 | 77.71 | 17.57 | 227.54 | 9717.6 | 0.03 | 1600 | 0.08 | \$0.10544 |
| `noop_agent` | `cold_sandbox` | 0/10 | 0.0% | 10/10 | 0 | 149.16 | 229.38 | 101.98 | 265.21 | 7284.0 | 37.53 | 0 | 37.54 | \$0.01590 |
| `noop_agent` | `warm_cas_visor` | 0/10 | 0.0% | 10/10 | 0 | 52.52 | 92.28 | 18.67 | 165.68 | 7283.6 | 0.03 | 1600 | 0.04 | \$0.01590 |
| `redteam_orphan_leaker` | `cold_sandbox` | 0/10 | 0.0% | 0/10 | 0 | 96.53 | 152.18 | 65.93 | 387.71 | 11651.6 | 37.53 | 0 | 37.59 | \$0.07900 |
| `redteam_orphan_leaker` | `warm_cas_visor` | 0/10 | 0.0% | 0/10 | 0 | 51.37 | 78.51 | 15.86 | 402.25 | 11656.0 | 0.03 | 1600 | 0.09 | \$0.07900 |
| `redteam_sandbox_escape` | `cold_sandbox` | 0/10 | 0.0% | 10/10 | 10 | 102.17 | 150.86 | 71.10 | 389.02 | 11468.0 | 37.53 | 0 | 37.59 | \$0.07850 |
| `redteam_sandbox_escape` | `warm_cas_visor` | 0/10 | 0.0% | 10/10 | 10 | 47.98 | 57.76 | 16.55 | 346.37 | 11471.2 | 0.03 | 1600 | 0.08 | \$0.07850 |
