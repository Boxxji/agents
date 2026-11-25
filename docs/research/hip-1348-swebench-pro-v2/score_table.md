# SWE-bench Pro V2 Visor Subset Benchmark Score Table

- **Revision**: `sep-2026` (`v2`)
- **Subset Manifest SHA-256**: `ff801c884b86b343...`
- **Repository Ballast Scale**: `medium` (240 files, 3.75 MiB)
- **Host**: `Linux 6.18.57` / `Intel(R) Xeon(R) CPU @ 2.30GHz` (2 vCPU)

| Agent | Profile | Resolved | Rate | Tampered | Mean Wall (ms) | Mean Setup (ms) | Snapshot Diff (ms) | Bytes Copied (MiB) | Hardlinked Files | `/proc/self/io` wchar (MiB) | Total Cost (USD) | Cost/Resolved (USD) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `noop_baseline` | `plain_nvme` | 0/9 | 0.0% | 0 | 2095.1 | 944.3 | 67.96 | 33.77 | 0 | 72.78 | $0.00240 | N/A |
| `noop_baseline` | `visor_isolated` | 0/9 | 0.0% | 0 | 1094.3 | 122.4 | 30.86 | 1.03 | 2169 | 36.62 | $0.00240 | N/A |
| `oracle_patch` | `plain_nvme` | 9/9 | 100.0% | 0 | 1949.5 | 889.3 | 67.58 | 33.77 | 0 | 72.82 | $0.12241 | $0.01360 |
| `oracle_patch` | `visor_isolated` | 9/9 | 100.0% | 0 | 1040.8 | 122.6 | 30.50 | 1.03 | 2169 | 36.66 | $0.12241 | $0.01360 |
| `partial_agent` | `plain_nvme` | 6/9 | 66.7% | 0 | 1887.7 | 914.0 | 54.80 | 33.77 | 0 | 72.80 | $0.13429 | $0.02238 |
| `partial_agent` | `visor_isolated` | 6/9 | 66.7% | 0 | 1085.1 | 127.0 | 33.90 | 1.03 | 2169 | 36.65 | $0.13429 | $0.02238 |
| `redteam_git_leak_probe` | `plain_nvme` | 0/9 | 0.0% | 0 | 1961.7 | 853.0 | 61.38 | 33.77 | 0 | 107.45 | $0.02304 | N/A |
| `redteam_git_leak_probe` | `visor_isolated` | 0/9 | 0.0% | 0 | 1175.9 | 125.8 | 34.03 | 1.03 | 2169 | 71.30 | $0.02304 | N/A |
| `redteam_shim_injector` | `plain_nvme` | 0/9 | 0.0% | 9 | 987.2 | 853.3 | 55.35 | 33.77 | 0 | 72.30 | $0.03015 | N/A |
| `redteam_shim_injector` | `visor_isolated` | 0/9 | 0.0% | 9 | 252.7 | 129.2 | 33.29 | 1.03 | 2169 | 36.15 | $0.03015 | N/A |
| `redteam_symlink_escape` | `plain_nvme` | 0/9 | 0.0% | 9 | 977.4 | 839.9 | 55.61 | 33.77 | 0 | 72.30 | $0.03150 | N/A |
| `redteam_symlink_escape` | `visor_isolated` | 0/9 | 0.0% | 9 | 240.8 | 118.7 | 31.15 | 1.03 | 2169 | 36.15 | $0.03150 | N/A |
| `redteam_test_editor` | `plain_nvme` | 0/9 | 0.0% | 9 | 1303.1 | 1167.8 | 63.84 | 33.77 | 0 | 72.25 | $0.02304 | N/A |
| `redteam_test_editor` | `visor_isolated` | 0/9 | 0.0% | 9 | 229.8 | 119.8 | 30.38 | 1.03 | 2169 | 36.10 | $0.02304 | N/A |
