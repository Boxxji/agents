# SWE-bench Pro V2 Visor Subset Benchmark Score Table

- **Revision**: `sep-2026` (`v2`)
- **Subset Manifest SHA-256**: `4ba05d9db01a238c...`
- **Repository Ballast Scale**: `medium` (240 files, 3.75 MiB)
- **Host**: `Linux 6.18.57` / `Intel(R) Xeon(R) CPU @ 2.30GHz` (2 vCPU)

| Agent | Profile | Resolved | Rate | Tampered | Mean Wall (ms) | Mean Setup (ms) | Snapshot Diff (ms) | Bytes Copied (MiB) | Hardlinked Files | `/proc/self/io` wchar (MiB) | Total Cost (USD) | Cost/Resolved (USD) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `noop_baseline` | `plain_nvme` | 0/9 | 0.0% | 0 | 2005.6 | 854.3 | 58.41 | 33.77 | 0 | 72.79 | $0.00240 | N/A |
| `noop_baseline` | `visor_isolated` | 0/9 | 0.0% | 0 | 1320.4 | 172.2 | 34.29 | 1.04 | 2160 | 36.65 | $0.00240 | N/A |
| `oracle_patch` | `plain_nvme` | 9/9 | 100.0% | 0 | 2168.6 | 964.8 | 73.73 | 33.77 | 0 | 72.82 | $0.12241 | $0.01360 |
| `oracle_patch` | `visor_isolated` | 9/9 | 100.0% | 0 | 1252.6 | 164.1 | 33.93 | 1.04 | 2160 | 36.68 | $0.12241 | $0.01360 |
| `partial_agent` | `plain_nvme` | 6/9 | 66.7% | 0 | 2077.3 | 903.0 | 58.38 | 33.77 | 0 | 72.81 | $0.13429 | $0.02238 |
| `partial_agent` | `visor_isolated` | 6/9 | 66.7% | 0 | 1272.2 | 177.9 | 33.57 | 1.04 | 2160 | 36.67 | $0.13429 | $0.02238 |
| `redteam_git_leak_probe` | `plain_nvme` | 0/9 | 0.0% | 0 | 2238.1 | 979.5 | 60.74 | 33.77 | 0 | 107.47 | $0.02304 | N/A |
| `redteam_git_leak_probe` | `visor_isolated` | 0/9 | 0.0% | 0 | 1475.0 | 178.9 | 39.83 | 1.04 | 2160 | 71.33 | $0.02304 | N/A |
| `redteam_shim_injector` | `plain_nvme` | 0/9 | 0.0% | 9 | 996.1 | 871.0 | 56.50 | 33.77 | 0 | 72.30 | $0.03015 | N/A |
| `redteam_shim_injector` | `visor_isolated` | 0/9 | 0.0% | 9 | 295.9 | 162.1 | 34.35 | 1.04 | 2160 | 36.16 | $0.03015 | N/A |
| `redteam_symlink_escape` | `plain_nvme` | 0/9 | 0.0% | 9 | 983.1 | 852.0 | 62.32 | 33.77 | 0 | 72.31 | $0.03150 | N/A |
| `redteam_symlink_escape` | `visor_isolated` | 0/9 | 0.0% | 9 | 288.7 | 158.4 | 33.35 | 1.04 | 2160 | 36.16 | $0.03150 | N/A |
| `redteam_test_editor` | `plain_nvme` | 0/9 | 0.0% | 9 | 1003.0 | 886.8 | 57.40 | 33.77 | 0 | 72.25 | $0.02304 | N/A |
| `redteam_test_editor` | `visor_isolated` | 0/9 | 0.0% | 9 | 412.1 | 223.8 | 50.91 | 1.04 | 2160 | 36.11 | $0.02304 | N/A |
