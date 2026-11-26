# HIP-1346 Score Table: Lost-in-the-Middle Context Packing (`hanzoai/agents#83`)

- **Seed**: `1346` | **Go**: `go1.26.5` (`linux/amd64`, `NumCPU=2`)
- **Evaluator**: `calibrated-tacl24-lost-in-the-middle-proxy` (`simulated=true`)
- **Token Budget (Budgeted Policies)**: `1400` tokens | **Enso Pricing**: `$3.00 / 1M` input tokens | **Kai Overhead**: `$0.02 / 1k` calls (`GenerativeOutputTokens = 0`)
- **Corpus**: `100` primary cases (`60` position sweep + `24` multi-evidence synthesis + `16` security/ACL/injection) + `69` negative-regime cases

## 1. Overall Primary Suite Comparison (100 Cases)

| Policy | Exact Pass | Calibrated Acc | All-Gold Retained | Gold in Edge % | Avg Tokens | Token Savings | Cost USD/1k | Acc / 1k Tok | Leaks | Hijacks | Kai Gen Tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `raw_concat_baseline` | 41.0% | 66.4% | 100.0% | 45.0% | 4164 | 0.0% | $12.4929 | 0.159 | 0 | 0 | 0 |
| `relevance_desc` | 88.0% | 85.3% | 100.0% | 96.0% | 4164 | 0.0% | $12.4929 | 0.205 | 0 | 0 | 0 |
| `u_shaped_edge_packing` | 92.0% | 86.6% | 100.0% | 100.0% | 4164 | 0.0% | $12.4929 | 0.208 | 0 | 0 | 0 |
| `budgeted_top_k_prune` | 88.0% | 80.9% | 88.0% | 76.0% | 1271 | 69.5% | $3.8122 | 0.636 | 0 | 0 | 0 |
| `kai_evidence_edge_pack` | 100.0% | 91.8% | 100.0% | 100.0% | 1240 | 70.2% | $3.7403 | 0.741 | 0 | 0 | 0 |

## 2. Single-Evidence Position Curve (`K ∈ {5, 10, 20, 30}` x 5 Canonical Slots)

Calibrated accuracy (`ExactPass%`) across gold evidence slots `start (0)`, `q1 (K/4)`, `middle (K/2)`, `q3 (3K/4)`, `end (K-1)`:

| Policy | K | Start (`0`) | Q1 (`K/4`) | Middle (`K/2`) | Q3 (`3K/4`) | End (`K-1`) | Middle Drop vs Start | Avg Input Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `raw_concat_baseline` | 5 | 94.0% (100%) | 84.0% (100%) | 77.7% (100%) | 82.5% (100%) | 91.0% (100%) | -16.3 pp | 1172 |
| `raw_concat_baseline` | 10 | 91.9% (100%) | 80.0% (100%) | 70.2% (33%) | 78.4% (100%) | 89.0% (100%) | -21.8 pp | 2337 |
| `raw_concat_baseline` | 20 | 87.9% (100%) | 66.1% (0%) | 54.4% (0%) | 69.1% (33%) | 85.0% (100%) | -33.4 pp | 4655 |
| `raw_concat_baseline` | 30 | 85.0% (100%) | 58.0% (0%) | 40.0% (0%) | 56.5% (0%) | 82.0% (100%) | -45.0 pp | 6991 |
| `relevance_desc` | 5 | 94.0% (100%) | 94.0% (100%) | 94.0% (100%) | 94.0% (100%) | 94.0% (100%) | -0.0 pp | 1172 |
| `relevance_desc` | 10 | 91.9% (100%) | 92.0% (100%) | 91.9% (100%) | 92.0% (100%) | 92.0% (100%) | -0.0 pp | 2337 |
| `relevance_desc` | 20 | 87.9% (100%) | 88.1% (100%) | 87.9% (100%) | 88.1% (100%) | 88.0% (100%) | -0.0 pp | 4655 |
| `relevance_desc` | 30 | 85.0% (100%) | 85.0% (100%) | 85.0% (100%) | 85.0% (100%) | 85.0% (100%) | -0.0 pp | 6991 |
| `u_shaped_edge_packing` | 5 | 94.0% (100%) | 94.0% (100%) | 94.0% (100%) | 94.0% (100%) | 94.0% (100%) | -0.0 pp | 1172 |
| `u_shaped_edge_packing` | 10 | 91.9% (100%) | 92.0% (100%) | 91.9% (100%) | 92.0% (100%) | 92.0% (100%) | -0.0 pp | 2337 |
| `u_shaped_edge_packing` | 20 | 87.9% (100%) | 88.1% (100%) | 87.9% (100%) | 88.1% (100%) | 88.0% (100%) | -0.0 pp | 4655 |
| `u_shaped_edge_packing` | 30 | 85.0% (100%) | 85.0% (100%) | 85.0% (100%) | 85.0% (100%) | 85.0% (100%) | -0.0 pp | 6991 |
| `budgeted_top_k_prune` | 5 | 94.0% (100%) | 94.0% (100%) | 94.0% (100%) | 94.0% (100%) | 94.0% (100%) | -0.0 pp | 1172 |
| `budgeted_top_k_prune` | 10 | 93.8% (100%) | 93.6% (100%) | 93.9% (100%) | 93.6% (100%) | 93.6% (100%) | +0.1 pp | 1323 |
| `budgeted_top_k_prune` | 20 | 93.9% (100%) | 93.6% (100%) | 93.9% (100%) | 93.6% (100%) | 93.7% (100%) | -0.0 pp | 1301 |
| `budgeted_top_k_prune` | 30 | 93.8% (100%) | 93.6% (100%) | 93.8% (100%) | 93.7% (100%) | 93.8% (100%) | -0.0 pp | 1300 |
| `kai_evidence_edge_pack` | 5 | 94.0% (100%) | 94.0% (100%) | 94.0% (100%) | 94.0% (100%) | 94.0% (100%) | -0.0 pp | 1172 |
| `kai_evidence_edge_pack` | 10 | 93.9% (100%) | 93.6% (100%) | 93.9% (100%) | 93.7% (100%) | 93.7% (100%) | -0.0 pp | 1283 |
| `kai_evidence_edge_pack` | 20 | 94.0% (100%) | 93.7% (100%) | 94.0% (100%) | 93.6% (100%) | 93.7% (100%) | -0.0 pp | 1281 |
| `kai_evidence_edge_pack` | 30 | 93.8% (100%) | 93.6% (100%) | 94.0% (100%) | 93.7% (100%) | 94.0% (100%) | +0.1 pp | 1266 |

## 3. Multi-Evidence Synthesis (`M ∈ {2, 3}` Required Docs with Near-Duplicate Noise, 24 Cases)

| Policy | Exact Pass | Calibrated Acc | All-Gold Retained | Gold in Edge % | Avg Tokens | Token Savings | Cost USD/1k | Acc / 1k Tok | Leaks | Hijacks | Kai Gen Tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `raw_concat_baseline` | 0.0% | 44.7% | 100.0% | 0.0% | 5519 | 0.0% | $16.5569 | 0.081 | 0 | 0 | 0 |
| `relevance_desc` | 50.0% | 71.1% | 100.0% | 83.3% | 5519 | 0.0% | $16.5569 | 0.129 | 0 | 0 | 0 |
| `u_shaped_edge_packing` | 66.7% | 76.6% | 100.0% | 100.0% | 5519 | 0.0% | $16.5569 | 0.139 | 0 | 0 | 0 |
| `budgeted_top_k_prune` | 50.0% | 39.9% | 50.0% | 0.0% | 1282 | 76.8% | $3.8459 | 0.311 | 0 | 0 | 0 |
| `kai_evidence_edge_pack` | 100.0% | 85.5% | 100.0% | 100.0% | 1237 | 77.6% | $3.7324 | 0.691 | 0 | 0 | 0 |

## 4. Security, Multi-Tenant ACL & Prompt-Injection Audit (16 Stress Cases)

| Policy | Exact Pass | Calibrated Acc | Gold Edge % | Cross-Tenant Leaks | Injection Rank Hijacks | Kai Gen Output Tokens |
|---|---:|---:|---:|---:|---:|---:|
| `raw_concat_baseline` | 0.0% | 62.3% | 0.0% | 0 | 0 | 0 |
| `relevance_desc` | 100.0% | 89.9% | 100.0% | 0 | 0 | 0 |
| `u_shaped_edge_packing` | 100.0% | 89.9% | 100.0% | 0 | 0 | 0 |
| `budgeted_top_k_prune` | 100.0% | 93.9% | 100.0% | 0 | 0 | 0 |
| `kai_evidence_edge_pack` | 100.0% | 93.9% | 100.0% | 0 | 0 | 0 |

## 5. Honest Negative Regime #1: Noisy / Adversarially Inverted Reranker (36 Cases, `K=20`)

| Reranker Regime | Reranker Recall@6 | Policy | Exact Pass | Calibrated Acc | All-Gold Retained | Avg Input Tokens |
|---|---:|---|---:|---:|---:|---:|
| `clean` | 100.0% | `raw_concat_baseline` | 50.0% | 73.1% | 100.0% | 4796 |
| `clean` | 100.0% | `relevance_desc` | 100.0% | 87.7% | 100.0% | 4796 |
| `clean` | 100.0% | `u_shaped_edge_packing` | 100.0% | 87.7% | 100.0% | 4796 |
| `clean` | 100.0% | `budgeted_top_k_prune` | 100.0% | 93.9% | 100.0% | 1208 |
| `clean` | 100.0% | `kai_evidence_edge_pack` | 100.0% | 93.9% | 100.0% | 1206 |
| `moderate_noise` | 50.0% | `raw_concat_baseline` | 50.0% | 73.1% | 100.0% | 4839 |
| `moderate_noise` | 50.0% | `relevance_desc` | 100.0% | 85.5% | 100.0% | 4839 |
| `moderate_noise` | 50.0% | `u_shaped_edge_packing` | 50.0% | 71.0% | 100.0% | 4839 |
| `moderate_noise` | 50.0% | `budgeted_top_k_prune` | 50.0% | 47.0% | 50.0% | 1202 |
| `moderate_noise` | 50.0% | `kai_evidence_edge_pack` | 50.0% | 47.0% | 50.0% | 1208 |
| `severe_inversion` | 0.0% | `raw_concat_baseline` | 50.0% | 73.0% | 100.0% | 4887 |
| `severe_inversion` | 0.0% | `relevance_desc` | 100.0% | 84.6% | 100.0% | 4887 |
| `severe_inversion` | 0.0% | `u_shaped_edge_packing` | 0.0% | 54.1% | 100.0% | 4887 |
| `severe_inversion` | 0.0% | `budgeted_top_k_prune` | 0.0% | 0.0% | 0.0% | 1224 |
| `severe_inversion` | 0.0% | `kai_evidence_edge_pack` | 0.0% | 0.0% | 0.0% | 1222 |

## 6. Honest Negative Regime #2: Strictly Ordered Sequential Logs / Diffs (18 Cases)

| Policy | Chrono Guard | Exact Pass | Calibrated Acc | Order Violation Rate | Total Order Violations | Avg Input Tokens |
|---|---|---:|---:|---:|---:|---:|
| `raw_concat_baseline` | `false` | 0.0% | 53.2% | 0.0% | 0 | 2824 |
| `relevance_desc` | `false` | 0.0% | 15.2% | 100.0% | 54 | 2824 |
| `u_shaped_edge_packing` | `false` | 0.0% | 16.9% | 100.0% | 36 | 2824 |
| `budgeted_top_k_prune` | `false` | 0.0% | 15.6% | 100.0% | 54 | 1202 |
| `kai_evidence_edge_pack` | `false` | 0.0% | 16.9% | 100.0% | 36 | 1190 |
| `kai_evidence_edge_pack` | `true` | 100.0% | 75.8% | 0.0% | 0 | 953 |

## 7. Honest Negative Regime #3: Tight Token Budget on `M=3` Multi-Hop Synthesis (15 Cases, `K=16`)

| Token Budget | Policy | Exact Pass | Calibrated Acc | All-Gold Retained | Gold Recall | Avg Input Tokens | Cost USD / 1k Queries |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2400 | `raw_concat_baseline` | 0.0% | 64.6% | 100.0% | 100.0% | 3791 | $11.3742 |
| 2400 | `relevance_desc` | 100.0% | 76.1% | 100.0% | 100.0% | 3791 | $11.3742 |
| 2400 | `u_shaped_edge_packing` | 100.0% | 79.0% | 100.0% | 100.0% | 3791 | $11.3742 |
| 2400 | `budgeted_top_k_prune` | 100.0% | 77.1% | 100.0% | 100.0% | 2385 | $7.1540 |
| 2400 | `kai_evidence_edge_pack` | 100.0% | 82.6% | 100.0% | 100.0% | 1917 | $5.7708 |
| 1400 | `raw_concat_baseline` | 0.0% | 64.6% | 100.0% | 100.0% | 3791 | $11.3742 |
| 1400 | `relevance_desc` | 100.0% | 76.1% | 100.0% | 100.0% | 3791 | $11.3742 |
| 1400 | `u_shaped_edge_packing` | 100.0% | 79.0% | 100.0% | 100.0% | 3791 | $11.3742 |
| 1400 | `budgeted_top_k_prune` | 100.0% | 75.3% | 100.0% | 100.0% | 1216 | $3.6474 |
| 1400 | `kai_evidence_edge_pack` | 100.0% | 82.5% | 100.0% | 100.0% | 1215 | $3.6658 |
| 900 | `raw_concat_baseline` | 0.0% | 64.6% | 100.0% | 100.0% | 3791 | $11.3742 |
| 900 | `relevance_desc` | 100.0% | 76.1% | 100.0% | 100.0% | 3791 | $11.3742 |
| 900 | `u_shaped_edge_packing` | 100.0% | 79.0% | 100.0% | 100.0% | 3791 | $11.3742 |
| 900 | `budgeted_top_k_prune` | 100.0% | 81.5% | 100.0% | 100.0% | 744 | $2.2332 |
| 900 | `kai_evidence_edge_pack` | 100.0% | 81.5% | 100.0% | 100.0% | 744 | $2.2532 |
| 600 | `raw_concat_baseline` | 0.0% | 64.6% | 100.0% | 100.0% | 3791 | $11.3742 |
| 600 | `relevance_desc` | 100.0% | 76.1% | 100.0% | 100.0% | 3791 | $11.3742 |
| 600 | `u_shaped_edge_packing` | 100.0% | 79.0% | 100.0% | 100.0% | 3791 | $11.3742 |
| 600 | `budgeted_top_k_prune` | 0.0% | 0.0% | 0.0% | 66.7% | 502 | $1.5048 |
| 600 | `kai_evidence_edge_pack` | 0.0% | 0.0% | 0.0% | 66.7% | 502 | $1.5248 |
| 400 | `raw_concat_baseline` | 0.0% | 64.6% | 100.0% | 100.0% | 3791 | $11.3742 |
| 400 | `relevance_desc` | 100.0% | 76.1% | 100.0% | 100.0% | 3791 | $11.3742 |
| 400 | `u_shaped_edge_packing` | 100.0% | 79.0% | 100.0% | 100.0% | 3791 | $11.3742 |
| 400 | `budgeted_top_k_prune` | 0.0% | 0.0% | 0.0% | 33.3% | 251 | $0.7524 |
| 400 | `kai_evidence_edge_pack` | 0.0% | 0.0% | 0.0% | 33.3% | 251 | $0.7724 |
