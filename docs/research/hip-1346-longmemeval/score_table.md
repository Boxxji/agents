# HIP-1346 LongMemEval Score Table (`--seed 1346`, 60 Cases)

## 1. Overall Policy Comparison (5 LongMemEval Dimensions)

| Policy | Overall | Update | Contradiction | Contra. Det. | Temporal | Abstention | Cross-Session | Stale Err | Unans. FPR | Abst. F1 | Tenant Leaks | Kai Gen Tok | Write Amp | Avg Tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `flat-kv-latest` | 61.67% | 100.00% | 50.00% | 0.00% | 8.33% | 100.00% | 50.00% | 13.33% | 0.00% | 100.00% | 0 | 0 | 2.00x | 10.0 |
| `hierarchical-scope-kv` | 45.00% | 66.67% | 0.00% | 0.00% | 8.33% | 100.00% | 50.00% | 30.00% | 0.00% | 100.00% | 0 | 0 | 2.00x | 9.8 |
| `append-only-history` | 11.67% | 0.00% | 50.00% | 0.00% | 8.33% | 0.00% | 0.00% | 50.00% | 100.00% | 0.00% | 0 | 0 | 2.00x | 23.6 |
| `vector-lexical-recency` | 48.33% | 100.00% | 50.00% | 0.00% | 8.33% | 33.33% | 50.00% | 56.67% | 66.67% | 50.00% | 0 | 0 | 0.95x | 43.9 |
| `atomic-reflective` | 96.67% | 100.00% | 100.00% | 100.00% | 83.33% | 100.00% | 100.00% | 3.33% | 0.00% | 100.00% | 0 | 0 | 3.73x | 17.9 |
| `kai-evidence-gated` | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% | 100.00% | 0 | 0 | 3.73x | 18.6 |

## 2. Negative Regime 1 — Recency-Bias Failure on Historical `AsOf` Queries (16 Cases)

| Policy | Historical Accuracy | Recency-Bias Error Rate | Abstention Rate | Correct / Total |
|---|---:|---:|---:|---:|
| `flat-kv-latest` | 0.00% | 100.00% | 0.00% | 0 / 16 |
| `hierarchical-scope-kv` | 0.00% | 100.00% | 0.00% | 0 / 16 |
| `vector-lexical-recency` | 0.00% | 100.00% | 0.00% | 0 / 16 |
| `atomic-reflective` | 0.00% | 100.00% | 0.00% | 0 / 16 |
| `kai-evidence-gated (no-asof ablation)` | 0.00% | 100.00% | 0.00% | 0 / 16 |
| `kai-evidence-gated (asof time-travel)` | 100.00% | 0.00% | 0.00% | 16 / 16 |

## 3. Negative Regime 2 — High-Distractor Long-Horizon Scaling (5 to 200 Sessions)

| Distractor Sessions | Policy | Needle Recall | Unanswerable FPR | Abstention F1 | Overall Acc | Avg Recall Tok | Mean Latency (us) | Tenant Leaks |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 5 | `vector-lexical-recency (unthresholded)` | 80.00% | 100.00% | 0.00% | 40.00% | 106.3 | 183.0 | 0 |
| 5 | `vector-lexical-recency (min_score=0.35)` | 100.00% | 40.00% | 75.00% | 80.00% | 106.3 | 28.8 | 0 |
| 5 | `kai-evidence-gated (semantic, theta=0.35)` | 100.00% | 40.00% | 75.00% | 80.00% | 106.3 | 37.9 | 0 |
| 25 | `vector-lexical-recency (unthresholded)` | 60.00% | 100.00% | 0.00% | 30.00% | 108.2 | 290.7 | 0 |
| 25 | `vector-lexical-recency (min_score=0.35)` | 90.00% | 80.00% | 33.33% | 55.00% | 108.2 | 35.8 | 0 |
| 25 | `kai-evidence-gated (semantic, theta=0.35)` | 100.00% | 80.00% | 33.33% | 60.00% | 108.2 | 72.2 | 0 |
| 50 | `vector-lexical-recency (unthresholded)` | 30.00% | 100.00% | 0.00% | 15.00% | 108.0 | 152.8 | 0 |
| 50 | `vector-lexical-recency (min_score=0.35)` | 90.00% | 80.00% | 33.33% | 55.00% | 108.0 | 275.5 | 0 |
| 50 | `kai-evidence-gated (semantic, theta=0.35)` | 100.00% | 80.00% | 33.33% | 60.00% | 108.0 | 380.9 | 0 |
| 100 | `vector-lexical-recency (unthresholded)` | 40.00% | 100.00% | 0.00% | 20.00% | 108.5 | 296.1 | 0 |
| 100 | `vector-lexical-recency (min_score=0.35)` | 90.00% | 90.00% | 18.18% | 50.00% | 108.5 | 213.7 | 0 |
| 100 | `kai-evidence-gated (semantic, theta=0.35)` | 100.00% | 90.00% | 18.18% | 55.00% | 108.5 | 463.1 | 0 |
| 200 | `vector-lexical-recency (unthresholded)` | 20.00% | 100.00% | 0.00% | 10.00% | 109.0 | 1413.7 | 0 |
| 200 | `vector-lexical-recency (min_score=0.35)` | 90.00% | 90.00% | 18.18% | 50.00% | 109.0 | 922.0 | 0 |
| 200 | `kai-evidence-gated (semantic, theta=0.35)` | 100.00% | 90.00% | 18.18% | 55.00% | 109.0 | 689.1 | 0 |

## 4. Negative Regime 3 — Abstention Threshold Pareto Tradeoff (`kai-evidence-gated`, 40 Semantic Cases)

| Threshold (`theta`) | Paraphrase Recall | Over-Abstention Rate | Unanswerable FPR | Abstention Prec | Abstention Recall | Abstention F1 | Overall Acc |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.10` | 85.00% | 5.00% | 100.00% | 0.00% | 0.00% | 0.00% | 42.50% |
| `0.20` | 70.00% | 25.00% | 100.00% | 0.00% | 0.00% | 0.00% | 35.00% |
| `0.30` | 60.00% | 40.00% | 100.00% | 0.00% | 0.00% | 0.00% | 30.00% |
| `0.40` | 45.00% | 55.00% | 50.00% | 47.62% | 50.00% | 48.78% | 47.50% |
| `0.50` | 30.00% | 70.00% | 20.00% | 53.33% | 80.00% | 64.00% | 55.00% |
| `0.60` | 15.00% | 85.00% | 0.00% | 54.05% | 100.00% | 70.17% | 57.50% |
| `0.70` | 0.00% | 100.00% | 0.00% | 50.00% | 100.00% | 66.67% | 50.00% |
| `0.80` | 0.00% | 100.00% | 0.00% | 50.00% | 100.00% | 66.67% | 50.00% |
| `0.90` | 0.00% | 100.00% | 0.00% | 50.00% | 100.00% | 66.67% | 50.00% |
