### Overall Policy Score Table

| Policy | Exact Match | Recall@K | Abstention F1 | Unanswerable FPR | Contradiction Det. | Temporal Order | Stale Shadowing | Cross-Actor Leaks | Avg Tokens | p50 (ms) | p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `kv_scoped` | 50.00% | 50.00% | 50.00% | 0.00% | 0.00% | 0.00% | 5 | 0 | 5.7 | 0.035 | 0.086 |
| `kv_fallback_chain` | 56.00% | 56.00% | 100.00% | 0.00% | 0.00% | 0.00% | 12 | 0 | 11.4 | 0.055 | 0.066 |
| `vector_top_k` | 34.00% | 80.00% | 0.00% | 100.00% | 0.00% | 0.00% | 16 | 0 | 59.4 | 0.501 | 0.749 |
| `vector_thresholded` | 88.00% | 98.00% | 100.00% | 0.00% | 0.00% | 100.00% | 6 | 0 | 27.3 | 0.478 | 0.681 |
| `hybrid_scope_policy` | 100.00% | 100.00% | 100.00% | 0.00% | 100.00% | 100.00% | 0 | 0 | 24.0 | 0.253 | 0.612 |

### Per-Category Exact Match Breakdown

| Policy | Knowledge Updates | Contradiction Handling | Temporal Ordering | Abstention | Cross-Session Retrieval |
|---|---:|---:|---:|---:|---:|
| `kv_scoped` | 100.00% (10/10) | 50.00% (5/10) | 0.00% (0/10) | 100.00% (10/10) | 0.00% (0/10) |
| `kv_fallback_chain` | 60.00% (6/10) | 20.00% (2/10) | 0.00% (0/10) | 100.00% (10/10) | 100.00% (10/10) |
| `vector_top_k` | 40.00% (4/10) | 50.00% (5/10) | 0.00% (0/10) | 0.00% (0/10) | 80.00% (8/10) |
| `vector_thresholded` | 100.00% (10/10) | 40.00% (4/10) | 100.00% (10/10) | 100.00% (10/10) | 100.00% (10/10) |
| `hybrid_scope_policy` | 100.00% (10/10) | 100.00% (10/10) | 100.00% (10/10) | 100.00% (10/10) | 100.00% (10/10) |
