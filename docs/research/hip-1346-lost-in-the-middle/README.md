# HIP-1346 Research Report: Evidence-Ordering Robustness & Context Packing ("Lost in the Middle")

- **Issue**: `hanzoai/agents#83` (`[Bounty #29] Evidence-ordering robustness · agents (Lost in the Middle)`)
- **Umbrella HIP**: HIP-1346 — Evidence-Grounded Retrieval and Decision Evaluation
- **Package**: [`control-plane/internal/eval/contextpack/`](../../../control-plane/internal/eval/contextpack/)
- **Reproducible CLI**: [`control-plane/cmd/contextpack-bench/`](../../../control-plane/cmd/contextpack-bench/)
- **Artifacts**: [`results.json`](./results.json) | [`score_table.md`](./score_table.md)

---

## 1. Primary Paper, Technical Summary, and License & Dependency Audit

### 1.1 Primary Literature & Reference Code
- **Paper**: Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang, *"Lost in the Middle: How Language Models Use Long Contexts"*, **TACL 2024** (Transactions of the Association for Computational Linguistics, Vol. 12, pp. 157–173).
  - ACL Anthology: https://aclanthology.org/2024.tacl-1.9/
  - arXiv: https://arxiv.org/abs/2307.03172
- **Reference Repository**: https://github.com/nelson-liu/lost-in-the-middle
- **Technical Summary**: Liu et al. (TACL 2024) demonstrate that long-context language models do not attend uniformly across their input window. On multi-document question answering and key-value retrieval across `K ∈ {5, 10, 20, 30}` documents, answer accuracy follows a pronounced **U-shaped serial-position curve**: performance is highest when relevant evidence appears at the very beginning (**primacy**) or very end (**recency**) of the context, and drops sharply—by 15 to 40+ percentage points as `K` grows—when relevant evidence is placed in the **middle**. Simply expanding raw context windows increases input token cost while actively degrading recall of buried evidence. In multi-document agent tool pipelines (`code_search`, `vfs_read`, `build_log`, `json_api`, `sql_graph`), optimizing **evidence placement, deduplication, and budgeted context packing** delivers higher grounded accuracy at dramatically lower input token cost than raw context expansion.

### 1.2 License, Dependency, and Invariant Audit
| Artifact / Dependency | License | Imported into Repo? | Audit Result |
|---|---|---|---|
| `nelson-liu/lost-in-the-middle` reference repo | MIT | **No** (0 lines copied; clean-room Go implementation) | Clear — methodology referenced only |
| External NaturalQuestions / MDQA datasets | CC-BY-SA / Apache | **No** (original deterministic `--seed 1346` agent tool corpus) | Clear — zero external downloads or redistribution |
| `control-plane/internal/eval/contextpack/` & `cmd/contextpack-bench/` | Apache-2.0 (repo license) | Yes (original Go stdlib code) | Clear — zero third-party packages |
| `control-plane/go.mod` & `control-plane/go.sum` | Apache-2.0 / existing | **Untouched** (0 new external dependencies) | Clear — zero dependency expansion |
| `control-plane/internal/server/` (130-route `routes_test.go` pin) | Apache-2.0 / existing | **Untouched** | Clear — 130-route pin intact |

**Clearance Verdict**: **PASS** — clean-room Apache-2.0 Go stdlib implementation, zero new external dependencies, zero route table or server modifications.

---

## 2. Architecture Under Test, Methodological Integrity, and Reproducible Commands

### 2.1 Canonical Hanzo Stack Boundaries
This harness preserves all canonical Hanzo architecture boundaries without introducing a competing runtime or storage layer:
- **`vfs -> hanzoai/s3 -> visor`**: Grounded multi-document tool outputs carry canonical source provenance (`code://`, `vfs://hanzoai-s3/visor/...`, `visor://runs/build-logs/`, `enso://v1/routing/...`, `sql://cloud/ledger/...`).
- **Cloud**: Owns lifecycle and billing; SQL/graph ledger fixtures reflect Cloud billing state without duplicating billing ownership inside the packer.
- **Enso**: Owns model routing and input token pricing (`$3.00 / 1M` input tokens modeled in `PackConfig.EnsoInputUSDPer1MTokens`).
- **Zen**: Owns generative answer synthesis, decoupled cleanly behind the [`ModelEvaluator`](../../../control-plane/internal/eval/contextpack/types.go) interface so live Zen/Enso endpoints plug directly into the exact same harness via `FuncEvaluator` or a custom struct.
- **Kai**: Owns structured, non-generative context-packing decisions (`KaiDecisionRecord.GenerativeOutputTokens == 0` enforced and asserted across all 5 policies and all cases, with amortized local decision compute priced at `$0.02 / 1k` calls).

### 2.2 Methodological Integrity & Offline Simulation Transparency
- **Zero Oracle Leakage**: None of the 5 context-packing policies in [`packer.go`](../../../control-plane/internal/eval/contextpack/packer.go) read `doc.IsGold` or `doc.EvidenceFactID` during filtering, deduplication, selection, or ordering. Those ground-truth labels are inspected strictly post-hoc for metric auditing.
- **Offline Calibration Disclosure**: Offline CI runs execute [`CalibratedLostInMiddleEvaluator`](../../../control-plane/internal/eval/contextpack/evaluator.go) (`IsSimulated() == true`), which combines:
  1. **Model-independent exact structural verification**: all required gold evidence facts retained (`AllGoldRetained`), exact extractive substring grounding, citation header grounding (`id=...`, `uri=...`), gold-in-edge-window rate, chronological sequence integrity (`OrderViolations`), cross-tenant ACL isolation (`CrossTenantLeaks`), prompt-injection rank-hijack immunity (`InjectionRankHijacks`), and Enso USD/1k input token cost.
  2. **Calibrated TACL '24 position-sensitivity proxy**: `PositionAttentionWeight(normPos, docCount, totalTokens)` modeling primacy > recency edges, a context-length-dependent middle valley (`K=5..30`), and a secondary token-volume dilution penalty. Live Zen/Enso hardware evaluations swap in via `ModelEvaluator` without modifying the corpus, packers, or reporter.

### 2.3 Five Matched Context-Packing Policies
1. `raw_concat_baseline` — Preserves raw tool arrival order after pre-packing ACL filtering; sends 100% of authorized tokens (`0%` savings).
2. `relevance_desc` — Sorts authorized documents strictly descending by relevance score; concentrates top hits at the start while ignoring U-shaped tail recency and sending 100% of tokens.
3. `u_shaped_edge_packing` — Implements the TACL '24 alternating edge placement (`rank 0 -> slot 0`, `rank 1 -> slot N-1`, `rank 2 -> slot 1`, `rank 3 -> slot N-2`, pushing distractors to the middle), sending 100% of authorized tokens with no chronological guard.
4. `budgeted_top_k_prune` — Greedily packs top-scoring documents up to `TokenBudget` (`1400` tokens) in descending order without deduplication, diversity preservation, U-shaped edge placement, or chronological locking.
5. `kai_evidence_edge_pack` — Kai non-generative structured packer (`GenerativeOutputTokens == 0`): pre-packing ACL filter, prompt-injection sanitization, token-Jaccard near-duplicate elimination (`threshold = 0.72`), chronological-group lock when `PreserveChronologicalOrder` or `RequiresOrder` is set, source-diversity-preserving budgeted selection (`TokenBudget = 1400`), and U-shaped edge placement.

### 2.4 Exact Reproduction Commands
From `control-plane/`:
```bash
# 1. Static analysis and race-detector verification (including 16-goroutine concurrency test)
GOWORK=off go vet ./internal/eval/contextpack/... ./cmd/contextpack-bench/...
GOWORK=off go test -v -race -count=1 ./internal/eval/contextpack/... ./cmd/contextpack-bench/...

# 2. Regenerate verified HIP-1346 JSON results and Markdown score tables (--seed 1346)
GOWORK=off go run ./cmd/contextpack-bench \
  --seed 1346 \
  --token-budget 1400 \
  --out-json ../docs/research/hip-1346-lost-in-the-middle/results.json \
  --out-md ../docs/research/hip-1346-lost-in-the-middle/score_table.md
```

**Execution Environment** (recorded in [`results.json`](./results.json)):
- Go `go1.26.5`, `linux/amd64`, `NumCPU = 2`, `--seed 1346`, `--token-budget 1400`.

---

## 3. Falsifiable Hypotheses, Measured Position Curves, Token Efficiency & Negative Regimes

### 3.1 Falsifiable Hypotheses
- **H1 (U-Shaped Middle Degradation)**: Under `raw_concat_baseline`, middle-slot accuracy degrades monotonically relative to start/end edges as `K` scales from `5 -> 10 -> 20 -> 30`.
- **H2 (Edge Packing + Budgeted Deduplication Dominates Pareto Frontier)**: Combining Kai non-generative deduplication, diversity-preserving pruning, and U-shaped edge placement (`kai_evidence_edge_pack`) achieves higher accuracy per 1k input tokens and lower USD/1k query cost than unpruned `u_shaped_edge_packing`, while avoiding the duplicate-crowding failures of naive `budgeted_top_k_prune` on multi-evidence synthesis.
- **H3 (Three Honest Negative Regimes)**: Context packing and pruning are **not** universally beneficial:
  1. **Noisy / Inverted Reranker**: When upstream reranker Recall@6 drops to `0%`, budgeted pruning drops true gold evidence completely (`0.0%`), performing **worse** than `raw_concat_baseline` (`73.0%`).
  2. **Strictly Ordered Sequential Logs / Diffs**: Unguarded relevance or U-shaped reordering breaks causal sequence order (`100%` order violation rate) unless guarded by `PreserveChronologicalOrder`.
  3. **Tight Budget on `M = 3` Multi-Hop Synthesis**: When `TokenBudget` drops below the physical token footprint of `M = 3` required documents (`< 705` tokens), pruned policies save up to `93.4%` tokens but collapse to `0.0%` exact synthesis accuracy.

### 3.2 Overall Primary Suite Comparison (`100` Cases, `--seed 1346`)
Measured directly from [`results.json`](./results.json) and [`score_table.md`](./score_table.md):

| Policy | Exact Pass | Calibrated Acc | All-Gold Retained | Gold in Edge % | Avg Tokens | Token Savings | Cost USD/1k | Acc / 1k Tok | Leaks | Hijacks | Kai Gen Tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `raw_concat_baseline` | 41.0% | 66.4% | 100.0% | 45.0% | 4164 | 0.0% | \$12.4929 | 0.159 | 0 | 0 | 0 |
| `relevance_desc` | 88.0% | 85.3% | 100.0% | 96.0% | 4164 | 0.0% | \$12.4929 | 0.205 | 0 | 0 | 0 |
| `u_shaped_edge_packing` | 92.0% | 86.6% | 100.0% | 100.0% | 4164 | 0.0% | \$12.4929 | 0.208 | 0 | 0 | 0 |
| `budgeted_top_k_prune` | 88.0% | 80.9% | 88.0% | 76.0% | 1271 | 69.5% | \$3.8122 | 0.636 | 0 | 0 | 0 |
| `kai_evidence_edge_pack` | **100.0%** | **91.8%** | **100.0%** | **100.0%** | **1240** | **70.2%** | **\$3.7403** | **0.741** | **0** | **0** | **0** |

**Key Primary Findings**:
- `kai_evidence_edge_pack` improves calibrated accuracy by **+25.4 pp** over `raw_concat_baseline` (`91.8%` vs `66.4%`) and **+5.2 pp** over unpruned `u_shaped_edge_packing` (`91.8%` vs `86.6%`), while cutting average input tokens by **70.2%** (`4,164 -> 1,240`), reducing Enso+Kai query cost by **70.1%** (`$12.4929 -> $3.7403` per 1k queries), and improving **accuracy per 1k input tokens by 4.66x** (`0.159 -> 0.741`).
- On the 24-case **Multi-Evidence Synthesis** (`M ∈ {2, 3}` with near-duplicate tool echoes), `budgeted_top_k_prune` loses half of all cases (`50.0%` All-Gold Retained, `39.9%` calibrated accuracy) because high-scoring duplicate echoes exhaust the 1400-token budget before all distinct supporting documents fit. `kai_evidence_edge_pack` deduplicates near-identical chunks and enforces cross-source diversity, achieving **100.0% All-Gold Retained**, **100.0% Exact Pass**, **85.5% calibrated accuracy**, and **77.6% token savings** (`5,519 -> 1,237` tokens; `0.691` vs `0.081` acc/1k tokens raw).

### 3.3 Single-Evidence Position Curve (`raw_concat_baseline` vs `kai_evidence_edge_pack`)
Full 5-policy curve is in [`score_table.md`](./score_table.md#2-single-evidence-position-curve-k--5-10-20-30-x-5-canonical-slots). Excerpt comparing raw arrival order vs Kai edge packing:

| Policy | K | Start (`0`) | Q1 (`K/4`) | Middle (`K/2`) | Q3 (`3K/4`) | End (`K-1`) | Middle Drop vs Start | Avg Input Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `raw_concat_baseline` | 5 | 94.0% (100%) | 84.0% (100%) | 77.7% (100%) | 82.5% (100%) | 91.0% (100%) | -16.3 pp | 1172 |
| `raw_concat_baseline` | 10 | 91.9% (100%) | 80.0% (100%) | 70.2% (33%) | 78.4% (100%) | 89.0% (100%) | -21.8 pp | 2337 |
| `raw_concat_baseline` | 20 | 87.9% (100%) | 66.1% (0%) | 54.4% (0%) | 69.1% (33%) | 85.0% (100%) | -33.4 pp | 4655 |
| `raw_concat_baseline` | 30 | 85.0% (100%) | 58.0% (0%) | 40.0% (0%) | 56.5% (0%) | 82.0% (100%) | **-45.0 pp** | 6991 |
| `kai_evidence_edge_pack` | 5 | 94.0% (100%) | 94.0% (100%) | 94.0% (100%) | 94.0% (100%) | 94.0% (100%) | -0.0 pp | 1172 |
| `kai_evidence_edge_pack` | 10 | 93.9% (100%) | 93.6% (100%) | 93.9% (100%) | 93.7% (100%) | 93.7% (100%) | -0.0 pp | 1283 |
| `kai_evidence_edge_pack` | 20 | 94.0% (100%) | 93.7% (100%) | 94.0% (100%) | 93.6% (100%) | 93.7% (100%) | -0.0 pp | 1281 |
| `kai_evidence_edge_pack` | 30 | 93.8% (100%) | 93.6% (100%) | 94.0% (100%) | 93.7% (100%) | 94.0% (100%) | **+0.1 pp** | **1266** |

### 3.4 Three Honest Negative Regimes (Where Packing / Pruning Degrades or Fails)

1. **Negative Regime #1 — Noisy / Adversarially Inverted Reranker (`36` cases, `K = 20`)**:
   - At `clean` (`Recall@6 = 100.0%`), `kai_evidence_edge_pack` (`93.9%`, `1,206` tokens) beats `raw_concat_baseline` (`73.1%`, `4,796` tokens).
   - At `moderate_noise` (`Recall@6 = 50.0%`), `budgeted_top_k_prune` and `kai_evidence_edge_pack` both drop to **`47.0%`** calibrated accuracy (`50.0%` All-Gold Retained), **underperforming `raw_concat_baseline` (`73.1%`)**.
   - At `severe_inversion` (`Recall@6 = 0.0%`), budgeted pruning drops the true gold document completely (**`0.0%` accuracy, `0.0%` All-Gold Retained**). Furthermore, unpruned `u_shaped_edge_packing` drops to **`54.1%` (`0.0%` Exact Pass)** because it deliberately places lowest-ranked documents into the dead middle—whereas `relevance_desc` accidentally rescues rank-20 via tail recency (`84.6%`). **Operational rule**: never enable aggressive token-budget pruning unless upstream reranker `Recall@k` is monitored and exceeds calibrated confidence floors.

2. **Negative Regime #2 — Strictly Ordered Sequential Logs / Diffs (`18` cases, 4-chunk migration traces)**:
   - Because failure keywords concentrate in later crash/rollback chunks, unguarded reordering scrambles causal order (`100.0%` order violation rate): `relevance_desc` (`15.2%`, `54` violations), `u_shaped_edge_packing` (`16.9%`, `36` violations), `budgeted_top_k_prune` (`15.6%`, `54` violations), and unguarded `kai_evidence_edge_pack (guard=false)` (`16.9%`, `36` violations) ALL fail worse than `raw_concat_baseline` (`53.2%`, `0` violations).
   - Enabling Kai's chronological lock (`EnforceChronologicalOrder: true`) eliminates all sequence breaks (**`0.0%` violation rate, `0` violations**), restores **`100.0%` Exact Pass (`75.8%` calibrated accuracy)**, and drops irrelevant non-log distractors (`2,824 -> 953` tokens, `-66.3%`).

3. **Negative Regime #3 — Tight Token Budget on `M = 3` Multi-Hop Synthesis (`15` cases, `K = 16`, ~705 gold tokens)**:
   - At budgets `>= 900` tokens (`2400`, `1400`, `900`), `kai_evidence_edge_pack` retains `100.0%` of all 3 gold documents (`81.5%–82.6%` calibrated accuracy, `100.0%` Exact Pass) while cutting cost from `\$11.3742` down to `\$2.2532 / 1k`.
   - Below the `~705`-token physical floor (`budget = 600` -> `66.7%` partial recall; `budget = 400` -> `33.3%` partial recall), both `budgeted_top_k_prune` and `kai_evidence_edge_pack` achieve cheap prompts (`\$0.75–\$1.52 / 1k`) but collapse to **`0.0%` Exact Pass / `0.0%` Calibrated Accuracy** on all-or-nothing synthesis—strictly worse than unpruned `raw_concat_baseline` (`64.6%`) and `u_shaped_edge_packing` (`79.0%`).

---

## 4. Bounded Integration Recommendation (Canonical Hanzo Architecture)

1. **Adopt `kai_evidence_edge_pack` as a Pre-Synthesis Packing Stage (Kai -> Enso -> Zen)**:
   - Run Kai's structured non-generative packer (`GenerativeOutputTokens == 0`) immediately after tool execution (`vfs -> hanzoai/s3 -> visor`, SQL/API queries) and strictly **after** pre-packing tenant/ACL verification, before submitting the packed context to Enso for Zen synthesis.
2. **Mandatory Production Guardrails**:
   - **Reranker Confidence Fallback**: When Kai's relevance distribution is flat/low-entropy or upstream reranker confidence falls below threshold, bypass hard pruning and fall back to full-context or larger budget window (mitigating Negative Regime #1).
   - **Chronological Group Metadata**: Tag sequentially coupled tool outputs (build logs, git diffs, stack traces) with `OrderGroupID`, `SequenceIndex`, and `RequiresOrder: true` so Kai locks ascending causal order (mitigating Negative Regime #2).
   - **Multi-Hop Budget Floor**: Set minimum `TokenBudget` dynamically from estimated required evidence count (`>= 300 * M` tokens) or emit an explicit `insufficient_budget` structured Kai decision instead of silent truncation (mitigating Negative Regime #3).
3. **Zero Invasive Control-Plane Surface Changes**:
   - Keep `control-plane/internal/server/` (130-route `routes_test.go` pin) and `control-plane/go.mod` completely untouched until live Zen/Enso hardware validation via `ModelEvaluator` confirms offline proxy predictions.

---

## 5. Correctness, Concurrency, Security, and Regression Evidence

Verified cleanly on Go `go1.26.5` (`linux/amd64`):
- **Static Analysis**: `GOWORK=off go vet ./internal/eval/contextpack/... ./cmd/contextpack-bench/...` -> **PASS (0 warnings)**.
- **Unit, Negative-Regime, CLI & Race Suite**: `GOWORK=off go test -v -race -count=1 ./internal/eval/contextpack/... ./cmd/contextpack-bench/...` -> **14/14 tests PASS with `-race`**.
- **Concurrency Safety**: `TestConcurrency_16GoroutinesRaceFreeAndDeterministic` runs 16 concurrent goroutines executing packing and evaluation simultaneously under `-race`, verifying **0 data races** and **bit-for-bit identical deterministic JSON outputs**.
- **Security Invariants**:
  - `TestPackContext_PrePackingACLIsolationZeroLeaks`: **0 cross-tenant, cross-workspace, or insufficient-role leaks** across all 100 primary cases x 5 policies (`500` runs).
  - `TestPackContext_PromptInjectionRankHijackImmunity`: **0 injection-induced rank hijacks**, exact rank-order invariance between clean and injected payloads, and complete sanitization of embedded override directives.
  - **Kai Boundary**: `MaxKaiGenOutputTokens == 0` verified across every policy and case.
- **Control-Plane Invariant**: `git diff --stat` confirms `control-plane/go.mod`, `control-plane/go.sum`, and `control-plane/internal/server/` are **100% untouched**.

---

## 6. HIP-1346 Draft Update Snippet

```markdown
### HIP-1346 Evaluation Entry — Issue #83 (Lost in the Middle / Context Packing in `hanzoai/agents`)

- **Harness**: `control-plane/internal/eval/contextpack/` + `control-plane/cmd/contextpack-bench/` (clean-room Apache-2.0 Go stdlib, zero new dependencies, `--seed 1346`).
- **Primary Result (100 cases, `K ∈ {5, 10, 20, 30}`, 5 slots + multi-evidence synthesis + ACL/injection stress)**:
  - `raw_concat_baseline` suffers a `-45.0 pp` middle-slot accuracy drop at `K = 30` (`85.0%` start -> `40.0%` middle -> `82.0%` end; overall `66.4%` calibrated acc, `4,164` avg tokens, `$12.49/1k` queries, `0.159` acc/1k tok).
  - `u_shaped_edge_packing` eliminates middle-slot decay (`86.6%` calibrated acc, `100.0%` gold-in-edge) but saves `0%` input tokens (`$12.49/1k`).
  - `kai_evidence_edge_pack` (`GenerativeOutputTokens == 0`) achieves `100.0%` Exact Pass, `91.8%` calibrated accuracy, `100.0%` gold-in-edge placement, `70.2%` token savings (`1,240` avg tokens, `$3.74/1k` queries), and `0.741` accuracy per 1k input tokens (`4.66x` raw baseline), with `0` cross-tenant ACL leaks and `0` prompt-injection rank hijacks.
- **Documented Negative Regimes**:
  1. Adversarial reranker inversion (`Recall@6 = 0%`) collapses budgeted pruning to `0.0%` vs `73.0%` raw baseline.
  2. Unguarded reordering breaks causally ordered sequential logs (`100%` order violations); Kai's `PreserveChronologicalOrder` guard restores `0%` violations and `100%` Exact Pass.
  3. Budgets below `M = 3` gold token footprint (`<= 600` tokens) collapse all-or-nothing multi-hop synthesis to `0.0%` despite `87%–93%` token savings.
- **Next Gate**: Bind live Enso/Zen synthesis endpoint to `contextpack.ModelEvaluator` for non-simulated hardware confirmation before enabling budgeted pruning by default in agent tool loops.
```
