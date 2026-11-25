# HIP-1347 Research Report: Evidence-Ordering Robustness & Context Packing ("Lost in the Middle")

- **Tracking Issue**: `hanzoai/agents#76` (`hip-1347`, `research`, `p2`, `test`)
- **Benchmark Package**: [`sdk/python/benchmarks/lost_in_the_middle/`](../../../sdk/python/benchmarks/lost_in_the_middle/)
- **Test Suite**: [`sdk/python/tests/test_lost_in_the_middle_benchmark.py`](../../../sdk/python/tests/test_lost_in_the_middle_benchmark.py)
- **Measured Artifacts**: [`score_table.md`](./score_table.md) | [`raw_results.json`](./raw_results.json)

---

## 1. Primary Paper, Summary, and License Audit

### Primary Reference
- **Paper**: Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. *"Lost in the Middle: How Language Models Use Long Contexts."* Transactions of the Association for Computational Linguistics (TACL), 2024, pp. 157–173. https://aclanthology.org/2024.tacl-1.9/
- **Upstream Code Reference**: https://github.com/nelson-liu/lost-in-the-middle (inspected for JSONL fixture schema compatibility only; **zero code or data copied**).

### Paper Summary
Liu et al. (2024) evaluate multi-document question answering and key-value retrieval across document positions `1..N` (`N ∈ {10, 20, 30}`), demonstrating that language model accuracy is typically highest when relevant evidence sits at the beginning (primacy) or end (recency) of the prompt and degrades substantially when evidence is buried in the middle. Crucially for agent runtimes, real-world degradation compounds with **hard context-window truncation**, **near-duplicate retrieval redundancy**, and **prompt-injection boundary exposure** during context assembly.

### License & Dependency Audit

| Component | License | Bundled / Imported? | Audit Status |
|---|---|---|---|
| Liu et al., TACL 2024 paper | CC-BY-4.0 | Referenced only (3 published calibration anchor points cited explicitly) | Cleared |
| `nelson-liu/lost-in-the-middle` code | MIT | **Not imported** — clean-room re-implementation | Cleared |
| NaturalQuestions / Wikipedia `qa_data` | CC-BY-SA-3.0 / Apache-2.0 | **Not bundled** — optional `--external-jsonl` loader reads user-supplied local files | Cleared |
| `sdk/python/benchmarks/lost_in_the_middle/` | MIT / Apache-2.0 (repo license) | Pure Python 3.12 stdlib + existing `hanzo_agents` SDK (`MemoryClient`, `ScopedMemoryClient`, `AIConfig`, `llm_adapters`) | Cleared |
| Model weights | N/A | None bundled; `--mode live_llm` calls user-configured Enso/Zen or OpenAI-compatible endpoints | Cleared |
| GPL / AGPL / LGPL code | None | Zero copyleft dependencies introduced | Cleared |

> **HIP-1347 Attribution Rule**: Published TACL 2024 numbers belong exclusively to Liu et al. (2024) and are **not** claimed as Hanzo model results. Live Hanzo Zen/Enso numbers require executing the `--mode live_llm` path documented below against a provisioned Hanzo evaluation endpoint.

---

## 2. Methodological Integrity & Reproducible Baselines

### Addressing Circularity (Non-Circular Physical Mechanisms vs. Calibrated Stress-Test)
A known methodological pitfall in offline Lost-in-the-Middle harnesses is baking a synthetic U-shaped attention probability curve into a mock reader and then reporting that edge placement beats middle placement — a circular proof that assumes its conclusion.

To prevent circularity, this harness evaluates all five packing policies over Hanzo's **actual** Python SDK components (`MemoryClient`, `ScopedMemoryClient`, `AIConfig`, and `llm_adapters`) across **four explicitly separated evaluation modes**:

1. **`budget_truncation` (`is_calibrated_stress_test=False`, Physical / Non-Circular)**:
   - Contains **zero assumed attention curves**.
   - Exercises Hanzo's real `AIConfig.trim_by_chars(prompt, limit=budget_chars, head_ratio=0.2)` (`hanzo_middle_trim`) — which physically excises the middle slice of oversized prompts and inserts `"\n…TRIMMED…\n"` — as well as serving-stack right-side clipping (`tail_clip`) across strict token budgets (`512, 1024, 2048, 4096`).
   - Correctness depends strictly on whether the gold evidence span physically survived budget enforcement intact and escaped prompt-injection hijacking.
2. **`distractor_interference` (`is_calibrated_stress_test=False`, Physical / Non-Circular)**:
   - Contains **zero position-probability bias**.
   - Parses surviving intact document blocks and aggregates lexical/claim support votes for the queried entity and attribute. Captures real-world duplicate-crowding interference where redundant stale distractors outvote a single authoritative passage unless deduplicated.
3. **`calibrated_tacl2024` (`is_calibrated_stress_test=True`, Offline Sensitivity Stress-Test)**:
   - Explicitly labeled in code, JSON metadata, and tables as a **calibrated offline sensitivity stress-test profile** parameterized from published Liu et al. (TACL 2024, Fig. 2) curves (`gpt-3.5-turbo-16k`, `claude-1.3`, `mpt-30b-instruct`), layered on top of physical token-budget retention. Never reported as a live Hanzo model result.
4. **`live_llm` (`is_calibrated_stress_test=False`, Live Hanzo Enso/Zen & OpenAI-Compatible Hook)**:
   - Uses the exact same CLI, dataset, packers, `AIConfig.get_llm_params()`, `hanzo_agents.llm_adapters.apply_provider_patches()`, and TACL-2024 substring/exact-match/token-F1 scorer against a live OpenAI-compatible endpoint (`HANZO_API_KEY` + `HANZO_API_BASE` or `OPENAI_API_KEY` + `OPENAI_BASE_URL`), recording actual server-reported `prompt_tokens` and `completion_tokens`.

### Five Context Packing Policies Evaluated
1. `chronological`: Original insertion/timestamp order, budget-unaware, naive plain framing.
2. `score_descending`: Naive top-k retriever relevance descending, budget-unaware, plain framing.
3. `bookend_primacy_recency`: Alternating primacy/recency placement (`[r0, r2, ..., r3, r1]`), no deduplication, budget-unaware, plain framing.
4. `dedup_budgeted_bookend`: 3-gram word-shingle Jaccard near-duplicate suppression (`threshold=0.65`) + strict token-budget packing (guaranteed `estimated_tokens <= budget_tokens`, zero uncontrolled downstream clipping) + primacy/recency bookend ordering + per-pack nonce-isolated XML boundary escaping.
5. `middle_buried_adversarial`: Worst-case baseline explicitly placing gold evidence at `N // 2` under distractor load.

### Reproduction Commands

```bash
# 1. Reproduce all offline physical + calibrated runs and regenerate raw_results.json + score_table.md
cd sdk/python
python3 -m benchmarks.lost_in_the_middle \
  --mode all \
  --doc-counts 5,10,20 \
  --budgets 512,1024,2048,4096 \
  --cases-per-position 2 \
  --negative-cases 10 \
  --seed 42 \
  --compact-json \
  --output-json ../../docs/research/hip-1347-lost-in-the-middle/raw_results.json \
  --output-markdown ../../docs/research/hip-1347-lost-in-the-middle/score_table.md

# 2. Run the exact same harness on Hanzo Enso/Zen hardware (live endpoint)
HANZO_API_KEY="<key>" HANZO_API_BASE="https://api.hanzo.ai/v1" \
python3 -m benchmarks.lost_in_the_middle \
  --mode live_llm \
  --model zen/zen-coder \
  --doc-counts 10,20 \
  --budgets 1024,2048,4096 \
  --cases-per-position 1 \
  --negative-cases 0 \
  --output-json live_zen_results.json

# 3. Run directly against upstream nelson-liu/lost-in-the-middle fixtures (100 matched questions)
HANZO_API_KEY="<key>" \
python3 -m benchmarks.lost_in_the_middle \
  --mode live_llm \
  --model zen/zen-coder \
  --external-jsonl /path/to/nq-open-10_total_documents_gold_at_4.jsonl.gz \
  --external-limit 100 \
  --budgets 2048
```

---

## 3. Falsifiable Hypothesis, Metrics, and Measured Failure Regimes

### Hypothesis (H1)
> Under constrained token budgets (`512–4096` tokens) and multi-document retrieval (`N ∈ {5, 10, 20}`), `dedup_budgeted_bookend` increases evidence retention and answer correctness over `chronological` packing by `>= +20pp` (paired bootstrap 95% CI strictly above 0), increases token efficiency (correctness per 1K tokens), eliminates uncontrolled downstream `AIConfig.trim_by_chars` truncation (`0%` truncation loss on well-ranked corpora), and maintains `100%` prompt-injection boundary integrity — **except** under three explicit failure/negative regimes:
> 1. **Small unconstrained context (`N <= 3`)**: zero accuracy gain (`0.0pp`) with added delimiter tokens and shingling latency.
> 2. **Mis-ranked retriever**: score-based eviction amplifies retriever errors and degrades accuracy below `chronological`.
> 3. **Un-deduplicated bookend under duplicate-heavy corpora**: naive `bookend_primacy_recency` fills both edge slots with redundant distractors and collapses in lexical interference.

### Measured Before / After Summary (`seed=42`, `110` cases, `8,800` total evaluations)

Full per-budget, per-`N`, per-regime, and paired bootstrap CI tables are in [`score_table.md`](./score_table.md) and [`raw_results.json`](./raw_results.json).

#### A. Standard Position Sweep (`N ∈ {5, 10, 20}`, All Positions `1..N`, Budgets `512..4096`, `280` paired evals/policy)

| Mode | Policy | Correctness | Retention | Trunc Loss | Mean Tokens | Eff (Acc/1K tok) | Boundary Integrity | p50 / p95 Lat (ms) | Paired Delta vs `chronological` (95% CI) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Physical: `hanzo_middle_trim` | `chronological` | 60.7% | 60.7% | 39.3% | 1475.6 | 0.411 | 100.0% | 0.077 / 0.165 | baseline |
| Physical: `hanzo_middle_trim` | `score_descending` | 100.0% | 100.0% | 0.0% | 1475.6 | 0.678 | 100.0% | 0.077 / 0.173 | +0.3929 |
| Physical: `hanzo_middle_trim` | `bookend_primacy_recency` | 100.0% | 100.0% | 0.0% | 1475.6 | 0.678 | 100.0% | 0.079 / 0.158 | +0.3929 `[+0.3321, +0.4536]` |
| Physical: `hanzo_middle_trim` | `dedup_budgeted_bookend` | **100.0%** | **100.0%** | **0.0%** | **1378.9** | **0.725** | **100.0%** | 5.823 / 11.568 | **+0.3929 `[+0.3321, +0.4536]`** |
| Physical: `hanzo_middle_trim` | `middle_buried_adversarial` | 39.3% | 39.3% | 60.7% | 1475.6 | 0.266 | 100.0% | 0.094 / 0.213 | -0.2143 |
| Physical: `tail_clip` | `chronological` | 62.9% | 62.9% | 37.1% | 1475.6 | 0.426 | 100.0% | 0.077 / 0.136 | baseline |
| Physical: `tail_clip` | `dedup_budgeted_bookend` | **100.0%** | **100.0%** | **0.0%** | **1378.9** | **0.725** | **100.0%** | 5.823 / 7.882 | **+0.3714 `[+0.3179, +0.4286]`** |
| Calibrated Stress-Test (`gpt-3.5-turbo-16k`)* | `chronological` | 39.6% | 60.7% | 39.3% | 1475.6 | 0.268 | 100.0% | 0.070 / 0.130 | baseline |
| Calibrated Stress-Test (`gpt-3.5-turbo-16k`)* | `bookend_primacy_recency` | 72.9% | 100.0% | 0.0% | 1475.6 | 0.494 | 100.0% | 0.072 / 0.147 | +0.3335 `[+0.2943, +0.3707]` |
| Calibrated Stress-Test (`gpt-3.5-turbo-16k`)* | `dedup_budgeted_bookend` | **75.1%** | **100.0%** | **0.0%** | **1378.9** | **0.545** | **100.0%** | 5.717 / 10.116 | **+0.3556 `[+0.3143, +0.3962]`** |

*\*Calibrated offline stress-test parameterized from Liu et al. (TACL 2024); NOT a live Hanzo model measurement.*

Key physical discovery: Hanzo's production `AIConfig.trim_by_chars(..., head_ratio=0.2)` creates a **real, physical U-shaped survival cliff** even without any transformer attention assumption — under `chronological`, Primacy (P1) = **100.0%**, Middle (`N/2`) = **50.0%**, Recency (PN) = **100.0%**! Under `512`-token budgets, `dedup_budgeted_bookend` achieves **100.0% correctness at 462 mean tokens (efficiency 2.16 Acc/1K tok)** vs `chronological` **25.7% at 512 tokens (efficiency 0.50)** and `middle_buried_adversarial` **0.0%**.

#### B. Honest Negative & Adversarial Failure Regimes (Measured)

| Regime | Mode | `chronological` | `bookend_primacy_recency` | `dedup_budgeted_bookend` | Takeaway |
|---|---|---:|---:|---:|---|
| `small_context` (`N=3`) | Physical (`middle_trim`) | 100.0% (565.1 tok) | 100.0% (565.1 tok) | 100.0% (606.1 tok) | **Negative**: `0.0pp` accuracy gain, `+41` framing tokens, `+5.7 ms` latency — gate packing off for `N <= 3` under ample budget. |
| `misranked_retriever` (`N=10`) | Physical (`middle_trim`) | **100.0%** (0.0% loss) | 52.5% (47.5% loss) | **42.5%** (57.5% loss) | **Negative**: `-57.5pp` regression when retriever ranks distractors above gold; packing cannot fix a mis-calibrated retriever. |
| `misranked_retriever` (`N=10`) | Physical (`tail_clip`) | **75.0%** (25.0% loss) | 72.5% (27.5% loss) | **42.5%** (57.5% loss) | **Negative**: Confirmed under both middle-trim and tail-clip truncation. |
| `duplicate_heavy` (`N=10`) | Physical (`distractor_interference`) | 20.0% (1291.5 tok) | **0.0%** (1291.5 tok) | **75.0%** (1156.2 tok) | **Failure of naive bookend**: Naive bookending wastes both primacy & recency slots on duplicate distractors; dedup recovers `+75.0pp` and saves `10.5%` tokens. |
| `prompt_injection` (`N=10`) | Physical (`middle_trim`) | 20.0% (50.0% integrity) | **0.0%** (**0.0%** integrity) | **100.0%** (**100.0%** integrity) | **Security risk of naive bookend**: Unescaped bookending moves high-scoring injected tickets to protected edge anchors; nonce XML escaping restores `100%` boundary integrity. |

---

## 4. Bounded Integration Recommendation (Canonical Hanzo Architecture)

Keeping Hanzo's canonical architecture fixed (`vfs -> Hanzo S3 -> Visor`, `Cloud` owns lifecycle & billing, `Enso` routes models, `Zen` generates, `Kai` decides):

1. **No parallel retrieval/orchestration stack**: Context packing belongs as a lightweight, opt-in pre-formatting pass inside the existing Python SDK (`hanzo_agents.types.AIConfig` and `hanzo_agents.memory.ScopedMemoryClient`), downstream of retrieval and upstream of `Enso`/`Zen` dispatch.
2. **Gated activation rule (driven directly by our negative regimes)**:
   - **Bypass** packing when `len(docs) <= 3` AND `estimated_tokens <= 0.75 * safe_input_budget` AND `untrusted_sources=False` (eliminates the `+41` token and `~5.8 ms` shingling tax in the `small_context` regime).
   - **Enable** `dedup_budgeted_bookend` when `len(docs) > 3` OR `estimated_tokens > 0.75 * safe_input_budget` OR `untrusted_sources=True`.
   - **Require calibrated retrieval scores** (or preserve a chronological diversity slot / MMR floor when score entropy is high) to avoid the `misranked_retriever` cliff (`-57.5pp`).
3. **Replace blind string slicing on structured contexts**: `AIConfig.trim_by_chars` currently slices raw strings blindly with `head_ratio=0.2`, silently amputating middle passages. Passing structured context items through a budget-aware packer before `trim_by_chars` eliminates uncontrolled mid-passage slicing (`39.3% -> 0.0%` truncation loss on well-ranked retrievals).
4. **Next Gate for Core Implementation Bounty**: Run `--mode live_llm` on a Hanzo-provisioned Zen/Enso checkpoint against 100 matched upstream `10_total_documents` and `20_total_documents` questions (`--external-jsonl`) to verify whether live Zen attention curves exhibit statistically significant paired gains (`95% CI > 0`) beyond the physical truncation/dedup/security wins already proven here.

---

## 5. Correctness, Concurrency, Security, and Regression Evidence

Verified by [`sdk/python/tests/test_lost_in_the_middle_benchmark.py`](../../../sdk/python/tests/test_lost_in_the_middle_benchmark.py) (`18` tests) alongside the full Hanzo Python SDK test suite:

- **Determinism**: Repeated seeded runs produce byte-for-byte identical datasets, packed prompts, and paired bootstrap 95% CIs.
- **Strict Token-Budget Enforcement**: `dedup_budgeted_bookend` enforces `estimated_tokens <= budget_tokens` across all 4 budgets (`512, 1024, 2048, 4096`) and both truncation modes (`hanzo_middle_trim`, `tail_clip`) with `0` uncontrolled downstream splices.
- **Permutation Invariance & No Content Corruption**: All non-dropping policies emit exact document permutations without mutating clean passage bodies; `dedup_budgeted_bookend` preserves clean entity/answer text verbatim.
- **Concurrency & Session Isolation**: Concurrent `asyncio.gather` execution over real `hanzo_agents.memory.MemoryClient` and `ScopedMemoryClient` verifies zero cross-session vector leakage and cosine similarity round-trip fidelity.
- **Prompt-Injection Security**: Adversarial breakout strings (`</evidence_doc>`, `<<<END_CONTEXT>>>`, `SYSTEM OVERRIDE:`, spoofed `[Document 99]`) achieve `100%` boundary integrity under `dedup_budgeted_bookend` while failing audit under unescaped baselines.
- **Negative-Regime Assertions**: Explicit unit/async tests assert the exact negative behaviors in `small_context`, `misranked_retriever`, and `duplicate_heavy`.
- **SDK Regression Checks**: Verified `AIConfig.trim_by_chars`, `AIConfig.get_llm_params`, `apply_provider_patches`, and full existing Python SDK pytest suite pass with 0 regressions.

---

## 6. HIP-1347 Draft Update Snippet

```markdown
### HIP-1347 Evaluation Entry — Issue #76 (Lost in the Middle / Context Packing)
- **Status**: Offline physical & calibrated stress-test benchmark complete; live Zen/Enso endpoint run ready via `--mode live_llm`.
- **Artifacts**: `sdk/python/benchmarks/lost_in_the_middle/`, `sdk/python/tests/test_lost_in_the_middle_benchmark.py`, `docs/research/hip-1347-lost-in-the-middle/{README.md,score_table.md,raw_results.json}`.
- **Key Physical Findings (Non-Circular)**:
  - Hanzo `AIConfig.trim_by_chars(head_ratio=0.2)` induces a 39.3% truncation loss (50.0% middle survival vs 100.0% edges) on chronological prompts; `dedup_budgeted_bookend` eliminates uncontrolled truncation (+39.3pp, 95% CI [+33.2pp, +45.4pp]) and lifts token efficiency from 0.411 to 0.725 Acc/1K tok.
  - On duplicate-heavy corpora, naive `bookend_primacy_recency` drops to 0.0% under lexical interference; deduplication recovers 75.0% accuracy and cuts tokens by 10.5%.
  - Under prompt-injection payloads, unescaped `bookend_primacy_recency` drops to 0.0% boundary integrity; nonce-isolated XML escaping maintains 100.0% integrity.
- **Documented Negative Regimes**:
  - `N <= 3` unconstrained contexts: 0.0pp gain with +41 framing tokens and ~5.8 ms p50 shingling latency -> gate packing on `N > 3` or token pressure `> 75%` or untrusted inputs.
  - Mis-ranked retriever: score-based budgeted packing drops to 42.5% vs 100.0% chronological -> requires retriever score calibration or diversity preservation.
- **Integration Recommendation**: Conditional-Go for gated SDK helper (`AIConfig` / `ScopedMemoryClient`); run `--mode live_llm` against Hanzo Zen/Enso before enabling by default in `Enso`/`Zen` routing paths.
```
