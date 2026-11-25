# HIP-1347: Adaptive Retrieve-or-Answer Policy with Kai (Self-RAG)

Issue: `hanzoai/agents#74`  
Package: `@hanzo/agents` (`sdk/typescript`)  
License: Apache-2.0

## 1. Primary Research Reference and Clean-Room License Audit

- **Primary paper**: Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi, *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*, ICLR 2024 (`https://proceedings.iclr.cc/paper_files/paper/2024/hash/25f7be9694d7b32d5cc670927b8091e1-Abstract-Conference.html`).
- **License audit**: Upstream reference code (`AkariAsai/self-rag`) is MIT-licensed, while its released checkpoints carry Llama-2 model weight restrictions. This implementation in `sdk/typescript/src/retrieval/` is a clean-room Apache-2.0 TypeScript implementation written from the decision-theoretic problem formulation and Hanzo's HIP-1347 architecture. Zero source files, prompt templates, or reflection-token weights were copied from external repositories.
- **Architectural mapping (Kai vs `hanzoai/index` vs Zen)**:
  - Self-RAG (ICLR 2024) trains a single autoregressive generator to emit inline special reflection tokens (`[Retrieve]`, `[IsREL]`, `[IsSUP]`, `[IsUSE]`), coupling discrete routing policy with free-form token decoding.
  - In Hanzo's architecture, routing and generation are strictly separated:
    1. **Kai (`KaiDecisionHead`)**: Non-generative discrete decision head (`outputTokens === 0`) returning calibrated 3-simplex probabilities `{ retrieve, answerDirect, clarifyOrAbstain }` and risk signals `{ freshnessRisk, citationRisk, ambiguityRisk }`. Any non-zero `outputTokens` or free-form prose field on Kai throws `KaiArchitectureViolationError`.
    2. **`hanzoai/index` (`IndexRetriever` / `GuardedIndexRetriever`)**: Enforces an immutable frozen `ImmutableIndexBinding` (`indexId`, `modelId`, `modelRevision`, `queryPrefix`, `docPrefix`, optional `dimension`) and executes mandatory pre-hit `tenantId` and role ACL filtering prior to similarity scoring or top-k truncation.
    3. **Zen (`ZenSynthesizer` / `AIClientZenSynthesizer`)**: Generative synthesizer (`zen3-vl`) invoked only after Kai approves `answer_direct` or `retrieve` with ACL-verified `EvidenceChunk[]`.

## 2. Matched Baselines

All four policies are evaluated by `evaluateSelfRagPolicy()` over the identical 40-query multi-domain workload, identical `BENCHMARK_INDEX_BINDING` and 21-record multi-tenant `InMemoryHanzoIndex` corpus, identical token/index cost model, and identical `BenchmarkZenSynthesizer`:

1. `always_retrieve`: Unconditional top-k RAG baseline querying `hanzoai/index` on 100% of turns before Zen synthesis.
2. `never_retrieve`: Direct parametric Zen baseline (`0` index calls).
3. `lexical_heuristic`: Uncalibrated surface heuristic retrieving when the query contains `?` or has `>= 10` words.
4. `kai_adaptive_gate`: Calibrated Kai 3-simplex gate (`theta = 0.50` selected on a separate 16-sample calibration split), freshness/citation risk floors, ambiguity safety routing, and post-retrieval insufficient-evidence abstention.

## 3. Falsifiable Hypothesis and Failure Criteria

- **Hypothesis**: On mixed multi-domain agent traffic (stable parametric knowledge, fresh mutable repo/billing/telemetry state, underspecified imperatives, and adversarial/cross-tenant queries), a calibrated Kai decision gate avoids `>= 80%` of unnecessary retrievals and lowers total USD cost per 1,000 queries by `>= 45%` relative to `always_retrieve` without reducing factual accuracy or mutable-query grounding rate, while maintaining `>= 95%` retrieval rate on a 100% volatile negative control.
- **Disconfirmation criteria**: The hypothesis fails if (a) unnecessary retrievals avoided drops below `80%`, (b) factual accuracy on mixed traffic falls below `always_retrieve`, (c) cost savings fall below `45%`, or (d) `kai_adaptive_gate` suppresses needed lookups (`< 95%` retrieval rate) on the 100% volatile boundary suite.

## 4. Bounded Deliverable

Scope is confined to `sdk/typescript/`:

- `sdk/typescript/src/retrieval/types.ts`: Domain interfaces and typed error hierarchy (`KaiValidationError`, `KaiArchitectureViolationError`, `IndexContractMismatchError`, `TenantIsolationError`, `CalibrationConstraintError`).
- `sdk/typescript/src/retrieval/index-guard.ts`: `createImmutableIndexBinding`, `assertIndexBindingMatch`, `enforcePreHitTenantAndAclFilter`, `InMemoryHanzoIndex`, and `GuardedIndexRetriever`.
- `sdk/typescript/src/retrieval/kai-gate.ts`: `normalizeSimplex`, `softmaxToSimplex`, `validateKaiDecisionOutput`, `CalibratedKaiHead`, `AdaptiveRetrievePolicy`, `KaiRetrieveOrAnswerGate`, and `AIClientZenSynthesizer`.
- `sdk/typescript/src/retrieval/calibration.ts`: `sweepThresholds` and `selectOptimalTheta` under `minFactualAccuracy`, cost, and p95 latency constraints.
- `sdk/typescript/src/retrieval/evaluation.ts`: Deterministic matched-baseline harness `evaluateSelfRagPolicy` plus 40-query main suite and 12-query volatile negative control.
- `sdk/typescript/src/index.ts`: Root barrel re-export (`./retrieval/index.js`) preserving all existing `@hanzo/agents` exports.
- `sdk/typescript/tests/kai-self-rag.test.ts`: 14 Vitest test suites covering boundaries, simplex safety, pre-hit ACL isolation, risk floors, calibration, 36-way `Promise.all` concurrency, and benchmark reproducibility.

## 5. Safety, Concurrency, and Isolation Guarantees

- **Simplex and numerical validation**: `validateKaiDecisionOutput` rejects `NaN`, `+/-Infinity`, negative weights, and unnormalized sums outside `simplexTolerance`; `softmaxToSimplex` uses max-shift log-sum-exp stabilization.
- **Immutable index provenance**: `ImmutableIndexBinding` is frozen via `Object.freeze`. Mismatches in `modelId`, `modelRevision`, `queryPrefix`, `docPrefix`, or vector `dimension` on either queries or stored chunks throw `IndexContractMismatchError`.
- **Pre-hit tenant and role ACL enforcement**: `enforcePreHitTenantAndAclFilter` strips cross-tenant (`record.tenantId !== tenantContext.tenantId`) and unauthorized role records before scoring, ensuring unauthorized chunks never displace valid tenant hits in `topK` or leak content/URIs.
- **Stateless async concurrency**: `KaiRetrieveOrAnswerGate` holds immutable configuration only; per-request state stays on the call stack and is verified across 36 concurrent multi-tenant `Promise.all` executions.

## 6. Reproducible Raw Measurements

Reproduce with:

```bash
pnpm --filter @hanzo/agents test
```

### Main 40-Query Multi-Domain Benchmark (`calibratedTheta = 0.50`)

| Policy | Retrieval Rate | Unnecessary Avoided | Factual Accuracy | Grounding Rate | p50 / p95 Latency (ms) | Cost / 1k Queries (USD) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always_retrieve` | 100.00% (40/40) | 0.00% (0/28) | 52.50% (21/40) | 100.00% | 216.00 / 216.00 | $2.3393 |
| `never_retrieve` | 0.00% (0/40) | 100.00% (28/28) | 40.00% (16/40) | 0.00% | 92.00 / 92.00 | $0.4800 |
| `lexical_heuristic` | 57.50% (23/40) | 28.57% (8/28) | 30.00% (12/40) | 25.00% | 216.00 / 216.00 | $1.5326 |
| `kai_adaptive_gate` | 40.00% (16/40) | 85.71% (24/28) | 92.50% (37/40) | 100.00% | 98.00 / 222.00 | $1.1125 |

**Deltas (`kai_adaptive_gate` vs `always_retrieve`)**:
- Unnecessary retrievals avoided: `85.71%` (`24` of `28` non-retrieval queries skipped; the remaining `4` are mutable-framed unauthorized tenant/ACL probes that Kai legitimately routes to the index, where pre-hit ACL filtering returns `0` chunks and triggers safe abstention).
- Factual accuracy delta: `+40.00` percentage points (`92.50%` vs `52.50%`), driven by avoiding distractor chunks on noisy parametric queries and abstaining on underspecified or unauthorized prompts.
- Median latency (`p50`): `-118.00 ms` (`98.00 ms` vs `216.00 ms`); `p95` adds `+6.00 ms` from the Kai decision pass on retrieved turns.
- Total cost per 1,000 queries: `-$1.2268` (`52.44%` reduction, from `$2.3393` to `$1.1125`).

### Boundary / Negative Control: 100% Volatile Queries (12 Queries)

| Policy | Retrieval Rate | Factual Accuracy | Grounding Rate | p50 / p95 Latency (ms) | Cost / 1k Queries (USD) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `always_retrieve` | 100.00% (12/12) | 100.00% (12/12) | 100.00% | 216.00 / 216.00 | $2.3780 |
| `never_retrieve` | 0.00% (0/12) | 0.00% (0/12) | 0.00% | 92.00 / 92.00 | $0.4800 |
| `lexical_heuristic` | 16.67% (2/12) | 16.67% (2/12) | 16.67% | 92.00 / 216.00 | $0.7963 |
| `kai_adaptive_gate` | 100.00% (12/12) | 100.00% (12/12) | 100.00% | 222.00 / 222.00 | $2.3980 |

On 100% volatile traffic, `kai_adaptive_gate` triggers `100.00%` (`12/12`) retrieval with `100.00%` grounding and `100.00%` factual accuracy (whereas `lexical_heuristic` misses short imperative status queries lacking `?`), incurring `+6.00 ms` latency and `+$0.0200 / 1k` Kai overhead when zero retrievals can be skipped.
