# HIP-1347 Research Report: Temporal Memory Benchmark (LongMemEval)

- **Tracking Issue**: [`hanzoai/agents#75`](https://github.com/hanzoai/agents/issues/75) (`help wanted`, `bounty`, `research`, `hip-1347`, `p2`, `test`)
- **HIP**: HIP-1347 (Eval Harnesses & Memory Policy Evaluation)
- **Scope**: Reproducible empirical evaluation of **existing unmodified Hanzo Agents Python SDK memory policies** (`MemoryClient`, `ScopedMemoryClient`, `GlobalMemoryClient`, `MemoryInterface`) over canonical `/v1/memory/*` control-plane semantics — **without adding a competing memory system**.
- **Complements**: Go control-plane KV/event evaluator in `control-plane/internal/eval/membench/` (which evaluates `storage.StorageProvider` KV + event history; this work extends HIP-1347 coverage end-to-end across the Python SDK HTTP/vector surface, scope fallback chain, and vector similarity search).

---

## 1. Primary Paper, Reference Code Summary & License Audit

### 1.1 Primary Literature & Reference Repository
- **Paper**: Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, Dong Yu, *"LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory"*, **ICLR 2025**.
  - Proceedings: https://proceedings.iclr.cc/paper_files/paper/2025/hash/d813d324dbf0598bbdc9c8e79740ed01-Abstract-Conference.html
- **Reference Code**: https://github.com/xiaowu0162/LongMemEval
- **Concise Technical Summary**: LongMemEval evaluates long-term conversational memory across five core interactive capabilities that break naive RAG and flat key-value stores: (1) **information extraction / cross-session retrieval**, (2) **knowledge updates** (superseding outdated facts across turns), (3) **temporal reasoning & event ordering**, (4) **contradiction / multi-session reconciliation**, and (5) **abstention** (refusing to answer when supporting evidence is absent or deleted). In the paper's own evaluations (which are the authors' claims and not Hanzo claims), commercial and open chat assistants exhibit 30%–60% accuracy drops on knowledge updates, temporal reasoning, and abstention when retrieval lacks temporal indexing, update-aware reconciliation, or calibrated abstention thresholds.

### 1.2 License, Dependency & Patent Audit
| Artifact / Dependency | License | Imported into Repo? | Audit Result |
|---|---|---|---|
| `xiaowu0162/LongMemEval` upstream repository | MIT | **No** (0 lines copied; clean-room implementation) | Clear — methodology referenced only |
| LongMemEval external benchmark dataset | MIT / HF Hub | **No** (original deterministic 50-case clean-room fixture) | Clear — zero external dataset downloads or license redistribution |
| Benchmark harness (`sdk/python/benchmarks/longmemeval/`) | Apache-2.0 (repo license) | Yes (original code) | Clear — Python 3.8+ standard library + existing `hanzo_agents` SDK only |
| `DeterministicHashEmbedder` | Apache-2.0 (original SHA-256 feature hashing) | Yes | Clear — zero third-party model weights, zero API keys, zero patent encumbrances |
| Test dependencies (`pytest`, `pytest-asyncio`) | MIT / Apache-2.0 | Already in `sdk/python/requirements-dev.txt` | Clear — zero new dependencies added |

**Clearance Verdict**: **PASS** — zero GPL/AGPL/LGPL/copyleft code, zero external model weights, zero third-party datasets, zero new runtime dependencies.

---

## 2. Reproducible Benchmark Methodology

### 2.1 Architecture Under Test (Zero Competing Memory Store)
Every evaluated policy drives the **unmodified** Python SDK classes in `sdk/python/hanzo_agents/memory.py`:
- `MemoryClient` (`set`, `get`, `exists`, `delete`, `list_keys`, `set_vector`, `delete_vector`, `similarity_search`)
- `ScopedMemoryClient` (`workflow`, `session`, `actor`)
- `GlobalMemoryClient` (`global_scope`)
- `MemoryInterface` (automatic context scoping + hierarchical fallback lookup)

Requests route through `ControlPlaneMemoryTransport` (`sdk/python/benchmarks/longmemeval/transport.py`), a deterministic, `asyncio.Lock`-guarded in-memory HTTP transport wired via `HanzoAgentsClient._async_request` that faithfully mirrors the Go control-plane handlers in:
- `control-plane/internal/handlers/memory.go` (`resolveScope`, `getScopeID`, `SetMemoryHandler`, `GetMemoryHandler` hierarchical fallback `workflow -> session -> actor -> global`, `DeleteMemoryHandler` 404/204, `ListMemoryHandler`)
- `control-plane/internal/handlers/vector_memory.go` (`SetVectorHandler`, `DeleteVectorHandler`, `SimilaritySearchHandler` unthresholded cosine `top_k` + metadata filters)

### 2.2 Dataset & Policies
- **Dataset** (`sdk/python/benchmarks/longmemeval/dataset.py`, `--seed 1347`): **50 deterministic scenarios** (10 per LongMemEval dimension):
  1. `knowledge_updates` (10): 6 multi-turn sequential overwrites + 4 cross-scope update scenarios where an older `session`/`workflow` record precedes a newer authoritative `actor` update.
  2. `contradiction_handling` (10): 5 cross-scope contradictions + 5 verified-observation vs low-confidence-hypothesis conflicts across sessions.
  3. `temporal_ordering` (10): 3–4 step timestamped timelines (5 with out-of-order / backfilled arrival to verify event-time vs arrival-time sorting).
  4. `abstention` (10): 5 never-stored facts with 4 in-scope distractors + 3 explicitly deleted facts (`action="delete"`) + 2 cross-actor-only facts (verifying both abstention and strict tenant isolation).
  5. `cross_session_retrieval` (10): 8 actor-scoped multi-session recall cases queried from a fresh `session_3` + 2 `global` org-knowledge fallback cases.
- **Policies Evaluated** (`sdk/python/benchmarks/longmemeval/policies.py`):
  1. `kv_scoped`: Strict `interface.session(session_id)` KV.
  2. `kv_fallback_chain`: Native `MemoryInterface.get()` (`workflow -> session -> actor -> global`) over scoped writes.
  3. `vector_top_k`: Unthresholded `set_vector` + `similarity_search(top_k=5)` mirroring Hanzo's current vector search API.
  4. `vector_thresholded`: `set_vector` + coordinated `delete_vector` + client-side cosine cutoff (`min_score=0.45`) + recency tie-breaking.
  5. `hybrid_scope_policy`: Multi-scope inspection across `workflow`/`session`/`actor`/`global`, verification-tier + timestamp conflict resolution, event-key timeline storage (`list_keys` + `get`), and thresholded vector fallback — implemented **entirely** over existing `MemoryInterface` primitives.

### 2.3 Exact Reproduction Commands
From repository root:
```bash
cd sdk/python
PYTHONPATH=. python3 -m benchmarks.longmemeval \
  --seed 1347 \
  --sweep-thresholds \
  --output ../../docs/research/hip-1347-longmemeval/raw_results.json \
  --markdown ../../docs/research/hip-1347-longmemeval/score_table.md
```
Run correctness, concurrency, security, and regression tests:
```bash
cd sdk/python
PYTHONPATH=. python3 -m pytest \
  tests/test_longmemeval_benchmark.py \
  tests/test_memory_client_core.py \
  tests/test_base_memory.py \
  tests/test_memory_flow_core.py \
  tests/test_memory_events.py \
  tests/test_memory_performance.py \
  -o addopts="-ra -q"
```
Run complementary Go control-plane membench tests:
```bash
cd control-plane
go test -v ./internal/eval/membench/...
```

**Execution Environment** (recorded in `raw_results.json`):
- Python `3.12.16`, `Linux-6.18.57-x86_64-with-glibc2.39`, 2 vCPU, base commit `2e8494c`.

---

## 3. Falsifiable Hypotheses & Measured Results

### 3.1 Falsifiable Hypotheses
- **H1 (KV Scope Isolation vs History Loss)**: Strict session-scoped KV (`kv_scoped`) achieves 0% on `cross_session_retrieval` and 0% on `temporal_ordering` because session isolation blocks cross-session reads and in-place `set()` overwrites erase earlier timeline states.
- **H2 (Hierarchical Precedence Shadowing)**: Native `MemoryInterface.get()` (`kv_fallback_chain`) solves `cross_session_retrieval` (100%) but suffers >= 4 stale-shadowing errors on `knowledge_updates` and fails `contradiction_handling` because `memory.go:189` resolves `workflow -> session -> actor -> global` strictly by scope precedence without comparing `updated_at` timestamps or surfacing cross-scope conflicts.
- **H3 (Unthresholded Vector Abstention Failure)**: Current `MemoryClient.similarity_search(top_k=5)` (`vector_top_k`) yields **0.00% Abstention F1 (100% Unanswerable FPR)** because neither the Python SDK nor `control-plane/internal/handlers/vector_memory.go` enforces a `min_score` cutoff; adding a calibrated threshold (`vector_thresholded`, `min_score=0.45`) raises Abstention F1 to >= 90% and Exact Match by >= +40 percentage points, while an overly strict threshold (`min_score=0.75`) degrades recall via false abstentions.
- **H4 (Hybrid Policy Over Existing Primitives)**: Coordinating existing Hanzo `ScopedMemoryClient`, `GlobalMemoryClient`, `list_keys`, and thresholded `similarity_search` (`hybrid_scope_policy`) resolves updates, contradictions, temporal ordering, abstention, and cross-session retrieval with **0 cross-actor leaks**, **0 stale-shadowing errors**, and lower recall latency and token footprint than pure vector search.

### 3.2 Reproducible Score Table per Policy (Seed `1347`, 50 Cases)
Measured directly from [`raw_results.json`](./raw_results.json) (also saved standalone in [`score_table.md`](./score_table.md)):

| Policy | Exact Match | Recall@K | Abstention F1 | Unanswerable FPR | Contradiction Det. | Temporal Order | Stale Shadowing | Cross-Actor Leaks | Avg Tokens | p50 (ms) | p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `kv_scoped` | 50.00% | 50.00% | 50.00% | 0.00% | 0.00% | 0.00% | 5 | 0 | 5.7 | 0.029 | 0.091 |
| `kv_fallback_chain` | 56.00% | 56.00% | 100.00% | 0.00% | 0.00% | 0.00% | 12 | 0 | 11.4 | 0.051 | 0.091 |
| `vector_top_k` | 34.00% | 80.00% | 0.00% | 100.00% | 0.00% | 0.00% | 16 | 0 | 54.4 | 0.515 | 0.771 |
| `vector_thresholded` | 88.00% | 98.00% | 100.00% | 0.00% | 0.00% | 100.00% | 6 | 0 | 27.3 | 0.519 | 0.794 |
| `hybrid_scope_policy` | 100.00% | 100.00% | 100.00% | 0.00% | 100.00% | 100.00% | 0 | 0 | 24.0 | 0.178 | 0.560 |

#### Per-Category Exact Match Breakdown
| Policy | Knowledge Updates | Contradiction Handling | Temporal Ordering | Abstention | Cross-Session Retrieval |
|---|---:|---:|---:|---:|---:|
| `kv_scoped` | 100.00% (10/10) | 50.00% (5/10) | 0.00% (0/10) | 100.00% (10/10) | 0.00% (0/10) |
| `kv_fallback_chain` | 60.00% (6/10) | 20.00% (2/10) | 0.00% (0/10) | 100.00% (10/10) | 100.00% (10/10) |
| `vector_top_k` | 40.00% (4/10) | 50.00% (5/10) | 0.00% (0/10) | 0.00% (0/10) | 80.00% (8/10) |
| `vector_thresholded` | 100.00% (10/10) | 40.00% (4/10) | 100.00% (10/10) | 100.00% (10/10) | 100.00% (10/10) |
| `hybrid_scope_policy` | 100.00% (10/10) | 100.00% (10/10) | 100.00% (10/10) | 100.00% (10/10) | 100.00% (10/10) |

#### Before/After Threshold Sensitivity Sweep (`vector_thresholded`, `top_k=8`)
| `min_score` | Exact Match | Recall@K | Abstention Precision | Abstention Recall | Abstention F1 | Unanswerable FPR | Avg Tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.00` | 70.00% | 80.00% | 100.00% | 10.00% | 18.18% | 90.00% | 31.4 |
| `0.25` | 88.00% | 98.00% | 100.00% | 100.00% | 100.00% | 0.00% | 27.3 |
| `0.35` | 88.00% | 98.00% | 100.00% | 100.00% | 100.00% | 0.00% | 27.3 |
| `0.45` | 88.00% | 98.00% | 100.00% | 100.00% | 100.00% | 0.00% | 27.3 |
| `0.55` | 84.00% | 94.00% | 90.91% | 100.00% | 95.24% | 0.00% | 26.7 |
| `0.65` | 62.00% | 70.00% | 58.82% | 100.00% | 74.07% | 0.00% | 18.1 |
| `0.75` | 42.00% | 46.00% | 27.03% | 100.00% | 42.55% | 0.00% | 7.4 |

All four hypotheses (H1–H4) are empirically confirmed.

---

## 4. Documented Failure Modes & Negative Results

1. **Silent Stale Shadowing in `MemoryInterface.get()` (`kv_fallback_chain`, 12 shadowing errors)**:
   - In `control-plane/internal/handlers/memory.go:189-202` and `sdk/python/hanzo_agents/base_memory.py:83-88`, unscoped `get()` returns the first hit in fixed order `workflow -> session -> actor -> global` without inspecting `updated_at`.
   - Whenever an older `session` or `workflow` record exists for key `K`, a newer authoritative update written to `actor` scope is **silently shadowed** (causing `kv_fallback_chain` to drop to **60% on `knowledge_updates`** and **20% on `contradiction_handling`**).
2. **Catastrophic Abstention Failure in Unthresholded `similarity_search()` (`vector_top_k`, 100% FPR)**:
   - Neither `MemoryClient.similarity_search()` (`sdk/python/hanzo_agents/memory.py:324-353`), `BaseMemory.similarity_search()` (`base_memory.py:189-244`), nor `VectorSearchRequest` (`control-plane/internal/handlers/vector_memory.go:26-31`) accepts a `min_score` parameter.
   - As a result, `vector_top_k` retrieves irrelevant distractors on 100% of unanswerable queries (`Abstention F1 = 0.00%`), hallucinates deleted facts (because naive vector policies lack coordinated `delete_vector` tracking), and inflates prompt token usage to **54.4 tokens/query** (2.27x `hybrid_scope_policy`).
3. **Recency-Only Thresholded Vector Search Still Fails Contradictions (`vector_thresholded`, 40% on `contradiction_handling`, 0% Contradiction Detection)**:
   - Even with `min_score=0.45`, pure timestamp tie-breaking blindly prefers a newer low-confidence chat hypothesis (`conf=0.35`) over an earlier verified tool observation (`conf=0.98`) in `ch-06..ch-10`, and lexical similarity skew pruned the newer authoritative `redis-cluster` record in `ch-04`.
   - Furthermore, raising `min_score` to `0.75` causes severe over-abstention (`Exact Match = 42.00%`, `Abstention Precision = 27.03%`).
4. **In-Place KV Overwrite Destroys Temporal Ordering (`kv_scoped` & `kv_fallback_chain`, 0% on `temporal_ordering`)**:
   - `SetMemoryHandler` overwrites `(scope, scope_id, key)` in place; previous values are only recorded on the change-event stream and are inaccessible via `MemoryClient.get()`.
5. **Pre-Existing Bug Audited in `MemoryClient.exists()` (`sdk/python/hanzo_agents/memory.py:233-250`)**:
   - `MemoryClient.exists()` wraps `await self.get(...)` in `try ... return True / except Exception: return False`. However, `MemoryClient.get()` explicitly catches HTTP 404 and returns `default` (`None`) without raising (`memory.py:214-215`). Consequently, **`MemoryClient.exists(key)` always returns `True` for missing keys** (verified in `test_documents_existing_memory_client_exists_404_quirk`). By contrast, `BaseMemory.exists()` (`base_memory.py:120`) correctly checks `bool(await self._read(...))`.
6. **Harness Scope Limitation (Honest Boundary)**:
   - `DeterministicHashEmbedder` measures lexical/morphological similarity reproducibly offline; dense neural embeddings (served via Enso/Zen in production) will exhibit higher paraphrase recall and a shifted optimal `min_score` operating band. The benchmark harness decouples the embedder interface so production neural embeddings can be swapped in directly without changing the benchmark or policy contracts.

---

## 5. Correctness, Concurrency, Security & Regression Evidence

All verification suites pass with **0 regressions**:

| Suite | Command | Result | Coverage |
|---|---|---|---|
| New LongMemEval Benchmark Suite | `pytest tests/test_longmemeval_benchmark.py` | **9 / 9 passed** | Wire contract fidelity, `MemoryClient.exists()` 404 audit, H1–H4 hypothesis & negative-result checks, threshold sweep, 20-way `asyncio.gather` concurrency, cross-actor & cross-session KV+vector isolation, bit-for-bit seed determinism, CLI JSON/Markdown output |
| Existing Python SDK Memory Suite | `pytest tests/test_memory_*.py tests/test_base_memory.py` | **70 / 70 passed** | `MemoryClient`, `ScopedMemoryClient`, `GlobalMemoryClient`, `MemoryInterface`, `BaseMemory`, `MemoryEventClient`, memory performance |
| Combined Python Memory Verification | `pytest tests/test_longmemeval_benchmark.py tests/test_memory_*.py tests/test_base_memory.py` | **79 / 79 passed** (7.20s) | Zero regressions |
| Go Control-Plane Membench Suite | `go test -v ./internal/eval/membench/...` | **5 / 5 passed** (0.017s) | Go `MemoryBackend` LongMemEval + Lost-in-the-Middle packer |

**Security / Tenant Isolation Verification**:
- Explicit unit tests confirm `actor-bob` cannot read, list, delete, or similarity-search `actor-alice`'s KV or vector records even when querying with identical keys and identical embedding vectors.
- All 5 benchmark policies recorded **`cross_actor_leaks = 0`** across all 50 cases (including `ab-09` and `ab-10` adversarial cross-actor secret probes).
- Documented operational invariant: `GlobalMemoryClient` (`scope="global"`) is intentionally shared across all actors and must strictly be restricted to public org-level policies/runbooks, never tenant/user-private memory.

---

## 6. HIP-1347 Integration Recommendation

Consistent with Hanzo's fixed canonical architecture (`vfs -> Hanzo S3 -> Visor`, Cloud lifecycle & billing, Enso model routing, Zen generation, Kai decision orchestration), **do not add a competing memory engine**. Instead, apply **four minimal, backwards-compatible improvements** to the existing memory primitives if maintainers open an implementation bounty on `#75`:

1. **Add optional `min_score: Optional[float] = None` to `similarity_search()`** in `control-plane/internal/handlers/vector_memory.go` (`VectorSearchRequest`), `sdk/python/hanzo_agents/memory.py`, and `sdk/python/hanzo_agents/base_memory.py`. Eliminates the 100% unanswerable false-positive rate with zero breaking changes (`min_score=None` preserves existing behavior).
2. **Fix `MemoryClient.exists()` 404 handling** (`sdk/python/hanzo_agents/memory.py:233-250`) by using a unique sentinel default (`_MISSING = object()`) so `return (await self.get(key, default=_MISSING, ...)) is not _MISSING`, matching `BaseMemory.exists()`.
3. **Add optional `include_metadata: bool = False` / `resolve_by: Literal["precedence", "newest"] = "precedence"` on `GetMemoryRequest`** (`control-plane/internal/handlers/memory.go`) so callers that need cross-scope contradiction detection or recency-ordered resolution (e.g., Kai orchestration loops) can opt in without issuing 4 separate HTTP round-trips.
4. **Wire this benchmark into CI as a non-flaky regression gate** (`pytest tests/test_longmemeval_benchmark.py` runs in < 1s offline with zero external dependencies) alongside `control-plane/internal/eval/membench/` to guard future HIP-1347 memory policy changes.
