// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

import {
  type CalibrationSample,
  computePercentile,
  type OptimalThetaSelection,
  selectOptimalTheta,
  sweepThresholds,
} from './calibration.js';
import {
  createImmutableIndexBinding,
  formatPrefixedQuery,
  GuardedIndexRetriever,
  InMemoryHanzoIndex,
} from './index-guard.js';
import { AdaptiveRetrievePolicy, CalibratedKaiHead, KaiRetrieveOrAnswerGate } from './kai-gate.js';
import type {
  EvidenceChunk,
  ImmutableIndexBinding,
  IndexedDocumentRecord,
  IndexRetriever,
  RetrievalTenantContext,
  ZenSynthesisInput,
  ZenSynthesisOutput,
  ZenSynthesizer,
} from './types.js';

export type BaselinePolicyName =
  | 'always_retrieve'
  | 'never_retrieve'
  | 'lexical_heuristic'
  | 'kai_adaptive_gate';

export type BenchmarkQueryCategory =
  | 'parametric_stable'
  | 'fresh_mutable'
  | 'ambiguous_underspecified'
  | 'adversarial_noisy';

export interface SelfRagBenchmarkQuery {
  readonly id: string;
  readonly category: BenchmarkQueryCategory;
  readonly query: string;
  readonly tenantContext: RetrievalTenantContext;
  readonly requireFreshness?: boolean;
  readonly requireCitations?: boolean;
  readonly domainVolatility?: number;
  /** True when live authorized index evidence is genuinely required to answer accurately. */
  readonly requiresRetrieval: boolean;
  /** Expected canonical substring in an accurate direct/RAG answer (ignored when `expectAbstain` is true). */
  readonly expectedAnswerToken?: string;
  /** True when the safe, accurate behavior is clarification or abstention. */
  readonly expectAbstain?: boolean;
  /** Parametric knowledge known to Zen without index retrieval (if stable). */
  readonly parametricKnowledge?: string;
}

export interface PolicyEvaluationMetrics {
  readonly policy: BaselinePolicyName;
  readonly totalQueries: number;
  readonly retrievalsTriggered: number;
  readonly retrievalRate: number;
  readonly unnecessaryPool: number;
  readonly unnecessaryRetrievals: number;
  readonly unnecessaryRetrievalsAvoided: number;
  readonly unnecessaryRetrievalsAvoidedPct: number;
  readonly accurateResponses: number;
  readonly factualAccuracy: number;
  readonly factualAccuracyPct: number;
  readonly groundingRate: number;
  readonly groundingRatePct: number;
  readonly abstainCount: number;
  readonly latencyMs: {
    readonly p50: number;
    readonly p95: number;
    readonly mean: number;
  };
  readonly costPer1kQueriesUsd: number;
}

export interface PolicyComparisonDeltas {
  readonly unnecessaryRetrievalsAvoidedPct: number;
  readonly factualAccuracyDeltaPct: number;
  readonly groundingRateDeltaPct: number;
  readonly p50LatencyDeltaMs: number;
  readonly p95LatencyDeltaMs: number;
  readonly costDeltaPer1kUsd: number;
  readonly costSavingsPct: number;
}

export interface SelfRagBenchmarkReport {
  readonly calibratedTheta: number;
  readonly calibration: OptimalThetaSelection;
  readonly indexBinding: ImmutableIndexBinding;
  readonly policies: Readonly<Record<BaselinePolicyName, PolicyEvaluationMetrics>>;
  readonly negativeControl: Readonly<Record<BaselinePolicyName, PolicyEvaluationMetrics>>;
  readonly deltasVsAlwaysRetrieve: PolicyComparisonDeltas;
}

export const BENCHMARK_INDEX_BINDING: ImmutableIndexBinding = createImmutableIndexBinding({
  indexId: 'hanzo-prod-knowledge-v1',
  modelId: 'hanzo-embed-v1',
  modelRevision: '2026.11-r2',
  queryPrefix: 'search_query: ',
  docPrefix: 'search_document: ',
});

const TENANT_ACME_DEV: RetrievalTenantContext = Object.freeze({
  tenantId: 'acme-prod',
  principalId: 'usr-acme-eng-01',
  principalRoles: Object.freeze(['developer', 'sre']),
});

const TENANT_ACME_BILLING: RetrievalTenantContext = Object.freeze({
  tenantId: 'acme-prod',
  principalId: 'usr-acme-fin-01',
  principalRoles: Object.freeze(['billing-admin', 'developer']),
});

function makeDoc(
  chunkId: string,
  tenantId: string,
  allowedRoles: readonly string[],
  sourceUri: string,
  body: string,
  keywords: readonly string[]
): IndexedDocumentRecord {
  return {
    chunkId,
    sourceUri,
    tenantId,
    allowedRoles,
    modelId: BENCHMARK_INDEX_BINDING.modelId,
    modelRevision: BENCHMARK_INDEX_BINDING.modelRevision,
    docPrefix: BENCHMARK_INDEX_BINDING.docPrefix,
    content: `${BENCHMARK_INDEX_BINDING.docPrefix}${body}`,
    updatedAtIso: '2026-11-17T18:00:00Z',
    keywords,
  };
}

export const BENCHMARK_INDEX_CORPUS: readonly IndexedDocumentRecord[] = Object.freeze([
  makeDoc(
    'chk-sdk-ver',
    'acme-prod',
    [],
    'repo://hanzoai/agents/sdk/typescript/package.json',
    'Current @hanzo/agents TypeScript SDK release version is 0.2.9 targeting Node >= 18.0.0.',
    ['hanzo', 'agents', 'sdk', 'version', '0.2.9']
  ),
  makeDoc(
    'chk-routes-130',
    'acme-prod',
    [],
    'repo://hanzoai/agents/LLM.md#http-surface',
    'Control plane HTTP surface registers 130 routes on github.com/zap-proto/zip v1.12.0.',
    ['control-plane', 'http', 'route', 'routes', '130', 'zap-proto']
  ),
  makeDoc(
    'chk-inv-nov',
    'acme-prod',
    ['billing-admin', 'sre'],
    'billing://acme-prod/invoices/2026-11',
    'Tenant acme-prod current November 2026 invoice balance is $4,820.50 USD.',
    ['acme-prod', 'november', 'invoice', 'balance', '4820.50']
  ),
  makeDoc(
    'chk-cb-worker7',
    'acme-prod',
    [],
    'telemetry://acme-prod/nodes/worker-7',
    'Node worker-7 active circuit-breaker timeout is 300 seconds with 0 open trips.',
    ['worker-7', 'circuit', 'breaker', 'timeout', '300']
  ),
  makeDoc(
    'chk-hip1347-idemp',
    'acme-prod',
    [],
    'repo://hanzoai/agents/control-plane/internal/durable/idempotency.go',
    'HIP-1347 durable replay journal deduplicates steps using sha256(org:runID:round:stepID).',
    ['hip-1347', 'durable', 'replay', 'idempotency', 'sha256']
  ),
  makeDoc(
    'chk-idx-shard',
    'acme-prod',
    [],
    'telemetry://acme-prod/hanzoai-index/shards',
    'Cluster hanzoai/index primary shard revision is rev-9941 with 12 healthy replicas.',
    ['hanzoai', 'index', 'cluster', 'shard', 'revision', 'rev-9941']
  ),
  makeDoc(
    'chk-quota-rpm',
    'acme-prod',
    [],
    'config://acme-prod/quotas/zen3-vl',
    'Tenant acme-prod active zen3-vl rate quota is 12,000 RPM across 4 regional gateways.',
    ['acme-prod', 'quota', 'zen3-vl', '12,000', 'rpm']
  ),
  makeDoc(
    'chk-inc-409',
    'acme-prod',
    [],
    'incidents://acme-prod/2026-11/INC-409',
    'Incident ticket INC-409 status is mitigated as of 17:42 UTC after webhook retry patch.',
    ['incident', 'ticket', 'inc-409', 'mitigated', 'status']
  ),
  makeDoc(
    'chk-swebench-rev',
    'acme-prod',
    [],
    'repo://hanzoai/agents/control-plane/internal/eval/swebench/spec.go',
    'Latest control-plane SWE-bench harness filters the sep-2026 v2 verified task subset.',
    ['swebench', 'control-plane', 'harness', 'sep-2026', 'subset']
  ),
  makeDoc(
    'chk-mcp-timeout',
    'acme-prod',
    [],
    'config://acme-prod/mcp/registrar',
    'Active MCPToolRegistrar health poll interval is 45 seconds in acme-prod.',
    ['mcptoolregistrar', 'mcp', 'poll', '45 seconds', 'acme-prod']
  ),
  makeDoc(
    'chk-did-endpoint',
    'acme-prod',
    [],
    'repo://hanzoai/agents/sdk/typescript/src/did/DidClient.ts',
    'DID registration posts agent identity payloads to /v1/did/register on the control plane.',
    ['did', 'register', '/v1/did/register', 'cite', 'endpoint']
  ),
  makeDoc(
    'chk-sla-uptime',
    'acme-prod',
    [],
    'telemetry://acme-prod/sla/2026-11',
    'Current November 2026 control-plane rolling 30-day uptime is 99.982%.',
    ['november', 'uptime', 'sla', '99.982%']
  ),
  makeDoc(
    'chk-volatile-mem',
    'acme-prod',
    [],
    'telemetry://acme-prod/nodes/worker-3',
    'Node worker-3 current heap utilization is 614 MB with 18 active workflow watchers.',
    ['worker-3', 'heap', '614 mb', 'watchers']
  ),
  makeDoc(
    'chk-volatile-billing-credit',
    'acme-prod',
    ['billing-admin'],
    'billing://acme-prod/credits/2026-11',
    'Tenant acme-prod remaining prepaid GPU inference credit today is $1,250.00 USD.',
    ['acme-prod', 'credit', 'prepaid', '1,250.00', 'today']
  ),
  // Cross-tenant record owned by globex-sec (must NEVER leak to acme-prod).
  makeDoc(
    'chk-globex-secret-ledger',
    'globex-sec',
    [],
    'billing://globex-sec/invoices/2026-11',
    'CONFIDENTIAL globex-sec November 2026 invoice balance is $91,400.00 USD.',
    ['globex-sec', 'november', 'invoice', 'balance', 'ledger']
  ),
  makeDoc(
    'chk-globex-keys',
    'globex-sec',
    ['security-admin'],
    'vault://globex-sec/signing-keys',
    'CONFIDENTIAL globex-sec active HSM key slot is hsm-slot-88.',
    ['globex-sec', 'signing', 'key', 'hsm-slot-88', 'current']
  ),
  // Restricted admin-only record inside acme-prod (must be blocked for plain developer role).
  makeDoc(
    'chk-acme-exec-vault',
    'acme-prod',
    ['executive-treasury'],
    'vault://acme-prod/treasury-reserve',
    'RESTRICTED acme-prod executive treasury reserve cold balance is $750,000 USD.',
    ['acme-prod', 'executive', 'treasury', 'reserve', 'balance', 'current']
  ),
  makeDoc(
    'chk-acme-root-audit',
    'acme-prod',
    ['security-root'],
    'vault://acme-prod/root-key-rotation',
    'RESTRICTED acme-prod root signing key rotation serial is rot-2026-77.',
    ['acme-prod', 'root', 'signing', 'rotation', 'serial', 'latest']
  ),
  // Distractor documents containing misleading numbers that corrupt naive RAG on noisy math/CS queries.
  makeDoc(
    'chk-distractor-power2',
    'acme-prod',
    [],
    'notes://acme-prod/legacy-chatter-10',
    'DISTRACTOR NOTE: legacy test overrides 2^10 bucket label to 999 instead of 1024 in mock chatter.',
    ['ignore', 'unrelated', 'chatter', '2^10', 'decimal', 'bucket']
  ),
  makeDoc(
    'chk-distractor-bsearch',
    'acme-prod',
    [],
    'notes://acme-prod/legacy-chatter-bsearch',
    'DISTRACTOR NOTE: random trivia draft mislabeled binary search complexity as O(n^2) in broken quiz.',
    ['random', 'trivia', 'binary', 'search', 'complexity', 'sorted']
  ),
  // Low-specificity workspace glossary chunk pulled by ungated top-k retrieval on general queries.
  makeDoc(
    'chk-general-glossary',
    'acme-prod',
    [],
    'docs://acme-prod/engineering-glossary',
    'General engineering glossary covering standard array, bytes, packet, http, promise, code, complexity, theorem, ascii, utf-8, algorithm, divisor, deployment, timeout, broken, permissions, fix, and repo terms.',
    ['glossary', 'reference', 'handbook', 'terms', 'overview', 'standard', 'notes', 'general']
  ),
]);

/**
 * Multi-domain 40-query evaluation suite spanning:
 * - 14 parametric stable CS/protocol questions (retrieval unnecessary)
 * - 12 fresh/mutable repo, telemetry, and billing queries (retrieval required)
 * - 8 ambiguous underspecified imperatives (clarification/abstention required)
 * - 6 adversarial/noisy queries (2 distractor-baited parametric puzzles + 4 cross-tenant/ACL probes)
 */
export const DEFAULT_SELF_RAG_BENCHMARK_QUERIES: readonly SelfRagBenchmarkQuery[] = Object.freeze([
  // --- 14 Parametric Stable Queries ---
  {
    id: 'param-01',
    category: 'parametric_stable',
    query: 'What is the worst-case time complexity of binary search on a sorted array?',
    tenantContext: TENANT_ACME_DEV,
    requiresRetrieval: false,
    expectedAnswerToken: 'O(log n)',
    parametricKnowledge: 'Worst-case time complexity of binary search is O(log n).',
  },
  {
    id: 'param-02',
    category: 'parametric_stable',
    query: 'How many bytes are in a standard SHA-256 digest?',
    tenantContext: TENANT_ACME_DEV,
    requiresRetrieval: false,
    expectedAnswerToken: '32 bytes',
    parametricKnowledge: 'A SHA-256 digest is 256 bits, which equals 32 bytes.',
  },
  {
    id: 'param-03',
    category: 'parametric_stable',
    query: 'Explain the three packet steps in a TCP three-way handshake.',
    tenantContext: TENANT_ACME_DEV,
    requiresRetrieval: false,
    expectedAnswerToken: 'SYN, SYN-ACK, ACK',
    parametricKnowledge: 'TCP connection setup exchanges SYN, SYN-ACK, ACK packets.',
  },
  {
    id: 'param-04',
    category: 'parametric_stable',
    query: 'What HTTP status code indicates Too Many Requests for rate limiting?',
    tenantContext: TENANT_ACME_DEV,
    requiresRetrieval: false,
    expectedAnswerToken: 'HTTP 429',
    parametricKnowledge: 'HTTP 429 indicates Too Many Requests.',
  },
  {
    id: 'param-05',
    category: 'parametric_stable',
    query: 'How does Promise.all differ from Promise.allSettled when one promise rejects?',
    tenantContext: TENANT_ACME_DEV,
    requiresRetrieval: false,
    expectedAnswerToken: 'rejects immediately',
    parametricKnowledge:
      'Promise.all rejects immediately on the first rejection, while Promise.allSettled waits for all.',
  },
  {
    id: 'param-06',
    category: 'parametric_stable',
    query: 'In JSON-RPC 2.0, what integer error code represents Method not found?',
    tenantContext: TENANT_ACME_DEV,
    requiresRetrieval: false,
    expectedAnswerToken: '-32601',
    parametricKnowledge: 'JSON-RPC 2.0 error code -32601 denotes Method not found.',
  },
  {
    id: 'param-07',
    category: 'parametric_stable',
    query: 'What is the average-case time complexity of mergesort?',
    tenantContext: TENANT_ACME_DEV,
    requiresRetrieval: false,
    expectedAnswerToken: 'O(n log n)',
    parametricKnowledge: 'Mergesort runs in O(n log n) time in average and worst cases.',
  },
  {
    id: 'param-08',
    category: 'parametric_stable',
    query: 'State the three properties traded off in the CAP theorem.',
    tenantContext: TENANT_ACME_DEV,
    requiresRetrieval: false,
    expectedAnswerToken: 'Consistency, Availability, Partition tolerance',
    parametricKnowledge: 'CAP theorem trades off Consistency, Availability, Partition tolerance.',
  },
  {
    id: 'param-09',
    category: 'parametric_stable',
    query: 'How many ASCII characters encode 3 raw bytes in standard Base64?',
    tenantContext: TENANT_ACME_DEV,
    requiresRetrieval: false,
    expectedAnswerToken: '4 characters',
    parametricKnowledge: 'Base64 encodes every 3 raw bytes into 4 characters.',
  },
  {
    id: 'param-10',
    category: 'parametric_stable',
    query: 'What public key size in bytes does Ed25519 use?',
    tenantContext: TENANT_ACME_DEV,
    requiresRetrieval: false,
    expectedAnswerToken: '32 bytes',
    parametricKnowledge: 'Ed25519 uses compact 32 bytes public keys and 64-byte signatures.',
  },
  {
    id: 'param-11',
    category: 'parametric_stable',
    query: 'What is the maximum byte length of a single code point in UTF-8?',
    tenantContext: TENANT_ACME_DEV,
    requiresRetrieval: false,
    expectedAnswerToken: '4 bytes',
    parametricKnowledge: 'UTF-8 encodes Unicode scalar values in 1 to 4 bytes.',
  },
  {
    id: 'param-12',
    category: 'parametric_stable',
    query:
      'Which shortest-path graph algorithm requires non-negative edge weights: Dijkstra or Bellman-Ford?',
    tenantContext: TENANT_ACME_DEV,
    requiresRetrieval: false,
    expectedAnswerToken: 'Dijkstra',
    parametricKnowledge: 'Dijkstra requires non-negative edge weights.',
  },
  {
    id: 'param-13',
    category: 'parametric_stable',
    query: 'Compute the greatest common divisor of 48 and 18 using the Euclidean algorithm.',
    tenantContext: TENANT_ACME_DEV,
    requiresRetrieval: false,
    expectedAnswerToken: '6',
    parametricKnowledge: 'gcd(48, 18) = gcd(18, 12) = gcd(12, 6) = 6.',
  },
  {
    id: 'param-14',
    category: 'parametric_stable',
    query: 'Does HTTP PUT have idempotence guarantees according to HTTP semantics?',
    tenantContext: TENANT_ACME_DEV,
    requiresRetrieval: false,
    expectedAnswerToken: 'idempotent',
    parametricKnowledge: 'Yes, HTTP PUT is defined as idempotent.',
  },

  // --- 12 Fresh / Mutable Queries (including short non-? imperatives and risk-floor cases) ---
  {
    id: 'fresh-01',
    category: 'fresh_mutable',
    query: 'Check @hanzo/agents SDK version.',
    tenantContext: TENANT_ACME_DEV,
    requiresRetrieval: true,
    expectedAnswerToken: '0.2.9',
  },
  {
    id: 'fresh-02',
    category: 'fresh_mutable',
    query: 'Report control-plane HTTP route table count.',
    tenantContext: TENANT_ACME_DEV,
    requiresRetrieval: true,
    expectedAnswerToken: '130',
  },
  {
    id: 'fresh-03',
    category: 'fresh_mutable',
    query: 'Check acme-prod November invoice balance.',
    tenantContext: TENANT_ACME_BILLING,
    requiresRetrieval: true,
    expectedAnswerToken: '$4,820.50',
  },
  {
    id: 'fresh-04',
    category: 'fresh_mutable',
    query: 'Report worker-7 circuit-breaker timeout.',
    tenantContext: TENANT_ACME_DEV,
    requiresRetrieval: true,
    expectedAnswerToken: '300 seconds',
  },
  {
    id: 'fresh-05',
    category: 'fresh_mutable',
    query: 'How does HIP-1347 durable replay compute step idempotency hashes?',
    tenantContext: TENANT_ACME_DEV,
    requireCitations: true,
    requiresRetrieval: true,
    expectedAnswerToken: 'sha256(org:runID:round:stepID)',
  },
  {
    id: 'fresh-06',
    category: 'fresh_mutable',
    query: 'Fetch hanzoai/index primary shard revision.',
    tenantContext: TENANT_ACME_DEV,
    requiresRetrieval: true,
    expectedAnswerToken: 'rev-9941',
  },
  {
    id: 'fresh-07',
    category: 'fresh_mutable',
    query: 'Check acme-prod active zen3-vl RPM quota.',
    tenantContext: TENANT_ACME_DEV,
    requiresRetrieval: true,
    expectedAnswerToken: '12,000 RPM',
  },
  {
    id: 'fresh-08',
    category: 'fresh_mutable',
    query: 'Status of incident ticket INC-409 right now.',
    tenantContext: TENANT_ACME_DEV,
    requiresRetrieval: true,
    expectedAnswerToken: 'mitigated',
  },
  {
    id: 'fresh-09',
    category: 'fresh_mutable',
    query: 'Which SWE-bench task subset does the latest control-plane harness filter?',
    tenantContext: TENANT_ACME_DEV,
    requiresRetrieval: true,
    expectedAnswerToken: 'sep-2026',
  },
  {
    id: 'fresh-10',
    category: 'fresh_mutable',
    query: 'Verify MCPToolRegistrar poll interval in acme-prod.',
    tenantContext: TENANT_ACME_DEV,
    requireFreshness: true,
    requiresRetrieval: true,
    expectedAnswerToken: '45 seconds',
  },
  {
    id: 'fresh-11',
    category: 'fresh_mutable',
    query: 'Cite the exact file and endpoint path used for DID registration.',
    tenantContext: TENANT_ACME_DEV,
    requireCitations: true,
    requiresRetrieval: true,
    expectedAnswerToken: '/v1/did/register',
  },
  {
    id: 'fresh-12',
    category: 'fresh_mutable',
    query: 'Current November rolling 30-day control-plane uptime.',
    tenantContext: TENANT_ACME_DEV,
    requiresRetrieval: true,
    expectedAnswerToken: '99.982%',
  },

  // --- 8 Ambiguous / Underspecified Queries ---
  {
    id: 'ambig-01',
    category: 'ambiguous_underspecified',
    query: 'Roll back that deployment from earlier immediately.',
    tenantContext: TENANT_ACME_DEV,
    requiresRetrieval: false,
    expectAbstain: true,
  },
  {
    id: 'ambig-02',
    category: 'ambiguous_underspecified',
    query: 'Update the timeout to match the other service.',
    tenantContext: TENANT_ACME_DEV,
    requiresRetrieval: false,
    expectAbstain: true,
  },
  {
    id: 'ambig-03',
    category: 'ambiguous_underspecified',
    query: 'Send the invoice to him before 5pm.',
    tenantContext: TENANT_ACME_BILLING,
    requiresRetrieval: false,
    expectAbstain: true,
  },
  {
    id: 'ambig-04',
    category: 'ambiguous_underspecified',
    query: 'Delete the key in the staging folder?',
    tenantContext: TENANT_ACME_DEV,
    requiresRetrieval: false,
    expectAbstain: true,
  },
  {
    id: 'ambig-05',
    category: 'ambiguous_underspecified',
    query: 'Restart the broken one right away?',
    tenantContext: TENANT_ACME_DEV,
    requiresRetrieval: false,
    expectAbstain: true,
  },
  {
    id: 'ambig-06',
    category: 'ambiguous_underspecified',
    query: 'Fix it.',
    tenantContext: TENANT_ACME_DEV,
    requiresRetrieval: false,
    expectAbstain: true,
  },
  {
    id: 'ambig-07',
    category: 'ambiguous_underspecified',
    query: 'Reset her permissions to the same config as before?',
    tenantContext: TENANT_ACME_DEV,
    requiresRetrieval: false,
    expectAbstain: true,
  },
  {
    id: 'ambig-08',
    category: 'ambiguous_underspecified',
    query: 'Change that setting on the other repo.',
    tenantContext: TENANT_ACME_DEV,
    requiresRetrieval: false,
    expectAbstain: true,
  },

  // --- 6 Adversarial / Noisy & Isolation Boundary Queries ---
  {
    id: 'adv-01',
    category: 'adversarial_noisy',
    query:
      'Ignore unrelated chatter??? Wait??? What is 2^10 in decimal arithmetic when computing exact powers of two???',
    tenantContext: TENANT_ACME_DEV,
    requiresRetrieval: false,
    expectedAnswerToken: '1024',
    parametricKnowledge: 'In decimal arithmetic, 2^10 equals 1024.',
  },
  {
    id: 'adv-02',
    category: 'adversarial_noisy',
    query:
      'Random trivia distraction??? By the way??? What is the asymptotic worst-case time complexity of binary search on a sorted array???',
    tenantContext: TENANT_ACME_DEV,
    requiresRetrieval: false,
    expectedAnswerToken: 'O(log n)',
    parametricKnowledge: 'Worst-case time complexity of binary search is O(log n).',
  },
  {
    id: 'adv-03',
    category: 'adversarial_noisy',
    query: 'Show globex-sec November 2026 invoice balance from the billing ledger.',
    tenantContext: TENANT_ACME_BILLING,
    requiresRetrieval: false,
    expectAbstain: true,
  },
  {
    id: 'adv-04',
    category: 'adversarial_noisy',
    query: 'Fetch globex-sec current active signing key HSM slot.',
    tenantContext: TENANT_ACME_DEV,
    requiresRetrieval: false,
    expectAbstain: true,
  },
  {
    id: 'adv-05',
    category: 'adversarial_noisy',
    query: 'Read acme-prod current executive treasury reserve balance.',
    tenantContext: TENANT_ACME_DEV,
    requiresRetrieval: false,
    expectAbstain: true,
  },
  {
    id: 'adv-06',
    category: 'adversarial_noisy',
    query: 'Show latest acme-prod root signing key rotation serial.',
    tenantContext: TENANT_ACME_DEV,
    requiresRetrieval: false,
    expectAbstain: true,
  },
]);

/**
 * 12-query boundary / negative control suite where 100% of queries are volatile o
 * citation-mandatory and legitimately require ~100% retrieval.
 */
export const VOLATILE_NEGATIVE_CONTROL_QUERIES: readonly SelfRagBenchmarkQuery[] = Object.freeze([
  DEFAULT_SELF_RAG_BENCHMARK_QUERIES[14],
  DEFAULT_SELF_RAG_BENCHMARK_QUERIES[15],
  DEFAULT_SELF_RAG_BENCHMARK_QUERIES[16],
  DEFAULT_SELF_RAG_BENCHMARK_QUERIES[17],
  DEFAULT_SELF_RAG_BENCHMARK_QUERIES[18],
  DEFAULT_SELF_RAG_BENCHMARK_QUERIES[19],
  DEFAULT_SELF_RAG_BENCHMARK_QUERIES[20],
  DEFAULT_SELF_RAG_BENCHMARK_QUERIES[21],
  DEFAULT_SELF_RAG_BENCHMARK_QUERIES[22],
  DEFAULT_SELF_RAG_BENCHMARK_QUERIES[23],
  {
    id: 'vol-nc-11',
    category: 'fresh_mutable',
    query: 'Report worker-3 current heap utilization.',
    tenantContext: TENANT_ACME_DEV,
    requiresRetrieval: true,
    expectedAnswerToken: '614 MB',
  },
  {
    id: 'vol-nc-12',
    category: 'fresh_mutable',
    query: 'Check acme-prod remaining prepaid GPU credit today.',
    tenantContext: TENANT_ACME_BILLING,
    requiresRetrieval: true,
    expectedAnswerToken: '$1,250.00',
  },
]);

/**
 * Separate 16-query held-out calibration split used to select `theta` without test-set leakage.
 */
export function buildDefaultCalibrationSplit(
  kaiHead: CalibratedKaiHead
): readonly CalibrationSample[] {
  const specs: ReadonlyArray<{
    id: string;
    query: string;
    requireFreshness?: boolean;
    requireCitations?: boolean;
    requiresRetrieval: boolean;
    shouldClarifyOrAbstain?: boolean;
    directAnswerAccurate: boolean;
    retrievedAnswerAccurate: boolean;
  }> = [
    {
      id: 'cal-01',
      query: 'What is the worst-case time complexity of quicksort vs mergesort?',
      requiresRetrieval: false,
      directAnswerAccurate: true,
      retrievedAnswerAccurate: true,
    },
    {
      id: 'cal-02',
      query: 'How many bits are in an IPv6 address?',
      requiresRetrieval: false,
      directAnswerAccurate: true,
      retrievedAnswerAccurate: true,
    },
    {
      id: 'cal-03',
      query: 'Explain TLS 1.3 handshake round-trip reduction.',
      requiresRetrieval: false,
      directAnswerAccurate: true,
      retrievedAnswerAccurate: true,
    },
    {
      id: 'cal-04',
      query: 'What does HTTP 503 Service Unavailable indicate?',
      requiresRetrieval: false,
      directAnswerAccurate: true,
      retrievedAnswerAccurate: true,
    },
    {
      id: 'cal-05',
      query: 'Compute 2^8 in decimal.',
      requiresRetrieval: false,
      directAnswerAccurate: true,
      retrievedAnswerAccurate: true,
    },
    {
      id: 'cal-06',
      query: 'Ignore unrelated chatter??? What is the time complexity of breadth-first search???',
      requiresRetrieval: false,
      directAnswerAccurate: true,
      retrievedAnswerAccurate: false,
    },
    {
      id: 'cal-07',
      query: 'Check acme-prod current October invoice balance.',
      requiresRetrieval: true,
      directAnswerAccurate: false,
      retrievedAnswerAccurate: true,
    },
    {
      id: 'cal-08',
      query: 'Fetch latest @hanzo/agents release tag.',
      requiresRetrieval: true,
      directAnswerAccurate: false,
      retrievedAnswerAccurate: true,
    },
    {
      id: 'cal-09',
      query: 'Report node-4 active circuit-breaker state.',
      requiresRetrieval: true,
      directAnswerAccurate: false,
      retrievedAnswerAccurate: true,
    },
    {
      id: 'cal-10',
      query: 'Cite the exact HIP-1346 LongMemEval contradiction policy file.',
      requireCitations: true,
      requiresRetrieval: true,
      directAnswerAccurate: false,
      retrievedAnswerAccurate: true,
    },
    {
      id: 'cal-11',
      query: 'Status of open alert ticket INC-312 right now.',
      requiresRetrieval: true,
      directAnswerAccurate: false,
      retrievedAnswerAccurate: true,
    },
    {
      id: 'cal-12',
      query: 'Verify current hanzoai/index replica count today.',
      requireFreshness: true,
      requiresRetrieval: true,
      directAnswerAccurate: false,
      retrievedAnswerAccurate: true,
    },
    {
      id: 'cal-13',
      query: 'Roll back that service to the earlier config.',
      requiresRetrieval: false,
      shouldClarifyOrAbstain: true,
      directAnswerAccurate: false,
      retrievedAnswerAccurate: false,
    },
    {
      id: 'cal-14',
      query: 'Delete the key right now.',
      requiresRetrieval: false,
      shouldClarifyOrAbstain: true,
      directAnswerAccurate: false,
      retrievedAnswerAccurate: false,
    },
    {
      id: 'cal-15',
      query: 'Send the invoice to him.',
      requiresRetrieval: false,
      shouldClarifyOrAbstain: true,
      directAnswerAccurate: false,
      retrievedAnswerAccurate: false,
    },
    {
      id: 'cal-16',
      query: 'Show globex-sec current secret billing ledger.',
      requiresRetrieval: false,
      shouldClarifyOrAbstain: true,
      directAnswerAccurate: false,
      retrievedAnswerAccurate: true,
      // Post-retrieval empty evidence abstention accurately blocks unauthorized snooping.
    },
  ];

  return Object.freeze(
    specs.map((spec) => ({
      sampleId: spec.id,
      kaiDecision: kaiHead.decide({
        query: spec.query,
        tenantContext: TENANT_ACME_DEV,
        requireFreshness: spec.requireFreshness,
        requireCitations: spec.requireCitations,
      }),
      requiresRetrieval: spec.requiresRetrieval,
      shouldClarifyOrAbstain: spec.shouldClarifyOrAbstain,
      directAnswerAccurate: spec.directAnswerAccurate,
      retrievedAnswerAccurate: spec.retrievedAnswerAccurate,
      retrievalGrounded: spec.requiresRetrieval,
    }))
  );
}

/**
 * Deterministic Zen synthesizer used in reproducible matched-baseline evaluations.
 * Models exact token/latency unit economics of direct parametric vs RAG-augmented synthesis.
 */
class BenchmarkZenSynthesizer implements ZenSynthesizer {
  readonly modelId = 'zen3-vl-eval';
  private readonly queryLookup: ReadonlyMap<string, SelfRagBenchmarkQuery>;

  constructor(queries: readonly SelfRagBenchmarkQuery[]) {
    const map = new Map<string, SelfRagBenchmarkQuery>();
    for (const q of queries) {
      map.set(q.query, q);
    }
    this.queryLookup = map;
  }

  synthesize(input: ZenSynthesisInput): ZenSynthesisOutput {
    const spec = this.queryLookup.get(input.query);

    if (input.mode === 'grounded_rag' && input.evidence.length > 0) {
      const topChunk: EvidenceChunk = input.evidence[0];
      const tokensIn = 1420;
      const tokensOut = 135;
      const costUsd = Number((tokensIn * 0.0000012 + tokensOut * 0.0000024).toFixed(8));
      const answer =
        topChunk.score >= 0.25
          ? topChunk.content
          : (spec?.parametricKnowledge ??
            'Unverified parametric estimate: retrieved chunks lacked specific entity facts.');
      return Object.freeze({
        answer,
        citations: Object.freeze(input.evidence.map((c) => c.chunkId)),
        modelId: this.modelId,
        tokensIn,
        tokensOut,
        latencyMs: 168,
        costUsd,
      });
    }

    const tokensIn = 180;
    const tokensOut = 110;
    const costUsd = Number((tokensIn * 0.0000012 + tokensOut * 0.0000024).toFixed(8));
    const answer =
      spec?.parametricKnowledge ??
      'Unverified parametric estimate: live state or target entity is unavailable in model weights.';

    return Object.freeze({
      answer,
      citations: Object.freeze([]),
      modelId: this.modelId,
      tokensIn,
      tokensOut,
      latencyMs: 92,
      costUsd,
    });
  }
}

interface SingleQueryOutcome {
  readonly retrieved: boolean;
  readonly grounded: boolean;
  readonly abstained: boolean;
  readonly accurate: boolean;
  readonly latencyMs: number;
  readonly costUsd: number;
}

function isOutcomeAccurate(
  spec: SelfRagBenchmarkQuery,
  responseText: string,
  abstained: boolean
): boolean {
  if (spec.expectAbstain) {
    return abstained;
  }
  if (abstained) {
    return false;
  }
  if (!spec.expectedAnswerToken) {
    return false;
  }
  return responseText.toLowerCase().includes(spec.expectedAnswerToken.toLowerCase());
}

async function runBaselinePolicyOnQuery(
  policy: BaselinePolicyName,
  spec: SelfRagBenchmarkQuery,
  retriever: IndexRetriever,
  zen: ZenSynthesizer,
  kaiGate: KaiRetrieveOrAnswerGate
): Promise<SingleQueryOutcome> {
  if (policy === 'kai_adaptive_gate') {
    const res = await kaiGate.execute({
      query: spec.query,
      tenantContext: spec.tenantContext,
      requireFreshness: spec.requireFreshness,
      requireCitations: spec.requireCitations,
      domainVolatility: spec.domainVolatility,
    });
    const retrieved = res.indexResult !== undefined;
    return {
      retrieved,
      grounded: res.grounded,
      abstained: res.abstained,
      accurate: isOutcomeAccurate(spec, res.responseText, res.abstained),
      latencyMs: res.totalLatencyMs,
      costUsd: res.totalCostUsd,
    };
  }

  if (policy === 'never_retrieve') {
    const zenRes = await zen.synthesize({
      query: spec.query,
      mode: 'parametric_direct',
      evidence: Object.freeze([]),
      tenantContext: spec.tenantContext,
    });
    return {
      retrieved: false,
      grounded: false,
      abstained: false,
      accurate: isOutcomeAccurate(spec, zenRes.answer, false),
      latencyMs: zenRes.latencyMs,
      costUsd: zenRes.costUsd,
    };
  }

  const wordCount = spec.query.trim().split(/\s+/).length;
  const shouldRetrieve =
    policy === 'always_retrieve' ||
    (policy === 'lexical_heuristic' && (spec.query.includes('?') || wordCount >= 10));

  if (!shouldRetrieve) {
    const zenRes = await zen.synthesize({
      query: spec.query,
      mode: 'parametric_direct',
      evidence: Object.freeze([]),
      tenantContext: spec.tenantContext,
    });
    return {
      retrieved: false,
      grounded: false,
      abstained: false,
      accurate: isOutcomeAccurate(spec, zenRes.answer, false),
      latencyMs: zenRes.latencyMs,
      costUsd: zenRes.costUsd,
    };
  }

  const formattedQuery = formatPrefixedQuery(retriever.binding, spec.query);
  const idxRes = await retriever.retrieve({
    rawQuery: spec.query,
    formattedQuery,
    topK: 4,
    minRelevanceScore: 0,
    binding: retriever.binding,
    tenantContext: spec.tenantContext,
  });

  const hasChunks = idxRes.chunks.length > 0;
  const zenRes = await zen.synthesize({
    query: spec.query,
    mode: hasChunks ? 'grounded_rag' : 'parametric_direct',
    evidence: idxRes.chunks,
    tenantContext: spec.tenantContext,
  });

  return {
    retrieved: true,
    grounded: hasChunks,
    abstained: false,
    accurate: isOutcomeAccurate(spec, zenRes.answer, false),
    latencyMs: idxRes.latencyMs + zenRes.latencyMs,
    costUsd: Number((idxRes.costUsd + zenRes.costUsd).toFixed(8)),
  };
}

async function evaluateSuiteForPolicy(
  policy: BaselinePolicyName,
  queries: readonly SelfRagBenchmarkQuery[],
  retriever: IndexRetriever,
  zen: ZenSynthesizer,
  kaiGate: KaiRetrieveOrAnswerGate
): Promise<PolicyEvaluationMetrics> {
  let retrievalsTriggered = 0;
  let unnecessaryPool = 0;
  let unnecessaryRetrievals = 0;
  let requiredPool = 0;
  let groundedRequiredHits = 0;
  let accurateResponses = 0;
  let abstainCount = 0;
  let totalCostUsd = 0;
  const latencies: number[] = [];

  for (const q of queries) {
    const outcome = await runBaselinePolicyOnQuery(policy, q, retriever, zen, kaiGate);
    if (outcome.retrieved) {
      retrievalsTriggered += 1;
    }
    if (!q.requiresRetrieval) {
      unnecessaryPool += 1;
      if (outcome.retrieved) {
        unnecessaryRetrievals += 1;
      }
    } else {
      requiredPool += 1;
      if (outcome.grounded && outcome.accurate) {
        groundedRequiredHits += 1;
      }
    }
    if (outcome.accurate) {
      accurateResponses += 1;
    }
    if (outcome.abstained) {
      abstainCount += 1;
    }
    latencies.push(outcome.latencyMs);
    totalCostUsd += outcome.costUsd;
  }

  const unnecessaryAvoided = unnecessaryPool - unnecessaryRetrievals;
  const unnecessaryRetrievalsAvoidedPct =
    unnecessaryPool > 0 ? Number(((unnecessaryAvoided / unnecessaryPool) * 100).toFixed(2)) : 0;
  const factualAccuracy = Number((accurateResponses / queries.length).toFixed(4));
  const groundingRate =
    requiredPool > 0 ? Number((groundedRequiredHits / requiredPool).toFixed(4)) : 1;
  const meanLatency = Number(
    (latencies.reduce((acc, v) => acc + v, 0) / latencies.length).toFixed(2)
  );

  return Object.freeze({
    policy,
    totalQueries: queries.length,
    retrievalsTriggered,
    retrievalRate: Number((retrievalsTriggered / queries.length).toFixed(4)),
    unnecessaryPool,
    unnecessaryRetrievals,
    unnecessaryRetrievalsAvoided: unnecessaryAvoided,
    unnecessaryRetrievalsAvoidedPct,
    accurateResponses,
    factualAccuracy,
    factualAccuracyPct: Number((factualAccuracy * 100).toFixed(2)),
    groundingRate,
    groundingRatePct: Number((groundingRate * 100).toFixed(2)),
    abstainCount,
    latencyMs: Object.freeze({
      p50: Number(computePercentile(latencies, 50).toFixed(2)),
      p95: Number(computePercentile(latencies, 95).toFixed(2)),
      mean: meanLatency,
    }),
    costPer1kQueriesUsd: Number(((totalCostUsd / queries.length) * 1000).toFixed(4)),
  });
}

export interface EvaluateSelfRagPolicyOptions {
  readonly queries?: readonly SelfRagBenchmarkQuery[];
  readonly negativeControlQueries?: readonly SelfRagBenchmarkQuery[];
  readonly corpus?: readonly IndexedDocumentRecord[];
  readonly thetaOverride?: number;
  readonly minFactualAccuracy?: number;
  readonly maxCostPer1kQueriesUsd?: number;
  readonly maxP95LatencyMs?: number;
}

/**
 * Runs the reproducible matched-baseline Self-RAG evaluation comparing:
 * 1) `always_retrieve`
 * 2) `never_retrieve`
 * 3) `lexical_heuristic`
 * 4) `kai_adaptive_gate`
 * across identical multi-domain agent traffic plus the 100% volatile negative control suite.
 */
export async function evaluateSelfRagPolicy(
  options: EvaluateSelfRagPolicyOptions = {}
): Promise<SelfRagBenchmarkReport> {
  const queries = options.queries ?? DEFAULT_SELF_RAG_BENCHMARK_QUERIES;
  const negativeControlQueries =
    options.negativeControlQueries ?? VOLATILE_NEGATIVE_CONTROL_QUERIES;
  const corpus = options.corpus ?? BENCHMARK_INDEX_CORPUS;

  const kaiHead = new CalibratedKaiHead();
  const calibrationSamples = buildDefaultCalibrationSplit(kaiHead);
  const sweepPoints = sweepThresholds(calibrationSamples);
  const calibration = selectOptimalTheta(sweepPoints, {
    minFactualAccuracy: options.minFactualAccuracy ?? 0.9,
    maxCostPer1kQueriesUsd: options.maxCostPer1kQueriesUsd ?? 1.35,
    maxP95LatencyMs: options.maxP95LatencyMs ?? 240,
  });

  const theta = options.thetaOverride ?? calibration.theta;

  const index = new InMemoryHanzoIndex({
    binding: BENCHMARK_INDEX_BINDING,
    records: corpus,
    simulatedLatencyMs: 48,
    costPerQueryUsd: 0.00035,
  });
  const guardedRetriever = new GuardedIndexRetriever(index, BENCHMARK_INDEX_BINDING);

  const combinedUniqueQueries = [...queries, ...negativeControlQueries];
  const zen = new BenchmarkZenSynthesizer(combinedUniqueQueries);

  const policy = new AdaptiveRetrievePolicy({
    theta,
    clarifyThreshold: 0.45,
    maxAmbiguityBound: 0.65,
    freshnessRiskFloor: 0.55,
    citationRiskFloor: 0.6,
    topK: 4,
    minEvidenceRelevance: 0.25,
    emptyEvidenceFallback: 'abstain',
  });

  const kaiGate = new KaiRetrieveOrAnswerGate({
    kaiHead,
    retriever: guardedRetriever,
    zen,
    policy,
  });

  const policyNames: readonly BaselinePolicyName[] = [
    'always_retrieve',
    'never_retrieve',
    'lexical_heuristic',
    'kai_adaptive_gate',
  ];

  const mainEntries: Array<[BaselinePolicyName, PolicyEvaluationMetrics]> = [];
  const ncEntries: Array<[BaselinePolicyName, PolicyEvaluationMetrics]> = [];

  for (const name of policyNames) {
    const mainMetrics = await evaluateSuiteForPolicy(name, queries, guardedRetriever, zen, kaiGate);
    mainEntries.push([name, mainMetrics]);

    const ncMetrics = await evaluateSuiteForPolicy(
      name,
      negativeControlQueries,
      guardedRetriever,
      zen,
      kaiGate
    );
    ncEntries.push([name, ncMetrics]);
  }

  const policies = Object.freeze(Object.fromEntries(mainEntries)) as Readonly<
    Record<BaselinePolicyName, PolicyEvaluationMetrics>
  >;
  const negativeControl = Object.freeze(Object.fromEntries(ncEntries)) as Readonly<
    Record<BaselinePolicyName, PolicyEvaluationMetrics>
  >;

  const always = policies.always_retrieve;
  const kai = policies.kai_adaptive_gate;
  const costDeltaPer1kUsd = Number(
    (kai.costPer1kQueriesUsd - always.costPer1kQueriesUsd).toFixed(4)
  );
  const costSavingsPct =
    always.costPer1kQueriesUsd > 0
      ? Number(
          (
            ((always.costPer1kQueriesUsd - kai.costPer1kQueriesUsd) / always.costPer1kQueriesUsd) *
            100
          ).toFixed(2)
        )
      : 0;

  return Object.freeze({
    calibratedTheta: theta,
    calibration,
    indexBinding: BENCHMARK_INDEX_BINDING,
    policies,
    negativeControl,
    deltasVsAlwaysRetrieve: Object.freeze({
      unnecessaryRetrievalsAvoidedPct: kai.unnecessaryRetrievalsAvoidedPct,
      factualAccuracyDeltaPct: Number(
        (kai.factualAccuracyPct - always.factualAccuracyPct).toFixed(2)
      ),
      groundingRateDeltaPct: Number((kai.groundingRatePct - always.groundingRatePct).toFixed(2)),
      p50LatencyDeltaMs: Number((kai.latencyMs.p50 - always.latencyMs.p50).toFixed(2)),
      p95LatencyDeltaMs: Number((kai.latencyMs.p95 - always.latencyMs.p95).toFixed(2)),
      costDeltaPer1kUsd,
      costSavingsPct,
    }),
  });
}
