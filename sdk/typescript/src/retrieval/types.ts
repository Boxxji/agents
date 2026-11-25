// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

/**
 * Discrete routing actions produced by Kai before Zen generates an answer.
 *
 * Architectural boundary (HIP-1347):
 * - Kai (`KaiDecisionHead`) is a structured non-generative decision head (`outputTokens === 0`).
 * - `hanzoai/index` (`IndexRetriever`) executes retrieval under an immutable embedding/prefix
 *   contract with mandatory pre-hit `tenantId` and role ACL filtering.
 * - Zen (`ZenSynthesizer`) synthesizes prose responses from grounded chunks or parametric
 *   knowledge without invoking the index directly.
 */
export type KaiAction = 'retrieve' | 'answer_direct' | 'clarify_or_abstain';

/**
 * Calibrated probability distribution on the 3-simplex (`retrieve + answerDirect + clarifyOrAbstain === 1`).
 */
export interface KaiSimplexProbabilities {
  readonly retrieve: number;
  readonly answerDirect: number;
  readonly clarifyOrAbstain: number;
}

/**
 * Auxiliary calibrated risk floors in `[0, 1]` accompanying Kai's discrete choice distribution.
 */
export interface KaiRiskSignals {
  /** Temporal or mutable-state volatility signal in `[0, 1]`. */
  readonly freshnessRisk: number;
  /** Verifiable source or exact-identifier citation requirement in `[0, 1]`. */
  readonly citationRisk: number;
  /** Underspecification or missing-referent ambiguity signal in `[0, 1]`. */
  readonly ambiguityRisk: number;
  /** Estimated risk that top-k retrieval pulls noisy distractor chunks in `[0, 1]`. */
  readonly distractorRisk?: number;
}

/**
 * Caller identity and authorization scope enforced pre-hit by `hanzoai/index`.
 */
export interface RetrievalTenantContext {
  readonly tenantId: string;
  readonly principalId: string;
  readonly principalRoles: readonly string[];
}

/**
 * Input context evaluated by Kai prior to any index lookup or Zen token generation.
 */
export interface RetrievalDecisionContext {
  readonly query: string;
  readonly tenantContext: RetrievalTenantContext;
  readonly conversationTurn?: number;
  readonly requireFreshness?: boolean;
  readonly requireCitations?: boolean;
  readonly domainVolatility?: number;
  readonly metadata?: Readonly<Record<string, string | number | boolean>>;
}

/**
 * Raw unvalidated payload returned by a `KaiDecisionHead` implementation.
 */
export interface KaiRawDecision {
  readonly probabilities: KaiSimplexProbabilities;
  readonly riskSignals?: Partial<KaiRiskSignals>;
  readonly outputTokens?: number;
  readonly modelId?: string;
  readonly latencyMs?: number;
  readonly costUsd?: number;
}

/**
 * Validated, normalized Kai decision output guaranteed to have `outputTokens === 0`.
 */
export interface KaiDecisionOutput {
  readonly probabilities: KaiSimplexProbabilities;
  readonly riskSignals: KaiRiskSignals;
  readonly outputTokens: 0;
  readonly modelId: string;
  readonly latencyMs: number;
  readonly costUsd: number;
}

/**
 * Non-generative structured decision head interface (Hanzo Kai).
 */
export interface KaiDecisionHead {
  readonly modelId: string;
  decide(context: RetrievalDecisionContext): Promise<KaiRawDecision> | KaiRawDecision;
}

/**
 * Immutable index and embedding provenance contract for `hanzoai/index`.
 * All fields are frozen on construction to prevent cross-model or cross-prefix drift.
 */
export interface ImmutableIndexBinding {
  readonly indexId: string;
  readonly modelId: string;
  readonly modelRevision: string;
  readonly queryPrefix: string;
  readonly docPrefix: string;
  readonly dimension?: number;
}

/**
 * Stored document chunk inside an index backend before tenant/ACL pre-filtering.
 */
export interface IndexedDocumentRecord {
  readonly chunkId: string;
  readonly sourceUri: string;
  readonly tenantId: string;
  readonly ownerId?: string;
  /** Empty array grants read access to any authenticated principal within `tenantId`. */
  readonly allowedRoles: readonly string[];
  readonly modelId: string;
  readonly modelRevision: string;
  readonly docPrefix: string;
  readonly content: string;
  readonly updatedAtIso: string;
  readonly keywords?: readonly string[];
  readonly embedding?: readonly number[];
}

/**
 * Sanitized evidence chunk delivered to Zen after passing immutable binding and pre-hit ACL checks.
 */
export interface EvidenceChunk {
  readonly chunkId: string;
  readonly sourceUri: string;
  readonly content: string;
  readonly score: number;
  readonly updatedAtIso: string;
}

/**
 * Pre-hit access-control and contract audit counters recorded during a single index query.
 */
export interface IndexPreFilterAudit {
  readonly candidatesScanned: number;
  readonly authorizedCandidates: number;
  readonly filteredOutByTenant: number;
  readonly filteredOutByAcl: number;
}

/**
 * Request dispatched to `hanzoai/index` when the Kai gate selects `retrieve`.
 */
export interface IndexSearchRequest {
  readonly rawQuery: string;
  readonly formattedQuery: string;
  readonly topK: number;
  readonly minRelevanceScore?: number;
  readonly queryEmbedding?: readonly number[];
  readonly binding: ImmutableIndexBinding;
  readonly tenantContext: RetrievalTenantContext;
}

/**
 * Result returned by `hanzoai/index` after pre-hit filtering and top-k scoring.
 */
export interface IndexSearchResult {
  readonly chunks: readonly EvidenceChunk[];
  readonly preFilterAudit: IndexPreFilterAudit;
  readonly latencyMs: number;
  readonly costUsd: number;
}

/**
 * Evidence retriever boundary (`hanzoai/index`).
 */
export interface IndexRetriever {
  readonly binding: ImmutableIndexBinding;
  retrieve(request: IndexSearchRequest): Promise<IndexSearchResult> | IndexSearchResult;
}

/**
 * Synthesis mode passed to Zen (`grounded_rag` with retrieved chunks or `parametric_direct`).
 */
export type ZenSynthesisMode = 'grounded_rag' | 'parametric_direct';

/**
 * Input passed to Zen when generating a final user-facing answer.
 */
export interface ZenSynthesisInput {
  readonly query: string;
  readonly mode: ZenSynthesisMode;
  readonly evidence: readonly EvidenceChunk[];
  readonly tenantContext: RetrievalTenantContext;
}

/**
 * Generative synthesis output returned by Zen.
 */
export interface ZenSynthesisOutput {
  readonly answer: string;
  readonly citations: readonly string[];
  readonly modelId: string;
  readonly tokensIn: number;
  readonly tokensOut: number;
  readonly latencyMs: number;
  readonly costUsd: number;
}

/**
 * Generative answer synthesizer boundary (Hanzo Zen).
 */
export interface ZenSynthesizer {
  readonly modelId: string;
  synthesize(input: ZenSynthesisInput): Promise<ZenSynthesisOutput> | ZenSynthesisOutput;
}

/**
 * Auditable explanation for why the gate selected a routing branch.
 */
export type GateDecisionReason =
  | 'clarify_safety_bound'
  | 'freshness_risk_floor'
  | 'citation_risk_floor'
  | 'kai_probability_above_theta'
  | 'kai_answer_direct'
  | 'post_retrieval_insufficient_evidence';

/**
 * Policy configuration governing `AdaptiveRetrievePolicy` and `KaiRetrieveOrAnswerGate`.
 */
export interface AdaptiveRetrievePolicyConfig {
  /** Calibrated retrieval probability threshold `theta` in `[0, 1]`. Default: `0.50`. */
  readonly theta?: number;
  /** Minimum `p(clarifyOrAbstain)` triggering immediate clarification/abstention. Default: `0.45`. */
  readonly clarifyThreshold?: number;
  /** Maximum tolerated `ambiguityRisk` before forcing `clarify_or_abstain`. Default: `0.65`. */
  readonly maxAmbiguityBound?: number;
  /** Minimum `freshnessRisk` forcing retrieval even when `p(retrieve) < theta`. Default: `0.55`. */
  readonly freshnessRiskFloor?: number;
  /** Minimum `citationRisk` forcing retrieval even when `p(retrieve) < theta`. Default: `0.60`. */
  readonly citationRiskFloor?: number;
  /** Allowed sum deviation from `1.0` when validating simplex probabilities. Default: `1e-3`. */
  readonly simplexTolerance?: number;
  /** Top-k chunks requested from `hanzoai/index` when retrieving. Default: `4`. */
  readonly topK?: number;
  /** Minimum chunk score treated as grounded evidence. Default: `0.15`. */
  readonly minEvidenceRelevance?: number;
  /** Behavior when retrieval returns zero authorized relevant chunks. Default: `'abstain'`. */
  readonly emptyEvidenceFallback?: 'abstain' | 'answer_direct';
}

/**
 * Pre-retrieval gate evaluation produced by `AdaptiveRetrievePolicy`.
 */
export interface GateEvaluation {
  readonly action: KaiAction;
  readonly reason: GateDecisionReason;
  readonly theta: number;
  readonly kai: KaiDecisionOutput;
}

/**
 * Full execution trace of a single query processed through `KaiRetrieveOrAnswerGate`.
 */
export interface SelfRagExecutionResult {
  readonly action: KaiAction;
  readonly decisionReason: GateDecisionReason;
  readonly theta: number;
  readonly kai: KaiDecisionOutput;
  readonly indexResult?: IndexSearchResult;
  readonly zenResult?: ZenSynthesisOutput;
  readonly responseText: string;
  readonly citations: readonly string[];
  readonly grounded: boolean;
  readonly abstained: boolean;
  readonly totalLatencyMs: number;
  readonly totalCostUsd: number;
}

/**
 * Raised when Kai outputs invalid probabilities, non-finite numbers, or out-of-range thresholds.
 */
export class KaiValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'KaiValidationError';
  }
}

/**
 * Raised when the Kai/Zen/Index architectural boundary is violated (for example, generative tokens on Kai).
 */
export class KaiArchitectureViolationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'KaiArchitectureViolationError';
  }
}

/**
 * Raised when an index query or stored document violates the immutable `ImmutableIndexBinding`.
 */
export class IndexContractMismatchError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'IndexContractMismatchError';
  }
}

/**
 * Raised when tenant or principal authorization context is missing or invalid.
 */
export class TenantIsolationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'TenantIsolationError';
  }
}

/**
 * Raised when threshold calibration inputs are invalid or strict constraints cannot be met.
 */
export class CalibrationConstraintError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'CalibrationConstraintError';
  }
}
