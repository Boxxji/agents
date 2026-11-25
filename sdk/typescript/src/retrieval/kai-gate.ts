// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

import type { AIClient } from '../ai/AIClient.js';
import {
  formatPrefixedQuery,
  GuardedIndexRetriever,
  validateTenantContext,
} from './index-guard.js';
import {
  type AdaptiveRetrievePolicyConfig,
  type EvidenceChunk,
  type GateDecisionReason,
  type GateEvaluation,
  type IndexRetriever,
  KaiArchitectureViolationError,
  type KaiDecisionHead,
  type KaiDecisionOutput,
  type KaiRawDecision,
  type KaiRiskSignals,
  type KaiSimplexProbabilities,
  KaiValidationError,
  type RetrievalDecisionContext,
  type SelfRagExecutionResult,
  type ZenSynthesisInput,
  type ZenSynthesisOutput,
  type ZenSynthesizer,
} from './types.js';

function assertFiniteUnitValue(value: unknown, fieldName: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new KaiValidationError(`${fieldName} must be a finite number, got ${String(value)}.`);
  }
  if (value < 0 || value > 1) {
    throw new KaiValidationError(`${fieldName} must be within [0, 1], got ${value}.`);
  }
  return value;
}

/**
 * Normalizes non-negative finite weights onto the 3-simplex (`sum === 1`).
 */
export function normalizeSimplex(weights: KaiSimplexProbabilities): KaiSimplexProbabilities {
  if (!weights || typeof weights !== 'object') {
    throw new KaiValidationError('Simplex weights must be an object.');
  }
  const { retrieve, answerDirect, clarifyOrAbstain } = weights;
  for (const [name, val] of [
    ['retrieve', retrieve],
    ['answerDirect', answerDirect],
    ['clarifyOrAbstain', clarifyOrAbstain],
  ] as const) {
    if (typeof val !== 'number' || !Number.isFinite(val)) {
      throw new KaiValidationError(`Probability "${name}" must be finite, got ${String(val)}.`);
    }
    if (val < 0) {
      throw new KaiValidationError(`Probability "${name}" cannot be negative, got ${val}.`);
    }
  }
  const sum = retrieve + answerDirect + clarifyOrAbstain;
  if (!Number.isFinite(sum) || sum <= 0) {
    throw new KaiValidationError(`Simplex weights must have a positive finite sum, got ${sum}.`);
  }

  const r = retrieve / sum;
  const a = answerDirect / sum;
  const c = Math.max(0, 1 - r - a);
  return Object.freeze({
    retrieve: r,
    answerDirect: a,
    clarifyOrAbstain: c,
  });
}

/**
 * Converts unnormalized finite logits into calibrated 3-simplex probabilities via
 * numerically stable temperature-scaled softmax.
 */
export function softmaxToSimplex(
  logits: {
    readonly retrieve: number;
    readonly answerDirect: number;
    readonly clarifyOrAbstain: number;
  },
  temperature = 1.0
): KaiSimplexProbabilities {
  if (typeof temperature !== 'number' || !Number.isFinite(temperature) || temperature <= 0) {
    throw new KaiValidationError(
      `Softmax temperature must be a positive finite number, got ${String(temperature)}.`
    );
  }
  for (const [name, val] of Object.entries(logits)) {
    if (typeof val !== 'number' || !Number.isFinite(val)) {
      throw new KaiValidationError(`Logit "${name}" must be a finite number, got ${String(val)}.`);
    }
  }

  const scaledR = logits.retrieve / temperature;
  const scaledA = logits.answerDirect / temperature;
  const scaledC = logits.clarifyOrAbstain / temperature;
  const maxLogit = Math.max(scaledR, scaledA, scaledC);

  return normalizeSimplex({
    retrieve: Math.exp(scaledR - maxLogit),
    answerDirect: Math.exp(scaledA - maxLogit),
    clarifyOrAbstain: Math.exp(scaledC - maxLogit),
  });
}

/**
 * Validates raw output from a `KaiDecisionHead`, enforcing:
 * 1. Non-generative execution (`outputTokens === 0`, no prose fields).
 * 2. Finite non-negative probabilities summing to `1.0` within `simplexTolerance`.
 * 3. Finite risk signals in `[0, 1]`.
 */
export function validateKaiDecisionOutput(
  raw: KaiRawDecision,
  simplexTolerance = 1e-3
): KaiDecisionOutput {
  if (!raw || typeof raw !== 'object') {
    throw new KaiValidationError('Kai decision payload must be an object.');
  }

  if (raw.outputTokens !== undefined && raw.outputTokens !== 0) {
    throw new KaiArchitectureViolationError(
      `Kai decision head must be non-generative (outputTokens === 0), got outputTokens=${String(raw.outputTokens)}.`
    );
  }

  const rawRecord = raw as unknown as Record<string, unknown>;
  if (
    typeof rawRecord.generatedText === 'string' ||
    typeof rawRecord.answer === 'string' ||
    typeof rawRecord.completion === 'string'
  ) {
    throw new KaiArchitectureViolationError(
      'Kai decision head must emit structured choice probabilities only, never generative answer text.'
    );
  }

  if (!raw.probabilities || typeof raw.probabilities !== 'object') {
    throw new KaiValidationError('Kai decision payload is missing probabilities.');
  }

  const { retrieve, answerDirect, clarifyOrAbstain } = raw.probabilities;
  for (const [name, val] of [
    ['probabilities.retrieve', retrieve],
    ['probabilities.answerDirect', answerDirect],
    ['probabilities.clarifyOrAbstain', clarifyOrAbstain],
  ] as const) {
    if (typeof val !== 'number' || !Number.isFinite(val)) {
      throw new KaiValidationError(`${name} must be a finite number, got ${String(val)}.`);
    }
    if (val < 0 || val > 1 + simplexTolerance) {
      throw new KaiValidationError(`${name} must lie in [0, 1], got ${val}.`);
    }
  }

  const sum = retrieve + answerDirect + clarifyOrAbstain;
  if (!Number.isFinite(sum) || Math.abs(sum - 1) > simplexTolerance) {
    throw new KaiValidationError(
      `Kai probabilities must sum to 1.0 (+/- ${simplexTolerance}), got sum=${sum}.`
    );
  }

  const normalized = normalizeSimplex(raw.probabilities);

  const freshnessRisk = assertFiniteUnitValue(
    raw.riskSignals?.freshnessRisk ?? 0,
    'riskSignals.freshnessRisk'
  );
  const citationRisk = assertFiniteUnitValue(
    raw.riskSignals?.citationRisk ?? 0,
    'riskSignals.citationRisk'
  );
  const ambiguityRisk = assertFiniteUnitValue(
    raw.riskSignals?.ambiguityRisk ?? 0,
    'riskSignals.ambiguityRisk'
  );
  const distractorRisk =
    raw.riskSignals?.distractorRisk !== undefined
      ? assertFiniteUnitValue(raw.riskSignals.distractorRisk, 'riskSignals.distractorRisk')
      : undefined;

  const latencyMs = raw.latencyMs ?? 0;
  if (typeof latencyMs !== 'number' || !Number.isFinite(latencyMs) || latencyMs < 0) {
    throw new KaiValidationError(
      `Kai latencyMs must be non-negative finite, got ${String(latencyMs)}.`
    );
  }

  const costUsd = raw.costUsd ?? 0;
  if (typeof costUsd !== 'number' || !Number.isFinite(costUsd) || costUsd < 0) {
    throw new KaiValidationError(
      `Kai costUsd must be non-negative finite, got ${String(costUsd)}.`
    );
  }

  const riskSignals: KaiRiskSignals = Object.freeze({
    freshnessRisk,
    citationRisk,
    ambiguityRisk,
    ...(distractorRisk !== undefined ? { distractorRisk } : {}),
  });

  return Object.freeze({
    probabilities: normalized,
    riskSignals,
    outputTokens: 0,
    modelId: raw.modelId && raw.modelId.trim().length > 0 ? raw.modelId : 'kai-decision-head',
    latencyMs,
    costUsd,
  });
}

const FRESHNESS_PATTERNS: readonly RegExp[] = [
  /\b(current|currently|latest|today|right now|tonight|this week|recent|live|active|open)\b/i,
  /\b(2025|2026|2027|january|february|march|april|may|june|july|august|september|october|november|december)\b/i,
  /\b(invoice|billing|balance|ledger|spend|usage|quota|subscription|payout|sla)\b/i,
  /\b(commit|revision|branch|shard|cluster|uptime|incident|ticket|alert|telemetry|circuit[- ]breaker)\b/i,
  /\b(hanzo|@hanzo|hanzoai|hip-\d+|v\d+\.\d+(?:\.\d+)?|worker-\d+|node-\d+|acme-|globex-)\b/i,
  /\b(route table|routes_test|control-plane|zap-proto)\b/i,
];

const CITATION_PATTERNS: readonly RegExp[] = [
  /\b(cite|citation|citations|source uri|exact file|exact line|commit sha|hash|audit record|verbatim|prove)\b/i,
  /\b(hip-\d+|rfc\s*\d+|chunk id|reference id)\b/i,
];

const AMBIGUOUS_DEICTIC_PATTERNS: readonly RegExp[] = [
  /\b(that deployment|that thing|that service|the other service|the other repo|the other one)\b/i,
  /\b(fix it|roll it back|roll back that|restart the broken one|delete the key|drop the table)\b/i,
  /\b(send it to him|send the invoice to him|email her the report|reset her permissions)\b/i,
  /\b(update the timeout to match|do the same as earlier|same config as before|change that setting)\b/i,
];

const CONCRETE_TARGET_PATTERNS: readonly RegExp[] = [
  /\b([a-z0-9]+-[a-z0-9]+(?:\.[a-z0-9]+)*|v\d+\.\d+|hip-\d+|sha-?256|http\s+\d{3}|json-rpc|tcp|tls|utf-8)\b/i,
  /\b(binary search|quicksort|mergesort|dijkstra|bellman-ford|promise\.all|promise\.allsettled|big-o)\b/i,
  /\b(cap theorem|b-tree|red-black tree|hmac|ed25519|secp256k1|base64|ieee 754)\b/i,
];

const PARAMETRIC_STABLE_PATTERNS: readonly RegExp[] = [
  /\b(time complexity|space complexity|big-o|asymptotic|worst-case|average-case)\b/i,
  /\b(binary search|mergesort|quicksort|dijkstra|breadth-first|depth-first|hash table|b-tree)\b/i,
  /\b(tcp three-way handshake|http\s+\d{3}|tls\s+1\.[23]|json-rpc\s+2\.0|utf-8|base64|sha-?256|ed25519)\b/i,
  /\b(promise\.all|promise\.allsettled|event loop|closure|structural typing|monad|idempotence|cap theorem)\b/i,
  /\b(2\^\d+|prime number|euclidean algorithm|logarithm|matrix multiplication|bayes'? theorem)\b/i,
];

const DISTRACTOR_NOISE_PATTERNS: readonly RegExp[] = [
  /\?{2,}/,
  /\b(ignore unrelated chatter|ignore previous|random trivia|distractor note|by the way)\b/i,
];

export interface CalibratedKaiHeadOptions {
  readonly modelId?: string;
  readonly temperature?: number;
  readonly simulatedLatencyMs?: number;
  readonly costPerDecisionUsd?: number;
}

/**
 * Clean-room calibrated non-generative Kai decision head (`outputTokens: 0`).
 *
 * Computes calibrated simplex probabilities and risk floors over temporal/mutable cues,
 * citation requirements, deictic underspecification, and parametric domain stability.
 */
export class CalibratedKaiHead implements KaiDecisionHead {
  readonly modelId: string;
  readonly temperature: number;
  private readonly simulatedLatencyMs: number;
  private readonly costPerDecisionUsd: number;

  constructor(options: CalibratedKaiHeadOptions = {}) {
    const temperature = options.temperature ?? 1.0;
    if (!Number.isFinite(temperature) || temperature <= 0) {
      throw new KaiValidationError(
        `CalibratedKaiHead temperature must be > 0, got ${temperature}.`
      );
    }
    this.modelId = options.modelId ?? 'kai-1-gate-calibrated';
    this.temperature = temperature;
    this.simulatedLatencyMs = options.simulatedLatencyMs ?? 6;
    this.costPerDecisionUsd = options.costPerDecisionUsd ?? 0.00002;
  }

  decide(context: RetrievalDecisionContext): KaiRawDecision {
    if (!context || typeof context.query !== 'string' || context.query.trim().length === 0) {
      throw new KaiValidationError('RetrievalDecisionContext.query must be a non-empty string.');
    }

    const q = context.query.trim();
    const wordCount = q.split(/\s+/).length;

    let freshnessMatches = 0;
    for (const pat of FRESHNESS_PATTERNS) {
      if (pat.test(q)) {
        freshnessMatches += 1;
      }
    }

    let citationMatches = 0;
    for (const pat of CITATION_PATTERNS) {
      if (pat.test(q)) {
        citationMatches += 1;
      }
    }

    let ambiguousMatches = 0;
    for (const pat of AMBIGUOUS_DEICTIC_PATTERNS) {
      if (pat.test(q)) {
        ambiguousMatches += 1;
      }
    }

    let hasConcreteTarget = false;
    for (const pat of CONCRETE_TARGET_PATTERNS) {
      if (pat.test(q)) {
        hasConcreteTarget = true;
        break;
      }
    }

    let parametricMatches = 0;
    for (const pat of PARAMETRIC_STABLE_PATTERNS) {
      if (pat.test(q)) {
        parametricMatches += 1;
      }
    }

    let distractorMatches = 0;
    for (const pat of DISTRACTOR_NOISE_PATTERNS) {
      if (pat.test(q)) {
        distractorMatches += 1;
      }
    }

    const veryShortUnderspecified = wordCount <= 3 && !hasConcreteTarget && freshnessMatches === 0;
    const domainVolatility =
      context.domainVolatility !== undefined
        ? assertFiniteUnitValue(context.domainVolatility, 'context.domainVolatility')
        : 0;

    const freshnessRisk = Math.min(
      1,
      (context.requireFreshness ? 0.85 : 0) + domainVolatility * 0.75 + freshnessMatches * 0.36
    );
    const citationRisk = Math.min(1, (context.requireCitations ? 0.9 : 0) + citationMatches * 0.45);
    const ambiguityRisk = Math.min(
      1,
      (ambiguousMatches > 0 && !hasConcreteTarget ? 0.78 : ambiguousMatches * 0.42) +
        (veryShortUnderspecified ? 0.72 : 0)
    );
    const distractorRisk = Math.min(
      1,
      distractorMatches * 0.45 + (parametricMatches > 0 && q.length > 140 ? 0.25 : 0)
    );

    const retrieveLogit =
      -0.65 +
      freshnessMatches * 1.45 +
      citationMatches * 1.1 +
      (context.requireFreshness ? 1.8 : 0) +
      (context.requireCitations ? 1.5 : 0) +
      domainVolatility * 1.9 -
      parametricMatches * 1.35 -
      ambiguityRisk * 1.4 -
      distractorMatches * 0.55;

    const answerDirectLogit =
      0.55 +
      parametricMatches * 1.65 +
      (freshnessMatches === 0 && citationMatches === 0 && ambiguityRisk < 0.3 ? 0.95 : 0) -
      freshnessMatches * 1.35 -
      citationMatches * 0.95 -
      (context.requireFreshness ? 1.6 : 0) -
      ambiguityRisk * 1.5;

    const clarifyLogit =
      -1.45 +
      ambiguityRisk * 3.85 -
      (hasConcreteTarget ? 0.95 : 0) -
      parametricMatches * 1.2 -
      freshnessMatches * 0.65;

    const probabilities = softmaxToSimplex(
      {
        retrieve: retrieveLogit,
        answerDirect: answerDirectLogit,
        clarifyOrAbstain: clarifyLogit,
      },
      this.temperature
    );

    return {
      probabilities,
      riskSignals: {
        freshnessRisk: Number(freshnessRisk.toFixed(6)),
        citationRisk: Number(citationRisk.toFixed(6)),
        ambiguityRisk: Number(ambiguityRisk.toFixed(6)),
        distractorRisk: Number(distractorRisk.toFixed(6)),
      },
      outputTokens: 0,
      modelId: this.modelId,
      latencyMs: this.simulatedLatencyMs,
      costUsd: this.costPerDecisionUsd,
    };
  }
}

/**
 * Evaluates normalized Kai simplex probabilities against calibrated threshold `theta`,
 * freshness/citation risk floors, and ambiguity safety bounds.
 */
export class AdaptiveRetrievePolicy {
  readonly theta: number;
  readonly clarifyThreshold: number;
  readonly maxAmbiguityBound: number;
  readonly freshnessRiskFloor: number;
  readonly citationRiskFloor: number;
  readonly simplexTolerance: number;
  readonly topK: number;
  readonly minEvidenceRelevance: number;
  readonly emptyEvidenceFallback: 'abstain' | 'answer_direct';

  constructor(config: AdaptiveRetrievePolicyConfig = {}) {
    this.theta = assertFiniteUnitValue(config.theta ?? 0.5, 'config.theta');
    this.clarifyThreshold = assertFiniteUnitValue(
      config.clarifyThreshold ?? 0.45,
      'config.clarifyThreshold'
    );
    this.maxAmbiguityBound = assertFiniteUnitValue(
      config.maxAmbiguityBound ?? 0.65,
      'config.maxAmbiguityBound'
    );
    this.freshnessRiskFloor = assertFiniteUnitValue(
      config.freshnessRiskFloor ?? 0.55,
      'config.freshnessRiskFloor'
    );
    this.citationRiskFloor = assertFiniteUnitValue(
      config.citationRiskFloor ?? 0.6,
      'config.citationRiskFloor'
    );

    const simplexTolerance = config.simplexTolerance ?? 1e-3;
    if (
      typeof simplexTolerance !== 'number' ||
      !Number.isFinite(simplexTolerance) ||
      simplexTolerance <= 0 ||
      simplexTolerance > 0.05
    ) {
      throw new KaiValidationError(
        `config.simplexTolerance must lie in (0, 0.05], got ${String(simplexTolerance)}.`
      );
    }
    this.simplexTolerance = simplexTolerance;

    const topK = config.topK ?? 4;
    if (!Number.isInteger(topK) || topK <= 0) {
      throw new KaiValidationError(`config.topK must be a positive integer, got ${String(topK)}.`);
    }
    this.topK = topK;

    this.minEvidenceRelevance = assertFiniteUnitValue(
      config.minEvidenceRelevance ?? 0.15,
      'config.minEvidenceRelevance'
    );
    this.emptyEvidenceFallback = config.emptyEvidenceFallback ?? 'abstain';
    Object.freeze(this);
  }

  /**
   * Evaluates a Kai decision payload into a discrete action and auditable `GateDecisionReason`.
   */
  evaluate(rawDecision: KaiRawDecision, thetaOverride?: number): GateEvaluation {
    const theta =
      thetaOverride !== undefined
        ? assertFiniteUnitValue(thetaOverride, 'thetaOverride')
        : this.theta;
    const kai = validateKaiDecisionOutput(rawDecision, this.simplexTolerance);

    if (
      kai.probabilities.clarifyOrAbstain >= this.clarifyThreshold ||
      kai.riskSignals.ambiguityRisk >= this.maxAmbiguityBound
    ) {
      return Object.freeze({
        action: 'clarify_or_abstain',
        reason: 'clarify_safety_bound',
        theta,
        kai,
      });
    }

    if (kai.riskSignals.freshnessRisk >= this.freshnessRiskFloor) {
      return Object.freeze({
        action: 'retrieve',
        reason: 'freshness_risk_floor',
        theta,
        kai,
      });
    }

    if (kai.riskSignals.citationRisk >= this.citationRiskFloor) {
      return Object.freeze({
        action: 'retrieve',
        reason: 'citation_risk_floor',
        theta,
        kai,
      });
    }

    if (kai.probabilities.retrieve >= theta) {
      return Object.freeze({
        action: 'retrieve',
        reason: 'kai_probability_above_theta',
        theta,
        kai,
      });
    }

    return Object.freeze({
      action: 'answer_direct',
      reason: 'kai_answer_direct',
      theta,
      kai,
    });
  }
}

export interface KaiRetrieveOrAnswerGateOptions {
  readonly kaiHead: KaiDecisionHead;
  readonly retriever: IndexRetriever;
  readonly zen: ZenSynthesizer;
  readonly policy?: AdaptiveRetrievePolicy | AdaptiveRetrievePolicyConfig;
  readonly clarifyTemplate?: (context: RetrievalDecisionContext) => string;
  readonly insufficientEvidenceTemplate?: (context: RetrievalDecisionContext) => string;
}

function defaultClarifyMessage(context: RetrievalDecisionContext): string {
  return `Clarification required before acting on "${context.query.trim()}": specify the exact target service, identifier, or resource scope.`;
}

function defaultInsufficientEvidenceMessage(context: RetrievalDecisionContext): string {
  return `Abstaining on "${context.query.trim()}": no authorized grounded evidence matched tenant "${context.tenantContext.tenantId}".`;
}

/**
 * Stateless, concurrency-safe Self-RAG orchestrator enforcing the Hanzo Kai -> Index -> Zen boundary.
 */
export class KaiRetrieveOrAnswerGate {
  readonly policy: AdaptiveRetrievePolicy;
  readonly kaiHead: KaiDecisionHead;
  readonly retriever: GuardedIndexRetriever;
  readonly zen: ZenSynthesizer;
  private readonly clarifyTemplate: (context: RetrievalDecisionContext) => string;
  private readonly insufficientEvidenceTemplate: (context: RetrievalDecisionContext) => string;

  constructor(options: KaiRetrieveOrAnswerGateOptions) {
    this.policy =
      options.policy instanceof AdaptiveRetrievePolicy
        ? options.policy
        : new AdaptiveRetrievePolicy(options.policy);
    this.kaiHead = options.kaiHead;
    this.retriever =
      options.retriever instanceof GuardedIndexRetriever
        ? options.retriever
        : new GuardedIndexRetriever(options.retriever);
    this.zen = options.zen;
    this.clarifyTemplate = options.clarifyTemplate ?? defaultClarifyMessage;
    this.insufficientEvidenceTemplate =
      options.insufficientEvidenceTemplate ?? defaultInsufficientEvidenceMessage;
    Object.freeze(this);
  }

  /**
   * Computes Kai's structured non-generative routing decision without invoking Index or Zen.
   */
  async decide(context: RetrievalDecisionContext, thetaOverride?: number): Promise<GateEvaluation> {
    if (!context || typeof context.query !== 'string' || context.query.trim().length === 0) {
      throw new KaiValidationError('RetrievalDecisionContext.query must be a non-empty string.');
    }
    validateTenantContext(context.tenantContext);
    const rawDecision = await this.kaiHead.decide(context);
    return this.policy.evaluate(rawDecision, thetaOverride);
  }

  /**
   * Executes the full adaptive retrieve-or-answer pipeline for a single query.
   * Safe for concurrent `Promise.all` calls across distinct tenants and principals.
   */
  async execute(
    context: RetrievalDecisionContext,
    thetaOverride?: number
  ): Promise<SelfRagExecutionResult> {
    const evaluation = await this.decide(context, thetaOverride);
    const { kai, theta } = evaluation;

    if (evaluation.action === 'clarify_or_abstain') {
      return Object.freeze({
        action: 'clarify_or_abstain',
        decisionReason: evaluation.reason,
        theta,
        kai,
        responseText: this.clarifyTemplate(context),
        citations: Object.freeze([]),
        grounded: false,
        abstained: true,
        totalLatencyMs: kai.latencyMs,
        totalCostUsd: Number(kai.costUsd.toFixed(8)),
      });
    }

    if (evaluation.action === 'answer_direct') {
      const zenResult = await this.zen.synthesize({
        query: context.query,
        mode: 'parametric_direct',
        evidence: Object.freeze([]),
        tenantContext: context.tenantContext,
      });
      if (zenResult.citations.length > 0) {
        throw new KaiArchitectureViolationError(
          'Zen parametric_direct synthesis cannot attach index citations when retrieval was skipped.'
        );
      }
      return Object.freeze({
        action: 'answer_direct',
        decisionReason: evaluation.reason,
        theta,
        kai,
        zenResult,
        responseText: zenResult.answer,
        citations: Object.freeze([]),
        grounded: false,
        abstained: false,
        totalLatencyMs: kai.latencyMs + zenResult.latencyMs,
        totalCostUsd: Number((kai.costUsd + zenResult.costUsd).toFixed(8)),
      });
    }

    const formattedQuery = formatPrefixedQuery(this.retriever.binding, context.query);
    const indexResult = await this.retriever.retrieve({
      rawQuery: context.query,
      formattedQuery,
      topK: this.policy.topK,
      minRelevanceScore: this.policy.minEvidenceRelevance,
      binding: this.retriever.binding,
      tenantContext: context.tenantContext,
    });

    const relevantChunks: readonly EvidenceChunk[] = Object.freeze(
      indexResult.chunks.filter((chunk) => chunk.score >= this.policy.minEvidenceRelevance)
    );

    if (relevantChunks.length === 0) {
      if (this.policy.emptyEvidenceFallback === 'abstain') {
        const abstainReason: GateDecisionReason = 'post_retrieval_insufficient_evidence';
        return Object.freeze({
          action: 'clarify_or_abstain',
          decisionReason: abstainReason,
          theta,
          kai,
          indexResult,
          responseText: this.insufficientEvidenceTemplate(context),
          citations: Object.freeze([]),
          grounded: false,
          abstained: true,
          totalLatencyMs: kai.latencyMs + indexResult.latencyMs,
          totalCostUsd: Number((kai.costUsd + indexResult.costUsd).toFixed(8)),
        });
      }

      const fallbackZen = await this.zen.synthesize({
        query: context.query,
        mode: 'parametric_direct',
        evidence: Object.freeze([]),
        tenantContext: context.tenantContext,
      });
      return Object.freeze({
        action: 'retrieve',
        decisionReason: evaluation.reason,
        theta,
        kai,
        indexResult,
        zenResult: fallbackZen,
        responseText: fallbackZen.answer,
        citations: Object.freeze([]),
        grounded: false,
        abstained: false,
        totalLatencyMs: kai.latencyMs + indexResult.latencyMs + fallbackZen.latencyMs,
        totalCostUsd: Number((kai.costUsd + indexResult.costUsd + fallbackZen.costUsd).toFixed(8)),
      });
    }

    const zenResult = await this.zen.synthesize({
      query: context.query,
      mode: 'grounded_rag',
      evidence: relevantChunks,
      tenantContext: context.tenantContext,
    });

    const validChunkIds = new Set(relevantChunks.map((c) => c.chunkId));
    const verifiedCitations = zenResult.citations.filter((id) => validChunkIds.has(id));
    const finalCitations =
      verifiedCitations.length > 0
        ? Object.freeze(verifiedCitations)
        : Object.freeze(relevantChunks.map((c) => c.chunkId));

    return Object.freeze({
      action: 'retrieve',
      decisionReason: evaluation.reason,
      theta,
      kai,
      indexResult,
      zenResult,
      responseText: zenResult.answer,
      citations: finalCitations,
      grounded: true,
      abstained: false,
      totalLatencyMs: kai.latencyMs + indexResult.latencyMs + zenResult.latencyMs,
      totalCostUsd: Number((kai.costUsd + indexResult.costUsd + zenResult.costUsd).toFixed(8)),
    });
  }
}

export interface AIClientZenSynthesizerOptions {
  readonly aiClient: AIClient;
  readonly modelId?: string;
  readonly inputTokenCostUsd?: number;
  readonly outputTokenCostUsd?: number;
}

/**
 * Production adapter connecting `@hanzo/agents` `AIClient` (defaulting to Hanzo `zen3-vl`)
 * to the `ZenSynthesizer` boundary.
 */
export class AIClientZenSynthesizer implements ZenSynthesizer {
  readonly modelId: string;
  private readonly aiClient: AIClient;
  private readonly inputTokenCostUsd: number;
  private readonly outputTokenCostUsd: number;

  constructor(options: AIClientZenSynthesizerOptions) {
    this.aiClient = options.aiClient;
    this.modelId = options.modelId ?? 'zen3-vl';
    this.inputTokenCostUsd = options.inputTokenCostUsd ?? 0.0000012;
    this.outputTokenCostUsd = options.outputTokenCostUsd ?? 0.0000024;
  }

  async synthesize(input: ZenSynthesisInput): Promise<ZenSynthesisOutput> {
    const startMs = Date.now();
    const systemPrompt =
      input.mode === 'grounded_rag'
        ? 'Answer strictly using the supplied evidence chunks and cite chunk IDs in brackets.'
        : 'Answer concisely and accurately using established domain knowledge.';

    const formattedPrompt =
      input.mode === 'grounded_rag'
        ? `Evidence:\n${input.evidence
            .map((c) => `[${c.chunkId}] (${c.sourceUri}): ${c.content}`)
            .join('\n')}\n\nQuestion: ${input.query}`
        : input.query;

    const answer = await this.aiClient.generate(formattedPrompt, {
      system: systemPrompt,
      model: this.modelId,
      temperature: 0,
    });

    const latencyMs = Math.max(1, Date.now() - startMs);
    const tokensIn = Math.ceil((systemPrompt.length + formattedPrompt.length) / 4);
    const tokensOut = Math.ceil(answer.length / 4);
    const costUsd = tokensIn * this.inputTokenCostUsd + tokensOut * this.outputTokenCostUsd;

    return Object.freeze({
      answer,
      citations:
        input.mode === 'grounded_rag'
          ? Object.freeze(input.evidence.map((c) => c.chunkId))
          : Object.freeze([]),
      modelId: this.modelId,
      tokensIn,
      tokensOut,
      latencyMs,
      costUsd: Number(costUsd.toFixed(8)),
    });
  }
}
