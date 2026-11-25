import crypto from 'crypto';
import type { MemoryScope } from '../types/agent.js';
import type { MemoryBackend } from './MemoryBackend.js';
import type { MemoryRequestMetadata, VectorSearchResult } from './MemoryClient.js';
import type { AIClient, AIEmbeddingOptions } from '../ai/AIClient.js';

export type PolicyAction = 'ADD' | 'UPDATE' | 'DELETE' | 'FORGET' | 'NOOP';
export type QueryAbstentionStatus = 'ANSWER' | 'INSUFFICIENT_EVIDENCE';

export interface PolicyMemoryCandidate {
  keySlot: string;
  content: string;
  sourceIdentity: string;
  sourceVersion?: number;
  confidence?: number;
  utilityHint?: number;
  expiresAt?: string;
  revokeKey?: boolean;
  containsRestrictedSensitiveData?: boolean;
  aclRoles?: string[];
  metadata?: Record<string, any>;
}

export interface PolicyManagedRecord {
  id: string;
  keySlot: string;
  content: string;
  sourceIdentity: string;
  sourceVersion: number;
  version: number;
  scope: MemoryScope;
  scopeId: string;
  confidence: number;
  utilityScore: number;
  expiresAt?: string;
  aclRoles: string[];
  metadata?: Record<string, any>;
  createdAt: string;
  updatedAt: string;
}

export interface PolicyDecisionTrace {
  decisionId: string;
  action: PolicyAction;
  overriddenByHardPolicy: boolean;
  reason: string;
  keySlot: string;
  record?: PolicyManagedRecord;
  features: {
    novelty: number;
    contradiction: number;
    confidence: number;
    utilityHint: number;
  };
}

export interface AbstentionQueryResult {
  status: QueryAbstentionStatus;
  reason: string;
  records: PolicyManagedRecord[];
}

export interface PolicyMemoryOptions {
  client: MemoryBackend;
  aiClient?: AIClient;
  defaultScope?: MemoryScope;
  defaultScopeId?: string;
  metadata?: MemoryRequestMetadata;
  learningRate?: number;
  minConfidenceForAdd?: number;
  minEvidenceCount?: number;
  minAnswerConfidence?: number;
  embeddingOptions?: AIEmbeddingOptions;
}

const POLICY_RECORD_PREFIX = 'policy_rec:';
const POLICY_SLOT_MAP_KEY = 'policy_slot_map';
const REVOKED_SOURCES_KEY = 'policy_revoked_sources';

function norm(val: string): string {
  return val.trim().toLowerCase().replace(/\s+/g, ' ');
}

export function computePolicyRecordId(
  scope: MemoryScope,
  scopeId: string,
  keySlot: string,
  content: string,
  version: number
): string {
  const canonical = `${scope}:${scopeId}:${norm(keySlot)}:${norm(content)}:v${version}`;
  return crypto.createHash('sha256').update(canonical, 'utf8').digest('hex').slice(0, 32);
}

/**
 * Memory-R1 style Outcome-Calibrated Memory Policy Controller & Store.
 *
 * Guarantees:
 * - Hard authorization, revocation, sensitive-data isolation, and retention/TTL
 *   policies strictly override learned controller preferences.
 * - Outcome calibration (`recordTaskOutcome`) updates action weights from real
 *   downstream task rewards rather than raw retrieval similarity alone.
 * - `queryWithAbstention` answers `'INSUFFICIENT_EVIDENCE'` whenever authorized,
 *   non-expired support falls below configured confidence or count thresholds.
 */
export class PolicyControlledMemory {
  private readonly client: MemoryBackend;
  private readonly aiClient?: AIClient;
  private readonly defaultScope: MemoryScope;
  private readonly defaultScopeId: string;
  private readonly metadata?: MemoryRequestMetadata;
  private readonly learningRate: number;
  private readonly minConfidenceForAdd: number;
  private readonly minEvidenceCount: number;
  private readonly minAnswerConfidence: number;
  private readonly embeddingOptions?: AIEmbeddingOptions;

  // Lightweight linear policy weights calibrated on downstream task outcomes
  private actionWeights: Record<PolicyAction, [number, number, number, number]> = {
    ADD: [0.9, 0.1, 0.6, 0.8],
    UPDATE: [0.2, 0.9, 0.7, 0.8],
    DELETE: [-0.5, 0.4, -0.6, -0.5],
    FORGET: [-0.4, 0.2, -0.8, -0.6],
    NOOP: [-0.6, -0.6, 0.2, -0.2]
  };

  private readonly decisionTraces = new Map<string, PolicyDecisionTrace>();

  constructor(options: PolicyMemoryOptions) {
    this.client = options.client;
    this.aiClient = options.aiClient;
    this.defaultScope = options.defaultScope ?? 'workflow';
    this.defaultScopeId = options.defaultScopeId ?? 'default';
    this.metadata = options.metadata;
    this.learningRate = options.learningRate ?? 0.15;
    this.minConfidenceForAdd = options.minConfidenceForAdd ?? 0.35;
    this.minEvidenceCount = options.minEvidenceCount ?? 1;
    this.minAnswerConfidence = options.minAnswerConfidence ?? 0.6;
    this.embeddingOptions = options.embeddingOptions;
  }

  /**
   * Evaluates a memory candidate under hard policy guardrails first, then applies
   * the outcome-calibrated controller to select ADD / UPDATE / DELETE / FORGET / NOOP.
   */
  async applyPolicy(
    candidate: PolicyMemoryCandidate,
    options?: {
      scope?: MemoryScope;
      scopeId?: string;
      nowIso?: string;
      embedding?: number[];
    }
  ): Promise<PolicyDecisionTrace> {
    const scope = options?.scope ?? this.defaultScope;
    const scopeId = options?.scopeId ?? this.defaultScopeId;
    const nowIso = options?.nowIso ?? new Date().toISOString();
    const slotKey = norm(candidate.keySlot);
    const decisionId = crypto.randomBytes(12).toString('hex');

    const slotMap =
      (await this.client.get<Record<string, string>>(POLICY_SLOT_MAP_KEY, {
        scope,
        scopeId,
        metadata: this.metadata
      })) ?? {};

    const existingId = slotMap[slotKey];
    const existing = existingId
      ? await this.getRecordById(existingId, scope, scopeId)
      : undefined;

    const revokedSources = new Set<string>(
      (await this.client.get<string[]>(REVOKED_SOURCES_KEY, {
        scope,
        scopeId,
        metadata: this.metadata
      })) ?? []
    );

    const confidence = candidate.confidence ?? 0.85;
    const utilityHint = candidate.utilityHint ?? 0.75;
    const novelty = existing ? 0.0 : 1.0;
    const contradiction =
      existing && norm(existing.content) !== norm(candidate.content) ? 1.0 : 0.0;
    const features = { novelty, contradiction, confidence, utilityHint };

    // 1. HARD POLICY OVERRIDE: Sensitive / restricted PII isolation guard
    if (candidate.containsRestrictedSensitiveData) {
      const trace: PolicyDecisionTrace = {
        decisionId,
        action: 'NOOP',
        overriddenByHardPolicy: true,
        reason: 'Hard policy override: candidate contains restricted sensitive data',
        keySlot: slotKey,
        features
      };
      this.decisionTraces.set(decisionId, trace);
      return trace;
    }

    // 2. HARD POLICY OVERRIDE: Explicit revocation or revoked source identity
    if (candidate.revokeKey || revokedSources.has(candidate.sourceIdentity)) {
      if (existing) {
        await this.removeRecord(existing.id, slotKey, slotMap, scope, scopeId);
      }
      const trace: PolicyDecisionTrace = {
        decisionId,
        action: 'DELETE',
        overriddenByHardPolicy: true,
        reason: 'Hard policy override: explicit revocation or revoked source identity',
        keySlot: slotKey,
        features
      };
      this.decisionTraces.set(decisionId, trace);
      return trace;
    }

    // 3. HARD POLICY OVERRIDE: Expired retention / TTL window
    if (candidate.expiresAt && candidate.expiresAt <= nowIso) {
      if (existing) {
        await this.removeRecord(existing.id, slotKey, slotMap, scope, scopeId);
      }
      const trace: PolicyDecisionTrace = {
        decisionId,
        action: 'FORGET',
        overriddenByHardPolicy: true,
        reason: `Hard retention override: candidate expired at ${candidate.expiresAt}`,
        keySlot: slotKey,
        features
      };
      this.decisionTraces.set(decisionId, trace);
      return trace;
    }

    // 4. Exact duplicate check -> NOOP
    if (existing && norm(existing.content) === norm(candidate.content)) {
      const trace: PolicyDecisionTrace = {
        decisionId,
        action: 'NOOP',
        overriddenByHardPolicy: false,
        reason: 'Identical content already active in slot',
        keySlot: slotKey,
        record: existing,
        features
      };
      this.decisionTraces.set(decisionId, trace);
      return trace;
    }

    // 5. Learned controller decision (ADD / UPDATE / FORGET / NOOP)
    const chosenAction = this.selectLearnedAction(features, Boolean(existing));

    if (chosenAction === 'NOOP' || chosenAction === 'FORGET' || chosenAction === 'DELETE') {
      if ((chosenAction === 'FORGET' || chosenAction === 'DELETE') && existing) {
        await this.removeRecord(existing.id, slotKey, slotMap, scope, scopeId);
      }
      const trace: PolicyDecisionTrace = {
        decisionId,
        action: chosenAction,
        overriddenByHardPolicy: false,
        reason: `Learned policy controller selected ${chosenAction}`,
        keySlot: slotKey,
        features
      };
      this.decisionTraces.set(decisionId, trace);
      return trace;
    }

    const nextVersion = existing ? existing.version + 1 : 1;
    const newId = computePolicyRecordId(
      scope,
      scopeId,
      slotKey,
      candidate.content,
      nextVersion
    );

    if (existing) {
      await this.client.delete(`${POLICY_RECORD_PREFIX}${existing.id}`, {
        scope,
        scopeId,
        metadata: this.metadata
      });
      await this.client.deleteVector(`${POLICY_RECORD_PREFIX}${existing.id}`, {
        scope,
        scopeId,
        metadata: this.metadata
      });
    }

    const record: PolicyManagedRecord = {
      id: newId,
      keySlot: slotKey,
      content: candidate.content.trim(),
      sourceIdentity: candidate.sourceIdentity,
      sourceVersion: candidate.sourceVersion ?? 1,
      version: nextVersion,
      scope,
      scopeId,
      confidence,
      utilityScore: utilityHint,
      expiresAt: candidate.expiresAt,
      aclRoles: candidate.aclRoles ? [...candidate.aclRoles] : [],
      metadata: candidate.metadata,
      createdAt: existing ? existing.createdAt : nowIso,
      updatedAt: nowIso
    };

    const recKey = `${POLICY_RECORD_PREFIX}${record.id}`;
    await this.client.set(recKey, record, { scope, scopeId, metadata: this.metadata });
    slotMap[slotKey] = record.id;
    await this.client.set(POLICY_SLOT_MAP_KEY, slotMap, {
      scope,
      scopeId,
      metadata: this.metadata
    });

    let embedding = options?.embedding;
    if (!embedding && this.aiClient) {
      embedding = await this.aiClient.embed(record.content, this.embeddingOptions);
    }
    if (embedding && embedding.length > 0) {
      await this.client.setVector(
        recKey,
        embedding,
        {
          recordId: record.id,
          keySlot: record.keySlot,
          scope,
          scopeId,
          version: record.version
        },
        { scope, scopeId, metadata: this.metadata }
      );
    }

    const finalAction: PolicyAction = existing ? 'UPDATE' : 'ADD';
    const trace: PolicyDecisionTrace = {
      decisionId,
      action: finalAction,
      overriddenByHardPolicy: false,
      reason: `Learned policy controller selected ${finalAction}`,
      keySlot: slotKey,
      record,
      features
    };
    this.decisionTraces.set(decisionId, trace);
    return trace;
  }

  /**
   * Calibrates the structured controller's action weights from downstream task
   * outcome rewards (-1.0 to +1.0).
   */
  recordTaskOutcome(decisionId: string, outcomeReward: number): boolean {
    const trace = this.decisionTraces.get(decisionId);
    if (!trace || trace.overriddenByHardPolicy) {
      return false;
    }

    const fVec: [number, number, number, number] = [
      trace.features.novelty,
      trace.features.contradiction,
      trace.features.confidence,
      trace.features.utilityHint
    ];
    const currentWeights = this.actionWeights[trace.action];
    this.actionWeights[trace.action] = currentWeights.map((w, i) =>
      Number((w + this.learningRate * outcomeReward * fVec[i]).toFixed(4))
    ) as [number, number, number, number];
    return true;
  }

  /**
   * Revokes a source identity and purges all records and derived vectors originating
   * from that source within the scope.
   */
  async revokeSource(
    sourceIdentity: string,
    options?: { scope?: MemoryScope; scopeId?: string }
  ): Promise<number> {
    const scope = options?.scope ?? this.defaultScope;
    const scopeId = options?.scopeId ?? this.defaultScopeId;

    const revoked =
      (await this.client.get<string[]>(REVOKED_SOURCES_KEY, {
        scope,
        scopeId,
        metadata: this.metadata
      })) ?? [];
    if (!revoked.includes(sourceIdentity)) {
      await this.client.set(REVOKED_SOURCES_KEY, [...revoked, sourceIdentity], {
        scope,
        scopeId,
        metadata: this.metadata
      });
    }

    const slotMap =
      (await this.client.get<Record<string, string>>(POLICY_SLOT_MAP_KEY, {
        scope,
        scopeId,
        metadata: this.metadata
      })) ?? {};

    let deletedCount = 0;
    for (const [slotKey, recId] of Object.entries(slotMap)) {
      const rec = await this.getRecordById(recId, scope, scopeId);
      if (rec && rec.sourceIdentity === sourceIdentity) {
        await this.removeRecord(rec.id, slotKey, slotMap, scope, scopeId);
        deletedCount += 1;
      }
    }

    return deletedCount;
  }

  /**
   * Forgets stale records whose retention TTL (`expiresAt`) has passed as of `nowIso`.
   */
  async forgetStale(options?: {
    nowIso?: string;
    scope?: MemoryScope;
    scopeId?: string;
  }): Promise<number> {
    const scope = options?.scope ?? this.defaultScope;
    const scopeId = options?.scopeId ?? this.defaultScopeId;
    const nowIso = options?.nowIso ?? new Date().toISOString();

    const slotMap =
      (await this.client.get<Record<string, string>>(POLICY_SLOT_MAP_KEY, {
        scope,
        scopeId,
        metadata: this.metadata
      })) ?? {};

    let forgottenCount = 0;
    for (const [slotKey, recId] of Object.entries(slotMap)) {
      const rec = await this.getRecordById(recId, scope, scopeId);
      if (rec && rec.expiresAt && rec.expiresAt <= nowIso) {
        await this.removeRecord(rec.id, slotKey, slotMap, scope, scopeId);
        forgottenCount += 1;
      }
    }

    return forgottenCount;
  }

  /**
   * Queries active memory with strict tenant, RBAC, and TTL filtering, and abstains
   * (`status: 'INSUFFICIENT_EVIDENCE'`) if verified support is insufficient.
   */
  async queryWithAbstention(options?: {
    scope?: MemoryScope;
    scopeId?: string;
    callerRoles?: string[];
    keySlots?: string[];
    queryEmbedding?: number[];
    nowIso?: string;
    minEvidenceCount?: number;
    minConfidence?: number;
    topK?: number;
  }): Promise<AbstentionQueryResult> {
    const scope = options?.scope ?? this.defaultScope;
    const scopeId = options?.scopeId ?? this.defaultScopeId;
    const nowIso = options?.nowIso ?? new Date().toISOString();
    const minCount = options?.minEvidenceCount ?? this.minEvidenceCount;
    const minConf = options?.minConfidence ?? this.minAnswerConfidence;

    let candidates: PolicyManagedRecord[] = [];

    if (options?.queryEmbedding) {
      const hits: VectorSearchResult[] = await this.client.searchVector(options.queryEmbedding, {
        topK: options.topK ?? 10,
        scope,
        scopeId,
        filters: { scope, scopeId },
        metadata: this.metadata
      });
      for (const hit of hits) {
        if (hit.scope && hit.scope !== scope) continue;
        if (hit.scopeId && hit.scopeId !== scopeId) continue;
        const rid = hit.metadata?.recordId ?? this.idFromKey(hit.key);
        if (!rid) continue;
        const rec = await this.getRecordById(rid, scope, scopeId);
        if (rec) candidates.push(rec);
      }
    } else {
      const slotMap =
        (await this.client.get<Record<string, string>>(POLICY_SLOT_MAP_KEY, {
          scope,
          scopeId,
          metadata: this.metadata
        })) ?? {};
      const targetSlots = options?.keySlots
        ? options.keySlots.map(norm)
        : Object.keys(slotMap);
      for (const sk of targetSlots) {
        const rid = slotMap[sk];
        if (!rid) continue;
        const rec = await this.getRecordById(rid, scope, scopeId);
        if (rec) candidates.push(rec);
      }
    }

    const valid = candidates.filter((rec) => {
      if (rec.scope !== scope || rec.scopeId !== scopeId) return false;
      if (rec.expiresAt && rec.expiresAt <= nowIso) return false;
      if (!this.isAuthorized(rec.aclRoles, options?.callerRoles)) return false;
      if (rec.confidence < minConf) return false;
      return true;
    });

    if (valid.length < minCount) {
      return {
        status: 'INSUFFICIENT_EVIDENCE',
        reason: `Found ${valid.length} verified record(s) meeting confidence >= ${minConf} (required >= ${minCount})`,
        records: []
      };
    }

    return {
      status: 'ANSWER',
      reason: `Verified ${valid.length} supporting memory record(s)`,
      records: options?.topK ? valid.slice(0, options.topK) : valid
    };
  }

  /**
   * Rebuilds derived vector indices from persisted active records.
   */
  async rebuildDerivedIndex(options?: {
    scope?: MemoryScope;
    scopeId?: string;
    embedFn: (text: string) => Promise<number[]>;
  }): Promise<{ rebuiltCount: number }> {
    const scope = options?.scope ?? this.defaultScope;
    const scopeId = options?.scopeId ?? this.defaultScopeId;
    const slotMap =
      (await this.client.get<Record<string, string>>(POLICY_SLOT_MAP_KEY, {
        scope,
        scopeId,
        metadata: this.metadata
      })) ?? {};

    let rebuiltCount = 0;
    for (const recId of Object.values(slotMap)) {
      const rec = await this.getRecordById(recId, scope, scopeId);
      if (!rec) continue;
      const key = `${POLICY_RECORD_PREFIX}${rec.id}`;
      const emb = await options!.embedFn(rec.content);
      await this.client.setVector(
        key,
        emb,
        {
          recordId: rec.id,
          keySlot: rec.keySlot,
          scope,
          scopeId,
          version: rec.version
        },
        { scope, scopeId, metadata: this.metadata }
      );
      rebuiltCount += 1;
    }

    return { rebuiltCount };
  }

  async getRecordById(
    id: string,
    scope: MemoryScope = this.defaultScope,
    scopeId: string = this.defaultScopeId
  ): Promise<PolicyManagedRecord | undefined> {
    const rec = await this.client.get<PolicyManagedRecord>(`${POLICY_RECORD_PREFIX}${id}`, {
      scope,
      scopeId,
      metadata: this.metadata
    });
    if (!rec || rec.scope !== scope || rec.scopeId !== scopeId) return undefined;
    return rec;
  }

  private selectLearnedAction(
    features: { novelty: number; contradiction: number; confidence: number; utilityHint: number },
    hasExisting: boolean
  ): PolicyAction {
    if (features.confidence < this.minConfidenceForAdd) {
      return 'NOOP';
    }
    const fVec = [
      features.novelty,
      features.contradiction,
      features.confidence,
      features.utilityHint
    ];
    const candidateActions: PolicyAction[] = hasExisting
      ? ['UPDATE', 'NOOP', 'FORGET']
      : ['ADD', 'NOOP'];

    let bestAction: PolicyAction = candidateActions[0];
    let bestScore = -Infinity;
    for (const act of candidateActions) {
      const w = this.actionWeights[act];
      const score = w[0] * fVec[0] + w[1] * fVec[1] + w[2] * fVec[2] + w[3] * fVec[3];
      if (score > bestScore) {
        bestScore = score;
        bestAction = act;
      }
    }
    return bestAction;
  }

  private async removeRecord(
    recordId: string,
    slotKey: string,
    slotMap: Record<string, string>,
    scope: MemoryScope,
    scopeId: string
  ) {
    const recKey = `${POLICY_RECORD_PREFIX}${recordId}`;
    await this.client.delete(recKey, { scope, scopeId, metadata: this.metadata });
    await this.client.deleteVector(recKey, { scope, scopeId, metadata: this.metadata });
    delete slotMap[slotKey];
    await this.client.set(POLICY_SLOT_MAP_KEY, slotMap, {
      scope,
      scopeId,
      metadata: this.metadata
    });
  }

  private isAuthorized(aclRoles: string[], callerRoles?: string[]): boolean {
    if (!aclRoles || aclRoles.length === 0) return true;
    if (!callerRoles || callerRoles.length === 0) return false;
    const callerSet = new Set(callerRoles);
    return aclRoles.some((r) => callerSet.has(r));
  }

  private idFromKey(key: string): string | undefined {
    return key.startsWith(POLICY_RECORD_PREFIX)
      ? key.slice(POLICY_RECORD_PREFIX.length)
      : undefined;
  }
}
