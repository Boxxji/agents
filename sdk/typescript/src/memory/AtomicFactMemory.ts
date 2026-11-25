import crypto from 'crypto';
import type { MemoryScope } from '../types/agent.js';
import type { MemoryBackend } from './MemoryBackend.js';
import type { MemoryRequestMetadata, VectorSearchResult } from './MemoryClient.js';
import type { AIClient, AIEmbeddingOptions } from '../ai/AIClient.js';

export type FactLifecycle = 'ACTIVE' | 'SUPERSEDED' | 'PENDING_REVIEW';
export type FactDecisionAction = 'ADD' | 'UPDATE' | 'NOOP' | 'REVIEW';

export interface AtomicFactInput {
  subject: string;
  predicate: string;
  object: string;
  sourceIdentity: string;
  sourceVersion?: number;
  confidence?: number;
  aclRoles?: string[];
  summary?: string;
  metadata?: Record<string, any>;
}

export interface AtomicFactRecord {
  id: string;
  slotKey: string;
  subject: string;
  predicate: string;
  object: string;
  sourceIdentity: string;
  sourceVersion: number;
  scope: MemoryScope;
  scopeId: string;
  confidence: number;
  version: number;
  supersedesId?: string;
  lifecycle: FactLifecycle;
  aclRoles: string[];
  summary?: string;
  metadata?: Record<string, any>;
  createdAt: string;
  updatedAt: string;
}

export interface FactDecision {
  action: FactDecisionAction;
  reason: string;
  targetFactId?: string;
  record?: AtomicFactRecord;
  supersededRecord?: AtomicFactRecord;
}

export interface FactDecisionModel {
  decide(
    candidate: AtomicFactInput,
    existingSlotFacts: AtomicFactRecord[],
    similarFacts: Array<{ record: AtomicFactRecord; score: number }>
  ): Promise<{ action: FactDecisionAction; reason: string; targetFact?: AtomicFactRecord }>;
}

export interface AtomicFactQueryOptions {
  scope?: MemoryScope;
  scopeId?: string;
  callerRoles?: string[];
  subject?: string;
  predicate?: string;
  queryEmbedding?: number[];
  queryText?: string;
  topK?: number;
  includeLifecycle?: FactLifecycle[];
}

export interface AtomicFactBenchmarkMetrics {
  eventsProcessed: number;
  durableWrites: number;
  addCount: number;
  updateCount: number;
  noopCount: number;
  reviewCount: number;
  writeAmplification: number;
}

export interface AtomicFactStoreOptions {
  client: MemoryBackend;
  aiClient?: AIClient;
  decisionModel?: FactDecisionModel;
  defaultScope?: MemoryScope;
  defaultScopeId?: string;
  metadata?: MemoryRequestMetadata;
  reviewConfidenceThreshold?: number;
  semanticDuplicateThreshold?: number;
  embeddingOptions?: AIEmbeddingOptions;
}

const FACT_KEY_PREFIX = 'fact_record:';
const FACT_SLOT_PREFIX = 'fact_slot:';

function normalizeToken(value: string): string {
  return value.trim().toLowerCase().replace(/\s+/g, ' ');
}

export function computeSlotHash(
  scope: MemoryScope,
  scopeId: string,
  subject: string,
  predicate: string
): string {
  const canonical = `${scope}:${scopeId}:${normalizeToken(subject)}:${normalizeToken(predicate)}`;
  return crypto.createHash('sha256').update(canonical, 'utf8').digest('hex').slice(0, 24);
}

export function computeFactId(
  scope: MemoryScope,
  scopeId: string,
  subject: string,
  predicate: string,
  object: string,
  version: number
): string {
  const canonical = `${scope}:${scopeId}:${normalizeToken(subject)}:${normalizeToken(predicate)}:${normalizeToken(object)}:v${version}`;
  return crypto.createHash('sha256').update(canonical, 'utf8').digest('hex').slice(0, 32);
}

/**
 * Default deterministic structured-decision model ("Kai" reconciliation judge).
 * Makes ADD / UPDATE / NOOP / REVIEW decisions over bounded neighborhood candidates.
 */
export class DefaultFactDecisionModel implements FactDecisionModel {
  private readonly reviewThreshold: number;
  private readonly duplicateScoreThreshold: number;

  constructor(options?: { reviewConfidenceThreshold?: number; semanticDuplicateThreshold?: number }) {
    this.reviewThreshold = options?.reviewConfidenceThreshold ?? 0.5;
    this.duplicateScoreThreshold = options?.semanticDuplicateThreshold ?? 0.96;
  }

  async decide(
    candidate: AtomicFactInput,
    existingSlotFacts: AtomicFactRecord[],
    similarFacts: Array<{ record: AtomicFactRecord; score: number }>
  ): Promise<{ action: FactDecisionAction; reason: string; targetFact?: AtomicFactRecord }> {
    const confidence = candidate.confidence ?? 1.0;

    if (confidence < this.reviewThreshold) {
      return {
        action: 'REVIEW',
        reason: `Candidate confidence (${confidence}) is below review threshold (${this.reviewThreshold})`
      };
    }

    const normSubject = normalizeToken(candidate.subject);
    const normPredicate = normalizeToken(candidate.predicate);
    const normObject = normalizeToken(candidate.object);

    const activeSlotFacts = existingSlotFacts.filter((f) => f.lifecycle === 'ACTIVE');

    // 1. Check exact match on (subject, predicate, object) -> NOOP
    for (const existing of activeSlotFacts) {
      if (
        normalizeToken(existing.subject) === normSubject &&
        normalizeToken(existing.predicate) === normPredicate &&
        normalizeToken(existing.object) === normObject
      ) {
        return {
          action: 'NOOP',
          reason: 'Identical active atomic fact already exists in scope',
          targetFact: existing
        };
      }
    }

    // 2. Check high-similarity vector hit with identical normalized triple -> NOOP
    for (const sim of similarFacts) {
      if (
        sim.record.lifecycle === 'ACTIVE' &&
        sim.score >= this.duplicateScoreThreshold &&
        normalizeToken(sim.record.subject) === normSubject &&
        normalizeToken(sim.record.predicate) === normPredicate &&
        normalizeToken(sim.record.object) === normObject
      ) {
        return {
          action: 'NOOP',
          reason: `Semantic duplicate detected (score=${sim.score.toFixed(4)})`,
          targetFact: sim.record
        };
      }
    }

    // 3. Same (subject, predicate) slot with updated/contradictory object -> UPDATE or REVIEW
    if (activeSlotFacts.length > 0) {
      const latest = activeSlotFacts.reduce((a, b) => (a.version >= b.version ? a : b));
      if (confidence < latest.confidence * 0.7) {
        return {
          action: 'REVIEW',
          reason: `Contradictory update has significantly lower confidence (${confidence}) than active v${latest.version} (${latest.confidence})`,
          targetFact: latest
        };
      }
      return {
        action: 'UPDATE',
        reason: `Reconciling (subject, predicate) slot from v${latest.version} to v${latest.version + 1}`,
        targetFact: latest
      };
    }

    // 4. Novel fact -> ADD
    return {
      action: 'ADD',
      reason: 'Novel atomic fact for scope'
    };
  }
}

/**
 * Mem0-style Atomic Fact Memory Store.
 * Separates immutable/versioned source fact records from derived vector representations
 * and enforces strict pre-retrieval tenant and RBAC isolation.
 */
export class AtomicFactMemory {
  private readonly client: MemoryBackend;
  private readonly aiClient?: AIClient;
  private readonly decisionModel: FactDecisionModel;
  private readonly defaultScope: MemoryScope;
  private readonly defaultScopeId: string;
  private readonly metadata?: MemoryRequestMetadata;
  private readonly embeddingOptions?: AIEmbeddingOptions;

  private eventsProcessed = 0;
  private durableWrites = 0;
  private addCount = 0;
  private updateCount = 0;
  private noopCount = 0;
  private reviewCount = 0;

  constructor(options: AtomicFactStoreOptions) {
    this.client = options.client;
    this.aiClient = options.aiClient;
    this.decisionModel =
      options.decisionModel ??
      new DefaultFactDecisionModel({
        reviewConfidenceThreshold: options.reviewConfidenceThreshold,
        semanticDuplicateThreshold: options.semanticDuplicateThreshold
      });
    this.defaultScope = options.defaultScope ?? 'workflow';
    this.defaultScopeId = options.defaultScopeId ?? 'default';
    this.metadata = options.metadata;
    this.embeddingOptions = options.embeddingOptions;
  }

  /**
   * Ingests a structured fact candidate, deduplicates and reconciles against existing
   * scoped memory via the decision model (ADD / UPDATE / NOOP / REVIEW), and updates
   * derived vector indices only on selective writes.
   */
  async ingestFact(
    input: AtomicFactInput,
    options?: {
      scope?: MemoryScope;
      scopeId?: string;
      embedding?: number[];
    }
  ): Promise<FactDecision> {
    this.eventsProcessed += 1;
    const scope = options?.scope ?? this.defaultScope;
    const scopeId = options?.scopeId ?? this.defaultScopeId;
    const slotHash = computeSlotHash(scope, scopeId, input.subject, input.predicate);
    const slotIndexKey = `${FACT_SLOT_PREFIX}${slotHash}`;

    // 1. Load bounded neighborhood of existing facts in the exact same (scope, scopeId, subject, predicate) slot
    const slotFactIds =
      (await this.client.get<string[]>(slotIndexKey, {
        scope,
        scopeId,
        metadata: this.metadata
      })) ?? [];

    const existingSlotFacts: AtomicFactRecord[] = [];
    for (const factId of slotFactIds) {
      const rec = await this.getFactById(factId, scope, scopeId);
      if (rec && rec.scope === scope && rec.scopeId === scopeId) {
        existingSlotFacts.push(rec);
      }
    }

    // 2. Optional vector neighborhood lookup within the same tenant boundary
    let candidateEmbedding = options?.embedding;
    if (!candidateEmbedding && this.aiClient) {
      candidateEmbedding = await this.aiClient.embed(
        this.formatFactText(input.subject, input.predicate, input.object),
        this.embeddingOptions
      );
    }

    const similarFacts: Array<{ record: AtomicFactRecord; score: number }> = [];
    if (candidateEmbedding && candidateEmbedding.length > 0) {
      const vectorHits = await this.client.searchVector(candidateEmbedding, {
        topK: 5,
        scope,
        scopeId,
        filters: { scope, scopeId },
        metadata: this.metadata
      });
      for (const hit of vectorHits) {
        if (hit.scope && hit.scope !== scope) continue;
        if (hit.scopeId && hit.scopeId !== scopeId) continue;
        const factId = hit.metadata?.factId ?? this.factIdFromKey(hit.key);
        if (!factId) continue;
        const rec = await this.getFactById(factId, scope, scopeId);
        if (rec && rec.scope === scope && rec.scopeId === scopeId) {
          similarFacts.push({ record: rec, score: hit.score });
        }
      }
    }

    // 3. Structured decision (ADD / UPDATE / NOOP / REVIEW)
    const decision = await this.decisionModel.decide(input, existingSlotFacts, similarFacts);

    if (decision.action === 'NOOP') {
      this.noopCount += 1;
      return {
        action: 'NOOP',
        reason: decision.reason,
        targetFactId: decision.targetFact?.id,
        record: decision.targetFact
      };
    }

    const nowIso = new Date().toISOString();
    const confidence = input.confidence ?? 1.0;
    const aclRoles = input.aclRoles ? [...input.aclRoles] : [];

    if (decision.action === 'REVIEW') {
      this.reviewCount += 1;
      const version = (decision.targetFact?.version ?? 0) + 1;
      const factId = computeFactId(scope, scopeId, input.subject, input.predicate, input.object, version);
      const reviewRecord: AtomicFactRecord = {
        id: factId,
        slotKey: slotHash,
        subject: input.subject.trim(),
        predicate: input.predicate.trim(),
        object: input.object.trim(),
        sourceIdentity: input.sourceIdentity,
        sourceVersion: input.sourceVersion ?? 1,
        scope,
        scopeId,
        confidence,
        version,
        supersedesId: decision.targetFact?.id,
        lifecycle: 'PENDING_REVIEW',
        aclRoles,
        summary: input.summary,
        metadata: input.metadata,
        createdAt: nowIso,
        updatedAt: nowIso
      };

      await this.persistFactRecord(reviewRecord, slotIndexKey, slotFactIds);
      return {
        action: 'REVIEW',
        reason: decision.reason,
        targetFactId: decision.targetFact?.id,
        record: reviewRecord
      };
    }

    if (decision.action === 'UPDATE' && decision.targetFact) {
      this.updateCount += 1;
      const prior = decision.targetFact;
      const supersededRecord: AtomicFactRecord = {
        ...prior,
        lifecycle: 'SUPERSEDED',
        updatedAt: nowIso
      };

      // Mark prior version as SUPERSEDED in durable store and remove its derived vector
      await this.client.set(this.recordKey(prior.id), supersededRecord, {
        scope,
        scopeId,
        metadata: this.metadata
      });
      this.durableWrites += 1;
      await this.client.deleteVector(this.recordKey(prior.id), {
        scope,
        scopeId,
        metadata: this.metadata
      });

      const nextVersion = prior.version + 1;
      const newId = computeFactId(scope, scopeId, input.subject, input.predicate, input.object, nextVersion);
      const updatedRecord: AtomicFactRecord = {
        id: newId,
        slotKey: slotHash,
        subject: input.subject.trim(),
        predicate: input.predicate.trim(),
        object: input.object.trim(),
        sourceIdentity: input.sourceIdentity,
        sourceVersion: input.sourceVersion ?? 1,
        scope,
        scopeId,
        confidence,
        version: nextVersion,
        supersedesId: prior.id,
        lifecycle: 'ACTIVE',
        aclRoles,
        summary: input.summary,
        metadata: input.metadata,
        createdAt: nowIso,
        updatedAt: nowIso
      };

      await this.persistFactRecord(updatedRecord, slotIndexKey, slotFactIds, candidateEmbedding);
      return {
        action: 'UPDATE',
        reason: decision.reason,
        targetFactId: prior.id,
        record: updatedRecord,
        supersededRecord
      };
    }

    // Default: ADD
    this.addCount += 1;
    const version = 1;
    const newId = computeFactId(scope, scopeId, input.subject, input.predicate, input.object, version);
    const addedRecord: AtomicFactRecord = {
      id: newId,
      slotKey: slotHash,
      subject: input.subject.trim(),
      predicate: input.predicate.trim(),
      object: input.object.trim(),
      sourceIdentity: input.sourceIdentity,
      sourceVersion: input.sourceVersion ?? 1,
      scope,
      scopeId,
      confidence,
      version,
      lifecycle: 'ACTIVE',
      aclRoles,
      summary: input.summary,
      metadata: input.metadata,
      createdAt: nowIso,
      updatedAt: nowIso
    };

    await this.persistFactRecord(addedRecord, slotIndexKey, slotFactIds, candidateEmbedding);
    return {
      action: 'ADD',
      reason: decision.reason,
      record: addedRecord
    };
  }

  /**
   * Retrieves atomic facts with strict pre-retrieval tenant (scope + scopeId)
   * and RBAC (callerRoles) filtering.
   */
  async queryFacts(options: AtomicFactQueryOptions = {}): Promise<AtomicFactRecord[]> {
    const scope = options.scope ?? this.defaultScope;
    const scopeId = options.scopeId ?? this.defaultScopeId;
    const allowedLifecycle = new Set<FactLifecycle>(options.includeLifecycle ?? ['ACTIVE']);

    let candidateRecords: AtomicFactRecord[] = [];

    // Fast path: if querying a specific (subject, predicate) slot
    if (options.subject && options.predicate) {
      const slotHash = computeSlotHash(scope, scopeId, options.subject, options.predicate);
      const slotIds =
        (await this.client.get<string[]>(`${FACT_SLOT_PREFIX}${slotHash}`, {
          scope,
          scopeId,
          metadata: this.metadata
        })) ?? [];
      for (const id of slotIds) {
        const rec = await this.getFactById(id, scope, scopeId);
        if (rec) candidateRecords.push(rec);
      }
    } else if (options.queryEmbedding || (options.queryText && this.aiClient)) {
      const embedding =
        options.queryEmbedding ??
        (await this.aiClient!.embed(options.queryText!, this.embeddingOptions));
      const vectorHits: VectorSearchResult[] = await this.client.searchVector(embedding, {
        topK: options.topK ?? 10,
        scope,
        scopeId,
        filters: { scope, scopeId },
        metadata: this.metadata
      });
      for (const hit of vectorHits) {
        if (hit.scope && hit.scope !== scope) continue;
        if (hit.scopeId && hit.scopeId !== scopeId) continue;
        const factId = hit.metadata?.factId ?? this.factIdFromKey(hit.key);
        if (!factId) continue;
        const rec = await this.getFactById(factId, scope, scopeId);
        if (rec) candidateRecords.push(rec);
      }
    } else {
      const allKeys = await this.client.listKeys(scope, {
        scope,
        scopeId,
        metadata: this.metadata
      });
      for (const key of allKeys) {
        if (!key.startsWith(FACT_KEY_PREFIX)) continue;
        const rec = await this.client.get<AtomicFactRecord>(key, {
          scope,
          scopeId,
          metadata: this.metadata
        });
        if (rec) candidateRecords.push(rec);
      }
    }

    // Strict tenant isolation + RBAC pre-filter + lifecycle filter
    const filtered = candidateRecords.filter((rec) => {
      if (rec.scope !== scope || rec.scopeId !== scopeId) return false;
      if (!allowedLifecycle.has(rec.lifecycle)) return false;
      if (options.subject && normalizeToken(rec.subject) !== normalizeToken(options.subject)) {
        return false;
      }
      if (options.predicate && normalizeToken(rec.predicate) !== normalizeToken(options.predicate)) {
        return false;
      }
      if (!this.isAuthorized(rec, options.callerRoles)) {
        return false;
      }
      return true;
    });

    return options.topK ? filtered.slice(0, options.topK) : filtered;
  }

  /**
   * Rebuilds derived vector indexes from immutable persisted source fact records
   * without mutating record identities, versions, or audit history.
   */
  async rebuildDerivedIndex(options?: {
    scope?: MemoryScope;
    scopeId?: string;
    embedFn?: (text: string) => Promise<number[]>;
  }): Promise<{ rebuiltCount: number; skippedCount: number }> {
    const scope = options?.scope ?? this.defaultScope;
    const scopeId = options?.scopeId ?? this.defaultScopeId;

    const allKeys = await this.client.listKeys(scope, {
      scope,
      scopeId,
      metadata: this.metadata
    });

    let rebuiltCount = 0;
    let skippedCount = 0;

    for (const key of allKeys) {
      if (!key.startsWith(FACT_KEY_PREFIX)) continue;
      const rec = await this.client.get<AtomicFactRecord>(key, {
        scope,
        scopeId,
        metadata: this.metadata
      });
      if (!rec || rec.scope !== scope || rec.scopeId !== scopeId) continue;

      if (rec.lifecycle !== 'ACTIVE') {
        await this.client.deleteVector(key, { scope, scopeId, metadata: this.metadata });
        skippedCount += 1;
        continue;
      }

      const text = this.formatFactText(rec.subject, rec.predicate, rec.object);
      let embedding: number[] | undefined;
      if (options?.embedFn) {
        embedding = await options.embedFn(text);
      } else if (this.aiClient) {
        embedding = await this.aiClient.embed(text, this.embeddingOptions);
      }

      if (embedding && embedding.length > 0) {
        await this.client.setVector(
          key,
          embedding,
          {
            factId: rec.id,
            slotKey: rec.slotKey,
            scope: rec.scope,
            scopeId: rec.scopeId,
            subject: rec.subject,
            predicate: rec.predicate,
            version: rec.version,
            sourceIdentity: rec.sourceIdentity
          },
          { scope, scopeId, metadata: this.metadata }
        );
        rebuiltCount += 1;
      } else {
        skippedCount += 1;
      }
    }

    return { rebuiltCount, skippedCount };
  }

  async getFactById(
    factId: string,
    scope: MemoryScope = this.defaultScope,
    scopeId: string = this.defaultScopeId
  ): Promise<AtomicFactRecord | undefined> {
    const rec = await this.client.get<AtomicFactRecord>(this.recordKey(factId), {
      scope,
      scopeId,
      metadata: this.metadata
    });
    if (!rec || rec.scope !== scope || rec.scopeId !== scopeId) {
      return undefined;
    }
    return rec;
  }

  getMetrics(): AtomicFactBenchmarkMetrics {
    return {
      eventsProcessed: this.eventsProcessed,
      durableWrites: this.durableWrites,
      addCount: this.addCount,
      updateCount: this.updateCount,
      noopCount: this.noopCount,
      reviewCount: this.reviewCount,
      writeAmplification:
        this.eventsProcessed > 0
          ? Number((this.durableWrites / this.eventsProcessed).toFixed(4))
          : 0
    };
  }

  private async persistFactRecord(
    record: AtomicFactRecord,
    slotIndexKey: string,
    existingSlotIds: string[],
    embedding?: number[]
  ) {
    const key = this.recordKey(record.id);
    await this.client.set(key, record, {
      scope: record.scope,
      scopeId: record.scopeId,
      metadata: this.metadata
    });
    this.durableWrites += 1;

    const updatedSlotIds = existingSlotIds.includes(record.id)
      ? existingSlotIds
      : [...existingSlotIds, record.id];
    await this.client.set(slotIndexKey, updatedSlotIds, {
      scope: record.scope,
      scopeId: record.scopeId,
      metadata: this.metadata
    });

    if (record.lifecycle === 'ACTIVE' && embedding && embedding.length > 0) {
      await this.client.setVector(
        key,
        embedding,
        {
          factId: record.id,
          slotKey: record.slotKey,
          scope: record.scope,
          scopeId: record.scopeId,
          subject: record.subject,
          predicate: record.predicate,
          version: record.version,
          sourceIdentity: record.sourceIdentity
        },
        {
          scope: record.scope,
          scopeId: record.scopeId,
          metadata: this.metadata
        }
      );
    }
  }

  private isAuthorized(record: AtomicFactRecord, callerRoles?: string[]): boolean {
    if (!record.aclRoles || record.aclRoles.length === 0) {
      return true;
    }
    if (!callerRoles || callerRoles.length === 0) {
      return false;
    }
    const callerSet = new Set(callerRoles);
    return record.aclRoles.some((role) => callerSet.has(role));
  }

  private recordKey(factId: string): string {
    return `${FACT_KEY_PREFIX}${factId}`;
  }

  private factIdFromKey(key: string): string | undefined {
    return key.startsWith(FACT_KEY_PREFIX) ? key.slice(FACT_KEY_PREFIX.length) : undefined;
  }

  private formatFactText(subject: string, predicate: string, object: string): string {
    return `${subject.trim()} ${predicate.trim()} ${object.trim()}`;
  }
}
