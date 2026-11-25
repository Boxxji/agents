import crypto from 'crypto';
import type { MemoryScope } from '../types/agent.js';
import type { MemoryBackend } from './MemoryBackend.js';
import type { MemoryRequestMetadata, VectorSearchResult } from './MemoryClient.js';
import type { AIClient, AIEmbeddingOptions } from '../ai/AIClient.js';

export type EpistemicKind = 'OBSERVATION' | 'CONCLUSION' | 'HYPOTHESIS';
export type BeliefStatus = 'ACTIVE' | 'CONTESTED' | 'REFUTED';

export interface RetainObservationInput {
  statement: string;
  topic: string;
  sourceIdentity: string;
  sourceVersion?: number;
  aclRoles?: string[];
  metadata?: Record<string, any>;
}

export interface ReflectBeliefInput {
  kind: 'CONCLUSION' | 'HYPOTHESIS';
  statement: string;
  topic: string;
  supportingObservationIds: string[];
  contradictingObservationIds?: string[];
  sourceIdentity: string;
  sourceVersion?: number;
  confidence?: number;
  aclRoles?: string[];
  metadata?: Record<string, any>;
}

export interface EpistemicMemoryRecord {
  id: string;
  kind: EpistemicKind;
  status: BeliefStatus;
  topic: string;
  statement: string;
  sourceIdentity: string;
  sourceVersion: number;
  version: number;
  scope: MemoryScope;
  scopeId: string;
  confidence: number;
  supportingObservationIds: string[];
  contradictingObservationIds: string[];
  aclRoles: string[];
  metadata?: Record<string, any>;
  createdAt: string;
  updatedAt: string;
}

export interface RecallOptions {
  kinds?: EpistemicKind[];
  statuses?: BeliefStatus[];
  topic?: string;
  scope?: MemoryScope;
  scopeId?: string;
  callerRoles?: string[];
  queryEmbedding?: number[];
  queryText?: string;
  topK?: number;
}

export interface ReflectiveMemoryOptions {
  client: MemoryBackend;
  aiClient?: AIClient;
  defaultScope?: MemoryScope;
  defaultScopeId?: string;
  metadata?: MemoryRequestMetadata;
  embeddingOptions?: AIEmbeddingOptions;
}

const EPISTEMIC_RECORD_PREFIX = 'epistemic_record:';
const EPISTEMIC_INDEX_PREFIX = 'epistemic_index:';

function norm(val: string): string {
  return val.trim().toLowerCase().replace(/\s+/g, ' ');
}

export function computeEpistemicId(
  scope: MemoryScope,
  scopeId: string,
  kind: EpistemicKind,
  topic: string,
  statement: string,
  version: number
): string {
  const canonical = `${scope}:${scopeId}:${kind}:${norm(topic)}:${norm(statement)}:v${version}`;
  return crypto.createHash('sha256').update(canonical, 'utf8').digest('hex').slice(0, 32);
}

/**
 * Hindsight-style Reflective Memory (Retain, Recall, Reflect) with strict
 * separation between immutable observed evidence ('OBSERVATION') and inferred
 * beliefs ('CONCLUSION' / 'HYPOTHESIS').
 *
 * Guarantees:
 * - Every conclusion or hypothesis must link to verified supporting observations
 *   in the same tenant scope (0% unsupported memory rate).
 * - Reflections never silently overwrite observed facts.
 * - New contradicting observations recalibrate confidence and transition refuted
 *   beliefs to 'CONTESTED' or 'REFUTED'.
 */
export class ReflectiveMemory {
  private readonly client: MemoryBackend;
  private readonly aiClient?: AIClient;
  private readonly defaultScope: MemoryScope;
  private readonly defaultScopeId: string;
  private readonly metadata?: MemoryRequestMetadata;
  private readonly embeddingOptions?: AIEmbeddingOptions;

  constructor(options: ReflectiveMemoryOptions) {
    this.client = options.client;
    this.aiClient = options.aiClient;
    this.defaultScope = options.defaultScope ?? 'workflow';
    this.defaultScopeId = options.defaultScopeId ?? 'default';
    this.metadata = options.metadata;
    this.embeddingOptions = options.embeddingOptions;
  }

  /**
   * RETAIN: Persists an immutable observed evidence record ('OBSERVATION').
   * Observations cannot be overwritten by later reflections.
   */
  async retain(
    input: RetainObservationInput,
    options?: { scope?: MemoryScope; scopeId?: string; embedding?: number[] }
  ): Promise<EpistemicMemoryRecord> {
    const scope = options?.scope ?? this.defaultScope;
    const scopeId = options?.scopeId ?? this.defaultScopeId;
    const id = computeEpistemicId(scope, scopeId, 'OBSERVATION', input.topic, input.statement, 1);

    const existing = await this.getRecordById(id, scope, scopeId);
    if (existing) {
      return existing;
    }

    const nowIso = new Date().toISOString();
    const record: EpistemicMemoryRecord = {
      id,
      kind: 'OBSERVATION',
      status: 'ACTIVE',
      topic: input.topic.trim(),
      statement: input.statement.trim(),
      sourceIdentity: input.sourceIdentity,
      sourceVersion: input.sourceVersion ?? 1,
      version: 1,
      scope,
      scopeId,
      confidence: 1.0,
      supportingObservationIds: [],
      contradictingObservationIds: [],
      aclRoles: input.aclRoles ? [...input.aclRoles] : [],
      metadata: input.metadata,
      createdAt: nowIso,
      updatedAt: nowIso
    };

    let embedding = options?.embedding;
    if (!embedding && this.aiClient) {
      embedding = await this.aiClient.embed(
        `${record.topic}: ${record.statement}`,
        this.embeddingOptions
      );
    }

    await this.persistRecord(record, embedding);
    return record;
  }

  /**
   * REFLECT: Synthesizes or updates an inferred 'CONCLUSION' or 'HYPOTHESIS'.
   * Enforces evidence attribution: every supportingObservationId must exist as
   * an 'OBSERVATION' within the same (scope, scopeId) tenant boundary, and
   * never mutates or overwrites any underlying observation record.
   */
  async reflect(
    input: ReflectBeliefInput,
    options?: { scope?: MemoryScope; scopeId?: string; embedding?: number[] }
  ): Promise<EpistemicMemoryRecord> {
    const scope = options?.scope ?? this.defaultScope;
    const scopeId = options?.scopeId ?? this.defaultScopeId;

    if (!input.supportingObservationIds || input.supportingObservationIds.length === 0) {
      throw new Error('Unsupported belief rejected: conclusions and hypotheses require at least one supportingObservationId');
    }

    // Verify all supporting observations exist in the same tenant scope and are true OBSERVATION records
    const verifiedSupportIds: string[] = [];
    const inheritedRoles = new Set<string>(input.aclRoles ?? []);

    for (const obsId of input.supportingObservationIds) {
      const obs = await this.getRecordById(obsId, scope, scopeId);
      if (!obs || obs.kind !== 'OBSERVATION') {
        throw new Error(`Invalid supporting observation ID '${obsId}': must reference an existing OBSERVATION in the same scope`);
      }
      verifiedSupportIds.push(obs.id);
      obs.aclRoles.forEach((r) => inheritedRoles.add(r));
    }

    const verifiedContradictIds: string[] = [];
    for (const cId of input.contradictingObservationIds ?? []) {
      const obs = await this.getRecordById(cId, scope, scopeId);
      if (!obs || obs.kind !== 'OBSERVATION') {
        throw new Error(`Invalid contradicting observation ID '${cId}': must reference an existing OBSERVATION in the same scope`);
      }
      verifiedContradictIds.push(obs.id);
    }

    const calibrated = this.calibrateBelief(
      input.confidence ?? 0.8,
      verifiedSupportIds.length,
      verifiedContradictIds.length
    );

    const id = computeEpistemicId(scope, scopeId, input.kind, input.topic, input.statement, 1);
    const existing = await this.getRecordById(id, scope, scopeId);
    const nowIso = new Date().toISOString();

    const record: EpistemicMemoryRecord = {
      id,
      kind: input.kind,
      status: calibrated.status,
      topic: input.topic.trim(),
      statement: input.statement.trim(),
      sourceIdentity: input.sourceIdentity,
      sourceVersion: input.sourceVersion ?? 1,
      version: existing ? existing.version + 1 : 1,
      scope,
      scopeId,
      confidence: calibrated.confidence,
      supportingObservationIds: Array.from(new Set(verifiedSupportIds)),
      contradictingObservationIds: Array.from(new Set(verifiedContradictIds)),
      aclRoles: Array.from(inheritedRoles),
      metadata: input.metadata,
      createdAt: existing ? existing.createdAt : nowIso,
      updatedAt: nowIso
    };

    let embedding = options?.embedding;
    if (!embedding && this.aiClient) {
      embedding = await this.aiClient.embed(
        `${record.topic}: ${record.statement}`,
        this.embeddingOptions
      );
    }

    await this.persistRecord(record, embedding);
    return record;
  }

  /**
   * Registers a new contradicting observation against an existing conclusion or
   * hypothesis, recalibrating its confidence and updating its BeliefStatus without
   * altering any observation record.
   */
  async challengeBelief(
    beliefId: string,
    contradictingObservationId: string,
    options?: { scope?: MemoryScope; scopeId?: string }
  ): Promise<EpistemicMemoryRecord> {
    const scope = options?.scope ?? this.defaultScope;
    const scopeId = options?.scopeId ?? this.defaultScopeId;

    const belief = await this.getRecordById(beliefId, scope, scopeId);
    if (!belief) {
      throw new Error(`Belief record '${beliefId}' not found in scope`);
    }
    if (belief.kind === 'OBSERVATION') {
      throw new Error('Observed facts (OBSERVATION) are immutable and cannot be overwritten or challenged as inferred beliefs');
    }

    const obs = await this.getRecordById(contradictingObservationId, scope, scopeId);
    if (!obs || obs.kind !== 'OBSERVATION') {
      throw new Error(`Contradicting record '${contradictingObservationId}' must be an existing OBSERVATION in the same scope`);
    }

    const contradictingIds = Array.from(
      new Set([...belief.contradictingObservationIds, obs.id])
    );
    const calibrated = this.calibrateBelief(
      belief.confidence,
      belief.supportingObservationIds.length,
      contradictingIds.length
    );

    const updated: EpistemicMemoryRecord = {
      ...belief,
      status: calibrated.status,
      confidence: calibrated.confidence,
      version: belief.version + 1,
      contradictingObservationIds: contradictingIds,
      updatedAt: new Date().toISOString()
    };

    await this.client.set(this.recordKey(updated.id), updated, {
      scope,
      scopeId,
      metadata: this.metadata
    });

    if (updated.status === 'REFUTED') {
      await this.client.deleteVector(this.recordKey(updated.id), {
        scope,
        scopeId,
        metadata: this.metadata
      });
    }

    return updated;
  }

  /**
   * RECALL: Queries observations, conclusions, and hypotheses with strict
   * pre-retrieval tenant and RBAC filtering.
   */
  async recall(options: RecallOptions = {}): Promise<EpistemicMemoryRecord[]> {
    const scope = options.scope ?? this.defaultScope;
    const scopeId = options.scopeId ?? this.defaultScopeId;
    const allowedKinds = new Set<EpistemicKind>(
      options.kinds ?? ['OBSERVATION', 'CONCLUSION', 'HYPOTHESIS']
    );
    const allowedStatuses = new Set<BeliefStatus>(options.statuses ?? ['ACTIVE']);

    let candidates: EpistemicMemoryRecord[] = [];

    if (options.queryEmbedding || (options.queryText && this.aiClient)) {
      const emb =
        options.queryEmbedding ??
        (await this.aiClient!.embed(options.queryText!, this.embeddingOptions));
      const hits: VectorSearchResult[] = await this.client.searchVector(emb, {
        topK: options.topK ?? 10,
        scope,
        scopeId,
        filters: { scope, scopeId },
        metadata: this.metadata
      });
      for (const hit of hits) {
        if (hit.scope && hit.scope !== scope) continue;
        if (hit.scopeId && hit.scopeId !== scopeId) continue;
        const recId = hit.metadata?.recordId ?? this.idFromKey(hit.key);
        if (!recId) continue;
        const rec = await this.getRecordById(recId, scope, scopeId);
        if (rec) candidates.push(rec);
      }
    } else {
      for (const kind of allowedKinds) {
        const indexKey = `${EPISTEMIC_INDEX_PREFIX}${kind}`;
        const ids =
          (await this.client.get<string[]>(indexKey, {
            scope,
            scopeId,
            metadata: this.metadata
          })) ?? [];
        for (const id of ids) {
          const rec = await this.getRecordById(id, scope, scopeId);
          if (rec) candidates.push(rec);
        }
      }
    }

    const filtered = candidates.filter((rec) => {
      if (rec.scope !== scope || rec.scopeId !== scopeId) return false;
      if (!allowedKinds.has(rec.kind)) return false;
      if (!allowedStatuses.has(rec.status)) return false;
      if (options.topic && norm(rec.topic) !== norm(options.topic)) return false;
      if (!this.isAuthorized(rec, options.callerRoles)) return false;
      return true;
    });

    return options.topK ? filtered.slice(0, options.topK) : filtered;
  }

  /**
   * Rebuilds derived vector representations from immutable source records.
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
      if (!key.startsWith(EPISTEMIC_RECORD_PREFIX)) continue;
      const rec = await this.client.get<EpistemicMemoryRecord>(key, {
        scope,
        scopeId,
        metadata: this.metadata
      });
      if (!rec || rec.scope !== scope || rec.scopeId !== scopeId) continue;

      if (rec.status !== 'ACTIVE') {
        await this.client.deleteVector(key, { scope, scopeId, metadata: this.metadata });
        skippedCount += 1;
        continue;
      }

      const text = `${rec.topic}: ${rec.statement}`;
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
            recordId: rec.id,
            kind: rec.kind,
            scope: rec.scope,
            scopeId: rec.scopeId,
            version: rec.version
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

  async getRecordById(
    id: string,
    scope: MemoryScope = this.defaultScope,
    scopeId: string = this.defaultScopeId
  ): Promise<EpistemicMemoryRecord | undefined> {
    const rec = await this.client.get<EpistemicMemoryRecord>(this.recordKey(id), {
      scope,
      scopeId,
      metadata: this.metadata
    });
    if (!rec || rec.scope !== scope || rec.scopeId !== scopeId) {
      return undefined;
    }
    return rec;
  }

  private calibrateBelief(
    baseConfidence: number,
    supportCount: number,
    contradictCount: number
  ): { confidence: number; status: BeliefStatus } {
    if (contradictCount === 0) {
      const boost = Math.min(0.15, (supportCount - 1) * 0.05);
      return {
        confidence: Number(Math.min(1.0, Math.max(0.05, baseConfidence + boost)).toFixed(4)),
        status: 'ACTIVE'
      };
    }

    const ratio = supportCount / (supportCount + contradictCount);
    const calibrated = Number(Math.max(0.05, baseConfidence * ratio).toFixed(4));
    if (contradictCount >= supportCount || calibrated < 0.35) {
      return { confidence: calibrated, status: 'REFUTED' };
    }
    return { confidence: calibrated, status: 'CONTESTED' };
  }

  private async persistRecord(rec: EpistemicMemoryRecord, embedding?: number[]) {
    const key = this.recordKey(rec.id);
    await this.client.set(key, rec, {
      scope: rec.scope,
      scopeId: rec.scopeId,
      metadata: this.metadata
    });

    const indexKey = `${EPISTEMIC_INDEX_PREFIX}${rec.kind}`;
    const existingIds =
      (await this.client.get<string[]>(indexKey, {
        scope: rec.scope,
        scopeId: rec.scopeId,
        metadata: this.metadata
      })) ?? [];
    if (!existingIds.includes(rec.id)) {
      await this.client.set(indexKey, [...existingIds, rec.id], {
        scope: rec.scope,
        scopeId: rec.scopeId,
        metadata: this.metadata
      });
    }

    if (rec.status === 'ACTIVE' && embedding && embedding.length > 0) {
      await this.client.setVector(
        key,
        embedding,
        {
          recordId: rec.id,
          kind: rec.kind,
          scope: rec.scope,
          scopeId: rec.scopeId,
          version: rec.version
        },
        {
          scope: rec.scope,
          scopeId: rec.scopeId,
          metadata: this.metadata
        }
      );
    }
  }

  private isAuthorized(rec: EpistemicMemoryRecord, callerRoles?: string[]): boolean {
    if (!rec.aclRoles || rec.aclRoles.length === 0) return true;
    if (!callerRoles || callerRoles.length === 0) return false;
    const callerSet = new Set(callerRoles);
    return rec.aclRoles.some((r) => callerSet.has(r));
  }

  private recordKey(id: string): string {
    return `${EPISTEMIC_RECORD_PREFIX}${id}`;
  }

  private idFromKey(key: string): string | undefined {
    return key.startsWith(EPISTEMIC_RECORD_PREFIX)
      ? key.slice(EPISTEMIC_RECORD_PREFIX.length)
      : undefined;
  }
}
