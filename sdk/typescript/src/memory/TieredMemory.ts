import crypto from 'crypto';
import type { MemoryScope } from '../types/agent.js';
import type { MemoryBackend } from './MemoryBackend.js';
import type { MemoryRequestMetadata, VectorSearchResult } from './MemoryClient.js';
import type { AIClient, AIEmbeddingOptions } from '../ai/AIClient.js';

export type MemoryTier = 'episodic' | 'semantic' | 'procedural';
export type TierRecordState = 'ACTIVE' | 'COMPACTED' | 'ARCHIVED';

export interface EpisodicPayload {
  runId: string;
  stepIndex: number;
  action: string;
  observation: string;
  outcome?: 'success' | 'failure' | 'observation';
  tokensUsed?: number;
}

export interface SemanticPayload {
  topic: string;
  statement: string;
  relationships?: string[];
  constraints?: string[];
}

export interface ProceduralPayload {
  name: string;
  triggerPattern: string;
  steps: string[];
  recoverySteps?: string[];
  successCount?: number;
}

export interface TieredMemoryInput {
  tier: MemoryTier;
  sourceIdentity: string;
  sourceVersion?: number;
  content: string;
  episodic?: EpisodicPayload;
  semantic?: SemanticPayload;
  procedural?: ProceduralPayload;
  lineageIds?: string[];
  aclRoles?: string[];
  importance?: number;
  metadata?: Record<string, any>;
}

export interface TieredMemoryRecord {
  id: string;
  tier: MemoryTier;
  sourceIdentity: string;
  sourceVersion: number;
  version: number;
  scope: MemoryScope;
  scopeId: string;
  state: TierRecordState;
  content: string;
  episodic?: EpisodicPayload;
  semantic?: SemanticPayload;
  procedural?: ProceduralPayload;
  lineageIds: string[];
  compactedIntoId?: string;
  aclRoles: string[];
  importance: number;
  metadata?: Record<string, any>;
  createdAt: string;
  updatedAt: string;
}

export interface TieredQueryOptions {
  tiers?: MemoryTier[];
  scope?: MemoryScope;
  scopeId?: string;
  callerRoles?: string[];
  runId?: string;
  triggerPattern?: string;
  queryEmbedding?: number[];
  queryText?: string;
  includeStates?: TierRecordState[];
  topK?: number;
}

export interface RunRecoveryState {
  runId: string;
  lastStepIndex: number;
  steps: TieredMemoryRecord[];
  semanticContext: TieredMemoryRecord[];
  applicableProcedures: TieredMemoryRecord[];
}

export interface TieredMemoryOptions {
  client: MemoryBackend;
  aiClient?: AIClient;
  defaultScope?: MemoryScope;
  defaultScopeId?: string;
  metadata?: MemoryRequestMetadata;
  episodicCompactionThreshold?: number;
  embeddingOptions?: AIEmbeddingOptions;
}

const TIER_RECORD_PREFIX = 'tiered_record:';
const TIER_INDEX_PREFIX = 'tiered_index:';

function normalizeStr(val: string): string {
  return val.trim().toLowerCase().replace(/\s+/g, ' ');
}

export function computeTieredRecordId(
  scope: MemoryScope,
  scopeId: string,
  tier: MemoryTier,
  sourceIdentity: string,
  content: string,
  version: number
): string {
  const canonical = `${scope}:${scopeId}:${tier}:${normalizeStr(sourceIdentity)}:${normalizeStr(content)}:v${version}`;
  return crypto.createHash('sha256').update(canonical, 'utf8').digest('hex').slice(0, 32);
}

/**
 * Unified Tiered Memory API (Episodic, Semantic, Procedural) with separate
 * retention, retrieval, and compaction over persistent storage.
 */
export class TieredMemory {
  private readonly client: MemoryBackend;
  private readonly aiClient?: AIClient;
  private readonly defaultScope: MemoryScope;
  private readonly defaultScopeId: string;
  private readonly metadata?: MemoryRequestMetadata;
  private readonly episodicCompactionThreshold: number;
  private readonly embeddingOptions?: AIEmbeddingOptions;

  constructor(options: TieredMemoryOptions) {
    this.client = options.client;
    this.aiClient = options.aiClient;
    this.defaultScope = options.defaultScope ?? 'workflow';
    this.defaultScopeId = options.defaultScopeId ?? 'default';
    this.metadata = options.metadata;
    this.episodicCompactionThreshold = options.episodicCompactionThreshold ?? 20;
    this.embeddingOptions = options.embeddingOptions;
  }

  /**
   * Writes a record to the specified tier ('episodic' | 'semantic' | 'procedural')
   * in persistent storage and indexes its derived vector representation.
   */
  async record(
    input: TieredMemoryInput,
    options?: {
      scope?: MemoryScope;
      scopeId?: string;
      embedding?: number[];
      version?: number;
    }
  ): Promise<TieredMemoryRecord> {
    const scope = options?.scope ?? this.defaultScope;
    const scopeId = options?.scopeId ?? this.defaultScopeId;
    const version = options?.version ?? 1;
    const id = computeTieredRecordId(
      scope,
      scopeId,
      input.tier,
      input.sourceIdentity,
      input.content,
      version
    );

    const existing = await this.getRecordById(id, scope, scopeId);
    if (existing) {
      return existing;
    }

    const nowIso = new Date().toISOString();
    const rec: TieredMemoryRecord = {
      id,
      tier: input.tier,
      sourceIdentity: input.sourceIdentity,
      sourceVersion: input.sourceVersion ?? 1,
      version,
      scope,
      scopeId,
      state: 'ACTIVE',
      content: input.content.trim(),
      episodic: input.episodic,
      semantic: input.semantic,
      procedural: input.procedural,
      lineageIds: input.lineageIds ? [...input.lineageIds] : [],
      aclRoles: input.aclRoles ? [...input.aclRoles] : [],
      importance: input.importance ?? 1.0,
      metadata: input.metadata,
      createdAt: nowIso,
      updatedAt: nowIso
    };

    let embedding = options?.embedding;
    if (!embedding && this.aiClient) {
      embedding = await this.aiClient.embed(rec.content, this.embeddingOptions);
    }

    await this.persistRecord(rec, embedding);
    return rec;
  }

  /**
   * Convenience helper to record an episodic observation/event step.
   */
  async recordEpisode(
    params: {
      runId: string;
      stepIndex: number;
      action: string;
      observation: string;
      outcome?: 'success' | 'failure' | 'observation';
      tokensUsed?: number;
      sourceIdentity: string;
      aclRoles?: string[];
      embedding?: number[];
    },
    scopeOptions?: { scope?: MemoryScope; scopeId?: string }
  ): Promise<TieredMemoryRecord> {
    return this.record(
      {
        tier: 'episodic',
        sourceIdentity: params.sourceIdentity,
        content: `[Run ${params.runId} #${params.stepIndex}] ${params.action} -> ${params.observation}`,
        episodic: {
          runId: params.runId,
          stepIndex: params.stepIndex,
          action: params.action,
          observation: params.observation,
          outcome: params.outcome ?? 'observation',
          tokensUsed: params.tokensUsed
        },
        aclRoles: params.aclRoles
      },
      { ...scopeOptions, embedding: params.embedding }
    );
  }

  /**
   * Convenience helper to store durable semantic knowledge/decisions.
   */
  async recordSemantic(
    params: {
      topic: string;
      statement: string;
      relationships?: string[];
      constraints?: string[];
      sourceIdentity: string;
      lineageIds?: string[];
      aclRoles?: string[];
      embedding?: number[];
    },
    scopeOptions?: { scope?: MemoryScope; scopeId?: string }
  ): Promise<TieredMemoryRecord> {
    return this.record(
      {
        tier: 'semantic',
        sourceIdentity: params.sourceIdentity,
        content: `${params.topic}: ${params.statement}`,
        semantic: {
          topic: params.topic,
          statement: params.statement,
          relationships: params.relationships,
          constraints: params.constraints
        },
        lineageIds: params.lineageIds,
        aclRoles: params.aclRoles
      },
      { ...scopeOptions, embedding: params.embedding }
    );
  }

  /**
   * Convenience helper to store reusable procedural skills/workflows/recovery steps.
   */
  async recordProcedure(
    params: {
      name: string;
      triggerPattern: string;
      steps: string[];
      recoverySteps?: string[];
      sourceIdentity: string;
      lineageIds?: string[];
      aclRoles?: string[];
      embedding?: number[];
    },
    scopeOptions?: { scope?: MemoryScope; scopeId?: string }
  ): Promise<TieredMemoryRecord> {
    return this.record(
      {
        tier: 'procedural',
        sourceIdentity: params.sourceIdentity,
        content: `${params.name} (trigger: ${params.triggerPattern}): ${params.steps.join(' -> ')}`,
        procedural: {
          name: params.name,
          triggerPattern: params.triggerPattern,
          steps: [...params.steps],
          recoverySteps: params.recoverySteps ? [...params.recoverySteps] : [],
          successCount: 1
        },
        lineageIds: params.lineageIds,
        aclRoles: params.aclRoles
      },
      { ...scopeOptions, embedding: params.embedding }
    );
  }

  /**
   * Compacts active episodic records into a consolidated semantic record while
   * preserving the immutable episodic records in persistent storage (marked 'COMPACTED')
   * with full lineage pointers.
   */
  async compactEpisodic(options?: {
    runId?: string;
    scope?: MemoryScope;
    scopeId?: string;
    topic?: string;
    summarizer?: (episodes: TieredMemoryRecord[]) => Promise<string> | string;
    embedding?: number[];
  }): Promise<{ semanticRecord?: TieredMemoryRecord; compactedIds: string[] }> {
    const scope = options?.scope ?? this.defaultScope;
    const scopeId = options?.scopeId ?? this.defaultScopeId;

    const episodes = await this.retrieve({
      tiers: ['episodic'],
      scope,
      scopeId,
      runId: options?.runId,
      includeStates: ['ACTIVE']
    });

    if (episodes.length === 0) {
      return { compactedIds: [] };
    }

    episodes.sort((a, b) => (a.episodic?.stepIndex ?? 0) - (b.episodic?.stepIndex ?? 0));

    const summaryText = options?.summarizer
      ? await options.summarizer(episodes)
      : episodes.map((e) => e.content).join(' | ');

    const topic = options?.topic ?? `compacted-run:${options?.runId ?? 'all'}`;
    const compactedIds = episodes.map((e) => e.id);
    const mergedAcl = Array.from(new Set(episodes.flatMap((e) => e.aclRoles)));

    const semanticRecord = await this.recordSemantic(
      {
        topic,
        statement: summaryText,
        sourceIdentity: `compaction:${compactedIds[0]}`,
        lineageIds: compactedIds,
        aclRoles: mergedAcl,
        embedding: options?.embedding
      },
      { scope, scopeId }
    );

    const nowIso = new Date().toISOString();
    for (const ep of episodes) {
      const updatedEp: TieredMemoryRecord = {
        ...ep,
        state: 'COMPACTED',
        compactedIntoId: semanticRecord.id,
        updatedAt: nowIso
      };
      await this.client.set(this.recordKey(ep.id), updatedEp, {
        scope,
        scopeId,
        metadata: this.metadata
      });
      await this.client.deleteVector(this.recordKey(ep.id), {
        scope,
        scopeId,
        metadata: this.metadata
      });
    }

    return { semanticRecord, compactedIds };
  }

  /**
   * Extracts a reusable procedural memory record from successful episodic steps
   * of a run, linking back to the source episodic record IDs.
   */
  async extractProcedureFromRun(params: {
    runId: string;
    name: string;
    triggerPattern: string;
    recoverySteps?: string[];
    scope?: MemoryScope;
    scopeId?: string;
    embedding?: number[];
  }): Promise<TieredMemoryRecord> {
    const scope = params.scope ?? this.defaultScope;
    const scopeId = params.scopeId ?? this.defaultScopeId;

    const episodes = await this.retrieve({
      tiers: ['episodic'],
      scope,
      scopeId,
      runId: params.runId,
      includeStates: ['ACTIVE', 'COMPACTED']
    });

    episodes.sort((a, b) => (a.episodic?.stepIndex ?? 0) - (b.episodic?.stepIndex ?? 0));
    const steps = episodes.map((e) => e.episodic?.action ?? e.content);
    const lineageIds = episodes.map((e) => e.id);
    const mergedAcl = Array.from(new Set(episodes.flatMap((e) => e.aclRoles)));

    return this.recordProcedure(
      {
        name: params.name,
        triggerPattern: params.triggerPattern,
        steps,
        recoverySteps: params.recoverySteps,
        sourceIdentity: `procedure-extraction:${params.runId}`,
        lineageIds,
        aclRoles: mergedAcl,
        embedding: params.embedding
      },
      { scope, scopeId }
    );
  }

  /**
   * Stateless process-restart recovery: reconstructs a run's execution trajectory,
   * semantic knowledge, and matching procedural workflows directly from persistent storage.
   */
  async recoverRunState(
    runId: string,
    options?: {
      scope?: MemoryScope;
      scopeId?: string;
      callerRoles?: string[];
      triggerPattern?: string;
    }
  ): Promise<RunRecoveryState> {
    const scope = options?.scope ?? this.defaultScope;
    const scopeId = options?.scopeId ?? this.defaultScopeId;

    const steps = await this.retrieve({
      tiers: ['episodic'],
      scope,
      scopeId,
      callerRoles: options?.callerRoles,
      runId,
      includeStates: ['ACTIVE', 'COMPACTED']
    });
    steps.sort((a, b) => (a.episodic?.stepIndex ?? 0) - (b.episodic?.stepIndex ?? 0));

    const semanticContext = await this.retrieve({
      tiers: ['semantic'],
      scope,
      scopeId,
      callerRoles: options?.callerRoles,
      includeStates: ['ACTIVE']
    });

    const applicableProcedures = await this.retrieve({
      tiers: ['procedural'],
      scope,
      scopeId,
      callerRoles: options?.callerRoles,
      triggerPattern: options?.triggerPattern,
      includeStates: ['ACTIVE']
    });

    const lastStepIndex =
      steps.length > 0
        ? Math.max(...steps.map((s) => s.episodic?.stepIndex ?? 0))
        : -1;

    return {
      runId,
      lastStepIndex,
      steps,
      semanticContext,
      applicableProcedures
    };
  }

  /**
   * Unified retrieval across any combination of ('episodic', 'semantic', 'procedural')
   * with strict pre-retrieval tenant (scope + scopeId) and RBAC (callerRoles) enforcement.
   */
  async retrieve(options: TieredQueryOptions = {}): Promise<TieredMemoryRecord[]> {
    const scope = options.scope ?? this.defaultScope;
    const scopeId = options.scopeId ?? this.defaultScopeId;
    const tiers = options.tiers ?? ['episodic', 'semantic', 'procedural'];
    const allowedTiers = new Set<MemoryTier>(tiers);
    const allowedStates = new Set<TierRecordState>(options.includeStates ?? ['ACTIVE']);

    let candidates: TieredMemoryRecord[] = [];

    if (options.queryEmbedding || (options.queryText && this.aiClient)) {
      const emb =
        options.queryEmbedding ??
        (await this.aiClient!.embed(options.queryText!, this.embeddingOptions));
      const vectorHits: VectorSearchResult[] = await this.client.searchVector(emb, {
        topK: options.topK ?? 10,
        scope,
        scopeId,
        filters: { scope, scopeId },
        metadata: this.metadata
      });
      for (const hit of vectorHits) {
        if (hit.scope && hit.scope !== scope) continue;
        if (hit.scopeId && hit.scopeId !== scopeId) continue;
        const recId = hit.metadata?.recordId ?? this.idFromKey(hit.key);
        if (!recId) continue;
        const rec = await this.getRecordById(recId, scope, scopeId);
        if (rec) candidates.push(rec);
      }
    } else {
      for (const tier of tiers) {
        const indexKey = `${TIER_INDEX_PREFIX}${tier}`;
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
      if (!allowedTiers.has(rec.tier)) return false;
      if (!allowedStates.has(rec.state)) return false;
      if (options.runId && rec.episodic?.runId !== options.runId) return false;
      if (
        options.triggerPattern &&
        rec.procedural &&
        !normalizeStr(rec.procedural.triggerPattern).includes(normalizeStr(options.triggerPattern))
      ) {
        return false;
      }
      if (!this.isAuthorized(rec, options.callerRoles)) return false;
      return true;
    });

    return options.topK ? filtered.slice(0, options.topK) : filtered;
  }

  /**
   * Purges and reconstructs derived vector indices from persisted source records
   * without mutating record identities, versions, or lineage.
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
      if (!key.startsWith(TIER_RECORD_PREFIX)) continue;
      const rec = await this.client.get<TieredMemoryRecord>(key, {
        scope,
        scopeId,
        metadata: this.metadata
      });
      if (!rec || rec.scope !== scope || rec.scopeId !== scopeId) continue;

      if (rec.state !== 'ACTIVE') {
        await this.client.deleteVector(key, { scope, scopeId, metadata: this.metadata });
        skippedCount += 1;
        continue;
      }

      let embedding: number[] | undefined;
      if (options?.embedFn) {
        embedding = await options.embedFn(rec.content);
      } else if (this.aiClient) {
        embedding = await this.aiClient.embed(rec.content, this.embeddingOptions);
      }

      if (embedding && embedding.length > 0) {
        await this.client.setVector(
          key,
          embedding,
          {
            recordId: rec.id,
            tier: rec.tier,
            scope: rec.scope,
            scopeId: rec.scopeId,
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

  async getRecordById(
    id: string,
    scope: MemoryScope = this.defaultScope,
    scopeId: string = this.defaultScopeId
  ): Promise<TieredMemoryRecord | undefined> {
    const rec = await this.client.get<TieredMemoryRecord>(this.recordKey(id), {
      scope,
      scopeId,
      metadata: this.metadata
    });
    if (!rec || rec.scope !== scope || rec.scopeId !== scopeId) {
      return undefined;
    }
    return rec;
  }

  private async persistRecord(rec: TieredMemoryRecord, embedding?: number[]) {
    const key = this.recordKey(rec.id);
    await this.client.set(key, rec, {
      scope: rec.scope,
      scopeId: rec.scopeId,
      metadata: this.metadata
    });

    const tierIndexKey = `${TIER_INDEX_PREFIX}${rec.tier}`;
    const existingIds =
      (await this.client.get<string[]>(tierIndexKey, {
        scope: rec.scope,
        scopeId: rec.scopeId,
        metadata: this.metadata
      })) ?? [];
    if (!existingIds.includes(rec.id)) {
      await this.client.set(tierIndexKey, [...existingIds, rec.id], {
        scope: rec.scope,
        scopeId: rec.scopeId,
        metadata: this.metadata
      });
    }

    if (rec.state === 'ACTIVE' && embedding && embedding.length > 0) {
      await this.client.setVector(
        key,
        embedding,
        {
          recordId: rec.id,
          tier: rec.tier,
          scope: rec.scope,
          scopeId: rec.scopeId,
          version: rec.version,
          sourceIdentity: rec.sourceIdentity
        },
        {
          scope: rec.scope,
          scopeId: rec.scopeId,
          metadata: this.metadata
        }
      );
    }
  }

  private isAuthorized(rec: TieredMemoryRecord, callerRoles?: string[]): boolean {
    if (!rec.aclRoles || rec.aclRoles.length === 0) return true;
    if (!callerRoles || callerRoles.length === 0) return false;
    const callerSet = new Set(callerRoles);
    return rec.aclRoles.some((r) => callerSet.has(r));
  }

  private recordKey(id: string): string {
    return `${TIER_RECORD_PREFIX}${id}`;
  }

  private idFromKey(key: string): string | undefined {
    return key.startsWith(TIER_RECORD_PREFIX) ? key.slice(TIER_RECORD_PREFIX.length) : undefined;
  }
}
