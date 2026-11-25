import { describe, it, expect, beforeEach } from 'vitest';
import type { MemoryScope } from '../src/types/agent.js';
import type { MemoryBackend } from '../src/memory/MemoryBackend.js';
import type {
  MemoryRequestOptions,
  VectorSearchOptions,
  VectorSearchResult
} from '../src/memory/MemoryClient.js';
import { MemoryInterface } from '../src/memory/MemoryInterface.js';
import { TieredMemory } from '../src/memory/TieredMemory.js';

class InMemoryBackend implements MemoryBackend {
  public kv = new Map<string, any>();
  public vectors = new Map<
    string,
    { key: string; scope: MemoryScope; scopeId: string; embedding: number[]; metadata?: any }
  >();

  private scopedKey(key: string, options?: MemoryRequestOptions) {
    const scope = options?.scope ?? 'workflow';
    const scopeId = options?.scopeId ?? 'default';
    return { fullKey: `${scope}:${scopeId}:${key}`, scope, scopeId };
  }

  async set(key: string, data: any, options?: MemoryRequestOptions) {
    const { fullKey } = this.scopedKey(key, options);
    this.kv.set(fullKey, JSON.parse(JSON.stringify(data)));
  }

  async get<T = any>(key: string, options?: MemoryRequestOptions): Promise<T | undefined> {
    const { fullKey } = this.scopedKey(key, options);
    const val = this.kv.get(fullKey);
    return val === undefined ? undefined : (JSON.parse(JSON.stringify(val)) as T);
  }

  async delete(key: string, options?: MemoryRequestOptions) {
    const { fullKey } = this.scopedKey(key, options);
    this.kv.delete(fullKey);
  }

  async listKeys(scope: MemoryScope, options?: MemoryRequestOptions): Promise<string[]> {
    const scopeId = options?.scopeId ?? 'default';
    const prefix = `${scope}:${scopeId}:`;
    return [...this.kv.keys()].filter((k) => k.startsWith(prefix)).map((k) => k.slice(prefix.length));
  }

  async exists(key: string, options?: MemoryRequestOptions): Promise<boolean> {
    const { fullKey } = this.scopedKey(key, options);
    return this.kv.has(fullKey);
  }

  async setVector(key: string, embedding: number[], metadata?: any, options?: MemoryRequestOptions) {
    const { fullKey, scope, scopeId } = this.scopedKey(key, options);
    this.vectors.set(fullKey, {
      key,
      scope,
      scopeId,
      embedding: [...embedding],
      metadata: metadata ? JSON.parse(JSON.stringify(metadata)) : undefined
    });
  }

  async deleteVector(key: string, options?: MemoryRequestOptions) {
    const { fullKey } = this.scopedKey(key, options);
    this.vectors.delete(fullKey);
  }

  async searchVector(_queryEmbedding: number[], options?: VectorSearchOptions): Promise<VectorSearchResult[]> {
    const scope = options?.scope ?? 'workflow';
    const scopeId = options?.scopeId ?? 'default';
    const hits: VectorSearchResult[] = [];
    for (const entry of this.vectors.values()) {
      if (entry.scope !== scope || entry.scopeId !== scopeId) continue;
      hits.push({
        key: entry.key,
        scope: entry.scope,
        scopeId: entry.scopeId,
        score: 0.99,
        metadata: entry.metadata
      });
    }
    return hits.slice(0, options?.topK ?? 10);
  }
}

describe('TieredMemory (Episodic, Semantic, Procedural behind one API)', () => {
  let backend: InMemoryBackend;
  let memory: MemoryInterface;
  let tiered: TieredMemory;

  beforeEach(() => {
    backend = new InMemoryBackend();
    memory = new MemoryInterface({
      client: backend,
      defaultScope: 'session',
      defaultScopeId: 'tenant-1',
      metadata: { sessionId: 'tenant-1' }
    });
    tiered = memory.tiered();
  });

  it('stores and retrieves episodic, semantic, and procedural records behind one unified API', async () => {
    await tiered.recordEpisode({
      runId: 'run-100',
      stepIndex: 0,
      action: 'read_config',
      observation: 'loaded config.yaml',
      outcome: 'success',
      sourceIdentity: 'event:1'
    });

    await tiered.recordSemantic({
      topic: 'db-schema',
      statement: 'Postgres uses strict tenant_id RLS policies',
      sourceIdentity: 'doc:arch-1'
    });

    await tiered.recordProcedure({
      name: 'recover-db-timeout',
      triggerPattern: 'ETIMEDOUT postgres',
      steps: ['backoff 500ms', 'reconnect pool', 'retry query'],
      sourceIdentity: 'runbook:1'
    });

    const all = await tiered.retrieve();
    expect(all).toHaveLength(3);

    const onlyProcedural = await tiered.retrieve({ tiers: ['procedural'] });
    expect(onlyProcedural).toHaveLength(1);
    expect(onlyProcedural[0].procedural?.name).toBe('recover-db-timeout');
  });

  it('compacts episodic steps into a semantic summary with lineage and extracts reusable procedures', async () => {
    const ep1 = await tiered.recordEpisode({
      runId: 'run-200',
      stepIndex: 0,
      action: 'check_port',
      observation: 'port 8080 busy',
      outcome: 'observation',
      sourceIdentity: 'ev:200-0'
    });
    const ep2 = await tiered.recordEpisode({
      runId: 'run-200',
      stepIndex: 1,
      action: ' switch_port_8081 ',
      observation: 'server started on 8081',
      outcome: 'success',
      sourceIdentity: 'ev:200-1'
    });

    const proc = await tiered.extractProcedureFromRun({
      runId: 'run-200',
      name: 'handle-port-conflict',
      triggerPattern: 'port busy'
    });
    expect(proc.tier).toBe('procedural');
    expect(proc.lineageIds).toEqual([ep1.id, ep2.id]);

    const compaction = await tiered.compactEpisodic({
      runId: 'run-200',
      topic: 'port-conflict-resolution'
    });
    expect(compaction.compactedIds).toEqual([ep1.id, ep2.id]);
    expect(compaction.semanticRecord?.lineageIds).toEqual([ep1.id, ep2.id]);

    // Active episodic records are now 0 (compacted into semantic), reducing context tokens
    const activeEpisodic = await tiered.retrieve({ tiers: ['episodic'], runId: 'run-200' });
    expect(activeEpisodic).toHaveLength(0);

    // Underlying source records remain intact in persistent storage under COMPACTED state
    const allEpisodic = await tiered.retrieve({
      tiers: ['episodic'],
      runId: 'run-200',
      includeStates: ['COMPACTED']
    });
    expect(allEpisodic).toHaveLength(2);
  });

  it('recovers full run state and procedures across stateless process restarts', async () => {
    await tiered.recordEpisode({
      runId: 'run-300',
      stepIndex: 0,
      action: 'init_migration',
      observation: 'started schema v2',
      sourceIdentity: 'ev:300-0'
    });
    await tiered.recordProcedure({
      name: 'resume-migration',
      triggerPattern: 'schema v2',
      steps: ['verify lock table', 'apply pending DDL'],
      sourceIdentity: 'proc:300'
    });

    // Simulate process restart with a brand new TieredMemory instance over persistent backend
    const restartedTiered = new TieredMemory({
      client: backend,
      defaultScope: 'session',
      defaultScopeId: 'tenant-1'
    });

    const recovery = await restartedTiered.recoverRunState('run-300', {
      triggerPattern: 'schema v2'
    });
    expect(recovery.lastStepIndex).toBe(0);
    expect(recovery.steps).toHaveLength(1);
    expect(recovery.applicableProcedures).toHaveLength(1);
    expect(recovery.applicableProcedures[0].procedural?.name).toBe('resume-migration');
  });

  it('enforces strict tenant & RBAC pre-filtering and rebuilds derived vector indices', async () => {
    const rec = await tiered.recordSemantic(
      {
        topic: 'audit-policy',
        statement: 'Sign all ledger writes with Ed25519',
        sourceIdentity: 'sec:1',
        aclRoles: ['auditor'],
        embedding: [0.4, 0.6, 0]
      },
      { scope: 'session', scopeId: 'tenant-1' }
    );

    // Other tenant gets 0 hits
    expect(
      await tiered.retrieve({ scope: 'session', scopeId: 'tenant-2', callerRoles: ['auditor'] })
    ).toHaveLength(0);

    // Unauthorized role in tenant-1 gets 0 hits
    expect(
      await tiered.retrieve({ scope: 'session', scopeId: 'tenant-1', callerRoles: ['guest'] })
    ).toHaveLength(0);

    // Purge derived vector store and rebuild from immutable source records
    backend.vectors.clear();
    expect(backend.vectors.size).toBe(0);

    const stats = await tiered.rebuildDerivedIndex({
      embedFn: async () => [0.4, 0.6, 0]
    });
    expect(stats.rebuiltCount).toBe(1);

    const hits = await tiered.retrieve({
      queryEmbedding: [0.4, 0.6, 0],
      callerRoles: ['auditor']
    });
    expect(hits).toHaveLength(1);
    expect(hits[0].id).toBe(rec.id);
  });
});
