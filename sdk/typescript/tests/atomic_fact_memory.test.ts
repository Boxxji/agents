import { describe, it, expect, beforeEach } from 'vitest';
import type { MemoryScope } from '../src/types/agent.js';
import type { MemoryBackend } from '../src/memory/MemoryBackend.js';
import type {
  MemoryRequestOptions,
  VectorSearchOptions,
  VectorSearchResult
} from '../src/memory/MemoryClient.js';
import { MemoryInterface } from '../src/memory/MemoryInterface.js';
import { AtomicFactMemory } from '../src/memory/AtomicFactMemory.js';

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

class InMemoryBackend implements MemoryBackend {
  public kv = new Map<string, any>();
  public vectors = new Map<
    string,
    { key: string; scope: MemoryScope; scopeId: string; embedding: number[]; metadata?: any }
  >();

  private scopedKey(key: string, options?: MemoryRequestOptions): {
    fullKey: string;
    scope: MemoryScope;
    scopeId: string;
  } {
    const scope = options?.scope ?? 'workflow';
    const scopeId = options?.scopeId ?? 'default';
    return {
      fullKey: `${scope}:${scopeId}:${key}`,
      scope,
      scopeId
    };
  }

  async set(key: string, data: any, options?: MemoryRequestOptions): Promise<void> {
    const { fullKey } = this.scopedKey(key, options);
    this.kv.set(fullKey, JSON.parse(JSON.stringify(data)));
  }

  async get<T = any>(key: string, options?: MemoryRequestOptions): Promise<T | undefined> {
    const { fullKey } = this.scopedKey(key, options);
    const val = this.kv.get(fullKey);
    return val === undefined ? undefined : (JSON.parse(JSON.stringify(val)) as T);
  }

  async delete(key: string, options?: MemoryRequestOptions): Promise<void> {
    const { fullKey } = this.scopedKey(key, options);
    this.kv.delete(fullKey);
  }

  async listKeys(scope: MemoryScope, options?: MemoryRequestOptions): Promise<string[]> {
    const scopeId = options?.scopeId ?? 'default';
    const prefix = `${scope}:${scopeId}:`;
    const keys: string[] = [];
    for (const fullKey of this.kv.keys()) {
      if (fullKey.startsWith(prefix)) {
        keys.push(fullKey.slice(prefix.length));
      }
    }
    return keys;
  }

  async exists(key: string, options?: MemoryRequestOptions): Promise<boolean> {
    const { fullKey } = this.scopedKey(key, options);
    return this.kv.has(fullKey);
  }

  async setVector(
    key: string,
    embedding: number[],
    metadata?: any,
    options?: MemoryRequestOptions
  ): Promise<void> {
    const { fullKey, scope, scopeId } = this.scopedKey(key, options);
    this.vectors.set(fullKey, {
      key,
      scope,
      scopeId,
      embedding: [...embedding],
      metadata: metadata ? JSON.parse(JSON.stringify(metadata)) : undefined
    });
  }

  async deleteVector(key: string, options?: MemoryRequestOptions): Promise<void> {
    const { fullKey } = this.scopedKey(key, options);
    this.vectors.delete(fullKey);
  }

  async searchVector(
    queryEmbedding: number[],
    options?: VectorSearchOptions
  ): Promise<VectorSearchResult[]> {
    const scope = options?.scope ?? 'workflow';
    const scopeId = options?.scopeId ?? 'default';
    const topK = options?.topK ?? 10;
    const hits: VectorSearchResult[] = [];

    for (const entry of this.vectors.values()) {
      if (entry.scope !== scope || entry.scopeId !== scopeId) continue;
      const score = cosineSimilarity(queryEmbedding, entry.embedding);
      hits.push({
        key: entry.key,
        scope: entry.scope,
        scopeId: entry.scopeId,
        score,
        metadata: entry.metadata
      });
    }

    hits.sort((a, b) => b.score - a.score);
    return hits.slice(0, topK);
  }
}

describe('AtomicFactMemory (Mem0-style selective writes & reconciliation)', () => {
  let backend: InMemoryBackend;
  let memory: MemoryInterface;
  let store: AtomicFactMemory;

  beforeEach(() => {
    backend = new InMemoryBackend();
    memory = new MemoryInterface({
      client: backend,
      defaultScope: 'session',
      defaultScopeId: 'tenant-alpha',
      metadata: { sessionId: 'tenant-alpha' }
    });
    store = memory.facts({ reviewConfidenceThreshold: 0.5 });
  });

  it('ingests a novel fact (ADD) and enforces idempotency (NOOP) on duplicate events', async () => {
    const first = await store.ingestFact(
      {
        subject: 'User:42',
        predicate: 'prefers_language',
        object: 'TypeScript',
        sourceIdentity: 'commit:abc1234',
        confidence: 0.95
      },
      { embedding: [1, 0, 0] }
    );

    expect(first.action).toBe('ADD');
    expect(first.record?.version).toBe(1);
    expect(first.record?.lifecycle).toBe('ACTIVE');

    // Re-ingesting identical fact with whitespace/case variation -> NOOP
    const duplicate = await store.ingestFact(
      {
        subject: '  user:42 ',
        predicate: 'PREFERS_LANGUAGE',
        object: 'typescript',
        sourceIdentity: 'commit:def5678',
        confidence: 0.99
      },
      { embedding: [1, 0, 0] }
    );

    expect(duplicate.action).toBe('NOOP');
    expect(duplicate.targetFactId).toBe(first.record?.id);

    const metrics = store.getMetrics();
    expect(metrics.eventsProcessed).toBe(2);
    expect(metrics.addCount).toBe(1);
    expect(metrics.noopCount).toBe(1);
    expect(metrics.durableWrites).toBe(1);
    expect(metrics.writeAmplification).toBe(0.5);
  });

  it('reconciles contradictory updates (UPDATE) by superseding prior version and preserving lineage', async () => {
    const v1 = await store.ingestFact(
      {
        subject: 'Service:Auth',
        predicate: 'deployment_region',
        object: 'eu-west-1',
        sourceIdentity: 'event:100',
        confidence: 0.9
      },
      { embedding: [0.8, 0.2, 0] }
    );

    const v2 = await store.ingestFact(
      {
        subject: 'Service:Auth',
        predicate: 'deployment_region',
        object: 'us-east-1',
        sourceIdentity: 'event:101',
        confidence: 0.95
      },
      { embedding: [0.2, 0.8, 0] }
    );

    expect(v2.action).toBe('UPDATE');
    expect(v2.record?.version).toBe(2);
    expect(v2.record?.supersedesId).toBe(v1.record?.id);
    expect(v2.supersededRecord?.lifecycle).toBe('SUPERSEDED');

    // Active query returns only v2
    const activeFacts = await store.queryFacts({
      subject: 'Service:Auth',
      predicate: 'deployment_region'
    });
    expect(activeFacts).toHaveLength(1);
    expect(activeFacts[0].object).toBe('us-east-1');
    expect(activeFacts[0].version).toBe(2);

    // Historical query preserves both v1 (SUPERSEDED) and v2 (ACTIVE)
    const history = await store.queryFacts({
      subject: 'Service:Auth',
      predicate: 'deployment_region',
      includeLifecycle: ['ACTIVE', 'SUPERSEDED']
    });
    expect(history).toHaveLength(2);
  });

  it('routes low-confidence facts to PENDING_REVIEW without polluting active memory', async () => {
    const lowConf = await store.ingestFact({
      subject: 'Database:Primary',
      predicate: 'max_connections',
      object: '5000',
      sourceIdentity: 'unverified:note',
      confidence: 0.3
    });

    expect(lowConf.action).toBe('REVIEW');
    expect(lowConf.record?.lifecycle).toBe('PENDING_REVIEW');

    const active = await store.queryFacts({ subject: 'Database:Primary' });
    expect(active).toHaveLength(0);

    const reviewQueue = await store.queryFacts({
      subject: 'Database:Primary',
      includeLifecycle: ['PENDING_REVIEW']
    });
    expect(reviewQueue).toHaveLength(1);
  });

  it('enforces strict pre-retrieval tenant isolation and RBAC filtering', async () => {
    await store.ingestFact(
      {
        subject: 'TenantSecret',
        predicate: 'api_tier',
        object: 'enterprise',
        sourceIdentity: 'doc:sec-1',
        aclRoles: ['security-admin']
      },
      { scope: 'session', scopeId: 'tenant-alpha', embedding: [0, 1, 0] }
    );

    await store.ingestFact(
      {
        subject: 'TenantSecret',
        predicate: 'api_tier',
        object: 'free',
        sourceIdentity: 'doc:sec-2'
      },
      { scope: 'session', scopeId: 'tenant-beta', embedding: [0, 1, 0] }
    );

    // 1. Cross-tenant query from tenant-beta never sees tenant-alpha's fact
    const betaHits = await store.queryFacts({
      scope: 'session',
      scopeId: 'tenant-beta',
      callerRoles: ['security-admin']
    });
    expect(betaHits).toHaveLength(1);
    expect(betaHits[0].object).toBe('free');

    // 2. Unauthorized caller in tenant-alpha without 'security-admin' role gets 0 hits
    const unauthorizedAlpha = await store.queryFacts({
      scope: 'session',
      scopeId: 'tenant-alpha',
      callerRoles: ['guest']
    });
    expect(unauthorizedAlpha).toHaveLength(0);

    // 3. Authorized caller in tenant-alpha retrieves the record
    const authorizedAlpha = await store.queryFacts({
      scope: 'session',
      scopeId: 'tenant-alpha',
      callerRoles: ['security-admin']
    });
    expect(authorizedAlpha).toHaveLength(1);
    expect(authorizedAlpha[0].object).toBe('enterprise');
  });

  it('rebuilds derived vector indices from immutable source records without mutating identities', async () => {
    const added = await store.ingestFact(
      {
        subject: 'Agent:Router',
        predicate: 'default_model',
        object: 'claude-sonnet-4-6',
        sourceIdentity: 'config:v1'
      },
      { embedding: [0.5, 0.5, 0] }
    );

    const originalId = added.record!.id;

    // Purge derived vector index completely
    backend.vectors.clear();
    expect(backend.vectors.size).toBe(0);

    // Rebuild derived vector index from source records
    const rebuildStats = await store.rebuildDerivedIndex({
      embedFn: async () => [0.5, 0.5, 0]
    });

    expect(rebuildStats.rebuiltCount).toBe(1);
    expect(backend.vectors.size).toBe(1);

    // Vector lookup works identically and preserves exact source record identity
    const vectorHits = await store.queryFacts({
      queryEmbedding: [0.5, 0.5, 0],
      topK: 3
    });
    expect(vectorHits).toHaveLength(1);
    expect(vectorHits[0].id).toBe(originalId);
    expect(vectorHits[0].object).toBe('claude-sonnet-4-6');
  });
});
