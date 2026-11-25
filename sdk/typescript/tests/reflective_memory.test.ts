import { describe, it, expect, beforeEach } from 'vitest';
import type { MemoryScope } from '../src/types/agent.js';
import type { MemoryBackend } from '../src/memory/MemoryBackend.js';
import type {
  MemoryRequestOptions,
  VectorSearchOptions,
  VectorSearchResult
} from '../src/memory/MemoryClient.js';
import { ReflectiveMemory } from '../src/memory/ReflectiveMemory.js';

class InMemoryBackend implements MemoryBackend {
  public kv = new Map<string, any>();
  public vectors = new Map<string, any>();
  private keyOf(k: string, o?: MemoryRequestOptions) {
    return `${o?.scope ?? 'workflow'}:${o?.scopeId ?? 'default'}:${k}`;
  }
  async set(k: string, d: any, o?: MemoryRequestOptions) {
    this.kv.set(this.keyOf(k, o), JSON.parse(JSON.stringify(d)));
  }
  async get<T = any>(k: string, o?: MemoryRequestOptions): Promise<T | undefined> {
    const v = this.kv.get(this.keyOf(k, o));
    return v === undefined ? undefined : (JSON.parse(JSON.stringify(v)) as T);
  }
  async delete(k: string, o?: MemoryRequestOptions) {
    this.kv.delete(this.keyOf(k, o));
  }
  async listKeys(s: MemoryScope, o?: MemoryRequestOptions): Promise<string[]> {
    const p = `${s}:${o?.scopeId ?? 'default'}:`;
    return [...this.kv.keys()].filter((k) => k.startsWith(p)).map((k) => k.slice(p.length));
  }
  async exists(k: string, o?: MemoryRequestOptions) {
    return this.kv.has(this.keyOf(k, o));
  }
  async setVector(k: string, e: number[], m?: any, o?: MemoryRequestOptions) {
    const s = o?.scope ?? 'workflow';
    const sid = o?.scopeId ?? 'default';
    this.vectors.set(`${s}:${sid}:${k}`, { key: k, scope: s, scopeId: sid, embedding: e, metadata: m });
  }
  async deleteVector(k: string, o?: MemoryRequestOptions) {
    this.vectors.delete(this.keyOf(k, o));
  }
  async searchVector(_q: number[], o?: VectorSearchOptions): Promise<VectorSearchResult[]> {
    const s = o?.scope ?? 'workflow';
    const sid = o?.scopeId ?? 'default';
    return [...this.vectors.values()]
      .filter((v) => v.scope === s && v.scopeId === sid)
      .map((v) => ({ key: v.key, scope: v.scope, scopeId: v.scopeId, score: 0.99, metadata: v.metadata }));
  }
}

describe('ReflectiveMemory (Retain, Recall, Reflect with belief separation)', () => {
  let backend: InMemoryBackend;
  let mem: ReflectiveMemory;

  beforeEach(() => {
    backend = new InMemoryBackend();
    mem = new ReflectiveMemory({
      client: backend,
      defaultScope: 'session',
      defaultScopeId: 'tenant-1'
    });
  });

  it('separates observed evidence from hypotheses and rejects unsupported conclusions', async () => {
    const obs = await mem.retain({
      topic: 'ci-build',
      statement: 'the regression test failed at allocator.c:142',
      sourceIdentity: 'ci:run-88'
    });

    await expect(
      mem.reflect({
        kind: 'HYPOTHESIS',
        topic: 'ci-build',
        statement: 'we suspect an allocator alias',
        supportingObservationIds: [],
        sourceIdentity: 'agent:reflect-1'
      })
    ).rejects.toThrow(/Unsupported belief rejected/);

    const hyp = await mem.reflect({
      kind: 'HYPOTHESIS',
      topic: 'ci-build',
      statement: 'we suspect an allocator alias',
      supportingObservationIds: [obs.id],
      sourceIdentity: 'agent:reflect-1',
      confidence: 0.75
    });

    expect(hyp.kind).toBe('HYPOTHESIS');
    expect(hyp.supportingObservationIds).toEqual([obs.id]);

    // Contradicting observation refutes the hypothesis without altering the original observation
    const obs2 = await mem.retain({
      topic: 'ci-build',
      statement: 'allocator alias check passed under ASan; failure caused by stale fixture',
      sourceIdentity: 'ci:run-89'
    });

    const challenged = await mem.challengeBelief(hyp.id, obs2.id);
    expect(challenged.status).toBe('REFUTED');

    // Original observation remains untouched and ACTIVE
    const activeObservations = await mem.recall({ kinds: ['OBSERVATION'], topic: 'ci-build' });
    expect(activeObservations).toHaveLength(2);
    expect(activeObservations[0].statement).toBe('the regression test failed at allocator.c:142');
  });
});
