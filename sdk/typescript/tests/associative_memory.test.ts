import { describe, it, expect, beforeEach } from 'vitest';
import type { MemoryScope } from '../src/types/agent.js';
import type { MemoryBackend } from '../src/memory/MemoryBackend.js';
import type {
  MemoryRequestOptions,
  VectorSearchOptions,
  VectorSearchResult
} from '../src/memory/MemoryClient.js';
import { AssociativeMemory } from '../src/memory/AssociativeMemory.js';

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

describe('AssociativeMemory (Self-organising associative memory with bounded consolidation)', () => {
  let backend: InMemoryBackend;
  let amem: AssociativeMemory;

  beforeEach(() => {
    backend = new InMemoryBackend();
    amem = new AssociativeMemory({
      client: backend,
      defaultScope: 'session',
      defaultScopeId: 'tenant-1',
      maxEdgesPerNode: 2,
      minAssociationScore: 0.15
    });
  });

  it('keeps write path fast, links related incidents/decisions/lessons, and enforces maxEdgesPerNode bound', async () => {
    const ev1 = await amem.ingestEvent({
      category: 'incident',
      content: 'Redis connection pool exhausted under spike',
      tags: ['redis', 'pool', 'outage'],
      sourceIdentity: 'inc:1'
    });
    const ev2 = await amem.ingestEvent({
      category: 'decision',
      content: 'Cap Redis pool per worker and enable circuit breaker',
      tags: ['redis', 'pool', 'circuit-breaker'],
      sourceIdentity: 'dec:1'
    });
    const ev3 = await amem.ingestEvent({
      category: 'code_change',
      content: 'Updated redis pool max_clients in config.ts',
      tags: ['redis', 'pool', 'config'],
      sourceIdentity: 'pr:10'
    });
    const ev4 = await amem.ingestEvent({
      category: 'lesson',
      content: 'Always bound redis pool size before load test',
      tags: ['redis', 'pool', 'loadtest'],
      sourceIdentity: 'postmortem:1'
    });

    const stats = await amem.consolidateBounded({ maxEdgesPerNode: 2 });
    expect(stats.processedEvents).toBe(4);
    expect(stats.edgesPrunedByBound).toBeGreaterThan(0);

    const node1 = await amem.getNodeById(`node_${ev1.id}`);
    expect(node1?.links.length).toBeLessThanOrEqual(2);
    expect(node1?.evidenceEventIds).toContain(ev1.id);

    // Multi-hop associative recall retrieves connected lessons/decisions
    const recalled = await amem.recallAssociative({
      categories: ['incident', 'decision', 'code_change', 'lesson'],
      tags: ['outage'],
      expandAssociativeHops: 1
    });
    expect(recalled.length).toBeGreaterThan(1);
    expect([ev2.id, ev3.id, ev4.id].some((id) => recalled.some((n) => n.eventId === id))).toBe(true);
  });
});
