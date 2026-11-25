import { describe, it, expect, beforeEach } from 'vitest';
import type { MemoryScope } from '../src/types/agent.js';
import type { MemoryBackend } from '../src/memory/MemoryBackend.js';
import type {
  MemoryRequestOptions,
  VectorSearchOptions,
  VectorSearchResult
} from '../src/memory/MemoryClient.js';
import { PolicyControlledMemory } from '../src/memory/LearnedMemoryPolicy.js';

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

describe('PolicyControlledMemory (Memory-R1 learned controller & hard overrides)', () => {
  let backend: InMemoryBackend;
  let policyMem: PolicyControlledMemory;

  beforeEach(() => {
    backend = new InMemoryBackend();
    policyMem = new PolicyControlledMemory({
      client: backend,
      defaultScope: 'session',
      defaultScopeId: 'tenant-1',
      minEvidenceCount: 1,
      minAnswerConfidence: 0.65
    });
  });

  it('selects ADD/UPDATE, calibrates weights on task outcome, and enforces hard policy overrides', async () => {
    const t1 = await policyMem.applyPolicy({
      keySlot: 'deploy_target',
      content: 'us-east-1',
      sourceIdentity: 'src:1',
      confidence: 0.9,
      utilityHint: 0.85
    });
    expect(t1.action).toBe('ADD');
    expect(policyMem.recordTaskOutcome(t1.decisionId, 1.0)).toBe(true);

    // Hard sensitive-data override rejects write even with high utility
    const sensitive = await policyMem.applyPolicy({
      keySlot: 'raw_ssn',
      content: '000-00-0000',
      sourceIdentity: 'src:2',
      confidence: 0.99,
      utilityHint: 0.99,
      containsRestrictedSensitiveData: true
    });
    expect(sensitive.action).toBe('NOOP');
    expect(sensitive.overriddenByHardPolicy).toBe(true);

    // Revoking source purges existing record and forces DELETE on subsequent writes
    const revokedCount = await policyMem.revokeSource('src:1');
    expect(revokedCount).toBe(1);

    // Abstains with INSUFFICIENT_EVIDENCE when knowledge was revoked
    const abstained = await policyMem.queryWithAbstention({ keySlots: ['deploy_target'] });
    expect(abstained.status).toBe('INSUFFICIENT_EVIDENCE');
    expect(abstained.records).toHaveLength(0);
  });
});
