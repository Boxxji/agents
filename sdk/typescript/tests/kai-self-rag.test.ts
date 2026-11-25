// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, vi } from 'vitest';
import {
  AdaptiveRetrievePolicy,
  Agent,
  AIClient,
  assertIndexBindingMatch,
  BENCHMARK_INDEX_BINDING,
  BENCHMARK_INDEX_CORPUS,
  CalibratedKaiHead,
  CalibrationConstraintError,
  type CalibrationSample,
  createImmutableIndexBinding,
  evaluateSelfRagPolicy,
  formatPrefixedQuery,
  GuardedIndexRetriever,
  IndexContractMismatchError,
  InMemoryHanzoIndex,
  KaiArchitectureViolationError,
  type KaiDecisionHead,
  type KaiRawDecision,
  KaiRetrieveOrAnswerGate,
  KaiValidationError,
  normalizeSimplex,
  type RetrievalTenantContext,
  selectOptimalTheta,
  softmaxToSimplex,
  sweepThresholds,
  TenantIsolationError,
  validateKaiDecisionOutput,
  type ZenSynthesisInput,
  type ZenSynthesisOutput,
  type ZenSynthesizer,
} from '../src/index.js';

const TENANT_ALPHA: RetrievalTenantContext = {
  tenantId: 'tenant-alpha',
  principalId: 'user-alpha-dev',
  principalRoles: ['developer'],
};

const TENANT_ALPHA_ADMIN: RetrievalTenantContext = {
  tenantId: 'tenant-alpha',
  principalId: 'user-alpha-cfo',
  principalRoles: ['billing-admin', 'developer'],
};

const TENANT_BETA: RetrievalTenantContext = {
  tenantId: 'tenant-beta',
  principalId: 'user-beta-dev',
  principalRoles: ['developer'],
};

function createTestZen(delayMs = 0): ZenSynthesizer & { calls: ZenSynthesisInput[] } {
  const calls: ZenSynthesisInput[] = [];
  return {
    modelId: 'zen3-vl-test',
    calls,
    async synthesize(input: ZenSynthesisInput): Promise<ZenSynthesisOutput> {
      if (delayMs > 0) {
        await new Promise((resolve) => setTimeout(resolve, delayMs));
      }
      calls.push(input);
      if (input.mode === 'grounded_rag') {
        return {
          answer: `Grounded[${input.tenantContext.tenantId}]: ${input.evidence.map((e) => e.content).join(' | ')}`,
          citations: input.evidence.map((e) => e.chunkId),
          modelId: 'zen3-vl-test',
          tokensIn: 900,
          tokensOut: 80,
          latencyMs: 140,
          costUsd: 0.001272,
        };
      }
      return {
        answer: `Direct[${input.tenantContext.tenantId}]: ${input.query}`,
        citations: [],
        modelId: 'zen3-vl-test',
        tokensIn: 120,
        tokensOut: 60,
        latencyMs: 85,
        costUsd: 0.000288,
      };
    },
  };
}

describe('HIP-1347 Kai Adaptive Retrieve-or-Answer Policy (Self-RAG)', () => {
  describe('Root SDK exports and Kai/Zen/Index architectural separation', () => {
    it('preserves existing @hanzo/agents exports alongside retrieval primitives', () => {
      expect(typeof Agent).toBe('function');
      expect(typeof AIClient).toBe('function');
      expect(typeof KaiRetrieveOrAnswerGate).toBe('function');
      expect(typeof AdaptiveRetrievePolicy).toBe('function');
      expect(typeof evaluateSelfRagPolicy).toBe('function');
    });

    it('enforces outputTokens === 0 on Kai and rejects generative payloads on Kai head', async () => {
      const binding = createImmutableIndexBinding({
        indexId: 'idx-arch',
        modelId: 'hanzo-embed-v1',
        modelRevision: 'rev-1',
        queryPrefix: 'query: ',
        docPrefix: 'passage: ',
      });
      const index = new InMemoryHanzoIndex({ binding });
      const zen = createTestZen();

      const leakyKaiHead: KaiDecisionHead = {
        modelId: 'misconfigured-generative-kai',
        decide: () => ({
          probabilities: { retrieve: 0.2, answerDirect: 0.7, clarifyOrAbstain: 0.1 },
          outputTokens: 14,
        }),
      };

      const gate = new KaiRetrieveOrAnswerGate({
        kaiHead: leakyKaiHead,
        retriever: index,
        zen,
      });

      await expect(
        gate.execute({ query: 'What is HTTP 429?', tenantContext: TENANT_ALPHA })
      ).rejects.toBeInstanceOf(KaiArchitectureViolationError);

      expect(() =>
        validateKaiDecisionOutput({
          probabilities: { retrieve: 0.2, answerDirect: 0.7, clarifyOrAbstain: 0.1 },
          outputTokens: 0,
          generatedText: 'Inline free-form prose should never come from Kai',
        } as unknown as KaiRawDecision)
      ).toThrow(KaiArchitectureViolationError);
    });

    it('skips index I/O on answer_direct and skips both index and Zen on clarify_or_abstain', async () => {
      const binding = createImmutableIndexBinding({
        indexId: 'idx-routing',
        modelId: 'hanzo-embed-v1',
        modelRevision: 'rev-1',
        queryPrefix: 'search_query: ',
        docPrefix: 'search_document: ',
      });
      const index = new InMemoryHanzoIndex({ binding });
      const retrieveSpy = vi.spyOn(index, 'retrieve');
      const zen = createTestZen();

      const gate = new KaiRetrieveOrAnswerGate({
        kaiHead: new CalibratedKaiHead(),
        retriever: index,
        zen,
        policy: { theta: 0.5 },
      });

      const directRes = await gate.execute({
        query: 'What is the worst-case time complexity of binary search on a sorted array?',
        tenantContext: TENANT_ALPHA,
      });
      expect(directRes.action).toBe('answer_direct');
      expect(directRes.kai.outputTokens).toBe(0);
      expect(directRes.grounded).toBe(false);
      expect(directRes.abstained).toBe(false);
      expect(retrieveSpy).toHaveBeenCalledTimes(0);
      expect(zen.calls.length).toBe(1);
      expect(zen.calls[0].mode).toBe('parametric_direct');

      const clarifyRes = await gate.execute({
        query: 'Roll back that deployment from earlier immediately.',
        tenantContext: TENANT_ALPHA,
      });
      expect(clarifyRes.action).toBe('clarify_or_abstain');
      expect(clarifyRes.decisionReason).toBe('clarify_safety_bound');
      expect(clarifyRes.abstained).toBe(true);
      expect(retrieveSpy).toHaveBeenCalledTimes(0);
      expect(zen.calls.length).toBe(1);
    });

    it('rejects Zen forging citations when synthesis mode is parametric_direct', async () => {
      const binding = createImmutableIndexBinding({
        indexId: 'idx-cite-guard',
        modelId: 'hanzo-embed-v1',
        modelRevision: 'rev-1',
        queryPrefix: 'search_query: ',
        docPrefix: 'search_document: ',
      });
      const index = new InMemoryHanzoIndex({ binding });
      const badZen: ZenSynthesizer = {
        modelId: 'zen-forged',
        synthesize: () => ({
          answer: 'Hallucinated answer',
          citations: ['forged-chunk-id'],
          modelId: 'zen-forged',
          tokensIn: 50,
          tokensOut: 20,
          latencyMs: 30,
          costUsd: 0.0001,
        }),
      };

      const gate = new KaiRetrieveOrAnswerGate({
        kaiHead: new CalibratedKaiHead(),
        retriever: index,
        zen: badZen,
      });

      await expect(
        gate.execute({
          query: 'How many bytes are in a standard SHA-256 digest?',
          tenantContext: TENANT_ALPHA,
        })
      ).rejects.toBeInstanceOf(KaiArchitectureViolationError);
    });
  });

  describe('Probability simplex, NaN/Inf, and numerical stability validation', () => {
    it('rejects NaN, Infinity, negative probabilities, and unnormalized simplex sums', () => {
      expect(() =>
        validateKaiDecisionOutput({
          probabilities: { retrieve: Number.NaN, answerDirect: 0.5, clarifyOrAbstain: 0.5 },
        })
      ).toThrow(KaiValidationError);

      expect(() =>
        validateKaiDecisionOutput({
          probabilities: {
            retrieve: Number.POSITIVE_INFINITY,
            answerDirect: 0,
            clarifyOrAbstain: 0,
          },
        })
      ).toThrow(KaiValidationError);

      expect(() =>
        validateKaiDecisionOutput({
          probabilities: { retrieve: -0.1, answerDirect: 0.9, clarifyOrAbstain: 0.2 },
        })
      ).toThrow(KaiValidationError);

      expect(() =>
        validateKaiDecisionOutput({
          probabilities: { retrieve: 0.7, answerDirect: 0.6, clarifyOrAbstain: 0.1 },
        })
      ).toThrow(KaiValidationError);

      expect(() => normalizeSimplex({ retrieve: 0, answerDirect: 0, clarifyOrAbstain: 0 })).toThrow(
        KaiValidationError
      );
    });

    it('stabilizes extreme logits in softmaxToSimplex without overflow and validates temperature', () => {
      const probs = softmaxToSimplex({
        retrieve: 1000,
        answerDirect: 999,
        clarifyOrAbstain: -1000,
      });
      expect(Number.isFinite(probs.retrieve)).toBe(true);
      expect(probs.retrieve).toBeCloseTo(Math.E / (Math.E + 1), 6);
      expect(probs.answerDirect).toBeCloseTo(1 / (Math.E + 1), 6);
      expect(probs.clarifyOrAbstain).toBeCloseTo(0, 8);
      expect(probs.retrieve + probs.answerDirect + probs.clarifyOrAbstain).toBeCloseTo(1, 9);

      expect(() =>
        softmaxToSimplex({ retrieve: 1, answerDirect: 1, clarifyOrAbstain: 1 }, 0)
      ).toThrow(KaiValidationError);
      expect(() =>
        softmaxToSimplex({ retrieve: 1, answerDirect: Number.NaN, clarifyOrAbstain: 1 }, 1)
      ).toThrow(KaiValidationError);
    });

    it('rejects out-of-range policy thresholds and non-finite risk signals', () => {
      expect(() => new AdaptiveRetrievePolicy({ theta: 1.2 })).toThrow(KaiValidationError);
      expect(() => new AdaptiveRetrievePolicy({ theta: -0.05 })).toThrow(KaiValidationError);
      expect(() => new AdaptiveRetrievePolicy({ topK: 0 })).toThrow(KaiValidationError);
      expect(() =>
        validateKaiDecisionOutput({
          probabilities: { retrieve: 0.4, answerDirect: 0.5, clarifyOrAbstain: 0.1 },
          riskSignals: { freshnessRisk: 1.5, citationRisk: 0, ambiguityRisk: 0 },
        })
      ).toThrow(KaiValidationError);
    });
  });

  describe('Immutable index contract and pre-hit tenant + role ACL filtering', () => {
    it('freezes ImmutableIndexBinding and rejects modelId, revision, or prefix drift', () => {
      const binding = createImmutableIndexBinding({
        indexId: 'idx-prod-1',
        modelId: 'hanzo-embed-v1',
        modelRevision: '2026.11-r2',
        queryPrefix: 'search_query: ',
        docPrefix: 'search_document: ',
        dimension: 4,
      });

      expect(Object.isFrozen(binding)).toBe(true);
      expect(() => {
        (binding as Record<string, unknown>).modelRevision = 'tampered';
      }).toThrow(TypeError);

      expect(() =>
        assertIndexBindingMatch(binding, {
          modelId: 'other-embed-v2',
          modelRevision: '2026.11-r2',
        })
      ).toThrow(IndexContractMismatchError);

      expect(() =>
        assertIndexBindingMatch(binding, {
          modelId: 'hanzo-embed-v1',
          modelRevision: 'stale-r1',
        })
      ).toThrow(IndexContractMismatchError);

      expect(() =>
        assertIndexBindingMatch(binding, {
          modelId: 'hanzo-embed-v1',
          modelRevision: '2026.11-r2',
          queryPrefix: 'wrong_prefix: ',
        })
      ).toThrow(IndexContractMismatchError);

      const index = new InMemoryHanzoIndex({ binding });
      expect(() =>
        index.upsert({
          chunkId: 'bad-model-doc',
          sourceUri: 'doc://bad',
          tenantId: 'tenant-alpha',
          allowedRoles: [],
          modelId: 'hanzo-embed-v1',
          modelRevision: 'wrong-revision',
          docPrefix: 'search_document: ',
          content: 'search_document: test body',
          updatedAtIso: '2026-11-17T00:00:00Z',
        })
      ).toThrow(IndexContractMismatchError);

      expect(() =>
        index.retrieve({
          rawQuery: 'test query',
          formattedQuery: 'missing_prefix: test query',
          topK: 2,
          binding,
          tenantContext: TENANT_ALPHA,
        })
      ).toThrow(IndexContractMismatchError);
    });

    it('filters cross-tenant and unauthorized ACL records BEFORE top-k ranking', async () => {
      const binding = createImmutableIndexBinding({
        indexId: 'idx-acl-test',
        modelId: 'hanzo-embed-v1',
        modelRevision: 'rev-1',
        queryPrefix: 'search_query: ',
        docPrefix: 'search_document: ',
      });

      const index = new InMemoryHanzoIndex({
        binding,
        records: [
          {
            chunkId: 'beta-exact-secret',
            sourceUri: 'vault://tenant-beta/invoice',
            tenantId: 'tenant-beta',
            allowedRoles: [],
            modelId: 'hanzo-embed-v1',
            modelRevision: 'rev-1',
            docPrefix: 'search_document: ',
            content: 'search_document: November invoice ledger balance secret is $999,999.',
            updatedAtIso: '2026-11-17T00:00:00Z',
            keywords: ['november', 'invoice', 'ledger', 'balance'],
          },
          {
            chunkId: 'alpha-admin-only',
            sourceUri: 'vault://tenant-alpha/cfo-ledger',
            tenantId: 'tenant-alpha',
            ownerId: 'user-alpha-cfo',
            allowedRoles: ['billing-admin'],
            modelId: 'hanzo-embed-v1',
            modelRevision: 'rev-1',
            docPrefix: 'search_document: ',
            content:
              'search_document: November invoice ledger restricted executive reserve $88,000.',
            updatedAtIso: '2026-11-17T00:00:00Z',
            keywords: ['november', 'invoice', 'ledger', 'balance'],
          },
          {
            chunkId: 'alpha-general-summary',
            sourceUri: 'docs://tenant-alpha/billing-tier',
            tenantId: 'tenant-alpha',
            allowedRoles: [],
            modelId: 'hanzo-embed-v1',
            modelRevision: 'rev-1',
            docPrefix: 'search_document: ',
            content: 'search_document: Standard billing invoice cycle closes on day 30.',
            updatedAtIso: '2026-11-17T00:00:00Z',
            keywords: ['invoice'],
          },
        ],
      });

      const guarded = new GuardedIndexRetriever(index, binding);
      const rawQuery = 'November invoice ledger balance';

      // Even with topK = 1, neither tenant-beta's exact match nor alpha's billing-admin chunk
      // may displace or leak to TENANT_ALPHA developer.
      const devResult = await guarded.retrieve({
        rawQuery,
        formattedQuery: formatPrefixedQuery(binding, rawQuery),
        topK: 1,
        binding,
        tenantContext: TENANT_ALPHA,
      });

      expect(devResult.preFilterAudit.candidatesScanned).toBe(3);
      expect(devResult.preFilterAudit.filteredOutByTenant).toBe(1);
      expect(devResult.preFilterAudit.filteredOutByAcl).toBe(1);
      expect(devResult.preFilterAudit.authorizedCandidates).toBe(1);
      expect(devResult.chunks.length).toBe(1);
      expect(devResult.chunks[0].chunkId).toBe('alpha-general-summary');
      expect(devResult.chunks[0].content).not.toContain('999,999');
      expect(devResult.chunks[0].content).not.toContain('88,000');

      // Authorized billing-admin within tenant-alpha sees alpha-admin-only, still zero beta leak.
      const adminResult = await guarded.retrieve({
        rawQuery,
        formattedQuery: formatPrefixedQuery(binding, rawQuery),
        topK: 2,
        binding,
        tenantContext: TENANT_ALPHA_ADMIN,
      });
      expect(adminResult.preFilterAudit.filteredOutByTenant).toBe(1);
      expect(adminResult.preFilterAudit.filteredOutByAcl).toBe(0);
      expect(adminResult.chunks[0].chunkId).toBe('alpha-admin-only');

      await expect(
        guarded.retrieve({
          rawQuery,
          formattedQuery: formatPrefixedQuery(binding, rawQuery),
          topK: 1,
          binding,
          tenantContext: { tenantId: '', principalId: 'u1', principalRoles: [] },
        })
      ).rejects.toBeInstanceOf(TenantIsolationError);
    });
  });

  describe('Freshness/citation risk floors and post-retrieval insufficient-evidence abstention', () => {
    it('forces retrieval via freshness and citation risk floors even when p(retrieve) < theta', () => {
      const policy = new AdaptiveRetrievePolicy({
        theta: 0.65,
        freshnessRiskFloor: 0.55,
        citationRiskFloor: 0.6,
      });

      const borderlineFresh = policy.evaluate({
        probabilities: { retrieve: 0.34, answerDirect: 0.6, clarifyOrAbstain: 0.06 },
        riskSignals: { freshnessRisk: 0.7, citationRisk: 0.1, ambiguityRisk: 0.05 },
        outputTokens: 0,
      });
      expect(borderlineFresh.action).toBe('retrieve');
      expect(borderlineFresh.reason).toBe('freshness_risk_floor');

      const borderlineCite = policy.evaluate({
        probabilities: { retrieve: 0.28, answerDirect: 0.66, clarifyOrAbstain: 0.06 },
        riskSignals: { freshnessRisk: 0.2, citationRisk: 0.75, ambiguityRisk: 0.05 },
        outputTokens: 0,
      });
      expect(borderlineCite.action).toBe('retrieve');
      expect(borderlineCite.reason).toBe('citation_risk_floor');
    });

    it('abstains safely after retrieval when pre-hit ACL filtering leaves zero authorized chunks', async () => {
      const index = new InMemoryHanzoIndex({
        binding: BENCHMARK_INDEX_BINDING,
        records: BENCHMARK_INDEX_CORPUS,
      });
      const zen = createTestZen();
      const gate = new KaiRetrieveOrAnswerGate({
        kaiHead: new CalibratedKaiHead(),
        retriever: index,
        zen,
      });

      const res = await gate.execute({
        query: 'Show globex-sec November 2026 invoice balance from the billing ledger.',
        tenantContext: TENANT_ALPHA_ADMIN,
      });

      expect(res.action).toBe('clarify_or_abstain');
      expect(res.decisionReason).toBe('post_retrieval_insufficient_evidence');
      expect(res.abstained).toBe(true);
      expect(res.grounded).toBe(false);
      expect(res.indexResult?.preFilterAudit.filteredOutByTenant).toBeGreaterThan(0);
      expect(zen.calls.length).toBe(0);
      expect(res.responseText).not.toContain('91,400');
    });
  });

  describe('Threshold calibration (sweepThresholds and selectOptimalTheta)', () => {
    const samples: readonly CalibrationSample[] = [
      {
        sampleId: 's1',
        kaiDecision: {
          probabilities: { retrieve: 0.12, answerDirect: 0.85, clarifyOrAbstain: 0.03 },
          outputTokens: 0,
        },
        requiresRetrieval: false,
        directAnswerAccurate: true,
        retrievedAnswerAccurate: true,
      },
      {
        sampleId: 's2',
        kaiDecision: {
          probabilities: { retrieve: 0.28, answerDirect: 0.68, clarifyOrAbstain: 0.04 },
          outputTokens: 0,
        },
        requiresRetrieval: false,
        directAnswerAccurate: true,
        retrievedAnswerAccurate: false,
      },
      {
        sampleId: 's3',
        kaiDecision: {
          probabilities: { retrieve: 0.62, answerDirect: 0.34, clarifyOrAbstain: 0.04 },
          outputTokens: 0,
        },
        requiresRetrieval: true,
        directAnswerAccurate: false,
        retrievedAnswerAccurate: true,
      },
      {
        sampleId: 's4',
        kaiDecision: {
          probabilities: { retrieve: 0.84, answerDirect: 0.12, clarifyOrAbstain: 0.04 },
          outputTokens: 0,
        },
        requiresRetrieval: true,
        directAnswerAccurate: false,
        retrievedAnswerAccurate: true,
      },
    ];

    it('produces monotone non-increasing retrievalRate across theta and honors minFactualAccuracy', () => {
      const points = sweepThresholds(samples, {
        thresholds: [0.1, 0.2, 0.4, 0.55, 0.7, 0.9],
      });

      for (let i = 1; i < points.length; i += 1) {
        expect(points[i].retrievalRate).toBeLessThanOrEqual(points[i - 1].retrievalRate);
        expect(points[i].unnecessaryRetrievalsAvoidedPct).toBeGreaterThanOrEqual(
          points[i - 1].unnecessaryRetrievalsAvoidedPct
        );
      }

      // At theta = 0.4 or 0.55, both s1 and s2 avoid retrieval (100% accuracy) while s3 and s4 retrieve.
      // At theta = 0.7 or 0.9, cost is lower (s3 stops retrieving) ONLY by dropping factualAccuracy to 0.75.
      const selected = selectOptimalTheta(points, {
        minFactualAccuracy: 0.95,
        maxCostPer1kQueriesUsd: 1.8,
      });

      expect(selected.accuracyFloorSatisfied).toBe(true);
      expect(selected.budgetSatisfied).toBe(true);
      expect(selected.selected.factualAccuracy).toBe(1);
      expect(selected.theta).toBeGreaterThanOrEqual(0.4);
      expect(selected.theta).toBeLessThanOrEqual(0.55);

      expect(() =>
        selectOptimalTheta(points, {
          minFactualAccuracy: 0.99,
          maxCostPer1kQueriesUsd: 0.1,
          strict: true,
        })
      ).toThrow(CalibrationConstraintError);
    });
  });

  describe('Concurrency safety across parallel async executions', () => {
    it('isolates tenant contexts, citations, and actions across 36 concurrent Promise.all runs', async () => {
      const binding = createImmutableIndexBinding({
        indexId: 'idx-concurrent',
        modelId: 'hanzo-embed-v1',
        modelRevision: 'rev-c1',
        queryPrefix: 'search_query: ',
        docPrefix: 'search_document: ',
      });

      const baseIndex = new InMemoryHanzoIndex({
        binding,
        records: [
          {
            chunkId: 'alpha-invoice',
            sourceUri: 'billing://tenant-alpha/nov',
            tenantId: 'tenant-alpha',
            allowedRoles: [],
            modelId: 'hanzo-embed-v1',
            modelRevision: 'rev-c1',
            docPrefix: 'search_document: ',
            content: 'search_document: Alpha November invoice balance is $1,111.',
            updatedAtIso: '2026-11-17T00:00:00Z',
            keywords: ['november', 'invoice', 'balance'],
          },
          {
            chunkId: 'beta-invoice',
            sourceUri: 'billing://tenant-beta/nov',
            tenantId: 'tenant-beta',
            allowedRoles: [],
            modelId: 'hanzo-embed-v1',
            modelRevision: 'rev-c1',
            docPrefix: 'search_document: ',
            content: 'search_document: Beta November invoice balance is $2,222.',
            updatedAtIso: '2026-11-17T00:00:00Z',
            keywords: ['november', 'invoice', 'balance'],
          },
        ],
      });

      const jitteredRetriever = {
        binding,
        async retrieve(req: Parameters<typeof baseIndex.retrieve>[0]) {
          await new Promise((r) => setTimeout(r, (req.tenantContext.tenantId.length % 5) + 1));
          return baseIndex.retrieve(req);
        },
      };

      const gate = new KaiRetrieveOrAnswerGate({
        kaiHead: new CalibratedKaiHead(),
        retriever: jitteredRetriever,
        zen: createTestZen(2),
      });

      const tasks: Array<Promise<{ kind: string; res: Awaited<ReturnType<typeof gate.execute>> }>> =
        [];
      for (let i = 0; i < 12; i += 1) {
        tasks.push(
          gate
            .execute({
              query: 'Check current November invoice balance.',
              tenantContext: TENANT_ALPHA,
            })
            .then((res) => ({ kind: 'alpha', res }))
        );
        tasks.push(
          gate
            .execute({
              query: 'Check current November invoice balance.',
              tenantContext: TENANT_BETA,
            })
            .then((res) => ({ kind: 'beta', res }))
        );
        tasks.push(
          gate
            .execute({
              query: 'What is the worst-case time complexity of binary search?',
              tenantContext: TENANT_ALPHA,
            })
            .then((res) => ({ kind: 'direct', res }))
        );
      }

      const outcomes = await Promise.all(tasks);
      expect(outcomes.length).toBe(36);

      for (const { kind, res } of outcomes) {
        if (kind === 'alpha') {
          expect(res.action).toBe('retrieve');
          expect(res.citations).toEqual(['alpha-invoice']);
          expect(res.responseText).toContain('$1,111');
          expect(res.responseText).not.toContain('$2,222');
        } else if (kind === 'beta') {
          expect(res.action).toBe('retrieve');
          expect(res.citations).toEqual(['beta-invoice']);
          expect(res.responseText).toContain('$2,222');
          expect(res.responseText).not.toContain('$1,111');
        } else {
          expect(res.action).toBe('answer_direct');
          expect(res.citations).toEqual([]);
        }
      }
    });
  });

  describe('Matched-baseline benchmark and 100% volatile negative control', () => {
    it('reproducibly outperforms naive baselines on mixed traffic and retrieves ~100% on volatile negative control', async () => {
      const reportA = await evaluateSelfRagPolicy();
      const reportB = await evaluateSelfRagPolicy();

      expect(reportA).toEqual(reportB);

      const { always_retrieve, never_retrieve, lexical_heuristic, kai_adaptive_gate } =
        reportA.policies;

      expect(always_retrieve.retrievalRate).toBe(1);
      expect(always_retrieve.unnecessaryRetrievalsAvoidedPct).toBe(0);
      expect(never_retrieve.retrievalRate).toBe(0);
      expect(never_retrieve.groundingRate).toBe(0);

      // Kai avoids >= 80% of unnecessary retrievals while achieving highest factual accuracy
      // and >= 45% lower dollar cost per 1,000 queries vs always_retrieve.
      expect(kai_adaptive_gate.unnecessaryRetrievalsAvoidedPct).toBeGreaterThanOrEqual(80);
      expect(kai_adaptive_gate.groundingRate).toBe(1);
      expect(kai_adaptive_gate.factualAccuracy).toBeGreaterThan(always_retrieve.factualAccuracy);
      expect(kai_adaptive_gate.factualAccuracy).toBeGreaterThan(never_retrieve.factualAccuracy);
      expect(kai_adaptive_gate.factualAccuracy).toBeGreaterThan(lexical_heuristic.factualAccuracy);
      expect(kai_adaptive_gate.latencyMs.p50).toBeLessThan(always_retrieve.latencyMs.p50);
      expect(reportA.deltasVsAlwaysRetrieve.costSavingsPct).toBeGreaterThanOrEqual(45);

      // Negative control: 100% volatile queries legitimately require ~100% retrieval.
      const ncKai = reportA.negativeControl.kai_adaptive_gate;
      const ncAlways = reportA.negativeControl.always_retrieve;
      const ncNever = reportA.negativeControl.never_retrieve;
      const ncLexical = reportA.negativeControl.lexical_heuristic;

      expect(ncKai.retrievalRate).toBeGreaterThanOrEqual(0.95);
      expect(ncKai.retrievalRate).toBe(1);
      expect(ncKai.groundingRate).toBe(1);
      expect(ncKai.factualAccuracy).toBe(ncAlways.factualAccuracy);
      expect(ncKai.factualAccuracy).toBeGreaterThan(ncLexical.factualAccuracy);
      expect(ncNever.factualAccuracy).toBe(0);
    });
  });
});
