// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

import {
  type EvidenceChunk,
  type ImmutableIndexBinding,
  IndexContractMismatchError,
  type IndexedDocumentRecord,
  type IndexPreFilterAudit,
  type IndexRetriever,
  type IndexSearchRequest,
  type IndexSearchResult,
  type RetrievalTenantContext,
  TenantIsolationError,
} from './types.js';

/**
 * Creates a deeply frozen `ImmutableIndexBinding` after validating non-empty provenance fields.
 */
export function createImmutableIndexBinding(spec: ImmutableIndexBinding): ImmutableIndexBinding {
  if (!spec || typeof spec !== 'object') {
    throw new IndexContractMismatchError('Index binding specification must be an object.');
  }
  const requiredFields: Array<keyof ImmutableIndexBinding> = [
    'indexId',
    'modelId',
    'modelRevision',
    'queryPrefix',
    'docPrefix',
  ];
  for (const field of requiredFields) {
    const value = spec[field];
    if (typeof value !== 'string' || value.trim().length === 0) {
      throw new IndexContractMismatchError(
        `Index binding field "${String(field)}" must be a non-empty string.`
      );
    }
  }
  if (spec.dimension !== undefined && (!Number.isInteger(spec.dimension) || spec.dimension <= 0)) {
    throw new IndexContractMismatchError(
      `Index binding dimension must be a positive integer, got ${String(spec.dimension)}.`
    );
  }

  return Object.freeze({
    indexId: spec.indexId,
    modelId: spec.modelId,
    modelRevision: spec.modelRevision,
    queryPrefix: spec.queryPrefix,
    docPrefix: spec.docPrefix,
    ...(spec.dimension !== undefined ? { dimension: spec.dimension } : {}),
  });
}

/**
 * Verifies that an incoming search binding or document provenance matches the expected immutable contract.
 */
export function assertIndexBindingMatch(
  expected: ImmutableIndexBinding,
  actual: Partial<ImmutableIndexBinding>,
  label = 'index request'
): void {
  if (actual.indexId !== undefined && actual.indexId !== expected.indexId) {
    throw new IndexContractMismatchError(
      `Index ID mismatch on ${label}: expected "${expected.indexId}", got "${actual.indexId}".`
    );
  }
  if (actual.modelId !== expected.modelId) {
    throw new IndexContractMismatchError(
      `Embedding modelId mismatch on ${label}: expected "${expected.modelId}", got "${String(actual.modelId)}".`
    );
  }
  if (actual.modelRevision !== expected.modelRevision) {
    throw new IndexContractMismatchError(
      `Embedding modelRevision mismatch on ${label}: expected "${expected.modelRevision}", got "${String(actual.modelRevision)}".`
    );
  }
  if (actual.queryPrefix !== undefined && actual.queryPrefix !== expected.queryPrefix) {
    throw new IndexContractMismatchError(
      `Embedding queryPrefix mismatch on ${label}: expected "${expected.queryPrefix}", got "${actual.queryPrefix}".`
    );
  }
  if (actual.docPrefix !== undefined && actual.docPrefix !== expected.docPrefix) {
    throw new IndexContractMismatchError(
      `Embedding docPrefix mismatch on ${label}: expected "${expected.docPrefix}", got "${actual.docPrefix}".`
    );
  }
  if (
    expected.dimension !== undefined &&
    actual.dimension !== undefined &&
    actual.dimension !== expected.dimension
  ) {
    throw new IndexContractMismatchError(
      `Embedding dimension mismatch on ${label}: expected ${expected.dimension}, got ${actual.dimension}.`
    );
  }
}

/**
 * Validates that tenant and principal identity are present before touching index state.
 */
export function validateTenantContext(tenantContext: RetrievalTenantContext): void {
  if (!tenantContext || typeof tenantContext !== 'object') {
    throw new TenantIsolationError('RetrievalTenantContext is required for all index operations.');
  }
  if (typeof tenantContext.tenantId !== 'string' || tenantContext.tenantId.trim().length === 0) {
    throw new TenantIsolationError('RetrievalTenantContext.tenantId must be a non-empty string.');
  }
  if (
    typeof tenantContext.principalId !== 'string' ||
    tenantContext.principalId.trim().length === 0
  ) {
    throw new TenantIsolationError(
      'RetrievalTenantContext.principalId must be a non-empty string.'
    );
  }
  if (!Array.isArray(tenantContext.principalRoles)) {
    throw new TenantIsolationError('RetrievalTenantContext.principalRoles must be an array.');
  }
  for (const role of tenantContext.principalRoles) {
    if (typeof role !== 'string' || role.trim().length === 0) {
      throw new TenantIsolationError('Every principal role must be a non-empty string.');
    }
  }
}

/**
 * Applies the immutable `queryPrefix` to a raw query string for `hanzoai/index`.
 */
export function formatPrefixedQuery(binding: ImmutableIndexBinding, rawQuery: string): string {
  if (typeof rawQuery !== 'string' || rawQuery.trim().length === 0) {
    throw new IndexContractMismatchError('Index search rawQuery must be a non-empty string.');
  }
  return `${binding.queryPrefix}${rawQuery.trim()}`;
}

export interface PreHitFilterOutcome {
  readonly authorizedRecords: readonly IndexedDocumentRecord[];
  readonly audit: IndexPreFilterAudit;
}

/**
 * Enforces immutable document binding verification and mandatory pre-hit `tenantId` + ACL filtering.
 *
 * Filtering executes strictly BEFORE similarity scoring or top-k truncation so cross-tenant o
 * unauthorized documents can never displace valid tenant hits or leak metadata via ranking side-effects.
 */
export function enforcePreHitTenantAndAclFilter(
  records: readonly IndexedDocumentRecord[],
  binding: ImmutableIndexBinding,
  tenantContext: RetrievalTenantContext
): PreHitFilterOutcome {
  validateTenantContext(tenantContext);

  const roleSet = new Set(tenantContext.principalRoles);
  const authorizedRecords: IndexedDocumentRecord[] = [];
  let filteredOutByTenant = 0;
  let filteredOutByAcl = 0;

  for (const record of records) {
    assertIndexBindingMatch(
      binding,
      {
        modelId: record.modelId,
        modelRevision: record.modelRevision,
        docPrefix: record.docPrefix,
        dimension: record.embedding ? record.embedding.length : undefined,
      },
      `indexed chunk "${record.chunkId}"`
    );

    if (!record.content.startsWith(binding.docPrefix)) {
      throw new IndexContractMismatchError(
        `Indexed chunk "${record.chunkId}" content is missing mandatory docPrefix "${binding.docPrefix}".`
      );
    }

    if (record.tenantId !== tenantContext.tenantId) {
      filteredOutByTenant += 1;
      continue;
    }

    if (record.allowedRoles.length > 0) {
      const isOwner = record.ownerId !== undefined && record.ownerId === tenantContext.principalId;
      const hasAllowedRole = record.allowedRoles.some((role) => roleSet.has(role));
      if (!isOwner && !hasAllowedRole) {
        filteredOutByAcl += 1;
        continue;
      }
    }

    authorizedRecords.push(record);
  }

  return {
    authorizedRecords,
    audit: Object.freeze({
      candidatesScanned: records.length,
      authorizedCandidates: authorizedRecords.length,
      filteredOutByTenant,
      filteredOutByAcl,
    }),
  };
}

const STOP_WORDS = new Set([
  'a',
  'an',
  'and',
  'are',
  'as',
  'at',
  'be',
  'by',
  'do',
  'for',
  'from',
  'how',
  'in',
  'is',
  'it',
  'of',
  'on',
  'or',
  'please',
  'show',
  'tell',
  'that',
  'the',
  'this',
  'to',
  'was',
  'what',
  'when',
  'where',
  'which',
  'who',
  'why',
  'with',
]);

function tokenizeTerms(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z0-9_.-]+/)
    .map((tok) => tok.replace(/^[._-]+|[._-]+$/g, ''))
    .filter((tok) => tok.length >= 2 && !STOP_WORDS.has(tok));
}

function cosineSimilarity(a: readonly number[], b: readonly number[]): number {
  if (a.length === 0 || a.length !== b.length) {
    return 0;
  }
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i += 1) {
    const va = a[i];
    const vb = b[i];
    if (!Number.isFinite(va) || !Number.isFinite(vb)) {
      throw new IndexContractMismatchError('Embedding vectors must contain only finite numbers.');
    }
    dot += va * vb;
    normA += va * va;
    normB += vb * vb;
  }
  if (normA <= 0 || normB <= 0) {
    return 0;
  }
  return Math.max(0, Math.min(1, dot / Math.sqrt(normA * normB)));
}

function stripPrefix(text: string, prefix: string): string {
  return text.startsWith(prefix) ? text.slice(prefix.length).trim() : text.trim();
}

export interface InMemoryHanzoIndexOptions {
  readonly binding: ImmutableIndexBinding;
  readonly records?: readonly IndexedDocumentRecord[];
  readonly simulatedLatencyMs?: number;
  readonly costPerQueryUsd?: number;
}

/**
 * Clean-room in-memory `hanzoai/index` retriever enforcing immutable model/prefix bindings
 * and mandatory pre-hit tenant + role ACL isolation before scoring.
 */
export class InMemoryHanzoIndex implements IndexRetriever {
  readonly binding: ImmutableIndexBinding;
  private readonly records: IndexedDocumentRecord[] = [];
  private readonly simulatedLatencyMs: number;
  private readonly costPerQueryUsd: number;

  constructor(options: InMemoryHanzoIndexOptions) {
    this.binding = createImmutableIndexBinding(options.binding);
    this.simulatedLatencyMs = options.simulatedLatencyMs ?? 48;
    this.costPerQueryUsd = options.costPerQueryUsd ?? 0.00035;

    if (options.records) {
      for (const rec of options.records) {
        this.upsert(rec);
      }
    }
  }

  upsert(record: IndexedDocumentRecord): void {
    if (!record.tenantId || record.tenantId.trim().length === 0) {
      throw new TenantIsolationError(
        `Indexed chunk "${record.chunkId}" must specify a non-empty tenantId.`
      );
    }
    assertIndexBindingMatch(
      this.binding,
      {
        modelId: record.modelId,
        modelRevision: record.modelRevision,
        docPrefix: record.docPrefix,
        dimension: record.embedding ? record.embedding.length : undefined,
      },
      `upsert chunk "${record.chunkId}"`
    );
    const prefixedContent = record.content.startsWith(this.binding.docPrefix)
      ? record.content
      : `${this.binding.docPrefix}${record.content}`;

    const frozenRecord: IndexedDocumentRecord = Object.freeze({
      ...record,
      allowedRoles: Object.freeze([...record.allowedRoles]),
      keywords: record.keywords ? Object.freeze([...record.keywords]) : undefined,
      embedding: record.embedding ? Object.freeze([...record.embedding]) : undefined,
      content: prefixedContent,
    });

    const existingIdx = this.records.findIndex((r) => r.chunkId === frozenRecord.chunkId);
    if (existingIdx >= 0) {
      this.records[existingIdx] = frozenRecord;
    } else {
      this.records.push(frozenRecord);
    }
  }

  retrieve(request: IndexSearchRequest): IndexSearchResult {
    validateTenantContext(request.tenantContext);
    assertIndexBindingMatch(this.binding, request.binding, 'search request');

    if (!request.formattedQuery.startsWith(this.binding.queryPrefix)) {
      throw new IndexContractMismatchError(
        `Search formattedQuery must start with queryPrefix "${this.binding.queryPrefix}".`
      );
    }
    if (
      request.queryEmbedding !== undefined &&
      this.binding.dimension !== undefined &&
      request.queryEmbedding.length !== this.binding.dimension
    ) {
      throw new IndexContractMismatchError(
        `Query embedding dimension mismatch: expected ${this.binding.dimension}, got ${request.queryEmbedding.length}.`
      );
    }

    const { authorizedRecords, audit } = enforcePreHitTenantAndAclFilter(
      this.records,
      this.binding,
      request.tenantContext
    );

    const queryTokens = tokenizeTerms(request.rawQuery);
    const queryTokenSet = new Set(queryTokens);
    const minScore = request.minRelevanceScore ?? 0;
    const topK = Math.max(1, Math.floor(request.topK));

    const scored: EvidenceChunk[] = [];
    for (const rec of authorizedRecords) {
      const cleanBody = stripPrefix(rec.content, this.binding.docPrefix);
      const docTokens = tokenizeTerms(
        `${cleanBody} ${rec.sourceUri} ${(rec.keywords ?? []).join(' ')}`
      );
      const docTokenSet = new Set(docTokens);

      let lexicalScore = 0;
      if (queryTokenSet.size > 0 && docTokenSet.size > 0) {
        let matched = 0;
        for (const qt of queryTokenSet) {
          if (docTokenSet.has(qt)) {
            matched += 1;
          } else {
            for (const dt of docTokenSet) {
              if ((dt.length >= 4 && dt.includes(qt)) || (qt.length >= 4 && qt.includes(dt))) {
                matched += 0.65;
                break;
              }
            }
          }
        }
        const recall = matched / queryTokenSet.size;
        const precision = matched / Math.max(4, Math.min(20, docTokenSet.size));
        lexicalScore = Math.min(1, 0.72 * recall + 0.28 * Math.min(1, precision * 2));
      }

      const vectorScore =
        request.queryEmbedding && rec.embedding
          ? cosineSimilarity(request.queryEmbedding, rec.embedding)
          : undefined;

      const combinedScore =
        vectorScore !== undefined ? 0.55 * vectorScore + 0.45 * lexicalScore : lexicalScore;

      if (combinedScore >= minScore && combinedScore > 0) {
        scored.push(
          Object.freeze({
            chunkId: rec.chunkId,
            sourceUri: rec.sourceUri,
            content: cleanBody,
            score: Number(combinedScore.toFixed(6)),
            updatedAtIso: rec.updatedAtIso,
          })
        );
      }
    }

    scored.sort((a, b) => {
      if (b.score !== a.score) {
        return b.score - a.score;
      }
      return a.chunkId.localeCompare(b.chunkId);
    });

    return {
      chunks: Object.freeze(scored.slice(0, topK)),
      preFilterAudit: audit,
      latencyMs: this.simulatedLatencyMs,
      costUsd: this.costPerQueryUsd,
    };
  }
}

/**
 * Guard wrapper around any external `IndexRetriever` enforcing immutable binding checks,
 * queryPrefix formatting verification, and tenant context validation before delegating I/O.
 */
export class GuardedIndexRetriever implements IndexRetriever {
  readonly binding: ImmutableIndexBinding;
  private readonly inner: IndexRetriever;

  constructor(inner: IndexRetriever, expectedBinding?: ImmutableIndexBinding) {
    const targetBinding = expectedBinding ?? inner.binding;
    this.binding = createImmutableIndexBinding(targetBinding);
    assertIndexBindingMatch(this.binding, inner.binding, 'wrapped retriever');
    this.inner = inner;
  }

  async retrieve(request: IndexSearchRequest): Promise<IndexSearchResult> {
    validateTenantContext(request.tenantContext);
    assertIndexBindingMatch(this.binding, request.binding, 'guarded search request');
    if (!request.formattedQuery.startsWith(this.binding.queryPrefix)) {
      throw new IndexContractMismatchError(
        `Guarded search formattedQuery must start with queryPrefix "${this.binding.queryPrefix}".`
      );
    }

    const result = await this.inner.retrieve({
      ...request,
      binding: this.binding,
    });

    return {
      chunks: Object.freeze([...result.chunks]),
      preFilterAudit: result.preFilterAudit,
      latencyMs: result.latencyMs,
      costUsd: result.costUsd,
    };
  }
}
