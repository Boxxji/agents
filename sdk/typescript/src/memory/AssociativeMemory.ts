import crypto from 'crypto';
import type { MemoryScope } from '../types/agent.js';
import type { MemoryBackend } from './MemoryBackend.js';
import type { MemoryRequestMetadata, VectorSearchResult } from './MemoryClient.js';
import type { AIClient, AIEmbeddingOptions } from '../ai/AIClient.js';

export type EventCategory = 'incident' | 'decision' | 'code_change' | 'lesson';

export interface RawMemoryEventInput {
  category: EventCategory;
  content: string;
  tags?: string[];
  contextDescription?: string;
  sourceIdentity: string;
  sourceVersion?: number;
  aclRoles?: string[];
  metadata?: Record<string, any>;
}

export interface ImmutableRawEvent {
  id: string;
  category: EventCategory;
  content: string;
  tags: string[];
  contextDescription?: string;
  sourceIdentity: string;
  sourceVersion: number;
  scope: MemoryScope;
  scopeId: string;
  aclRoles: string[];
  metadata?: Record<string, any>;
  createdAt: string;
}

export interface AssociativeLink {
  targetNodeId: string;
  relationType: string;
  weight: number;
  sharedTags: string[];
  evidenceEventIds: string[];
}

export interface AssociativeMemoryNode {
  id: string;
  eventId: string;
  category: EventCategory;
  content: string;
  contextDescription: string;
  tags: string[];
  links: AssociativeLink[];
  evidenceEventIds: string[];
  scope: MemoryScope;
  scopeId: string;
  aclRoles: string[];
  version: number;
  updatedAt: string;
}

export interface ConsolidationStats {
  processedEvents: number;
  nodesUpdated: number;
  edgesCreatedOrUpdated: number;
  edgesPrunedByBound: number;
}

export interface AssociativeRecallOptions {
  scope?: MemoryScope;
  scopeId?: string;
  callerRoles?: string[];
  categories?: EventCategory[];
  tags?: string[];
  queryEmbedding?: number[];
  queryText?: string;
  expandAssociativeHops?: number;
  topK?: number;
}

export interface AssociativeMemoryOptions {
  client: MemoryBackend;
  aiClient?: AIClient;
  defaultScope?: MemoryScope;
  defaultScopeId?: string;
  metadata?: MemoryRequestMetadata;
  maxEdgesPerNode?: number;
  minAssociationScore?: number;
  embeddingOptions?: AIEmbeddingOptions;
}

const RAW_EVENT_PREFIX = 'amem_raw_event:';
const RAW_EVENT_LIST_KEY = 'amem_raw_event_ids';
const PENDING_QUEUE_KEY = 'amem_pending_queue';
const ASSOC_NODE_PREFIX = 'amem_node:';
const ASSOC_NODE_LIST_KEY = 'amem_node_ids';

function norm(val: string): string {
  return val.trim().toLowerCase().replace(/\s+/g, ' ');
}

function extractAutoTags(content: string, existingTags: string[] = []): string[] {
  const tokens = content
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, ' ')
    .split(/\s+/)
    .filter((w) => w.length >= 4);
  const stop = new Set(['that', 'this', 'with', 'from', 'when', 'were', 'have', 'been']);
  const auto = tokens.filter((t) => !stop.has(t)).slice(0, 8);
  return Array.from(new Set([...existingTags.map(norm), ...auto]));
}

export function computeRawEventId(
  scope: MemoryScope,
  scopeId: string,
  category: EventCategory,
  sourceIdentity: string,
  content: string
): string {
  const canonical = `${scope}:${scopeId}:${category}:${norm(sourceIdentity)}:${norm(content)}`;
  return crypto.createHash('sha256').update(canonical, 'utf8').digest('hex').slice(0, 32);
}

/**
 * Clean-room Apache-2.0 Self-Organising Associative Memory with Bounded Consolidation.
 *
 * Guarantees:
 * - Fast write path (`ingestEvent`) stores immutable raw events and enqueues IDs
 *   without blocking on graph traversal.
 * - Asynchronous bounded consolidation (`consolidateBounded`) enriches derived
 *   associative nodes with contextual descriptions, tags, and bidirectional links
 *   while strictly enforcing `maxEdgesPerNode` degree bounds.
 * - Derived associative nodes and vector indices can be completely rebuilt from
 *   immutable raw events (`rebuildDerivedGraphAndIndex`).
 */
export class AssociativeMemory {
  private readonly client: MemoryBackend;
  private readonly aiClient?: AIClient;
  private readonly defaultScope: MemoryScope;
  private readonly defaultScopeId: string;
  private readonly metadata?: MemoryRequestMetadata;
  private readonly maxEdgesPerNode: number;
  private readonly minAssociationScore: number;
  private readonly embeddingOptions?: AIEmbeddingOptions;

  constructor(options: AssociativeMemoryOptions) {
    this.client = options.client;
    this.aiClient = options.aiClient;
    this.defaultScope = options.defaultScope ?? 'workflow';
    this.defaultScopeId = options.defaultScopeId ?? 'default';
    this.metadata = options.metadata;
    this.maxEdgesPerNode = options.maxEdgesPerNode ?? 6;
    this.minAssociationScore = options.minAssociationScore ?? 0.2;
    this.embeddingOptions = options.embeddingOptions;
  }

  /**
   * Fast, O(1) write path: persists an immutable raw event and appends its ID
   * to the bounded consolidation queue without blocking on graph linking.
   */
  async ingestEvent(
    input: RawMemoryEventInput,
    options?: { scope?: MemoryScope; scopeId?: string }
  ): Promise<ImmutableRawEvent> {
    const scope = options?.scope ?? this.defaultScope;
    const scopeId = options?.scopeId ?? this.defaultScopeId;
    const id = computeRawEventId(
      scope,
      scopeId,
      input.category,
      input.sourceIdentity,
      input.content
    );

    const existing = await this.getRawEventById(id, scope, scopeId);
    if (existing) {
      return existing;
    }

    const rawEvent: ImmutableRawEvent = {
      id,
      category: input.category,
      content: input.content.trim(),
      tags: extractAutoTags(input.content, input.tags),
      contextDescription: input.contextDescription,
      sourceIdentity: input.sourceIdentity,
      sourceVersion: input.sourceVersion ?? 1,
      scope,
      scopeId,
      aclRoles: input.aclRoles ? [...input.aclRoles] : [],
      metadata: input.metadata,
      createdAt: new Date().toISOString()
    };

    await this.client.set(`${RAW_EVENT_PREFIX}${id}`, rawEvent, {
      scope,
      scopeId,
      metadata: this.metadata
    });

    const eventIds =
      (await this.client.get<string[]>(RAW_EVENT_LIST_KEY, {
        scope,
        scopeId,
        metadata: this.metadata
      })) ?? [];
    if (!eventIds.includes(id)) {
      await this.client.set(RAW_EVENT_LIST_KEY, [...eventIds, id], {
        scope,
        scopeId,
        metadata: this.metadata
      });
    }

    const pendingQueue =
      (await this.client.get<string[]>(PENDING_QUEUE_KEY, {
        scope,
        scopeId,
        metadata: this.metadata
      })) ?? [];
    if (!pendingQueue.includes(id)) {
      await this.client.set(PENDING_QUEUE_KEY, [...pendingQueue, id], {
        scope,
        scopeId,
        metadata: this.metadata
      });
    }

    return rawEvent;
  }

  /**
   * Asynchronous bounded consolidation: processes up to `maxBatchSize` queued raw events,
   * constructs/updates derived associative nodes with contextual descriptions and tags,
   * links related incidents, decisions, code changes, and lessons, and enforces `maxEdgesPerNode`.
   */
  async consolidateBounded(options?: {
    scope?: MemoryScope;
    scopeId?: string;
    maxBatchSize?: number;
    maxEdgesPerNode?: number;
    embedFn?: (text: string) => Promise<number[]>;
  }): Promise<ConsolidationStats> {
    const scope = options?.scope ?? this.defaultScope;
    const scopeId = options?.scopeId ?? this.defaultScopeId;
    const maxBatch = options?.maxBatchSize ?? 25;
    const maxEdges = options?.maxEdgesPerNode ?? this.maxEdgesPerNode;

    const pendingQueue =
      (await this.client.get<string[]>(PENDING_QUEUE_KEY, {
        scope,
        scopeId,
        metadata: this.metadata
      })) ?? [];

    if (pendingQueue.length === 0) {
      return {
        processedEvents: 0,
        nodesUpdated: 0,
        edgesCreatedOrUpdated: 0,
        edgesPrunedByBound: 0
      };
    }

    const batchIds = pendingQueue.slice(0, maxBatch);
    const remainingIds = pendingQueue.slice(maxBatch);

    const existingNodeIds =
      (await this.client.get<string[]>(ASSOC_NODE_LIST_KEY, {
        scope,
        scopeId,
        metadata: this.metadata
      })) ?? [];

    const nodeMap = new Map<string, AssociativeMemoryNode>();
    for (const nid of existingNodeIds) {
      const n = await this.getNodeById(nid, scope, scopeId);
      if (n) nodeMap.set(n.id, n);
    }

    let nodesUpdated = 0;
    let edgesCreatedOrUpdated = 0;
    let edgesPrunedByBound = 0;
    const touchedNodeIds = new Set<string>();

    for (const eventId of batchIds) {
      const raw = await this.getRawEventById(eventId, scope, scopeId);
      if (!raw) continue;

      const nodeId = `node_${raw.id}`;
      const contextDesc =
        raw.contextDescription ??
        `[${raw.category.toUpperCase()}] ${raw.content} (tags: ${raw.tags.join(', ')})`;

      const newNode: AssociativeMemoryNode = nodeMap.get(nodeId) ?? {
        id: nodeId,
        eventId: raw.id,
        category: raw.category,
        content: raw.content,
        contextDescription: contextDesc,
        tags: [...raw.tags],
        links: [],
        evidenceEventIds: [raw.id],
        scope,
        scopeId,
        aclRoles: [...raw.aclRoles],
        version: 1,
        updatedAt: new Date().toISOString()
      };

      // Evaluate associations against existing nodes in the same tenant scope
      for (const other of nodeMap.values()) {
        if (other.id === newNode.id) continue;
        if (other.scope !== scope || other.scopeId !== scopeId) continue;

        const { score, sharedTags } = this.computeAssociationScore(newNode, other);
        if (score < this.minAssociationScore) continue;

        const relationType = `${newNode.category}<->${other.category}`;
        const combinedEvidence = Array.from(new Set([raw.id, other.eventId]));

        const prunedA = this.upsertBoundedLink(
          newNode,
          {
            targetNodeId: other.id,
            relationType,
            weight: score,
            sharedTags,
            evidenceEventIds: combinedEvidence
          },
          maxEdges
        );
        const prunedB = this.upsertBoundedLink(
          other,
          {
            targetNodeId: newNode.id,
            relationType,
            weight: score,
            sharedTags,
            evidenceEventIds: combinedEvidence
          },
          maxEdges
        );

        edgesCreatedOrUpdated += 2;
        edgesPrunedByBound += prunedA + prunedB;
        other.version += 1;
        other.updatedAt = new Date().toISOString();
        touchedNodeIds.add(other.id);
      }

      nodeMap.set(newNode.id, newNode);
      touchedNodeIds.add(newNode.id);
    }

    // Persist all touched associative nodes and update derived vector indices
    for (const nid of touchedNodeIds) {
      const node = nodeMap.get(nid)!;
      const key = `${ASSOC_NODE_PREFIX}${node.id}`;
      await this.client.set(key, node, { scope, scopeId, metadata: this.metadata });
      nodesUpdated += 1;

      let embedding: number[] | undefined;
      if (options?.embedFn) {
        embedding = await options.embedFn(node.contextDescription);
      } else if (this.aiClient) {
        embedding = await this.aiClient.embed(node.contextDescription, this.embeddingOptions);
      }
      if (embedding && embedding.length > 0) {
        await this.client.setVector(
          key,
          embedding,
          {
            nodeId: node.id,
            eventId: node.eventId,
            category: node.category,
            scope,
            scopeId
          },
          { scope, scopeId, metadata: this.metadata }
        );
      }
    }

    await this.client.set(ASSOC_NODE_LIST_KEY, Array.from(nodeMap.keys()), {
      scope,
      scopeId,
      metadata: this.metadata
    });
    await this.client.set(PENDING_QUEUE_KEY, remainingIds, {
      scope,
      scopeId,
      metadata: this.metadata
    });

    return {
      processedEvents: batchIds.length,
      nodesUpdated,
      edgesCreatedOrUpdated,
      edgesPrunedByBound
    };
  }

  /**
   * Associative recall with optional multi-hop link expansion and strict
   * pre-retrieval tenant & RBAC filtering on every node and hop.
   */
  async recallAssociative(
    options: AssociativeRecallOptions = {}
  ): Promise<AssociativeMemoryNode[]> {
    const scope = options.scope ?? this.defaultScope;
    const scopeId = options.scopeId ?? this.defaultScopeId;
    const allowedCategories = options.categories ? new Set(options.categories) : undefined;
    const requiredTags = options.tags ? options.tags.map(norm) : [];

    let seedNodes: AssociativeMemoryNode[] = [];

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
        const nid = hit.metadata?.nodeId ?? this.nodeIdFromKey(hit.key);
        if (!nid) continue;
        const n = await this.getNodeById(nid, scope, scopeId);
        if (n) seedNodes.push(n);
      }
    } else {
      const nodeIds =
        (await this.client.get<string[]>(ASSOC_NODE_LIST_KEY, {
          scope,
          scopeId,
          metadata: this.metadata
        })) ?? [];
      for (const nid of nodeIds) {
        const n = await this.getNodeById(nid, scope, scopeId);
        if (n) seedNodes.push(n);
      }
    }

    // Apply pre-retrieval tenant, RBAC, category, and tag filters to seed nodes
    const authorizedSeeds = seedNodes.filter((n) => {
      if (n.scope !== scope || n.scopeId !== scopeId) return false;
      if (!this.isAuthorized(n.aclRoles, options.callerRoles)) return false;
      if (allowedCategories && !allowedCategories.has(n.category)) return false;
      if (requiredTags.length > 0 && !requiredTags.some((t) => n.tags.includes(t))) return false;
      return true;
    });

    const resultMap = new Map<string, AssociativeMemoryNode>();
    for (const s of authorizedSeeds) {
      resultMap.set(s.id, s);
    }

    // Optional associative link traversal (strictly enforcing tenant & RBAC on neighbor nodes)
    const hops = options.expandAssociativeHops ?? 0;
    let frontier = [...authorizedSeeds];
    for (let h = 0; h < hops; h++) {
      const nextFrontier: AssociativeMemoryNode[] = [];
      for (const node of frontier) {
        for (const link of node.links) {
          if (resultMap.has(link.targetNodeId)) continue;
          const neighbor = await this.getNodeById(link.targetNodeId, scope, scopeId);
          if (!neighbor) continue;
          if (neighbor.scope !== scope || neighbor.scopeId !== scopeId) continue;
          if (!this.isAuthorized(neighbor.aclRoles, options.callerRoles)) continue;
          if (allowedCategories && !allowedCategories.has(neighbor.category)) continue;
          resultMap.set(neighbor.id, neighbor);
          nextFrontier.push(neighbor);
        }
      }
      frontier = nextFrontier;
    }

    const results = Array.from(resultMap.values());
    return options.topK ? results.slice(0, options.topK) : results;
  }

  /**
   * Purges derived associative nodes and vector indices and reconstructs the entire
   * associative graph from immutable raw events without mutating any raw event.
   */
  async rebuildDerivedGraphAndIndex(options?: {
    scope?: MemoryScope;
    scopeId?: string;
    maxEdgesPerNode?: number;
    embedFn?: (text: string) => Promise<number[]>;
  }): Promise<ConsolidationStats> {
    const scope = options?.scope ?? this.defaultScope;
    const scopeId = options?.scopeId ?? this.defaultScopeId;

    const existingNodeIds =
      (await this.client.get<string[]>(ASSOC_NODE_LIST_KEY, {
        scope,
        scopeId,
        metadata: this.metadata
      })) ?? [];

    for (const nid of existingNodeIds) {
      const key = `${ASSOC_NODE_PREFIX}${nid}`;
      await this.client.delete(key, { scope, scopeId, metadata: this.metadata });
      await this.client.deleteVector(key, { scope, scopeId, metadata: this.metadata });
    }
    await this.client.set(ASSOC_NODE_LIST_KEY, [], { scope, scopeId, metadata: this.metadata });

    const allEventIds =
      (await this.client.get<string[]>(RAW_EVENT_LIST_KEY, {
        scope,
        scopeId,
        metadata: this.metadata
      })) ?? [];

    await this.client.set(PENDING_QUEUE_KEY, [...allEventIds], {
      scope,
      scopeId,
      metadata: this.metadata
    });

    return this.consolidateBounded({
      scope,
      scopeId,
      maxBatchSize: Math.max(1, allEventIds.length),
      maxEdgesPerNode: options?.maxEdgesPerNode,
      embedFn: options?.embedFn
    });
  }

  async getRawEventById(
    id: string,
    scope: MemoryScope = this.defaultScope,
    scopeId: string = this.defaultScopeId
  ): Promise<ImmutableRawEvent | undefined> {
    const rec = await this.client.get<ImmutableRawEvent>(`${RAW_EVENT_PREFIX}${id}`, {
      scope,
      scopeId,
      metadata: this.metadata
    });
    if (!rec || rec.scope !== scope || rec.scopeId !== scopeId) return undefined;
    return rec;
  }

  async getNodeById(
    nodeId: string,
    scope: MemoryScope = this.defaultScope,
    scopeId: string = this.defaultScopeId
  ): Promise<AssociativeMemoryNode | undefined> {
    const rec = await this.client.get<AssociativeMemoryNode>(`${ASSOC_NODE_PREFIX}${nodeId}`, {
      scope,
      scopeId,
      metadata: this.metadata
    });
    if (!rec || rec.scope !== scope || rec.scopeId !== scopeId) return undefined;
    return rec;
  }

  private computeAssociationScore(
    a: AssociativeMemoryNode,
    b: AssociativeMemoryNode
  ): { score: number; sharedTags: string[] } {
    const setB = new Set(b.tags);
    const sharedTags = a.tags.filter((t) => setB.has(t));
    const unionSize = new Set([...a.tags, ...b.tags]).size;
    const jaccard = unionSize > 0 ? sharedTags.length / unionSize : 0;
    const crossCategoryBonus = a.category !== b.category && sharedTags.length > 0 ? 0.15 : 0;
    return {
      score: Number(Math.min(1.0, jaccard + crossCategoryBonus).toFixed(4)),
      sharedTags
    };
  }

  private upsertBoundedLink(
    node: AssociativeMemoryNode,
    newLink: AssociativeLink,
    maxEdges: number
  ): number {
    const filtered = node.links.filter((l) => l.targetNodeId !== newLink.targetNodeId);
    filtered.push(newLink);
    filtered.sort((a, b) => b.weight - a.weight);

    const pruned = Math.max(0, filtered.length - maxEdges);
    node.links = filtered.slice(0, maxEdges);
    node.evidenceEventIds = Array.from(
      new Set([node.eventId, ...node.links.flatMap((l) => l.evidenceEventIds)])
    );
    return pruned;
  }

  private isAuthorized(aclRoles: string[], callerRoles?: string[]): boolean {
    if (!aclRoles || aclRoles.length === 0) return true;
    if (!callerRoles || callerRoles.length === 0) return false;
    const callerSet = new Set(callerRoles);
    return aclRoles.some((r) => callerSet.has(r));
  }

  private nodeIdFromKey(key: string): string | undefined {
    return key.startsWith(ASSOC_NODE_PREFIX) ? key.slice(ASSOC_NODE_PREFIX.length) : undefined;
  }
}
