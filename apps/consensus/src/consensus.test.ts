import { describe, expect, it } from 'vitest';
import type { AgentCallFn, AgentConfig, AgentVoteFn } from './consensus.js';
import { AgentConsensus } from './consensus.js';
import type { AgentProposal, QuasarVote } from './quasar-voting.js';
import { computeConsensusScore, hasReachedFinality, synthesizeTopK } from './quasar-voting.js';

// --- quasar-voting unit tests ---

describe('computeConsensusScore', () => {
  it('scores proposals by weighted votes', () => {
    const proposalIds = ['p1', 'p2', 'p3'];
    const votes: QuasarVote[] = [
      {
        voterId: 'a1',
        preference: ['p1', 'p2', 'p3'],
        scores: new Map([
          ['p1', 0.9],
          ['p2', 0.6],
          ['p3', 0.3],
        ]),
        confidence: 0.8,
      },
      {
        voterId: 'a2',
        preference: ['p2', 'p1', 'p3'],
        scores: new Map([
          ['p1', 0.7],
          ['p2', 0.8],
          ['p3', 0.2],
        ]),
        confidence: 0.9,
      },
    ];

    const scores = computeConsensusScore(votes, proposalIds);
    // p1 should score well (high scores, ranked #1 by a1 and #2 by a2)
    // p2 should also score well (ranked #1 by a2 with higher confidence)
    expect(scores.get('p1')!).toBeGreaterThan(scores.get('p3')!);
    expect(scores.get('p2')!).toBeGreaterThan(scores.get('p3')!);
  });

  it('returns zeros when all confidences are zero', () => {
    const votes: QuasarVote[] = [
      {
        voterId: 'a1',
        preference: ['p1'],
        scores: new Map([['p1', 0.9]]),
        confidence: 0,
      },
    ];
    const scores = computeConsensusScore(votes, ['p1']);
    expect(scores.get('p1')).toBe(0);
  });

  it('weights high-confidence voters more heavily', () => {
    const proposalIds = ['p1', 'p2'];
    // a1 (low confidence) prefers p1, a2 (high confidence) prefers p2
    const votes: QuasarVote[] = [
      {
        voterId: 'a1',
        preference: ['p1', 'p2'],
        scores: new Map([
          ['p1', 0.9],
          ['p2', 0.1],
        ]),
        confidence: 0.2,
      },
      {
        voterId: 'a2',
        preference: ['p2', 'p1'],
        scores: new Map([
          ['p1', 0.3],
          ['p2', 0.95],
        ]),
        confidence: 0.95,
      },
    ];

    const scores = computeConsensusScore(votes, proposalIds);
    // High-confidence voter prefers p2, so p2 should win
    expect(scores.get('p2')!).toBeGreaterThan(scores.get('p1')!);
  });
});

describe('hasReachedFinality', () => {
  it('returns true when top score exceeds threshold with margin', () => {
    const scores = new Map([
      ['p1', 0.85],
      ['p2', 0.3],
    ]);
    expect(hasReachedFinality(scores, 0.67)).toBe(true);
  });

  it('returns false when top score is below threshold', () => {
    const scores = new Map([
      ['p1', 0.5],
      ['p2', 0.4],
    ]);
    expect(hasReachedFinality(scores, 0.67)).toBe(false);
  });

  it('returns false when margin between #1 and #2 is too small', () => {
    // Both above threshold but gap < threshold * 0.1 = 0.067
    const scores = new Map([
      ['p1', 0.72],
      ['p2', 0.7],
    ]);
    expect(hasReachedFinality(scores, 0.67)).toBe(false);
  });

  it('returns true for single proposal above threshold', () => {
    const scores = new Map([['p1', 0.8]]);
    expect(hasReachedFinality(scores, 0.67)).toBe(true);
  });

  it('returns false for empty scores', () => {
    expect(hasReachedFinality(new Map(), 0.67)).toBe(false);
  });
});

describe('synthesizeTopK', () => {
  it('returns single proposal content unchanged', () => {
    const proposals: AgentProposal[] = [
      { id: 'p1', agentId: 'a1', content: 'The answer is 42.', confidence: 0.9, timestamp: 1 },
    ];
    const scores = new Map([['p1', 0.9]]);
    expect(synthesizeTopK(proposals, scores, 1)).toBe('The answer is 42.');
  });

  it('merges unique sentences from top proposals', () => {
    const proposals: AgentProposal[] = [
      {
        id: 'p1',
        agentId: 'a1',
        content: 'The sky is blue. Water is wet.',
        confidence: 0.9,
        timestamp: 1,
      },
      {
        id: 'p2',
        agentId: 'a2',
        content: 'The sky is blue. Fire is hot.',
        confidence: 0.8,
        timestamp: 2,
      },
    ];
    const scores = new Map([
      ['p1', 0.9],
      ['p2', 0.7],
    ]);
    const result = synthesizeTopK(proposals, scores, 2);
    // p1 ranked first, so its sentences come first
    expect(result).toContain('The sky is blue.');
    expect(result).toContain('Water is wet.');
    expect(result).toContain('Fire is hot.');
    // "The sky is blue." should only appear once (deduplicated)
    const count = (result.match(/The sky is blue\./g) ?? []).length;
    expect(count).toBe(1);
  });

  it('respects K limit', () => {
    const proposals: AgentProposal[] = [
      { id: 'p1', agentId: 'a1', content: 'First.', confidence: 0.9, timestamp: 1 },
      { id: 'p2', agentId: 'a2', content: 'Second.', confidence: 0.8, timestamp: 2 },
      { id: 'p3', agentId: 'a3', content: 'Third.', confidence: 0.7, timestamp: 3 },
    ];
    const scores = new Map([
      ['p1', 0.9],
      ['p2', 0.5],
      ['p3', 0.1],
    ]);
    const result = synthesizeTopK(proposals, scores, 2);
    expect(result).toContain('First.');
    expect(result).toContain('Second.');
    expect(result).not.toContain('Third.');
  });
});

// --- AgentConsensus integration tests ---

describe('AgentConsensus', () => {
  it('propose and vote workflow with 3 agents', () => {
    const engine = new AgentConsensus(0.5);

    const id1 = engine.propose('agent-a', 'Answer A is correct.', 0.9);
    const id2 = engine.propose('agent-b', 'Answer B is correct.', 0.7);
    const id3 = engine.propose('agent-c', 'Answer A is correct. Also C.', 0.8);

    // All agents vote: strong consensus for id1
    engine.submitQuasarVote({
      voterId: 'agent-a',
      preference: [id2, id3],
      scores: new Map([
        [id2, 0.5],
        [id3, 0.7],
      ]),
      confidence: 0.9,
    });
    engine.submitQuasarVote({
      voterId: 'agent-b',
      preference: [id1, id3],
      scores: new Map([
        [id1, 0.9],
        [id3, 0.6],
      ]),
      confidence: 0.7,
    });
    engine.submitQuasarVote({
      voterId: 'agent-c',
      preference: [id1, id2],
      scores: new Map([
        [id1, 0.85],
        [id2, 0.4],
      ]),
      confidence: 0.8,
    });

    expect(engine.hasConverged()).toBe(true);

    const result = engine.synthesize();
    expect(result.proposals).toHaveLength(3);
    expect(result.votes).toHaveLength(3);
    expect(result.synthesis.length).toBeGreaterThan(0);
    expect(result.agreement).toBeGreaterThan(0);
  });

  it('does not converge with high disagreement', () => {
    const engine = new AgentConsensus(0.8); // High threshold

    engine.propose('a', 'X.', 0.5);
    engine.propose('b', 'Y.', 0.5);

    // Votes are split evenly
    engine.submitQuasarVote({
      voterId: 'a',
      preference: ['proposal-b-1'],
      scores: new Map([['proposal-b-1', 0.5]]),
      confidence: 0.5,
    });
    engine.submitQuasarVote({
      voterId: 'b',
      preference: ['proposal-a-0'],
      scores: new Map([['proposal-a-0', 0.5]]),
      confidence: 0.5,
    });

    expect(engine.hasConverged()).toBe(false);
  });

  it('vote rejects unknown proposal', () => {
    const engine = new AgentConsensus();
    expect(() => engine.vote('a', 'nonexistent', 0.5, 'test')).toThrow('Unknown proposal');
  });

  it('clamps confidence and score to 0-1', () => {
    const engine = new AgentConsensus();
    const id = engine.propose('a', 'test', 1.5); // Should clamp to 1
    engine.vote('b', id, -0.5, 'too low'); // Should clamp to 0

    const result = engine.synthesize();
    const proposal = result.proposals.find((p) => p.id === id);
    expect(proposal!.confidence).toBe(1);
  });

  it('reset clears all state', () => {
    const engine = new AgentConsensus();
    engine.propose('a', 'test', 0.8);
    engine.reset();
    const result = engine.synthesize();
    expect(result.proposals).toHaveLength(0);
    expect(result.synthesis).toBe('');
  });
});

describe('AgentConsensus.runConsensusRound', () => {
  it('runs full round with mock agents and converges', async () => {
    const agents: AgentConfig[] = [
      { id: 'researcher', model: 'claude-opus-4-6', role: 'researcher', systemPrompt: 'Research.' },
      { id: 'coder', model: 'claude-sonnet-4-6', role: 'coder', systemPrompt: 'Code.' },
      { id: 'reviewer', model: 'claude-sonnet-4-6', role: 'reviewer', systemPrompt: 'Review.' },
    ];

    // Mock agent calls: each returns a deterministic response
    const callAgent: AgentCallFn = async (agentId) => ({
      content: `Response from ${agentId}. This is good.`,
      confidence: 0.85,
    });

    // Mock voting: each agent gives high scores to the first proposal it sees
    const voteAgent: AgentVoteFn = async (_agentId, _prompt, proposals) => {
      const scores = new Map<string, number>();
      const preference: string[] = [];
      for (let i = 0; i < proposals.length; i++) {
        const score = 1 - i * 0.2; // First gets 1.0, second 0.8, etc.
        scores.set(proposals[i].id, score);
        preference.push(proposals[i].id);
      }
      return { preference, scores, confidence: 0.9 };
    };

    const engine = new AgentConsensus(0.5);
    const result = await engine.runConsensusRound(
      'What is the best approach?',
      agents,
      callAgent,
      voteAgent,
      3
    );

    expect(result.proposals.length).toBeGreaterThanOrEqual(3);
    expect(result.synthesis.length).toBeGreaterThan(0);
    expect(result.agreement).toBeGreaterThan(0);
    expect(result.convergenceRound).toBeGreaterThanOrEqual(1);
  });

  it('runs multiple rounds when consensus is not reached', async () => {
    const agents: AgentConfig[] = [
      { id: 'a', model: 'claude-sonnet-4-6', role: 'a', systemPrompt: '' },
      { id: 'b', model: 'claude-sonnet-4-6', role: 'b', systemPrompt: '' },
    ];

    let callCount = 0;
    const callAgent: AgentCallFn = async (agentId) => {
      callCount++;
      return { content: `Unique response ${callCount} from ${agentId}.`, confidence: 0.3 };
    };

    // Low scores and split votes -- should not converge
    const voteAgent: AgentVoteFn = async (_agentId, _prompt, proposals) => {
      const scores = new Map<string, number>();
      const preference: string[] = [];
      for (const p of proposals) {
        scores.set(p.id, 0.4);
        preference.push(p.id);
      }
      return { preference, scores, confidence: 0.3 };
    };

    const engine = new AgentConsensus(0.95); // Very high threshold
    const result = await engine.runConsensusRound('test', agents, callAgent, voteAgent, 2);

    // Should have called agents in 2 rounds (2 agents * 2 rounds = 4 calls minimum)
    expect(callCount).toBeGreaterThanOrEqual(4);
    expect(result.convergenceRound).toBe(2);
  });
});
