// Agent consensus engine using Quasar-inspired voting.
// Agents propose responses, vote on each other's proposals,
// and the system converges to a synthesized result.

import type { AgentProposal, ConsensusResult, QuasarVote } from './quasar-voting.js';
import { computeConsensusScore, hasReachedFinality, synthesizeTopK } from './quasar-voting.js';

export interface AgentConfig {
  id: string;
  model: string; // claude-opus-4-6, claude-sonnet-4-6, etc.
  role: string; // researcher, coder, reviewer, etc.
  systemPrompt: string;
}

export interface Vote {
  voterId: string;
  proposalId: string;
  score: number; // 0-1
  reason: string;
}

// Callback type for MCP-based agent communication.
// The bridge layer implements this to actually call agents.
export type AgentCallFn = (
  agentId: string,
  systemPrompt: string,
  userPrompt: string
) => Promise<{ content: string; confidence: number }>;

// Callback for voting. Agent receives all proposals and returns votes.
export type AgentVoteFn = (
  agentId: string,
  systemPrompt: string,
  proposals: AgentProposal[]
) => Promise<{ preference: string[]; scores: Map<string, number>; confidence: number }>;

export class AgentConsensus {
  private proposals: Map<string, AgentProposal> = new Map();
  private votes: QuasarVote[] = [];
  private flatVotes: Vote[] = [];
  private threshold: number;
  private proposalCounter = 0;

  constructor(threshold = 0.67) {
    this.threshold = threshold;
  }

  // Agent submits a proposal. Returns proposal ID.
  propose(agentId: string, content: string, confidence: number): string {
    const id = `proposal-${agentId}-${this.proposalCounter++}`;
    this.proposals.set(id, {
      id,
      agentId,
      content,
      confidence: Math.max(0, Math.min(1, confidence)),
      timestamp: Date.now(),
    });
    return id;
  }

  // Agent votes on a proposal.
  vote(voterId: string, proposalId: string, score: number, reason: string): void {
    if (!this.proposals.has(proposalId)) {
      throw new Error(`Unknown proposal: ${proposalId}`);
    }
    this.flatVotes.push({
      voterId,
      proposalId,
      score: Math.max(0, Math.min(1, score)),
      reason,
    });
  }

  // Submit a full QuasarVote (ranked preference + scores).
  submitQuasarVote(qv: QuasarVote): void {
    this.votes.push(qv);
    // Also record as flat votes for the result
    for (const [proposalId, score] of qv.scores) {
      this.flatVotes.push({
        voterId: qv.voterId,
        proposalId,
        score,
        reason: `rank: ${qv.preference.indexOf(proposalId) + 1}`,
      });
    }
  }

  // Check if consensus has been reached.
  hasConverged(): boolean {
    if (this.proposals.size === 0 || this.votes.length === 0) return false;
    const proposalIds = [...this.proposals.keys()];
    const scores = computeConsensusScore(this.votes, proposalIds);
    return hasReachedFinality(scores, this.threshold);
  }

  // Synthesize the final result from current proposals and votes.
  synthesize(): ConsensusResult {
    const proposals = [...this.proposals.values()];
    const proposalIds = proposals.map((p) => p.id);

    if (proposals.length === 0) {
      return {
        synthesis: '',
        proposals: [],
        votes: this.votes,
        convergenceRound: 0,
        agreement: 0,
      };
    }

    const scores = computeConsensusScore(this.votes, proposalIds);
    const sortedScores = [...scores.values()].sort((a, b) => b - a);
    const topScore = sortedScores[0] ?? 0;

    // Agreement = how close the top score is to 1.0, penalized by spread
    const spread = sortedScores.length > 1 ? topScore - sortedScores[sortedScores.length - 1] : 0;
    const agreement = Math.min(1, topScore * (1 + spread));

    const synthesis = synthesizeTopK(proposals, scores, Math.min(3, proposals.length));

    return {
      synthesis,
      proposals,
      votes: this.votes,
      convergenceRound: 1,
      agreement,
    };
  }

  // Run a full consensus round using agent callbacks.
  async runConsensusRound(
    prompt: string,
    agents: AgentConfig[],
    callAgent: AgentCallFn,
    voteAgent: AgentVoteFn,
    maxRounds = 3
  ): Promise<ConsensusResult> {
    for (let round = 1; round <= maxRounds; round++) {
      // Phase 1: Collect proposals (parallel)
      const proposalPromises = agents.map(async (agent) => {
        const response = await callAgent(agent.id, agent.systemPrompt, prompt);
        return this.propose(agent.id, response.content, response.confidence);
      });
      await Promise.all(proposalPromises);

      const proposals = [...this.proposals.values()];

      // Phase 2: Collect votes (parallel)
      const votePromises = agents.map(async (agent) => {
        // Agents only vote on proposals from other agents
        const othersProposals = proposals.filter((p) => p.agentId !== agent.id);
        if (othersProposals.length === 0) return;
        const voteResult = await voteAgent(agent.id, agent.systemPrompt, othersProposals);
        this.submitQuasarVote({
          voterId: agent.id,
          preference: voteResult.preference,
          scores: voteResult.scores,
          confidence: voteResult.confidence,
        });
      });
      await Promise.all(votePromises);

      // Phase 3: Check convergence
      if (this.hasConverged()) {
        const result = this.synthesize();
        result.convergenceRound = round;
        return result;
      }
    }

    // Did not converge within maxRounds -- return best effort
    const result = this.synthesize();
    result.convergenceRound = maxRounds;
    return result;
  }

  // Reset state for a new round.
  reset(): void {
    this.proposals.clear();
    this.votes = [];
    this.flatVotes = [];
    this.proposalCounter = 0;
  }
}
