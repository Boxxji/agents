// Quasar-inspired voting for agent consensus.
// Two-round protocol: propose -> vote -> converge.
//
// Scoring: weighted sum of (score * confidence * rank_weight) normalized
// by total confidence. Finality requires the top proposal to exceed
// the threshold AND maintain a margin over second place.

export interface AgentProposal {
  id: string;
  agentId: string;
  content: string;
  confidence: number; // 0-1
  timestamp: number;
}

export interface QuasarVote {
  voterId: string;
  preference: string[]; // Ranked proposal IDs, most preferred first
  scores: Map<string, number>; // Per-proposal quality scores 0-1
  confidence: number; // Voter's self-assessed confidence 0-1
}

export interface ConsensusResult {
  synthesis: string;
  proposals: AgentProposal[];
  votes: QuasarVote[];
  convergenceRound: number;
  agreement: number; // 0-1
}

// Compute consensus score for each proposal from all votes.
// Score_p = sum(score_i * confidence_i * rank_weight_i) / sum(confidence_i)
// where rank_weight = 1 / (rank_position + 1) for the voter's preference order.
export function computeConsensusScore(
  votes: QuasarVote[],
  proposalIds: string[]
): Map<string, number> {
  const scores = new Map<string, number>();
  const totalConfidence = votes.reduce((sum, v) => sum + v.confidence, 0);

  if (totalConfidence === 0) {
    for (const id of proposalIds) scores.set(id, 0);
    return scores;
  }

  for (const id of proposalIds) {
    let weighted = 0;
    for (const vote of votes) {
      const rawScore = vote.scores.get(id) ?? 0;
      const rankIndex = vote.preference.indexOf(id);
      // If not ranked, use worst possible rank weight
      const rankWeight = rankIndex >= 0 ? 1 / (rankIndex + 1) : 1 / (proposalIds.length + 1);
      weighted += rawScore * vote.confidence * rankWeight;
    }
    scores.set(id, weighted / totalConfidence);
  }

  return scores;
}

// Check if the top proposal has reached finality:
// 1. Top score exceeds the threshold
// 2. Gap between #1 and #2 exceeds a minimum margin (10% of threshold)
export function hasReachedFinality(scores: Map<string, number>, threshold: number): boolean {
  const sorted = [...scores.values()].sort((a, b) => b - a);
  if (sorted.length === 0) return false;
  const top = sorted[0];
  if (top < threshold) return false;
  if (sorted.length === 1) return true;
  const margin = threshold * 0.1;
  return top - sorted[1] > margin;
}

// Synthesize top K proposals into a single response.
// Takes unique sentences from each proposal in score order, deduplicating
// by normalized content. Returns the merged text.
export function synthesizeTopK(
  proposals: AgentProposal[],
  scores: Map<string, number>,
  k: number
): string {
  const ranked = [...proposals]
    .sort((a, b) => (scores.get(b.id) ?? 0) - (scores.get(a.id) ?? 0))
    .slice(0, k);

  if (ranked.length === 0) return '';
  if (ranked.length === 1) return ranked[0].content;

  // Split each proposal into sentences, collect unique ones in rank order
  const seen = new Set<string>();
  const parts: string[] = [];

  for (const proposal of ranked) {
    const sentences = proposal.content
      .split(/(?<=[.!?])\s+/)
      .map((s) => s.trim())
      .filter((s) => s.length > 0);

    for (const sentence of sentences) {
      const normalized = sentence.toLowerCase().replace(/\s+/g, ' ');
      if (!seen.has(normalized)) {
        seen.add(normalized);
        parts.push(sentence);
      }
    }
  }

  return parts.join(' ');
}
