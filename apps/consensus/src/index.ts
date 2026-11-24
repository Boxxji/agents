export type { AgentCallFn, AgentConfig, AgentVoteFn, Vote } from './consensus.js';
export { AgentConsensus } from './consensus.js';
export type { McpClient } from './mcp-bridge.js';
export { runMeshConsensus } from './mcp-bridge.js';
export type { AgentProposal, ConsensusResult, QuasarVote } from './quasar-voting.js';
export {
  computeConsensusScore,
  hasReachedFinality,
  synthesizeTopK,
} from './quasar-voting.js';
