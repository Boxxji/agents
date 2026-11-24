// MCP bridge for agent consensus.
// Wraps the consensus engine with MCP tool-call semantics.
// The caller provides an MCP client; this module handles
// formatting prompts and parsing responses.

import type { AgentCallFn, AgentConfig, AgentVoteFn } from './consensus.js';
import { AgentConsensus } from './consensus.js';
import type { AgentProposal, ConsensusResult } from './quasar-voting.js';

export interface McpClient {
  // Send a prompt to a model and get a text response.
  call(model: string, systemPrompt: string, userPrompt: string): Promise<string>;
}

// Build the proposal prompt sent to each agent.
function buildProposalPrompt(prompt: string, role: string): string {
  return [
    `You are acting as a ${role} in a multi-agent consensus process.`,
    'Respond to the following prompt with your best answer.',
    'At the end, state your confidence as a number between 0 and 1 on a separate line prefixed with "Confidence: ".',
    '',
    prompt,
  ].join('\n');
}

// Build the voting prompt sent to each agent.
function buildVotingPrompt(proposals: AgentProposal[]): string {
  const header = [
    'You are voting on proposals from other agents.',
    'For each proposal, provide a quality score between 0 and 1.',
    'Then rank all proposals from best to worst by ID.',
    'Format your response as:',
    'SCORES:',
    '<proposal-id>: <score>',
    '...',
    'RANKING:',
    '<proposal-id>',
    '...',
    'Confidence: <your confidence 0-1>',
    '',
    'Proposals:',
  ];

  const body = proposals.map(
    (p) => `--- ${p.id} (by ${p.agentId}, confidence ${p.confidence}) ---\n${p.content}\n`
  );

  return [...header, ...body].join('\n');
}

// Parse confidence from agent response. Returns 0.5 if not found.
function parseConfidence(text: string): number {
  const match = text.match(/Confidence:\s*([\d.]+)/i);
  if (!match) return 0.5;
  const val = Number.parseFloat(match[1]);
  return Number.isNaN(val) ? 0.5 : Math.max(0, Math.min(1, val));
}

// Parse voting response into scores, preference, and confidence.
function parseVotingResponse(
  text: string,
  proposalIds: string[]
): { preference: string[]; scores: Map<string, number>; confidence: number } {
  const scores = new Map<string, number>();
  const preference: string[] = [];

  // Parse SCORES section
  const scoresMatch = text.match(/SCORES:\s*\n([\s\S]*?)(?:RANKING:|$)/i);
  if (scoresMatch) {
    const lines = scoresMatch[1].trim().split('\n');
    for (const line of lines) {
      const m = line.match(/^([\w-]+):\s*([\d.]+)/);
      if (m && proposalIds.includes(m[1])) {
        scores.set(m[1], Math.max(0, Math.min(1, Number.parseFloat(m[2]))));
      }
    }
  }

  // Parse RANKING section
  const rankingMatch = text.match(/RANKING:\s*\n([\s\S]*?)(?:Confidence:|$)/i);
  if (rankingMatch) {
    const lines = rankingMatch[1].trim().split('\n');
    for (const line of lines) {
      const id = line.trim();
      if (proposalIds.includes(id)) {
        preference.push(id);
      }
    }
  }

  // Fill in any missing proposals with default scores
  for (const id of proposalIds) {
    if (!scores.has(id)) scores.set(id, 0.5);
    if (!preference.includes(id)) preference.push(id);
  }

  return {
    preference,
    scores,
    confidence: parseConfidence(text),
  };
}

// Run multi-agent consensus via an MCP client.
export async function runMeshConsensus(
  prompt: string,
  agents: AgentConfig[],
  mcpClient: McpClient,
  options?: {
    threshold?: number;
    maxRounds?: number;
  }
): Promise<ConsensusResult> {
  const threshold = options?.threshold ?? 0.67;
  const maxRounds = options?.maxRounds ?? 3;

  const engine = new AgentConsensus(threshold);

  const callAgent: AgentCallFn = async (agentId, systemPrompt, userPrompt) => {
    const agent = agents.find((a) => a.id === agentId);
    const role = agent?.role ?? 'assistant';
    const model = agent?.model ?? 'claude-sonnet-4-6';
    const fullPrompt = buildProposalPrompt(userPrompt, role);
    const response = await mcpClient.call(model, systemPrompt, fullPrompt);
    const confidence = parseConfidence(response);
    // Strip the confidence line from content
    const content = response.replace(/\nConfidence:\s*[\d.]+\s*$/i, '').trim();
    return { content, confidence };
  };

  const voteAgent: AgentVoteFn = async (agentId, systemPrompt, proposals) => {
    const agent = agents.find((a) => a.id === agentId);
    const model = agent?.model ?? 'claude-sonnet-4-6';
    const votingPrompt = buildVotingPrompt(proposals);
    const response = await mcpClient.call(model, systemPrompt, votingPrompt);
    const proposalIds = proposals.map((p) => p.id);
    return parseVotingResponse(response, proposalIds);
  };

  return engine.runConsensusRound(prompt, agents, callAgent, voteAgent, maxRounds);
}
