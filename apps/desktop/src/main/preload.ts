/**
 * Desktop Bridge Types for Hanzo Agents (Tauri v2 + Node Orchestration Sidecar).
 *
 * Exports the TypeScript API contracts consumed by the React renderer (`global.d.ts`).
 * At runtime, the Rust Tauri v2 shell injects these globals via `initialization_script`
 * backed by the local Node sidecar RPC bus and native Tauri window/dialog commands.
 */
import type {
  AddWorkspaceOptions,
  GitInfo,
  RecentWorkspace,
  RecentWorkspacesAPI,
  SessionFileChangeEvent,
  SessionWatcherAPI,
  CodingAgentType as SharedCodingAgentType,
  TerminalSessionAPI,
  TerminalSessionState,
} from '@hanzo/agents-shared';
import type { CodingAgentState } from '../../types/coding-agent-status';
import type {
  AgentCapabilities,
  CodingAgentAPI,
  CodingAgentType,
  ContinueOptions,
  ForkOptions,
  GenerateRequest,
  GenerateResponse,
  MessageFilterOptions,
  SessionContent,
  SessionFilterOptions,
  SessionIdentifier,
  SessionSummary,
  StreamingChunk,
} from './services/coding-agent';
import type {
  ChatRequest,
  ChatResponse,
  LLMCapabilities,
  ModelInfo,
  VendorId,
} from './services/llm';
import type {
  AnyRepresentationOutput,
  AudioRepresentationOutput,
  AudioTransformOptions,
  ImageRepresentationOutput,
  ImageTransformOptions,
  RepresentationCapabilities,
  RepresentationInput,
  RepresentationType,
  SummaryRepresentationOutput,
  SummaryTransformOptions,
} from './services/representation';
import type { CanvasMetadata, CanvasState } from './types/database';
import type {
  WorktreeInfo,
  WorktreeProvisionOptions,
  WorktreeReleaseOptions,
} from './types/worktree';

export type {
  AddWorkspaceOptions,
  AgentCapabilities,
  CodingAgentAPI,
  GenerateResponse,
  GitInfo,
  RecentWorkspace,
  RecentWorkspacesAPI,
  SessionContent,
  SessionFileChangeEvent,
  SessionSummary,
  SessionWatcherAPI,
  SharedCodingAgentType,
  TerminalSessionAPI,
  TerminalSessionState,
};

export interface McpBridgeRequest {
  requestId: string;
  channel: string;
  payload: unknown;
}

export interface McpBridgeResponse {
  requestId: string;
  result?: unknown;
  error?: string;
}

export interface ElectronAPI {
  createTerminal: (terminalId: string, workspacePath?: string) => void;
  onTerminalData: (callback: (data: { terminalId: string; data: string }) => void) => void;
  onTerminalExit: (
    callback: (data: { terminalId: string; code: number; signal?: number }) => void
  ) => void;
  sendTerminalInput: (terminalId: string, data: string) => void;
  sendTerminalResize: (terminalId: string, cols: number, rows: number) => void;
  destroyTerminal: (terminalId: string) => void;
  removeAllListeners: (channel: string) => void;
  getHomeDir: () => string;
  onMcpBridgeRequest?: (callback: (request: McpBridgeRequest) => void) => void;
  sendMcpBridgeResponse?: (response: McpBridgeResponse) => void;
}

export interface CanvasAPI {
  saveCanvas: (canvasId: string, state: CanvasState) => Promise<void>;
  loadCanvas: (canvasId: string) => Promise<CanvasState | null>;
  listCanvases: () => Promise<CanvasMetadata[]>;
  deleteCanvas: (canvasId: string) => Promise<void>;
  getCurrentCanvasId: () => Promise<string | null>;
  setCurrentCanvasId: (canvasId: string) => Promise<void>;
}

export interface WorktreeAPI {
  provision: (
    repoPath: string,
    branchName: string,
    options?: WorktreeProvisionOptions
  ) => Promise<WorktreeInfo>;
  release: (worktreeId: string, options?: WorktreeReleaseOptions) => Promise<void>;
  get: (worktreeId: string) => Promise<WorktreeInfo | null>;
  list: (repoPath?: string) => Promise<WorktreeInfo[]>;
}

export interface AgentStatusAPI {
  saveAgentStatus: (agentId: string, state: CodingAgentState) => Promise<void>;
  loadAgentStatus: (agentId: string) => Promise<CodingAgentState | null>;
  deleteAgentStatus: (agentId: string) => Promise<void>;
  loadAllAgentStatuses: () => Promise<CodingAgentState[]>;
}

export interface LLMAPI {
  chat: (request: ChatRequest) => Promise<ChatResponse>;
  chatStream: (
    requestId: string,
    request: ChatRequest,
    onChunk: (chunk: string) => void
  ) => Promise<ChatResponse>;
  chatWithTools: (request: ChatRequest, maxIterations?: number) => Promise<ChatResponse>;
  setApiKey: (vendor: VendorId, apiKey: string) => Promise<void>;
  deleteApiKey: (vendor: VendorId) => Promise<void>;
  hasApiKey: (vendor: VendorId) => Promise<boolean>;
  listVendorsWithKeys: () => Promise<VendorId[]>;
  getAvailableModels: () => Promise<ModelInfo[]>;
  isConfigured: () => Promise<boolean>;
  getCapabilities: () => Promise<LLMCapabilities>;
  onStreamChunk: (callback: (data: { requestId: string; chunk: string }) => void) => () => void;
}

export interface ProviderInfo {
  providerId: string;
  providerName: string;
  representationType: RepresentationType;
  capabilities: RepresentationCapabilities;
}

export interface RepresentationAPI {
  getAvailableTypes: () => Promise<RepresentationType[]>;
  transform: (providerId: string, input: RepresentationInput) => Promise<AnyRepresentationOutput>;
  transformToImage: (
    input: RepresentationInput,
    options?: ImageTransformOptions
  ) => Promise<ImageRepresentationOutput>;
  transformToSummary: (
    input: RepresentationInput,
    options?: SummaryTransformOptions
  ) => Promise<SummaryRepresentationOutput>;
  transformToAudio: (
    input: RepresentationInput,
    options?: AudioTransformOptions
  ) => Promise<AudioRepresentationOutput>;
  getAllProviders: () => Promise<ProviderInfo[]>;
}

export type EditorApp = 'vscode' | 'cursor' | 'zed' | 'sublime' | 'atom' | 'webstorm' | 'finder';

export interface ShellAPI {
  openWithEditor: (directoryPath: string, editor: EditorApp) => Promise<void>;
  getAvailableEditors: () => Promise<EditorApp[]>;
  showInFolder: (path: string) => Promise<void>;
  openDirectoryDialog: (options?: {
    title?: string;
    defaultPath?: string;
  }) => Promise<string | null>;
}

export interface FileAPI {
  readFile: (filePath: string) => Promise<string>;
  exists: (filePath: string) => Promise<boolean>;
}

export interface GitAPI {
  getInfo: (workspacePath: string) => Promise<GitInfo>;
  listBranches: (workspacePath: string) => Promise<string[] | null>;
  createBranch: (
    workspacePath: string,
    branchName: string
  ) => Promise<{ success: boolean; error?: string }>;
  checkoutBranch: (
    workspacePath: string,
    branchName: string
  ) => Promise<{ success: boolean; error?: string }>;
  getGithubUsername: () => Promise<{ success: boolean; username?: string; error?: string }>;
}

export interface WindowAPI {
  minimize: () => void;
  maximize: () => void;
  close: () => void;
  isMaximized: () => Promise<boolean>;
}

export interface SessionSummaryCacheAPI {
  getSummary: (
    sessionId: string,
    workspacePath: string
  ) => Promise<{ summary: string; messageCount: number } | null>;
  saveSummary: (
    sessionId: string,
    workspacePath: string,
    summary: string,
    messageCount: number
  ) => Promise<void>;
  isStale: (
    sessionId: string,
    workspacePath: string,
    currentMessageCount: number
  ) => Promise<boolean>;
  deleteSummary: (sessionId: string, workspacePath: string) => Promise<void>;
}

export type {
  CodingAgentType,
  ContinueOptions,
  ForkOptions,
  GenerateRequest,
  MessageFilterOptions,
  SessionFilterOptions,
  SessionIdentifier,
  StreamingChunk,
};
