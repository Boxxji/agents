/**
 * Global Type Declarations
 *
 * Extends Window interface with Electron IPC APIs.
 */

import type {
  RecentWorkspacesAPI,
  SessionWatcherAPI,
  TerminalSessionAPI,
} from '@hanzo/agents-shared';
import type {
  AgentStatusAPI,
  ElectronAPI,
  GitAPI,
  LLMAPI,
  RepresentationAPI,
  SessionSummaryCacheAPI,
  ShellAPI,
  WorktreeAPI,
} from '../main/preload';
import type { CodingAgentAPI } from '../main/services/coding-agent';

// SVG module declarations moved to vite-env.d.ts

declare global {
  interface Window {
    electronAPI?: ElectronAPI;
    worktreeAPI?: WorktreeAPI;
    codingAgentAPI?: CodingAgentAPI;
    agentStatusAPI?: AgentStatusAPI;
    llmAPI?: LLMAPI;
    representationAPI?: RepresentationAPI;
    shellAPI?: ShellAPI;
    canvasAPI?: import('../main/preload').CanvasAPI;
    gitAPI?: GitAPI;
    terminalSessionAPI?: TerminalSessionAPI;
    sessionWatcherAPI?: SessionWatcherAPI;
    recentWorkspacesAPI?: RecentWorkspacesAPI;
    sessionSummaryCacheAPI?: SessionSummaryCacheAPI;
  }
}
