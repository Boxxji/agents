import type { CodingAgentType, SessionFileChangeEvent } from '@hanzo/agents-shared';
import { ipcBus } from '../../ipc-bus';
import { getCodingAgent } from '../coding-agent';
import { SessionFileWatcher } from './SessionFileWatcher';

interface IPCResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

function successResponse<T>(data: T): IPCResponse<T> {
  return { success: true, data };
}

function errorResponse(error: string): IPCResponse<never> {
  return { success: false, error };
}

let sessionFileWatcher: SessionFileWatcher | null = null;

export function registerSessionWatcherIpcHandlers(): void {
  sessionFileWatcher = new SessionFileWatcher({
    debounceMs: 300,
    maxDebounceMs: 1000,
  });

  sessionFileWatcher.initialize();

  sessionFileWatcher.on('change', (event: SessionFileChangeEvent) => {
    ipcBus.broadcast('session:file-changed', event);
  });

  ipcBus.handle(
    'session-watcher:watch',
    async (_event, agentType: CodingAgentType): Promise<IPCResponse<void>> => {
      try {
        if (!sessionFileWatcher) {
          return errorResponse('SessionFileWatcher not initialized');
        }

        const agentResult = await getCodingAgent(agentType, {
          skipCliVerification: true,
        });

        if (!agentResult.success) {
          return errorResponse(agentResult.error.message);
        }

        const dataPaths = agentResult.data.getDataPaths();
        sessionFileWatcher.watchAgent(agentType, dataPaths);

        console.log('[Main] Started session file watching', { agentType, dataPaths });
        return successResponse(undefined);
      } catch (error) {
        console.error('[Main] Error starting session watch', { agentType, error });
        return errorResponse((error as Error).message);
      }
    }
  );

  ipcBus.handle(
    'session-watcher:unwatch',
    async (_event, agentType: CodingAgentType): Promise<IPCResponse<void>> => {
      try {
        if (sessionFileWatcher) {
          sessionFileWatcher.unwatchAgent(agentType);
        }
        console.log('[Main] Stopped session file watching', { agentType });
        return successResponse(undefined);
      } catch (error) {
        console.error('[Main] Error stopping session watch', { agentType, error });
        return errorResponse((error as Error).message);
      }
    }
  );

  ipcBus.handle(
    'session-watcher:status',
    async (): Promise<
      IPCResponse<{
        isActive: boolean;
        watchedAgents: CodingAgentType[];
        watchedPaths: string[];
      }>
    > => {
      if (!sessionFileWatcher) {
        return successResponse({
          isActive: false,
          watchedAgents: [],
          watchedPaths: [],
        });
      }
      return successResponse(sessionFileWatcher.getWatchStatus());
    }
  );

  console.log('[Main] Session watcher IPC handlers registered');
}

export function disposeSessionWatcher(): void {
  if (sessionFileWatcher) {
    sessionFileWatcher.dispose();
    sessionFileWatcher = null;
  }
}
