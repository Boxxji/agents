import * as fs from 'node:fs';
import * as path from 'node:path';
import type { AddWorkspaceOptions, RecentWorkspace } from '@hanzo/agents-shared';
import type { CodingAgentState } from '../../../types/coding-agent-status';
import type { IDatabase } from '../database/IDatabase';
import { ipcBus } from '../ipc-bus';
import type { CanvasState } from '../types/database';

export function registerPersistenceHandlers(getDatabase: () => IDatabase | undefined): void {
  const requireDb = (): IDatabase => {
    const db = getDatabase();
    if (!db) {
      throw new Error('Database is not initialized');
    }
    return db;
  };

  ipcBus.handle('canvas:save', async (_event, canvasId: string, state: CanvasState) => {
    try {
      await requireDb().saveCanvas(canvasId, state);
      return { success: true };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  });

  ipcBus.handle('canvas:load', async (_event, canvasId: string) => {
    try {
      const canvas = await requireDb().loadCanvas(canvasId);
      return { success: true, data: canvas };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  });

  ipcBus.handle('canvas:list', async () => {
    try {
      const canvases = await requireDb().listCanvases();
      return { success: true, data: canvases };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  });

  ipcBus.handle('canvas:delete', async (_event, canvasId: string) => {
    try {
      await requireDb().deleteCanvas(canvasId);
      return { success: true };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  });

  ipcBus.handle('canvas:get-current-id', async () => {
    try {
      const canvasId = await requireDb().getCurrentCanvasId();
      return { success: true, data: canvasId };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  });

  ipcBus.handle('canvas:set-current-id', async (_event, canvasId: string) => {
    try {
      await requireDb().setCurrentCanvasId(canvasId);
      return { success: true };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  });

  ipcBus.handle('agent-status:save', async (_event, agentId: string, state: CodingAgentState) => {
    try {
      await requireDb().saveAgentStatus(agentId, state);
      return { success: true };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  });

  ipcBus.handle('agent-status:load', async (_event, agentId: string) => {
    try {
      const state = await requireDb().loadAgentStatus(agentId);
      return { success: true, data: state };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  });

  ipcBus.handle('agent-status:delete', async (_event, agentId: string) => {
    try {
      await requireDb().deleteAgentStatus(agentId);
      return { success: true };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  });

  ipcBus.handle('agent-status:load-all', async () => {
    try {
      const states = await requireDb().loadAllAgentStatuses();
      return { success: true, data: states };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  });

  ipcBus.handle('file:read', async (_event, filePath: string) => {
    try {
      if (!fs.existsSync(filePath)) {
        return { success: false, error: `File does not exist: ${filePath}` };
      }
      const content = fs.readFileSync(filePath, 'utf-8');
      return { success: true, data: content };
    } catch (error: any) {
      return { success: false, error: error.message || 'Unknown error' };
    }
  });

  ipcBus.handle('file:exists', async (_event, filePath: string) => {
    try {
      return { success: true, exists: fs.existsSync(filePath) };
    } catch (error: any) {
      return { success: false, exists: false, error: error.message || 'Unknown error' };
    }
  });

  ipcBus.handle(
    'recent-workspaces:add',
    async (_event, workspacePath: string, options?: AddWorkspaceOptions) => {
      try {
        const db = requireDb();
        const now = Date.now();
        const existing = await db.getRecentWorkspaceByPath(workspacePath);
        const workspace: RecentWorkspace = {
          path: workspacePath,
          name: options?.name || path.basename(workspacePath),
          lastOpenedAt: now,
          createdAt: existing?.createdAt || now,
        };
        await db.upsertRecentWorkspace(workspace);
        return { success: true };
      } catch (error) {
        return { success: false, error: (error as Error).message };
      }
    }
  );

  ipcBus.handle('recent-workspaces:get', async (_event, limit?: number) => {
    try {
      const workspaces = await requireDb().getRecentWorkspaces(limit);
      return { success: true, data: workspaces };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  });

  ipcBus.handle('recent-workspaces:remove', async (_event, workspacePath: string) => {
    try {
      await requireDb().removeRecentWorkspace(workspacePath);
      return { success: true };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  });

  ipcBus.handle('recent-workspaces:clear', async () => {
    try {
      await requireDb().clearAllRecentWorkspaces();
      return { success: true };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  });

  ipcBus.handle('recent-workspaces:has', async (_event, workspacePath: string) => {
    try {
      const workspace = await requireDb().getRecentWorkspaceByPath(workspacePath);
      return { success: true, data: workspace !== null };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  });

  ipcBus.handle('session-summary:get', async (_event, sessionId: string, workspacePath: string) => {
    try {
      const cached = await requireDb().getSessionSummary(sessionId, workspacePath);
      return { success: true, data: cached };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  });

  ipcBus.handle(
    'session-summary:save',
    async (
      _event,
      sessionId: string,
      workspacePath: string,
      summary: string,
      messageCount: number
    ) => {
      try {
        await requireDb().saveSessionSummary(sessionId, workspacePath, summary, messageCount);
        return { success: true };
      } catch (error) {
        return { success: false, error: (error as Error).message };
      }
    }
  );

  ipcBus.handle(
    'session-summary:is-stale',
    async (_event, sessionId: string, workspacePath: string, currentMessageCount: number) => {
      try {
        const isStale = await requireDb().isSessionSummaryStale(
          sessionId,
          workspacePath,
          currentMessageCount
        );
        return { success: true, data: isStale };
      } catch (error) {
        return { success: false, error: (error as Error).message };
      }
    }
  );

  ipcBus.handle(
    'session-summary:delete',
    async (_event, sessionId: string, workspacePath: string) => {
      try {
        await requireDb().deleteSessionSummary(sessionId, workspacePath);
        return { success: true };
      } catch (error) {
        return { success: false, error: (error as Error).message };
      }
    }
  );
}
