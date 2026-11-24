import type { TerminalSessionState } from '@hanzo/agents-shared';
import * as pty from 'node-pty';
import { ipcBus } from '../ipc-bus';
import { getHomePath } from '../paths';
import type { AgentHooksService } from '../services/agent-hooks';

const TERMINAL_BUFFER_LINES = 1000;

const terminalProcesses = new Map<string, pty.IPty>();
const terminalBuffers = new Map<string, string[]>();
const terminalSessionStates = new Map<string, TerminalSessionState>();
const terminalsBeingCreated = new Set<string>();
const terminalCreationTimes = new Map<string, number>();

export function registerTerminalHandlers(
  getAgentHooksService: () => AgentHooksService | undefined
): void {
  ipcBus.on('get-home-dir', () => getHomePath());

  ipcBus.on('terminal-create', (_event, terminalId: string, workspacePath?: string) => {
    const callTime = Date.now();
    if (terminalProcesses.has(terminalId) || terminalsBeingCreated.has(terminalId)) {
      return;
    }

    terminalsBeingCreated.add(terminalId);
    terminalCreationTimes.set(terminalId, callTime);

    const shell = process.platform === 'win32' ? 'cmd.exe' : process.env.SHELL || '/bin/bash';
    const shellArgs: string[] = process.platform === 'win32' ? [] : ['-l'];

    let hookEnv: Record<string, string> = {};
    const agentHooksService = getAgentHooksService();

    if (workspacePath && agentHooksService) {
      hookEnv = agentHooksService.getTerminalEnv({
        terminalId,
        workspacePath,
        agentId: `agent-${terminalId}`,
      });

      agentHooksService.ensureWorkspaceHooks(workspacePath).catch((error) => {
        console.error('[Main] Failed to set up workspace hooks:', error);
      });
    }

    const home = getHomePath();
    const ptyProcess = pty.spawn(shell, shellArgs, {
      name: 'xterm-256color',
      cols: 80,
      rows: 30,
      cwd: workspacePath || home || process.cwd(),
      env: {
        ...process.env,
        ...hookEnv,
        TERM: 'xterm-256color',
        COLORTERM: 'truecolor',
        PS1: process.env.PS1 || '$ ',
        ...(shell.includes('zsh') ? { ZDOTDIR: home } : {}),
      } as { [key: string]: string },
    });

    terminalProcesses.set(terminalId, ptyProcess);
    terminalsBeingCreated.delete(terminalId);

    if (!terminalBuffers.has(terminalId)) {
      terminalBuffers.set(terminalId, []);
    }

    ptyProcess.onData((data: string) => {
      const buffer = terminalBuffers.get(terminalId);
      if (buffer) {
        const lines = data.split('\n');
        buffer.push(...lines);
        while (buffer.length > TERMINAL_BUFFER_LINES) {
          buffer.shift();
        }
      }
      ipcBus.broadcast('terminal-data', { terminalId, data });
    });

    ptyProcess.onExit((exitInfo: { exitCode: number; signal?: number }) => {
      ipcBus.broadcast('terminal-exit', {
        terminalId,
        code: exitInfo.exitCode,
        signal: exitInfo.signal,
      });
      terminalProcesses.delete(terminalId);
      terminalsBeingCreated.delete(terminalId);
      terminalBuffers.delete(terminalId);
      terminalSessionStates.delete(terminalId);
    });
  });

  ipcBus.on(
    'terminal-resize',
    (_event, { terminalId, cols, rows }: { terminalId: string; cols: number; rows: number }) => {
      const ptyProcess = terminalProcesses.get(terminalId);
      if (ptyProcess && cols > 0 && rows > 0) {
        ptyProcess.resize(cols, rows);
      }
    }
  );

  ipcBus.on(
    'terminal-input',
    (_event, { terminalId, data }: { terminalId: string; data: string }) => {
      const ptyProcess = terminalProcesses.get(terminalId);
      if (ptyProcess) {
        ptyProcess.write(data);
      }
    }
  );

  ipcBus.on('terminal-destroy', (_event, terminalId: string) => {
    const ptyProcess = terminalProcesses.get(terminalId);
    if (ptyProcess) {
      ptyProcess.kill();
      terminalProcesses.delete(terminalId);
      terminalsBeingCreated.delete(terminalId);
      terminalCreationTimes.delete(terminalId);
      terminalBuffers.delete(terminalId);
      terminalSessionStates.delete(terminalId);
    }
  });

  ipcBus.handle('terminal-get-buffer', async (_event, terminalId: string) => {
    const buffer = terminalBuffers.get(terminalId);
    return { success: true, data: buffer ? buffer.join('\n') : '' };
  });

  ipcBus.handle('terminal-get-session-state', async (_event, terminalId: string) => {
    const state = terminalSessionStates.get(terminalId);
    return { success: true, data: state || null };
  });

  ipcBus.handle(
    'terminal-set-session-state',
    async (_event, terminalId: string, state: TerminalSessionState) => {
      terminalSessionStates.set(terminalId, state);
      return { success: true };
    }
  );

  ipcBus.handle('terminal-clear-session-state', async (_event, terminalId: string) => {
    terminalSessionStates.delete(terminalId);
    return { success: true };
  });
}

export function disposeAllTerminals(): void {
  terminalProcesses.forEach((ptyProcess) => {
    try {
      ptyProcess.kill();
    } catch {
      // Ignore exit race
    }
  });
  terminalProcesses.clear();
  terminalBuffers.clear();
  terminalSessionStates.clear();
}
