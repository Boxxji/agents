import * as crypto from 'node:crypto';
import * as path from 'node:path';
import { getLogServer, isMcpMode, startMcpServer, stopMcpServer } from '../instrumentation';
import { DatabaseFactory } from './database';
import type { IDatabase } from './database/IDatabase';
import { registerCodingAgentHandlers } from './handlers/coding-agent-handlers';
import { registerPersistenceHandlers } from './handlers/persistence-handlers';
import { registerRepresentationHandlers } from './handlers/representation-handlers';
import { registerShellAndGitHandlers } from './handlers/shell-git-handlers';
import { disposeAllTerminals, registerTerminalHandlers } from './handlers/terminal-handlers';
import { ipcBus } from './ipc-bus';
import { fixPath, getHomePath, getUserDataPath } from './paths';
import { type AgentHooksService, createAgentHooksService } from './services/agent-hooks';
import { disposeAllCodingAgents } from './services/coding-agent';
import { registerAgentActionHandlers } from './services/coding-agent/agent-event-bridge';
import { DEFAULT_LLM_CONFIG, LLMServiceFactory, registerLLMIpcHandlers } from './services/llm';
import { type IIdGenerator, type ILogger, RepresentationService } from './services/representation';
import {
  disposeSessionWatcher,
  registerSessionWatcherIpcHandlers,
} from './services/session-watcher';
import { SidecarRpcServer } from './sidecar-server';
import { WorktreeManagerFactory } from './worktree';
import { registerWorktreeIpcHandlers } from './worktree/ipc';

fixPath();

let database: IDatabase | undefined;
let agentHooksService: AgentHooksService | undefined;
let rpcServer: SidecarRpcServer | undefined;
let shuttingDown = false;

const representationLogger: ILogger = {
  info: (message: string, context?: Record<string, unknown>) =>
    console.log(`[Representation] ${message}`, context || ''),
  warn: (message: string, context?: Record<string, unknown>) =>
    console.warn(`[Representation] ${message}`, context || ''),
  error: (message: string, context?: Record<string, unknown>) =>
    console.error(`[Representation] ${message}`, context || ''),
};

const representationIdGenerator: IIdGenerator = {
  generate: () => crypto.randomUUID(),
};

const representationService = new RepresentationService(
  { defaultTimeout: 30000, initializeProvidersOnStart: true },
  { logger: representationLogger, idGenerator: representationIdGenerator }
);

async function bootstrap(): Promise<void> {
  const mcpMode = isMcpMode() || process.env.HANZO_AGENTS_MCP === '1';
  const port = Number(process.env.HANZO_AGENTS_PORT || '13880');
  const token = process.env.HANZO_AGENTS_TOKEN;

  registerTerminalHandlers(() => agentHooksService);
  registerPersistenceHandlers(() => database);
  registerCodingAgentHandlers();
  registerRepresentationHandlers(representationService);
  registerShellAndGitHandlers();
  registerAgentActionHandlers();
  registerSessionWatcherIpcHandlers();

  try {
    database = await DatabaseFactory.getDatabase('sqlite');
    console.log('[Main] Database initialized successfully');
  } catch (error) {
    console.error('[Main] Error initializing database', error);
  }

  try {
    const worktreeBaseDir = path.join(getUserDataPath(), 'worktrees');
    WorktreeManagerFactory.configure({
      baseWorktreeDirectory: worktreeBaseDir,
    });
    await WorktreeManagerFactory.getManager();
    registerWorktreeIpcHandlers();
    console.log('[Main] WorktreeManager initialized successfully');
  } catch (error) {
    console.error('[Main] Error initializing WorktreeManager', error);
  }

  try {
    LLMServiceFactory.configure(DEFAULT_LLM_CONFIG);
    await LLMServiceFactory.getService();
    registerLLMIpcHandlers();
    console.log('[Main] LLM Service initialized successfully');
  } catch (error) {
    console.error('[Main] Error initializing LLM Service', error);
  }

  try {
    const initResult = await representationService.initialize();
    if (initResult.success) {
      console.log('[Main] RepresentationService initialized successfully');
    } else {
      console.error('[Main] Error initializing RepresentationService', initResult.error);
    }
  } catch (error) {
    console.error('[Main] Error initializing RepresentationService', error);
  }

  try {
    const homeDir = getHomePath();
    agentHooksService = createAgentHooksService(homeDir);
    await agentHooksService.ensureSetup();
    agentHooksService.startServer();

    agentHooksService.on('lifecycle', (event) => {
      ipcBus.broadcast('agent-lifecycle', event);
    });
    console.log('[Main] AgentHooksService initialized successfully');
  } catch (error) {
    console.error('[Main] Error initializing AgentHooksService', error);
  }

  if (mcpMode) {
    try {
      await startMcpServer();
      getLogServer().log('info', 'mcp', 'MCP server started successfully');
      console.log('[Main] MCP server started successfully');
    } catch (error) {
      console.error('[Main] Error starting MCP server', error);
      process.exit(1);
    }
  }

  rpcServer = new SidecarRpcServer({ port, host: '127.0.0.1', token });
  await rpcServer.start();
}

async function shutdown(): Promise<void> {
  if (shuttingDown) return;
  shuttingDown = true;
  console.log('[Main] Sidecar shutting down...');
  disposeAllTerminals();
  DatabaseFactory.closeDatabase();
  WorktreeManagerFactory.closeManager();
  disposeSessionWatcher();
  await disposeAllCodingAgents();
  await LLMServiceFactory.dispose();
  await representationService.dispose();
  agentHooksService?.dispose();
  await stopMcpServer();
  await rpcServer?.stop();
  process.exit(0);
}

process.on('SIGINT', () => void shutdown());
process.on('SIGTERM', () => void shutdown());

bootstrap().catch((error) => {
  console.error('[Main] Fatal sidecar startup error:', error);
  process.exit(1);
});
