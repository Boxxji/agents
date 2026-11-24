import type {
  AgentActionResponse,
  AgentEvent,
  EventRegistry,
  EventResult,
  PermissionPayload,
} from '@hanzo/agents-shared';
import { ipcBus } from '../ipc-bus';
import type {
  CodingAgentType,
  ContinueOptions,
  ForkOptions,
  GenerateRequest,
  MessageFilterOptions,
  SessionFilterOptions,
  SessionIdentifier,
  StreamingChunk,
} from '../services/coding-agent';
import { getCodingAgent } from '../services/coding-agent';
import {
  awaitAgentActionResponse,
  emitAgentEvent,
} from '../services/coding-agent/agent-event-bridge';

interface EventRegistryProvider {
  getEventRegistry: () => EventRegistry;
}

const bridgedAgents = new WeakSet<object>();

function isEventRegistryProvider(agent: unknown): agent is EventRegistryProvider {
  return typeof (agent as EventRegistryProvider)?.getEventRegistry === 'function';
}

function sanitizeAgentEvent(event: AgentEvent): AgentEvent {
  if (!event.raw || typeof event.raw !== 'object') {
    return event;
  }
  const raw = event.raw as { toolInput?: unknown; toolUseId?: string };
  return {
    ...event,
    raw: {
      toolInput: raw.toolInput,
      toolUseId: raw.toolUseId,
    },
  };
}

function mapActionResponseToEventResult(
  response: AgentActionResponse,
  event: AgentEvent<PermissionPayload>
): EventResult {
  if (response.type === 'tool_approval') {
    return response.decision === 'allow'
      ? { action: 'allow' }
      : { action: 'deny', message: response.message || 'Permission denied' };
  }
  if (response.type === 'clarifying_question') {
    const toolInput =
      (event.raw as { toolInput?: Record<string, unknown> } | undefined)?.toolInput ?? {};
    return {
      action: 'modify',
      modifiedPayload: {
        ...toolInput,
        answers: response.answers,
      },
    };
  }
  return { action: 'continue' };
}

function ensureAgentEventBridge(agent: unknown): void {
  if (!isEventRegistryProvider(agent)) {
    return;
  }
  const agentObject = agent as object;
  if (bridgedAgents.has(agentObject)) {
    return;
  }
  bridgedAgents.add(agentObject);
  const registry = agent.getEventRegistry();

  registry.on<PermissionPayload>('permission:request', async (event) => {
    emitAgentEvent(sanitizeAgentEvent(event));
    try {
      const response = await awaitAgentActionResponse(
        event.id,
        (event.raw as { signal?: AbortSignal } | undefined)?.signal
      );
      return mapActionResponseToEventResult(response, event);
    } catch (error) {
      return {
        action: 'deny',
        message: error instanceof Error ? error.message : 'Permission request aborted',
      };
    }
  });
}

export function registerCodingAgentHandlers(): void {
  ipcBus.handle(
    'coding-agent:generate',
    async (_event, agentType: CodingAgentType, request: GenerateRequest) => {
      try {
        const agentResult = await getCodingAgent(agentType);
        if (agentResult.success === false) {
          return { success: false, error: agentResult.error.message };
        }
        ensureAgentEventBridge(agentResult.data);
        const result = await agentResult.data.generate(request);
        if (result.success === false) {
          return { success: false, error: result.error.message };
        }
        return { success: true, data: result.data };
      } catch (error) {
        return { success: false, error: (error as Error).message };
      }
    }
  );

  ipcBus.handle(
    'coding-agent:continue-session',
    async (
      _event,
      agentType: CodingAgentType,
      identifier: SessionIdentifier,
      prompt: string,
      options?: ContinueOptions
    ) => {
      try {
        const agentResult = await getCodingAgent(agentType);
        if (agentResult.success === false) {
          return { success: false, error: agentResult.error.message };
        }
        ensureAgentEventBridge(agentResult.data);
        const result = await agentResult.data.continueSession(identifier, prompt, options);
        if (result.success === false) {
          return { success: false, error: result.error.message };
        }
        return { success: true, data: result.data };
      } catch (error) {
        return { success: false, error: (error as Error).message };
      }
    }
  );

  ipcBus.handle(
    'coding-agent:continue-session-streaming',
    async (
      event,
      requestId: string,
      agentType: CodingAgentType,
      identifier: SessionIdentifier,
      prompt: string,
      options?: ContinueOptions
    ) => {
      try {
        const agentResult = await getCodingAgent(agentType);
        if (agentResult.success === false) {
          return { success: false, error: agentResult.error.message };
        }
        const agent = agentResult.data;
        ensureAgentEventBridge(agent);
        const result = await agent.continueSessionStreaming(
          identifier,
          prompt,
          (chunk: string) => {
            event.sender.send('coding-agent:stream-chunk', { requestId, chunk });
          },
          options
        );
        if (result.success === false) {
          return { success: false, error: result.error.message };
        }
        return { success: true, data: result.data };
      } catch (error) {
        return { success: false, error: (error as Error).message };
      }
    }
  );

  ipcBus.handle(
    'coding-agent:fork-session',
    async (_event, agentType: CodingAgentType, options: ForkOptions) => {
      try {
        const agentResult = await getCodingAgent(agentType);
        if (agentResult.success === false) {
          return { success: false, error: agentResult.error.message };
        }
        const agent = agentResult.data;
        ensureAgentEventBridge(agent);
        const result = await agent.forkSession(options);
        if (result.success === false) {
          return { success: false, error: result.error.message };
        }
        return { success: true, data: result.data };
      } catch (error) {
        return { success: false, error: (error as Error).message };
      }
    }
  );

  ipcBus.handle('coding-agent:get-available', async () => {
    const available: CodingAgentType[] = ['claude_code'];
    return { success: true, data: available };
  });

  ipcBus.handle('coding-agent:get-capabilities', async (_event, agentType: CodingAgentType) => {
    try {
      const agentResult = await getCodingAgent(agentType);
      if (agentResult.success === false) {
        return { success: false, error: agentResult.error.message };
      }
      return { success: true, data: agentResult.data.getCapabilities() };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  });

  ipcBus.handle('coding-agent:is-available', async (_event, agentType: CodingAgentType) => {
    try {
      const result = await getCodingAgent(agentType);
      return { success: true, data: result.success };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  });

  ipcBus.handle(
    'coding-agent:list-session-summaries',
    async (_event, agentType: CodingAgentType, filter?: SessionFilterOptions) => {
      try {
        const agentResult = await getCodingAgent(agentType, { skipCliVerification: true });
        if (agentResult.success === false) {
          return { success: false, error: agentResult.error.message };
        }
        const result = await agentResult.data.listSessionSummaries(filter);
        if (result.success === false) {
          return { success: false, error: result.error.message };
        }
        return { success: true, data: result.data };
      } catch (error) {
        return { success: false, error: (error as Error).message };
      }
    }
  );

  ipcBus.handle(
    'coding-agent:get-session',
    async (
      _event,
      agentType: CodingAgentType,
      sessionId: string,
      filter?: MessageFilterOptions
    ) => {
      try {
        const agentResult = await getCodingAgent(agentType, { skipCliVerification: true });
        if (agentResult.success === false) {
          return { success: false, error: agentResult.error.message };
        }
        const result = await agentResult.data.getSession(sessionId, filter);
        if (result.success === false) {
          return { success: false, error: result.error.message };
        }
        return { success: true, data: result.data };
      } catch (error) {
        return { success: false, error: (error as Error).message };
      }
    }
  );

  ipcBus.handle(
    'coding-agent:get-latest-session',
    async (_event, agentType: CodingAgentType, workspacePath: string) => {
      try {
        const agentResult = await getCodingAgent(agentType, { skipCliVerification: true });
        if (agentResult.success === false) {
          return { success: false, error: agentResult.error.message };
        }
        const result = await agentResult.data.listSessionSummaries({ projectPath: workspacePath });
        if (result.success === false) {
          return { success: false, error: result.error.message };
        }
        if (!result.data.length) {
          return { success: true, data: null };
        }
        const latestSession = [...result.data].sort((a, b) => {
          const aTime = new Date(a.updatedAt ?? a.timestamp ?? a.createdAt).getTime();
          const bTime = new Date(b.updatedAt ?? b.timestamp ?? b.createdAt).getTime();
          return bTime - aTime;
        })[0];
        return {
          success: true,
          data: { id: latestSession.id, updatedAt: latestSession.updatedAt },
        };
      } catch (error) {
        return { success: false, error: (error as Error).message };
      }
    }
  );

  ipcBus.handle(
    'coding-agent:session-file-exists',
    async (_event, agentType: CodingAgentType, sessionId: string, workspacePath: string) => {
      try {
        const agentResult = await getCodingAgent(agentType, { skipCliVerification: true });
        if (agentResult.success === false) {
          return { success: false, error: agentResult.error.message };
        }
        const exists = await agentResult.data.sessionFileExists(sessionId, workspacePath);
        return { success: true, data: exists };
      } catch (error) {
        return { success: false, error: (error as Error).message };
      }
    }
  );

  ipcBus.handle(
    'coding-agent:generate-streaming',
    async (event, requestId: string, agentType: CodingAgentType, request: GenerateRequest) => {
      try {
        const agentResult = await getCodingAgent(agentType);
        if (agentResult.success === false) {
          return { success: false, error: agentResult.error.message };
        }
        ensureAgentEventBridge(agentResult.data);
        const result = await agentResult.data.generateStreaming(request, (chunk: string) => {
          event.sender.send('coding-agent:stream-chunk', { requestId, chunk });
        });
        if (result.success === false) {
          return { success: false, error: result.error.message };
        }
        return { success: true, data: result.data };
      } catch (error) {
        return { success: false, error: (error as Error).message };
      }
    }
  );

  ipcBus.handle(
    'coding-agent:generate-streaming-structured',
    async (event, requestId: string, agentType: CodingAgentType, request: GenerateRequest) => {
      try {
        const agentResult = await getCodingAgent(agentType);
        if (agentResult.success === false) {
          return { success: false, error: agentResult.error.message };
        }
        const agent = agentResult.data;
        if (!('generateStreamingStructured' in agent)) {
          return { success: false, error: 'Agent does not support structured streaming' };
        }
        ensureAgentEventBridge(agent);
        const agentWithStructured = agent as typeof agent & {
          generateStreamingStructured: (
            req: GenerateRequest,
            onChunk: (chunk: StreamingChunk) => void
          ) => ReturnType<typeof agent.generateStreaming>;
        };
        const result = await agentWithStructured.generateStreamingStructured(
          request,
          (chunk: StreamingChunk) => {
            event.sender.send('coding-agent:stream-chunk-structured', { requestId, chunk });
          }
        );
        if (result.success === false) {
          return { success: false, error: result.error.message };
        }
        return { success: true, data: result.data };
      } catch (error) {
        return { success: false, error: (error as Error).message };
      }
    }
  );

  ipcBus.handle('coding-agent:abort', async (_event, agentType: CodingAgentType) => {
    try {
      const agentResult = await getCodingAgent(agentType);
      if (agentResult.success === false) {
        return { success: false, error: agentResult.error.message };
      }
      await agentResult.data.cancelAll();
      return { success: true };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  });
}
