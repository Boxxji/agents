export interface RpcInvokeContext {
  sender: {
    send: (channel: string, payload: unknown) => void;
  };
}

export type RpcHandler = (context: RpcInvokeContext, ...args: any[]) => Promise<unknown> | unknown;
export type RpcEventListener = (context: RpcInvokeContext, ...args: any[]) => void;
export type BroadcastSubscriber = (channel: string, payload: unknown) => void;

class SidecarIpcBus {
  private readonly handlers = new Map<string, RpcHandler>();
  private readonly listeners = new Map<string, Set<RpcEventListener>>();
  private readonly broadcastSubscribers = new Set<BroadcastSubscriber>();

  private readonly context: RpcInvokeContext = {
    sender: {
      send: (channel: string, payload: unknown) => {
        this.broadcast(channel, payload);
      },
    },
  };

  handle(channel: string, handler: RpcHandler): void {
    this.handlers.set(channel, handler);
  }

  on(channel: string, listener: RpcEventListener): void {
    let set = this.listeners.get(channel);
    if (!set) {
      set = new Set();
      this.listeners.set(channel, set);
    }
    set.add(listener);
  }

  removeAllListeners(channel: string): void {
    this.listeners.delete(channel);
  }

  async invoke(channel: string, args: unknown[] = []): Promise<unknown> {
    const handler = this.handlers.get(channel);
    if (!handler) {
      throw new Error(`No RPC handler registered for channel: ${channel}`);
    }
    return await handler(this.context, ...args);
  }

  emitFromClient(channel: string, args: unknown[] = []): void {
    const set = this.listeners.get(channel);
    if (!set) return;
    for (const listener of set) {
      try {
        listener(this.context, ...args);
      } catch (error) {
        console.error(`[IpcBus] Error in listener for ${channel}:`, error);
      }
    }
  }

  broadcast(channel: string, payload: unknown): void {
    for (const sub of this.broadcastSubscribers) {
      try {
        sub(channel, payload);
      } catch (error) {
        console.error(`[IpcBus] Error broadcasting ${channel}:`, error);
      }
    }
  }

  subscribeBroadcast(subscriber: BroadcastSubscriber): () => void {
    this.broadcastSubscribers.add(subscriber);
    return () => {
      this.broadcastSubscribers.delete(subscriber);
    };
  }
}

export const ipcBus = new SidecarIpcBus();
