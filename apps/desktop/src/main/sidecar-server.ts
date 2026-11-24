import * as http from 'node:http';
import { ipcBus } from './ipc-bus';
import { getHomePath } from './paths';

export interface SidecarServerOptions {
  port: number;
  host?: string;
  token?: string;
}

function readJsonBody(req: http.IncomingMessage): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on('data', (chunk: Buffer) => chunks.push(chunk));
    req.on('end', () => {
      const raw = Buffer.concat(chunks).toString('utf8').trim();
      if (!raw) {
        resolve({});
        return;
      }
      try {
        resolve(JSON.parse(raw));
      } catch (err) {
        reject(err);
      }
    });
    req.on('error', reject);
  });
}

function isAuthorized(req: http.IncomingMessage, url: URL, expectedToken?: string): boolean {
  if (!expectedToken) return true;
  const auth = req.headers.authorization;
  if (auth === `Bearer ${expectedToken}`) return true;
  if (url.searchParams.get('token') === expectedToken) return true;
  return false;
}

export class SidecarRpcServer {
  private server: http.Server | null = null;
  private unsubscribeBroadcast: (() => void) | null = null;
  private readonly sseClients = new Set<http.ServerResponse>();

  constructor(private readonly options: SidecarServerOptions) {}

  async start(): Promise<number> {
    const host = this.options.host || '127.0.0.1';
    const expectedToken = this.options.token?.trim() || undefined;

    this.unsubscribeBroadcast = ipcBus.subscribeBroadcast((channel, payload) => {
      const frame = `data: ${JSON.stringify({ channel, payload })}\n\n`;
      for (const client of this.sseClients) {
        try {
          client.write(frame);
        } catch {
          this.sseClients.delete(client);
        }
      }
    });

    this.server = http.createServer(async (req, res) => {
      res.setHeader('Access-Control-Allow-Origin', '*');
      res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
      res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

      if (req.method === 'OPTIONS') {
        res.writeHead(204);
        res.end();
        return;
      }

      const url = new URL(req.url || '/', `http://${host}:${this.options.port}`);

      if (url.pathname === '/healthz') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ ok: true, homeDir: getHomePath() }));
        return;
      }

      if (!isAuthorized(req, url, expectedToken)) {
        res.writeHead(401, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ ok: false, error: 'Unauthorized' }));
        return;
      }

      if (req.method === 'GET' && url.pathname === '/rpc/events') {
        res.writeHead(200, {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache, no-transform',
          Connection: 'keep-alive',
        });
        res.write(': connected\n\n');
        this.sseClients.add(res);
        req.on('close', () => {
          this.sseClients.delete(res);
        });
        return;
      }

      if (req.method === 'POST' && url.pathname === '/rpc/invoke') {
        try {
          const body = (await readJsonBody(req)) as { channel?: string; args?: unknown[] };
          if (!body.channel || typeof body.channel !== 'string') {
            res.writeHead(400, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ ok: false, error: 'Missing channel' }));
            return;
          }
          const result = await ipcBus.invoke(
            body.channel,
            Array.isArray(body.args) ? body.args : []
          );
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ ok: true, result }));
        } catch (error) {
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(
            JSON.stringify({
              ok: false,
              error: error instanceof Error ? error.message : String(error),
            })
          );
        }
        return;
      }

      if (req.method === 'POST' && url.pathname === '/rpc/send') {
        try {
          const body = (await readJsonBody(req)) as { channel?: string; args?: unknown[] };
          if (!body.channel || typeof body.channel !== 'string') {
            res.writeHead(400, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ ok: false, error: 'Missing channel' }));
            return;
          }
          ipcBus.emitFromClient(body.channel, Array.isArray(body.args) ? body.args : []);
          res.writeHead(204);
          res.end();
        } catch (error) {
          res.writeHead(500, { 'Content-Type': 'application/json' });
          res.end(
            JSON.stringify({
              ok: false,
              error: error instanceof Error ? error.message : String(error),
            })
          );
        }
        return;
      }

      res.writeHead(404, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ ok: false, error: 'Not Found' }));
    });

    const boundPort = await new Promise<number>((resolve, reject) => {
      this.server!.once('error', reject);
      this.server!.listen(this.options.port, host, () => {
        const addr = this.server!.address();
        resolve(typeof addr === 'object' && addr ? addr.port : this.options.port);
      });
    });

    console.log(`[Sidecar] Hanzo Agents RPC server listening on http://${host}:${boundPort}`);
    return boundPort;
  }

  async stop(): Promise<void> {
    this.unsubscribeBroadcast?.();
    this.unsubscribeBroadcast = null;
    for (const client of this.sseClients) {
      try {
        client.end();
      } catch {
        // Ignore close errors
      }
    }
    this.sseClients.clear();
    if (this.server) {
      await new Promise<void>((resolve) => {
        this.server!.close(() => resolve());
      });
      this.server = null;
    }
  }
}
