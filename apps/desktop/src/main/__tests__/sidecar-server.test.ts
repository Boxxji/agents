import * as http from 'node:http';
import * as os from 'node:os';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { ipcBus } from '../ipc-bus';
import { SidecarRpcServer } from '../sidecar-server';

const TEST_TOKEN = '0123456789abcdef0123456789abcdef0123456789abcdef';

interface RawHttpResponse {
  status: number;
  headers: http.IncomingHttpHeaders;
  body: string;
}

function rawRequest(options: {
  port: number;
  method: string;
  path: string;
  headers?: Record<string, string>;
  body?: string;
}): Promise<RawHttpResponse> {
  return new Promise((resolve, reject) => {
    const req = http.request(
      {
        hostname: '127.0.0.1',
        port: options.port,
        method: options.method,
        path: options.path,
        headers: options.headers,
      },
      (res) => {
        const chunks: Buffer[] = [];
        res.on('data', (chunk: Buffer) => chunks.push(chunk));
        res.on('end', () => {
          resolve({
            status: res.statusCode ?? 0,
            headers: res.headers,
            body: Buffer.concat(chunks).toString('utf8'),
          });
        });
      }
    );
    req.on('error', reject);
    if (options.body) {
      req.write(options.body);
    }
    req.end();
  });
}

describe('SidecarRpcServer security', () => {
  let server: SidecarRpcServer;
  let port: number;

  beforeEach(async () => {
    server = new SidecarRpcServer({ port: 0, host: '127.0.0.1', token: TEST_TOKEN });
    port = await server.start();
    ipcBus.handle('test:echo', (_ctx, value: string) => ({ echoed: value }));
  });

  afterEach(async () => {
    ipcBus.removeAllListeners('test:notify');
    await server.stop();
  });

  it('refuses to start without a non-empty auth token', async () => {
    const unauthenticated = new SidecarRpcServer({ port: 0, host: '127.0.0.1', token: '   ' });
    await expect(unauthenticated.start()).rejects.toThrow(/token/i);
  });

  it('never emits CORS headers and rejects OPTIONS preflight requests', async () => {
    const res = await rawRequest({
      port,
      method: 'OPTIONS',
      path: '/rpc/invoke',
      headers: {
        Origin: 'https://evil.example',
        'Access-Control-Request-Method': 'POST',
      },
    });
    expect(res.status).toBe(403);
    expect(res.headers['access-control-allow-origin']).toBeUndefined();
    expect(res.headers['access-control-allow-methods']).toBeUndefined();
    expect(res.headers['access-control-allow-headers']).toBeUndefined();
  });

  it('rejects cross-origin requests even when carrying a valid bearer token', async () => {
    const res = await rawRequest({
      port,
      method: 'POST',
      path: '/rpc/invoke',
      headers: {
        Origin: 'https://evil.example',
        Authorization: `Bearer ${TEST_TOKEN}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ channel: 'test:echo', args: ['pwn'] }),
    });
    expect(res.status).toBe(403);
    expect(res.headers['access-control-allow-origin']).toBeUndefined();
  });

  it('rejects DNS-rebinding non-loopback Host headers even with a valid token', async () => {
    const res = await rawRequest({
      port,
      method: 'GET',
      path: '/healthz',
      headers: {
        Host: 'attacker.example:13880',
        Authorization: `Bearer ${TEST_TOKEN}`,
      },
    });
    expect(res.status).toBe(403);
    expect(res.headers['access-control-allow-origin']).toBeUndefined();
  });

  it('requires authentication on /healthz and never leaks homeDir', async () => {
    const unauth = await rawRequest({
      port,
      method: 'GET',
      path: '/healthz',
    });
    expect(unauth.status).toBe(401);
    expect(unauth.headers['access-control-allow-origin']).toBeUndefined();
    expect(unauth.body).not.toContain('homeDir');
    expect(unauth.body).not.toContain(os.homedir());

    const queryToken = await rawRequest({
      port,
      method: 'GET',
      path: `/healthz?token=${TEST_TOKEN}`,
    });
    expect(queryToken.status).toBe(401);

    const wrongToken = await rawRequest({
      port,
      method: 'GET',
      path: '/healthz',
      headers: { Authorization: 'Bearer wrong-token' },
    });
    expect(wrongToken.status).toBe(401);

    const authed = await rawRequest({
      port,
      method: 'GET',
      path: '/healthz',
      headers: { Authorization: `Bearer ${TEST_TOKEN}` },
    });
    expect(authed.status).toBe(200);
    expect(authed.headers['access-control-allow-origin']).toBeUndefined();
    const json = JSON.parse(authed.body);
    expect(json).toEqual({ ok: true });
    expect(json.homeDir).toBeUndefined();
  });

  it('protects /rpc/invoke with Authorization header only and invokes handlers when authorized', async () => {
    const noAuth = await rawRequest({
      port,
      method: 'POST',
      path: '/rpc/invoke',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ channel: 'test:echo', args: ['hello'] }),
    });
    expect(noAuth.status).toBe(401);

    const queryAuth = await rawRequest({
      port,
      method: 'POST',
      path: `/rpc/invoke?token=${TEST_TOKEN}`,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ channel: 'test:echo', args: ['hello'] }),
    });
    expect(queryAuth.status).toBe(401);

    const authed = await rawRequest({
      port,
      method: 'POST',
      path: '/rpc/invoke',
      headers: {
        Authorization: `Bearer ${TEST_TOKEN}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ channel: 'test:echo', args: ['hello'] }),
    });
    expect(authed.status).toBe(200);
    expect(authed.headers['access-control-allow-origin']).toBeUndefined();
    expect(JSON.parse(authed.body)).toEqual({ ok: true, result: { echoed: 'hello' } });
  });

  it('protects /rpc/send with Authorization header and dispatches events when authorized', async () => {
    const received: unknown[] = [];
    ipcBus.on('test:notify', (_ctx, payload: unknown) => received.push(payload));

    const noAuth = await rawRequest({
      port,
      method: 'POST',
      path: '/rpc/send',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ channel: 'test:notify', args: [{ n: 1 }] }),
    });
    expect(noAuth.status).toBe(401);
    expect(received).toHaveLength(0);

    const authed = await rawRequest({
      port,
      method: 'POST',
      path: '/rpc/send',
      headers: {
        Authorization: `Bearer ${TEST_TOKEN}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ channel: 'test:notify', args: [{ n: 2 }] }),
    });
    expect(authed.status).toBe(204);
    expect(received).toEqual([{ n: 2 }]);
  });

  it('rejects query-string tokens on /rpc/events and streams events when Authorization header is valid', async () => {
    const queryAuth = await rawRequest({
      port,
      method: 'GET',
      path: `/rpc/events?token=${TEST_TOKEN}`,
    });
    expect(queryAuth.status).toBe(401);

    const receivedFrame = await new Promise<string>((resolve, reject) => {
      const req = http.request(
        {
          hostname: '127.0.0.1',
          port,
          method: 'GET',
          path: '/rpc/events',
          headers: { Authorization: `Bearer ${TEST_TOKEN}` },
        },
        (res) => {
          expect(res.statusCode).toBe(200);
          expect(res.headers['access-control-allow-origin']).toBeUndefined();
          let buf = '';
          res.on('data', (chunk: Buffer) => {
            buf += chunk.toString('utf8');
            if (buf.includes(': connected') && !buf.includes('data:')) {
              ipcBus.broadcast('terminal-data', { terminalId: 't1', data: 'ok' });
            }
            if (buf.includes('"terminal-data"')) {
              req.destroy();
              resolve(buf);
            }
          });
        }
      );
      req.on('error', (err: NodeJS.ErrnoException) => {
        if (err.code !== 'ECONNRESET') reject(err);
      });
      req.end();
    });

    expect(receivedFrame).toContain(
      'data: {"channel":"terminal-data","payload":{"terminalId":"t1","data":"ok"}}'
    );
  });
});
