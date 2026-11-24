#!/usr/bin/env node
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const sidecarEntry = path.join(__dirname, '..', 'dist', 'main', 'src', 'main', 'main.js');

async function runTest() {
  console.log('Starting Hanzo Agents sidecar MCP server test...\n');

  const child = spawn(process.execPath, [sidecarEntry, '--mcp'], {
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, HANZO_AGENTS_PORT: '13991' },
  });

  child.stdout.on('data', (d) => process.stdout.write(d));
  child.stderr.on('data', (d) => process.stderr.write(d));

  await new Promise((resolve) => setTimeout(resolve, 3000));

  const res = await fetch('http://127.0.0.1:13991/healthz');
  console.log('Sidecar healthz:', res.status, await res.json());

  child.kill('SIGTERM');
  process.exit(0);
}

runTest().catch((err) => {
  console.error(err);
  process.exit(1);
});
