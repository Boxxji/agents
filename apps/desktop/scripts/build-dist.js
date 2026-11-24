#!/usr/bin/env node
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const APP_DIR = path.resolve(__dirname, '..');
const MONOREPO_ROOT = path.resolve(APP_DIR, '../..');

function run(command, options = {}) {
  const result = spawnSync(command, {
    cwd: APP_DIR,
    stdio: 'inherit',
    shell: true,
    ...options,
  });
  if (result.status !== 0) {
    throw new Error(`Command failed: ${command}`);
  }
}

function main() {
  console.log('\n[1/4] Building @hanzo/agents-shared...');
  run('pnpm --filter @hanzo/agents-shared build', { cwd: MONOREPO_ROOT });

  console.log('\n[2/4] Building Vite renderer...');
  run('pnpm run build');

  console.log('\n[3/4] Compiling Node orchestration sidecar...');
  run('pnpm run build:main');

  console.log('\n[4/4] Building Rust Tauri v2 binary...');
  run('cargo build --release --manifest-path src-tauri/Cargo.toml');

  console.log('\nHanzo Agents Tauri v2 build complete.');
}

main();
