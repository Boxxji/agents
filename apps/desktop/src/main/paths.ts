import { execFileSync } from 'node:child_process';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';

const APP_IDENTIFIER = 'ai.hanzo.agents';
const APP_NAME = 'HanzoAgents';

export function getHomePath(): string {
  return process.env.HOME || os.homedir();
}

export function getAppName(): string {
  return APP_NAME;
}

export function getUserDataPath(): string {
  const explicit = process.env.HANZO_AGENTS_USER_DATA?.trim();
  if (explicit) {
    fs.mkdirSync(explicit, { recursive: true });
    return explicit;
  }

  const home = getHomePath();
  let base: string;
  if (process.platform === 'darwin') {
    base = path.join(home, 'Library', 'Application Support', APP_IDENTIFIER);
  } else if (process.platform === 'win32') {
    base = path.join(process.env.APPDATA || path.join(home, 'AppData', 'Roaming'), APP_IDENTIFIER);
  } else {
    base = path.join(
      process.env.XDG_DATA_HOME || path.join(home, '.local', 'share'),
      APP_IDENTIFIER
    );
  }

  fs.mkdirSync(base, { recursive: true });
  return base;
}

/**
 * Fix PATH when launched from macOS Finder/Dock where login shell PATH is absent.
 */
export function fixPath(): void {
  if (process.platform !== 'darwin') return;
  if (process.env.NODE_ENV === 'development') return;

  try {
    const userShell = process.env.SHELL || '/bin/zsh';
    const result = execFileSync(userShell, ['-ilc', 'echo $PATH'], {
      encoding: 'utf8',
      timeout: 5000,
    }).trim();

    if (result) {
      process.env.PATH = result;
    }
  } catch {
    const home = getHomePath();
    const commonPaths = [
      '/usr/local/bin',
      '/opt/homebrew/bin',
      '/opt/homebrew/sbin',
      `${home}/.local/bin`,
      `${home}/.hanzo/bin`,
      `${home}/.cargo/bin`,
    ];
    const currentPath = process.env.PATH || '';
    process.env.PATH = [...commonPaths, currentPath].join(':');
  }
}
