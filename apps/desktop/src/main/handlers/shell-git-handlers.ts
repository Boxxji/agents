import { spawn } from 'node:child_process';
import * as fs from 'node:fs';
import * as path from 'node:path';
import type { GitInfo } from '@hanzo/agents-shared';
import { ipcBus } from '../ipc-bus';
import { getHomePath } from '../paths';
import { gitBranchService } from '../services/git';

type EditorApp = 'vscode' | 'cursor' | 'zed' | 'sublime' | 'atom' | 'webstorm' | 'finder';

const EDITOR_COMMANDS: Record<
  EditorApp,
  { app?: string; command?: string; args: (dir: string) => string[] }
> = {
  vscode: { command: 'code', args: (dir) => [dir] },
  cursor: { command: 'cursor', args: (dir) => [dir] },
  zed: { command: 'zed', args: (dir) => [dir] },
  sublime: { command: 'subl', args: (dir) => [dir] },
  atom: { command: 'atom', args: (dir) => [dir] },
  webstorm: { app: 'WebStorm', command: 'webstorm', args: (dir) => [dir] },
  finder: {
    command:
      process.platform === 'darwin'
        ? 'open'
        : process.platform === 'win32'
          ? 'explorer'
          : 'xdg-open',
    args: (dir) => [dir],
  },
};

function commandExists(command: string): Promise<boolean> {
  const probe = process.platform === 'win32' ? 'where' : 'which';
  return new Promise((resolve) => {
    const child = spawn(probe, [command]);
    child.on('close', (code) => resolve(code === 0));
    child.on('error', () => resolve(false));
  });
}

function appExists(appName: string): Promise<boolean> {
  if (process.platform !== 'darwin') return Promise.resolve(false);
  const appPath = `/Applications/${appName}.app`;
  return new Promise((resolve) => {
    fs.access(appPath, fs.constants.F_OK, (err) => resolve(!err));
  });
}

function openInSystemFileManager(targetPath: string, revealItem = false): void {
  if (process.platform === 'darwin') {
    const args = revealItem ? ['-R', targetPath] : [targetPath];
    spawn('open', args, { detached: true, stdio: 'ignore' }).unref();
    return;
  }
  if (process.platform === 'win32') {
    const args = revealItem ? [`/select,${targetPath}`] : [targetPath];
    spawn('explorer', args, { detached: true, stdio: 'ignore' }).unref();
    return;
  }
  const dir =
    revealItem && fs.existsSync(targetPath) && fs.statSync(targetPath).isFile()
      ? path.dirname(targetPath)
      : targetPath;
  spawn('xdg-open', [dir], { detached: true, stdio: 'ignore' }).unref();
}

function runGitCommand(cwd: string, args: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    const git = spawn('git', args, { cwd });
    let stdout = '';
    let stderr = '';
    git.stdout.on('data', (d) => {
      stdout += d.toString();
    });
    git.stderr.on('data', (d) => {
      stderr += d.toString();
    });
    git.on('close', (code) => {
      if (code === 0) {
        resolve(stdout.trim());
      } else {
        reject(new Error(stderr.trim() || `git command failed with code ${code}`));
      }
    });
    git.on('error', (err) => reject(err));
  });
}

export function registerShellAndGitHandlers(): void {
  ipcBus.handle(
    'shell:open-with-editor',
    async (_event, directoryPath: string, editor: EditorApp) => {
      try {
        if (!fs.existsSync(directoryPath)) {
          return { success: false, error: `Directory does not exist: ${directoryPath}` };
        }
        const config = EDITOR_COMMANDS[editor];
        if (!config) {
          return { success: false, error: `Unknown editor: ${editor}` };
        }
        if (editor === 'finder') {
          openInSystemFileManager(directoryPath, false);
          return { success: true };
        }
        if (config.command && (await commandExists(config.command))) {
          spawn(config.command, config.args(directoryPath), {
            detached: true,
            stdio: 'ignore',
          }).unref();
          return { success: true };
        }
        if (config.app && (await appExists(config.app))) {
          spawn('open', ['-a', config.app, directoryPath], {
            detached: true,
            stdio: 'ignore',
          }).unref();
          return { success: true };
        }
        return { success: false, error: `Editor ${editor} is not installed or not found in PATH` };
      } catch (error) {
        return { success: false, error: (error as Error).message };
      }
    }
  );

  ipcBus.handle('shell:get-available-editors', async () => {
    try {
      const available: EditorApp[] = [];
      for (const [editor, config] of Object.entries(EDITOR_COMMANDS) as [
        EditorApp,
        (typeof EDITOR_COMMANDS)[EditorApp],
      ][]) {
        if (config.command && (await commandExists(config.command))) {
          available.push(editor);
          continue;
        }
        if (config.app && (await appExists(config.app))) {
          available.push(editor);
        }
      }
      if (!available.includes('finder')) {
        available.push('finder');
      }
      return { success: true, data: available };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  });

  ipcBus.handle('shell:show-in-folder', async (_event, filePath: string) => {
    try {
      openInSystemFileManager(filePath, true);
      return { success: true };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  });

  ipcBus.handle(
    'shell:open-directory-dialog',
    async (_event, options?: { title?: string; defaultPath?: string }) => {
      // Native Tauri dialog handles this directly in the renderer shim when running inside Tauri;
      // provide a fallback using zenity/osascript when invoked directly on the sidecar.
      const defaultPath = options?.defaultPath || getHomePath();
      try {
        if (process.platform === 'darwin') {
          const prompt = (options?.title || 'Select Directory').replace(/"/g, '\\"');
          const dir = await new Promise<string | null>((resolve) => {
            const osa = spawn('osascript', [
              '-e',
              `POSIX path of (choose folder with prompt "${prompt}")`,
            ]);
            let out = '';
            osa.stdout.on('data', (d) => {
              out += d.toString();
            });
            osa.on('close', (code) => resolve(code === 0 ? out.trim() || null : null));
            osa.on('error', () => resolve(null));
          });
          return { success: true, data: dir };
        }
        if (process.platform === 'linux' && (await commandExists('zenity'))) {
          const dir = await new Promise<string | null>((resolve) => {
            const z = spawn('zenity', [
              '--file-selection',
              '--directory',
              `--title=${options?.title || 'Select Directory'}`,
              `--filename=${defaultPath}/`,
            ]);
            let out = '';
            z.stdout.on('data', (d) => {
              out += d.toString();
            });
            z.on('close', (code) => resolve(code === 0 ? out.trim() || null : null));
            z.on('error', () => resolve(null));
          });
          return { success: true, data: dir };
        }
        return { success: true, data: null };
      } catch (error) {
        return { success: false, error: (error as Error).message };
      }
    }
  );

  ipcBus.handle(
    'git:list-branches',
    async (
      _event,
      workspacePath: string
    ): Promise<{ success: boolean; data?: string[]; error?: string }> => {
      try {
        if (!fs.existsSync(workspacePath)) {
          return { success: false, error: `Path does not exist: ${workspacePath}` };
        }
        try {
          const branchesOutput = await runGitCommand(workspacePath, [
            'branch',
            '--format=%(refname:short)',
          ]);
          const branches = branchesOutput
            .split('\n')
            .map((b) => b.trim())
            .filter((b) => b.length > 0);
          return { success: true, data: branches };
        } catch {
          return { success: false, error: 'Not a git repository' };
        }
      } catch (error) {
        return { success: false, error: (error as Error).message };
      }
    }
  );

  ipcBus.handle('git:get-info-strict', async (_event, workspacePath: string): Promise<GitInfo> => {
    return gitBranchService.getGitInfo(workspacePath);
  });

  ipcBus.handle(
    'git:create-branch',
    async (
      _event,
      workspacePath: string,
      branchName: string
    ): Promise<{ success: boolean; error?: string }> => {
      try {
        if (!fs.existsSync(workspacePath)) {
          return { success: false, error: `Path does not exist: ${workspacePath}` };
        }
        if (!branchName || !branchName.trim()) {
          return { success: false, error: 'Branch name is required' };
        }
        const sanitizedBranchName = branchName.trim().replace(/[^a-zA-Z0-9._/-]/g, '-');
        if (!sanitizedBranchName) {
          return { success: false, error: 'Invalid branch name' };
        }
        try {
          await runGitCommand(workspacePath, ['checkout', '-b', sanitizedBranchName]);
          return { success: true };
        } catch (error) {
          const errorMessage = (error as Error).message;
          if (errorMessage.includes('already exists')) {
            try {
              await runGitCommand(workspacePath, ['checkout', sanitizedBranchName]);
              return { success: true };
            } catch (checkoutError) {
              return { success: false, error: (checkoutError as Error).message };
            }
          }
          return { success: false, error: errorMessage };
        }
      } catch (error) {
        return { success: false, error: (error as Error).message };
      }
    }
  );

  ipcBus.handle(
    'git:checkout-branch',
    async (
      _event,
      workspacePath: string,
      branchName: string
    ): Promise<{ success: boolean; error?: string }> => {
      try {
        if (!fs.existsSync(workspacePath)) {
          return { success: false, error: `Path does not exist: ${workspacePath}` };
        }
        if (!branchName || !branchName.trim()) {
          return { success: false, error: 'Branch name is required' };
        }
        await runGitCommand(workspacePath, ['checkout', branchName.trim()]);
        return { success: true };
      } catch (error) {
        return { success: false, error: (error as Error).message };
      }
    }
  );

  ipcBus.handle(
    'git:get-github-username',
    async (): Promise<{ success: boolean; data?: { username: string }; error?: string }> => {
      try {
        const gh = spawn('gh', ['api', 'user', '--jq', '.login']);
        let stdout = '';
        let stderr = '';
        gh.stdout.on('data', (d) => {
          stdout += d.toString();
        });
        gh.stderr.on('data', (d) => {
          stderr += d.toString();
        });
        const username = await new Promise<string>((resolve, reject) => {
          gh.on('close', (code) => {
            if (code === 0 && stdout.trim()) {
              resolve(stdout.trim());
            } else {
              reject(new Error(stderr.trim() || `GitHub CLI failed with code ${code}`));
            }
          });
          gh.on('error', (err) =>
            reject(new Error(`Failed to execute GitHub CLI: ${err.message}`))
          );
        });
        return { success: true, data: { username } };
      } catch (error) {
        return { success: false, error: (error as Error).message };
      }
    }
  );
}
