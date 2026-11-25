use crate::AppState;
use serde::Deserialize;
use serde_json::Value;
use std::path::PathBuf;
use tauri::{AppHandle, Manager, State};
use tauri_plugin_dialog::DialogExt;

pub const ALLOWED_INVOKE_CHANNELS: &[&str] = &[
    // Terminal session
    "terminal-get-session-state",
    "terminal-set-session-state",
    "terminal-clear-session-state",
    "terminal-get-buffer",
    // Canvas
    "canvas:save",
    "canvas:load",
    "canvas:list",
    "canvas:delete",
    "canvas:get-current-id",
    "canvas:set-current-id",
    // Worktree
    "worktree:provision",
    "worktree:release",
    "worktree:get",
    "worktree:list",
    // Agent status
    "agent-status:save",
    "agent-status:load",
    "agent-status:delete",
    "agent-status:load-all",
    // Coding agent
    "coding-agent:generate",
    "coding-agent:generate-streaming",
    "coding-agent:generate-streaming-structured",
    "coding-agent:continue-session",
    "coding-agent:continue-session-streaming",
    "coding-agent:fork-session",
    "coding-agent:get-available",
    "coding-agent:get-capabilities",
    "coding-agent:is-available",
    "coding-agent:list-session-summaries",
    "coding-agent:get-session",
    "coding-agent:respond-to-action",
    "coding-agent:get-latest-session",
    "coding-agent:session-file-exists",
    "coding-agent:abort",
    // LLM
    "llm:chat",
    "llm:chat-stream",
    "llm:chat-with-tools",
    "llm:set-api-key",
    "llm:delete-api-key",
    "llm:has-api-key",
    "llm:list-vendors-with-keys",
    "llm:get-available-models",
    "llm:is-configured",
    "llm:get-capabilities",
    // Representation
    "representation:get-available-types",
    "representation:transform",
    "representation:transform-to-image",
    "representation:transform-to-summary",
    "representation:transform-to-audio",
    "representation:get-all-providers",
    // Shell
    "shell:open-with-editor",
    "shell:get-available-editors",
    "shell:show-in-folder",
    "shell:open-directory-dialog",
    // File
    "file:read",
    "file:exists",
    // Git
    "git:get-info-strict",
    "git:list-branches",
    "git:create-branch",
    "git:checkout-branch",
    "git:get-github-username",
    // Session watcher
    "session-watcher:watch",
    "session-watcher:unwatch",
    // Session summary cache
    "session-summary:get",
    "session-summary:save",
    "session-summary:is-stale",
    "session-summary:delete",
    // Recent workspaces
    "recent-workspaces:add",
    "recent-workspaces:get",
    "recent-workspaces:remove",
    "recent-workspaces:clear",
    "recent-workspaces:has",
];

pub const ALLOWED_SEND_CHANNELS: &[&str] = &[
    "terminal-create",
    "terminal-input",
    "terminal-resize",
    "terminal-destroy",
    "mcp-bridge-response",
];

pub fn is_allowed_invoke_channel(channel: &str) -> bool {
    ALLOWED_INVOKE_CHANNELS.contains(&channel)
}

pub fn is_allowed_send_channel(channel: &str) -> bool {
    ALLOWED_SEND_CHANNELS.contains(&channel)
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenDirectoryOptions {
    pub title: Option<String>,
    pub default_path: Option<String>,
}

#[tauri::command]
pub async fn sidecar_rpc_invoke(
    state: State<'_, AppState>,
    channel: String,
    args: Option<Vec<Value>>,
) -> Result<Value, String> {
    if !is_allowed_invoke_channel(&channel) {
        return Err(format!("Disallowed RPC invoke channel: {channel}"));
    }
    state
        .client
        .invoke(&channel, args.unwrap_or_default())
        .await
}

#[tauri::command]
pub async fn sidecar_rpc_send(
    state: State<'_, AppState>,
    channel: String,
    args: Option<Vec<Value>>,
) -> Result<(), String> {
    if !is_allowed_send_channel(&channel) {
        return Err(format!("Disallowed RPC send channel: {channel}"));
    }
    state.client.send(&channel, args.unwrap_or_default()).await
}

#[tauri::command]
pub fn desktop_window_minimize(app: AppHandle) -> Result<(), String> {
    if let Some(win) = app.get_webview_window("main") {
        win.minimize().map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[tauri::command]
pub fn desktop_window_toggle_maximize(app: AppHandle) -> Result<(), String> {
    if let Some(win) = app.get_webview_window("main") {
        if win.is_maximized().unwrap_or(false) {
            win.unmaximize().map_err(|e| e.to_string())?;
        } else {
            win.maximize().map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

#[tauri::command]
pub fn desktop_window_close(app: AppHandle) -> Result<(), String> {
    if let Some(win) = app.get_webview_window("main") {
        win.close().map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[tauri::command]
pub fn desktop_window_is_maximized(app: AppHandle) -> Result<bool, String> {
    if let Some(win) = app.get_webview_window("main") {
        return win.is_maximized().map_err(|e| e.to_string());
    }
    Ok(false)
}

#[tauri::command]
pub async fn desktop_open_directory_dialog(
    app: AppHandle,
    options: Option<OpenDirectoryOptions>,
) -> Result<Option<String>, String> {
    tokio::task::spawn_blocking(move || {
        let mut dialog = app.dialog().file();
        if let Some(ref opts) = options {
            if let Some(ref title) = opts.title {
                dialog = dialog.set_title(title);
            }
            if let Some(ref dir) = opts.default_path {
                let p = PathBuf::from(dir);
                if p.is_dir() {
                    dialog = dialog.set_directory(p);
                }
            }
        }
        Ok(dialog
            .blocking_pick_folder()
            .and_then(|fp| fp.into_path().ok())
            .map(|p| p.to_string_lossy().into_owned()))
    })
    .await
    .map_err(|e| e.to_string())?
}
