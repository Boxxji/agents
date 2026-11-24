use serde::Deserialize;
use std::path::PathBuf;
use tauri::{AppHandle, Manager};
use tauri_plugin_dialog::DialogExt;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenDirectoryOptions {
    pub title: Option<String>,
    pub default_path: Option<String>,
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
