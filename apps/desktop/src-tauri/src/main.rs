#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod bridge;
mod bridge_apis;
mod commands;
mod sidecar;

use crate::bridge::build_initialization_script;
use crate::sidecar::{
    generate_random_token, pick_available_port, SidecarClient, SidecarRuntimeConfig,
    SidecarSupervisor,
};
use std::env;
use tauri::{Manager, RunEvent, WebviewUrl, WebviewWindowBuilder};

pub(crate) struct AppState {
    pub supervisor: SidecarSupervisor,
    pub client: SidecarClient,
}

#[cfg(target_os = "linux")]
fn configure_linux_webkit_env() {
    if std::env::var_os("WEBKIT_DISABLE_DMABUF_RENDERER").is_none() {
        std::env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
    }
    if std::env::var_os("WEBKIT_DISABLE_COMPOSITING_MODE").is_none() {
        std::env::set_var("WEBKIT_DISABLE_COMPOSITING_MODE", "1");
    }
}

fn main() {
    #[cfg(target_os = "linux")]
    configure_linux_webkit_env();

    let dev_vite_url = env::var("VITE_DEV_SERVER_URL")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    let preferred_port = env::var("HANZO_AGENTS_PORT")
        .ok()
        .and_then(|s| s.trim().parse::<u16>().ok())
        .unwrap_or(13880);
    let port = pick_available_port(preferred_port);
    let token = generate_random_token(24);
    let home_dir = dirs::home_dir()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| "/".to_string());
    let mcp_mode = env::args().any(|a| a == "--mcp")
        || env::var("HANZO_AGENTS_MCP")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

    let init_script = build_initialization_script(&home_dir);

    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(move |app| {
            let user_data_dir = app
                .path()
                .app_data_dir()
                .unwrap_or_else(|_| dirs::data_local_dir().unwrap_or_default().join("ai.hanzo.agents"));

            let supervisor = SidecarSupervisor::start(SidecarRuntimeConfig {
                port,
                token: token.clone(),
                user_data_dir,
                mcp_mode,
            });
            let client = SidecarClient::new(port, token.clone());
            client.spawn_event_relay(app.handle().clone(), supervisor.shutdown_flag());
            app.manage(AppState { supervisor, client });

            let webview_url = match &dev_vite_url {
                Some(raw) => WebviewUrl::External(raw.parse()?),
                None => WebviewUrl::App("index.html".into()),
            };

            let win_builder = WebviewWindowBuilder::new(app, "main", webview_url)
                .title("Hanzo Agents")
                .inner_size(1200.0, 800.0)
                .min_inner_size(800.0, 600.0)
                .decorations(false)
                .resizable(true)
                .center()
                .initialization_script(&init_script);

            win_builder.build()?;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::desktop_window_minimize,
            commands::desktop_window_toggle_maximize,
            commands::desktop_window_close,
            commands::desktop_window_is_maximized,
            commands::desktop_open_directory_dialog,
            commands::sidecar_rpc_invoke,
            commands::sidecar_rpc_send,
        ])
        .build(tauri::generate_context!())
        .expect("error while building Hanzo Agents Tauri application");

    app.run(|app_handle, event| match event {
        RunEvent::Exit => {
            if let Some(state) = app_handle.try_state::<AppState>() {
                state.supervisor.shutdown();
            }
        }
        #[cfg(target_os = "macos")]
        RunEvent::Reopen {
            has_visible_windows,
            ..
        } => {
            if !has_visible_windows {
                if let Some(win) = app_handle.get_webview_window("main") {
                    let _ = win.show();
                    let _ = win.set_focus();
                }
            }
        }
        _ => {}
    });
}
