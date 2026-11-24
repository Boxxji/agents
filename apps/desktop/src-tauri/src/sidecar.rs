use rand::RngCore;
use std::env;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::process::Command;

#[derive(Clone)]
pub struct SidecarRuntimeConfig {
    pub port: u16,
    pub token: String,
    pub user_data_dir: PathBuf,
    pub mcp_mode: bool,
}

pub struct SidecarSupervisor {
    shutting_down: Arc<AtomicBool>,
    child_pid: Arc<AtomicU32>,
}

pub fn generate_random_token(byte_len: usize) -> String {
    let mut bytes = vec![0u8; byte_len];
    rand::thread_rng().fill_bytes(&mut bytes);
    hex::encode(bytes)
}

pub fn pick_available_port(preferred_start: u16) -> u16 {
    for candidate in preferred_start..=preferred_start.saturating_add(200) {
        if TcpListener::bind(("127.0.0.1", candidate)).is_ok() {
            return candidate;
        }
    }
    TcpListener::bind(("127.0.0.1", 0))
        .and_then(|sock| sock.local_addr())
        .map(|addr| addr.port())
        .unwrap_or(preferred_start)
}

pub fn resolve_node_executable() -> PathBuf {
    if let Ok(explicit) = env::var("HANZO_NODE_BINARY") {
        let p = PathBuf::from(explicit.trim());
        if p.is_file() {
            return p;
        }
    }
    #[cfg(windows)]
    let candidates = [
        r"C:\Program Files\nodejs\node.exe",
        r"C:\Program Files (x86)\nodejs\node.exe",
    ];
    #[cfg(not(windows))]
    let candidates = [
        "/opt/homebrew/bin/node",
        "/usr/local/bin/node",
        "/usr/bin/node",
    ];
    for candidate in candidates {
        let p = PathBuf::from(candidate);
        if p.is_file() {
            return p;
        }
    }
    if let Some(home) = dirs::home_dir() {
        for rel in [
            ".nvm/current/bin/node",
            ".volta/bin/node",
            ".local/share/mise/shims/node",
            ".fnm/current/bin/node",
        ] {
            let p = home.join(rel);
            if p.is_file() {
                return p;
            }
        }
    }
    PathBuf::from(if cfg!(windows) { "node.exe" } else { "node" })
}

fn check_sidecar_script_in(app_dir: &Path) -> Option<(PathBuf, PathBuf)> {
    let entry = app_dir.join("dist/main/src/main/main.js");
    if entry.is_file() {
        return Some((entry, app_dir.to_path_buf()));
    }
    None
}

pub fn resolve_sidecar_entry() -> Option<(PathBuf, PathBuf)> {
    if let Ok(explicit) = env::var("HANZO_AGENTS_SIDECAR_ENTRY") {
        let p = PathBuf::from(explicit.trim());
        if p.is_file() {
            let cwd = p
                .parent()
                .and_then(Path::parent)
                .and_then(Path::parent)
                .and_then(Path::parent)
                .map(Path::to_path_buf)
                .unwrap_or_else(|| PathBuf::from("."));
            return Some((p, cwd));
        }
    }

    if let Ok(exe) = env::current_exe() {
        let mut cursor = exe.parent();
        while let Some(dir) = cursor {
            if let Some(found) = check_sidecar_script_in(dir) {
                return Some(found);
            }
            let resources = dir.join("Resources");
            if let Some(found) = check_sidecar_script_in(&resources) {
                return Some(found);
            }
            cursor = dir.parent();
        }
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if let Some(app_dir) = manifest_dir.parent() {
        if let Some(found) = check_sidecar_script_in(app_dir) {
            return Some(found);
        }
    }

    None
}

impl SidecarSupervisor {
    pub fn start(config: SidecarRuntimeConfig) -> Self {
        let shutting_down = Arc::new(AtomicBool::new(false));
        let child_pid = Arc::new(AtomicU32::new(0));

        let stop_flag = Arc::clone(&shutting_down);
        let pid_slot = Arc::clone(&child_pid);

        tauri::async_runtime::spawn(async move {
            let mut backoff_ms = 500u64;
            while !stop_flag.load(Ordering::SeqCst) {
                let Some((entry_js, cwd)) = resolve_sidecar_entry() else {
                    eprintln!("[hanzo-agents] Sidecar entry dist/main/src/main/main.js not found; retrying...");
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    continue;
                };

                let node_bin = resolve_node_executable();
                let mut cmd = Command::new(&node_bin);
                cmd.arg(&entry_js);
                if config.mcp_mode {
                    cmd.arg("--mcp");
                }
                cmd.current_dir(&cwd)
                    .env("HANZO_AGENTS_PORT", config.port.to_string())
                    .env("HANZO_AGENTS_TOKEN", &config.token)
                    .env(
                        "HANZO_AGENTS_USER_DATA",
                        config.user_data_dir.to_string_lossy().as_ref(),
                    )
                    .stdin(Stdio::null())
                    .stdout(Stdio::inherit())
                    .stderr(Stdio::inherit());

                let started = std::time::Instant::now();
                match cmd.spawn() {
                    Ok(mut child) => {
                        if let Some(pid) = child.id() {
                            pid_slot.store(pid, Ordering::SeqCst);
                        }
                        let _ = child.wait().await;
                        pid_slot.store(0, Ordering::SeqCst);
                        if stop_flag.load(Ordering::SeqCst) {
                            break;
                        }
                    }
                    Err(err) => {
                        eprintln!(
                            "[hanzo-agents] Failed to spawn Node sidecar {}: {err}",
                            node_bin.display()
                        );
                    }
                }

                if started.elapsed() > Duration::from_secs(15) {
                    backoff_ms = 500;
                }
                tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                backoff_ms = (backoff_ms * 2).min(10_000);
            }
        });

        Self {
            shutting_down,
            child_pid,
        }
    }

    pub fn shutdown(&self) {
        self.shutting_down.store(true, Ordering::SeqCst);
        let pid = self.child_pid.swap(0, Ordering::SeqCst);
        if pid == 0 {
            return;
        }
        #[cfg(unix)]
        unsafe {
            libc::kill(pid as i32, libc::SIGTERM);
        }
        #[cfg(windows)]
        {
            let _ = std::process::Command::new("taskkill")
                .args(["/PID", &pid.to_string(), "/T", "/F"])
                .status();
        }
    }
}
