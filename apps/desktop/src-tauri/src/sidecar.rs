use rand::RngCore;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::env;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tauri::{AppHandle, Emitter};
use tokio::process::Command;

pub const SIDECAR_EVENT_NAME: &str = "hanzo-sidecar-event";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SidecarEventEnvelope {
    pub channel: String,
    #[serde(default)]
    pub payload: Value,
}

#[derive(Clone)]
pub struct SidecarClient {
    base_url: String,
    auth_header: String,
    http: reqwest::Client,
}

impl SidecarClient {
    pub fn new(port: u16, token: String) -> Self {
        let http = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("failed to build localhost sidecar HTTP client");
        Self {
            base_url: format!("http://127.0.0.1:{port}"),
            auth_header: format!("Bearer {token}"),
            http,
        }
    }

    pub async fn healthz(&self) -> Result<bool, String> {
        let res = self
            .http
            .get(format!("{}/healthz", self.base_url))
            .header(reqwest::header::AUTHORIZATION, &self.auth_header)
            .send()
            .await
            .map_err(|e| format!("Sidecar health check failed: {e}"))?;
        if !res.status().is_success() {
            return Err(format!("Sidecar health check returned HTTP {}", res.status()));
        }
        let body: Value = res
            .json()
            .await
            .map_err(|e| format!("Invalid healthz response: {e}"))?;
        Ok(body.get("ok").and_then(Value::as_bool).unwrap_or(false))
    }

    pub async fn invoke(&self, channel: &str, args: Vec<Value>) -> Result<Value, String> {
        let url = format!("{}/rpc/invoke", self.base_url);
        let payload = json!({ "channel": channel, "args": args });
        let mut last_err: Option<String> = None;

        for attempt in 0..12 {
            match self
                .http
                .post(&url)
                .header(reqwest::header::AUTHORIZATION, &self.auth_header)
                .json(&payload)
                .send()
                .await
            {
                Ok(res) => {
                    let status = res.status();
                    let body: Value = res
                        .json()
                        .await
                        .map_err(|e| format!("Invalid RPC response for {channel}: {e}"))?;
                    if !status.is_success() {
                        let msg = body
                            .get("error")
                            .and_then(Value::as_str)
                            .unwrap_or("Sidecar RPC request rejected");
                        return Err(format!("{msg} (HTTP {status})"));
                    }
                    if body.get("ok").and_then(Value::as_bool).unwrap_or(false) {
                        return Ok(body.get("result").cloned().unwrap_or(Value::Null));
                    }
                    let err_msg = body
                        .get("error")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                        .unwrap_or_else(|| format!("RPC failed: {channel}"));
                    return Err(err_msg);
                }
                Err(err) => {
                    let retryable = err.is_connect() || err.is_request();
                    last_err = Some(err.to_string());
                    if retryable && attempt < 11 {
                        tokio::time::sleep(Duration::from_millis(250)).await;
                        continue;
                    }
                    return Err(format!("Sidecar unreachable for {channel}: {err}"));
                }
            }
        }

        Err(last_err
            .map(|e| format!("Sidecar unreachable for {channel}: {e}"))
            .unwrap_or_else(|| format!("Sidecar unreachable: {channel}")))
    }

    pub async fn send(&self, channel: &str, args: Vec<Value>) -> Result<(), String> {
        let url = format!("{}/rpc/send", self.base_url);
        let payload = json!({ "channel": channel, "args": args });
        let res = self
            .http
            .post(&url)
            .header(reqwest::header::AUTHORIZATION, &self.auth_header)
            .json(&payload)
            .send()
            .await
            .map_err(|e| format!("Sidecar send failed for {channel}: {e}"))?;
        if !res.status().is_success() {
            return Err(format!(
                "Sidecar send rejected for {channel}: HTTP {}",
                res.status()
            ));
        }
        Ok(())
    }

    pub fn spawn_event_relay(&self, app_handle: AppHandle, shutdown_flag: Arc<AtomicBool>) {
        let client = self.clone();
        tauri::async_runtime::spawn(async move {
            let url = format!("{}/rpc/events", client.base_url);
            while !shutdown_flag.load(Ordering::SeqCst) {
                let response = client
                    .http
                    .get(&url)
                    .header(reqwest::header::AUTHORIZATION, &client.auth_header)
                    .header(reqwest::header::ACCEPT, "text/event-stream")
                    .send()
                    .await;

                match response {
                    Ok(mut res) if res.status().is_success() => {
                        let mut buffer = String::new();
                        while !shutdown_flag.load(Ordering::SeqCst) {
                            match res.chunk().await {
                                Ok(Some(bytes)) => {
                                    let chunk = String::from_utf8_lossy(&bytes).replace("\r\n", "\n");
                                    buffer.push_str(&chunk);
                                    while let Some(idx) = buffer.find("\n\n") {
                                        let frame = buffer[..idx].to_string();
                                        buffer.drain(..idx + 2);
                                        if let Some(envelope) = parse_sse_frame(&frame) {
                                            let _ = app_handle.emit_to(
                                                "main",
                                                SIDECAR_EVENT_NAME,
                                                envelope,
                                            );
                                        }
                                    }
                                }
                                Ok(None) | Err(_) => break,
                            }
                        }
                    }
                    _ => {}
                }

                if shutdown_flag.load(Ordering::SeqCst) {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(750)).await;
            }
        });
    }
}

pub(crate) fn parse_sse_frame(frame: &str) -> Option<SidecarEventEnvelope> {
    let mut data_lines = Vec::new();
    for line in frame.lines() {
        if let Some(rest) = line.strip_prefix("data:") {
            data_lines.push(rest.trim_start());
        }
    }
    if data_lines.is_empty() {
        return None;
    }
    let joined = data_lines.join("\n");
    serde_json::from_str::<SidecarEventEnvelope>(&joined).ok()
}

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

    pub fn shutdown_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.shutting_down)
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

#[cfg(test)]
mod tests {
    use super::{parse_sse_frame, SidecarClient, SidecarEventEnvelope};
    use serde_json::json;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[test]
    fn parse_sse_frame_parses_events_and_ignores_comments() {
        assert_eq!(parse_sse_frame(": connected"), None);
        assert_eq!(parse_sse_frame(""), None);

        let parsed = parse_sse_frame(
            r#"data: {"channel":"terminal-data","payload":{"terminalId":"t1","data":"hi"}}"#,
        )
        .expect("should parse valid event");
        assert_eq!(
            parsed,
            SidecarEventEnvelope {
                channel: "terminal-data".to_string(),
                payload: json!({ "terminalId": "t1", "data": "hi" }),
            }
        );
    }

    #[tokio::test]
    async fn sidecar_client_sends_bearer_token_and_never_sends_origin() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let token = "secret-sidecar-token-abc123".to_string();

        let server = tokio::spawn(async move {
            let mut captured = Vec::new();
            for _ in 0..3 {
                let (mut sock, _) = listener.accept().await.unwrap();
                let mut buf = vec![0u8; 4096];
                let n = sock.read(&mut buf).await.unwrap();
                let raw = String::from_utf8_lossy(&buf[..n]).to_string();
                let first_line = raw.lines().next().unwrap_or("").to_string();
                captured.push(raw.clone());

                let response = if first_line.starts_with("GET /healthz") {
                    let body = r#"{"ok":true}"#;
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    )
                } else if first_line.starts_with("POST /rpc/invoke") {
                    let body = r#"{"ok":true,"result":"pong"}"#;
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    )
                } else {
                    "HTTP/1.1 204 No Content\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                        .to_string()
                };
                sock.write_all(response.as_bytes()).await.unwrap();
            }
            captured
        });

        let client = SidecarClient::new(port, token.clone());
        assert!(client.healthz().await.unwrap());
        let res = client
            .invoke("canvas:list", vec![json!("arg1")])
            .await
            .unwrap();
        assert_eq!(res, json!("pong"));
        client
            .send("terminal-destroy", vec![json!("term-1")])
            .await
            .unwrap();

        let requests = server.await.unwrap();
        assert_eq!(requests.len(), 3);
        for req in requests {
            let lower = req.to_ascii_lowercase();
            assert!(
                lower.contains(&format!("authorization: bearer {token}")),
                "missing bearer token header in request: {req}"
            );
            assert!(
                !lower.contains("\norigin:"),
                "unexpected Origin header sent by Rust client: {req}"
            );
            assert!(
                !lower.contains("?token="),
                "token must never be placed in query string: {req}"
            );
        }
    }
}
