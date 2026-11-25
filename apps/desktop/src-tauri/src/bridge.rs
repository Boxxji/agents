use crate::bridge_apis::DOMAIN_APIS_JS;
use crate::sidecar::SIDECAR_EVENT_NAME;
use serde_json::json;

pub fn build_initialization_script(home_dir: &str) -> String {
    let cfg = json!({
        "homeDir": home_dir,
    });
    let serialized = serde_json::to_string(&cfg).unwrap_or_else(|_| "{}".to_string());
    let sidecar_event_name =
        serde_json::to_string(SIDECAR_EVENT_NAME).unwrap_or_else(|_| "\"hanzo-sidecar-event\"".to_string());

    format!(
        r#"(function () {{
  if (typeof window === "undefined" || window.__HANZO_AGENTS_TAURI_SHIM__) return;
  window.__HANZO_AGENTS_TAURI_SHIM__ = true;
  const __CFG__ = {serialized};
  const __SIDECAR_EVENT__ = {sidecar_event_name};

  function tauriInvoke(cmd, args) {{
    if (window.__TAURI_INTERNALS__ && typeof window.__TAURI_INTERNALS__.invoke === "function") {{
      return window.__TAURI_INTERNALS__.invoke(cmd, args || {{}});
    }}
    if (window.__TAURI__ && window.__TAURI__.core && typeof window.__TAURI__.core.invoke === "function") {{
      return window.__TAURI__.core.invoke(cmd, args || {{}});
    }}
    return Promise.reject(new Error("Tauri IPC unavailable for " + cmd));
  }}

  const listeners = new Map();
  function addChannelListener(channel, fn) {{
    let set = listeners.get(channel);
    if (!set) {{
      set = new Set();
      listeners.set(channel, set);
    }}
    set.add(fn);
    return function () {{
      set.delete(fn);
    }};
  }}

  function removeChannelListeners(channel) {{
    listeners.delete(channel);
  }}

  function dispatchEvent(channel, payload) {{
    const set = listeners.get(channel);
    if (!set) return;
    for (const fn of Array.from(set)) {{
      try {{
        fn(payload);
      }} catch (err) {{
        console.error("[HanzoAgentsShim] listener error on " + channel, err);
      }}
    }}
  }}

  let sidecarEventsSubscribed = false;
  function subscribeSidecarEvents() {{
    if (sidecarEventsSubscribed) return;
    if (window.__TAURI__ && window.__TAURI__.event && typeof window.__TAURI__.event.listen === "function") {{
      sidecarEventsSubscribed = true;
      window.__TAURI__.event.listen(__SIDECAR_EVENT__, function (ev) {{
        const msg = ev && ev.payload;
        if (msg && typeof msg.channel === "string") {{
          dispatchEvent(msg.channel, msg.payload);
        }}
      }}).catch(function () {{
        sidecarEventsSubscribed = false;
        setTimeout(subscribeSidecarEvents, 500);
      }});
      return;
    }}
    setTimeout(subscribeSidecarEvents, 50);
  }}
  subscribeSidecarEvents();

  async function rpcInvoke(channel) {{
    const args = Array.prototype.slice.call(arguments, 1);
    return await tauriInvoke("sidecar_rpc_invoke", {{ channel: channel, args: args }});
  }}

  function rpcSend(channel) {{
    const args = Array.prototype.slice.call(arguments, 1);
    tauriInvoke("sidecar_rpc_send", {{ channel: channel, args: args }}).catch(function () {{}});
  }}

  async function unwrapResponse(promise) {{
    const response = await promise;
    if (!response || !response.success) {{
      throw new Error((response && response.error) || "Unknown error");
    }}
    return response.data;
  }}

  window.electronAPI = {{
    createTerminal: function (terminalId, workspacePath) {{
      rpcSend("terminal-create", terminalId, workspacePath);
    }},
    onTerminalData: function (cb) {{
      addChannelListener("terminal-data", cb);
    }},
    onTerminalExit: function (cb) {{
      addChannelListener("terminal-exit", cb);
    }},
    sendTerminalInput: function (terminalId, data) {{
      rpcSend("terminal-input", {{ terminalId: terminalId, data: data }});
    }},
    sendTerminalResize: function (terminalId, cols, rows) {{
      rpcSend("terminal-resize", {{ terminalId: terminalId, cols: cols, rows: rows }});
    }},
    destroyTerminal: function (terminalId) {{
      rpcSend("terminal-destroy", terminalId);
    }},
    removeAllListeners: function (channel) {{
      removeChannelListeners(channel);
    }},
    getHomeDir: function () {{
      return __CFG__.homeDir;
    }},
    onMcpBridgeRequest: function (cb) {{
      addChannelListener("mcp-bridge-request", cb);
    }},
    sendMcpBridgeResponse: function (response) {{
      rpcSend("mcp-bridge-response", response);
    }}
  }};

  window.terminalSessionAPI = {{
    getTerminalSessionState: function (tid) {{
      return unwrapResponse(rpcInvoke("terminal-get-session-state", tid));
    }},
    setTerminalSessionState: async function (tid, state) {{
      await unwrapResponse(rpcInvoke("terminal-set-session-state", tid, state));
    }},
    clearTerminalSessionState: async function (tid) {{
      await unwrapResponse(rpcInvoke("terminal-clear-session-state", tid));
    }},
    getTerminalBuffer: function (tid) {{
      return unwrapResponse(rpcInvoke("terminal-get-buffer", tid));
    }}
  }};

  window.canvasAPI = {{
    saveCanvas: async function (id, state) {{
      await unwrapResponse(rpcInvoke("canvas:save", id, state));
    }},
    loadCanvas: function (id) {{
      return unwrapResponse(rpcInvoke("canvas:load", id));
    }},
    listCanvases: function () {{
      return unwrapResponse(rpcInvoke("canvas:list"));
    }},
    deleteCanvas: async function (id) {{
      await unwrapResponse(rpcInvoke("canvas:delete", id));
    }},
    getCurrentCanvasId: function () {{
      return unwrapResponse(rpcInvoke("canvas:get-current-id"));
    }},
    setCurrentCanvasId: async function (id) {{
      await unwrapResponse(rpcInvoke("canvas:set-current-id", id));
    }}
  }};

  window.worktreeAPI = {{
    provision: function (repoPath, branchName, options) {{
      return unwrapResponse(rpcInvoke("worktree:provision", repoPath, branchName, options));
    }},
    release: async function (worktreeId, options) {{
      await unwrapResponse(rpcInvoke("worktree:release", worktreeId, options));
    }},
    get: function (worktreeId) {{
      return unwrapResponse(rpcInvoke("worktree:get", worktreeId));
    }},
    list: function (repoPath) {{
      return unwrapResponse(rpcInvoke("worktree:list", repoPath));
    }}
  }};

  window.agentStatusAPI = {{
    saveAgentStatus: async function (agentId, state) {{
      await unwrapResponse(rpcInvoke("agent-status:save", agentId, state));
    }},
    loadAgentStatus: function (agentId) {{
      return unwrapResponse(rpcInvoke("agent-status:load", agentId));
    }},
    deleteAgentStatus: async function (agentId) {{
      await unwrapResponse(rpcInvoke("agent-status:delete", agentId));
    }},
    loadAllAgentStatuses: function () {{
      return unwrapResponse(rpcInvoke("agent-status:load-all"));
    }}
  }};
{DOMAIN_APIS_JS}
}})();"#
    )
}

#[cfg(test)]
mod tests {
    use super::build_initialization_script;
    use crate::commands::{
        is_allowed_invoke_channel, is_allowed_send_channel, ALLOWED_INVOKE_CHANNELS,
        ALLOWED_SEND_CHANNELS,
    };
    use std::collections::BTreeSet;

    fn extract_call_channels(script: &str, fn_name: &str) -> BTreeSet<String> {
        let needle = format!("{fn_name}(\"");
        let mut out = BTreeSet::new();
        let mut rest = script;
        while let Some(idx) = rest.find(&needle) {
            let after = &rest[idx + needle.len()..];
            if let Some(end) = after.find('"') {
                out.insert(after[..end].to_string());
                rest = &after[end + 1..];
            } else {
                break;
            }
        }
        out
    }

    #[test]
    fn init_script_exposes_all_renderer_apis() {
        let script = build_initialization_script("/home/test");
        for api in [
            "window.electronAPI",
            "window.terminalSessionAPI",
            "window.canvasAPI",
            "window.worktreeAPI",
            "window.agentStatusAPI",
            "window.codingAgentAPI",
            "window.llmAPI",
            "window.representationAPI",
            "window.shellAPI",
            "window.fileAPI",
            "window.gitAPI",
            "window.windowAPI",
            "window.sessionWatcherAPI",
            "window.sessionSummaryCacheAPI",
            "window.recentWorkspacesAPI",
        ] {
            assert!(script.contains(api), "missing {api}");
        }
    }

    #[test]
    fn init_script_never_exposes_sidecar_token_or_direct_http_calls() {
        let script = build_initialization_script("/home/test");
        for forbidden in [
            "token",
            "baseUrl",
            "127.0.0.1",
            "EventSource",
            "fetch(",
            "/rpc/invoke",
            "/rpc/send",
            "/rpc/events",
            "Authorization",
        ] {
            assert!(
                !script.contains(forbidden),
                "initialization script must not contain {forbidden:?}"
            );
        }
        assert!(script.contains("sidecar_rpc_invoke"));
        assert!(script.contains("sidecar_rpc_send"));
        assert!(script.contains("hanzo-sidecar-event"));
    }

    #[test]
    fn bridge_channels_match_rust_command_allowlists() {
        let script = build_initialization_script("/home/test");

        let invoke_channels = extract_call_channels(&script, "rpcInvoke");
        let send_channels = extract_call_channels(&script, "rpcSend");

        for ch in &invoke_channels {
            assert!(
                is_allowed_invoke_channel(ch),
                "bridge rpcInvoke channel {ch:?} is missing from ALLOWED_INVOKE_CHANNELS"
            );
        }
        for ch in ALLOWED_INVOKE_CHANNELS {
            assert!(
                invoke_channels.contains(*ch),
                "ALLOWED_INVOKE_CHANNELS entry {ch:?} is unused in bridge script"
            );
        }

        for ch in &send_channels {
            assert!(
                is_allowed_send_channel(ch),
                "bridge rpcSend channel {ch:?} is missing from ALLOWED_SEND_CHANNELS"
            );
        }
        for ch in ALLOWED_SEND_CHANNELS {
            assert!(
                send_channels.contains(*ch),
                "ALLOWED_SEND_CHANNELS entry {ch:?} is unused in bridge script"
            );
        }

        assert!(!is_allowed_invoke_channel("arbitrary:evil"));
        assert!(!is_allowed_send_channel("arbitrary:evil"));
    }
}
