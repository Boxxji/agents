use crate::bridge_apis::DOMAIN_APIS_JS;
use serde_json::json;

pub fn build_initialization_script(port: u16, token: &str, home_dir: &str) -> String {
    let cfg = json!({
        "baseUrl": format!("http://127.0.0.1:{port}"),
        "token": token,
        "homeDir": home_dir,
    });
    let serialized = serde_json::to_string(&cfg).unwrap_or_else(|_| "{}".to_string());

    format!(
        r#"(function () {{
  if (typeof window === "undefined" || window.__HANZO_AGENTS_TAURI_SHIM__) return;
  window.__HANZO_AGENTS_TAURI_SHIM__ = true;
  const __CFG__ = {serialized};

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

  let eventSource = null;
  function connectEvents() {{
    try {{
      const url = __CFG__.baseUrl + "/rpc/events?token=" + encodeURIComponent(__CFG__.token);
      eventSource = new EventSource(url);
      eventSource.onmessage = function (ev) {{
        try {{
          const msg = JSON.parse(ev.data);
          if (msg && typeof msg.channel === "string") {{
            dispatchEvent(msg.channel, msg.payload);
          }}
        }} catch (_e) {{}}
      }};
      eventSource.onerror = function () {{
        if (eventSource) {{
          eventSource.close();
          eventSource = null;
        }}
        setTimeout(connectEvents, 750);
      }};
    }} catch (_err) {{
      setTimeout(connectEvents, 1000);
    }}
  }}
  connectEvents();

  async function rpcInvoke(channel) {{
    const args = Array.prototype.slice.call(arguments, 1);
    let lastErr = null;
    for (let attempt = 0; attempt < 12; attempt++) {{
      try {{
        const res = await fetch(__CFG__.baseUrl + "/rpc/invoke", {{
          method: "POST",
          headers: {{
            "Content-Type": "application/json",
            "Authorization": "Bearer " + __CFG__.token
          }},
          body: JSON.stringify({{ channel: channel, args: args }})
        }});
        const body = await res.json();
        if (!body.ok) {{
          throw new Error(body.error || ("RPC failed: " + channel));
        }}
        return body.result;
      }} catch (err) {{
        lastErr = err;
        if (err && err.name === "TypeError" && attempt < 11) {{
          await new Promise(function (r) {{ setTimeout(r, 250); }});
          continue;
        }}
        throw err;
      }}
    }}
    throw lastErr || new Error("Sidecar unreachable: " + channel);
  }}

  function rpcSend(channel) {{
    const args = Array.prototype.slice.call(arguments, 1);
    fetch(__CFG__.baseUrl + "/rpc/send", {{
      method: "POST",
      headers: {{
        "Content-Type": "application/json",
        "Authorization": "Bearer " + __CFG__.token
      }},
      body: JSON.stringify({{ channel: channel, args: args }})
    }}).catch(function () {{}});
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

    #[test]
    fn init_script_exposes_all_renderer_apis() {
        let script = build_initialization_script(13880, "test-token", "/home/test");
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
}
