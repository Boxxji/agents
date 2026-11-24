pub const DOMAIN_APIS_JS: &str = r#"
  window.codingAgentAPI = {
    generate: function (agentType, request) {
      return unwrapResponse(rpcInvoke("coding-agent:generate", agentType, request));
    },
    generateStreaming: async function (agentType, request, onChunk) {
      const requestId = crypto.randomUUID();
      const unsub = addChannelListener("coding-agent:stream-chunk", function (data) {
        if (data && data.requestId === requestId) onChunk(data.chunk);
      });
      try {
        return await unwrapResponse(rpcInvoke("coding-agent:generate-streaming", requestId, agentType, request));
      } finally {
        unsub();
      }
    },
    generateStreamingStructured: async function (agentType, request, onChunk) {
      const requestId = crypto.randomUUID();
      const unsub = addChannelListener("coding-agent:stream-chunk-structured", function (data) {
        if (data && data.requestId === requestId) onChunk(data.chunk);
      });
      try {
        return await unwrapResponse(rpcInvoke("coding-agent:generate-streaming-structured", requestId, agentType, request));
      } finally {
        unsub();
      }
    },
    continueSession: function (agentType, identifier, prompt, options) {
      return unwrapResponse(rpcInvoke("coding-agent:continue-session", agentType, identifier, prompt, options));
    },
    continueSessionStreaming: async function (agentType, identifier, prompt, onChunk, options) {
      const requestId = crypto.randomUUID();
      const unsub = addChannelListener("coding-agent:stream-chunk", function (data) {
        if (data && data.requestId === requestId) onChunk(data.chunk);
      });
      try {
        return await unwrapResponse(rpcInvoke("coding-agent:continue-session-streaming", requestId, agentType, identifier, prompt, options));
      } finally {
        unsub();
      }
    },
    forkSession: function (agentType, options) {
      return rpcInvoke("coding-agent:fork-session", agentType, options);
    },
    getAvailableAgents: function () {
      return unwrapResponse(rpcInvoke("coding-agent:get-available"));
    },
    getCapabilities: function (agentType) {
      return unwrapResponse(rpcInvoke("coding-agent:get-capabilities", agentType));
    },
    isAgentAvailable: function (agentType) {
      return unwrapResponse(rpcInvoke("coding-agent:is-available", agentType));
    },
    listSessionSummaries: function (agentType, filter) {
      return unwrapResponse(rpcInvoke("coding-agent:list-session-summaries", agentType, filter));
    },
    getSession: function (agentType, sessionId, filter) {
      return unwrapResponse(rpcInvoke("coding-agent:get-session", agentType, sessionId, filter));
    },
    onStreamChunk: function (cb) {
      return addChannelListener("coding-agent:stream-chunk", cb);
    },
    onStreamChunkStructured: function (cb) {
      return addChannelListener("coding-agent:stream-chunk-structured", cb);
    },
    onAgentEvent: function (cb) {
      return addChannelListener("coding-agent:event", cb);
    },
    onAgentLifecycle: function (cb) {
      return addChannelListener("agent-lifecycle", cb);
    },
    respondToAction: async function (response) {
      await unwrapResponse(rpcInvoke("coding-agent:respond-to-action", response));
    },
    getLatestSession: function (agentType, workspacePath) {
      return unwrapResponse(rpcInvoke("coding-agent:get-latest-session", agentType, workspacePath));
    },
    sessionFileExists: function (agentType, sessionId, workspacePath) {
      return unwrapResponse(rpcInvoke("coding-agent:session-file-exists", agentType, sessionId, workspacePath));
    },
    abort: async function (agentType) {
      await unwrapResponse(rpcInvoke("coding-agent:abort", agentType));
    }
  };

  window.llmAPI = {
    chat: function (request) {
      return unwrapResponse(rpcInvoke("llm:chat", request));
    },
    chatStream: async function (requestId, request, onChunk) {
      const unsub = addChannelListener("llm:stream-chunk", function (data) {
        if (data && data.requestId === requestId) onChunk(data.chunk);
      });
      try {
        return await unwrapResponse(rpcInvoke("llm:chat-stream", requestId, request));
      } finally {
        unsub();
      }
    },
    chatWithTools: function (request, maxIterations) {
      return unwrapResponse(rpcInvoke("llm:chat-with-tools", request, maxIterations));
    },
    setApiKey: async function (vendor, apiKey) {
      await unwrapResponse(rpcInvoke("llm:set-api-key", vendor, apiKey));
    },
    deleteApiKey: async function (vendor) {
      await unwrapResponse(rpcInvoke("llm:delete-api-key", vendor));
    },
    hasApiKey: function (vendor) {
      return unwrapResponse(rpcInvoke("llm:has-api-key", vendor));
    },
    listVendorsWithKeys: function () {
      return unwrapResponse(rpcInvoke("llm:list-vendors-with-keys"));
    },
    getAvailableModels: function () {
      return unwrapResponse(rpcInvoke("llm:get-available-models"));
    },
    isConfigured: function () {
      return unwrapResponse(rpcInvoke("llm:is-configured"));
    },
    getCapabilities: function () {
      return unwrapResponse(rpcInvoke("llm:get-capabilities"));
    },
    onStreamChunk: function (cb) {
      return addChannelListener("llm:stream-chunk", cb);
    }
  };

  window.representationAPI = {
    getAvailableTypes: function () {
      return unwrapResponse(rpcInvoke("representation:get-available-types"));
    },
    transform: function (providerId, input) {
      return unwrapResponse(rpcInvoke("representation:transform", providerId, input));
    },
    transformToImage: function (input, options) {
      return unwrapResponse(rpcInvoke("representation:transform-to-image", input, options));
    },
    transformToSummary: function (input, options) {
      return unwrapResponse(rpcInvoke("representation:transform-to-summary", input, options));
    },
    transformToAudio: function (input, options) {
      return unwrapResponse(rpcInvoke("representation:transform-to-audio", input, options));
    },
    getAllProviders: function () {
      return unwrapResponse(rpcInvoke("representation:get-all-providers"));
    }
  };

  window.shellAPI = {
    openWithEditor: async function (directoryPath, editor) {
      await unwrapResponse(rpcInvoke("shell:open-with-editor", directoryPath, editor));
    },
    getAvailableEditors: function () {
      return unwrapResponse(rpcInvoke("shell:get-available-editors"));
    },
    showInFolder: async function (filePath) {
      await unwrapResponse(rpcInvoke("shell:show-in-folder", filePath));
    },
    openDirectoryDialog: async function (options) {
      try {
        return await tauriInvoke("desktop_open_directory_dialog", { options: options || null });
      } catch (_err) {
        return unwrapResponse(rpcInvoke("shell:open-directory-dialog", options));
      }
    }
  };

  window.fileAPI = {
    readFile: function (filePath) {
      return unwrapResponse(rpcInvoke("file:read", filePath));
    },
    exists: async function (filePath) {
      const result = await rpcInvoke("file:exists", filePath);
      return Boolean(result && result.success && result.exists);
    }
  };

  window.gitAPI = {
    getInfo: function (workspacePath) {
      return rpcInvoke("git:get-info-strict", workspacePath);
    },
    listBranches: async function (workspacePath) {
      try {
        return await unwrapResponse(rpcInvoke("git:list-branches", workspacePath));
      } catch (_e) {
        return null;
      }
    },
    createBranch: async function (workspacePath, branchName) {
      try {
        return await rpcInvoke("git:create-branch", workspacePath, branchName);
      } catch (err) {
        return { success: false, error: err && err.message ? err.message : String(err) };
      }
    },
    checkoutBranch: async function (workspacePath, branchName) {
      try {
        return await rpcInvoke("git:checkout-branch", workspacePath, branchName);
      } catch (err) {
        return { success: false, error: err && err.message ? err.message : String(err) };
      }
    },
    getGithubUsername: async function () {
      try {
        const res = await rpcInvoke("git:get-github-username");
        if (res && res.success && res.data) {
          return { success: true, username: res.data.username };
        }
        return { success: false, error: (res && res.error) || "Failed to get GitHub username" };
      } catch (err) {
        return { success: false, error: err && err.message ? err.message : String(err) };
      }
    }
  };

  window.windowAPI = {
    minimize: function () {
      tauriInvoke("desktop_window_minimize").catch(function () {});
    },
    maximize: function () {
      tauriInvoke("desktop_window_toggle_maximize").catch(function () {});
    },
    close: function () {
      tauriInvoke("desktop_window_close").catch(function () {});
    },
    isMaximized: function () {
      return tauriInvoke("desktop_window_is_maximized");
    }
  };

  window.sessionWatcherAPI = {
    watch: async function (agentType) {
      const res = await rpcInvoke("session-watcher:watch", agentType);
      if (!res || !res.success) throw new Error((res && res.error) || "Failed to watch sessions");
    },
    unwatch: async function (agentType) {
      const res = await rpcInvoke("session-watcher:unwatch", agentType);
      if (!res || !res.success) throw new Error((res && res.error) || "Failed to unwatch sessions");
    },
    onSessionFileChanged: function (cb) {
      return addChannelListener("session:file-changed", cb);
    }
  };

  window.sessionSummaryCacheAPI = {
    getSummary: function (sessionId, workspacePath) {
      return unwrapResponse(rpcInvoke("session-summary:get", sessionId, workspacePath));
    },
    saveSummary: async function (sessionId, workspacePath, summary, messageCount) {
      await unwrapResponse(rpcInvoke("session-summary:save", sessionId, workspacePath, summary, messageCount));
    },
    isStale: function (sessionId, workspacePath, currentMessageCount) {
      return unwrapResponse(rpcInvoke("session-summary:is-stale", sessionId, workspacePath, currentMessageCount));
    },
    deleteSummary: async function (sessionId, workspacePath) {
      await unwrapResponse(rpcInvoke("session-summary:delete", sessionId, workspacePath));
    }
  };

  window.recentWorkspacesAPI = {
    addWorkspace: async function (workspacePath, options) {
      await unwrapResponse(rpcInvoke("recent-workspaces:add", workspacePath, options));
    },
    getRecentWorkspaces: function (limit) {
      return unwrapResponse(rpcInvoke("recent-workspaces:get", limit));
    },
    removeWorkspace: async function (workspacePath) {
      await unwrapResponse(rpcInvoke("recent-workspaces:remove", workspacePath));
    },
    clearAll: async function () {
      await unwrapResponse(rpcInvoke("recent-workspaces:clear"));
    },
    hasWorkspace: function (workspacePath) {
      return unwrapResponse(rpcInvoke("recent-workspaces:has", workspacePath));
    }
  };
"#;
