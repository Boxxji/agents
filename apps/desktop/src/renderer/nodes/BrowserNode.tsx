import { type NodeProps, NodeResizer } from '@xyflow/react';
import type React from 'react';
import { useCallback, useEffect, useRef, useState } from 'react';
import './BrowserNode.css';

interface BrowserNodeData {
  browserId: string;
  url?: string;
}

function BrowserNode({ data, selected }: NodeProps) {
  const nodeData = data as unknown as BrowserNodeData;
  const iframeRef = useRef<HTMLIFrameElement | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const initialUrl = nodeData.url || '';
  const [currentUrl, setCurrentUrl] = useState(initialUrl);
  const [inputUrl, setInputUrl] = useState(initialUrl);
  const [history, setHistory] = useState<string[]>(initialUrl ? [initialUrl] : []);
  const [historyIndex, setHistoryIndex] = useState<number>(initialUrl ? 0 : -1);
  const [reloadNonce, setReloadNonce] = useState(0);
  const [isLoading, setIsLoading] = useState(false);

  const canGoBack = historyIndex > 0;
  const canGoForward = historyIndex >= 0 && historyIndex < history.length - 1;

  const normalizeUrl = useCallback((url: string): string => {
    return url.trim();
  }, []);

  const handleUrlSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      const normalizedUrl = normalizeUrl(inputUrl);
      if (!normalizedUrl) return;
      setIsLoading(true);
      setCurrentUrl(normalizedUrl);
      setInputUrl(normalizedUrl);
      setHistory((prev) => {
        const truncated = historyIndex >= 0 ? prev.slice(0, historyIndex + 1) : [];
        if (truncated[truncated.length - 1] === normalizedUrl) {
          return truncated;
        }
        const next = [...truncated, normalizedUrl];
        setHistoryIndex(next.length - 1);
        return next;
      });
    },
    [historyIndex, inputUrl, normalizeUrl]
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Escape') {
        setInputUrl(currentUrl);
        inputRef.current?.blur();
      }
    },
    [currentUrl]
  );

  const handleBack = useCallback(() => {
    if (!canGoBack) return;
    const nextIdx = historyIndex - 1;
    const target = history[nextIdx];
    if (target) {
      setIsLoading(true);
      setHistoryIndex(nextIdx);
      setCurrentUrl(target);
      setInputUrl(target);
    }
  }, [canGoBack, history, historyIndex]);

  const handleForward = useCallback(() => {
    if (!canGoForward) return;
    const nextIdx = historyIndex + 1;
    const target = history[nextIdx];
    if (target) {
      setIsLoading(true);
      setHistoryIndex(nextIdx);
      setCurrentUrl(target);
      setInputUrl(target);
    }
  }, [canGoForward, history, historyIndex]);

  const handleReload = useCallback(() => {
    if (!currentUrl) return;
    setIsLoading(true);
    setReloadNonce((n) => n + 1);
  }, [currentUrl]);

  useEffect(() => {
    if (selected && !currentUrl && inputRef.current) {
      inputRef.current.focus();
    }
  }, [selected, currentUrl]);

  useEffect(() => {
    const iframe = iframeRef.current;
    if (!iframe || !selected) return;

    const handleWheel = (e: Event) => {
      e.stopPropagation();
    };

    iframe.addEventListener('wheel', handleWheel, { passive: false });
    return () => {
      iframe.removeEventListener('wheel', handleWheel);
    };
  }, [selected, currentUrl]);

  return (
    <div className={`browser-node ${selected ? 'selected' : ''}`}>
      <NodeResizer
        minWidth={400}
        minHeight={300}
        isVisible={true}
        lineStyle={{ borderColor: 'transparent' }}
        handleStyle={{
          width: 8,
          height: 8,
          borderRadius: '50%',
        }}
      />

      <div className="browser-node-header">
        <div className="browser-controls">
          <button
            type="button"
            className="browser-nav-button"
            onClick={handleBack}
            disabled={!canGoBack}
            title="Go back"
          >
            ←
          </button>
          <button
            type="button"
            className="browser-nav-button"
            onClick={handleForward}
            disabled={!canGoForward}
            title="Go forward"
          >
            →
          </button>
          <button
            type="button"
            className="browser-nav-button"
            onClick={handleReload}
            title="Reload"
          >
            {isLoading ? '×' : '↻'}
          </button>
        </div>

        <form className="browser-url-form" onSubmit={handleUrlSubmit}>
          <input
            ref={inputRef}
            type="text"
            className="browser-url-input"
            value={inputUrl}
            onChange={(e) => setInputUrl(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Enter URL or search..."
            spellCheck={false}
          />
        </form>
      </div>

      <div className="browser-content">
        {currentUrl ? (
          <iframe
            key={`${currentUrl}:${reloadNonce}`}
            ref={iframeRef}
            src={currentUrl}
            title={`Browser ${nodeData.browserId}`}
            className="browser-webview"
            onLoad={() => setIsLoading(false)}
            onError={() => setIsLoading(false)}
          />
        ) : (
          <div className="browser-empty-state">
            <div className="browser-empty-icon">🌐</div>
            <div className="browser-empty-text">Enter a URL above to start browsing</div>
          </div>
        )}
      </div>
    </div>
  );
}

export default BrowserNode;
