package mcp

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// readServerLogs opens packages/mcp/<alias>/<alias>.log and honors the lines parameter.
// Note: follow is not supported yet.
// TODO: Implement follow functionality.
func readServerLogs(projectDir string, alias string, _ bool, lines int) (io.ReadCloser, error) {
	logFile := filepath.Join(projectDir, "packages", "mcp", alias, fmt.Sprintf("%s.log", alias))

	if _, err := os.Stat(logFile); os.IsNotExist(err) {
		return nil, fmt.Errorf("log file not found for server %s", alias)
	}

	if lines <= 0 {
		file, err := os.Open(logFile)
		if err != nil {
			return nil, fmt.Errorf("failed to open log file: %w", err)
		}
		return file, nil
	}

	data, err := os.ReadFile(logFile)
	if err != nil {
		return nil, fmt.Errorf("failed to open log file: %w", err)
	}

	content := string(data)
	hasTrailingNewline := strings.HasSuffix(content, "\n")
	trimmed := strings.TrimSuffix(content, "\n")
	if trimmed == "" {
		return io.NopCloser(bytes.NewReader(data)), nil
	}

	allLines := strings.Split(trimmed, "\n")
	if lines < len(allLines) {
		allLines = allLines[len(allLines)-lines:]
	}

	out := strings.Join(allLines, "\n")
	if hasTrailingNewline {
		out += "\n"
	}
	return io.NopCloser(strings.NewReader(out)), nil
}
