// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/hanzoai/agents/control-plane/internal/eval/contextpack"
)

func TestCLI_EndToEndJSONAndMarkdown(t *testing.T) {
	tmpDir := t.TempDir()
	jsonPath := filepath.Join(tmpDir, "results.json")
	mdPath := filepath.Join(tmpDir, "score_table.md")

	var stdout, stderr bytes.Buffer
	err := run([]string{
		"--seed", "1346",
		"--token-budget", "1400",
		"--out-json", jsonPath,
		"--out-md", mdPath,
	}, &stdout, &stderr)
	if err != nil {
		t.Fatalf("CLI run failed: %v (stderr=%s)", err, stderr.String())
	}

	if !strings.Contains(stdout.String(), "kai_evidence_edge_pack") {
		t.Fatalf("expected stdout summary to mention kai_evidence_edge_pack, got:\n%s", stdout.String())
	}

	rawJSON, err := os.ReadFile(jsonPath)
	if err != nil {
		t.Fatalf("read json output: %v", err)
	}
	var report contextpack.BenchmarkReport
	if err := json.Unmarshal(rawJSON, &report); err != nil {
		t.Fatalf("unmarshal json report: %v", err)
	}
	if report.Metadata.Seed != 1346 {
		t.Errorf("expected seed 1346, got %d", report.Metadata.Seed)
	}
	if len(report.PrimaryOverall) != 5 {
		t.Errorf("expected 5 policy summaries, got %d", len(report.PrimaryOverall))
	}
	if report.Metadata.PrimaryCaseCount != 100 {
		t.Errorf("expected 100 primary cases, got %d", report.Metadata.PrimaryCaseCount)
	}

	rawMD, err := os.ReadFile(mdPath)
	if err != nil {
		t.Fatalf("read markdown output: %v", err)
	}
	md := string(rawMD)
	for _, needle := range []string{
		"HIP-1346 Score Table",
		"raw_concat_baseline",
		"u_shaped_edge_packing",
		"kai_evidence_edge_pack",
		"Honest Negative Regime #1",
		"Honest Negative Regime #2",
		"Honest Negative Regime #3",
	} {
		if !strings.Contains(md, needle) {
			t.Errorf("markdown score table missing expected section %q", needle)
		}
	}
}

func TestCLI_InvalidTokenBudgetRejected(t *testing.T) {
	var stdout, stderr bytes.Buffer
	err := run([]string{"--token-budget", "0"}, &stdout, &stderr)
	if err == nil {
		t.Fatalf("expected error for --token-budget 0, got nil")
	}
}
