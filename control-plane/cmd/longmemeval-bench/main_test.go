package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/hanzoai/agents/control-plane/internal/eval/longmemeval"
	"github.com/stretchr/testify/require"
)

func TestCLI_SmokeAndArtifactGeneration(t *testing.T) {
	tmp := t.TempDir()
	jsonPath := filepath.Join(tmp, "out", "results.json")
	mdPath := filepath.Join(tmp, "out", "score_table.md")

	var stdout, stderr bytes.Buffer
	err := run([]string{
		"--seed", "1346",
		"--horizons", "5,25",
		"--thresholds", "0.20,0.50,0.80",
		"--out-json", jsonPath,
		"--out-md", mdPath,
	}, &stdout, &stderr)
	require.NoError(t, err, "stderr: %s", stderr.String())

	// Verify stdout contains score table.
	require.Contains(t, stdout.String(), "HIP-1346 LongMemEval Score Table")
	require.Contains(t, stdout.String(), "kai-evidence-gated")
	require.Contains(t, stdout.String(), "Negative Regime 1")
	require.Contains(t, stdout.String(), "Negative Regime 2")
	require.Contains(t, stdout.String(), "Negative Regime 3")

	// Verify JSON artifact parses into BenchmarkReport cleanly.
	rawJSON, err := os.ReadFile(jsonPath)
	require.NoError(t, err)
	var rep longmemeval.BenchmarkReport
	require.NoError(t, json.Unmarshal(rawJSON, &rep))
	require.Equal(t, int64(1346), rep.Seed)
	require.Equal(t, "HIP-1346", rep.HIP)
	require.Equal(t, 60, rep.DatasetCases)
	require.Len(t, rep.PolicyResults, 6)
	require.NotNil(t, rep.HistoricalRegime)
	require.NotNil(t, rep.LongHorizonRegime)
	require.NotNil(t, rep.AbstentionPareto)

	// Verify Markdown artifact matches stdout.
	rawMD, err := os.ReadFile(mdPath)
	require.NoError(t, err)
	require.Equal(t, stdout.String(), string(rawMD))
}

func TestCLI_CategoryFilterAndQuietFlag(t *testing.T) {
	tmp := t.TempDir()
	jsonPath := filepath.Join(tmp, "filtered.json")

	var stdout, stderr bytes.Buffer
	err := run([]string{
		"--seed", "1346",
		"--categories", "knowledge_update,abstention",
		"--skip-negative-regimes",
		"--quiet",
		"--out-json", jsonPath,
	}, &stdout, &stderr)
	require.NoError(t, err)
	require.Empty(t, stdout.String())

	rawJSON, err := os.ReadFile(jsonPath)
	require.NoError(t, err)
	var rep longmemeval.BenchmarkReport
	require.NoError(t, json.Unmarshal(rawJSON, &rep))
	require.Equal(t, 24, rep.DatasetCases)
	require.Nil(t, rep.HistoricalRegime)
	require.Nil(t, rep.LongHorizonRegime)
	require.Nil(t, rep.AbstentionPareto)
}

func TestCLI_FlagValidationErrors(t *testing.T) {
	var stdout, stderr bytes.Buffer

	err := run([]string{"--categories", "bogus_category"}, &stdout, &stderr)
	require.Error(t, err)
	require.Contains(t, err.Error(), "unknown category")

	err = run([]string{"--horizons", "-5"}, &stdout, &stderr)
	require.Error(t, err)
	require.Contains(t, err.Error(), "invalid --horizons")

	err = run([]string{"--thresholds", "1.5"}, &stdout, &stderr)
	require.Error(t, err)
	require.Contains(t, err.Error(), "invalid --thresholds")
}
