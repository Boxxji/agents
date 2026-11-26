package main

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestRunBenchmarkSuite_Smoke(t *testing.T) {
	rep, err := RunBenchmarkSuite(context.Background(), SuiteOptions{
		Model:   "zen/zen-coder",
		Seed:    42,
		Workers: 1,
		Smoke:   true,
	})
	require.NoError(t, err)
	require.NotEmpty(t, rep.ManifestDigest)
	require.NotEmpty(t, rep.MainCohorts)
	require.NotEmpty(t, rep.ColdTinyCohorts)
	require.NotEmpty(t, rep.LargeMonorepo)
	require.Len(t, rep.SecurityGateChecks, 6)
	for _, chk := range rep.SecurityGateChecks {
		require.True(t, chk.Caught, "security gate check %s must be caught", chk.Name)
	}

	// Verify JSON serialization roundtrip and Markdown rendering.
	raw, err := json.Marshal(rep)
	require.NoError(t, err)
	require.NotEmpty(t, raw)

	md := RenderScoreTableMarkdown(rep)
	require.Contains(t, md, "HIP-1347 SWE-bench Pro V2 Score Table")
	require.Contains(t, md, "iterative_kai_gated")
	require.Contains(t, md, "adversarial_tamper_agent")
}
