package swebench

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func buildTestSubsetSpec(t *testing.T, corpus []CorpusTask) *SubsetSpec {
	t.Helper()
	tasks := make([]TaskSpec, 0, len(corpus))
	for i, ct := range corpus {
		baseCommit := fmt.Sprintf("%040x", i+1)
		tasks = append(tasks, ct.ToTaskSpec(baseCommit))
	}
	spec, err := FilterSep2026V2(tasks, SubsetFilterOptions{
		Name:       "hip-1347-go-v2-test",
		CutoffDate: DefaultSep2026Cutoff(),
		Seed:       42,
		Strict:     true,
	})
	require.NoError(t, err)
	return spec
}

func makeTenantResolver(tenantID, apiKey string, scopes []string) RuntimeResolver {
	return func(_ context.Context, _ TaskSpec, _ StorageBackend) (TaskRuntime, error) {
		return TaskRuntime{
			Credentials: CredentialDescriptor{
				Env: map[string]string{
					DefaultAPIKeyEnv: apiKey,
				},
				TokenScopes: scopes,
				TenantID:    tenantID,
				KeyID:       "key-inf-eval",
			},
		}, nil
	}
}

func TestCleanRoomGoCorpus_MatchedAgentPoliciesAndStorageParity(t *testing.T) {
	ctx := context.Background()
	corpus := BuildCleanRoomGoCorpus()
	require.Len(t, corpus, 8)
	spec := buildTestSubsetSpec(t, corpus)

	vfs, err := NewVFSStore("")
	require.NoError(t, err)
	defer vfs.Close()
	require.NoError(t, vfs.PreWarm(ctx, corpus, ScaleTiny, DefaultSep2026Cutoff()))

	policies := []struct {
		policy       AgentPolicy
		wantResolved int
	}{
		{NoOpPolicy{}, 0},
		{SingleShotPolicy{}, 4},
		{IterativeKaiGatedPolicy{MaxIterations: 3}, 8},
		{IterativeKaiGatedPolicy{PolicyLabel: "budget_capped_iterative", MaxIterations: 3, TokenBudget: 2500}, 4},
		{OracleReferencePolicy{}, 8},
	}

	for _, tc := range policies {
		t.Run(tc.policy.Name(), func(t *testing.T) {
			exec, err := NewLocalVisorExecutor(LocalVisorConfig{
				Corpus:        corpus,
				Policy:        tc.policy,
				SandboxPolicy: DefaultVisorSandboxPolicy("tenant-eval"),
				Scale:         ScaleTiny,
				CutoffDate:    DefaultSep2026Cutoff(),
				VFS:           vfs,
				ModelName:     "zen/zen-coder",
			})
			require.NoError(t, err)
			defer exec.Close()

			runner, err := NewRunner(exec, makeTenantResolver("tenant-eval", "hz_live_ai_inference_secret_001", []string{ScopeAIInference}), RunnerConfig{
				Backends:         []StorageBackend{BackendNVMe, BackendVFS},
				MaxWorkers:       4,
				CredentialPolicy: DefaultSandboxCredentialPolicy("tenant-eval"),
			})
			require.NoError(t, err)

			report, err := runner.Run(ctx, spec)
			require.NoError(t, err)
			require.NotNil(t, report.VFSvsNVMe)

			nvme := report.Summaries[BackendNVMe]
			vfsSum := report.Summaries[BackendVFS]
			require.Equal(t, tc.wantResolved, nvme.PassedTasks, "nvme resolved count mismatch")
			require.Equal(t, tc.wantResolved, vfsSum.PassedTasks, "vfs resolved count mismatch")
			require.InDelta(t, 0.0, report.VFSvsNVMe.PassRateDelta, 1e-9, "storage parity must be 0.0pp")
			require.Zero(t, nvme.TamperedTasks)
			require.Zero(t, vfsSum.TamperedTasks)

			if tc.policy.Name() == "noop_baseline" {
				for _, out := range report.Outcomes {
					require.Zero(t, out.FailToPassPassed, "noop must fail all F2P tests on %s", out.InstanceID)
					require.Equal(t, out.PassToPassTotal, out.PassToPassPassed, "noop must pass all P2P tests on %s", out.InstanceID)
				}
			}
		})
	}
}

func TestAdversarialLeakageAndTampering_FailClosed(t *testing.T) {
	ctx := context.Background()
	corpus := BuildCleanRoomGoCorpus()[:2]
	spec := buildTestSubsetSpec(t, corpus)

	t.Run("scrubbed workspace leaks zero reference artifacts to leak probe", func(t *testing.T) {
		exec, err := NewLocalVisorExecutor(LocalVisorConfig{
			Corpus:        corpus,
			Policy:        AdversarialLeakProbePolicy{},
			SandboxPolicy: DefaultVisorSandboxPolicy("tenant-sec"),
			Scale:         ScaleTiny,
		})
		require.NoError(t, err)
		defer exec.Close()

		runner, err := NewRunner(exec, makeTenantResolver("tenant-sec", "hz_live_ai_inference_secret_002", []string{ScopeAIInference}), RunnerConfig{
			Backends:         []StorageBackend{BackendNVMe, BackendVFS},
			MaxWorkers:       2,
			CredentialPolicy: DefaultSandboxCredentialPolicy("tenant-sec"),
		})
		require.NoError(t, err)

		report, err := runner.Run(ctx, spec)
		require.NoError(t, err)
		for _, out := range report.Outcomes {
			require.False(t, out.Passed)
			require.NotNil(t, out.WorkspaceAudit)
			require.True(t, out.WorkspaceAudit.Clean)
		}
	})

	t.Run("unscrubbed leakage injection is 100% caught and blocked before agent runs", func(t *testing.T) {
		exec, err := NewLocalVisorExecutor(LocalVisorConfig{
			Corpus:        corpus,
			Policy:        OracleReferencePolicy{},
			SandboxPolicy: DefaultVisorSandboxPolicy("tenant-sec"),
			Scale:         ScaleTiny,
			SkipScrub:     true, // deliberately inject unscrubbed future commit/tag/remote/gold_patch.diff
		})
		require.NoError(t, err)
		defer exec.Close()

		runner, err := NewRunner(exec, makeTenantResolver("tenant-sec", "hz_live_ai_inference_secret_003", []string{ScopeAIInference}), RunnerConfig{
			Backends:         []StorageBackend{BackendNVMe},
			MaxWorkers:       2,
			CredentialPolicy: DefaultSandboxCredentialPolicy("tenant-sec"),
		})
		require.NoError(t, err)

		report, err := runner.Run(ctx, spec)
		require.NoError(t, err)
		require.Equal(t, len(corpus), report.Summaries[BackendNVMe].BlockedTasks)
		require.Zero(t, report.Summaries[BackendNVMe].PassedTasks)
		for _, out := range report.Outcomes {
			require.Equal(t, StatusLeakageBlocked, out.Status)
			require.False(t, out.Passed)
			require.NotNil(t, out.WorkspaceAudit)
			require.False(t, out.WorkspaceAudit.Clean)
			codes := make(map[ViolationCode]bool)
			for _, v := range out.WorkspaceAudit.Violations {
				codes[v.Code] = true
			}
			require.True(t, codes[ViolationRemotePresent])
			require.True(t, codes[ViolationTagPresent])
			require.True(t, codes[ViolationFutureCommit])
			require.True(t, codes[ViolationLeakArtifact])
			require.True(t, codes[ViolationSolutionFingerprint])
		}
	})

	t.Run("tampering agent rewriting protected tests and planting symlink is 100% blocked even with valid patch", func(t *testing.T) {
		vfs, err := NewVFSStore("")
		require.NoError(t, err)
		defer vfs.Close()
		require.NoError(t, vfs.PreWarm(ctx, corpus, ScaleTiny, DefaultSep2026Cutoff()))

		// Snapshot CAS protected test digests before tamper run to verify CAS immutability!
		casBefore := make(map[string]string)
		for _, img := range vfs.images {
			for rel, want := range img.ProtectedHashes {
				got, hashErr := fileSHA256(filepath.Join(img.ImageDir, filepath.FromSlash(rel)))
				require.NoError(t, hashErr)
				require.Equal(t, want, got)
				casBefore[filepath.Join(img.ImageDir, filepath.FromSlash(rel))] = got
			}
		}

		exec, err := NewLocalVisorExecutor(LocalVisorConfig{
			Corpus:        corpus,
			Policy:        AdversarialTamperPolicy{},
			SandboxPolicy: DefaultVisorSandboxPolicy("tenant-sec"),
			Scale:         ScaleTiny,
			VFS:           vfs,
		})
		require.NoError(t, err)
		defer exec.Close()

		runner, err := NewRunner(exec, makeTenantResolver("tenant-sec", "hz_live_ai_inference_secret_004", []string{ScopeAIInference}), RunnerConfig{
			Backends:         []StorageBackend{BackendNVMe, BackendVFS},
			MaxWorkers:       2,
			CredentialPolicy: DefaultSandboxCredentialPolicy("tenant-sec"),
		})
		require.NoError(t, err)

		report, err := runner.Run(ctx, spec)
		require.NoError(t, err)
		for _, backend := range []StorageBackend{BackendNVMe, BackendVFS} {
			require.Zero(t, report.Summaries[backend].PassedTasks)
			require.Equal(t, len(corpus), report.Summaries[backend].TamperedTasks)
		}
		for _, out := range report.Outcomes {
			require.Equal(t, StatusTampered, out.Status)
			require.True(t, out.Tampered)
			require.False(t, out.Passed)
			require.NotNil(t, out.PostExecAudit)
			codes := make(map[ViolationCode]bool)
			for _, v := range out.PostExecAudit.Violations {
				codes[v.Code] = true
			}
			require.True(t, codes[ViolationHarnessTampered])
			require.True(t, codes[ViolationSymlinkEscape])
		}

		// Verify CAS store protected test files were NEVER poisoned by chmod+edit!
		for absPath, wantDigest := range casBefore {
			gotDigest, hashErr := fileSHA256(absPath)
			require.NoError(t, hashErr)
			require.Equal(t, wantDigest, gotDigest, "CAS protected file must stay detached and immutable")
		}
	})
}

func TestVisorSandboxPolicyAndMultiTenantIsolation(t *testing.T) {
	ctx := context.Background()
	corpus := BuildCleanRoomGoCorpus()[:1]
	spec := buildTestSubsetSpec(t, corpus)

	// Invalid sandbox policies must fail closed on construction.
	badPolicies := []VisorSandboxPolicy{
		{TenantID: "../escape", ReadOnlyBaseImage: true, MaxMemoryMB: 512, MaxCPUCores: 1, WallTimeout: time.Second, NetworkDisabled: true},
		{TenantID: "tenant-a", ReadOnlyBaseImage: true, MaxMemoryMB: 512, MaxCPUCores: 1, WallTimeout: time.Second, NetworkDisabled: false},
		{TenantID: "tenant-a", ReadOnlyBaseImage: true, MaxMemoryMB: 512, MaxCPUCores: 1, WallTimeout: time.Second, NetworkDisabled: true, AllowHostEscape: true},
		{TenantID: "tenant-a", ReadOnlyBaseImage: true, MaxMemoryMB: 0, MaxCPUCores: 1, WallTimeout: time.Second, NetworkDisabled: true},
	}
	for _, bp := range badPolicies {
		_, err := NewLocalVisorExecutor(LocalVisorConfig{
			Corpus:        corpus,
			SandboxPolicy: bp,
		})
		require.ErrorIs(t, err, ErrSandboxPolicyViolation)
	}

	// Cross-tenant execution must be rejected fail-closed.
	exec, err := NewLocalVisorExecutor(LocalVisorConfig{
		Corpus:        corpus,
		Policy:        OracleReferencePolicy{},
		SandboxPolicy: DefaultVisorSandboxPolicy("tenant-alpha"),
		Scale:         ScaleTiny,
	})
	require.NoError(t, err)
	defer exec.Close()

	_, err = exec.ExecuteTask(ctx, SandboxExecRequest{
		Task:        spec.Tasks[0],
		Backend:     BackendVFS,
		Env:         map[string]string{DefaultAPIKeyEnv: "hz_live_ai_inference_alpha"},
		TenantID:    "tenant-beta",
		TokenScopes: []string{ScopeAIInference},
	})
	require.ErrorIs(t, err, ErrSandboxPolicyViolation)

	// Secret redaction helper must scrub raw secrets cleanly.
	secret := "hz_live_super_secret_token_98765"
	redacted := RedactSecretsInLog("error calling upstream with key "+secret+" failed", secret)
	require.NotContains(t, redacted, secret)
	require.Contains(t, redacted, "[REDACTED]")
}

func TestVFSvsNVMe_MediumScaleIOCoWSavings(t *testing.T) {
	ctx := context.Background()
	corpus := BuildCleanRoomGoCorpus()[:2]
	spec := buildTestSubsetSpec(t, corpus)

	vfs, err := NewVFSStore("")
	require.NoError(t, err)
	defer vfs.Close()
	require.NoError(t, vfs.PreWarm(ctx, corpus, ScaleMedium, DefaultSep2026Cutoff()))

	exec, err := NewLocalVisorExecutor(LocalVisorConfig{
		Corpus:        corpus,
		Policy:        OracleReferencePolicy{},
		SandboxPolicy: DefaultVisorSandboxPolicy("tenant-io"),
		Scale:         ScaleMedium,
		VFS:           vfs,
	})
	require.NoError(t, err)
	defer exec.Close()

	runner, err := NewRunner(exec, makeTenantResolver("tenant-io", "hz_live_ai_inference_io", []string{ScopeAIInference}), RunnerConfig{
		Backends:         []StorageBackend{BackendNVMe, BackendVFS},
		MaxWorkers:       1,
		CredentialPolicy: DefaultSandboxCredentialPolicy("tenant-io"),
	})
	require.NoError(t, err)

	report, err := runner.Run(ctx, spec)
	require.NoError(t, err)
	nvme := report.Summaries[BackendNVMe]
	vfsSum := report.Summaries[BackendVFS]

	require.Equal(t, 2, nvme.PassedTasks)
	require.Equal(t, 2, vfsSum.PassedTasks)
	require.Greater(t, vfsSum.AggregateIO.HardlinkedFiles, int64(400), "VFS should hardlink immutable ballast files")
	require.Greater(t, vfsSum.AggregateIO.CacheHits, int64(0), "warm VFS should record CAS cache hits")
	require.Less(t, vfsSum.AggregateIO.BytesWritten, nvme.AggregateIO.BytesWritten/4, "warm VFS must cut bytes written by >75%")
	require.Less(t, vfsSum.AggregateIO.BytesRead, nvme.AggregateIO.BytesRead/4, "VFS CoW snapshot diff must cut bytes read by >75%")
}

func TestEnsoLiveHookAndKaiVerificationGate(t *testing.T) {
	var seenAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seenAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"diff --git a/x.go b/x.go"}}]}`))
	}))
	defer srv.Close()

	router := NewDefaultEnsoRouter("zen/zen-coder")
	router.EndpointURL = srv.URL
	out, err := router.LiveGenerateHook(context.Background(), "hz_live_ai_inference_only_xyz", "", "fix bug")
	require.NoError(t, err)
	require.Contains(t, out, "diff --git")
	require.Equal(t, "Bearer hz_live_ai_inference_only_xyz", seenAuth)

	gate := KaiVerificationGate{MaxSummaryBytes: 64}
	require.Equal(t, DecisionAccept, gate.Decide(KaiGateInput{Iteration: 1, MaxIterations: 3, TestsCompiled: true, FailingTests: 0}).Decision)
	require.Equal(t, DecisionIterate, gate.Decide(KaiGateInput{Iteration: 1, MaxIterations: 3, TestsCompiled: false, FailingTests: 2, DiagnosticTail: strings.Repeat("x", 200)}).Decision)
	require.Equal(t, DecisionAbortBudget, gate.Decide(KaiGateInput{Iteration: 1, MaxIterations: 3, TokensUsed: 4000, TokenBudget: 3500, FailingTests: 1}).Decision)
	require.Equal(t, DecisionAbortMaxIt, gate.Decide(KaiGateInput{Iteration: 3, MaxIterations: 3, FailingTests: 1}).Decision)

	// Verify symlink escape detection directly on dangling external symlink.
	tmp := t.TempDir()
	mustGit(t, tmp, time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC), "init", "-b", "main")
	writeTestFile(t, filepath.Join(tmp, "a.go"), "package a\n")
	mustGit(t, tmp, time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC), "add", ".")
	mustGit(t, tmp, time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC), "commit", "-m", "init")
	require.NoError(t, os.Symlink("/nonexistent_host_secret_file_xyz", filepath.Join(tmp, "dangling_escape")))
	rep, err := VerifyPostExecutionIntegrity(context.Background(), tmp, DefaultSanitizeConfig("", DefaultSep2026Cutoff()))
	require.ErrorIs(t, err, ErrLeakageDetected)
	require.False(t, rep.Clean)
	require.Equal(t, ViolationSymlinkEscape, rep.Violations[0].Code)
}
