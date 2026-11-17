package swebench

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestFilterSep2026V2_ReproducibleAndStrict(t *testing.T) {
	cutoff := time.Date(2026, 9, 15, 0, 0, 0, 0, time.UTC)
	candidates := []TaskSpec{
		{
			InstanceID:    "django__django-1002",
			Repo:          "django/django",
			BaseCommit:    "1111111abcdef1111111",
			Revision:      "sep-2026",
			Version:       "v2",
			Language:      "python",
			Difficulty:    "hard",
			TestPatchHash: "hash-b",
			BaseTimestamp: cutoff.Add(-48 * time.Hour),
		},
		{
			InstanceID:    "golang__go-1001",
			Repo:          "golang/go",
			BaseCommit:    "2222222abcdef2222222",
			Revision:      "SEP-2026",
			Version:       "pro-v2",
			Language:      "go",
			Difficulty:    "medium",
			TestPatchHash: "hash-a",
			BaseTimestamp: cutoff.Add(-24 * time.Hour),
		},
		{
			InstanceID:    "stale__repo-0001",
			Repo:          "stale/repo",
			BaseCommit:    "3333333abcdef3333333",
			Revision:      "jan-2026",
			Version:       "v2",
			Language:      "go",
			BaseTimestamp: cutoff.Add(-72 * time.Hour),
		},
		{
			InstanceID:    "leaked__repo-0002",
			Repo:          "leaked/repo",
			BaseCommit:    "4444444abcdef4444444",
			Revision:      "sep-2026",
			Version:       "v2",
			Language:      "python",
			BaseTimestamp: cutoff.Add(24 * time.Hour), // post-cutoff leakage
		},
	}

	spec1, err := FilterSep2026V2(candidates, SubsetFilterOptions{
		Name:       "wave1-smoke",
		CutoffDate: cutoff,
		Seed:       42,
	})
	require.NoError(t, err)
	require.Len(t, spec1.Tasks, 2)
	require.NoError(t, spec1.Validate())

	// Reverse input order and verify identical manifest digest and ordering.
	reversed := []TaskSpec{candidates[3], candidates[2], candidates[1], candidates[0]}
	spec2, err := FilterSep2026V2(reversed, SubsetFilterOptions{
		Name:       "wave1-smoke",
		CutoffDate: cutoff,
		Seed:       42,
	})
	require.NoError(t, err)
	require.Equal(t, spec1.ManifestDigest, spec2.ManifestDigest)
	require.Equal(t, spec1.Tasks, spec2.Tasks)

	// Strict mode rejects non-sep-2026 or post-cutoff entries.
	_, err = FilterSep2026V2(candidates, SubsetFilterOptions{
		CutoffDate: cutoff,
		Strict:     true,
	})
	require.ErrorIs(t, err, ErrInvalidSubsetSpec)

	// Language filter + MaxTasks=1.
	goOnly, err := FilterSep2026V2(candidates, SubsetFilterOptions{
		CutoffDate: cutoff,
		Languages:  []string{"go"},
		MaxTasks:   1,
	})
	require.NoError(t, err)
	require.Len(t, goOnly.Tasks, 1)
	require.Equal(t, "golang__go-1001", goOnly.Tasks[0].InstanceID)
}

func TestSanitizeAndVerifyGitWorkspace_StripsLeakageAndDetectsTampering(t *testing.T) {
	ctx := context.Background()
	repoDir := t.TempDir()

	baseTime := time.Date(2026, 9, 1, 12, 0, 0, 0, time.UTC)
	futureTime := time.Date(2026, 9, 20, 12, 0, 0, 0, time.UTC)
	cutoff := time.Date(2026, 9, 15, 0, 0, 0, 0, time.UTC)

	mustGit(t, repoDir, baseTime, "init", "-b", "main")
	writeTestFile(t, filepath.Join(repoDir, "app.go"), "package app\nfunc Buggy() int { return 1 }\n")
	writeTestFile(t, filepath.Join(repoDir, "harness_test.go"), "package app\n// immutable eval test\n")
	mustGit(t, repoDir, baseTime, "add", ".")
	mustGit(t, repoDir, baseTime, "commit", "-m", "initial commit")

	writeTestFile(t, filepath.Join(repoDir, "app.go"), "package app\nfunc Buggy() int { return 2 }\n")
	mustGit(t, repoDir, baseTime.Add(time.Hour), "add", "app.go")
	mustGit(t, repoDir, baseTime.Add(time.Hour), "commit", "-m", "base commit")
	baseSHA := mustGitTrim(t, repoDir, "rev-parse", "HEAD")

	// Add future reference fix commit, leaked tag, leaked branch, remote, stash, FETCH_HEAD, and gold patch.
	writeTestFile(t, filepath.Join(repoDir, "app.go"), "package app\n// SECRET_FUTURE_FIX\nfunc Buggy() int { return 42 }\n")
	mustGit(t, repoDir, futureTime, "add", "app.go")
	mustGit(t, repoDir, futureTime, "commit", "-m", "future reference fix")
	futureSHA := mustGitTrim(t, repoDir, "rev-parse", "HEAD")

	mustGit(t, repoDir, futureTime, "tag", "v2.0-reference-fix")
	mustGit(t, repoDir, futureTime, "branch", "fix/future-solution", futureSHA)
	mustGit(t, repoDir, futureTime, "remote", "add", "origin", "https://example.invalid/repo.git")
	writeTestFile(t, filepath.Join(repoDir, "scratch.txt"), "stashed fix")
	mustGit(t, repoDir, futureTime, "add", "scratch.txt")
	mustGit(t, repoDir, futureTime, "stash", "push", "-m", "hidden stash")
	writeTestFile(t, filepath.Join(repoDir, ".git", "FETCH_HEAD"), futureSHA+"\tbranch 'fix'\n")
	writeTestFile(t, filepath.Join(repoDir, "gold_patch.diff"), "diff --git a/app.go b/app.go\n")

	harnessBytes, err := os.ReadFile(filepath.Join(repoDir, "harness_test.go"))
	require.NoError(t, err)
	sum := sha256.Sum256(harnessBytes)
	harnessDigest := hex.EncodeToString(sum[:])

	sanCfg := DefaultSanitizeConfig(baseSHA, cutoff)
	sanCfg.ForbiddenCommits = []string{futureSHA}
	sanCfg.ExpectedHarnessHashes = map[string]string{
		"harness_test.go": harnessDigest,
	}

	// Pre-sanitization audit must detect all leakage vectors.
	dirtyReport, err := VerifyWorkspaceClean(ctx, repoDir, sanCfg)
	require.ErrorIs(t, err, ErrLeakageDetected)
	require.False(t, dirtyReport.Clean)
	codes := make(map[ViolationCode]bool)
	for _, v := range dirtyReport.Violations {
		codes[v.Code] = true
	}
	require.True(t, codes[ViolationRemotePresent])
	require.True(t, codes[ViolationTagPresent])
	require.True(t, codes[ViolationExtraBranch])
	require.True(t, codes[ViolationStashPresent])
	require.True(t, codes[ViolationReflogPresent])
	require.True(t, codes[ViolationStateFilePresent])
	require.True(t, codes[ViolationFutureCommit])
	require.True(t, codes[ViolationLeakArtifact])

	// Sanitize workspace and assert clean verification + purged future object.
	cleanReport, err := SanitizeGitWorkspace(ctx, repoDir, sanCfg)
	require.NoError(t, err)
	require.True(t, cleanReport.Clean)
	require.Equal(t, baseSHA, cleanReport.HeadCommit)
	require.Contains(t, cleanReport.StrippedRemotes, "origin")
	require.Contains(t, cleanReport.StrippedTags, "v2.0-reference-fix")
	require.Contains(t, cleanReport.RemovedPaths, "gold_patch.diff")

	// Future commit must no longer exist in the git object database.
	cmd := exec.Command("git", "cat-file", "-e", futureSHA+"^{commit}")
	cmd.Dir = repoDir
	require.Error(t, cmd.Run(), "future fix commit must be purged from object store")

	// Tampering with protected harness file must fail verification.
	writeTestFile(t, filepath.Join(repoDir, "harness_test.go"), "package app\n// tampered!\n")
	tamperReport, err := VerifyWorkspaceClean(ctx, repoDir, sanCfg)
	require.ErrorIs(t, err, ErrLeakageDetected)
	require.Len(t, tamperReport.Violations, 1)
	require.Equal(t, ViolationHarnessTampered, tamperReport.Violations[0].Code)
}

func TestValidateSandboxCredentials_EnforcesAIInferenceOnlyAndTenantIsolation(t *testing.T) {
	policy := DefaultSandboxCredentialPolicy("tenant-alpha")

	// Valid ai:inference-only descriptor passes.
	audit, err := ValidateSandboxCredentials(CredentialDescriptor{
		Env: map[string]string{
			"HANZO_API_KEY": "hz_live_inference_1234567890",
			"PATH":          "/usr/local/bin:/usr/bin",
		},
		TokenScopes: []string{ScopeAIInference},
		TenantID:    "tenant-alpha",
		KeyID:       "key-inf-1",
	}, policy)
	require.NoError(t, err)
	require.True(t, audit.Valid)
	require.Equal(t, "hz_l...90", audit.KeyPreview)
	require.Equal(t, []string{ScopeAIInference}, audit.GrantedScopes)

	// Rejects wildcard, admin scope, forbidden env vars, and cross-tenant mismatch.
	cases := []struct {
		name string
		desc CredentialDescriptor
	}{
		{
			name: "missing api key",
			desc: CredentialDescriptor{
				Env:         map[string]string{},
				TokenScopes: []string{ScopeAIInference},
				TenantID:    "tenant-alpha",
			},
		},
		{
			name: "wildcard scope",
			desc: CredentialDescriptor{
				Env:         map[string]string{"HANZO_API_KEY": "hz_live_123456789"},
				TokenScopes: []string{"ai:*"},
				TenantID:    "tenant-alpha",
			},
		},
		{
			name: "excess privilege scope",
			desc: CredentialDescriptor{
				Env:         map[string]string{"HANZO_API_KEY": "hz_live_123456789"},
				TokenScopes: []string{ScopeAIInference, "billing:write"},
				TenantID:    "tenant-alpha",
			},
		},
		{
			name: "forbidden github token leaked",
			desc: CredentialDescriptor{
				Env: map[string]string{
					"HANZO_API_KEY": "hz_live_123456789",
					"GITHUB_TOKEN":  "ghp_secret",
				},
				TokenScopes: []string{ScopeAIInference},
				TenantID:    "tenant-alpha",
			},
		},
		{
			name: "cross-tenant isolation violation",
			desc: CredentialDescriptor{
				Env:         map[string]string{"HANZO_API_KEY": "hz_live_123456789"},
				TokenScopes: []string{ScopeAIInference},
				TenantID:    "tenant-beta",
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			res, err := ValidateSandboxCredentials(tc.desc, policy)
			require.ErrorIs(t, err, ErrCredentialPolicyViolation)
			require.False(t, res.Valid)
			require.NotEmpty(t, res.Violations)
		})
	}

	cleanEnv, stripped := SanitizeSandboxEnv(map[string]string{
		"HANZO_API_KEY":         "hz_live_123456789",
		"AWS_SECRET_ACCESS_KEY": "secret",
		"GITHUB_TOKEN":          "ghp_secret",
		"LANG":                  "C.UTF-8",
	}, policy)
	require.Equal(t, []string{"AWS_SECRET_ACCESS_KEY", "GITHUB_TOKEN"}, stripped)
	require.Equal(t, "hz_live_123456789", cleanEnv["HANZO_API_KEY"])
	require.NotContains(t, cleanEnv, "GITHUB_TOKEN")
}

type stubVisorExecutor struct {
	mu    sync.Mutex
	calls int
}

func (s *stubVisorExecutor) ExecuteTask(_ context.Context, req SandboxExecRequest) (TaskOutcome, error) {
	s.mu.Lock()
	s.calls++
	s.mu.Unlock()

	if req.Backend == BackendVFS {
		return TaskOutcome{
			Passed:            true,
			WallTimeMS:        120,
			ModelTokensIn:     4000,
			ModelTokensOut:    800,
			ModelTokensCached: 2000,
			ModelCostUSD:      0.015,
			PatchApplied:      true,
			IO: IOCounters{
				ReadOps:        150,
				WriteOps:       40,
				MetadataOps:    90,
				SyncOps:        4,
				BytesRead:      4 * 1024 * 1024,
				BytesWritten:   512 * 1024,
				ReadLatencyNS:  18_000_000,
				WriteLatencyNS: 6_000_000,
				CacheHits:      110,
				CacheMisses:    40,
			},
		}, nil
	}

	return TaskOutcome{
		Passed:         req.Task.InstanceID == "repo__a-1",
		WallTimeMS:     90,
		ModelTokensIn:  4000,
		ModelTokensOut: 850,
		ModelCostUSD:   0.018,
		PatchApplied:   true,
		IO: IOCounters{
			ReadOps:        150,
			WriteOps:       40,
			MetadataOps:    20,
			SyncOps:        12,
			BytesRead:      4 * 1024 * 1024,
			BytesWritten:   512 * 1024,
			ReadLatencyNS:  8_000_000,
			WriteLatencyNS: 3_000_000,
		},
	}, nil
}

func TestRunner_ComparesVFSvsNVMeAndBlocksTaintedRuns(t *testing.T) {
	ctx := context.Background()
	cutoff := time.Date(2026, 9, 15, 0, 0, 0, 0, time.UTC)

	spec, err := FilterSep2026V2([]TaskSpec{
		{
			InstanceID:    "repo__a-1",
			Repo:          "org/a",
			BaseCommit:    "aaaaaaa1111111",
			Revision:      "sep-2026",
			Version:       "v2",
			Language:      "go",
			BaseTimestamp: cutoff.Add(-48 * time.Hour),
		},
		{
			InstanceID:    "repo__b-2",
			Repo:          "org/b",
			BaseCommit:    "bbbbbbb2222222",
			Revision:      "sep-2026",
			Version:       "v2",
			Language:      "go",
			BaseTimestamp: cutoff.Add(-24 * time.Hour),
		},
	}, SubsetFilterOptions{Name: "vfs-vs-nvme", CutoffDate: cutoff})
	require.NoError(t, err)

	execStub := &stubVisorExecutor{}
	runner, err := NewRunner(execStub, func(_ context.Context, _ TaskSpec, _ StorageBackend) (TaskRuntime, error) {
		return TaskRuntime{
			Credentials: CredentialDescriptor{
				Env:         map[string]string{"HANZO_API_KEY": "hz_test_inference_key"},
				TokenScopes: []string{ScopeAIInference},
				TenantID:    "tenant-eval",
			},
		}, nil
	}, RunnerConfig{
		Backends:         []StorageBackend{BackendVFS, BackendNVMe},
		MaxWorkers:       4,
		CredentialPolicy: DefaultSandboxCredentialPolicy("tenant-eval"),
	})
	require.NoError(t, err)

	report, err := runner.Run(ctx, spec)
	require.NoError(t, err)
	require.NotNil(t, report.VFSvsNVMe)
	require.Equal(t, 2, report.Summaries[BackendVFS].TotalTasks)
	require.Equal(t, 1.0, report.Summaries[BackendVFS].PassRate)
	require.Equal(t, 0.5, report.Summaries[BackendNVMe].PassRate)
	require.InDelta(t, 0.030, report.Summaries[BackendVFS].TotalModelCostUSD, 1e-9)
	require.InDelta(t, 0.036, report.Summaries[BackendNVMe].TotalModelCostUSD, 1e-9)
	require.Equal(t, int64(300), report.Summaries[BackendVFS].AggregateIO.ReadOps)
	require.Equal(t, int64(140), report.VFSvsNVMe.MetadataOpsDelta)
	require.Contains(t, report.FormatMarkdown(), "SWE-bench Pro V2 (sep-2026 / v2)")

	// Pre-flight must block execution when credentials violate ai:inference-only policy.
	var invoked atomic.Int64
	blockingExec := &countingExecutor{count: &invoked}
	badRunner, err := NewRunner(blockingExec, func(_ context.Context, _ TaskSpec, _ StorageBackend) (TaskRuntime, error) {
		return TaskRuntime{
			Credentials: CredentialDescriptor{
				Env:         map[string]string{"HANZO_API_KEY": "hz_test_key", "GITHUB_TOKEN": "leaked"},
				TokenScopes: []string{ScopeAIInference, "admin"},
				TenantID:    "tenant-eval",
			},
		}, nil
	}, RunnerConfig{
		Backends:         []StorageBackend{BackendVFS},
		CredentialPolicy: DefaultSandboxCredentialPolicy("tenant-eval"),
	})
	require.NoError(t, err)

	blockedReport, err := badRunner.Run(ctx, spec)
	require.NoError(t, err)
	require.Zero(t, invoked.Load(), "executor must never run when policy check fails")
	require.Equal(t, 2, blockedReport.Summaries[BackendVFS].BlockedTasks)
	for _, out := range blockedReport.Outcomes {
		require.Equal(t, StatusPolicyBlocked, out.Status)
	}

	// Pre-flight must block tainted git workspaces when AutoSanitize=false, and succeed when AutoSanitize=true.
	dirtyRepo := t.TempDir()
	baseWhen := cutoff.Add(-48 * time.Hour)
	mustGit(t, dirtyRepo, baseWhen, "init", "-b", "main")
	writeTestFile(t, filepath.Join(dirtyRepo, "main.go"), "package main\n")
	mustGit(t, dirtyRepo, baseWhen, "add", "main.go")
	mustGit(t, dirtyRepo, baseWhen, "commit", "-m", "base")
	repoBase := mustGitTrim(t, dirtyRepo, "rev-parse", "HEAD")
	mustGit(t, dirtyRepo, baseWhen, "remote", "add", "origin", "https://example.invalid/leak.git")

	singleSpec, err := FilterSep2026V2([]TaskSpec{{
		InstanceID:    "dirty__repo-1",
		Repo:          "org/dirty",
		BaseCommit:    repoBase,
		Revision:      "sep-2026",
		Version:       "v2",
		Language:      "go",
		BaseTimestamp: baseWhen,
	}}, SubsetFilterOptions{CutoffDate: cutoff})
	require.NoError(t, err)

	wsResolver := func(_ context.Context, _ TaskSpec, _ StorageBackend) (TaskRuntime, error) {
		return TaskRuntime{
			WorkspaceDir: dirtyRepo,
			Credentials: CredentialDescriptor{
				Env:         map[string]string{"HANZO_API_KEY": "hz_test_key"},
				TokenScopes: []string{ScopeAIInference},
				TenantID:    "tenant-eval",
			},
		}, nil
	}

	var wsInvoked atomic.Int64
	noSanRunner, err := NewRunner(&countingExecutor{count: &wsInvoked}, wsResolver, RunnerConfig{
		Backends:         []StorageBackend{BackendVFS},
		AutoSanitize:     false,
		CredentialPolicy: DefaultSandboxCredentialPolicy("tenant-eval"),
	})
	require.NoError(t, err)
	dirtyRunReport, err := noSanRunner.Run(ctx, singleSpec)
	require.NoError(t, err)
	require.Zero(t, wsInvoked.Load())
	require.Equal(t, StatusLeakageBlocked, dirtyRunReport.Outcomes[0].Status)

	autoSanRunner, err := NewRunner(&countingExecutor{count: &wsInvoked}, wsResolver, RunnerConfig{
		Backends:         []StorageBackend{BackendVFS},
		AutoSanitize:     true,
		CredentialPolicy: DefaultSandboxCredentialPolicy("tenant-eval"),
	})
	require.NoError(t, err)
	cleanRunReport, err := autoSanRunner.Run(ctx, singleSpec)
	require.NoError(t, err)
	require.Equal(t, int64(1), wsInvoked.Load())
	require.Equal(t, StatusPassed, cleanRunReport.Outcomes[0].Status)
	require.True(t, cleanRunReport.Outcomes[0].WorkspaceAudit.Clean)
}

type countingExecutor struct {
	count *atomic.Int64
}

func (c *countingExecutor) ExecuteTask(_ context.Context, _ SandboxExecRequest) (TaskOutcome, error) {
	c.count.Add(1)
	return TaskOutcome{Passed: true}, nil
}

func mustGit(t *testing.T, dir string, when time.Time, args ...string) {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	stamp := when.UTC().Format(time.RFC3339)
	cmd.Env = append(os.Environ(),
		"GIT_CONFIG_GLOBAL=/dev/null",
		"GIT_CONFIG_NOSYSTEM=1",
		"GIT_TERMINAL_PROMPT=0",
		"GIT_AUTHOR_NAME=Eval",
		"GIT_AUTHOR_EMAIL=eval@hanzo.ai",
		"GIT_COMMITTER_NAME=Eval",
		"GIT_COMMITTER_EMAIL=eval@hanzo.ai",
		"GIT_AUTHOR_DATE="+stamp,
		"GIT_COMMITTER_DATE="+stamp,
	)
	out, err := cmd.CombinedOutput()
	require.NoError(t, err, "git %v failed: %s", args, string(out))
}

func mustGitTrim(t *testing.T, dir string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "GIT_CONFIG_GLOBAL=/dev/null", "GIT_CONFIG_NOSYSTEM=1")
	out, err := cmd.CombinedOutput()
	require.NoError(t, err, "git %v failed: %s", args, string(out))
	return string(bytesTrimSpace(out))
}

func bytesTrimSpace(b []byte) []byte {
	start := 0
	for start < len(b) && (b[start] == ' ' || b[start] == '\n' || b[start] == '\r' || b[start] == '\t') {
		start++
	}
	end := len(b)
	for end > start && (b[end-1] == ' ' || b[end-1] == '\n' || b[end-1] == '\r' || b[end-1] == '\t') {
		end--
	}
	return b[start:end]
}

func writeTestFile(t *testing.T, path, content string) {
	t.Helper()
	require.NoError(t, os.MkdirAll(filepath.Dir(path), 0o755))
	require.NoError(t, os.WriteFile(path, []byte(content), 0o644))
}
