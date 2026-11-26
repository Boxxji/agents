// Command swebench-pro-v2 runs the reproducible HIP-1347 SWE-bench Pro V2
// (sep-2026 anti-leakage revision) subset benchmark comparing plain NVMe vs
// Hanzo VFS-backed Visor sandboxes across matched coding-agent policies and
// honest negative regimes.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/hanzoai/agents/control-plane/internal/eval/swebench"
)

// CohortSummary aggregates all 4 acceptance dimensions for one (suite, policy, backend) cohort.
type CohortSummary struct {
	Suite              string                  `json:"suite"`
	Scale              string                  `json:"scale"`
	WarmVFS            bool                    `json:"warm_vfs"`
	Policy             string                  `json:"policy"`
	Backend            swebench.StorageBackend `json:"backend"`
	Passed             int                     `json:"passed"`
	Total              int                     `json:"total"`
	PassRate           float64                 `json:"pass_rate"`
	Blocked            int                     `json:"blocked"`
	Tampered           int                     `json:"tampered"`
	MeanWallMS         float64                 `json:"mean_wall_ms"`
	P50WallMS          int64                   `json:"p50_wall_ms"`
	P95WallMS          int64                   `json:"p95_wall_ms"`
	P99WallMS          int64                   `json:"p99_wall_ms"`
	MeanBootMS         float64                 `json:"mean_boot_ms"`
	MeanSetupScrubMS   float64                 `json:"mean_setup_scrub_ms"`
	MeanAgentLoopMS    float64                 `json:"mean_agent_loop_ms"`
	MeanPatchVerifyMS  float64                 `json:"mean_patch_verify_ms"`
	MeanTestExecMS     float64                 `json:"mean_test_exec_ms"`
	MeanSnapshotDiffMS float64                 `json:"mean_snapshot_diff_ms"`
	MeanSnapshotCloneM float64                 `json:"mean_snapshot_clone_ms"`
	TotalTokensIn      int64                   `json:"total_tokens_in"`
	TotalTokensOut     int64                   `json:"total_tokens_out"`
	TotalTokensCached  int64                   `json:"total_tokens_cached"`
	TotalCostUSD       float64                 `json:"total_cost_usd"`
	CostPerResolvedUSD float64                 `json:"cost_per_resolved_usd"`
	ReadOps            int64                   `json:"read_ops"`
	WriteOps           int64                   `json:"write_ops"`
	MetadataOps        int64                   `json:"metadata_ops"`
	SyncOps            int64                   `json:"sync_ops"`
	BytesRead          int64                   `json:"bytes_read"`
	BytesWritten       int64                   `json:"bytes_written"`
	DirtyBytes         int64                   `json:"dirty_bytes"`
	HardlinkedFiles    int64                   `json:"hardlinked_files"`
	CacheHits          int64                   `json:"cache_hits"`
	CacheMisses        int64                   `json:"cache_misses"`
	ProcIOAvailable    bool                    `json:"proc_io_available"`
	ProcRChar          int64                   `json:"proc_rchar"`
	ProcWChar          int64                   `json:"proc_wchar"`
	ProcSyscR          int64                   `json:"proc_syscr"`
	ProcSyscW          int64                   `json:"proc_syscw"`
}

// SecurityGateCheck records an automated adversarial leakage or credential gate test.
type SecurityGateCheck struct {
	Name          string `json:"name"`
	Vector        string `json:"vector"`
	ExpectedState string `json:"expected_state"`
	ActualState   string `json:"actual_state"`
	Caught        bool   `json:"caught"`
	Detail        string `json:"detail"`
}

// FullBenchmarkReport is the machine-readable raw JSON artifact emitted for HIP-1347.
type FullBenchmarkReport struct {
	Benchmark          string                 `json:"benchmark"`
	Issue              string                 `json:"issue"`
	HIP                string                 `json:"hip"`
	Revision           string                 `json:"revision"`
	Version            string                 `json:"version"`
	Model              string                 `json:"model"`
	Seed               uint64                 `json:"seed"`
	ManifestDigest     string                 `json:"manifest_digest"`
	GeneratedAt        time.Time              `json:"generated_at"`
	Environment        map[string]any         `json:"environment"`
	MainCohorts        []CohortSummary        `json:"main_cohorts"`
	ColdTinyCohorts    []CohortSummary        `json:"cold_tiny_cohorts"`
	LargeMonorepo      []CohortSummary        `json:"large_monorepo_cohorts"`
	SecurityGateChecks []SecurityGateCheck    `json:"security_gate_checks"`
	RawOutcomes        []swebench.TaskOutcome `json:"raw_outcomes"`
}

func main() {
	outDir := flag.String("output-dir", "", "Optional directory to write results.json and score_table.md")
	model := flag.String("model", "zen/zen-coder", "Enso-routed model identifier for cost accounting")
	seed := flag.Uint64("seed", 42, "Deterministic subset selection seed")
	workers := flag.Int("workers", 1, "Worker concurrency (use 1 for clean per-task /proc/self/io attribution)")
	smoke := flag.Bool("smoke", false, "Run fast 2-task smoke subset")
	flag.Parse()

	ctx := context.Background()
	report, err := RunBenchmarkSuite(ctx, SuiteOptions{
		Model:   *model,
		Seed:    *seed,
		Workers: *workers,
		Smoke:   *smoke,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "swebench-pro-v2 error: %v\n", err)
		os.Exit(1)
	}

	md := RenderScoreTableMarkdown(report)
	fmt.Print(md)

	if strings.TrimSpace(*outDir) != "" {
		if err := os.MkdirAll(*outDir, 0o755); err != nil {
			fmt.Fprintf(os.Stderr, "mkdir %s: %v\n", *outDir, err)
			os.Exit(1)
		}
		rawJSON, err := json.MarshalIndent(report, "", "  ")
		if err != nil {
			fmt.Fprintf(os.Stderr, "marshal results.json: %v\n", err)
			os.Exit(1)
		}
		if err := os.WriteFile(filepath.Join(*outDir, "results.json"), append(rawJSON, '\n'), 0o644); err != nil {
			fmt.Fprintf(os.Stderr, "write results.json: %v\n", err)
			os.Exit(1)
		}
		if err := os.WriteFile(filepath.Join(*outDir, "score_table.md"), []byte(md), 0o644); err != nil {
			fmt.Fprintf(os.Stderr, "write score_table.md: %v\n", err)
			os.Exit(1)
		}
	}
}

// SuiteOptions configures RunBenchmarkSuite.
type SuiteOptions struct {
	Model   string
	Seed    uint64
	Workers int
	Smoke   bool
}

// RunBenchmarkSuite executes the complete reproducible benchmark matrix.
func RunBenchmarkSuite(ctx context.Context, opts SuiteOptions) (*FullBenchmarkReport, error) {
	if strings.TrimSpace(opts.Model) == "" {
		opts.Model = "zen/zen-coder"
	}
	if opts.Workers <= 0 {
		opts.Workers = 1
	}

	corpus := swebench.BuildCleanRoomGoCorpus()
	if opts.Smoke && len(corpus) > 2 {
		corpus = corpus[:2]
	}

	cutoff := swebench.DefaultSep2026Cutoff()
	spec, err := buildSubset(corpus, "hip-1347-swebench-pro-v2-go", cutoff, opts.Seed)
	if err != nil {
		return nil, err
	}

	report := &FullBenchmarkReport{
		Benchmark:      "SWE-bench Pro V2 (Go Clean-Room Subset)",
		Issue:          "hanzoai/agents#84",
		HIP:            "HIP-1347",
		Revision:       swebench.RevisionSep2026,
		Version:        swebench.BenchmarkVersionV2,
		Model:          opts.Model,
		Seed:           opts.Seed,
		ManifestDigest: spec.ManifestDigest,
		GeneratedAt:    time.Now().UTC(),
		Environment: map[string]any{
			"go_version":        runtime.Version(),
			"goos":              runtime.GOOS,
			"goarch":            runtime.GOARCH,
			"num_cpu":           runtime.NumCPU(),
			"proc_io_available": swebench.ReadProcSelfIO().Available,
			"tasks_count":       len(corpus),
			"workers":           opts.Workers,
		},
	}

	// Suite A: Main comparison on realistic medium repo scale (240 files / 3.75 MiB), warm VFS vs NVMe.
	mainScale := swebench.ScaleMedium
	if opts.Smoke {
		mainScale = swebench.ScaleTiny
	}
	warmVFS, err := swebench.NewVFSStore("")
	if err != nil {
		return nil, err
	}
	defer warmVFS.Close()
	if err := warmVFS.PreWarm(ctx, corpus, mainScale, cutoff); err != nil {
		return nil, err
	}

	mainPolicies := []swebench.AgentPolicy{
		swebench.NoOpPolicy{},
		swebench.SingleShotPolicy{},
		swebench.IterativeKaiGatedPolicy{PolicyLabel: "iterative_kai_gated", MaxIterations: 3},
		swebench.IterativeKaiGatedPolicy{PolicyLabel: "budget_capped_iterative", MaxIterations: 3, TokenBudget: 2500},
		swebench.OracleReferencePolicy{},
		swebench.AdversarialLeakProbePolicy{},
		swebench.AdversarialTamperPolicy{},
	}
	if opts.Smoke {
		mainPolicies = []swebench.AgentPolicy{
			swebench.NoOpPolicy{},
			swebench.IterativeKaiGatedPolicy{PolicyLabel: "iterative_kai_gated", MaxIterations: 3},
			swebench.AdversarialTamperPolicy{},
		}
	}

	for _, pol := range mainPolicies {
		cohorts, outcomes, err := runCohortPair(ctx, "main_medium_warm", corpus, spec, pol, mainScale, true, warmVFS, opts.Model, opts.Workers, cutoff)
		if err != nil {
			return nil, err
		}
		report.MainCohorts = append(report.MainCohorts, cohorts...)
		report.RawOutcomes = append(report.RawOutcomes, outcomes...)
	}

	// Suite B (Honest Negative Regime #1): Cold-start tiny repo (8 files / 16 KiB) — cold VFS CAS vs NVMe.
	coldVFS, err := swebench.NewVFSStore("")
	if err != nil {
		return nil, err
	}
	defer coldVFS.Close()
	coldCohorts, coldOutcomes, err := runCohortPair(ctx, "neg_cold_tiny", corpus, spec, swebench.OracleReferencePolicy{}, swebench.ScaleTiny, false, coldVFS, opts.Model, opts.Workers, cutoff)
	if err != nil {
		return nil, err
	}
	report.ColdTinyCohorts = append(report.ColdTinyCohorts, coldCohorts...)
	report.RawOutcomes = append(report.RawOutcomes, coldOutcomes...)

	// Suite C (Honest Negative Regime #2): Oversized monorepo (640 files / 20 MiB) — cold NVMe git-scrub vs warm VFS CoW.
	largeCorpus := corpus
	if len(largeCorpus) > 4 {
		largeCorpus = corpus[:4]
	}
	largeScale := swebench.ScaleLarge
	if opts.Smoke {
		largeScale = swebench.ScaleMedium
	}
	largeSpec, err := buildSubset(largeCorpus, "hip-1347-large-monorepo", cutoff, opts.Seed)
	if err != nil {
		return nil, err
	}
	largeVFS, err := swebench.NewVFSStore("")
	if err != nil {
		return nil, err
	}
	defer largeVFS.Close()
	if err := largeVFS.PreWarm(ctx, largeCorpus, largeScale, cutoff); err != nil {
		return nil, err
	}
	largeCohorts, largeOutcomes, err := runCohortPair(ctx, "neg_large_monorepo", largeCorpus, largeSpec, swebench.OracleReferencePolicy{}, largeScale, true, largeVFS, opts.Model, opts.Workers, cutoff)
	if err != nil {
		return nil, err
	}
	report.LargeMonorepo = append(report.LargeMonorepo, largeCohorts...)
	report.RawOutcomes = append(report.RawOutcomes, largeOutcomes...)

	// Suite D: Adversarial leakage-injection & API-key scope isolation gate checks.
	checks, err := runSecurityGateChecks(ctx, corpus[:1], cutoff)
	if err != nil {
		return nil, err
	}
	report.SecurityGateChecks = checks

	return report, nil
}

func buildSubset(corpus []swebench.CorpusTask, name string, cutoff time.Time, seed uint64) (*swebench.SubsetSpec, error) {
	specs := make([]swebench.TaskSpec, 0, len(corpus))
	for i, ct := range corpus {
		baseSHA := fmt.Sprintf("%040x", i+1)
		specs = append(specs, ct.ToTaskSpec(baseSHA))
	}
	return swebench.FilterSep2026V2(specs, swebench.SubsetFilterOptions{
		Name:       name,
		CutoffDate: cutoff,
		Seed:       seed,
		Strict:     true,
	})
}

func runCohortPair(
	ctx context.Context,
	suiteName string,
	corpus []swebench.CorpusTask,
	spec *swebench.SubsetSpec,
	policy swebench.AgentPolicy,
	scale swebench.RepoScale,
	warmVFS bool,
	vfs *swebench.VFSStore,
	model string,
	workers int,
	cutoff time.Time,
) ([]CohortSummary, []swebench.TaskOutcome, error) {
	exec, err := swebench.NewLocalVisorExecutor(swebench.LocalVisorConfig{
		Corpus:        corpus,
		Policy:        policy,
		SandboxPolicy: swebench.DefaultVisorSandboxPolicy("tenant-hip1347"),
		Scale:         scale,
		CutoffDate:    cutoff,
		VFS:           vfs,
		ModelName:     model,
	})
	if err != nil {
		return nil, nil, err
	}
	defer exec.Close()

	resolver := func(_ context.Context, _ swebench.TaskSpec, _ swebench.StorageBackend) (swebench.TaskRuntime, error) {
		return swebench.TaskRuntime{
			Credentials: swebench.CredentialDescriptor{
				Env:         map[string]string{swebench.DefaultAPIKeyEnv: "hz_eval_ai_inference_only_token_001"},
				TokenScopes: []string{swebench.ScopeAIInference},
				TenantID:    "tenant-hip1347",
				KeyID:       "key-hip1347-inf",
			},
		}, nil
	}

	runner, err := swebench.NewRunner(exec, resolver, swebench.RunnerConfig{
		Backends:         []swebench.StorageBackend{swebench.BackendNVMe, swebench.BackendVFS},
		MaxWorkers:       workers,
		CredentialPolicy: swebench.DefaultSandboxCredentialPolicy("tenant-hip1347"),
	})
	if err != nil {
		return nil, nil, err
	}

	comp, err := runner.Run(ctx, spec)
	if err != nil {
		return nil, nil, err
	}

	var cohorts []CohortSummary
	for _, backend := range []swebench.StorageBackend{swebench.BackendNVMe, swebench.BackendVFS} {
		sum := comp.Summaries[backend]
		var filtered []swebench.TaskOutcome
		for _, o := range comp.Outcomes {
			if o.Backend == backend {
				filtered = append(filtered, o)
			}
		}
		cohorts = append(cohorts, buildCohortSummary(suiteName, scale.Name, warmVFS, policy.Name(), backend, sum, filtered))
	}
	return cohorts, comp.Outcomes, nil
}

func buildCohortSummary(
	suite, scale string,
	warmVFS bool,
	policy string,
	backend swebench.StorageBackend,
	sum swebench.BackendSummary,
	outcomes []swebench.TaskOutcome,
) CohortSummary {
	n := float64(max(1, len(outcomes)))
	var boot, scrub, agent, patch, testExec, snapDiff, snapClone float64
	for _, o := range outcomes {
		boot += o.BootTimeMS
		scrub += o.SetupScrubMS
		agent += o.AgentLoopMS
		patch += o.PatchVerifyMS
		testExec += o.TestExecMS
		snapDiff += o.SnapshotDiffMS
		snapClone += o.IO.SnapshotCloneMS
	}
	return CohortSummary{
		Suite:              suite,
		Scale:              scale,
		WarmVFS:            warmVFS,
		Policy:             policy,
		Backend:            backend,
		Passed:             sum.PassedTasks,
		Total:              sum.TotalTasks,
		PassRate:           sum.PassRate,
		Blocked:            sum.BlockedTasks,
		Tampered:           sum.TamperedTasks,
		MeanWallMS:         round2(sum.MeanWallTimeMS),
		P50WallMS:          sum.P50WallTimeMS,
		P95WallMS:          sum.P95WallTimeMS,
		P99WallMS:          sum.P99WallTimeMS,
		MeanBootMS:         round2(boot / n),
		MeanSetupScrubMS:   round2(scrub / n),
		MeanAgentLoopMS:    round2(agent / n),
		MeanPatchVerifyMS:  round2(patch / n),
		MeanTestExecMS:     round2(testExec / n),
		MeanSnapshotDiffMS: round2(snapDiff / n),
		MeanSnapshotCloneM: round2(snapClone / n),
		TotalTokensIn:      sum.TotalTokensIn,
		TotalTokensOut:     sum.TotalTokensOut,
		TotalTokensCached:  sum.TotalTokensCached,
		TotalCostUSD:       round6(sum.TotalModelCostUSD),
		CostPerResolvedUSD: round6(sum.CostPerResolvedUSD),
		ReadOps:            sum.AggregateIO.ReadOps,
		WriteOps:           sum.AggregateIO.WriteOps,
		MetadataOps:        sum.AggregateIO.MetadataOps,
		SyncOps:            sum.AggregateIO.SyncOps,
		BytesRead:          sum.AggregateIO.BytesRead,
		BytesWritten:       sum.AggregateIO.BytesWritten,
		DirtyBytes:         sum.AggregateIO.DirtyBytes,
		HardlinkedFiles:    sum.AggregateIO.HardlinkedFiles,
		CacheHits:          sum.AggregateIO.CacheHits,
		CacheMisses:        sum.AggregateIO.CacheMisses,
		ProcIOAvailable:    sum.AggregateIO.ProcIOAvailable,
		ProcRChar:          sum.AggregateIO.ProcRChar,
		ProcWChar:          sum.AggregateIO.ProcWChar,
		ProcSyscR:          sum.AggregateIO.ProcSyscR,
		ProcSyscW:          sum.AggregateIO.ProcSyscW,
	}
}

func runSecurityGateChecks(ctx context.Context, corpus []swebench.CorpusTask, cutoff time.Time) ([]SecurityGateCheck, error) {
	spec, err := buildSubset(corpus, "hip-1347-sec-gate", cutoff, 42)
	if err != nil {
		return nil, err
	}
	var checks []SecurityGateCheck

	// 1. Leakage injection (unscrubbed repo with future commit + tag + remote + gold_patch.diff + fingerprint).
	leakExec, err := swebench.NewLocalVisorExecutor(swebench.LocalVisorConfig{
		Corpus:        corpus,
		Policy:        swebench.OracleReferencePolicy{},
		SandboxPolicy: swebench.DefaultVisorSandboxPolicy("tenant-sec"),
		Scale:         swebench.ScaleTiny,
		CutoffDate:    cutoff,
		SkipScrub:     true,
	})
	if err != nil {
		return nil, err
	}
	defer leakExec.Close()
	leakRunner, err := swebench.NewRunner(leakExec, func(_ context.Context, _ swebench.TaskSpec, _ swebench.StorageBackend) (swebench.TaskRuntime, error) {
		return swebench.TaskRuntime{
			Credentials: swebench.CredentialDescriptor{
				Env:         map[string]string{swebench.DefaultAPIKeyEnv: "hz_eval_ai_inference_sec"},
				TokenScopes: []string{swebench.ScopeAIInference},
				TenantID:    "tenant-sec",
			},
		}, nil
	}, swebench.RunnerConfig{
		Backends:         []swebench.StorageBackend{swebench.BackendNVMe},
		CredentialPolicy: swebench.DefaultSandboxCredentialPolicy("tenant-sec"),
	})
	if err != nil {
		return nil, err
	}
	leakRep, err := leakRunner.Run(ctx, spec)
	if err != nil {
		return nil, err
	}
	st := leakRep.Outcomes[0].Status
	violations := 0
	if leakRep.Outcomes[0].WorkspaceAudit != nil {
		violations = len(leakRep.Outcomes[0].WorkspaceAudit.Violations)
	}
	checks = append(checks, SecurityGateCheck{
		Name:          "unscrubbed_git_leakage_injection",
		Vector:        "future commit + v-reference-fix tag + remote + gold_patch.diff + solution fingerprint",
		ExpectedState: string(swebench.StatusLeakageBlocked),
		ActualState:   string(st),
		Caught:        st == swebench.StatusLeakageBlocked && !leakRep.Outcomes[0].Passed,
		Detail:        fmt.Sprintf("%d audit violation(s) detected pre-exec", violations),
	})

	// 2. Credential scope escalation (admin / billing:write / wildcard / leaked GITHUB_TOKEN / cross-tenant).
	credCases := []struct {
		name   string
		vector string
		desc   swebench.CredentialDescriptor
	}{
		{
			name:   "excess_admin_scope",
			vector: "HANZO_API_KEY with [ai:inference, admin:write]",
			desc: swebench.CredentialDescriptor{
				Env:         map[string]string{swebench.DefaultAPIKeyEnv: "hz_eval_token_1"},
				TokenScopes: []string{swebench.ScopeAIInference, "admin:write"},
				TenantID:    "tenant-sec",
			},
		},
		{
			name:   "wildcard_scope_rejection",
			vector: "HANZO_API_KEY with [ai:*]",
			desc: swebench.CredentialDescriptor{
				Env:         map[string]string{swebench.DefaultAPIKeyEnv: "hz_eval_token_2"},
				TokenScopes: []string{"ai:*"},
				TenantID:    "tenant-sec",
			},
		},
		{
			name:   "unscoped_token_rejection",
			vector: "HANZO_API_KEY with empty scopes",
			desc: swebench.CredentialDescriptor{
				Env:         map[string]string{swebench.DefaultAPIKeyEnv: "hz_eval_token_3"},
				TokenScopes: nil,
				TenantID:    "tenant-sec",
			},
		},
		{
			name:   "forbidden_secret_env_var",
			vector: "GITHUB_TOKEN + AWS_SECRET_ACCESS_KEY present in sandbox env",
			desc: swebench.CredentialDescriptor{
				Env: map[string]string{
					swebench.DefaultAPIKeyEnv: "hz_eval_token_4",
					"GITHUB_TOKEN":            "ghp_leaked_token",
					"AWS_SECRET_ACCESS_KEY":   "aws_leaked_secret",
				},
				TokenScopes: []string{swebench.ScopeAIInference},
				TenantID:    "tenant-sec",
			},
		},
		{
			name:   "cross_tenant_isolation",
			vector: "tenant-other attempting execution in tenant-sec sandbox",
			desc: swebench.CredentialDescriptor{
				Env:         map[string]string{swebench.DefaultAPIKeyEnv: "hz_eval_token_5"},
				TokenScopes: []string{swebench.ScopeAIInference},
				TenantID:    "tenant-other",
			},
		},
	}

	policy := swebench.DefaultSandboxCredentialPolicy("tenant-sec")
	for _, cc := range credCases {
		audit, vErr := swebench.ValidateSandboxCredentials(cc.desc, policy)
		blocked := vErr != nil && !audit.Valid
		state := "allowed"
		if blocked {
			state = string(swebench.StatusPolicyBlocked)
		}
		checks = append(checks, SecurityGateCheck{
			Name:          cc.name,
			Vector:        cc.vector,
			ExpectedState: string(swebench.StatusPolicyBlocked),
			ActualState:   state,
			Caught:        blocked,
			Detail:        strings.Join(audit.Violations, "; "),
		})
	}
	return checks, nil
}

// RenderScoreTableMarkdown formats verified benchmark results into GitHub-flavored Markdown.
func RenderScoreTableMarkdown(r *FullBenchmarkReport) string {
	var b strings.Builder
	_, _ = fmt.Fprintf(&b, "# HIP-1347 SWE-bench Pro V2 Score Table (`hanzoai/agents#84`)\n\n")
	_, _ = fmt.Fprintf(&b, "- **Revision**: `%s` (`%s`)\n", r.Revision, r.Version)
	_, _ = fmt.Fprintf(&b, "- **Manifest SHA-256**: `%s`\n", r.ManifestDigest[:16]+"...")
	_, _ = fmt.Fprintf(&b, "- **Model**: `%s` | **Seed**: `%d` | **Go**: `%v` (`%v/%v`, %v vCPU)\n\n",
		r.Model, r.Seed, r.Environment["go_version"], r.Environment["goos"], r.Environment["goarch"], r.Environment["num_cpu"])

	writeTable := func(title string, cohorts []CohortSummary) {
		_, _ = fmt.Fprintf(&b, "## %s\n\n", title)
		b.WriteString("| Policy | Backend | Resolved | Pass@1 | Tampered | Wall p50 / p95 / p99 (ms) | Setup+Scrub (ms) | Test (ms) | Diff (ms) | Written (MiB) | Hardlinked | `/proc/self/io` wchar (MiB) | Total Cost (USD) | Cost/Resolved |\n")
		b.WriteString("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
		for _, c := range cohorts {
			writtenMiB := float64(c.BytesWritten) / (1024 * 1024)
			wcharMiB := float64(c.ProcWChar) / (1024 * 1024)
			cpr := "N/A"
			if c.Passed > 0 {
				cpr = fmt.Sprintf("\\$%.6f", c.CostPerResolvedUSD)
			}
			_, _ = fmt.Fprintf(&b, "| `%s` | `%s` | %d/%d | %.1f%% | %d | %d / %d / %d | %.1f | %.1f | %.2f | %.2f | %d | %.2f | \\$%.6f | %s |\n",
				c.Policy, c.Backend, c.Passed, c.Total, c.PassRate*100.0, c.Tampered,
				c.P50WallMS, c.P95WallMS, c.P99WallMS,
				c.MeanSetupScrubMS, c.MeanTestExecMS, c.MeanSnapshotDiffMS,
				writtenMiB, c.HardlinkedFiles, wcharMiB, c.TotalCostUSD, cpr)
		}
		b.WriteString("\n")
	}

	writeTable("1. Main Benchmark (`medium`: 240 files / 3.75 MiB per repo, Warm VFS CAS vs Plain NVMe)", r.MainCohorts)
	writeTable("2. Negative Regime A: Cold-Start Tiny Repo (`tiny`: 8 files / 16 KiB, Cold VFS CAS vs Plain NVMe)", r.ColdTinyCohorts)
	writeTable("3. Negative Regime B: Oversized Monorepo (`large`: 640 files / 20.0 MiB, Cold NVMe Git-Scrub vs Warm VFS CoW)", r.LargeMonorepo)

	b.WriteString("## 4. Adversarial Anti-Leakage & API-Key Scope Gate Verification\n\n")
	b.WriteString("| Check | Attack Vector | Expected | Actual | Caught |\n")
	b.WriteString("| --- | --- | --- | --- | --- |\n")
	for _, chk := range r.SecurityGateChecks {
		caught := "PASS"
		if !chk.Caught {
			caught = "FAIL"
		}
		_, _ = fmt.Fprintf(&b, "| `%s` | %s | `%s` | `%s` | **%s** |\n",
			chk.Name, chk.Vector, chk.ExpectedState, chk.ActualState, caught)
	}
	b.WriteString("\n")
	return b.String()
}

func round2(v float64) float64 {
	return float64(int64(v*100+0.5)) / 100.0
}

func round6(v float64) float64 {
	return float64(int64(v*1_000_000+0.5)) / 1_000_000.0
}
