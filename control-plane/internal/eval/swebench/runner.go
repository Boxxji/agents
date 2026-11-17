package swebench

import (
	"context"
	"errors"
	"sort"
	"sync"
	"time"
)

// StorageBackend identifies the sandbox storage layer under test.
type StorageBackend string

const (
	// BackendVFS represents the canonical Hanzo stack (vfs -> hanzoai/s3 -> visor).
	BackendVFS StorageBackend = "vfs"
	// BackendNVMe represents a plain local NVMe baseline inside Visor.
	BackendNVMe StorageBackend = "nvme"
)

// TaskStatus captures the terminal classification of a single SWE-bench task run.
type TaskStatus string

const (
	StatusPassed         TaskStatus = "passed"
	StatusFailed         TaskStatus = "failed"
	StatusTimeout        TaskStatus = "timeout"
	StatusLeakageBlocked TaskStatus = "leakage_blocked"
	StatusPolicyBlocked  TaskStatus = "policy_blocked"
	StatusError          TaskStatus = "error"
)

// IOCounters captures filesystem IO telemetry collected during sandbox execution.
type IOCounters struct {
	Backend        StorageBackend `json:"backend"`
	ReadOps        int64          `json:"read_ops"`
	WriteOps       int64          `json:"write_ops"`
	MetadataOps    int64          `json:"metadata_ops"`
	SyncOps        int64          `json:"sync_ops"`
	BytesRead      int64          `json:"bytes_read"`
	BytesWritten   int64          `json:"bytes_written"`
	ReadLatencyNS  int64          `json:"read_latency_ns"`
	WriteLatencyNS int64          `json:"write_latency_ns"`
	CacheHits      int64          `json:"cache_hits"`
	CacheMisses    int64          `json:"cache_misses"`
}

// TotalOps returns the sum of read, write, metadata, and sync operations.
func (io IOCounters) TotalOps() int64 {
	return io.ReadOps + io.WriteOps + io.MetadataOps + io.SyncOps
}

// Add accumulates other into io.
func (io *IOCounters) Add(other IOCounters) {
	io.ReadOps += other.ReadOps
	io.WriteOps += other.WriteOps
	io.MetadataOps += other.MetadataOps
	io.SyncOps += other.SyncOps
	io.BytesRead += other.BytesRead
	io.BytesWritten += other.BytesWritten
	io.ReadLatencyNS += other.ReadLatencyNS
	io.WriteLatencyNS += other.WriteLatencyNS
	io.CacheHits += other.CacheHits
	io.CacheMisses += other.CacheMisses
}

// TaskOutcome holds the verified execution metrics for one (task, backend) pair.
type TaskOutcome struct {
	InstanceID        string                 `json:"instance_id"`
	Backend           StorageBackend         `json:"backend"`
	Passed            bool                   `json:"passed"`
	Status            TaskStatus             `json:"status"`
	WallTimeMS        int64                  `json:"wall_time_ms"`
	ModelTokensIn     int64                  `json:"model_tokens_in"`
	ModelTokensOut    int64                  `json:"model_tokens_out"`
	ModelTokensCached int64                  `json:"model_tokens_cached"`
	ModelCostUSD      float64                `json:"model_cost_usd"`
	IO                IOCounters             `json:"io"`
	WorkspaceAudit    *WorkspaceAuditReport  `json:"workspace_audit,omitempty"`
	CredentialAudit   *CredentialAuditResult `json:"credential_audit,omitempty"`
	PatchApplied      bool                   `json:"patch_applied"`
	Error             string                 `json:"error,omitempty"`
}

// SandboxExecRequest is passed to VisorSandboxExecutor after pre-flight checks pass.
type SandboxExecRequest struct {
	Task        TaskSpec          `json:"task"`
	Backend     StorageBackend    `json:"backend"`
	Workspace   string            `json:"workspace,omitempty"`
	Env         map[string]string `json:"-"`
	TenantID    string            `json:"tenant_id,omitempty"`
	TokenScopes []string          `json:"token_scopes"`
}

// VisorSandboxExecutor runs a sanitized task inside a Visor sandbox using the specified backend.
type VisorSandboxExecutor interface {
	ExecuteTask(ctx context.Context, req SandboxExecRequest) (TaskOutcome, error)
}

// TaskRuntime describes per-task workspace and credential inputs for the runner.
type TaskRuntime struct {
	WorkspaceDir   string
	SanitizeConfig *SanitizeConfig
	Credentials    CredentialDescriptor
}

// RuntimeResolver supplies workspace and credential bindings per (task, backend).
type RuntimeResolver func(ctx context.Context, task TaskSpec, backend StorageBackend) (TaskRuntime, error)

// RunnerConfig controls execution concurrency and pre-flight enforcement.
type RunnerConfig struct {
	Backends           []StorageBackend
	MaxWorkers         int
	AutoSanitize       bool
	CredentialPolicy   CredentialPolicy
	FailOnPolicyErrors bool
}

// Runner executes SWE-bench Pro V2 subsets reproducibly across storage backends.
type Runner struct {
	executor VisorSandboxExecutor
	resolver RuntimeResolver
	cfg      RunnerConfig
}

// NewRunner constructs a Runner with strict sep-2026 anti-leakage and credential enforcement.
func NewRunner(executor VisorSandboxExecutor, resolver RuntimeResolver, cfg RunnerConfig) (*Runner, error) {
	if executor == nil {
		return nil, errors.New("swebench: executor is required")
	}
	if len(cfg.Backends) == 0 {
		cfg.Backends = []StorageBackend{BackendVFS, BackendNVMe}
	}
	if cfg.MaxWorkers <= 0 {
		cfg.MaxWorkers = 1
	}
	if cfg.CredentialPolicy.RequiredKeyEnv == "" && len(cfg.CredentialPolicy.AllowedScopes) == 0 {
		cfg.CredentialPolicy = DefaultSandboxCredentialPolicy(cfg.CredentialPolicy.AllowedTenantID)
	}
	return &Runner{
		executor: executor,
		resolver: resolver,
		cfg:      cfg,
	}, nil
}

// Run executes the validated subset on all configured backends and returns a comparison report.
func (r *Runner) Run(ctx context.Context, spec *SubsetSpec) (*ComparisonReport, error) {
	if err := spec.Validate(); err != nil {
		return nil, err
	}

	type workItem struct {
		index   int
		task    TaskSpec
		backend StorageBackend
	}

	items := make([]workItem, 0, len(spec.Tasks)*len(r.cfg.Backends))
	for _, backend := range r.cfg.Backends {
		for _, task := range spec.Tasks {
			items = append(items, workItem{
				index:   len(items),
				task:    task,
				backend: backend,
			})
		}
	}

	outcomes := make([]TaskOutcome, len(items))
	jobs := make(chan workItem, len(items))
	for _, item := range items {
		jobs <- item
	}
	close(jobs)

	workers := r.cfg.MaxWorkers
	if workers > len(items) {
		workers = len(items)
	}

	var (
		wg       sync.WaitGroup
		errOnce  sync.Once
		firstErr error
	)
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for item := range jobs {
				outcome, err := r.executeOne(ctx, spec, item.task, item.backend)
				outcomes[item.index] = outcome
				if err != nil && r.cfg.FailOnPolicyErrors {
					errOnce.Do(func() {
						firstErr = err
					})
				}
			}
		}()
	}
	wg.Wait()

	if firstErr != nil {
		return nil, firstErr
	}

	// Sort deterministically by backend then instance_id.
	sort.Slice(outcomes, func(i, j int) bool {
		if outcomes[i].Backend != outcomes[j].Backend {
			return outcomes[i].Backend < outcomes[j].Backend
		}
		return outcomes[i].InstanceID < outcomes[j].InstanceID
	})

	return BuildComparisonReport(spec, outcomes)
}

func (r *Runner) executeOne(ctx context.Context, spec *SubsetSpec, task TaskSpec, backend StorageBackend) (TaskOutcome, error) {
	outcome := TaskOutcome{
		InstanceID: task.InstanceID,
		Backend:    backend,
		IO:         IOCounters{Backend: backend},
	}

	var runtime TaskRuntime
	if r.resolver != nil {
		var err error
		runtime, err = r.resolver(ctx, task, backend)
		if err != nil {
			outcome.Status = StatusError
			outcome.Error = err.Error()
			return outcome, err
		}
	}

	credAudit, err := ValidateSandboxCredentials(runtime.Credentials, r.cfg.CredentialPolicy)
	outcome.CredentialAudit = credAudit
	if err != nil {
		outcome.Status = StatusPolicyBlocked
		outcome.Error = err.Error()
		return outcome, err
	}

	if runtime.WorkspaceDir != "" {
		sanCfg := DefaultSanitizeConfig(task.BaseCommit, spec.CutoffDate)
		if runtime.SanitizeConfig != nil {
			sanCfg = *runtime.SanitizeConfig
			if sanCfg.BaseCommit == "" {
				sanCfg.BaseCommit = task.BaseCommit
			}
			if sanCfg.CutoffTimestamp.IsZero() {
				sanCfg.CutoffTimestamp = spec.CutoffDate
			}
		}

		var wsAudit *WorkspaceAuditReport
		if r.cfg.AutoSanitize {
			wsAudit, err = SanitizeGitWorkspace(ctx, runtime.WorkspaceDir, sanCfg)
		} else {
			wsAudit, err = VerifyWorkspaceClean(ctx, runtime.WorkspaceDir, sanCfg)
		}
		outcome.WorkspaceAudit = wsAudit
		if err != nil {
			outcome.Status = StatusLeakageBlocked
			outcome.Error = err.Error()
			return outcome, err
		}
	}

	cleanEnv, _ := SanitizeSandboxEnv(runtime.Credentials.Env, r.cfg.CredentialPolicy)
	start := time.Now()
	execOutcome, err := r.executor.ExecuteTask(ctx, SandboxExecRequest{
		Task:        task,
		Backend:     backend,
		Workspace:   runtime.WorkspaceDir,
		Env:         cleanEnv,
		TenantID:    runtime.Credentials.TenantID,
		TokenScopes: append([]string(nil), runtime.Credentials.TokenScopes...),
	})
	elapsedMS := time.Since(start).Milliseconds()

	execOutcome.InstanceID = task.InstanceID
	execOutcome.Backend = backend
	execOutcome.IO.Backend = backend
	execOutcome.WorkspaceAudit = outcome.WorkspaceAudit
	execOutcome.CredentialAudit = outcome.CredentialAudit
	if execOutcome.WallTimeMS <= 0 {
		execOutcome.WallTimeMS = elapsedMS
	}
	if err != nil {
		if execOutcome.Status == "" {
			execOutcome.Status = StatusError
		}
		if execOutcome.Error == "" {
			execOutcome.Error = err.Error()
		}
		return execOutcome, err
	}
	if execOutcome.Passed {
		execOutcome.Status = StatusPassed
	} else if execOutcome.Status == "" {
		execOutcome.Status = StatusFailed
	}
	return execOutcome, nil
}
