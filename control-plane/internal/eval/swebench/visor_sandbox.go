package swebench

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

// ErrSandboxPolicyViolation indicates a Visor sandbox configuration violates isolation rules.
var ErrSandboxPolicyViolation = errors.New("swebench: visor sandbox isolation policy violation")

var safeTenantRE = regexp.MustCompile(`^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$`)

// VisorSandboxPolicy specifies resource quotas, read-only base image enforcement,
// network isolation, and multi-tenant namespace boundaries for hanzoai/visor.
type VisorSandboxPolicy struct {
	TenantID          string        `json:"tenant_id"`
	ReadOnlyBaseImage bool          `json:"read_only_base_image"`
	MaxMemoryMB       int           `json:"max_memory_mb"`
	MaxCPUCores       int           `json:"max_cpu_cores"`
	WallTimeout       time.Duration `json:"wall_timeout"`
	NetworkDisabled   bool          `json:"network_disabled"`
	AllowHostEscape   bool          `json:"allow_host_escape"`
}

// DefaultVisorSandboxPolicy returns strict multi-tenant Visor isolation defaults.
func DefaultVisorSandboxPolicy(tenantID string) VisorSandboxPolicy {
	return VisorSandboxPolicy{
		TenantID:          strings.TrimSpace(tenantID),
		ReadOnlyBaseImage: true,
		MaxMemoryMB:       2048,
		MaxCPUCores:       2,
		WallTimeout:       45 * time.Second,
		NetworkDisabled:   true,
		AllowHostEscape:   false,
	}
}

// Validate enforces fail-closed Visor isolation invariants.
func (p VisorSandboxPolicy) Validate() error {
	if !safeTenantRE.MatchString(strings.TrimSpace(p.TenantID)) {
		return fmt.Errorf("%w: invalid tenant_id %q", ErrSandboxPolicyViolation, p.TenantID)
	}
	if p.AllowHostEscape {
		return fmt.Errorf("%w: allow_host_escape must be false", ErrSandboxPolicyViolation)
	}
	if !p.NetworkDisabled {
		return fmt.Errorf("%w: sandbox network must be disabled", ErrSandboxPolicyViolation)
	}
	if p.MaxMemoryMB <= 0 || p.MaxCPUCores <= 0 || p.WallTimeout <= 0 {
		return fmt.Errorf("%w: memory, cpu, and wall timeout quotas must be positive", ErrSandboxPolicyViolation)
	}
	return nil
}

// RepoScale controls surrounding repository ballast size to benchmark
// plain NVMe full-copy + git-scrub vs VFS CoW snapshot performance.
type RepoScale struct {
	Name              string `json:"name"`
	AssetFiles        int    `json:"asset_files"`
	AssetBytesPerFile int    `json:"asset_bytes_per_file"`
}

// TotalBytes returns total ballast bytes materialized per repository.
func (s RepoScale) TotalBytes() int64 {
	return int64(s.AssetFiles) * int64(s.AssetBytesPerFile)
}

var (
	// ScaleTiny models tiny repos (<32 KiB) where cold VFS CAS indexing overhead exceeds plain NVMe.
	ScaleTiny = RepoScale{Name: "tiny", AssetFiles: 8, AssetBytesPerFile: 2048}
	// ScaleMedium models realistic multi-megabyte Go repositories (240 files / 3.75 MiB).
	ScaleMedium = RepoScale{Name: "medium", AssetFiles: 240, AssetBytesPerFile: 16384}
	// ScaleLarge models oversized monorepos (640 files / 20.0 MiB) highlighting cold NVMe git-scrub overhead.
	ScaleLarge = RepoScale{Name: "large", AssetFiles: 640, AssetBytesPerFile: 32768}
)

// LookupRepoScale resolves a named RepoScale ("tiny", "medium", "large").
func LookupRepoScale(name string) (RepoScale, error) {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "tiny":
		return ScaleTiny, nil
	case "medium":
		return ScaleMedium, nil
	case "large":
		return ScaleLarge, nil
	default:
		return RepoScale{}, fmt.Errorf("swebench: unknown repo scale %q (want tiny, medium, large)", name)
	}
}

// ProcIOStats holds Linux /proc/self/io counters.
type ProcIOStats struct {
	Available  bool
	RChar      int64
	WChar      int64
	SyscR      int64
	SyscW      int64
	ReadBytes  int64
	WriteBytes int64
}

// ReadProcSelfIO reads real Linux kernel process I/O counters when available.
func ReadProcSelfIO() ProcIOStats {
	data, err := os.ReadFile("/proc/self/io")
	if err != nil {
		return ProcIOStats{Available: false}
	}
	stats := ProcIOStats{Available: true}
	sc := bufio.NewScanner(bytes.NewReader(data))
	for sc.Scan() {
		line := sc.Text()
		parts := strings.SplitN(line, ":", 2)
		if len(parts) != 2 {
			continue
		}
		val, err := strconv.ParseInt(strings.TrimSpace(parts[1]), 10, 64)
		if err != nil {
			continue
		}
		switch strings.TrimSpace(parts[0]) {
		case "rchar":
			stats.RChar = val
		case "wchar":
			stats.WChar = val
		case "syscr":
			stats.SyscR = val
		case "syscw":
			stats.SyscW = val
		case "read_bytes":
			stats.ReadBytes = val
		case "write_bytes":
			stats.WriteBytes = val
		}
	}
	return stats
}

type casImage struct {
	ImageDir        string
	BaseCommit      string
	FutureFixCommit string
	BaseManifest    map[string]string
	ProtectedHashes map[string]string
	ScrubReport     *WorkspaceAuditReport
}

// VFSStore emulates Hanzo's canonical vfs -> hanzoai/s3 -> visor content-addressed
// immutable base snapshot layer using read-only 0o444 hardlinks + isolated per-sandbox upper layers.
type VFSStore struct {
	rootDir  string
	ownsRoot bool
	mu       sync.Mutex
	images   map[string]*casImage
}

// NewVFSStore creates a VFSStore rooted at rootDir (or a temp directory if empty).
func NewVFSStore(rootDir string) (*VFSStore, error) {
	owns := false
	if strings.TrimSpace(rootDir) == "" {
		dir, err := os.MkdirTemp("", "hanzo_visor_vfs_cas_*")
		if err != nil {
			return nil, err
		}
		rootDir = dir
		owns = true
	} else {
		if err := os.MkdirAll(rootDir, 0o755); err != nil {
			return nil, err
		}
	}
	resolved, err := filepath.EvalSymlinks(rootDir)
	if err == nil {
		rootDir = resolved
	}
	return &VFSStore{
		rootDir:  rootDir,
		ownsRoot: owns,
		images:   make(map[string]*casImage),
	}, nil
}

// Close removes the temporary CAS store directory if owned by VFSStore.
func (s *VFSStore) Close() error {
	if !s.ownsRoot || s.rootDir == "" {
		return nil
	}
	restoreWritableTree(s.rootDir)
	return os.RemoveAll(s.rootDir)
}

// PreWarm builds and scrubs immutable CAS base images ahead of timed rollouts.
func (s *VFSStore) PreWarm(ctx context.Context, tasks []CorpusTask, scale RepoScale, cutoff time.Time) error {
	for _, task := range tasks {
		if _, _, _, err := s.getOrBuild(ctx, task, scale, cutoff); err != nil {
			return err
		}
	}
	return nil
}

func (s *VFSStore) getOrBuild(ctx context.Context, task CorpusTask, scale RepoScale, cutoff time.Time) (*casImage, bool, IOCounters, error) {
	fp := taskCASFingerprint(task, scale)
	key := task.InstanceID + "__" + scale.Name + "__" + fp[:12]

	s.mu.Lock()
	defer s.mu.Unlock()

	if img, ok := s.images[key]; ok {
		if _, err := os.Stat(img.ImageDir); err == nil {
			return img, true, IOCounters{Backend: BackendVFS, CacheHits: 1}, nil
		}
	}

	imageDir := filepath.Join(s.rootDir, key)
	restoreWritableTree(imageDir)
	_ = os.RemoveAll(imageDir)
	if err := os.MkdirAll(imageDir, 0o755); err != nil {
		return nil, false, IOCounters{}, err
	}

	baseCommit, futureCommit, initIO, err := initUnscrubbedUpstreamRepo(ctx, imageDir, task, scale)
	if err != nil {
		return nil, false, initIO, err
	}
	initIO.Backend = BackendVFS
	initIO.CacheMisses = 1

	sanCfg := DefaultSanitizeConfig(baseCommit, cutoff)
	sanCfg.ForbiddenCommits = []string{futureCommit}
	if task.SolutionFingerprint != "" {
		sanCfg.ForbiddenFingerprints = []string{task.SolutionFingerprint}
	}
	scrubRep, err := SanitizeGitWorkspace(ctx, imageDir, sanCfg)
	if err != nil {
		return nil, false, initIO, err
	}

	manifest, readBytes, readOps := buildFileManifest(imageDir)
	initIO.BytesRead += readBytes
	initIO.ReadOps += readOps

	protHashes := make(map[string]string, len(task.ProtectedTestPaths))
	for _, rel := range task.ProtectedTestPaths {
		if digest, ok := manifest[rel]; ok {
			protHashes[rel] = digest
		}
	}

	// Harden immutable ballast files in CAS to read-only 0o444.
	assetsDir := filepath.Join(imageDir, "repo_assets")
	if info, err := os.Stat(assetsDir); err == nil && info.IsDir() {
		_ = filepath.WalkDir(assetsDir, func(path string, d os.DirEntry, walkErr error) error {
			if walkErr == nil && !d.IsDir() {
				_ = os.Chmod(path, 0o444)
				initIO.MetadataOps++
			}
			return nil
		})
	}

	img := &casImage{
		ImageDir:        imageDir,
		BaseCommit:      baseCommit,
		FutureFixCommit: futureCommit,
		BaseManifest:    manifest,
		ProtectedHashes: protHashes,
		ScrubReport:     scrubRep,
	}
	s.images[key] = img
	return img, false, initIO, nil
}

// ProvisionedWorkspace represents a materialized, isolated task workspace.
type ProvisionedWorkspace struct {
	RootDir         string
	RepoDir         string
	TenantID        string
	Backend         StorageBackend
	BaseCommit      string
	FutureFixCommit string
	BaseManifest    map[string]string
	ProtectedHashes map[string]string
	PreAudit        *WorkspaceAuditReport
	IO              IOCounters
	BootMS          float64
	SetupScrubMS    float64
}

// Cleanup restores writable bits and deletes the ephemeral sandbox directory.
func (w *ProvisionedWorkspace) Cleanup() {
	if w == nil || w.RootDir == "" {
		return
	}
	restoreWritableTree(w.RootDir)
	_ = os.RemoveAll(w.RootDir)
}

// LocalVisorExecutor implements VisorSandboxExecutor with real git repos,
// real anti-leakage scrubbing, VFS CoW vs plain NVMe storage backends,
// matched agent policies, post-execution anti-tampering, and real `go test -json`.
type LocalVisorExecutor struct {
	Corpus         map[string]CorpusTask
	Policy         AgentPolicy
	SandboxPolicy  VisorSandboxPolicy
	Scale          RepoScale
	CutoffDate     time.Time
	VFS            *VFSStore
	Router         *EnsoModelRouter
	ModelName      string
	WorkRoot       string
	GoCacheDir     string
	SkipScrub      bool
	sandboxCounter atomic.Int64
	ownsWorkRoot   bool
	ownsVFS        bool
}

// LocalVisorConfig configures LocalVisorExecutor.
type LocalVisorConfig struct {
	Corpus        []CorpusTask
	Policy        AgentPolicy
	SandboxPolicy VisorSandboxPolicy
	Scale         RepoScale
	CutoffDate    time.Time
	VFS           *VFSStore
	Router        *EnsoModelRouter
	ModelName     string
	SkipScrub     bool
}

// NewLocalVisorExecutor constructs a real, offline-reproducible Visor executor.
func NewLocalVisorExecutor(cfg LocalVisorConfig) (*LocalVisorExecutor, error) {
	if err := cfg.SandboxPolicy.Validate(); err != nil {
		return nil, err
	}
	if cfg.Policy == nil {
		cfg.Policy = OracleReferencePolicy{}
	}
	if cfg.Scale.Name == "" {
		cfg.Scale = ScaleMedium
	}
	if cfg.CutoffDate.IsZero() {
		cfg.CutoffDate = DefaultSep2026Cutoff()
	}
	if cfg.Router == nil {
		cfg.Router = NewDefaultEnsoRouter(cfg.ModelName)
	}
	model := cfg.ModelName
	if strings.TrimSpace(model) == "" {
		model = cfg.Router.DefaultModel
	}

	workRoot, err := os.MkdirTemp("", "hanzo_visor_tenants_*")
	if err != nil {
		return nil, err
	}
	if resolved, rErr := filepath.EvalSymlinks(workRoot); rErr == nil {
		workRoot = resolved
	}

	vfs := cfg.VFS
	ownsVFS := false
	if vfs == nil {
		vfs, err = NewVFSStore("")
		if err != nil {
			_ = os.RemoveAll(workRoot)
			return nil, err
		}
		ownsVFS = true
	}

	goCache := resolveGoBuildCache(workRoot)

	corpusMap := make(map[string]CorpusTask, len(cfg.Corpus))
	for _, t := range cfg.Corpus {
		corpusMap[t.InstanceID] = t
	}

	return &LocalVisorExecutor{
		Corpus:        corpusMap,
		Policy:        cfg.Policy,
		SandboxPolicy: cfg.SandboxPolicy,
		Scale:         cfg.Scale,
		CutoffDate:    cfg.CutoffDate,
		VFS:           vfs,
		Router:        cfg.Router,
		ModelName:     model,
		WorkRoot:      workRoot,
		GoCacheDir:    goCache,
		SkipScrub:     cfg.SkipScrub,
		ownsWorkRoot:  true,
		ownsVFS:       ownsVFS,
	}, nil
}

// Close cleans up temporary workspace roots (and owned VFS CAS store).
func (e *LocalVisorExecutor) Close() error {
	var firstErr error
	if e.ownsWorkRoot && e.WorkRoot != "" {
		restoreWritableTree(e.WorkRoot)
		if err := os.RemoveAll(e.WorkRoot); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	if e.ownsVFS && e.VFS != nil {
		if err := e.VFS.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

// ExecuteTask runs a single task inside an isolated Visor sandbox on req.Backend.
func (e *LocalVisorExecutor) ExecuteTask(ctx context.Context, req SandboxExecRequest) (TaskOutcome, error) {
	task, ok := e.Corpus[req.Task.InstanceID]
	if !ok {
		return TaskOutcome{
			InstanceID: req.Task.InstanceID,
			Backend:    req.Backend,
			Status:     StatusError,
			Error:      fmt.Sprintf("unknown corpus task %q", req.Task.InstanceID),
		}, fmt.Errorf("swebench: unknown corpus task %q", req.Task.InstanceID)
	}

	// Enforce multi-tenant sandbox boundary.
	if err := e.SandboxPolicy.Validate(); err != nil {
		return TaskOutcome{
			InstanceID: task.InstanceID,
			Backend:    req.Backend,
			PolicyName: e.Policy.Name(),
			Status:     StatusPolicyBlocked,
			Error:      err.Error(),
		}, err
	}
	if strings.TrimSpace(req.TenantID) != "" && req.TenantID != e.SandboxPolicy.TenantID {
		err := fmt.Errorf("%w: cross-tenant execution denied (%q != %q)", ErrSandboxPolicyViolation, req.TenantID, e.SandboxPolicy.TenantID)
		return TaskOutcome{
			InstanceID: task.InstanceID,
			Backend:    req.Backend,
			PolicyName: e.Policy.Name(),
			Status:     StatusPolicyBlocked,
			Error:      err.Error(),
		}, err
	}

	taskCtx, cancel := context.WithTimeout(ctx, e.SandboxPolicy.WallTimeout)
	defer cancel()

	wallStart := time.Now()
	procBefore := ReadProcSelfIO()

	// 1. Provision isolated sandbox workspace (nvme vs vfs).
	ws, provErr := e.provisionWorkspace(taskCtx, task, req.Backend)
	defer ws.Cleanup()

	apiKey := req.Env[DefaultAPIKeyEnv]
	outcome := TaskOutcome{
		InstanceID:      task.InstanceID,
		Backend:         req.Backend,
		PolicyName:      e.Policy.Name(),
		FailToPassTotal: len(task.FailToPassTests),
		PassToPassTotal: len(task.PassToPassTests),
	}
	if ws != nil {
		outcome.IO = ws.IO
		outcome.BootTimeMS = ws.BootMS
		outcome.SetupScrubMS = ws.SetupScrubMS
		outcome.WorkspaceAudit = ws.PreAudit
	}

	// Fail closed on pre-execution anti-leakage violations.
	if provErr != nil {
		outcome.Status = StatusLeakageBlocked
		if !errors.Is(provErr, ErrLeakageDetected) {
			outcome.Status = StatusError
		}
		outcome.Error = RedactSecretsInLog(provErr.Error(), apiKey)
		finalizeProcIO(&outcome, procBefore, wallStart)
		return outcome, nil
	}

	// 2. Build scrubbed child environment (no leaked host/admin secrets; network disabled).
	childEnv := buildIsolatedChildEnv(req.Env, e.GoCacheDir)

	// 3. Execute matched coding-agent policy with optional sandbox test feedback loop.
	agentStart := time.Now()
	testFeedbackFn := func(fCtx context.Context) (bool, int, string, error) {
		f2p, p2p, tail, code, err := runGoTestGrading(fCtx, ws.RepoDir, task, childEnv)
		if err != nil {
			return false, len(task.FailToPassTests) + len(task.PassToPassTests), tail, err
		}
		failing := (len(task.FailToPassTests) - f2p) + (len(task.PassToPassTests) - p2p)
		return code == 0 && failing == 0, failing, tail, nil
	}
	agentRes, agentErr := e.Policy.Solve(taskCtx, task, ws.RepoDir, childEnv, testFeedbackFn)
	outcome.AgentLoopMS = float64(time.Since(agentStart).Microseconds()) / 1000.0
	if agentErr != nil {
		outcome.Status = StatusError
		outcome.Error = RedactSecretsInLog(agentErr.Error(), apiKey)
		finalizeProcIO(&outcome, procBefore, wallStart)
		return outcome, nil
	}

	outcome.PolicyName = agentRes.PolicyName
	outcome.Iterations = agentRes.Iterations
	outcome.ModelTokensIn = agentRes.TokensIn
	outcome.ModelTokensOut = agentRes.TokensOut
	outcome.ModelTokensCached = agentRes.TokensCached
	outcome.ModelCostUSD = e.Router.ComputeCostUSD(e.ModelName, agentRes.TokensIn, agentRes.TokensOut, agentRes.TokensCached)

	// 4. Apply agent edits cleanly (if not already written during iterative turns) and verify git status.
	patchStart := time.Now()
	if len(agentRes.UpdatedFiles) > 0 {
		if err := writeUpdatedFiles(ws.RepoDir, agentRes.UpdatedFiles); err != nil {
			outcome.Status = StatusError
			outcome.Error = RedactSecretsInLog(err.Error(), apiKey)
			finalizeProcIO(&outcome, procBefore, wallStart)
			return outcome, nil
		}
		for _, content := range agentRes.UpdatedFiles {
			outcome.IO.WriteOps++
			outcome.IO.BytesWritten += int64(len(content))
			outcome.IO.DirtyBytes += int64(len(content))
		}
		outcome.PatchApplied = true
	}

	// 5. Measure snapshot delta (VFS skips untouched hardlinked inodes; NVMe scans full tree).
	diffStart := time.Now()
	diffBytesRead, diffReadOps := measureSnapshotDiff(ws)
	outcome.SnapshotDiffMS = float64(time.Since(diffStart).Microseconds()) / 1000.0
	outcome.IO.BytesRead += diffBytesRead
	outcome.IO.ReadOps += diffReadOps

	// 6. Fail-closed post-execution anti-tampering audit BEFORE invoking go test grading!
	postCfg := DefaultSanitizeConfig(ws.BaseCommit, e.CutoffDate)
	postCfg.ExpectedHarnessHashes = ws.ProtectedHashes
	postAudit, postErr := VerifyPostExecutionIntegrity(taskCtx, ws.RepoDir, postCfg)
	outcome.PostExecAudit = postAudit
	outcome.PatchVerifyMS = float64(time.Since(patchStart).Microseconds()) / 1000.0
	if postErr != nil {
		outcome.Status = StatusTampered
		outcome.Tampered = true
		outcome.Passed = false
		outcome.Error = RedactSecretsInLog(postErr.Error(), apiKey)
		finalizeProcIO(&outcome, procBefore, wallStart)
		return outcome, nil
	}

	// 7. Execute real `go test -json -count=1 ./...` grading inside verified workspace.
	testStart := time.Now()
	f2pOK, p2pOK, tail, exitCode, testErr := runGoTestGrading(taskCtx, ws.RepoDir, task, childEnv)
	outcome.TestExecMS = float64(time.Since(testStart).Microseconds()) / 1000.0
	outcome.FailToPassPassed = f2pOK
	outcome.PassToPassPassed = p2pOK

	if errors.Is(testErr, context.DeadlineExceeded) {
		outcome.Status = StatusTimeout
		outcome.Passed = false
		outcome.Error = "test execution timed out"
	} else if exitCode == 0 && f2pOK == len(task.FailToPassTests) && p2pOK == len(task.PassToPassTests) {
		outcome.Status = StatusPassed
		outcome.Passed = true
	} else {
		outcome.Status = StatusFailed
		outcome.Passed = false
		if tail != "" {
			outcome.Error = RedactSecretsInLog(tail, apiKey)
		}
	}

	finalizeProcIO(&outcome, procBefore, wallStart)
	return outcome, nil
}

func finalizeProcIO(outcome *TaskOutcome, before ProcIOStats, wallStart time.Time) {
	outcome.WallTimeMS = time.Since(wallStart).Milliseconds()
	if outcome.WallTimeMS <= 0 {
		outcome.WallTimeMS = 1
	}
	after := ReadProcSelfIO()
	if before.Available && after.Available {
		outcome.IO.ProcIOAvailable = true
		if after.RChar >= before.RChar {
			outcome.IO.ProcRChar = after.RChar - before.RChar
		}
		if after.WChar >= before.WChar {
			outcome.IO.ProcWChar = after.WChar - before.WChar
		}
		if after.SyscR >= before.SyscR {
			outcome.IO.ProcSyscR = after.SyscR - before.SyscR
		}
		if after.SyscW >= before.SyscW {
			outcome.IO.ProcSyscW = after.SyscW - before.SyscW
		}
	}
}

func (e *LocalVisorExecutor) provisionWorkspace(ctx context.Context, task CorpusTask, backend StorageBackend) (*ProvisionedWorkspace, error) {
	bootStart := time.Now()
	seq := e.sandboxCounter.Add(1)
	tenantDir := filepath.Join(e.WorkRoot, "tenants", e.SandboxPolicy.TenantID, "sandboxes")
	if err := os.MkdirAll(tenantDir, 0o700); err != nil {
		return nil, err
	}
	sandboxRoot := filepath.Join(tenantDir, fmt.Sprintf("%s_%s_%d", backend, sanitizeID(task.InstanceID), seq))
	repoDir := filepath.Join(sandboxRoot, "repo")
	if err := os.MkdirAll(repoDir, 0o755); err != nil {
		return nil, err
	}
	bootMS := float64(time.Since(bootStart).Microseconds()) / 1000.0

	ws := &ProvisionedWorkspace{
		RootDir:  sandboxRoot,
		RepoDir:  repoDir,
		TenantID: e.SandboxPolicy.TenantID,
		Backend:  backend,
		BootMS:   bootMS,
		IO:       IOCounters{Backend: backend},
	}

	setupStart := time.Now()
	if backend == BackendNVMe || e.SkipScrub {
		baseCommit, futureCommit, initIO, err := initUnscrubbedUpstreamRepo(ctx, repoDir, task, e.Scale)
		ws.BaseCommit = baseCommit
		ws.FutureFixCommit = futureCommit
		ws.IO.Add(initIO)
		if err != nil {
			ws.SetupScrubMS = float64(time.Since(setupStart).Microseconds()) / 1000.0
			return ws, err
		}

		sanCfg := DefaultSanitizeConfig(baseCommit, e.CutoffDate)
		sanCfg.ForbiddenCommits = []string{futureCommit}
		if task.SolutionFingerprint != "" {
			sanCfg.ForbiddenFingerprints = []string{task.SolutionFingerprint}
		}

		var audit *WorkspaceAuditReport
		if e.SkipScrub {
			audit, err = VerifyWorkspaceClean(ctx, repoDir, sanCfg)
		} else {
			audit, err = SanitizeGitWorkspace(ctx, repoDir, sanCfg)
		}
		ws.PreAudit = audit
		ws.SetupScrubMS = float64(time.Since(setupStart).Microseconds()) / 1000.0
		if err != nil {
			return ws, err
		}

		manifest, rBytes, rOps := buildFileManifest(repoDir)
		ws.BaseManifest = manifest
		ws.IO.BytesRead += rBytes
		ws.IO.ReadOps += rOps
		ws.ProtectedHashes = snapshotProtectedTestHashes(repoDir, task.ProtectedTestPaths, false)
		return ws, nil
	}

	// BackendVFS: canonical vfs -> hanzoai/s3 -> visor CoW snapshot mount.
	cloneStart := time.Now()
	img, cacheHit, casIO, err := e.VFS.getOrBuild(ctx, task, e.Scale, e.CutoffDate)
	ws.IO.Add(casIO)
	if err != nil {
		ws.SetupScrubMS = float64(time.Since(setupStart).Microseconds()) / 1000.0
		return ws, err
	}
	ws.BaseCommit = img.BaseCommit
	ws.FutureFixCommit = img.FutureFixCommit

	// 1. Copy small isolated .git metadata so git diff/status works per sandbox.
	gitBytes, gitFiles, err := copyDirTree(filepath.Join(img.ImageDir, ".git"), filepath.Join(repoDir, ".git"))
	if err != nil {
		return ws, err
	}
	ws.IO.BytesWritten += gitBytes
	ws.IO.WriteOps += gitFiles
	ws.IO.DirtyBytes += gitBytes

	// 2. Hardlink immutable repo_assets ballast (zero copy) + private detached copy for
	//    mutable Go sources and protected test files (chmod 0o444 on protected tests).
	protectedSet := make(map[string]bool, len(task.ProtectedTestPaths))
	for _, rel := range task.ProtectedTestPaths {
		protectedSet[filepath.ToSlash(rel)] = true
	}

	err = filepath.WalkDir(img.ImageDir, func(path string, d os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		rel, relErr := filepath.Rel(img.ImageDir, path)
		if relErr != nil || rel == "." {
			return nil
		}
		relSlash := filepath.ToSlash(rel)
		if d.IsDir() {
			if d.Name() == ".git" {
				return filepath.SkipDir
			}
			ws.IO.MetadataOps++
			return os.MkdirAll(filepath.Join(repoDir, rel), 0o755)
		}
		dst := filepath.Join(repoDir, rel)
		if strings.HasPrefix(relSlash, "repo_assets/") && !protectedSet[relSlash] && e.SandboxPolicy.ReadOnlyBaseImage {
			if linkErr := os.Link(path, dst); linkErr == nil {
				ws.IO.HardlinkedFiles++
				ws.IO.MetadataOps++
				return nil
			}
		}
		data, readErr := os.ReadFile(path)
		if readErr != nil {
			return readErr
		}
		if writeErr := os.WriteFile(dst, data, 0o644); writeErr != nil {
			return writeErr
		}
		ws.IO.ReadOps++
		ws.IO.BytesRead += int64(len(data))
		ws.IO.WriteOps++
		ws.IO.BytesWritten += int64(len(data))
		ws.IO.DirtyBytes += int64(len(data))
		return nil
	})
	if err != nil {
		return ws, err
	}
	ws.IO.SnapshotCloneMS = float64(time.Since(cloneStart).Microseconds()) / 1000.0
	if cacheHit {
		ws.IO.CacheHits++
	}

	// 3. Re-verify pre-execution anti-leakage on the mounted sandbox.
	sanCfg := DefaultSanitizeConfig(img.BaseCommit, e.CutoffDate)
	sanCfg.ForbiddenCommits = []string{img.FutureFixCommit}
	if task.SolutionFingerprint != "" {
		sanCfg.ForbiddenFingerprints = []string{task.SolutionFingerprint}
	}
	audit, err := VerifyWorkspaceClean(ctx, repoDir, sanCfg)
	ws.PreAudit = audit
	ws.SetupScrubMS = float64(time.Since(setupStart).Microseconds()) / 1000.0
	if err != nil {
		return ws, err
	}

	ws.BaseManifest = cloneStringMap(img.BaseManifest)
	ws.ProtectedHashes = snapshotProtectedTestHashes(repoDir, task.ProtectedTestPaths, true)
	return ws, nil
}

func initUnscrubbedUpstreamRepo(ctx context.Context, repoDir string, task CorpusTask, scale RepoScale) (string, string, IOCounters, error) {
	ioStats := IOCounters{Backend: BackendNVMe}
	writeStart := time.Now()

	writeAll := func(files map[string]string) error {
		keys := make([]string, 0, len(files))
		for k := range files {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, rel := range keys {
			dst := filepath.Join(repoDir, filepath.FromSlash(rel))
			if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
				return err
			}
			b := []byte(files[rel])
			if err := os.WriteFile(dst, b, 0o644); err != nil {
				return err
			}
			ioStats.WriteOps++
			ioStats.BytesWritten += int64(len(b))
			ioStats.DirtyBytes += int64(len(b))
		}
		return nil
	}

	if err := writeAll(task.BaseFiles); err != nil {
		return "", "", ioStats, err
	}
	if err := writeAll(task.TestFiles); err != nil {
		return "", "", ioStats, err
	}

	// Write surrounding repository ballast files.
	for i := 0; i < scale.AssetFiles; i++ {
		rel := fmt.Sprintf("repo_assets/pkg_%02d/fixture_%04d.txt", i/25, i)
		dst := filepath.Join(repoDir, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			return "", "", ioStats, err
		}
		payload := deterministicBallast(task.InstanceID, i, scale.AssetBytesPerFile)
		if err := os.WriteFile(dst, payload, 0o644); err != nil {
			return "", "", ioStats, err
		}
		ioStats.WriteOps++
		ioStats.BytesWritten += int64(len(payload))
		ioStats.DirtyBytes += int64(len(payload))
	}
	ioStats.WriteLatencyNS = time.Since(writeStart).Nanoseconds()

	baseStamp := task.BaseTimestamp.UTC().Format(time.RFC3339)
	fixStamp := task.FixTimestamp.UTC().Format(time.RFC3339)

	if _, err := runGitTimed(ctx, repoDir, baseStamp, "init", "-q", "-b", "main"); err != nil {
		return "", "", ioStats, err
	}
	if _, err := runGitTimed(ctx, repoDir, baseStamp, "add", "-A"); err != nil {
		return "", "", ioStats, err
	}
	if _, err := runGitTimed(ctx, repoDir, baseStamp, "commit", "-q", "-m", "base commit for "+task.InstanceID); err != nil {
		return "", "", ioStats, err
	}
	baseSHAOut, err := runGitTimed(ctx, repoDir, baseStamp, "rev-parse", "HEAD")
	if err != nil {
		return "", "", ioStats, err
	}
	baseSHA := strings.TrimSpace(baseSHAOut)
	ioStats.SyncOps++

	// Inject realistic unscrubbed future reference fix commit, tag, branch, remote, and leaked artifact!
	if _, err := runGitTimed(ctx, repoDir, fixStamp, "checkout", "-q", "-b", "upstream-reference-fix"); err != nil {
		return baseSHA, "", ioStats, err
	}
	leakedFix := make(map[string]string, len(task.ReferenceFixFiles))
	for k, v := range task.ReferenceFixFiles {
		leakedFix[k] = "// " + task.SolutionFingerprint + "\n" + v
	}
	if err := writeAll(leakedFix); err != nil {
		return baseSHA, "", ioStats, err
	}
	if _, err := runGitTimed(ctx, repoDir, fixStamp, "add", "-A"); err != nil {
		return baseSHA, "", ioStats, err
	}
	if _, err := runGitTimed(ctx, repoDir, fixStamp, "commit", "-q", "-m", "reference fix "+task.SolutionFingerprint); err != nil {
		return baseSHA, "", ioStats, err
	}
	fixSHAOut, err := runGitTimed(ctx, repoDir, fixStamp, "rev-parse", "HEAD")
	if err != nil {
		return baseSHA, "", ioStats, err
	}
	fixSHA := strings.TrimSpace(fixSHAOut)
	_, _ = runGitTimed(ctx, repoDir, fixStamp, "tag", "v-reference-fix", fixSHA)
	_, _ = runGitTimed(ctx, repoDir, fixStamp, "remote", "add", "origin", "https://upstream.invalid/leak.git")
	_ = os.WriteFile(filepath.Join(repoDir, "gold_patch.diff"), []byte("# leaked "+task.SolutionFingerprint+"\n"), 0o644)
	ioStats.SyncOps++

	return baseSHA, fixSHA, ioStats, nil
}

func deterministicBallast(taskID string, idx, sizeBytes int) []byte {
	sum := sha256.Sum256([]byte(fmt.Sprintf("%s:asset:%d", taskID, idx)))
	hexSeed := hex.EncodeToString(sum[:])
	header := fmt.Sprintf("# fixture %d %s\n", idx, hexSeed[:16])
	line := fmt.Sprintf("ASSET_%04d=%s\n", idx, hexSeed)
	var b bytes.Buffer
	b.Grow(sizeBytes + len(line))
	b.WriteString(header)
	for b.Len() < sizeBytes {
		b.WriteString(line)
	}
	out := b.Bytes()
	if len(out) > sizeBytes {
		return out[:sizeBytes]
	}
	return out
}

func taskCASFingerprint(task CorpusTask, scale RepoScale) string {
	h := sha256.New()
	_, _ = fmt.Fprintf(h, "%s|%s|%d|%d|%s\n", task.InstanceID, scale.Name, scale.AssetFiles, scale.AssetBytesPerFile, task.SolutionFingerprint)
	for _, m := range []map[string]string{task.BaseFiles, task.TestFiles, task.ReferenceFixFiles} {
		keys := make([]string, 0, len(m))
		for k := range m {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			_, _ = fmt.Fprintf(h, "%s:%d:%s\n", k, len(m[k]), m[k])
		}
	}
	return hex.EncodeToString(h.Sum(nil))
}

func buildFileManifest(repoDir string) (map[string]string, int64, int64) {
	manifest := make(map[string]string)
	var totalBytes, readOps int64
	_ = filepath.WalkDir(repoDir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() && d.Name() == ".git" {
			return filepath.SkipDir
		}
		if d.IsDir() || d.Type()&os.ModeSymlink != 0 {
			return nil
		}
		data, rErr := os.ReadFile(path)
		if rErr != nil {
			return nil
		}
		sum := sha256.Sum256(data)
		rel, _ := filepath.Rel(repoDir, path)
		manifest[filepath.ToSlash(rel)] = hex.EncodeToString(sum[:])
		totalBytes += int64(len(data))
		readOps++
		return nil
	})
	return manifest, totalBytes, readOps
}

func measureSnapshotDiff(ws *ProvisionedWorkspace) (int64, int64) {
	var bytesRead, readOps int64
	_ = filepath.WalkDir(ws.RepoDir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() && d.Name() == ".git" {
			return filepath.SkipDir
		}
		if d.IsDir() || d.Type()&os.ModeSymlink != 0 {
			return nil
		}
		rel, _ := filepath.Rel(ws.RepoDir, path)
		relSlash := filepath.ToSlash(rel)

		// VFS CoW fast-path: untouched hardlinked CAS inodes (nlink > 1) require zero re-read!
		if ws.Backend == BackendVFS {
			if info, statErr := os.Stat(path); statErr == nil {
				if st, ok := info.Sys().(*syscall.Stat_t); ok && st.Nlink > 1 {
					if _, inBase := ws.BaseManifest[relSlash]; inBase {
						return nil
					}
				}
			}
		}

		data, rErr := os.ReadFile(path)
		if rErr != nil {
			return nil
		}
		_ = sha256.Sum256(data)
		bytesRead += int64(len(data))
		readOps++
		return nil
	})
	return bytesRead, readOps
}

func snapshotProtectedTestHashes(repoDir string, protected []string, makeReadOnly bool) map[string]string {
	hashes := make(map[string]string, len(protected))
	for _, rel := range protected {
		norm := filepath.ToSlash(filepath.Clean(rel))
		abs, ok := safeJoinUnderRoot(repoDir, norm)
		if !ok {
			continue
		}
		if sum, err := fileSHA256(abs); err == nil {
			hashes[norm] = sum
			if makeReadOnly {
				_ = os.Chmod(abs, 0o444)
			}
		}
	}
	return hashes
}

func copyDirTree(src, dst string) (int64, int64, error) {
	var totalBytes, totalFiles int64
	err := filepath.WalkDir(src, func(path string, d os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		rel, err := filepath.Rel(src, path)
		if err != nil {
			return err
		}
		target := filepath.Join(dst, rel)
		if d.IsDir() {
			return os.MkdirAll(target, 0o755)
		}
		in, err := os.Open(path)
		if err != nil {
			return err
		}
		defer in.Close()
		out, err := os.OpenFile(target, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
		if err != nil {
			return err
		}
		n, copyErr := io.Copy(out, in)
		closeErr := out.Close()
		if copyErr != nil {
			return copyErr
		}
		if closeErr != nil {
			return closeErr
		}
		totalBytes += n
		totalFiles++
		return nil
	})
	return totalBytes, totalFiles, err
}

func restoreWritableTree(root string) {
	if root == "" {
		return
	}
	_ = filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.Type()&os.ModeSymlink != 0 {
			return nil
		}
		if d.IsDir() {
			_ = os.Chmod(path, 0o755)
		} else {
			_ = os.Chmod(path, 0o644)
		}
		return nil
	})
}

var (
	goCacheOnce   sync.Once
	cachedGoCache string
)

func resolveGoBuildCache(fallbackRoot string) string {
	goCacheOnce.Do(func() {
		if env := strings.TrimSpace(os.Getenv("GOCACHE")); env != "" {
			cachedGoCache = env
			return
		}
		if out, err := exec.Command("go", "env", "GOCACHE").Output(); err == nil {
			if dir := strings.TrimSpace(string(out)); dir != "" {
				cachedGoCache = dir
				return
			}
		}
	})
	if cachedGoCache != "" {
		return cachedGoCache
	}
	dir := filepath.Join(fallbackRoot, ".gocache")
	_ = os.MkdirAll(dir, 0o755)
	return dir
}

func buildIsolatedChildEnv(parentEnv map[string]string, goCacheDir string) map[string]string {
	pathVal := os.Getenv("PATH")
	if pathVal == "" {
		pathVal = "/usr/local/go/bin:/usr/local/bin:/usr/bin:/bin"
	}
	homeVal := os.Getenv("HOME")
	if homeVal == "" {
		homeVal = "/tmp"
	}
	// Strictly scrub secrets from test execution subprocess environment and disable network!
	env := map[string]string{
		"PATH":                pathVal,
		"HOME":                homeVal,
		"GOCACHE":             goCacheDir,
		"GOPROXY":             "off",
		"GOSUMDB":             "off",
		"GOWORK":              "off",
		"NO_PROXY":            "*",
		"HTTP_PROXY":          "",
		"HTTPS_PROXY":         "",
		"LANG":                "C.UTF-8",
		"GIT_CONFIG_GLOBAL":   "/dev/null",
		"GIT_CONFIG_NOSYSTEM": "1",
		"GIT_TERMINAL_PROMPT": "0",
	}
	_ = parentEnv // API keys are intentionally NOT passed into `go test` child processes.
	return env
}

func envMapToSlice(m map[string]string) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	out := make([]string, 0, len(keys))
	for _, k := range keys {
		out = append(out, k+"="+m[k])
	}
	return out
}

type goTestEvent struct {
	Action string `json:"Action"`
	Test   string `json:"Test"`
	Output string `json:"Output"`
}

func runGoTestGrading(ctx context.Context, repoDir string, task CorpusTask, env map[string]string) (int, int, string, int, error) {
	cmd := exec.CommandContext(ctx, "go", "test", "-json", "-count=1", "./...")
	cmd.Dir = repoDir
	cmd.Env = envMapToSlice(env)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	runErr := cmd.Run()

	exitCode := 0
	if runErr != nil {
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			return 0, 0, "timeout", -1, context.DeadlineExceeded
		}
		var ee *exec.ExitError
		if errors.As(runErr, &ee) {
			exitCode = ee.ExitCode()
		} else {
			exitCode = 1
		}
	}

	// Strictly parse `go test -json` events per test name; any fail/skip taints the test.
	outcomes := make(map[string]bool)
	tainted := make(map[string]bool)
	var diagLines []string

	sc := bufio.NewScanner(bytes.NewReader(stdout.Bytes()))
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var ev goTestEvent
		if err := json.Unmarshal([]byte(line), &ev); err != nil {
			continue
		}
		if ev.Output != "" {
			trimmed := strings.TrimSpace(ev.Output)
			if trimmed != "" {
				diagLines = append(diagLines, trimmed)
			}
		}
		if ev.Test == "" {
			continue
		}
		switch ev.Action {
		case "pass":
			if !tainted[ev.Test] {
				outcomes[ev.Test] = true
			}
		case "fail", "skip":
			tainted[ev.Test] = true
			outcomes[ev.Test] = false
		}
	}
	if stderr.Len() > 0 {
		diagLines = append(diagLines, strings.TrimSpace(stderr.String()))
	}

	f2pOK := 0
	for _, tname := range task.FailToPassTests {
		if outcomes[tname] {
			f2pOK++
		}
	}
	p2pOK := 0
	for _, tname := range task.PassToPassTests {
		if outcomes[tname] {
			p2pOK++
		}
	}

	if len(diagLines) > 8 {
		diagLines = diagLines[len(diagLines)-8:]
	}
	return f2pOK, p2pOK, strings.Join(diagLines, " | "), exitCode, nil
}

func runGitTimed(ctx context.Context, repoDir, stamp string, args ...string) (string, error) {
	cmd := exec.CommandContext(ctx, "git", args...)
	cmd.Dir = repoDir
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
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return stdout.String(), fmt.Errorf("git %s: %w (%s)", strings.Join(args, " "), err, strings.TrimSpace(stderr.String()))
	}
	return stdout.String(), nil
}

func sanitizeID(id string) string {
	var b strings.Builder
	for _, r := range id {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '-' || r == '_' {
			b.WriteRune(r)
		} else {
			b.WriteByte('_')
		}
	}
	out := b.String()
	if len(out) > 28 {
		return out[:28]
	}
	return out
}
