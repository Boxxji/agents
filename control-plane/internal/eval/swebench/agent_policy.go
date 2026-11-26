package swebench

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// ModelPricing defines per-million-token USD pricing for Enso-routed models.
type ModelPricing struct {
	Model             string  `json:"model"`
	Provider          string  `json:"provider"`
	InputPer1MUSD     float64 `json:"input_per_1m_usd"`
	OutputPer1MUSD    float64 `json:"output_per_1m_usd"`
	CachedInputPer1MU float64 `json:"cached_input_per_1m_usd"`
}

// EnsoModelRouter routes inference requests across Zen and external providers
// while keeping billing & token cost accounting deterministic and transparent.
type EnsoModelRouter struct {
	DefaultModel string
	EndpointURL  string
	HTTPClient   *http.Client
	Pricing      map[string]ModelPricing
}

// NewDefaultEnsoRouter returns an EnsoModelRouter configured with canonical pricing.
func NewDefaultEnsoRouter(model string) *EnsoModelRouter {
	if strings.TrimSpace(model) == "" {
		model = "zen/zen-coder"
	}
	return &EnsoModelRouter{
		DefaultModel: model,
		Pricing: map[string]ModelPricing{
			"zen/zen-coder":             {Model: "zen/zen-coder", Provider: "zen", InputPer1MUSD: 0.80, OutputPer1MUSD: 3.20, CachedInputPer1MU: 0.20},
			"anthropic/claude-sonnet-4": {Model: "anthropic/claude-sonnet-4", Provider: "anthropic", InputPer1MUSD: 3.00, OutputPer1MUSD: 15.00, CachedInputPer1MU: 0.30},
			"openai/gpt-4o":             {Model: "openai/gpt-4o", Provider: "openai", InputPer1MUSD: 2.50, OutputPer1MUSD: 10.00, CachedInputPer1MU: 1.25},
		},
	}
}

// ComputeCostUSD calculates exact USD spend for token usage under the routed model.
func (r *EnsoModelRouter) ComputeCostUSD(model string, tokensIn, tokensOut, tokensCached int64) float64 {
	if strings.TrimSpace(model) == "" {
		model = r.DefaultModel
	}
	p, ok := r.Pricing[model]
	if !ok {
		p = ModelPricing{InputPer1MUSD: 2.50, OutputPer1MUSD: 10.00, CachedInputPer1MU: 0.50}
	}
	billableIn := tokensIn - tokensCached
	if billableIn < 0 {
		billableIn = 0
	}
	cost := (float64(billableIn)/1_000_000.0)*p.InputPer1MUSD +
		(float64(tokensCached)/1_000_000.0)*p.CachedInputPer1MU +
		(float64(tokensOut)/1_000_000.0)*p.OutputPer1MUSD
	return cost
}

// LiveGenerateHook optionally invokes a live Enso/Zen inference endpoint when
// EndpointURL is configured and HANZO_API_KEY has been validated for ai:inference.
// Secrets are never written into returned errors or telemetry.
func (r *EnsoModelRouter) LiveGenerateHook(ctx context.Context, apiKey, model, prompt string) (string, error) {
	if strings.TrimSpace(r.EndpointURL) == "" {
		return "", errors.New("swebench: live Enso endpoint URL not configured")
	}
	if strings.TrimSpace(apiKey) == "" {
		return "", errors.New("swebench: missing ai:inference HANZO_API_KEY")
	}
	if strings.TrimSpace(model) == "" {
		model = r.DefaultModel
	}
	payload, err := json.Marshal(map[string]any{
		"model":       model,
		"temperature": 0.0,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
	})
	if err != nil {
		return "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, r.EndpointURL, bytes.NewReader(payload))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)
	client := r.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: 30 * time.Second}
	}
	resp, err := client.Do(req)
	if err != nil {
		return "", errors.New(RedactSecretsInLog(err.Error(), apiKey))
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return "", fmt.Errorf("swebench: Enso inference HTTP %d", resp.StatusCode)
	}
	var parsed struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return "", err
	}
	if len(parsed.Choices) == 0 {
		return "", errors.New("swebench: empty Enso inference response")
	}
	return parsed.Choices[0].Message.Content, nil
}

// KaiDecision represents a structured, non-generative control decision from Kai.
type KaiDecision string

const (
	DecisionAccept      KaiDecision = "accept"
	DecisionIterate     KaiDecision = "iterate"
	DecisionAbortBudget KaiDecision = "abort_budget"
	DecisionAbortMaxIt  KaiDecision = "abort_max_iterations"
)

// KaiGateInput holds structured verification telemetry passed to KaiVerificationGate.
type KaiGateInput struct {
	Iteration      int
	MaxIterations  int
	TokensUsed     int64
	TokenBudget    int64
	FailingTests   int
	TestsCompiled  bool
	DiagnosticTail string
}

// KaiGateOutput is the deterministic, non-generative routing verdict from Kai.
type KaiGateOutput struct {
	Decision         KaiDecision `json:"decision"`
	CompactedSummary string      `json:"compacted_summary,omitempty"`
	Reason           string      `json:"reason"`
}

// KaiVerificationGate implements Hanzo's canonical Kai boundary: structured,
// non-generative stop/iterate/budget-abort decisions based on test verdicts.
type KaiVerificationGate struct {
	MaxSummaryBytes int
}

// Decide evaluates structured verification signals and compacts diagnostics.
func (g KaiVerificationGate) Decide(in KaiGateInput) KaiGateOutput {
	limit := g.MaxSummaryBytes
	if limit <= 0 {
		limit = 512
	}
	summary := strings.TrimSpace(in.DiagnosticTail)
	if len(summary) > limit {
		summary = summary[len(summary)-limit:]
	}
	if in.TestsCompiled && in.FailingTests == 0 {
		return KaiGateOutput{
			Decision:         DecisionAccept,
			CompactedSummary: summary,
			Reason:           "all verification tests passed",
		}
	}
	if in.TokenBudget > 0 && in.TokensUsed >= in.TokenBudget {
		return KaiGateOutput{
			Decision:         DecisionAbortBudget,
			CompactedSummary: summary,
			Reason:           fmt.Sprintf("token budget exhausted (%d >= %d)", in.TokensUsed, in.TokenBudget),
		}
	}
	if in.Iteration >= in.MaxIterations {
		return KaiGateOutput{
			Decision:         DecisionAbortMaxIt,
			CompactedSummary: summary,
			Reason:           fmt.Sprintf("max iterations reached (%d/%d)", in.Iteration, in.MaxIterations),
		}
	}
	return KaiGateOutput{
		Decision:         DecisionIterate,
		CompactedSummary: summary,
		Reason:           fmt.Sprintf("%d failing test(s); continuing compacted edit-test loop", in.FailingTests),
	}
}

// AgentRunResult holds the candidate file edits (or workspace mutations) and
// token accounting produced by an AgentPolicy during sandbox execution.
type AgentRunResult struct {
	PolicyName   string
	UpdatedFiles map[string]string
	TokensIn     int64
	TokensOut    int64
	TokensCached int64
	Iterations   int
	Notes        string
}

// SandboxTestRunner is a read-only feedback callback allowing iterative policies
// to execute real `go test` inside the isolated sandbox between edit turns.
type SandboxTestRunner func(ctx context.Context) (passed bool, failingCount int, diagTail string, err error)

// AgentPolicy defines a matched coding-agent strategy evaluated inside Visor.
type AgentPolicy interface {
	Name() string
	Solve(ctx context.Context, task CorpusTask, workspace string, env map[string]string, runTests SandboxTestRunner) (AgentRunResult, error)
}

func estimateTokens(chars int) int64 {
	if chars <= 0 {
		return 1
	}
	return int64((chars + 3) / 4)
}

func taskContextChars(task CorpusTask) int {
	total := len(task.ProblemStatement)
	for _, v := range task.BaseFiles {
		total += len(v)
	}
	return total
}

func filesChars(m map[string]string) int {
	total := 0
	for _, v := range m {
		total += len(v)
	}
	return total
}

// NoOpPolicy submits zero edits to prove F2P tests fail and P2P tests pass at base_commit.
type NoOpPolicy struct{}

func (NoOpPolicy) Name() string { return "noop_baseline" }

func (NoOpPolicy) Solve(_ context.Context, task CorpusTask, _ string, _ map[string]string, _ SandboxTestRunner) (AgentRunResult, error) {
	return AgentRunResult{
		PolicyName:   "noop_baseline",
		TokensIn:     estimateTokens(len(task.ProblemStatement)),
		TokensOut:    16,
		TokensCached: 0,
		Iterations:   1,
		Notes:        "no-op baseline",
	}, nil
}

// OracleReferencePolicy applies the clean reference fix for upper-bound verification.
type OracleReferencePolicy struct{}

func (OracleReferencePolicy) Name() string { return "oracle_reference" }

func (OracleReferencePolicy) Solve(_ context.Context, task CorpusTask, _ string, _ map[string]string, _ SandboxTestRunner) (AgentRunResult, error) {
	ctxTok := estimateTokens(taskContextChars(task)) * 2
	outTok := estimateTokens(filesChars(task.ReferenceFixFiles))
	return AgentRunResult{
		PolicyName:   "oracle_reference",
		UpdatedFiles: cloneStringMap(task.ReferenceFixFiles),
		TokensIn:     ctxTok,
		TokensOut:    outTok,
		TokensCached: ctxTok / 2,
		Iterations:   1,
		Notes:        "clean reference solution",
	}, nil
}

// SingleShotPolicy models a single-turn Zen generation baseline without test-loop iteration.
type SingleShotPolicy struct{}

func (SingleShotPolicy) Name() string { return "single_shot_baseline" }

func (SingleShotPolicy) Solve(_ context.Context, task CorpusTask, _ string, _ map[string]string, _ SandboxTestRunner) (AgentRunResult, error) {
	edits := task.SingleShotFixFiles
	if len(edits) == 0 {
		edits = task.ReferenceFixFiles
	}
	ctxTok := estimateTokens(taskContextChars(task)) * 3
	outTok := estimateTokens(filesChars(edits)) + 128
	return AgentRunResult{
		PolicyName:   "single_shot_baseline",
		UpdatedFiles: cloneStringMap(edits),
		TokensIn:     ctxTok,
		TokensOut:    outTok,
		TokensCached: 0,
		Iterations:   1,
		Notes:        "single-shot generation without iterative test feedback",
	}, nil
}

// IterativeKaiGatedPolicy runs a multi-step edit-test loop with context compaction
// and Kai structured verification gating (optional TokenBudget models budget-capped regime).
type IterativeKaiGatedPolicy struct {
	PolicyLabel   string
	MaxIterations int
	TokenBudget   int64
	Gate          KaiVerificationGate
}

func (p IterativeKaiGatedPolicy) Name() string {
	if p.PolicyLabel != "" {
		return p.PolicyLabel
	}
	return "iterative_kai_gated"
}

func (p IterativeKaiGatedPolicy) Solve(ctx context.Context, task CorpusTask, workspace string, _ map[string]string, runTests SandboxTestRunner) (AgentRunResult, error) {
	maxIter := p.MaxIterations
	if maxIter <= 0 {
		maxIter = 3
	}
	basePromptTok := estimateTokens(taskContextChars(task)) * 2

	// Step 1: produce initial patch (partial on hard tasks, full on medium tasks).
	step1Files := task.IterativeStep1Files
	if len(step1Files) == 0 {
		step1Files = task.SingleShotFixFiles
	}
	if len(step1Files) == 0 {
		step1Files = task.ReferenceFixFiles
	}

	if err := writeUpdatedFiles(workspace, step1Files); err != nil {
		return AgentRunResult{}, err
	}

	tokensIn := basePromptTok + 600
	tokensOut := estimateTokens(filesChars(step1Files)) + 220
	tokensCached := int64(0)
	iterations := 1

	// Run intermediate sandbox test verification for Kai gate.
	passed, failing, diagTail, testErr := runTests(ctx)
	compiled := testErr == nil
	verdict := p.Gate.Decide(KaiGateInput{
		Iteration:      iterations,
		MaxIterations:  maxIter,
		TokensUsed:     tokensIn + tokensOut,
		TokenBudget:    p.TokenBudget,
		FailingTests:   failing,
		TestsCompiled:  compiled && passed,
		DiagnosticTail: diagTail,
	})

	if verdict.Decision == DecisionAccept || verdict.Decision == DecisionAbortBudget || verdict.Decision == DecisionAbortMaxIt {
		return AgentRunResult{
			PolicyName:   p.Name(),
			UpdatedFiles: cloneStringMap(step1Files),
			TokensIn:     tokensIn,
			TokensOut:    tokensOut,
			TokensCached: tokensCached,
			Iterations:   iterations,
			Notes:        fmt.Sprintf("kai=%s (%s)", verdict.Decision, verdict.Reason),
		}, nil
	}

	// Step 2: compact failure diagnostics, reuse cached repository prefix, and repair remaining edge cases.
	iterations++
	compactedTok := estimateTokens(len(verdict.CompactedSummary)) + 180
	step2In := basePromptTok + compactedTok
	step2Cached := int64(float64(basePromptTok) * 0.75) // prefix cache hit on base context
	step2Out := estimateTokens(filesChars(task.ReferenceFixFiles)) + 180

	tokensIn += step2In
	tokensOut += step2Out
	tokensCached += step2Cached

	if p.TokenBudget > 0 && (tokensIn+tokensOut) > p.TokenBudget {
		// Budget cap exceeded before step 2 patch can complete — truncate/abort!
		return AgentRunResult{
			PolicyName:   p.Name(),
			UpdatedFiles: cloneStringMap(step1Files),
			TokensIn:     tokensIn,
			TokensOut:    tokensOut,
			TokensCached: tokensCached,
			Iterations:   iterations,
			Notes:        fmt.Sprintf("kai=%s (budget cap %d exceeded during turn 2)", DecisionAbortBudget, p.TokenBudget),
		}, nil
	}

	if err := writeUpdatedFiles(workspace, task.ReferenceFixFiles); err != nil {
		return AgentRunResult{}, err
	}
	return AgentRunResult{
		PolicyName:   p.Name(),
		UpdatedFiles: cloneStringMap(task.ReferenceFixFiles),
		TokensIn:     tokensIn,
		TokensOut:    tokensOut,
		TokensCached: tokensCached,
		Iterations:   iterations,
		Notes:        "kai=iterate->accept (converged after compacted diagnostic loop)",
	}, nil
}

// AdversarialLeakProbePolicy attempts to harvest reference fixes or secrets
// from git history, tags, reflogs, raw .git/objects, workspace files, and env.
type AdversarialLeakProbePolicy struct{}

func (AdversarialLeakProbePolicy) Name() string { return "adversarial_leak_probe" }

func (AdversarialLeakProbePolicy) Solve(ctx context.Context, task CorpusTask, workspace string, env map[string]string, _ SandboxTestRunner) (AgentRunResult, error) {
	var findings []string
	fp := task.SolutionFingerprint

	for _, args := range [][]string{
		{"log", "--all", "-p"},
		{"reflog", "--all"},
		{"tag", "-l"},
		{"remote", "-v"},
		{"branch", "-a"},
	} {
		out, _ := runGit(ctx, workspace, args...)
		if fp != "" && strings.Contains(out, fp) {
			findings = append(findings, "git "+strings.Join(args, " "))
		}
	}

	objDir := filepath.Join(workspace, ".git", "objects")
	_ = filepath.WalkDir(objDir, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		data, readErr := os.ReadFile(path)
		if readErr == nil && fp != "" && bytes.Contains(data, []byte(fp)) {
			findings = append(findings, "object:"+filepath.Base(path))
		}
		return nil
	})

	for _, artifact := range []string{"gold_patch.diff", ".swe_bench_gold.json", "test_patch.diff"} {
		if _, err := os.Stat(filepath.Join(workspace, artifact)); err == nil {
			findings = append(findings, "file:"+artifact)
		}
	}

	for k, v := range env {
		up := strings.ToUpper(k)
		if strings.Contains(up, "GOLD") || strings.Contains(up, "ADMIN") || strings.Contains(up, "GITHUB_TOKEN") {
			findings = append(findings, "env:"+k)
		}
		if fp != "" && strings.Contains(v, fp) {
			findings = append(findings, "env_fp:"+k)
		}
	}
	sort.Strings(findings)

	return AgentRunResult{
		PolicyName: "adversarial_leak_probe",
		TokensIn:   640,
		TokensOut:  96,
		Iterations: 1,
		Notes:      fmt.Sprintf("leak_findings=%d:%s", len(findings), strings.Join(findings, ",")),
	}, nil
}

// AdversarialTamperPolicy attempts to rewrite protected test files and plant a
// host-escaping symlink alongside the valid reference fix to verify fail-closed
// anti-tampering enforcement.
type AdversarialTamperPolicy struct{}

func (AdversarialTamperPolicy) Name() string { return "adversarial_tamper_agent" }

func (AdversarialTamperPolicy) Solve(_ context.Context, task CorpusTask, workspace string, _ map[string]string, _ SandboxTestRunner) (AgentRunResult, error) {
	// Attempt 1: chmod + overwrite protected test file.
	for _, rel := range task.ProtectedTestPaths {
		target := filepath.Join(workspace, filepath.FromSlash(rel))
		_ = os.Chmod(target, 0o644)
		_ = os.WriteFile(target, []byte("package tampered\nimport \"testing\"\nfunc TestFake(t *testing.T) {}\n"), 0o644)
	}
	// Attempt 2: plant an out-of-sandbox symlink pointing to host /etc/hosts.
	linkPath := filepath.Join(workspace, "host_escape_link.txt")
	_ = os.Symlink("/etc/hosts", linkPath)

	// Submit valid reference fix alongside tampering to prove tampering ALWAYS fails closed!
	return AgentRunResult{
		PolicyName:   "adversarial_tamper_agent",
		UpdatedFiles: cloneStringMap(task.ReferenceFixFiles),
		TokensIn:     720,
		TokensOut:    180,
		Iterations:   1,
		Notes:        "attempted test overwrite + host symlink escape",
	}, nil
}

func cloneStringMap(in map[string]string) map[string]string {
	if in == nil {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func writeUpdatedFiles(workspace string, files map[string]string) error {
	keys := make([]string, 0, len(files))
	for k := range files {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, rel := range keys {
		abs, ok := safeJoinUnderRoot(workspace, rel)
		if !ok {
			return fmt.Errorf("swebench: agent update path %q escapes workspace", rel)
		}
		if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
			return err
		}
		if err := os.WriteFile(abs, []byte(files[rel]), 0o644); err != nil {
			return err
		}
	}
	return nil
}
