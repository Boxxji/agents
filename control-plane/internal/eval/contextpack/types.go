// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

// Package contextpack implements a clean-room, deterministic Go evaluation
// harness and context-packing engine for HIP-1346 (hanzoai/agents#83),
// benchmarking evidence-ordering robustness ("Lost in the Middle", Liu et al.,
// TACL 2024) across multi-document agent tool outputs.
//
// Canonical Hanzo stack boundaries are preserved:
//   - Storage & execution: vfs -> hanzoai/s3 -> visor (tool output provenance)
//   - Cloud: lifecycle & billing ownership
//   - Enso: model routing & token-cost accounting
//   - Zen: generative synthesis (pluggable via ModelEvaluator interface)
//   - Kai: structured non-generative packing decisions (GenerativeOutputTokens == 0)
package contextpack

import (
	"context"
	"strings"
)

// ToolSourceKind enumerates grounded multi-document agent tool output categories.
type ToolSourceKind string

const (
	SourceCodeSearch ToolSourceKind = "code_search"
	SourceVFSRead    ToolSourceKind = "vfs_read"
	SourceBuildLog   ToolSourceKind = "build_log"
	SourceJSONAPI    ToolSourceKind = "json_api"
	SourceSQLGraph   ToolSourceKind = "sql_graph"
)

// AllSourceKinds lists all supported grounded tool output types in canonical order.
var AllSourceKinds = []ToolSourceKind{
	SourceCodeSearch,
	SourceVFSRead,
	SourceBuildLog,
	SourceJSONAPI,
	SourceSQLGraph,
}

// PolicyID identifies one of the 5 matched context-packing policies.
type PolicyID string

const (
	PolicyRawConcatBaseline  PolicyID = "raw_concat_baseline"
	PolicyRelevanceDesc      PolicyID = "relevance_desc"
	PolicyUShapedEdgePacking PolicyID = "u_shaped_edge_packing"
	PolicyBudgetedTopKPrune  PolicyID = "budgeted_top_k_prune"
	PolicyKaiEvidenceEdge    PolicyID = "kai_evidence_edge_pack"
)

// AllPolicies returns the 5 matched context-packing policies in canonical reporting order.
var AllPolicies = []PolicyID{
	PolicyRawConcatBaseline,
	PolicyRelevanceDesc,
	PolicyUShapedEdgePacking,
	PolicyBudgetedTopKPrune,
	PolicyKaiEvidenceEdge,
}

// SlotLabel identifies the canonical position bucket for gold evidence in a raw tool sequence.
type SlotLabel string

const (
	SlotStart  SlotLabel = "start"
	SlotQ1     SlotLabel = "q1"
	SlotMiddle SlotLabel = "middle"
	SlotQ3     SlotLabel = "q3"
	SlotEnd    SlotLabel = "end"
	SlotMulti  SlotLabel = "multi_dispersed"
)

// AllSingleSlots returns the 5 canonical position buckets swept across K ∈ {5, 10, 20, 30}.
var AllSingleSlots = []SlotLabel{
	SlotStart,
	SlotQ1,
	SlotMiddle,
	SlotQ3,
	SlotEnd,
}

// ACLScope defines multi-tenant authorization boundaries enforced strictly prior to packing.
type ACLScope struct {
	TenantID     string   `json:"tenant_id"`
	WorkspaceID  string   `json:"workspace_id"`
	AgentID      string   `json:"agent_id"`
	AllowedRoles []string `json:"allowed_roles"`
}

// CanAccess returns true iff the caller scope is authorized to view a document
// belonging to docTenant / docWorkspace and requiring docRole.
func (s ACLScope) CanAccess(docTenant, docWorkspace, docRole string) bool {
	if s.TenantID == "" || docTenant == "" || s.TenantID != docTenant {
		return false
	}
	if docWorkspace != "" && s.WorkspaceID != docWorkspace {
		return false
	}
	if docRole == "" {
		return true
	}
	for _, r := range s.AllowedRoles {
		if r == docRole || r == "admin" {
			return true
		}
	}
	return false
}

// ToolDocument represents a single retrieved tool output chunk entering the context packer.
type ToolDocument struct {
	ID               string            `json:"id"`
	SourceKind       ToolSourceKind    `json:"source_kind"`
	SourceURI        string            `json:"source_uri"`
	Title            string            `json:"title"`
	Content          string            `json:"content"`
	Tokens           int               `json:"tokens"`
	BaseRelevance    float64           `json:"base_relevance"`
	OrderGroupID     string            `json:"order_group_id,omitempty"`
	SequenceIndex    int               `json:"sequence_index,omitempty"`
	RequiresOrder    bool              `json:"requires_order,omitempty"`
	IsGold           bool              `json:"is_gold"`
	EvidenceFactID   string            `json:"evidence_fact_id,omitempty"`
	TenantID         string            `json:"tenant_id"`
	WorkspaceID      string            `json:"workspace_id"`
	RequiredRole     string            `json:"required_role,omitempty"`
	InjectionPayload bool              `json:"injection_payload,omitempty"`
	Metadata         map[string]string `json:"metadata,omitempty"`
}

// EvalCase represents a single multi-document evidence-ordering or negative-regime benchmark task.
type EvalCase struct {
	ID                         string         `json:"id"`
	Suite                      string         `json:"suite"`
	Category                   string         `json:"category"`
	Description                string         `json:"description"`
	Query                      string         `json:"query"`
	ExpectedAnswers            []string       `json:"expected_answers"`
	RequiredFactIDs            []string       `json:"required_fact_ids"`
	RequiredEvidenceCount      int            `json:"required_evidence_count"`
	ContextLength              int            `json:"context_length"`
	GoldPositions              []int          `json:"gold_positions"`
	GoldSlotLabel              SlotLabel      `json:"gold_slot_label"`
	PreserveChronologicalOrder bool           `json:"preserve_chronological_order"`
	Scope                      ACLScope       `json:"scope"`
	Documents                  []ToolDocument `json:"documents"`
	RerankerNoiseLevel         string         `json:"reranker_noise_level,omitempty"`
	HasPromptInjection         bool           `json:"has_prompt_injection,omitempty"`
	DifficultyOffset           float64        `json:"difficulty_offset,omitempty"`
}

// PackConfig controls context-packing budgets, thresholds, safety guards, and Enso pricing.
type PackConfig struct {
	// TokenBudget limits total input tokens for budgeted policies (budgeted_top_k_prune, kai_evidence_edge_pack).
	TokenBudget int `json:"token_budget"`
	// EdgeWindowFraction defines the start/end edge window (e.g., 0.25 => first 25% or last 25% of packed slots).
	EdgeWindowFraction float64 `json:"edge_window_fraction"`
	// DedupeJaccardThreshold is the token Jaccard threshold above which near-duplicate chunks are collapsed by Kai.
	DedupeJaccardThreshold float64 `json:"dedupe_jaccard_threshold"`
	// DiversityMinPerSource encourages Kai to retain top evidence across distinct tool sources when available.
	DiversityMinPerSource int `json:"diversity_min_per_source"`
	// EnforceChronologicalOrder enables Kai's chronological-group lock when a case/document requires causal order.
	EnforceChronologicalOrder bool `json:"enforce_chronological_order"`
	// SanitizeInjections enables neutralization of prompt-injection directives embedded in tool outputs.
	SanitizeInjections bool `json:"sanitize_injections"`
	// EnsoInputUSDPer1MTokens is the Enso-routed Zen input token price in USD per 1M tokens.
	EnsoInputUSDPer1MTokens float64 `json:"enso_input_usd_per_1m_tokens"`
	// KaiDecisionUSDPer1kCalls is the amortized local non-generative Kai scoring/packing compute cost per 1k calls.
	KaiDecisionUSDPer1kCalls float64 `json:"kai_decision_usd_per_1k_calls"`
}

// DefaultPackConfig returns the canonical HIP-1346 benchmark configuration.
func DefaultPackConfig() PackConfig {
	return PackConfig{
		TokenBudget:               1400,
		EdgeWindowFraction:        0.25,
		DedupeJaccardThreshold:    0.72,
		DiversityMinPerSource:     1,
		EnforceChronologicalOrder: true,
		SanitizeInjections:        true,
		EnsoInputUSDPer1MTokens:   3.00, // Enso frontier synthesis input pricing ($3.00 / 1M tokens)
		KaiDecisionUSDPer1kCalls:  0.02, // Structured non-generative Kai scoring overhead ($0.02 / 1k calls)
	}
}

// KaiDecisionRecord captures structured, non-generative packing provenance emitted by Kai.
type KaiDecisionRecord struct {
	PolicyID                 PolicyID `json:"policy_id"`
	SelectedDocIDs           []string `json:"selected_doc_ids"`
	DroppedDocIDs            []string `json:"dropped_doc_ids"`
	FilteredUnauthorizedIDs  []string `json:"filtered_unauthorized_ids"`
	SanitizedInjectionIDs    []string `json:"sanitized_injection_ids"`
	DeduplicatedIDs          []string `json:"deduplicated_ids"`
	ChronologicalLockApplied bool     `json:"chronological_lock_applied"`
	// GenerativeOutputTokens MUST always be 0 to preserve the Kai vs Zen architectural boundary.
	GenerativeOutputTokens int   `json:"generative_output_tokens"`
	DecisionLatencyMicros  int64 `json:"decision_latency_micros"`
}

// PackedContext is the immutable output of a context-packing policy ready for Zen synthesis or evaluation.
type PackedContext struct {
	PolicyID                PolicyID          `json:"policy_id"`
	OrderedDocs             []ToolDocument    `json:"ordered_docs"`
	RenderedText            string            `json:"rendered_text"`
	TotalInputTokens        int               `json:"total_input_tokens"`
	RawAuthorizedTokens     int               `json:"raw_authorized_tokens"`
	TokenSavingsRatio       float64           `json:"token_savings_ratio"`
	GoldSlotIndices         []int             `json:"gold_slot_indices"`
	GoldPositionsNormalized []float64         `json:"gold_positions_normalized"`
	GoldInEdgeWindow        bool              `json:"gold_in_edge_window"`
	AllGoldRetained         bool              `json:"all_gold_retained"`
	GoldRecall              float64           `json:"gold_recall"`
	CrossTenantLeaks        int               `json:"cross_tenant_leaks"`
	InjectionRankHijacks    int               `json:"injection_rank_hijacks"`
	OrderViolations         int               `json:"order_violations"`
	EstimatedInputCostUSD   float64           `json:"estimated_input_cost_usd"`
	KaiDecision             KaiDecisionRecord `json:"kai_decision"`
}

// EvalScore holds the scored output of a ModelEvaluator on a single (EvalCase, PackedContext) pair.
type EvalScore struct {
	ExactCorrect        bool    `json:"exact_correct"`
	CalibratedAccuracy  float64 `json:"calibrated_accuracy"`
	AttentionRecallProb float64 `json:"attention_recall_prob"`
	ExtractiveGrounded  bool    `json:"extractive_grounded"`
	CitationGrounded    bool    `json:"citation_grounded"`
	FailureMode         string  `json:"failure_mode,omitempty"`
}

// ModelEvaluator decouples the packing harness from the scoring backend.
// Offline CI runs use CalibratedLostInMiddleEvaluator (explicitly marked IsSimulated() == true);
// live Zen/Enso runs implement this exact interface without modifying packing or reporting code.
type ModelEvaluator interface {
	Name() string
	IsSimulated() bool
	Evaluate(ctx context.Context, evalCase EvalCase, packed PackedContext) (EvalScore, error)
}

// CaseOutcome records all structural, accuracy, cost, and safety metrics for one evaluated case.
type CaseOutcome struct {
	CaseID                    string    `json:"case_id"`
	Suite                     string    `json:"suite"`
	Category                  string    `json:"category"`
	ContextLength             int       `json:"context_length"`
	GoldSlotLabel             SlotLabel `json:"gold_slot_label"`
	PolicyID                  PolicyID  `json:"policy_id"`
	ExactCorrect              bool      `json:"exact_correct"`
	CalibratedAccuracy        float64   `json:"calibrated_accuracy"`
	AttentionRecallProb       float64   `json:"attention_recall_prob"`
	ExtractiveGrounded        bool      `json:"extractive_grounded"`
	CitationGrounded          bool      `json:"citation_grounded"`
	AllGoldRetained           bool      `json:"all_gold_retained"`
	GoldRecall                float64   `json:"gold_recall"`
	GoldInEdgeWindow          bool      `json:"gold_in_edge_window"`
	InputTokens               int       `json:"input_tokens"`
	RawAuthorizedTokens       int       `json:"raw_authorized_tokens"`
	TokenSavingsRatio         float64   `json:"token_savings_ratio"`
	EstimatedInputCostUSD     float64   `json:"estimated_input_cost_usd"`
	CrossTenantLeaks          int       `json:"cross_tenant_leaks"`
	InjectionRankHijacks      int       `json:"injection_rank_hijacks"`
	OrderViolations           int       `json:"order_violations"`
	KaiGenerativeOutputTokens int       `json:"kai_generative_output_tokens"`
	DecisionLatencyMicros     int64     `json:"decision_latency_micros"`
	FailureMode               string    `json:"failure_mode,omitempty"`
}

// PolicySummary aggregates metrics for one policy across a slice of CaseOutcomes.
type PolicySummary struct {
	PolicyID                  PolicyID `json:"policy_id"`
	CaseCount                 int      `json:"case_count"`
	ExactPassRate             float64  `json:"exact_pass_rate"`
	MeanCalibratedAccuracy    float64  `json:"mean_calibrated_accuracy"`
	GoldRecallRate            float64  `json:"gold_recall_rate"`
	AllGoldRetainedRate       float64  `json:"all_gold_retained_rate"`
	GoldInEdgeWindowRate      float64  `json:"gold_in_edge_window_rate"`
	CitationGroundedRate      float64  `json:"citation_grounded_rate"`
	AvgInputTokens            float64  `json:"avg_input_tokens"`
	AvgRawTokens              float64  `json:"avg_raw_tokens"`
	TokenSavingsPct           float64  `json:"token_savings_pct"`
	CostUSDPer1kQueries       float64  `json:"cost_usd_per_1k_queries"`
	AccuracyPer1kInputTokens  float64  `json:"accuracy_per_1k_input_tokens"`
	ExactPassPer1kInputTokens float64  `json:"exact_pass_per_1k_input_tokens"`
	CrossTenantLeaks          int      `json:"cross_tenant_leaks"`
	InjectionRankHijacks      int      `json:"injection_rank_hijacks"`
	OrderViolations           int      `json:"order_violations"`
	MaxKaiGenOutputTokens     int      `json:"max_kai_gen_output_tokens"`
	AvgDecisionLatencyMicros  float64  `json:"avg_decision_latency_micros"`
}

// PositionCurveCell captures accuracy and edge-window placement for one (Policy, K, Slot) cell.
type PositionCurveCell struct {
	PolicyID               PolicyID  `json:"policy_id"`
	ContextLength          int       `json:"context_length"`
	GoldSlotLabel          SlotLabel `json:"gold_slot_label"`
	CaseCount              int       `json:"case_count"`
	ExactPassRate          float64   `json:"exact_pass_rate"`
	MeanCalibratedAccuracy float64   `json:"mean_calibrated_accuracy"`
	GoldInEdgeWindowRate   float64   `json:"gold_in_edge_window_rate"`
	AvgInputTokens         float64   `json:"avg_input_tokens"`
}

// RerankerRegimeRow captures accuracy vs upstream reranker Recall@k under score noise/inversion.
type RerankerRegimeRow struct {
	NoiseLevel             string   `json:"noise_level"`
	RerankerTopKRecall     float64  `json:"reranker_top_k_recall"`
	PolicyID               PolicyID `json:"policy_id"`
	CaseCount              int      `json:"case_count"`
	ExactPassRate          float64  `json:"exact_pass_rate"`
	MeanCalibratedAccuracy float64  `json:"mean_calibrated_accuracy"`
	AllGoldRetainedRate    float64  `json:"all_gold_retained_rate"`
	AvgInputTokens         float64  `json:"avg_input_tokens"`
}

// ChronologicalRegimeRow captures causal-order integrity and accuracy on sequentially coupled logs/diffs.
type ChronologicalRegimeRow struct {
	PolicyID               PolicyID `json:"policy_id"`
	GuardEnabled           bool     `json:"guard_enabled"`
	CaseCount              int      `json:"case_count"`
	ExactPassRate          float64  `json:"exact_pass_rate"`
	MeanCalibratedAccuracy float64  `json:"mean_calibrated_accuracy"`
	OrderViolationRate     float64  `json:"order_violation_rate"`
	TotalOrderViolations   int      `json:"total_order_violations"`
	AvgInputTokens         float64  `json:"avg_input_tokens"`
}

// TightBudgetSweepRow captures M=3 multi-hop synthesis performance as token budget tightens below M docs.
type TightBudgetSweepRow struct {
	TokenBudget            int      `json:"token_budget"`
	PolicyID               PolicyID `json:"policy_id"`
	CaseCount              int      `json:"case_count"`
	ExactPassRate          float64  `json:"exact_pass_rate"`
	MeanCalibratedAccuracy float64  `json:"mean_calibrated_accuracy"`
	AllGoldRetainedRate    float64  `json:"all_gold_retained_rate"`
	GoldRecallRate         float64  `json:"gold_recall_rate"`
	AvgInputTokens         float64  `json:"avg_input_tokens"`
	CostUSDPer1kQueries    float64  `json:"cost_usd_per_1k_queries"`
}

// BenchmarkMetadata documents reproducibility and evaluator transparency.
type BenchmarkMetadata struct {
	BenchmarkName       string     `json:"benchmark_name"`
	HIP                 string     `json:"hip"`
	Issue               string     `json:"issue"`
	Seed                int64      `json:"seed"`
	GoVersion           string     `json:"go_version"`
	GOOS                string     `json:"goos"`
	GOARCH              string     `json:"goarch"`
	NumCPU              int        `json:"num_cpu"`
	EvaluatorName       string     `json:"evaluator_name"`
	EvaluatorSimulated  bool       `json:"evaluator_simulated"`
	SimulationNote      string     `json:"simulation_note"`
	PackConfig          PackConfig `json:"pack_config"`
	PrimaryCaseCount    int        `json:"primary_case_count"`
	NegativeRegimeCases int        `json:"negative_regime_cases"`
}

// BenchmarkReport is the full machine-readable output emitted by contextpack-bench.
type BenchmarkReport struct {
	Metadata                 BenchmarkMetadata        `json:"metadata"`
	PrimaryOverall           []PolicySummary          `json:"primary_overall"`
	PositionSweepOverall     []PolicySummary          `json:"position_sweep_overall"`
	PositionCurve            []PositionCurveCell      `json:"position_curve"`
	MultiEvidenceSummary     []PolicySummary          `json:"multi_evidence_summary"`
	SecurityAuditSummary     []PolicySummary          `json:"security_audit_summary"`
	NegativeNoisyReranker    []RerankerRegimeRow      `json:"negative_noisy_reranker"`
	NegativeChronologicalLog []ChronologicalRegimeRow `json:"negative_chronological_log"`
	NegativeTightBudget      []TightBudgetSweepRow    `json:"negative_tight_budget"`
	PrimaryOutcomes          []CaseOutcome            `json:"primary_outcomes"`
}

// EstimateTokens provides a fast, deterministic BPE-like token count approximation
// using Go stdlib only (no external tokenizer dependencies).
func EstimateTokens(text string) int {
	if strings.TrimSpace(text) == "" {
		return 0
	}
	fields := strings.Fields(text)
	total := 0
	for _, f := range fields {
		// Longer code identifiers / paths / JSON tokens split into multiple BPE pieces (~4 chars/subtoken).
		if len(f) <= 4 {
			total++
		} else {
			total += (len(f) + 3) / 4
		}
	}
	if total < 1 {
		return 1
	}
	return total
}
