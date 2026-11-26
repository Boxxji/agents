// Package longmemeval implements a clean-room Apache-2.0 LongMemEval (ICLR 2025)
// temporal memory evaluation harness for HIP-1346 (hanzoai/agents#82).
//
// It evaluates existing Hanzo Agents memory storage and policy semantics
// (#62-#66, control-plane/internal/handlers/memory.go, vector_memory.go, and
// Kai non-generative evidence gating) directly over the canonical MemoryStore
// surface without introducing a competing memory engine.
package longmemeval

import "time"

// TaskCategory identifies each of the five LongMemEval evaluation dimensions.
type TaskCategory string

const (
	CategoryKnowledgeUpdate  TaskCategory = "knowledge_update"
	CategoryContradiction    TaskCategory = "contradiction"
	CategoryTemporalOrdering TaskCategory = "temporal_ordering"
	CategoryAbstention       TaskCategory = "abstention"
	CategoryCrossSession     TaskCategory = "cross_session"
)

// AllCategories lists all five LongMemEval categories in canonical order.
var AllCategories = []TaskCategory{
	CategoryKnowledgeUpdate,
	CategoryContradiction,
	CategoryTemporalOrdering,
	CategoryAbstention,
	CategoryCrossSession,
}

// MemoryAction models selective memory lifecycle operations (#62, #66).
type MemoryAction string

const (
	ActionAdd     MemoryAction = "add"
	ActionUpdate  MemoryAction = "update"
	ActionDelete  MemoryAction = "delete"
	ActionForget  MemoryAction = "forget"
	ActionReflect MemoryAction = "reflect"
	ActionNoop    MemoryAction = "noop"
)

// EpistemicKind separates direct observations from inferred hypotheses (#64).
type EpistemicKind string

const (
	KindObservation EpistemicKind = "observation"
	KindConclusion  EpistemicKind = "conclusion"
	KindHypothesis  EpistemicKind = "hypothesis"
)

// TemporalMode defines the temporal reasoning operation requested by a query.
type TemporalMode string

const (
	TemporalLatest    TemporalMode = "latest"
	TemporalSequence  TemporalMode = "sequence"
	TemporalFirst     TemporalMode = "first"
	TemporalLast      TemporalMode = "last"
	TemporalDuration  TemporalMode = "duration"
	TemporalWindow    TemporalMode = "window"
	TemporalAsOf      TemporalMode = "as_of"
	TemporalAggregate TemporalMode = "aggregate"
)

// QueryMode controls whether structured entity/attribute hints are available
// or the policy must rely purely on semantic/lexical evidence matching.
type QueryMode string

const (
	QueryStructured QueryMode = "structured"
	QuerySemantic   QueryMode = "semantic"
)

// KaiDecisionStatus captures the non-generative structured decision emitted by
// a memory retrieval/gating policy under Hanzo's Kai boundary.
type KaiDecisionStatus string

const (
	DecisionAnswer    KaiDecisionStatus = "ANSWER"
	DecisionAbstain   KaiDecisionStatus = "ABSTAIN_INSUFFICIENT_EVIDENCE"
	DecisionContested KaiDecisionStatus = "CONTESTED_CONTRADICTION"
)

// SessionTurn represents a single conversational, tool, or reflection event
// ingested into a canonical MemoryStore by a memory policy adapter.
type SessionTurn struct {
	TurnID       string        `json:"turn_id"`
	TenantID     string        `json:"tenant_id"`
	ActorID      string        `json:"actor_id"`
	SessionID    string        `json:"session_id"`
	WorkflowID   string        `json:"workflow_id,omitempty"`
	Scope        string        `json:"scope,omitempty"` // workflow | session | actor | global
	ACLRoles     []string      `json:"acl_roles,omitempty"`
	Timestamp    time.Time     `json:"timestamp"`
	Entity       string        `json:"entity"`
	Attribute    string        `json:"attribute"`
	Value        string        `json:"value"`
	Statement    string        `json:"statement"`
	Action       MemoryAction  `json:"action"`
	Kind         EpistemicKind `json:"kind,omitempty"`
	Confidence   float64       `json:"confidence,omitempty"`
	SourceID     string        `json:"source_id,omitempty"`
	IsDistractor bool          `json:"is_distractor,omitempty"`
}

// RecallQuery specifies a LongMemEval query executed against a populated MemoryStore.
type RecallQuery struct {
	CaseID             string       `json:"case_id"`
	Category           TaskCategory `json:"category"`
	TenantID           string       `json:"tenant_id"`
	ActorID            string       `json:"actor_id"`
	WorkflowID         string       `json:"workflow_id,omitempty"`
	SessionIDs         []string     `json:"session_ids,omitempty"`
	CallerRoles        []string     `json:"caller_roles,omitempty"`
	Question           string       `json:"question"`
	Entity             string       `json:"entity,omitempty"`
	Attribute          string       `json:"attribute,omitempty"`
	QueryMode          QueryMode    `json:"query_mode,omitempty"`
	TemporalMode       TemporalMode `json:"temporal_mode,omitempty"`
	QueryTime          time.Time    `json:"query_time"`
	AsOf               time.Time    `json:"as_of,omitempty"`
	WindowStart        time.Time    `json:"window_start,omitempty"`
	WindowEnd          time.Time    `json:"window_end,omitempty"`
	DurationStartValue string       `json:"duration_start_value,omitempty"`
	DurationEndValue   string       `json:"duration_end_value,omitempty"`
	TokenBudget        int          `json:"token_budget,omitempty"`
}

// KaiDecision captures the structured, non-generative recall result and audit
// provenance returned by a MemoryPolicy.
type KaiDecision struct {
	Status                 KaiDecisionStatus `json:"status"`
	Answer                 string            `json:"answer,omitempty"`
	OrderedAnswers         []string          `json:"ordered_answers,omitempty"`
	AggregatedValues       []string          `json:"aggregated_values,omitempty"`
	Abstained              bool              `json:"abstained"`
	ContradictionDetected  bool              `json:"contradiction_detected,omitempty"`
	ContradictingValues    []string          `json:"contradicting_values,omitempty"`
	EvidenceKeys           []string          `json:"evidence_keys,omitempty"`
	EvidenceTenants        []string          `json:"evidence_tenants,omitempty"`
	EvidenceScopes         []string          `json:"evidence_scopes,omitempty"`
	Confidence             float64           `json:"confidence"`
	EvidenceScore          float64           `json:"evidence_score"`
	RetrievalTokens        int               `json:"retrieval_tokens"`
	GenerativeOutputTokens int               `json:"generative_output_tokens"`
}

// IngestStats records canonical store mutations per ingested turn.
type IngestStats struct {
	KVWrites     int `json:"kv_writes"`
	VectorWrites int `json:"vector_writes"`
	EventWrites  int `json:"event_writes"`
	SkippedNoops int `json:"skipped_noops"`
}

// TotalWrites returns the sum of KV, vector, and event mutations.
func (s IngestStats) TotalWrites() int {
	return s.KVWrites + s.VectorWrites + s.EventWrites
}

// EvalCase defines a reproducible multi-session LongMemEval scenario.
type EvalCase struct {
	ID                  string        `json:"id"`
	Category            TaskCategory  `json:"category"`
	Subcategory         string        `json:"subcategory,omitempty"`
	TenantID            string        `json:"tenant_id"`
	ActorID             string        `json:"actor_id"`
	SessionCount        int           `json:"session_count"`
	Turns               []SessionTurn `json:"turns"`
	Query               RecallQuery   `json:"query"`
	ExpectedAnswer      string        `json:"expected_answer,omitempty"`
	ExpectedOrder       []string      `json:"expected_order,omitempty"`
	ExpectedSet         []string      `json:"expected_set,omitempty"`
	ShouldAbstain       bool          `json:"should_abstain,omitempty"`
	ExpectContradiction bool          `json:"expect_contradiction,omitempty"`
	ForbiddenStale      []string      `json:"forbidden_stale,omitempty"`
	IsParaphrase        bool          `json:"is_paraphrase,omitempty"`
}

// CategoryMetrics aggregates accuracy and error modes for a single TaskCategory.
type CategoryMetrics struct {
	Total                  int     `json:"total"`
	Correct                int     `json:"correct"`
	Accuracy               float64 `json:"accuracy"`
	StaleErrors            int     `json:"stale_errors"`
	FalseMemories          int     `json:"false_memories"`
	OverAbstentions        int     `json:"over_abstentions"`
	ContradictionsExpected int     `json:"contradictions_expected,omitempty"`
	ContradictionsDetected int     `json:"contradictions_detected,omitempty"`
}

// CaseOutcome records per-case grading diagnostics for reproducible auditing.
type CaseOutcome struct {
	CaseID                string       `json:"case_id"`
	Category              TaskCategory `json:"category"`
	Subcategory           string       `json:"subcategory,omitempty"`
	Passed                bool         `json:"passed"`
	Abstained             bool         `json:"abstained"`
	ShouldAbstain         bool         `json:"should_abstain"`
	StaleError            bool         `json:"stale_error,omitempty"`
	ContradictionDetected bool         `json:"contradiction_detected,omitempty"`
	CrossTenantLeak       bool         `json:"cross_tenant_leak,omitempty"`
	PredictedAnswer       string       `json:"predicted_answer,omitempty"`
	ExpectedAnswer        string       `json:"expected_answer,omitempty"`
	EvidenceScore         float64      `json:"evidence_score"`
	RetrievalTokens       int          `json:"retrieval_tokens"`
	LatencyUS             int64        `json:"latency_us"`
}

// PolicyMetrics summarizes end-to-end LongMemEval results for one policy.
type PolicyMetrics struct {
	PolicyName            string                           `json:"policy_name"`
	Description           string                           `json:"description"`
	TotalCases            int                              `json:"total_cases"`
	CorrectCases          int                              `json:"correct_cases"`
	OverallAccuracy       float64                          `json:"overall_accuracy"`
	ByCategory            map[TaskCategory]CategoryMetrics `json:"by_category"`
	StaleFactErrorRate    float64                          `json:"stale_fact_error_rate"`
	ContradictionDetRate  float64                          `json:"contradiction_detection_rate"`
	AbstentionPrecision   float64                          `json:"abstention_precision"`
	AbstentionRecall      float64                          `json:"abstention_recall"`
	AbstentionF1          float64                          `json:"abstention_f1"`
	UnanswerableFPR       float64                          `json:"unanswerable_fpr"`
	OverAbstentionRate    float64                          `json:"over_abstention_rate"`
	CrossTenantLeaks      int                              `json:"cross_tenant_leaks"`
	GenerativeTokensTotal int                              `json:"generative_tokens_total"`
	TotalWrites           int                              `json:"total_writes"`
	TotalSkippedNoops     int                              `json:"total_skipped_noops"`
	WriteAmplification    float64                          `json:"write_amplification"`
	AvgRecallTokens       float64                          `json:"avg_recall_tokens"`
	MeanLatencyUS         float64                          `json:"mean_latency_us"`
	P50LatencyUS          int64                            `json:"p50_latency_us"`
	P95LatencyUS          int64                            `json:"p95_latency_us"`
	Outcomes              []CaseOutcome                    `json:"outcomes,omitempty"`
}

// HistoricalRegimeResult captures Negative Regime 1 (recency bias on pre-migration
// point-in-time AsOf queries).
type HistoricalRegimeResult struct {
	RegimeName string                `json:"regime_name"`
	CaseCount  int                   `json:"case_count"`
	Policies   []HistoricalPolicyRow `json:"policies"`
}

// HistoricalPolicyRow reports one policy's accuracy and recency-bias error rate
// on historical point-in-time queries.
type HistoricalPolicyRow struct {
	PolicyName          string  `json:"policy_name"`
	Accuracy            float64 `json:"accuracy"`
	RecencyBiasRate     float64 `json:"recency_bias_rate"`
	AbstentionRate      float64 `json:"abstention_rate"`
	CorrectCases        int     `json:"correct_cases"`
	RecencyBiasFailures int     `json:"recency_bias_failures"`
	TotalCases          int     `json:"total_cases"`
}

// HorizonScalingPoint captures Negative Regime 2 metrics at a single distractor
// session horizon (e.g. 5, 25, 50, 100, 200 sessions).
type HorizonScalingPoint struct {
	DistractorSessions int     `json:"distractor_sessions"`
	PolicyName         string  `json:"policy_name"`
	NeedleRecall       float64 `json:"needle_recall"`
	UnanswerableFPR    float64 `json:"unanswerable_fpr"`
	AbstentionF1       float64 `json:"abstention_f1"`
	OverallAccuracy    float64 `json:"overall_accuracy"`
	AvgRecallTokens    float64 `json:"avg_recall_tokens"`
	MeanLatencyUS      float64 `json:"mean_latency_us"`
	CrossTenantLeaks   int     `json:"cross_tenant_leaks"`
}

// LongHorizonRegimeResult summarizes Negative Regime 2 across session scales.
type LongHorizonRegimeResult struct {
	RegimeName string                `json:"regime_name"`
	Horizons   []int                 `json:"horizons"`
	Points     []HorizonScalingPoint `json:"points"`
}

// ParetoPoint captures Negative Regime 3 metrics at one abstention threshold theta.
type ParetoPoint struct {
	Threshold          float64 `json:"threshold"`
	ParaphraseRecall   float64 `json:"paraphrase_recall"`
	UnanswerableFPR    float64 `json:"unanswerable_fpr"`
	OverAbstentionRate float64 `json:"over_abstention_rate"`
	AbstentionPrec     float64 `json:"abstention_precision"`
	AbstentionRecall   float64 `json:"abstention_recall"`
	AbstentionF1       float64 `json:"abstention_f1"`
	OverallAccuracy    float64 `json:"overall_accuracy"`
	AvgRecallTokens    float64 `json:"avg_recall_tokens"`
}

// AbstentionParetoResult summarizes Negative Regime 3 across evidence thresholds.
type AbstentionParetoResult struct {
	RegimeName       string        `json:"regime_name"`
	PolicyName       string        `json:"policy_name"`
	ValidCases       int           `json:"valid_paraphrase_cases"`
	UnanswerableCase int           `json:"unanswerable_cases"`
	BestThreshold    float64       `json:"best_threshold"`
	BestAccuracy     float64       `json:"best_accuracy"`
	Points           []ParetoPoint `json:"points"`
}

// BenchmarkReport is the complete reproducible HIP-1346 LongMemEval output.
type BenchmarkReport struct {
	SchemaVersion      string                   `json:"schema_version"`
	HIP                string                   `json:"hip"`
	Issue              string                   `json:"issue"`
	Seed               int64                    `json:"seed"`
	GoVersion          string                   `json:"go_version"`
	GOOS               string                   `json:"goos"`
	GOARCH             string                   `json:"goarch"`
	NumCPU             int                      `json:"num_cpu"`
	DatasetCases       int                      `json:"dataset_cases"`
	Categories         []TaskCategory           `json:"categories"`
	PolicyResults      []PolicyMetrics          `json:"policy_results"`
	HistoricalRegime   *HistoricalRegimeResult  `json:"negative_regime_1_historical_recency_bias,omitempty"`
	LongHorizonRegime  *LongHorizonRegimeResult `json:"negative_regime_2_long_horizon_distractors,omitempty"`
	AbstentionPareto   *AbstentionParetoResult  `json:"negative_regime_3_abstention_pareto,omitempty"`
	CanonicalStackNote string                   `json:"canonical_stack_note"`
}
