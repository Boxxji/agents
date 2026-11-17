package membench

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/hanzoai/agents/control-plane/pkg/types"
)

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

// MemoryKind separates direct observed evidence from inferred hypotheses (#64).
type MemoryKind string

const (
	KindObservation MemoryKind = "observation"
	KindHypothesis  MemoryKind = "hypothesis"
)

// SessionTurn represents a single conversational or tool event ingested by a memory policy.
type SessionTurn struct {
	TurnID     string       `json:"turn_id"`
	TenantID   string       `json:"tenant_id"`
	ActorID    string       `json:"actor_id"`
	SessionID  string       `json:"session_id"`
	Scope      string       `json:"scope,omitempty"`
	Timestamp  time.Time    `json:"timestamp"`
	Entity     string       `json:"entity"`
	Attribute  string       `json:"attribute"`
	Value      string       `json:"value"`
	Statement  string       `json:"statement"`
	Action     MemoryAction `json:"action"`
	Kind       MemoryKind   `json:"kind,omitempty"`
	Confidence float64      `json:"confidence,omitempty"`
	SourceID   string       `json:"source_id,omitempty"`
}

// RecallQuery specifies a LongMemEval evaluation query.
type RecallQuery struct {
	CaseID      string       `json:"case_id"`
	Category    TaskCategory `json:"category"`
	TenantID    string       `json:"tenant_id"`
	ActorID     string       `json:"actor_id"`
	SessionIDs  []string     `json:"session_ids,omitempty"`
	Question    string       `json:"question"`
	Entity      string       `json:"entity"`
	Attribute   string       `json:"attribute"`
	QueryTime   time.Time    `json:"query_time"`
	TokenBudget int          `json:"token_budget,omitempty"`
}

// RecallResult captures a policy's grounded recall output and resource usage.
type RecallResult struct {
	Answer          string   `json:"answer"`
	OrderedAnswers  []string `json:"ordered_answers,omitempty"`
	Abstained       bool     `json:"abstained"`
	EvidenceKeys    []string `json:"evidence_keys,omitempty"`
	EvidenceTenants []string `json:"evidence_tenants,omitempty"`
	RetrievalTokens int      `json:"retrieval_tokens"`
	Confidence      float64  `json:"confidence"`
}

// IngestStats counts storage writes and deduplicated no-ops per ingested turn.
type IngestStats struct {
	MemoryWrites int `json:"memory_writes"`
	EventWrites  int `json:"event_writes"`
	SkippedNoops int `json:"skipped_noops"`
}

// MemoryPolicy defines an adapter evaluating an existing Hanzo memory policy
// over canonical MemoryBackend primitives without creating a competing engine.
type MemoryPolicy interface {
	Name() string
	Ingest(ctx context.Context, backend MemoryBackend, turn SessionTurn) (IngestStats, error)
	Recall(ctx context.Context, backend MemoryBackend, query RecallQuery) (RecallResult, error)
}

// EvalCase defines one LongMemEval scenario across memory sessions.
type EvalCase struct {
	ID                    string        `json:"id"`
	Category              TaskCategory  `json:"category"`
	TenantID              string        `json:"tenant_id"`
	ActorID               string        `json:"actor_id"`
	Turns                 []SessionTurn `json:"turns"`
	Query                 RecallQuery   `json:"query"`
	ExpectedAnswer        string        `json:"expected_answer,omitempty"`
	ExpectedOrder         []string      `json:"expected_order,omitempty"`
	ShouldAbstain         bool          `json:"should_abstain,omitempty"`
	ForbiddenStaleAnswers []string      `json:"forbidden_stale_answers,omitempty"`
}

// CategoryMetrics summarizes accuracy and failure modes for one TaskCategory.
type CategoryMetrics struct {
	Total            int     `json:"total"`
	Correct          int     `json:"correct"`
	Accuracy         float64 `json:"accuracy"`
	StaleErrors      int     `json:"stale_errors"`
	FalseMemories    int     `json:"false_memories"`
	MissedAbstention int     `json:"missed_abstentions"`
}

// PolicyReport holds complete LongMemEval metrics for a single MemoryPolicy.
type PolicyReport struct {
	PolicyName         string                           `json:"policy_name"`
	TotalCases         int                              `json:"total_cases"`
	CorrectCases       int                              `json:"correct_cases"`
	OverallAccuracy    float64                          `json:"overall_accuracy"`
	ByCategory         map[TaskCategory]CategoryMetrics `json:"by_category"`
	StaleFactErrorRate float64                          `json:"stale_fact_error_rate"`
	FalseMemoryRate    float64                          `json:"false_memory_rate"`
	AbstentionPrec     float64                          `json:"abstention_precision"`
	AbstentionRecall   float64                          `json:"abstention_recall"`
	CrossTenantLeaks   int                              `json:"cross_tenant_leaks"`
	TotalWrites        int                              `json:"total_writes"`
	WriteAmplification float64                          `json:"write_amplification"`
	TotalRecallTokens  int                              `json:"total_recall_tokens"`
	MeanLatencyUS      float64                          `json:"mean_latency_us"`
}

// BackendFactory creates an isolated MemoryBackend instance (e.g. LocalStorage or InMemoryBackend)
// along with an optional cleanup callback.
type BackendFactory func() (MemoryBackend, func(), error)

type storedMemoryPayload struct {
	TenantID   string       `json:"tenant_id"`
	ActorID    string       `json:"actor_id"`
	SessionID  string       `json:"session_id"`
	Entity     string       `json:"entity"`
	Attribute  string       `json:"attribute"`
	Value      string       `json:"value"`
	Statement  string       `json:"statement"`
	Action     MemoryAction `json:"action"`
	Kind       MemoryKind   `json:"kind"`
	Confidence float64      `json:"confidence"`
	Timestamp  time.Time    `json:"timestamp"`
	Deleted    bool         `json:"deleted,omitempty"`
}

// EvaluatePolicies runs each policy on identical EvalCase scenarios using isolated MemoryBackends.
func EvaluatePolicies(ctx context.Context, policies []MemoryPolicy, cases []EvalCase, factory BackendFactory) ([]PolicyReport, error) {
	if len(policies) == 0 {
		return nil, errors.New("membench: at least one memory policy is required")
	}
	if len(cases) == 0 {
		return nil, errors.New("membench: evaluation cases cannot be empty")
	}
	if factory == nil {
		factory = func() (MemoryBackend, func(), error) {
			return NewInMemoryBackend(), nil, nil
		}
	}

	reports := make([]PolicyReport, 0, len(policies))
	for _, policy := range policies {
		rep := PolicyReport{
			PolicyName: policy.Name(),
			TotalCases: len(cases),
			ByCategory: make(map[TaskCategory]CategoryMetrics, len(AllCategories)),
		}
		for _, cat := range AllCategories {
			rep.ByCategory[cat] = CategoryMetrics{}
		}

		var (
			totalIngestTurns int
			totalLatUS       int64
			staleErrors      int
			falseMemories    int
			abstainExpected  int
			abstainPredicted int
			abstainTP        int
		)

		for _, tc := range cases {
			backend, cleanup, err := factory()
			if err != nil {
				return nil, fmt.Errorf("create backend for case %s: %w", tc.ID, err)
			}

			runCase := func() error {
				if cleanup != nil {
					defer cleanup()
				}
				for _, turn := range tc.Turns {
					totalIngestTurns++
					stats, err := policy.Ingest(ctx, backend, turn)
					if err != nil {
						return fmt.Errorf("policy %s ingest case %s turn %s: %w", policy.Name(), tc.ID, turn.TurnID, err)
					}
					rep.TotalWrites += stats.MemoryWrites + stats.EventWrites
				}

				start := time.Now()
				res, err := policy.Recall(ctx, backend, tc.Query)
				totalLatUS += time.Since(start).Microseconds()
				if err != nil {
					return fmt.Errorf("policy %s recall case %s: %w", policy.Name(), tc.ID, err)
				}
				rep.TotalRecallTokens += res.RetrievalTokens

				for _, evTenant := range res.EvidenceTenants {
					if evTenant != "" && tc.Query.TenantID != "" && evTenant != tc.Query.TenantID {
						rep.CrossTenantLeaks++
					}
				}

				catStats := rep.ByCategory[tc.Category]
				catStats.Total++

				if tc.ShouldAbstain {
					abstainExpected++
				}
				if res.Abstained {
					abstainPredicted++
				}

				passed := false
				if tc.ShouldAbstain {
					if res.Abstained {
						abstainTP++
						passed = true
					} else {
						falseMemories++
						catStats.FalseMemories++
					}
				} else {
					if res.Abstained {
						catStats.MissedAbstention++
					} else if len(tc.ExpectedOrder) > 0 {
						passed = equalNormalizedSlice(res.OrderedAnswers, tc.ExpectedOrder)
					} else {
						passed = normalizeAnswer(res.Answer) == normalizeAnswer(tc.ExpectedAnswer)
					}
					for _, stale := range tc.ForbiddenStaleAnswers {
						if !res.Abstained && normalizeAnswer(res.Answer) == normalizeAnswer(stale) {
							staleErrors++
							catStats.StaleErrors++
							break
						}
					}
				}

				if passed {
					rep.CorrectCases++
					catStats.Correct++
				}
				rep.ByCategory[tc.Category] = catStats
				return nil
			}

			if err := runCase(); err != nil {
				return nil, err
			}
		}

		for cat, m := range rep.ByCategory {
			if m.Total > 0 {
				m.Accuracy = float64(m.Correct) / float64(m.Total)
				rep.ByCategory[cat] = m
			}
		}
		rep.OverallAccuracy = float64(rep.CorrectCases) / float64(rep.TotalCases)
		rep.StaleFactErrorRate = float64(staleErrors) / float64(rep.TotalCases)
		if abstainExpected > 0 {
			rep.FalseMemoryRate = float64(falseMemories) / float64(abstainExpected)
			rep.AbstentionRecall = float64(abstainTP) / float64(abstainExpected)
		}
		if abstainPredicted > 0 {
			rep.AbstentionPrec = float64(abstainTP) / float64(abstainPredicted)
		}
		if totalIngestTurns > 0 {
			rep.WriteAmplification = float64(rep.TotalWrites) / float64(totalIngestTurns)
		}
		rep.MeanLatencyUS = float64(totalLatUS) / float64(rep.TotalCases)
		reports = append(reports, rep)
	}
	return reports, nil
}

// FormatLongMemEvalMarkdown renders a reproducible comparison table across policies.
func FormatLongMemEvalMarkdown(reports []PolicyReport) string {
	var b strings.Builder
	b.WriteString("| Policy | Overall | Update | Contradiction | Ordering | Abstention | Cross-Session | Stale Err | Tenant Leaks | Write Amp |\n")
	b.WriteString("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
	for _, r := range reports {
		_, _ = fmt.Fprintf(&b, "| `%s` | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %d | %.2fx |\n",
			r.PolicyName,
			r.OverallAccuracy*100,
			r.ByCategory[CategoryTemporalUpdate].Accuracy*100,
			r.ByCategory[CategoryContradiction].Accuracy*100,
			r.ByCategory[CategoryTemporalOrdering].Accuracy*100,
			r.ByCategory[CategoryAbstention].Accuracy*100,
			r.ByCategory[CategoryCrossSession].Accuracy*100,
			r.StaleFactErrorRate*100,
			r.CrossTenantLeaks,
			r.WriteAmplification,
		)
	}
	return b.String()
}

// BaselineAppendPolicy stores raw session turns in session scope without reconciling updates,
// belief separation, or abstention boundaries.
type BaselineAppendPolicy struct{}

func (BaselineAppendPolicy) Name() string { return "baseline-append" }

func (BaselineAppendPolicy) Ingest(ctx context.Context, backend MemoryBackend, turn SessionTurn) (IngestStats, error) {
	payload := payloadFromTurn(turn)
	raw, err := json.Marshal(payload)
	if err != nil {
		return IngestStats{}, err
	}
	mem := &types.Memory{
		Scope:     "session",
		ScopeID:   turn.SessionID,
		Key:       fmt.Sprintf("%020d:%s", turn.Timestamp.UTC().UnixNano(), turn.TurnID),
		Data:      raw,
		CreatedAt: turn.Timestamp,
		UpdatedAt: turn.Timestamp,
	}
	if err := backend.SetMemory(ctx, mem); err != nil {
		return IngestStats{}, err
	}
	return IngestStats{MemoryWrites: 1}, nil
}

func (BaselineAppendPolicy) Recall(ctx context.Context, backend MemoryBackend, query RecallQuery) (RecallResult, error) {
	var candidates []storedMemoryPayload
	var keys []string
	var tenants []string
	tokens := 0
	for _, sid := range query.SessionIDs {
		mems, err := backend.ListMemory(ctx, "session", sid)
		if err != nil {
			return RecallResult{}, err
		}
		for _, m := range mems {
			var p storedMemoryPayload
			if err := json.Unmarshal(m.Data, &p); err != nil {
				continue
			}
			tokens += estimateTokens(p.Statement)
			if matchEntityAttr(p, query.Entity, query.Attribute) {
				candidates = append(candidates, p)
				keys = append(keys, m.Key)
				tenants = append(tenants, p.TenantID)
			}
		}
	}
	if len(candidates) == 0 {
		// Baseline fails open with a guess when query asks for unrecorded/deleted facts if session has text.
		return RecallResult{Answer: "unknown", RetrievalTokens: tokens}, nil
	}
	// Baseline returns the earliest entry (fails on temporal updates & ordering) and never abstains on deleted/low-confidence entries.
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Timestamp.Before(candidates[j].Timestamp)
	})
	ordered := make([]string, 0, len(candidates))
	for _, c := range candidates {
		ordered = append(ordered, c.Value)
	}
	return RecallResult{
		Answer:          candidates[0].Value,
		OrderedAnswers:  ordered,
		EvidenceKeys:    keys,
		EvidenceTenants: tenants,
		RetrievalTokens: tokens,
		Confidence:      candidates[0].Confidence,
	}, nil
}

// AtomicFactPolicy implements selective ADD/UPDATE/DELETE/NOOP reconciliation (#62/#63)
// keyed by (entity, attribute) inside tenant-scoped actor memory.
type AtomicFactPolicy struct{}

func (AtomicFactPolicy) Name() string { return "selective-atomic-fact" }

func (AtomicFactPolicy) Ingest(ctx context.Context, backend MemoryBackend, turn SessionTurn) (IngestStats, error) {
	scopeID := tenantActorScopeID(turn.TenantID, turn.ActorID)
	key := factKey(turn.Entity, turn.Attribute)

	existing, err := backend.GetMemory(ctx, "actor", scopeID, key)
	if err == nil && existing != nil {
		var prev storedMemoryPayload
		if json.Unmarshal(existing.Data, &prev) == nil &&
			prev.Value == turn.Value && !prev.Deleted &&
			turn.Action != ActionDelete && turn.Action != ActionForget {
			return IngestStats{SkippedNoops: 1}, nil
		}
	}

	if turn.Action == ActionDelete || turn.Action == ActionForget {
		_ = backend.DeleteMemory(ctx, "actor", scopeID, key)
		_ = backend.StoreEvent(ctx, &types.MemoryChangeEvent{
			Type:      "memory_change",
			Scope:     "actor",
			ScopeID:   scopeID,
			Key:       key,
			Action:    "delete",
			Timestamp: turn.Timestamp,
			Metadata:  types.EventMetadata{ActorID: turn.ActorID},
		})
		return IngestStats{MemoryWrites: 1, EventWrites: 1}, nil
	}

	payload := payloadFromTurn(turn)
	raw, err := json.Marshal(payload)
	if err != nil {
		return IngestStats{}, err
	}
	if err := backend.SetMemory(ctx, &types.Memory{
		Scope:     "actor",
		ScopeID:   scopeID,
		Key:       key,
		Data:      raw,
		CreatedAt: turn.Timestamp,
		UpdatedAt: turn.Timestamp,
	}); err != nil {
		return IngestStats{}, err
	}
	if err := backend.StoreEvent(ctx, &types.MemoryChangeEvent{
		Type:      "memory_change",
		Scope:     "actor",
		ScopeID:   scopeID,
		Key:       key,
		Action:    "set",
		Data:      raw,
		Timestamp: turn.Timestamp,
		Metadata:  types.EventMetadata{ActorID: turn.ActorID},
	}); err != nil {
		return IngestStats{}, err
	}
	return IngestStats{MemoryWrites: 1, EventWrites: 1}, nil
}

func (AtomicFactPolicy) Recall(ctx context.Context, backend MemoryBackend, query RecallQuery) (RecallResult, error) {
	scopeID := tenantActorScopeID(query.TenantID, query.ActorID)
	key := factKey(query.Entity, query.Attribute)
	mem, err := backend.GetMemory(ctx, "actor", scopeID, key)
	if err != nil || mem == nil {
		return RecallResult{Abstained: true, Answer: "insufficient evidence"}, nil
	}
	var p storedMemoryPayload
	if err := json.Unmarshal(mem.Data, &p); err != nil || p.TenantID != query.TenantID || p.Deleted {
		return RecallResult{Abstained: true, Answer: "insufficient evidence"}, nil
	}
	tokens := estimateTokens(p.Statement)
	return RecallResult{
		Answer:          p.Value,
		OrderedAnswers:  []string{p.Value},
		EvidenceKeys:    []string{key},
		EvidenceTenants: []string{p.TenantID},
		RetrievalTokens: tokens,
		Confidence:      p.Confidence,
	}, nil
}

// TemporalBeliefPolicy evaluates #63/#64/#66 combined semantics over MemoryBackend:
// tenant-isolated actor/session memories, observation-over-hypothesis precedence,
// immutable timeline event logs for ordering, and confidence-calibrated abstention.
type TemporalBeliefPolicy struct {
	MinConfidence float64
}

func (p TemporalBeliefPolicy) Name() string { return "temporal-belief-calibrated" }

func (p TemporalBeliefPolicy) minConf() float64 {
	if p.MinConfidence <= 0 {
		return 0.55
	}
	return p.MinConfidence
}

func (p TemporalBeliefPolicy) Ingest(ctx context.Context, backend MemoryBackend, turn SessionTurn) (IngestStats, error) {
	if turn.Action == ActionNoop {
		return IngestStats{SkippedNoops: 1}, nil
	}

	scopeID := tenantActorScopeID(turn.TenantID, turn.ActorID)
	beliefKey := "belief:" + factKey(turn.Entity, turn.Attribute)
	payload := payloadFromTurn(turn)

	if turn.Action == ActionDelete || turn.Action == ActionForget {
		payload.Deleted = true
		raw, err := json.Marshal(payload)
		if err != nil {
			return IngestStats{}, err
		}
		if err := backend.SetMemory(ctx, &types.Memory{
			Scope:     "actor",
			ScopeID:   scopeID,
			Key:       beliefKey,
			Data:      raw,
			CreatedAt: turn.Timestamp,
			UpdatedAt: turn.Timestamp,
		}); err != nil {
			return IngestStats{}, err
		}
		if err := backend.StoreEvent(ctx, &types.MemoryChangeEvent{
			Type:      "memory_change",
			Scope:     "actor",
			ScopeID:   scopeID,
			Key:       beliefKey,
			Action:    "delete",
			Data:      raw,
			Timestamp: turn.Timestamp,
			Metadata:  types.EventMetadata{ActorID: turn.ActorID},
		}); err != nil {
			return IngestStats{}, err
		}
		return IngestStats{MemoryWrites: 1, EventWrites: 1}, nil
	}

	// Belief separation (#64): an inferred hypothesis never silently overwrites
	// a higher-precedence direct observation unless no observation exists.
	var prevPayload *storedMemoryPayload
	if existing, err := backend.GetMemory(ctx, "actor", scopeID, beliefKey); err == nil && existing != nil {
		var prev storedMemoryPayload
		if json.Unmarshal(existing.Data, &prev) == nil && prev.TenantID == turn.TenantID {
			prevPayload = &prev
		}
	}

	var stats IngestStats
	shouldReplaceBelief := prevPayload == nil || prevPayload.Deleted
	if !shouldReplaceBelief {
		switch {
		case prevPayload.Kind == KindObservation && payload.Kind == KindHypothesis:
			shouldReplaceBelief = false
		case payload.Kind == KindObservation && prevPayload.Kind == KindHypothesis:
			shouldReplaceBelief = true
		case !payload.Timestamp.Before(prevPayload.Timestamp):
			shouldReplaceBelief = true
		}
	}

	raw, err := json.Marshal(payload)
	if err != nil {
		return stats, err
	}

	if shouldReplaceBelief {
		if prevPayload != nil && !prevPayload.Deleted && prevPayload.Value == payload.Value && prevPayload.Kind == payload.Kind {
			stats.SkippedNoops++
		} else {
			if err := backend.SetMemory(ctx, &types.Memory{
				Scope:     "actor",
				ScopeID:   scopeID,
				Key:       beliefKey,
				Data:      raw,
				CreatedAt: turn.Timestamp,
				UpdatedAt: turn.Timestamp,
			}); err != nil {
				return stats, err
			}
			stats.MemoryWrites++
		}
	}

	// Immutable observation/timeline record for temporal ordering queries.
	if payload.Kind == KindObservation {
		tlKey := fmt.Sprintf("timeline:%s:%s:%020d:%s",
			strings.ToLower(turn.Entity), strings.ToLower(turn.Attribute),
			turn.Timestamp.UTC().UnixNano(), turn.TurnID)
		if err := backend.SetMemory(ctx, &types.Memory{
			Scope:     "actor",
			ScopeID:   scopeID,
			Key:       tlKey,
			Data:      raw,
			CreatedAt: turn.Timestamp,
			UpdatedAt: turn.Timestamp,
		}); err != nil {
			return stats, err
		}
		stats.MemoryWrites++
	}

	if err := backend.StoreEvent(ctx, &types.MemoryChangeEvent{
		Type:      "memory_change",
		Scope:     "actor",
		ScopeID:   scopeID,
		Key:       beliefKey,
		Action:    "set",
		Data:      raw,
		Timestamp: turn.Timestamp,
		Metadata:  types.EventMetadata{ActorID: turn.ActorID},
	}); err != nil {
		return stats, err
	}
	stats.EventWrites++
	return stats, nil
}

func (p TemporalBeliefPolicy) Recall(ctx context.Context, backend MemoryBackend, query RecallQuery) (RecallResult, error) {
	scopeID := tenantActorScopeID(query.TenantID, query.ActorID)
	beliefKey := "belief:" + factKey(query.Entity, query.Attribute)

	if query.Category == CategoryTemporalOrdering {
		mems, err := backend.ListMemory(ctx, "actor", scopeID)
		if err != nil {
			return RecallResult{}, err
		}
		tlPrefix := fmt.Sprintf("timeline:%s:%s:", strings.ToLower(query.Entity), strings.ToLower(query.Attribute))
		var entries []storedMemoryPayload
		var keys []string
		var tenants []string
		tokens := 0
		for _, m := range mems {
			if !strings.HasPrefix(m.Key, tlPrefix) {
				continue
			}
			var sp storedMemoryPayload
			if err := json.Unmarshal(m.Data, &sp); err != nil {
				continue
			}
			// Enforce tenant ACL BEFORE including any evidence hit.
			if sp.TenantID != query.TenantID || sp.Deleted {
				continue
			}
			if !query.QueryTime.IsZero() && sp.Timestamp.After(query.QueryTime) {
				continue
			}
			entries = append(entries, sp)
			keys = append(keys, m.Key)
			tenants = append(tenants, sp.TenantID)
			tokens += estimateTokens(sp.Statement)
		}
		if len(entries) == 0 {
			return RecallResult{Abstained: true, Answer: "insufficient evidence", RetrievalTokens: tokens}, nil
		}
		sort.Slice(entries, func(i, j int) bool {
			return entries[i].Timestamp.Before(entries[j].Timestamp)
		})
		ordered := make([]string, 0, len(entries))
		for _, e := range entries {
			ordered = append(ordered, e.Value)
		}
		return RecallResult{
			Answer:          strings.Join(ordered, " -> "),
			OrderedAnswers:  ordered,
			EvidenceKeys:    keys,
			EvidenceTenants: tenants,
			RetrievalTokens: tokens,
			Confidence:      entries[len(entries)-1].Confidence,
		}, nil
	}

	mem, err := backend.GetMemory(ctx, "actor", scopeID, beliefKey)
	if err != nil || mem == nil {
		return RecallResult{Abstained: true, Answer: "insufficient evidence"}, nil
	}
	var sp storedMemoryPayload
	if err := json.Unmarshal(mem.Data, &sp); err != nil {
		return RecallResult{Abstained: true, Answer: "insufficient evidence"}, nil
	}
	// Strict tenant filter before surfacing evidence.
	if sp.TenantID != query.TenantID || sp.Deleted {
		return RecallResult{Abstained: true, Answer: "insufficient evidence"}, nil
	}
	tokens := estimateTokens(sp.Statement)
	if sp.Confidence > 0 && sp.Confidence < p.minConf() {
		return RecallResult{
			Abstained:       true,
			Answer:          "insufficient evidence",
			EvidenceKeys:    []string{beliefKey},
			EvidenceTenants: []string{sp.TenantID},
			RetrievalTokens: tokens,
			Confidence:      sp.Confidence,
		}, nil
	}

	return RecallResult{
		Answer:          sp.Value,
		OrderedAnswers:  []string{sp.Value},
		EvidenceKeys:    []string{beliefKey},
		EvidenceTenants: []string{sp.TenantID},
		RetrievalTokens: tokens,
		Confidence:      sp.Confidence,
	}, nil
}

// DefaultLongMemEvalSuite constructs a reproducible benchmark covering all five
// LongMemEval dimensions plus multi-tenant isolation.
func DefaultLongMemEvalSuite() []EvalCase {
	t0 := time.Date(2026, 8, 1, 9, 0, 0, 0, time.UTC)
	return []EvalCase{
		{
			ID:       "lme-temporal-update-1",
			Category: CategoryTemporalUpdate,
			TenantID: "tenant-a",
			ActorID:  "user-1",
			Turns: []SessionTurn{
				{
					TurnID: "t1", TenantID: "tenant-a", ActorID: "user-1", SessionID: "s1",
					Timestamp: t0, Entity: "user", Attribute: "home_city", Value: "New York",
					Statement: "I currently live in New York.", Action: ActionAdd, Kind: KindObservation, Confidence: 0.95,
				},
				{
					TurnID: "t2", TenantID: "tenant-a", ActorID: "user-1", SessionID: "s2",
					Timestamp: t0.Add(48 * time.Hour), Entity: "user", Attribute: "home_city", Value: "San Francisco",
					Statement: "I relocated to San Francisco this week.", Action: ActionUpdate, Kind: KindObservation, Confidence: 0.98,
				},
			},
			Query: RecallQuery{
				CaseID: "lme-temporal-update-1", Category: CategoryTemporalUpdate,
				TenantID: "tenant-a", ActorID: "user-1", SessionIDs: []string{"s1", "s2"},
				Question: "Where does the user live right now?", Entity: "user", Attribute: "home_city",
				QueryTime: t0.Add(72 * time.Hour),
			},
			ExpectedAnswer:        "San Francisco",
			ForbiddenStaleAnswers: []string{"New York"},
		},
		{
			ID:       "lme-contradiction-1",
			Category: CategoryContradiction,
			TenantID: "tenant-a",
			ActorID:  "user-2",
			Turns: []SessionTurn{
				{
					TurnID: "t1", TenantID: "tenant-a", ActorID: "user-2", SessionID: "s1",
					Timestamp: t0, Entity: "build", Attribute: "root_cause", Value: "pool_exhaustion",
					Statement: "Observed connection pool timeout metrics spiking to 100%.", Action: ActionAdd, Kind: KindObservation, Confidence: 0.96,
				},
				{
					TurnID: "t2", TenantID: "tenant-a", ActorID: "user-2", SessionID: "s1",
					Timestamp: t0.Add(1 * time.Hour), Entity: "build", Attribute: "root_cause", Value: "dns_flake",
					Statement: "Hypothesis: maybe it was a transient DNS flake.", Action: ActionReflect, Kind: KindHypothesis, Confidence: 0.45,
				},
			},
			Query: RecallQuery{
				CaseID: "lme-contradiction-1", Category: CategoryContradiction,
				TenantID: "tenant-a", ActorID: "user-2", SessionIDs: []string{"s1"},
				Question: "What is the verified root cause?", Entity: "build", Attribute: "root_cause",
				QueryTime: t0.Add(2 * time.Hour),
			},
			ExpectedAnswer:        "pool_exhaustion",
			ForbiddenStaleAnswers: []string{"dns_flake"},
		},
		{
			ID:       "lme-temporal-ordering-1",
			Category: CategoryTemporalOrdering,
			TenantID: "tenant-a",
			ActorID:  "user-3",
			Turns: []SessionTurn{
				{
					TurnID: "t2", TenantID: "tenant-a", ActorID: "user-3", SessionID: "s2",
					Timestamp: t0.Add(24 * time.Hour), Entity: "deploy", Attribute: "stage", Value: "staging",
					Statement: "Promoted service to staging.", Action: ActionUpdate, Kind: KindObservation, Confidence: 0.95,
				},
				{
					TurnID: "t1", TenantID: "tenant-a", ActorID: "user-3", SessionID: "s1",
					Timestamp: t0, Entity: "deploy", Attribute: "stage", Value: "dev",
					Statement: "Deployed build to dev.", Action: ActionAdd, Kind: KindObservation, Confidence: 0.95,
				},
				{
					TurnID: "t3", TenantID: "tenant-a", ActorID: "user-3", SessionID: "s3",
					Timestamp: t0.Add(72 * time.Hour), Entity: "deploy", Attribute: "stage", Value: "prod",
					Statement: "Rolled out release to prod.", Action: ActionUpdate, Kind: KindObservation, Confidence: 0.99,
				},
			},
			Query: RecallQuery{
				CaseID: "lme-temporal-ordering-1", Category: CategoryTemporalOrdering,
				TenantID: "tenant-a", ActorID: "user-3", SessionIDs: []string{"s2", "s1", "s3"},
				Question: "List deployment stages chronologically.", Entity: "deploy", Attribute: "stage",
				QueryTime: t0.Add(96 * time.Hour),
			},
			ExpectedOrder: []string{"dev", "staging", "prod"},
		},
		{
			ID:       "lme-abstention-revoked-1",
			Category: CategoryAbstention,
			TenantID: "tenant-a",
			ActorID:  "user-4",
			Turns: []SessionTurn{
				{
					TurnID: "t1", TenantID: "tenant-a", ActorID: "user-4", SessionID: "s1",
					Timestamp: t0, Entity: "user", Attribute: "phone_number", Value: "555-0199",
					Statement: "My phone number is 555-0199.", Action: ActionAdd, Kind: KindObservation, Confidence: 0.95,
				},
				{
					TurnID: "t2", TenantID: "tenant-a", ActorID: "user-4", SessionID: "s2",
					Timestamp: t0.Add(12 * time.Hour), Entity: "user", Attribute: "phone_number", Value: "",
					Statement: "Forget my phone number immediately.", Action: ActionForget, Kind: KindObservation, Confidence: 1.0,
				},
			},
			Query: RecallQuery{
				CaseID: "lme-abstention-revoked-1", Category: CategoryAbstention,
				TenantID: "tenant-a", ActorID: "user-4", SessionIDs: []string{"s1", "s2"},
				Question: "What is the user's phone number?", Entity: "user", Attribute: "phone_number",
				QueryTime: t0.Add(24 * time.Hour),
			},
			ShouldAbstain:         true,
			ForbiddenStaleAnswers: []string{"555-0199"},
		},
		{
			ID:       "lme-abstention-tenant-isolation-1",
			Category: CategoryAbstention,
			TenantID: "tenant-b",
			ActorID:  "shared-actor",
			Turns: []SessionTurn{
				{
					TurnID: "t1", TenantID: "tenant-a", ActorID: "shared-actor", SessionID: "s1",
					Timestamp: t0, Entity: "secret", Attribute: "api_quota", Value: "9000",
					Statement: "Tenant A quota is 9000.", Action: ActionAdd, Kind: KindObservation, Confidence: 0.99,
				},
			},
			Query: RecallQuery{
				CaseID: "lme-abstention-tenant-isolation-1", Category: CategoryAbstention,
				TenantID: "tenant-b", ActorID: "shared-actor", SessionIDs: []string{"s1"},
				Question: "What is the api quota?", Entity: "secret", Attribute: "api_quota",
				QueryTime: t0.Add(2 * time.Hour),
			},
			ShouldAbstain:         true,
			ForbiddenStaleAnswers: []string{"9000"},
		},
		{
			ID:       "lme-cross-session-1",
			Category: CategoryCrossSession,
			TenantID: "tenant-a",
			ActorID:  "user-5",
			Turns: []SessionTurn{
				{
					TurnID: "t1", TenantID: "tenant-a", ActorID: "user-5", SessionID: "s1",
					Timestamp: t0, Entity: "project", Attribute: "target_db", Value: "postgres",
					Statement: "We standardized the project on postgres.", Action: ActionAdd, Kind: KindObservation, Confidence: 0.94,
				},
				{
					TurnID: "t2", TenantID: "tenant-a", ActorID: "user-5", SessionID: "s2",
					Timestamp: t0.Add(36 * time.Hour), Entity: "project", Attribute: "ci_runner", Value: "visor",
					Statement: "CI runs inside visor sandboxes.", Action: ActionAdd, Kind: KindObservation, Confidence: 0.97,
				},
			},
			Query: RecallQuery{
				CaseID: "lme-cross-session-1", Category: CategoryCrossSession,
				TenantID: "tenant-a", ActorID: "user-5", SessionIDs: []string{"s2"},
				Question: "Which database did the user standardize on in an earlier session?", Entity: "project", Attribute: "target_db",
				QueryTime: t0.Add(48 * time.Hour),
			},
			ExpectedAnswer: "postgres",
		},
	}
}

func payloadFromTurn(turn SessionTurn) storedMemoryPayload {
	kind := turn.Kind
	if kind == "" {
		kind = KindObservation
	}
	conf := turn.Confidence
	if conf == 0 {
		conf = 0.9
	}
	return storedMemoryPayload{
		TenantID:   turn.TenantID,
		ActorID:    turn.ActorID,
		SessionID:  turn.SessionID,
		Entity:     turn.Entity,
		Attribute:  turn.Attribute,
		Value:      turn.Value,
		Statement:  turn.Statement,
		Action:     turn.Action,
		Kind:       kind,
		Confidence: conf,
		Timestamp:  turn.Timestamp.UTC(),
	}
}

func tenantActorScopeID(tenantID, actorID string) string {
	return tenantID + ":" + actorID
}

func factKey(entity, attribute string) string {
	return strings.ToLower(strings.TrimSpace(entity)) + ":" + strings.ToLower(strings.TrimSpace(attribute))
}

func matchEntityAttr(p storedMemoryPayload, entity, attribute string) bool {
	return strings.EqualFold(strings.TrimSpace(p.Entity), strings.TrimSpace(entity)) &&
		strings.EqualFold(strings.TrimSpace(p.Attribute), strings.TrimSpace(attribute))
}

func normalizeAnswer(s string) string {
	return strings.ToLower(strings.TrimSpace(s))
}

func equalNormalizedSlice(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if normalizeAnswer(a[i]) != normalizeAnswer(b[i]) {
			return false
		}
	}
	return true
}
