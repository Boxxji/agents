package longmemeval

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"github.com/hanzoai/agents/control-plane/pkg/types"
)

// MemoryPolicy defines an evaluation adapter exercising an existing Hanzo
// Agents memory policy over canonical MemoryStore primitives.
type MemoryPolicy interface {
	Name() string
	Description() string
	Ingest(ctx context.Context, store MemoryStore, turn SessionTurn) (IngestStats, error)
	Recall(ctx context.Context, store MemoryStore, query RecallQuery) (KaiDecision, error)
}

type storedRecord struct {
	TurnID     string        `json:"turn_id"`
	TenantID   string        `json:"tenant_id"`
	ActorID    string        `json:"actor_id"`
	SessionID  string        `json:"session_id"`
	WorkflowID string        `json:"workflow_id,omitempty"`
	Scope      string        `json:"scope,omitempty"`
	ACLRoles   []string      `json:"acl_roles,omitempty"`
	Entity     string        `json:"entity"`
	Attribute  string        `json:"attribute"`
	Value      string        `json:"value"`
	Statement  string        `json:"statement"`
	Action     MemoryAction  `json:"action"`
	Kind       EpistemicKind `json:"kind"`
	Confidence float64       `json:"confidence"`
	SourceID   string        `json:"source_id,omitempty"`
	Timestamp  time.Time     `json:"timestamp"`
	Deleted    bool          `json:"deleted,omitempty"`
}

type contradictionLog struct {
	Values []string `json:"values"`
}

func recordFromTurn(turn SessionTurn) storedRecord {
	kind := turn.Kind
	if kind == "" {
		kind = KindObservation
	}
	conf := turn.Confidence
	if conf <= 0 {
		conf = 0.90
	}
	return storedRecord{
		TurnID:     turn.TurnID,
		TenantID:   turn.TenantID,
		ActorID:    turn.ActorID,
		SessionID:  turn.SessionID,
		WorkflowID: turn.WorkflowID,
		Scope:      turn.Scope,
		ACLRoles:   append([]string(nil), turn.ACLRoles...),
		Entity:     turn.Entity,
		Attribute:  turn.Attribute,
		Value:      turn.Value,
		Statement:  turn.Statement,
		Action:     turn.Action,
		Kind:       kind,
		Confidence: conf,
		SourceID:   turn.SourceID,
		Timestamp:  turn.Timestamp.UTC(),
	}
}

func actorScopeID(tenantID, actorID string) string {
	return tenantID + ":actor:" + actorID
}

func sessionScopeID(tenantID, sessionID string) string {
	return tenantID + ":session:" + sessionID
}

func workflowScopeID(tenantID, workflowID string) string {
	return tenantID + ":workflow:" + workflowID
}

func factKey(entity, attribute string) string {
	return strings.ToLower(strings.TrimSpace(entity)) + ":" + strings.ToLower(strings.TrimSpace(attribute))
}

func normalizeAnswer(s string) string {
	return strings.ToLower(strings.TrimSpace(s))
}

func epistemicRank(k EpistemicKind) int {
	switch k {
	case KindObservation:
		return 3
	case KindConclusion:
		return 2
	case KindHypothesis:
		return 1
	default:
		return 3
	}
}

// aclPermitted enforces pre-hit tenant and role-based ACL boundaries.
// Empty record ACLRoles means accessible to any caller within the same tenant.
func aclPermitted(recTenant string, recRoles []string, queryTenant string, callerRoles []string) bool {
	if recTenant != "" && queryTenant != "" && recTenant != queryTenant {
		return false
	}
	if len(recRoles) == 0 {
		return true
	}
	if len(callerRoles) == 0 {
		return false
	}
	roleSet := make(map[string]struct{}, len(callerRoles))
	for _, r := range callerRoles {
		roleSet[strings.ToLower(strings.TrimSpace(r))] = struct{}{}
	}
	for _, req := range recRoles {
		if _, ok := roleSet[strings.ToLower(strings.TrimSpace(req))]; ok {
			return true
		}
	}
	return false
}

func embedTextForTurn(turn SessionTurn) string {
	var parts []string
	if turn.Statement != "" {
		parts = append(parts, turn.Statement)
	}
	if turn.Entity != "" {
		parts = append(parts, turn.Entity)
	}
	if turn.Attribute != "" {
		parts = append(parts, strings.ReplaceAll(turn.Attribute, "_", " "))
	}
	if turn.Value != "" {
		parts = append(parts, turn.Value)
	}
	return strings.Join(parts, " ")
}

func embedTextForQuery(q RecallQuery) string {
	if q.QueryMode == QuerySemantic {
		return q.Question
	}
	var parts []string
	if q.Question != "" {
		parts = append(parts, q.Question)
	}
	if q.Entity != "" {
		parts = append(parts, q.Entity)
	}
	if q.Attribute != "" {
		parts = append(parts, strings.ReplaceAll(q.Attribute, "_", " "))
	}
	return strings.Join(parts, " ")
}

func abstainDecision(tokens int, score, conf float64, keys, tenants, scopes []string) KaiDecision {
	return KaiDecision{
		Status:                 DecisionAbstain,
		Answer:                 "INSUFFICIENT_EVIDENCE",
		Abstained:              true,
		EvidenceKeys:           keys,
		EvidenceTenants:        tenants,
		EvidenceScopes:         scopes,
		Confidence:             conf,
		EvidenceScore:          score,
		RetrievalTokens:        tokens,
		GenerativeOutputTokens: 0,
	}
}

func formatDurationHours(d time.Duration) string {
	if d < 0 {
		d = -d
	}
	hours := int(math.Round(d.Hours()))
	return fmt.Sprintf("%dh", hours)
}

func appendUniqueStr(dst []string, val string) []string {
	norm := normalizeAnswer(val)
	if norm == "" {
		return dst
	}
	for _, existing := range dst {
		if normalizeAnswer(existing) == norm {
			return dst
		}
	}
	return append(dst, val)
}

func resolveTimelineMode(entries []storedRecord, query RecallQuery, tokens int, keys, tenants, scopes []string) KaiDecision {
	mode := query.TemporalMode
	if mode == "" {
		mode = TemporalLatest
	}

	// Filter out future events after QueryTime when specified.
	filtered := make([]storedRecord, 0, len(entries))
	for _, e := range entries {
		if !query.QueryTime.IsZero() && e.Timestamp.After(query.QueryTime) {
			continue
		}
		filtered = append(filtered, e)
	}
	if len(filtered) == 0 {
		return abstainDecision(tokens, 0, 0, nil, nil, nil)
	}

	sort.Slice(filtered, func(i, j int) bool {
		if filtered[i].Timestamp.Equal(filtered[j].Timestamp) {
			return filtered[i].TurnID < filtered[j].TurnID
		}
		return filtered[i].Timestamp.Before(filtered[j].Timestamp)
	})

	switch mode {
	case TemporalFirst:
		first := filtered[0]
		return KaiDecision{
			Status:                 DecisionAnswer,
			Answer:                 first.Value,
			OrderedAnswers:         []string{first.Value},
			EvidenceKeys:           keys,
			EvidenceTenants:        tenants,
			EvidenceScopes:         scopes,
			Confidence:             first.Confidence,
			EvidenceScore:          1.0,
			RetrievalTokens:        tokens,
			GenerativeOutputTokens: 0,
		}

	case TemporalLast:
		last := filtered[len(filtered)-1]
		return KaiDecision{
			Status:                 DecisionAnswer,
			Answer:                 last.Value,
			OrderedAnswers:         []string{last.Value},
			EvidenceKeys:           keys,
			EvidenceTenants:        tenants,
			EvidenceScopes:         scopes,
			Confidence:             last.Confidence,
			EvidenceScore:          1.0,
			RetrievalTokens:        tokens,
			GenerativeOutputTokens: 0,
		}

	case TemporalDuration:
		var startTS, endTS time.Time
		var hasStart, hasEnd bool
		startNorm := normalizeAnswer(query.DurationStartValue)
		endNorm := normalizeAnswer(query.DurationEndValue)
		for _, e := range filtered {
			v := normalizeAnswer(e.Value)
			if !hasStart && v == startNorm {
				startTS = e.Timestamp
				hasStart = true
			}
			if v == endNorm {
				endTS = e.Timestamp
				hasEnd = true
			}
		}
		if !hasStart || !hasEnd {
			return abstainDecision(tokens, 0, 0, nil, nil, nil)
		}
		ans := formatDurationHours(endTS.Sub(startTS))
		return KaiDecision{
			Status:                 DecisionAnswer,
			Answer:                 ans,
			EvidenceKeys:           keys,
			EvidenceTenants:        tenants,
			EvidenceScopes:         scopes,
			Confidence:             0.98,
			EvidenceScore:          1.0,
			RetrievalTokens:        tokens,
			GenerativeOutputTokens: 0,
		}

	case TemporalWindow:
		var windowed []string
		for _, e := range filtered {
			if !query.WindowStart.IsZero() && e.Timestamp.Before(query.WindowStart) {
				continue
			}
			if !query.WindowEnd.IsZero() && e.Timestamp.After(query.WindowEnd) {
				continue
			}
			windowed = appendUniqueStr(windowed, e.Value)
		}
		if len(windowed) == 0 {
			return abstainDecision(tokens, 0, 0, nil, nil, nil)
		}
		return KaiDecision{
			Status:                 DecisionAnswer,
			Answer:                 strings.Join(windowed, " -> "),
			OrderedAnswers:         windowed,
			EvidenceKeys:           keys,
			EvidenceTenants:        tenants,
			EvidenceScopes:         scopes,
			Confidence:             0.95,
			EvidenceScore:          1.0,
			RetrievalTokens:        tokens,
			GenerativeOutputTokens: 0,
		}

	case TemporalAsOf:
		var latestValid *storedRecord
		for i := range filtered {
			e := &filtered[i]
			if !query.AsOf.IsZero() && e.Timestamp.After(query.AsOf) {
				continue
			}
			if e.Deleted {
				latestValid = nil
				continue
			}
			latestValid = e
		}
		if latestValid == nil {
			return abstainDecision(tokens, 0, 0, nil, nil, nil)
		}
		return KaiDecision{
			Status:                 DecisionAnswer,
			Answer:                 latestValid.Value,
			OrderedAnswers:         []string{latestValid.Value},
			EvidenceKeys:           keys,
			EvidenceTenants:        tenants,
			EvidenceScopes:         scopes,
			Confidence:             latestValid.Confidence,
			EvidenceScore:          1.0,
			RetrievalTokens:        tokens,
			GenerativeOutputTokens: 0,
		}

	case TemporalAggregate:
		active := make(map[string]string)
		for _, e := range filtered {
			norm := normalizeAnswer(e.Value)
			if e.Deleted {
				if norm == "" {
					clear(active)
				} else {
					delete(active, norm)
				}
				continue
			}
			if norm != "" {
				active[norm] = e.Value
			}
		}
		if len(active) == 0 {
			return abstainDecision(tokens, 0, 0, nil, nil, nil)
		}
		vals := make([]string, 0, len(active))
		for _, v := range active {
			vals = append(vals, v)
		}
		sort.Slice(vals, func(i, j int) bool {
			return normalizeAnswer(vals[i]) < normalizeAnswer(vals[j])
		})
		return KaiDecision{
			Status:                 DecisionAnswer,
			Answer:                 strings.Join(vals, ", "),
			AggregatedValues:       vals,
			EvidenceKeys:           keys,
			EvidenceTenants:        tenants,
			EvidenceScopes:         scopes,
			Confidence:             0.95,
			EvidenceScore:          1.0,
			RetrievalTokens:        tokens,
			GenerativeOutputTokens: 0,
		}

	default: // TemporalSequence
		ordered := make([]string, 0, len(filtered))
		for _, e := range filtered {
			if e.Deleted {
				if normalizeAnswer(e.Value) == "" {
					ordered = ordered[:0]
				}
				continue
			}
			ordered = appendUniqueStr(ordered, e.Value)
		}
		if len(ordered) == 0 {
			return abstainDecision(tokens, 0, 0, nil, nil, nil)
		}
		return KaiDecision{
			Status:                 DecisionAnswer,
			Answer:                 strings.Join(ordered, " -> "),
			OrderedAnswers:         ordered,
			EvidenceKeys:           keys,
			EvidenceTenants:        tenants,
			EvidenceScopes:         scopes,
			Confidence:             filtered[len(filtered)-1].Confidence,
			EvidenceScore:          1.0,
			RetrievalTokens:        tokens,
			GenerativeOutputTokens: 0,
		}
	}
}

// ============================================================================
// Policy 1: FlatKVLatestPolicy
// Adapts control-plane/internal/handlers/memory.go flat actor-scope KV
// overwrite (SetMemory / GetMemory / DeleteMemory).
// ============================================================================

type FlatKVLatestPolicy struct{}

func (FlatKVLatestPolicy) Name() string { return "flat-kv-latest" }

func (FlatKVLatestPolicy) Description() string {
	return "Flat actor-scoped KV latest-write overwrite (handlers/memory.go baseline)"
}

func (FlatKVLatestPolicy) Ingest(ctx context.Context, store MemoryStore, turn SessionTurn) (IngestStats, error) {
	scopeID := actorScopeID(turn.TenantID, turn.ActorID)
	key := factKey(turn.Entity, turn.Attribute)

	if turn.Action == ActionDelete || turn.Action == ActionForget {
		_ = store.DeleteMemory(ctx, "actor", scopeID, key)
		_ = store.StoreEvent(ctx, &types.MemoryChangeEvent{
			Type:      "memory_change",
			Scope:     "actor",
			ScopeID:   scopeID,
			Key:       key,
			Action:    "delete",
			Timestamp: turn.Timestamp,
			Metadata:  types.EventMetadata{ActorID: turn.ActorID},
		})
		return IngestStats{KVWrites: 1, EventWrites: 1}, nil
	}

	rec := recordFromTurn(turn)
	raw, err := json.Marshal(rec)
	if err != nil {
		return IngestStats{}, err
	}
	if err := store.SetMemory(ctx, &types.Memory{
		Scope:     "actor",
		ScopeID:   scopeID,
		Key:       key,
		Data:      raw,
		CreatedAt: turn.Timestamp,
		UpdatedAt: turn.Timestamp,
	}); err != nil {
		return IngestStats{}, err
	}
	if err := store.StoreEvent(ctx, &types.MemoryChangeEvent{
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
	return IngestStats{KVWrites: 1, EventWrites: 1}, nil
}

func (FlatKVLatestPolicy) Recall(ctx context.Context, store MemoryStore, query RecallQuery) (KaiDecision, error) {
	scopeID := actorScopeID(query.TenantID, query.ActorID)
	if query.QueryMode == QuerySemantic {
		mems, err := store.ListMemory(ctx, "actor", scopeID)
		if err != nil {
			return KaiDecision{}, err
		}
		emb := NewHashEmbedder(256)
		qVec := emb.Embed(query.Question)
		var bestRec *storedRecord
		var bestKey string
		bestScore := -1.0
		tokens := 0
		for _, m := range mems {
			var r storedRecord
			if err := json.Unmarshal(m.Data, &r); err != nil {
				continue
			}
			if !aclPermitted(r.TenantID, r.ACLRoles, query.TenantID, query.CallerRoles) {
				continue
			}
			tokens += estimateTokens(r.Statement)
			score := cosineSim32(qVec, emb.Embed(r.Statement+" "+r.Entity+" "+r.Attribute))
			if score > bestScore {
				bestScore = score
				cp := r
				bestRec = &cp
				bestKey = m.Key
			}
		}
		if bestRec == nil || bestScore <= 0 {
			return abstainDecision(tokens, 0, 0, nil, nil, nil), nil
		}
		return KaiDecision{
			Status:                 DecisionAnswer,
			Answer:                 bestRec.Value,
			EvidenceKeys:           []string{bestKey},
			EvidenceTenants:        []string{bestRec.TenantID},
			EvidenceScopes:         []string{"actor"},
			Confidence:             bestRec.Confidence,
			EvidenceScore:          bestScore,
			RetrievalTokens:        tokens,
			GenerativeOutputTokens: 0,
		}, nil
	}

	key := factKey(query.Entity, query.Attribute)
	mem, err := store.GetMemory(ctx, "actor", scopeID, key)
	if err != nil || mem == nil {
		return abstainDecision(0, 0, 0, nil, nil, nil), nil
	}
	var rec storedRecord
	if err := json.Unmarshal(mem.Data, &rec); err != nil {
		return abstainDecision(0, 0, 0, nil, nil, nil), nil
	}
	if !aclPermitted(rec.TenantID, rec.ACLRoles, query.TenantID, query.CallerRoles) || rec.Deleted {
		return abstainDecision(0, 0, 0, nil, nil, nil), nil
	}
	tokens := estimateTokens(rec.Statement)
	return KaiDecision{
		Status:                 DecisionAnswer,
		Answer:                 rec.Value,
		OrderedAnswers:         []string{rec.Value},
		AggregatedValues:       []string{rec.Value},
		EvidenceKeys:           []string{key},
		EvidenceTenants:        []string{rec.TenantID},
		EvidenceScopes:         []string{"actor"},
		Confidence:             rec.Confidence,
		EvidenceScore:          1.0,
		RetrievalTokens:        tokens,
		GenerativeOutputTokens: 0,
	}, nil
}

// ============================================================================
// Policy 2: HierarchicalScopeKVPolicy
// Faithfully mirrors control-plane/internal/handlers/memory.go:189 fixed scope
// precedence lookup (workflow -> session -> actor -> global) without timestamp
// reconciliation across scopes.
// ============================================================================

type HierarchicalScopeKVPolicy struct{}

func (HierarchicalScopeKVPolicy) Name() string { return "hierarchical-scope-kv" }

func (HierarchicalScopeKVPolicy) Description() string {
	return "Fixed workflow->session->actor->global precedence KV (handlers/memory.go:189)"
}

func resolveTurnScope(turn SessionTurn) (string, string) {
	scope := strings.ToLower(strings.TrimSpace(turn.Scope))
	switch scope {
	case "workflow":
		wf := turn.WorkflowID
		if wf == "" {
			wf = "default-wf"
		}
		return "workflow", workflowScopeID(turn.TenantID, wf)
	case "session":
		return "session", sessionScopeID(turn.TenantID, turn.SessionID)
	case "global":
		return "global", "global"
	default:
		return "actor", actorScopeID(turn.TenantID, turn.ActorID)
	}
}

func (HierarchicalScopeKVPolicy) Ingest(ctx context.Context, store MemoryStore, turn SessionTurn) (IngestStats, error) {
	scope, scopeID := resolveTurnScope(turn)
	key := factKey(turn.Entity, turn.Attribute)

	if turn.Action == ActionDelete || turn.Action == ActionForget {
		_ = store.DeleteMemory(ctx, scope, scopeID, key)
		_ = store.StoreEvent(ctx, &types.MemoryChangeEvent{
			Type:      "memory_change",
			Scope:     scope,
			ScopeID:   scopeID,
			Key:       key,
			Action:    "delete",
			Timestamp: turn.Timestamp,
			Metadata:  types.EventMetadata{ActorID: turn.ActorID, WorkflowID: turn.WorkflowID},
		})
		return IngestStats{KVWrites: 1, EventWrites: 1}, nil
	}

	rec := recordFromTurn(turn)
	rec.Scope = scope
	raw, err := json.Marshal(rec)
	if err != nil {
		return IngestStats{}, err
	}
	if err := store.SetMemory(ctx, &types.Memory{
		Scope:     scope,
		ScopeID:   scopeID,
		Key:       key,
		Data:      raw,
		CreatedAt: turn.Timestamp,
		UpdatedAt: turn.Timestamp,
	}); err != nil {
		return IngestStats{}, err
	}
	if err := store.StoreEvent(ctx, &types.MemoryChangeEvent{
		Type:      "memory_change",
		Scope:     scope,
		ScopeID:   scopeID,
		Key:       key,
		Action:    "set",
		Data:      raw,
		Timestamp: turn.Timestamp,
		Metadata:  types.EventMetadata{ActorID: turn.ActorID, WorkflowID: turn.WorkflowID},
	}); err != nil {
		return IngestStats{}, err
	}
	return IngestStats{KVWrites: 1, EventWrites: 1}, nil
}

func (HierarchicalScopeKVPolicy) Recall(ctx context.Context, store MemoryStore, query RecallQuery) (KaiDecision, error) {
	key := factKey(query.Entity, query.Attribute)
	type scopeProbe struct {
		scope   string
		scopeID string
	}
	var probes []scopeProbe
	if query.WorkflowID != "" {
		probes = append(probes, scopeProbe{"workflow", workflowScopeID(query.TenantID, query.WorkflowID)})
	}
	for _, sid := range query.SessionIDs {
		probes = append(probes, scopeProbe{"session", sessionScopeID(query.TenantID, sid)})
	}
	if query.ActorID != "" {
		probes = append(probes, scopeProbe{"actor", actorScopeID(query.TenantID, query.ActorID)})
	}
	probes = append(probes, scopeProbe{"global", "global"})

	for _, p := range probes {
		mem, err := store.GetMemory(ctx, p.scope, p.scopeID, key)
		if err != nil || mem == nil {
			continue
		}
		var rec storedRecord
		if err := json.Unmarshal(mem.Data, &rec); err != nil {
			continue
		}
		if !aclPermitted(rec.TenantID, rec.ACLRoles, query.TenantID, query.CallerRoles) || rec.Deleted {
			continue
		}
		tokens := estimateTokens(rec.Statement)
		return KaiDecision{
			Status:                 DecisionAnswer,
			Answer:                 rec.Value,
			OrderedAnswers:         []string{rec.Value},
			AggregatedValues:       []string{rec.Value},
			EvidenceKeys:           []string{key},
			EvidenceTenants:        []string{rec.TenantID},
			EvidenceScopes:         []string{p.scope},
			Confidence:             rec.Confidence,
			EvidenceScore:          1.0,
			RetrievalTokens:        tokens,
			GenerativeOutputTokens: 0,
		}, nil
	}
	return abstainDecision(0, 0, 0, nil, nil, nil), nil
}

// ============================================================================
// Policy 3: AppendOnlyHistoryPolicy
// Adapts membench.BaselineAppendPolicy & #65 raw immutable session event append
// without reconciliation, deletion tombstones, or cross-session consolidation.
// ============================================================================

type AppendOnlyHistoryPolicy struct{}

func (AppendOnlyHistoryPolicy) Name() string { return "append-only-history" }

func (AppendOnlyHistoryPolicy) Description() string {
	return "Immutable session-scoped append-only log without reconciliation (membench baseline)"
}

func (AppendOnlyHistoryPolicy) Ingest(ctx context.Context, store MemoryStore, turn SessionTurn) (IngestStats, error) {
	scopeID := sessionScopeID(turn.TenantID, turn.SessionID)
	rec := recordFromTurn(turn)
	raw, err := json.Marshal(rec)
	if err != nil {
		return IngestStats{}, err
	}
	key := fmt.Sprintf("%020d:%s", turn.Timestamp.UTC().UnixNano(), turn.TurnID)
	if err := store.SetMemory(ctx, &types.Memory{
		Scope:     "session",
		ScopeID:   scopeID,
		Key:       key,
		Data:      raw,
		CreatedAt: turn.Timestamp,
		UpdatedAt: turn.Timestamp,
	}); err != nil {
		return IngestStats{}, err
	}
	if err := store.StoreEvent(ctx, &types.MemoryChangeEvent{
		Type:      "memory_change",
		Scope:     "session",
		ScopeID:   scopeID,
		Key:       key,
		Action:    "set",
		Data:      raw,
		Timestamp: turn.Timestamp,
		Metadata:  types.EventMetadata{ActorID: turn.ActorID},
	}); err != nil {
		return IngestStats{}, err
	}
	return IngestStats{KVWrites: 1, EventWrites: 1}, nil
}

func (AppendOnlyHistoryPolicy) Recall(ctx context.Context, store MemoryStore, query RecallQuery) (KaiDecision, error) {
	var candidates []storedRecord
	var keys, tenants, scopes []string
	tokens := 0
	for _, sid := range query.SessionIDs {
		scopeID := sessionScopeID(query.TenantID, sid)
		mems, err := store.ListMemory(ctx, "session", scopeID)
		if err != nil {
			return KaiDecision{}, err
		}
		for _, m := range mems {
			var rec storedRecord
			if err := json.Unmarshal(m.Data, &rec); err != nil {
				continue
			}
			if !aclPermitted(rec.TenantID, rec.ACLRoles, query.TenantID, query.CallerRoles) {
				continue
			}
			tokens += estimateTokens(rec.Statement)
			if strings.EqualFold(rec.Entity, query.Entity) && strings.EqualFold(rec.Attribute, query.Attribute) && rec.Value != "" {
				candidates = append(candidates, rec)
				keys = append(keys, m.Key)
				tenants = append(tenants, rec.TenantID)
				scopes = append(scopes, "session")
			}
		}
	}
	if len(candidates) == 0 {
		// Append-only baseline fails open rather than emitting calibrated abstention.
		return KaiDecision{
			Status:                 DecisionAnswer,
			Answer:                 "unknown",
			Abstained:              false,
			RetrievalTokens:        tokens,
			GenerativeOutputTokens: 0,
		}, nil
	}
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].Timestamp.Equal(candidates[j].Timestamp) {
			return candidates[i].TurnID < candidates[j].TurnID
		}
		return candidates[i].Timestamp.Before(candidates[j].Timestamp)
	})
	ordered := make([]string, 0, len(candidates))
	for _, c := range candidates {
		ordered = appendUniqueStr(ordered, c.Value)
	}
	// Baseline returns earliest record for scalar queries (fails on updates & deletions).
	return KaiDecision{
		Status:                 DecisionAnswer,
		Answer:                 candidates[0].Value,
		OrderedAnswers:         ordered,
		AggregatedValues:       ordered,
		EvidenceKeys:           keys,
		EvidenceTenants:        tenants,
		EvidenceScopes:         scopes,
		Confidence:             candidates[0].Confidence,
		EvidenceScore:          1.0,
		RetrievalTokens:        tokens,
		GenerativeOutputTokens: 0,
	}, nil
}

// ============================================================================
// Policy 4: VectorLexicalRecencyPolicy
// Adapts control-plane/internal/handlers/vector_memory.go (SetVector +
// SimilaritySearch) with recency-weighted re-ranking. By default MinScore=0
// matches current VectorSearchRequest (unthresholded top-K).
// ============================================================================

type VectorLexicalRecencyPolicy struct {
	Embedder          Embedder
	TopK              int
	RecencyWeight     float64
	MinScore          float64
	CoordinatedDelete bool
	CustomName        string
}

func (p VectorLexicalRecencyPolicy) Name() string {
	if p.CustomName != "" {
		return p.CustomName
	}
	return "vector-lexical-recency"
}

func (p VectorLexicalRecencyPolicy) Description() string {
	return "Canonical vector SimilaritySearch with recency-weighted re-ranking (handlers/vector_memory.go)"
}

func (p VectorLexicalRecencyPolicy) embedder() Embedder {
	if p.Embedder != nil {
		return p.Embedder
	}
	return NewHashEmbedder(256)
}

func (p VectorLexicalRecencyPolicy) topK() int {
	if p.TopK <= 0 {
		return 8
	}
	return p.TopK
}

func (p VectorLexicalRecencyPolicy) recencyWeight() float64 {
	if p.RecencyWeight < 0 {
		return 0
	}
	if p.RecencyWeight == 0 {
		return 0.25
	}
	return p.RecencyWeight
}

func (p VectorLexicalRecencyPolicy) Ingest(ctx context.Context, store MemoryStore, turn SessionTurn) (IngestStats, error) {
	if turn.Action == ActionNoop {
		return IngestStats{SkippedNoops: 1}, nil
	}
	scopeID := actorScopeID(turn.TenantID, turn.ActorID)
	prefix := factKey(turn.Entity, turn.Attribute) + ":"

	if turn.Action == ActionDelete || turn.Action == ActionForget {
		if p.CoordinatedDelete {
			_, _ = store.DeleteVectorsByPrefix(ctx, "actor", scopeID, prefix)
			return IngestStats{VectorWrites: 1}, nil
		}
		return IngestStats{SkippedNoops: 1}, nil
	}

	rec := recordFromTurn(turn)
	vec := p.embedder().Embed(embedTextForTurn(turn))
	key := fmt.Sprintf("%s%020d:%s", prefix, turn.Timestamp.UTC().UnixNano(), turn.TurnID)
	meta := map[string]interface{}{
		"turn_id":      rec.TurnID,
		"tenant_id":    rec.TenantID,
		"actor_id":     rec.ActorID,
		"session_id":   rec.SessionID,
		"entity":       rec.Entity,
		"attribute":    rec.Attribute,
		"value":        rec.Value,
		"statement":    rec.Statement,
		"kind":         string(rec.Kind),
		"confidence":   rec.Confidence,
		"timestamp_ns": rec.Timestamp.UnixNano(),
		"acl_roles":    strings.Join(rec.ACLRoles, ","),
	}
	if err := store.SetVector(ctx, &types.VectorRecord{
		Scope:     "actor",
		ScopeID:   scopeID,
		Key:       key,
		Embedding: vec,
		Metadata:  meta,
		CreatedAt: turn.Timestamp,
		UpdatedAt: turn.Timestamp,
	}); err != nil {
		return IngestStats{}, err
	}
	return IngestStats{VectorWrites: 1}, nil
}

func vectorMetaRoles(meta map[string]interface{}) []string {
	raw, _ := meta["acl_roles"].(string)
	if strings.TrimSpace(raw) == "" {
		return nil
	}
	parts := strings.Split(raw, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if t := strings.TrimSpace(p); t != "" {
			out = append(out, t)
		}
	}
	return out
}

func vectorMetaRecord(meta map[string]interface{}) storedRecord {
	val, _ := meta["value"].(string)
	stmt, _ := meta["statement"].(string)
	tenant, _ := meta["tenant_id"].(string)
	actor, _ := meta["actor_id"].(string)
	session, _ := meta["session_id"].(string)
	entity, _ := meta["entity"].(string)
	attr, _ := meta["attribute"].(string)
	turnID, _ := meta["turn_id"].(string)
	kindStr, _ := meta["kind"].(string)
	conf, _ := meta["confidence"].(float64)
	var ts time.Time
	if ns, ok := meta["timestamp_ns"].(int64); ok {
		ts = time.Unix(0, ns).UTC()
	} else if fns, ok := meta["timestamp_ns"].(float64); ok {
		ts = time.Unix(0, int64(fns)).UTC()
	}
	return storedRecord{
		TurnID:     turnID,
		TenantID:   tenant,
		ActorID:    actor,
		SessionID:  session,
		ACLRoles:   vectorMetaRoles(meta),
		Entity:     entity,
		Attribute:  attr,
		Value:      val,
		Statement:  stmt,
		Kind:       EpistemicKind(kindStr),
		Confidence: conf,
		Timestamp:  ts,
	}
}

func (p VectorLexicalRecencyPolicy) Recall(ctx context.Context, store MemoryStore, query RecallQuery) (KaiDecision, error) {
	scopeID := actorScopeID(query.TenantID, query.ActorID)
	qVec := p.embedder().Embed(embedTextForQuery(query))

	hits, err := store.SimilaritySearch(ctx, "actor", scopeID, qVec, p.topK(), nil)
	if err != nil {
		return KaiDecision{}, err
	}

	type scoredHit struct {
		rec   storedRecord
		key   string
		score float64
	}
	var valid []scoredHit
	tokens := 0
	var minTS, maxTS int64
	for _, h := range hits {
		rec := vectorMetaRecord(h.Metadata)
		if !aclPermitted(rec.TenantID, rec.ACLRoles, query.TenantID, query.CallerRoles) {
			continue
		}
		tokens += estimateTokens(rec.Statement)
		if h.Score < p.MinScore {
			continue
		}
		ns := rec.Timestamp.UnixNano()
		if len(valid) == 0 || ns < minTS {
			minTS = ns
		}
		if len(valid) == 0 || ns > maxTS {
			maxTS = ns
		}
		valid = append(valid, scoredHit{rec: rec, key: h.Key, score: h.Score})
	}

	if len(valid) == 0 {
		return abstainDecision(tokens, 0, 0, nil, nil, nil), nil
	}

	wRec := p.recencyWeight()
	span := float64(maxTS - minTS)
	sort.Slice(valid, func(i, j int) bool {
		recI := 1.0
		recJ := 1.0
		if span > 0 {
			recI = float64(valid[i].rec.Timestamp.UnixNano()-minTS) / span
			recJ = float64(valid[j].rec.Timestamp.UnixNano()-minTS) / span
		}
		combinedI := (1.0-wRec)*valid[i].score + wRec*recI
		combinedJ := (1.0-wRec)*valid[j].score + wRec*recJ
		if combinedI == combinedJ {
			return valid[i].key < valid[j].key
		}
		return combinedI > combinedJ
	})

	top := valid[0]
	ordered := make([]string, 0, len(valid))
	keys := make([]string, 0, len(valid))
	tenants := make([]string, 0, len(valid))
	scopes := make([]string, 0, len(valid))
	for _, v := range valid {
		ordered = appendUniqueStr(ordered, v.rec.Value)
		keys = append(keys, v.key)
		tenants = append(tenants, v.rec.TenantID)
		scopes = append(scopes, "actor")
	}

	return KaiDecision{
		Status:                 DecisionAnswer,
		Answer:                 top.rec.Value,
		OrderedAnswers:         ordered,
		AggregatedValues:       ordered,
		EvidenceKeys:           keys,
		EvidenceTenants:        tenants,
		EvidenceScopes:         scopes,
		Confidence:             top.rec.Confidence,
		EvidenceScore:          top.score,
		RetrievalTokens:        tokens,
		GenerativeOutputTokens: 0,
	}, nil
}

// ============================================================================
// Policy 5: AtomicReflectivePolicy
// Adapts #62 (atomic_facts deduplication/selective writes), #63 (tiered
// episodic timeline + semantic beliefs), and #64 (reflective epistemic belief
// separation: OBSERVATION > CONCLUSION > HYPOTHESIS with contradiction flags).
// Intentionally reads latest semantic belief on scalar queries (demonstrating
// recency bias on Negative Regime 1 AsOf queries without time-travel routing).
// ============================================================================

type AtomicReflectivePolicy struct {
	MinConfidence float64
	Embedder      Embedder
}

func (p AtomicReflectivePolicy) Name() string { return "atomic-reflective" }

func (p AtomicReflectivePolicy) Description() string {
	return "Selective atomic writes (#62) + tiered timeline (#63) + epistemic belief separation (#64)"
}

func (p AtomicReflectivePolicy) minConf() float64 {
	if p.MinConfidence <= 0 {
		return 0.55
	}
	return p.MinConfidence
}

func (p AtomicReflectivePolicy) embedder() Embedder {
	if p.Embedder != nil {
		return p.Embedder
	}
	return NewHashEmbedder(256)
}

func (p AtomicReflectivePolicy) Ingest(ctx context.Context, store MemoryStore, turn SessionTurn) (IngestStats, error) {
	if turn.Action == ActionNoop {
		return IngestStats{SkippedNoops: 1}, nil
	}
	scopeID := actorScopeID(turn.TenantID, turn.ActorID)
	fk := factKey(turn.Entity, turn.Attribute)
	beliefKey := "belief:" + fk
	conflictKey := "conflict:" + fk
	rec := recordFromTurn(turn)

	if turn.Action == ActionDelete || turn.Action == ActionForget {
		rec.Deleted = true
		raw, err := json.Marshal(rec)
		if err != nil {
			return IngestStats{}, err
		}
		if err := store.SetMemory(ctx, &types.Memory{
			Scope:     "actor",
			ScopeID:   scopeID,
			Key:       beliefKey,
			Data:      raw,
			CreatedAt: turn.Timestamp,
			UpdatedAt: turn.Timestamp,
		}); err != nil {
			return IngestStats{}, err
		}
		tlKey := fmt.Sprintf("timeline:%s:%020d:%s", fk, turn.Timestamp.UTC().UnixNano(), turn.TurnID)
		_ = store.SetMemory(ctx, &types.Memory{
			Scope:     "actor",
			ScopeID:   scopeID,
			Key:       tlKey,
			Data:      raw,
			CreatedAt: turn.Timestamp,
			UpdatedAt: turn.Timestamp,
		})
		_ = store.DeleteVector(ctx, "actor", scopeID, beliefKey)
		_ = store.StoreEvent(ctx, &types.MemoryChangeEvent{
			Type:      "memory_change",
			Scope:     "actor",
			ScopeID:   scopeID,
			Key:       beliefKey,
			Action:    "delete",
			Data:      raw,
			Timestamp: turn.Timestamp,
			Metadata:  types.EventMetadata{ActorID: turn.ActorID},
		})
		return IngestStats{KVWrites: 2, VectorWrites: 1, EventWrites: 1}, nil
	}

	var prev *storedRecord
	if existing, err := store.GetMemory(ctx, "actor", scopeID, beliefKey); err == nil && existing != nil {
		var tmp storedRecord
		if json.Unmarshal(existing.Data, &tmp) == nil && tmp.TenantID == turn.TenantID {
			prev = &tmp
		}
	}

	// Selective deduplication (#62): skip redundant identical belief writes.
	if prev != nil && !prev.Deleted &&
		normalizeAnswer(prev.Value) == normalizeAnswer(rec.Value) &&
		prev.Kind == rec.Kind && turn.Action != ActionAdd {
		return IngestStats{SkippedNoops: 1}, nil
	}

	var stats IngestStats
	shouldReplace := prev == nil || prev.Deleted
	contradiction := false
	var contestedVal string

	if !shouldReplace {
		prevRank := epistemicRank(prev.Kind)
		newRank := epistemicRank(rec.Kind)
		if normalizeAnswer(prev.Value) != normalizeAnswer(rec.Value) {
			// Conflict detected if hypothesis contradicts observation, or competing sources at same/newer turn.
			if newRank < prevRank {
				contradiction = true
				contestedVal = rec.Value
				shouldReplace = false
			} else if newRank > prevRank {
				contradiction = true
				contestedVal = prev.Value
				shouldReplace = true
			} else if !rec.Timestamp.Before(prev.Timestamp) {
				if turn.Action == ActionReflect || (prev.SourceID != "" && rec.SourceID != "" && prev.SourceID != rec.SourceID) {
					contradiction = true
					contestedVal = prev.Value
				}
				shouldReplace = true
			} else {
				// Out-of-order older observation arriving late: preserve newer authoritative belief.
				shouldReplace = false
			}
		} else if !rec.Timestamp.Before(prev.Timestamp) && newRank >= prevRank {
			shouldReplace = true
		}
	}

	raw, err := json.Marshal(rec)
	if err != nil {
		return stats, err
	}

	if contradiction && contestedVal != "" {
		var clog contradictionLog
		if cm, err := store.GetMemory(ctx, "actor", scopeID, conflictKey); err == nil && cm != nil {
			_ = json.Unmarshal(cm.Data, &clog)
		}
		clog.Values = appendUniqueStr(clog.Values, contestedVal)
		if cRaw, err := json.Marshal(clog); err == nil {
			_ = store.SetMemory(ctx, &types.Memory{
				Scope:     "actor",
				ScopeID:   scopeID,
				Key:       conflictKey,
				Data:      cRaw,
				CreatedAt: turn.Timestamp,
				UpdatedAt: turn.Timestamp,
			})
			stats.KVWrites++
		}
	}

	if shouldReplace {
		if err := store.SetMemory(ctx, &types.Memory{
			Scope:     "actor",
			ScopeID:   scopeID,
			Key:       beliefKey,
			Data:      raw,
			CreatedAt: turn.Timestamp,
			UpdatedAt: turn.Timestamp,
		}); err != nil {
			return stats, err
		}
		stats.KVWrites++

		// Keep vector index synchronized with active semantic belief.
		if rec.Kind == KindObservation || rec.Kind == KindConclusion {
			_ = store.SetVector(ctx, &types.VectorRecord{
				Scope:     "actor",
				ScopeID:   scopeID,
				Key:       beliefKey,
				Embedding: p.embedder().Embed(embedTextForTurn(turn)),
				Metadata: map[string]interface{}{
					"turn_id":      rec.TurnID,
					"tenant_id":    rec.TenantID,
					"actor_id":     rec.ActorID,
					"session_id":   rec.SessionID,
					"entity":       rec.Entity,
					"attribute":    rec.Attribute,
					"value":        rec.Value,
					"statement":    rec.Statement,
					"kind":         string(rec.Kind),
					"confidence":   rec.Confidence,
					"timestamp_ns": rec.Timestamp.UnixNano(),
					"acl_roles":    strings.Join(rec.ACLRoles, ","),
				},
				CreatedAt: turn.Timestamp,
				UpdatedAt: turn.Timestamp,
			})
			stats.VectorWrites++
		}
	}

	// Immutable episodic timeline tier (#63) for direct observations.
	if rec.Kind == KindObservation {
		tlKey := fmt.Sprintf("timeline:%s:%020d:%s", fk, turn.Timestamp.UTC().UnixNano(), turn.TurnID)
		if err := store.SetMemory(ctx, &types.Memory{
			Scope:     "actor",
			ScopeID:   scopeID,
			Key:       tlKey,
			Data:      raw,
			CreatedAt: turn.Timestamp,
			UpdatedAt: turn.Timestamp,
		}); err != nil {
			return stats, err
		}
		stats.KVWrites++
	}

	if err := store.StoreEvent(ctx, &types.MemoryChangeEvent{
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

func loadTimelineEntries(ctx context.Context, store MemoryStore, scopeID, fk string, query RecallQuery) ([]storedRecord, []string, []string, []string, int, error) {
	mems, err := store.ListMemory(ctx, "actor", scopeID)
	if err != nil {
		return nil, nil, nil, nil, 0, err
	}
	prefix := "timeline:" + fk + ":"
	var entries []storedRecord
	var keys, tenants, scopes []string
	tokens := 0
	for _, m := range mems {
		if !strings.HasPrefix(m.Key, prefix) {
			continue
		}
		var r storedRecord
		if err := json.Unmarshal(m.Data, &r); err != nil {
			continue
		}
		if !aclPermitted(r.TenantID, r.ACLRoles, query.TenantID, query.CallerRoles) {
			continue
		}
		entries = append(entries, r)
		keys = append(keys, m.Key)
		tenants = append(tenants, r.TenantID)
		scopes = append(scopes, "actor")
		tokens += estimateTokens(r.Statement)
	}
	return entries, keys, tenants, scopes, tokens, nil
}

func (p AtomicReflectivePolicy) Recall(ctx context.Context, store MemoryStore, query RecallQuery) (KaiDecision, error) {
	scopeID := actorScopeID(query.TenantID, query.ActorID)
	fk := factKey(query.Entity, query.Attribute)

	// Timeline-backed modes (excluding TemporalAsOf: demonstrates recency bias
	// when a policy lacks explicit point-in-time historical snapshot routing).
	switch query.TemporalMode {
	case TemporalSequence, TemporalFirst, TemporalLast, TemporalDuration, TemporalWindow, TemporalAggregate:
		entries, keys, tenants, scopes, tokens, err := loadTimelineEntries(ctx, store, scopeID, fk, query)
		if err != nil {
			return KaiDecision{}, err
		}
		return resolveTimelineMode(entries, query, tokens, keys, tenants, scopes), nil
	}

	if query.QueryMode == QuerySemantic {
		qVec := p.embedder().Embed(query.Question)
		hits, err := store.SimilaritySearch(ctx, "actor", scopeID, qVec, 5, nil)
		if err != nil {
			return KaiDecision{}, err
		}
		tokens := 0
		for _, h := range hits {
			rec := vectorMetaRecord(h.Metadata)
			if !aclPermitted(rec.TenantID, rec.ACLRoles, query.TenantID, query.CallerRoles) {
				continue
			}
			tokens += estimateTokens(rec.Statement)
			if h.Score >= 0.15 {
				return KaiDecision{
					Status:                 DecisionAnswer,
					Answer:                 rec.Value,
					EvidenceKeys:           []string{h.Key},
					EvidenceTenants:        []string{rec.TenantID},
					EvidenceScopes:         []string{"actor"},
					Confidence:             rec.Confidence,
					EvidenceScore:          h.Score,
					RetrievalTokens:        tokens,
					GenerativeOutputTokens: 0,
				}, nil
			}
		}
		return abstainDecision(tokens, 0, 0, nil, nil, nil), nil
	}

	beliefKey := "belief:" + fk
	mem, err := store.GetMemory(ctx, "actor", scopeID, beliefKey)
	if err != nil || mem == nil {
		return abstainDecision(0, 0, 0, nil, nil, nil), nil
	}
	var rec storedRecord
	if err := json.Unmarshal(mem.Data, &rec); err != nil {
		return abstainDecision(0, 0, 0, nil, nil, nil), nil
	}
	if !aclPermitted(rec.TenantID, rec.ACLRoles, query.TenantID, query.CallerRoles) || rec.Deleted {
		return abstainDecision(0, 0, 0, nil, nil, nil), nil
	}

	tokens := estimateTokens(rec.Statement)
	if rec.Confidence > 0 && rec.Confidence < p.minConf() {
		return abstainDecision(tokens, 1.0, rec.Confidence, []string{beliefKey}, []string{rec.TenantID}, []string{"actor"}), nil
	}

	var clog contradictionLog
	if cm, err := store.GetMemory(ctx, "actor", scopeID, "conflict:"+fk); err == nil && cm != nil {
		_ = json.Unmarshal(cm.Data, &clog)
	}
	status := DecisionAnswer
	if len(clog.Values) > 0 {
		status = DecisionContested
	}

	return KaiDecision{
		Status:                 status,
		Answer:                 rec.Value,
		OrderedAnswers:         []string{rec.Value},
		AggregatedValues:       []string{rec.Value},
		ContradictionDetected:  len(clog.Values) > 0,
		ContradictingValues:    clog.Values,
		EvidenceKeys:           []string{beliefKey},
		EvidenceTenants:        []string{rec.TenantID},
		EvidenceScopes:         []string{"actor"},
		Confidence:             rec.Confidence,
		EvidenceScore:          1.0,
		RetrievalTokens:        tokens,
		GenerativeOutputTokens: 0,
	}, nil
}

// ============================================================================
// Policy 6: KaiEvidenceGatedPolicy
// Adapts #66 (learned_memory_policy ADD/UPDATE/DELETE/FORGET/NOOP) + #74 Kai
// non-generative evidence gate over #62-#65 canonical MemoryStore primitives.
// Supports calibrated semantic abstention threshold (MinEvidenceScore),
// epistemic contradiction tracking, pre-hit ACL isolation, multi-session
// aggregation, and explicit point-in-time AsOf snapshot queries.
// ============================================================================

type KaiEvidenceGatedPolicy struct {
	MinConfidence         float64
	MinEvidenceScore      float64
	TopK                  int
	DisableHistoricalAsOf bool // Ablation flag to verify Negative Regime 1 root cause
	Embedder              Embedder
	CustomName            string
}

func (p KaiEvidenceGatedPolicy) Name() string {
	if p.CustomName != "" {
		return p.CustomName
	}
	return "kai-evidence-gated"
}

func (p KaiEvidenceGatedPolicy) Description() string {
	return "Kai non-generative evidence gate (#66/#74) over tiered reflective MemoryStore (#62-#65)"
}

func (p KaiEvidenceGatedPolicy) minConf() float64 {
	if p.MinConfidence <= 0 {
		return 0.55
	}
	return p.MinConfidence
}

func (p KaiEvidenceGatedPolicy) minEvidenceScore() float64 {
	if p.MinEvidenceScore <= 0 {
		return 0.35
	}
	return p.MinEvidenceScore
}

func (p KaiEvidenceGatedPolicy) topK() int {
	if p.TopK <= 0 {
		return 8
	}
	return p.TopK
}

func (p KaiEvidenceGatedPolicy) embedder() Embedder {
	if p.Embedder != nil {
		return p.Embedder
	}
	return NewHashEmbedder(256)
}

func (p KaiEvidenceGatedPolicy) Ingest(ctx context.Context, store MemoryStore, turn SessionTurn) (IngestStats, error) {
	// Reuse canonical #62/#63/#64 tiered + belief + vector ingestion, adding
	// coordinated vector pruning on delete/forget and low-confidence filtering (#66).
	base := AtomicReflectivePolicy{
		MinConfidence: p.minConf(),
		Embedder:      p.embedder(),
	}
	return base.Ingest(ctx, store, turn)
}

func (p KaiEvidenceGatedPolicy) Recall(ctx context.Context, store MemoryStore, query RecallQuery) (KaiDecision, error) {
	scopeID := actorScopeID(query.TenantID, query.ActorID)
	fk := factKey(query.Entity, query.Attribute)

	// 1. Temporal & multi-session timeline modes (including point-in-time AsOf).
	switch query.TemporalMode {
	case TemporalSequence, TemporalFirst, TemporalLast, TemporalDuration, TemporalWindow, TemporalAggregate:
		entries, keys, tenants, scopes, tokens, err := loadTimelineEntries(ctx, store, scopeID, fk, query)
		if err != nil {
			return KaiDecision{}, err
		}
		return resolveTimelineMode(entries, query, tokens, keys, tenants, scopes), nil
	case TemporalAsOf:
		if !p.DisableHistoricalAsOf {
			entries, keys, tenants, scopes, tokens, err := loadTimelineEntries(ctx, store, scopeID, fk, query)
			if err != nil {
				return KaiDecision{}, err
			}
			return resolveTimelineMode(entries, query, tokens, keys, tenants, scopes), nil
		}
	}

	// 2. Semantic-only mode: route through Kai evidence-score abstention gate.
	if query.QueryMode == QuerySemantic {
		qVec := p.embedder().Embed(query.Question)
		hits, err := store.SimilaritySearch(ctx, "actor", scopeID, qVec, p.topK(), nil)
		if err != nil {
			return KaiDecision{}, err
		}
		tokens := 0
		var bestRec *storedRecord
		var bestKey string
		bestScore := 0.0
		for _, h := range hits {
			rec := vectorMetaRecord(h.Metadata)
			if !aclPermitted(rec.TenantID, rec.ACLRoles, query.TenantID, query.CallerRoles) {
				continue
			}
			tokens += estimateTokens(rec.Statement)
			if bestRec == nil || h.Score > bestScore {
				cp := rec
				bestRec = &cp
				bestKey = h.Key
				bestScore = h.Score
			}
		}
		if bestRec == nil || bestScore < p.minEvidenceScore() || (bestRec.Confidence > 0 && bestRec.Confidence < p.minConf()) {
			return abstainDecision(tokens, bestScore, 0, nil, nil, nil), nil
		}
		return KaiDecision{
			Status:                 DecisionAnswer,
			Answer:                 bestRec.Value,
			EvidenceKeys:           []string{bestKey},
			EvidenceTenants:        []string{bestRec.TenantID},
			EvidenceScopes:         []string{"actor"},
			Confidence:             bestRec.Confidence,
			EvidenceScore:          bestScore,
			RetrievalTokens:        tokens,
			GenerativeOutputTokens: 0,
		}, nil
	}

	// 3. Structured belief lookup with epistemic contradiction surfacing and Kai confidence gate.
	beliefKey := "belief:" + fk
	mem, err := store.GetMemory(ctx, "actor", scopeID, beliefKey)
	if err != nil || mem == nil {
		return abstainDecision(0, 0, 0, nil, nil, nil), nil
	}
	var rec storedRecord
	if err := json.Unmarshal(mem.Data, &rec); err != nil {
		return abstainDecision(0, 0, 0, nil, nil, nil), nil
	}
	if !aclPermitted(rec.TenantID, rec.ACLRoles, query.TenantID, query.CallerRoles) || rec.Deleted {
		return abstainDecision(0, 0, 0, nil, nil, nil), nil
	}

	tokens := estimateTokens(rec.Statement)
	if rec.Confidence > 0 && rec.Confidence < p.minConf() {
		return abstainDecision(tokens, 1.0, rec.Confidence, []string{beliefKey}, []string{rec.TenantID}, []string{"actor"}), nil
	}

	var clog contradictionLog
	if cm, err := store.GetMemory(ctx, "actor", scopeID, "conflict:"+fk); err == nil && cm != nil {
		_ = json.Unmarshal(cm.Data, &clog)
	}
	status := DecisionAnswer
	if len(clog.Values) > 0 {
		status = DecisionContested
	}

	return KaiDecision{
		Status:                 status,
		Answer:                 rec.Value,
		OrderedAnswers:         []string{rec.Value},
		AggregatedValues:       []string{rec.Value},
		ContradictionDetected:  len(clog.Values) > 0,
		ContradictingValues:    clog.Values,
		EvidenceKeys:           []string{beliefKey},
		EvidenceTenants:        []string{rec.TenantID},
		EvidenceScopes:         []string{"actor"},
		Confidence:             rec.Confidence,
		EvidenceScore:          1.0,
		RetrievalTokens:        tokens,
		GenerativeOutputTokens: 0,
	}, nil
}

// DefaultPolicyMatrix returns the canonical matched set of 6 existing Hanzo
// Agents memory policy adapters evaluated in HIP-1346 LongMemEval.
func DefaultPolicyMatrix() []MemoryPolicy {
	emb := NewHashEmbedder(256)
	return []MemoryPolicy{
		FlatKVLatestPolicy{},
		HierarchicalScopeKVPolicy{},
		AppendOnlyHistoryPolicy{},
		VectorLexicalRecencyPolicy{Embedder: emb, TopK: 8, RecencyWeight: 0.25, MinScore: 0.0, CoordinatedDelete: false},
		AtomicReflectivePolicy{MinConfidence: 0.55, Embedder: emb},
		KaiEvidenceGatedPolicy{MinConfidence: 0.55, MinEvidenceScore: 0.35, TopK: 8, Embedder: emb},
	}
}
