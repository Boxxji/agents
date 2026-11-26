package replay

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"time"

	"github.com/hanzoai/agents/control-plane/pkg/types"
)

// EventType enumerates deterministic append-only workflow history event types.
type EventType string

const (
	EventWorkflowStarted   EventType = "WorkflowStarted"
	EventActivityScheduled EventType = "ActivityScheduled"
	EventActivityCompleted EventType = "ActivityCompleted"
	EventActivityFailed    EventType = "ActivityFailed"
	EventTimerFired        EventType = "TimerFired"
	EventSignalReceived    EventType = "SignalReceived"
	EventCheckpointSaved   EventType = "CheckpointSaved"
	EventWorkflowSuspended EventType = "WorkflowSuspended"
	EventWorkflowCompleted EventType = "WorkflowCompleted"
	EventWorkflowFailed    EventType = "WorkflowFailed"
)

// WorkflowStatus represents the persisted lifecycle state of a durable workflow.
type WorkflowStatus string

const (
	StatusPending   WorkflowStatus = "pending"
	StatusRunning   WorkflowStatus = "running"
	StatusSuspended WorkflowStatus = "suspended"
	StatusCompleted WorkflowStatus = "completed"
	StatusFailed    WorkflowStatus = "failed"
)

// GenesisDigest is the initial PrevDigest for sequence 1.
const GenesisDigest = "0000000000000000000000000000000000000000000000000000000000000000"

// HistoryEvent is a single immutable, cryptographically chained record in a
// workflow's append-only event log.
type HistoryEvent struct {
	Sequence       int64           `json:"sequence"`
	Type           EventType       `json:"type"`
	TenantID       string          `json:"tenant_id"`
	WorkflowID     string          `json:"workflow_id"`
	ActivityName   string          `json:"activity_name,omitempty"`
	IdempotencyKey string          `json:"idempotency_key,omitempty"`
	TimerID        string          `json:"timer_id,omitempty"`
	SignalName     string          `json:"signal_name,omitempty"`
	Label          string          `json:"label,omitempty"`
	ScheduledSeq   int64           `json:"scheduled_seq,omitempty"`
	InputDigest    string          `json:"input_digest,omitempty"`
	Payload        json.RawMessage `json:"payload,omitempty"`
	PayloadURI     string          `json:"payload_uri,omitempty"`
	Error          string          `json:"error,omitempty"`
	LogicalTimeNS  int64           `json:"logical_time_ns"`
	RecordedAt     time.Time       `json:"recorded_at"`
	PrevDigest     string          `json:"prev_digest"`
	Digest         string          `json:"digest"`
}

func (e HistoryEvent) clone() HistoryEvent {
	cp := e
	if len(e.Payload) > 0 {
		cp.Payload = make(json.RawMessage, len(e.Payload))
		copy(cp.Payload, e.Payload)
	}
	return cp
}

// ComputeEventDigest computes the canonical SHA-256 digest chaining this event
// to prevDigest. Wall-clock RecordedAt is excluded so deterministic replays
// produce identical history digests regardless of execution host clock.
func ComputeEventDigest(e HistoryEvent, prevDigest string) string {
	if prevDigest == "" {
		prevDigest = GenesisDigest
	}
	h := sha256.New()
	writeLenPrefixed(h, "hanzo.replay.event.v1")
	writeLenPrefixed(h, prevDigest)
	writeLenPrefixed(h, fmt.Sprintf("%d", e.Sequence))
	writeLenPrefixed(h, string(e.Type))
	writeLenPrefixed(h, e.TenantID)
	writeLenPrefixed(h, e.WorkflowID)
	writeLenPrefixed(h, e.ActivityName)
	writeLenPrefixed(h, e.IdempotencyKey)
	writeLenPrefixed(h, e.TimerID)
	writeLenPrefixed(h, e.SignalName)
	writeLenPrefixed(h, e.Label)
	writeLenPrefixed(h, fmt.Sprintf("%d", e.ScheduledSeq))
	writeLenPrefixed(h, e.InputDigest)
	writeLenPrefixed(h, digestBytes(e.Payload))
	writeLenPrefixed(h, e.PayloadURI)
	writeLenPrefixed(h, e.Error)
	writeLenPrefixed(h, fmt.Sprintf("%d", e.LogicalTimeNS))
	return hex.EncodeToString(h.Sum(nil))
}

// SealEvent sets PrevDigest and computes Digest on e.
func SealEvent(e *HistoryEvent, prevDigest string) {
	if prevDigest == "" {
		prevDigest = GenesisDigest
	}
	e.PrevDigest = prevDigest
	e.Digest = ComputeEventDigest(*e, prevDigest)
}

// ComputeHistoryDigest computes the terminal canonical SHA-256 digest over an
// ordered event slice without mutating the events.
func ComputeHistoryDigest(events []HistoryEvent) string {
	digest := GenesisDigest
	for _, ev := range events {
		digest = ComputeEventDigest(ev, digest)
	}
	return digest
}

// VerifyHistoryIntegrity validates monotonic sequence numbering, tenant/workflow
// scoping, and cryptographic SHA-256 digest chaining across events.
// If startSeq > 1, prevDigest must equal the digest of sequence startSeq-1.
func VerifyHistoryIntegrity(tenantID, workflowID string, events []HistoryEvent, startSeq int64, prevDigest string) (string, error) {
	if prevDigest == "" {
		prevDigest = GenesisDigest
	}
	expectedSeq := startSeq
	if expectedSeq <= 0 {
		expectedSeq = 1
	}
	for _, ev := range events {
		if ev.TenantID != tenantID {
			return "", fmt.Errorf("%w: event seq %d belongs to mismatched tenant", ErrTenantAccessDenied, ev.Sequence)
		}
		if ev.WorkflowID != workflowID {
			return "", fmt.Errorf("%w: event seq %d belongs to mismatched workflow %q", ErrCorruptedHistory, ev.Sequence, ev.WorkflowID)
		}
		if ev.Sequence != expectedSeq {
			return "", fmt.Errorf("%w: expected sequence %d, got %d", ErrCorruptedHistory, expectedSeq, ev.Sequence)
		}
		if ev.PrevDigest != prevDigest {
			return "", fmt.Errorf("%w: prev_digest mismatch at sequence %d", ErrCorruptedHistory, ev.Sequence)
		}
		want := ComputeEventDigest(ev, prevDigest)
		if ev.Digest != want {
			return "", fmt.Errorf("%w: digest mismatch at sequence %d", ErrCorruptedHistory, ev.Sequence)
		}
		prevDigest = ev.Digest
		expectedSeq++
	}
	return prevDigest, nil
}

// ToWorkflowRunEvent maps a HistoryEvent onto control-plane's canonical
// types.WorkflowRunEvent schema.
func (e HistoryEvent) ToWorkflowRunEvent() types.WorkflowRunEvent {
	payload := e.Payload
	if len(payload) == 0 {
		payload = json.RawMessage(`{}`)
	}
	var statusReason *string
	if e.Error != "" {
		errCopy := e.Error
		statusReason = &errCopy
	}
	prevSeq := e.Sequence - 1
	if prevSeq < 0 {
		prevSeq = 0
	}
	emitted := time.Unix(0, e.LogicalTimeNS).UTC()
	recorded := e.RecordedAt
	if recorded.IsZero() {
		recorded = emitted
	}
	return types.WorkflowRunEvent{
		EventID:          e.Sequence,
		RunID:            e.WorkflowID,
		Sequence:         e.Sequence,
		PreviousSequence: prevSeq,
		EventType:        string(e.Type),
		StatusReason:     statusReason,
		Payload:          append(json.RawMessage(nil), payload...),
		EmittedAt:        emitted,
		RecordedAt:       recorded,
	}
}

// Checkpoint persists a periodic or explicit workflow snapshot allowing O(K)
// cold-start recovery by seeking directly to LogByteOffset in the append-only
// event log and replaying only delta events with Sequence > Checkpoint.Sequence.
type Checkpoint struct {
	TenantID         string                     `json:"tenant_id"`
	WorkflowID       string                     `json:"workflow_id"`
	WorkflowName     string                     `json:"workflow_name"`
	Label            string                     `json:"label"`
	Sequence         int64                      `json:"sequence"`
	StepIndex        int                        `json:"step_index"`
	ReplayCursor     int                        `json:"replay_cursor"`
	LogByteOffset    int64                      `json:"log_byte_offset"`
	HistoryDigest    string                     `json:"history_digest"`
	StartTimeNS      int64                      `json:"start_time_ns"`
	LogicalTimeNS    int64                      `json:"logical_time_ns"`
	RNGState         uint64                     `json:"rng_state"`
	RNGDraws         uint64                     `json:"rng_draws"`
	Status           WorkflowStatus             `json:"status"`
	AwaitingSignal   string                     `json:"awaiting_signal,omitempty"`
	AwaitingTimer    string                     `json:"awaiting_timer,omitempty"`
	DeliveredSignals map[string]json.RawMessage `json:"delivered_signals,omitempty"`
	FiredTimers      map[string]int64           `json:"fired_timers,omitempty"`
	UserState        json.RawMessage            `json:"user_state,omitempty"`
	CreatedAt        time.Time                  `json:"created_at"`
}

func (c *Checkpoint) clone() *Checkpoint {
	if c == nil {
		return nil
	}
	cp := *c
	if len(c.UserState) > 0 {
		cp.UserState = make(json.RawMessage, len(c.UserState))
		copy(cp.UserState, c.UserState)
	}
	if len(c.DeliveredSignals) > 0 {
		cp.DeliveredSignals = make(map[string]json.RawMessage, len(c.DeliveredSignals))
		for k, v := range c.DeliveredSignals {
			cp.DeliveredSignals[k] = append(json.RawMessage(nil), v...)
		}
	}
	if len(c.FiredTimers) > 0 {
		cp.FiredTimers = make(map[string]int64, len(c.FiredTimers))
		for k, v := range c.FiredTimers {
			cp.FiredTimers[k] = v
		}
	}
	return &cp
}

// WorkflowMeta summarizes persisted workflow metadata for canonical conversion
// and fast inspection without reading the full event history.
type WorkflowMeta struct {
	TenantID         string                     `json:"tenant_id"`
	WorkflowID       string                     `json:"workflow_id"`
	WorkflowName     string                     `json:"workflow_name"`
	Status           WorkflowStatus             `json:"status"`
	Sequence         int64                      `json:"sequence"`
	HistoryDigest    string                     `json:"history_digest"`
	StartTimeNS      int64                      `json:"start_time_ns"`
	LogicalTimeNS    int64                      `json:"logical_time_ns"`
	CompletedSteps   int                        `json:"completed_steps"`
	FailedSteps      int                        `json:"failed_steps"`
	CheckpointsCount int                        `json:"checkpoints_count"`
	AwaitingSignal   string                     `json:"awaiting_signal,omitempty"`
	AwaitingTimer    string                     `json:"awaiting_timer,omitempty"`
	DeliveredSignals map[string]json.RawMessage `json:"delivered_signals,omitempty"`
	FiredTimers      map[string]int64           `json:"fired_timers,omitempty"`
	LastError        string                     `json:"last_error,omitempty"`
	CreatedAt        time.Time                  `json:"created_at"`
	UpdatedAt        time.Time                  `json:"updated_at"`
	CompletedAt      *time.Time                 `json:"completed_at,omitempty"`
}

func (m *WorkflowMeta) clone() *WorkflowMeta {
	if m == nil {
		return nil
	}
	cp := *m
	if m.CompletedAt != nil {
		t := *m.CompletedAt
		cp.CompletedAt = &t
	}
	if len(m.DeliveredSignals) > 0 {
		cp.DeliveredSignals = make(map[string]json.RawMessage, len(m.DeliveredSignals))
		for k, v := range m.DeliveredSignals {
			cp.DeliveredSignals[k] = append(json.RawMessage(nil), v...)
		}
	}
	if len(m.FiredTimers) > 0 {
		cp.FiredTimers = make(map[string]int64, len(m.FiredTimers))
		for k, v := range m.FiredTimers {
			cp.FiredTimers[k] = v
		}
	}
	return &cp
}

// ToWorkflowRun converts WorkflowMeta into control-plane's canonical types.WorkflowRun.
func (m *WorkflowMeta) ToWorkflowRun() *types.WorkflowRun {
	if m == nil {
		return nil
	}
	metaBytes, _ := json.Marshal(map[string]any{
		"tenant_id":      m.TenantID,
		"history_digest": m.HistoryDigest,
		"checkpoints":    m.CheckpointsCount,
		"awaiting_sig":   m.AwaitingSignal,
		"awaiting_timer": m.AwaitingTimer,
	})
	return &types.WorkflowRun{
		RunID:             m.WorkflowID,
		RootWorkflowID:    m.WorkflowName,
		Status:            string(m.Status),
		TotalSteps:        m.CompletedSteps + m.FailedSteps,
		CompletedSteps:    m.CompletedSteps,
		FailedSteps:       m.FailedSteps,
		StateVersion:      m.Sequence,
		LastEventSequence: m.Sequence,
		Metadata:          metaBytes,
		CreatedAt:         m.CreatedAt,
		UpdatedAt:         m.UpdatedAt,
		CompletedAt:       m.CompletedAt,
	}
}

// ToExecution converts WorkflowMeta into control-plane's canonical types.Execution.
func (m *WorkflowMeta) ToExecution(agentNodeID, reasonerID string) *types.Execution {
	if m == nil {
		return nil
	}
	var errPtr *string
	if m.LastError != "" {
		s := m.LastError
		errPtr = &s
	}
	var durPtr *int64
	if m.CompletedAt != nil && !m.CreatedAt.IsZero() {
		d := m.CompletedAt.Sub(m.CreatedAt).Milliseconds()
		durPtr = &d
	}
	tenantCopy := m.TenantID
	return &types.Execution{
		ExecutionID:  m.WorkflowID,
		RunID:        m.WorkflowID,
		AgentNodeID:  agentNodeID,
		ReasonerID:   reasonerID,
		NodeID:       agentNodeID,
		Status:       string(m.Status),
		ErrorMessage: errPtr,
		StartedAt:    m.CreatedAt,
		CompletedAt:  m.CompletedAt,
		DurationMS:   durPtr,
		ActorID:      &tenantCopy,
		CreatedAt:    m.CreatedAt,
		UpdatedAt:    m.UpdatedAt,
	}
}

// ToWorkflowSteps projects activity events from a workflow history into
// control-plane's canonical types.WorkflowStep records.
func ToWorkflowSteps(events []HistoryEvent) []types.WorkflowStep {
	byKey := make(map[string]*types.WorkflowStep)
	order := make([]string, 0)

	for _, ev := range events {
		switch ev.Type {
		case EventActivityScheduled:
			if ev.IdempotencyKey == "" {
				continue
			}
			target := ev.ActivityName
			t := time.Unix(0, ev.LogicalTimeNS).UTC()
			step := &types.WorkflowStep{
				StepID:    ev.IdempotencyKey,
				RunID:     ev.WorkflowID,
				Target:    &target,
				Status:    string(StatusRunning),
				Attempt:   1,
				StartedAt: &t,
				CreatedAt: t,
				UpdatedAt: t,
			}
			if ev.PayloadURI != "" {
				u := ev.PayloadURI
				step.InputURI = &u
			}
			byKey[ev.IdempotencyKey] = step
			order = append(order, ev.IdempotencyKey)
		case EventActivityCompleted:
			step, ok := byKey[ev.IdempotencyKey]
			if !ok {
				continue
			}
			t := time.Unix(0, ev.LogicalTimeNS).UTC()
			step.Status = string(ActivityStatusCompleted)
			step.CompletedAt = &t
			step.UpdatedAt = t
			if ev.PayloadURI != "" {
				u := ev.PayloadURI
				step.ResultURI = &u
			}
		case EventActivityFailed:
			step, ok := byKey[ev.IdempotencyKey]
			if !ok {
				continue
			}
			t := time.Unix(0, ev.LogicalTimeNS).UTC()
			step.Status = string(ActivityStatusFailed)
			step.UpdatedAt = t
			if ev.Error != "" {
				s := ev.Error
				step.ErrorMessage = &s
			}
		}
	}

	out := make([]types.WorkflowStep, 0, len(order))
	for _, k := range order {
		out = append(out, *byKey[k])
	}
	return out
}
