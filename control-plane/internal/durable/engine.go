package durable

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"
)

var (
	// ErrSuspended signals that a workflow checkpointed and paused waiting for an external event.
	ErrSuspended = errors.New("durable: workflow suspended awaiting event")
	// ErrNonDeterministicReplay indicates workflow execution diverged from persisted journal history.
	ErrNonDeterministicReplay = errors.New("durable: non-deterministic workflow replay")
	// ErrInvalidActivation indicates a missing org, runID, or workflow handler.
	ErrInvalidActivation = errors.New("durable: invalid activation event")
)

// ActivationEventType identifies what triggered a stateless workflow activation.
type ActivationEventType string

const (
	EventStart        ActivationEventType = "workflow_started"
	EventResume       ActivationEventType = "workflow_resume"
	EventSignal       ActivationEventType = "workflow_signal"
	EventStepCallback ActivationEventType = "step_callback"
)

// ActivationEvent is an incoming event that wakes a workflow for one activation pass
// without maintaining any always-on worker or daemon process.
type ActivationEvent struct {
	Org          string              `json:"org"`
	RunID        string              `json:"run_id"`
	WorkflowName string              `json:"workflow_name"`
	Type         ActivationEventType `json:"type"`
	SignalName   string              `json:"signal_name,omitempty"`
	Payload      json.RawMessage     `json:"payload,omitempty"`
	TriggeredAt  time.Time           `json:"triggered_at"`
}

// ActivityInput carries deterministic context and the sha256(org:runID:round:stepID)
// idempotency key to an activity implementation.
type ActivityInput struct {
	Org            string          `json:"org"`
	RunID          string          `json:"run_id"`
	Round          int             `json:"round"`
	StepID         string          `json:"step_id"`
	IdempotencyKey string          `json:"idempotency_key"`
	Payload        json.RawMessage `json:"payload,omitempty"`
}

// ActivityFunc executes a single potentially side-effecting step.
type ActivityFunc func(ctx context.Context, input ActivityInput) (json.RawMessage, error)

// WorkflowHandler implements deterministic event-driven workflow logic.
type WorkflowHandler func(ctx context.Context, wctx *WorkflowContext, event ActivationEvent) error

// ActivationResult summarizes what happened during a single event-triggered activation.
type ActivationResult struct {
	Org                string    `json:"org"`
	RunID              string    `json:"run_id"`
	Status             RunStatus `json:"status"`
	Round              int       `json:"round"`
	Sequence           int64     `json:"sequence"`
	ExecutedActivities int       `json:"executed_activities"`
	ReplayedActivities int       `json:"replayed_activities"`
	DeduplicatedHits   int       `json:"deduplicated_hits"`
	CheckpointsSaved   int       `json:"checkpoints_saved"`
	AwaitingSignal     string    `json:"awaiting_signal,omitempty"`
	Error              string    `json:"error,omitempty"`
}

// Engine executes event-triggered durable workflows on demand with zero background
// goroutines or always-on daemon processes.
type Engine struct {
	store    JournalStore
	locksMu  sync.Mutex
	runLocks map[string]*sync.Mutex
}

// NewEngine creates a stateless event-driven replay Engine backed by store.
func NewEngine(store JournalStore) (*Engine, error) {
	if store == nil {
		return nil, errors.New("durable: journal store is required")
	}
	return &Engine{
		store:    store,
		runLocks: make(map[string]*sync.Mutex),
	}, nil
}

func (e *Engine) lockForRun(org, runID string) *sync.Mutex {
	k := runMapKey(org, runID)
	e.locksMu.Lock()
	defer e.locksMu.Unlock()
	mu, ok := e.runLocks[k]
	if !ok {
		mu = &sync.Mutex{}
		e.runLocks[k] = mu
	}
	return mu
}

// Activate executes one event-triggered activation for (event.Org, event.RunID),
// replaying completed activities from persisted journal checkpoints and persisting
// new checkpoints before returning.
func (e *Engine) Activate(ctx context.Context, event ActivationEvent, handler WorkflowHandler) (*ActivationResult, error) {
	if strings.TrimSpace(event.Org) == "" {
		return nil, fmt.Errorf("%w: org is required", ErrInvalidActivation)
	}
	if strings.TrimSpace(event.RunID) == "" {
		return nil, fmt.Errorf("%w: run_id is required", ErrInvalidActivation)
	}
	if handler == nil {
		return nil, fmt.Errorf("%w: handler is required", ErrInvalidActivation)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	runMu := e.lockForRun(event.Org, event.RunID)
	runMu.Lock()
	defer runMu.Unlock()

	workflowName := strings.TrimSpace(event.WorkflowName)
	if workflowName == "" {
		workflowName = "durable-workflow"
	}

	state, err := e.store.LoadOrCreateRun(ctx, event.Org, event.RunID, workflowName)
	if err != nil {
		return nil, err
	}

	if state.Status == RunStatusCompleted {
		return &ActivationResult{
			Org:                state.Org,
			RunID:              state.RunID,
			Status:             state.Status,
			Round:              state.Round,
			Sequence:           state.Sequence,
			ReplayedActivities: len(state.Ledger),
			DeduplicatedHits:   len(state.Ledger),
		}, nil
	}

	now := event.TriggeredAt
	if now.IsZero() {
		now = time.Now().UTC()
	}

	var initEntries []JournalEntry
	if state.Sequence == 0 {
		entry := JournalEntry{
			Sequence:         1,
			PreviousSequence: 0,
			Type:             JournalWorkflowStarted,
			Org:              state.Org,
			RunID:            state.RunID,
			Round:            0,
			Payload:          cloneRaw(event.Payload),
			Timestamp:        now,
		}
		state.applyEntry(entry)
		initEntries = append(initEntries, entry)
	}

	if event.Type == EventSignal && strings.TrimSpace(event.SignalName) != "" {
		entry := JournalEntry{
			Sequence:         state.Sequence + 1,
			PreviousSequence: state.Sequence,
			Type:             JournalWorkflowResumed,
			Org:              state.Org,
			RunID:            state.RunID,
			Round:            state.Round,
			Label:            strings.TrimSpace(event.SignalName),
			Payload:          cloneRaw(event.Payload),
			Timestamp:        now,
		}
		state.applyEntry(entry)
		initEntries = append(initEntries, entry)
	} else if state.Status != RunStatusRunning {
		state.Status = RunStatusRunning
		state.UpdatedAt = now
	}

	if len(initEntries) > 0 {
		if err := e.store.SaveRunState(ctx, state, initEntries); err != nil {
			return nil, err
		}
	}

	wctx := newWorkflowContext(ctx, e.store, state)
	runErr := handler(ctx, wctx, event)

	result := &ActivationResult{
		Org:                state.Org,
		RunID:              state.RunID,
		Round:              state.Round,
		Sequence:           state.Sequence,
		ExecutedActivities: wctx.executedActivities,
		ReplayedActivities: wctx.replayedActivities,
		DeduplicatedHits:   wctx.dedupHits,
		CheckpointsSaved:   wctx.checkpointsSaved,
		AwaitingSignal:     state.AwaitingSignal,
	}

	if runErr == nil {
		doneEntry := JournalEntry{
			Sequence:         state.Sequence + 1,
			PreviousSequence: state.Sequence,
			Type:             JournalWorkflowCompleted,
			Org:              state.Org,
			RunID:            state.RunID,
			Round:            wctx.round,
			Timestamp:        time.Now().UTC(),
		}
		state.applyEntry(doneEntry)
		if err := e.store.SaveRunState(ctx, state, []JournalEntry{doneEntry}); err != nil {
			return nil, err
		}
		result.Status = state.Status
		result.Sequence = state.Sequence
		return result, nil
	}

	if errors.Is(runErr, ErrSuspended) {
		result.Status = RunStatusSuspended
		result.AwaitingSignal = state.AwaitingSignal
		result.Sequence = state.Sequence
		return result, nil
	}

	failEntry := JournalEntry{
		Sequence:         state.Sequence + 1,
		PreviousSequence: state.Sequence,
		Type:             JournalWorkflowFailed,
		Org:              state.Org,
		RunID:            state.RunID,
		Round:            wctx.round,
		Error:            runErr.Error(),
		Timestamp:        time.Now().UTC(),
	}
	state.applyEntry(failEntry)
	if saveErr := e.store.SaveRunState(context.Background(), state, []JournalEntry{failEntry}); saveErr != nil {
		return result, errors.Join(runErr, saveErr)
	}
	result.Status = state.Status
	result.Sequence = state.Sequence
	result.Error = runErr.Error()
	return result, runErr
}

// WorkflowContext provides deterministic replay, checkpointing, and idempotent activity execution.
type WorkflowContext struct {
	ctx                  context.Context
	store                JournalStore
	state                *RunState
	round                int
	recordedActivities   []JournalEntry
	replayCursor         int
	jumpedFromCheckpoint bool
	executedActivities   int
	replayedActivities   int
	dedupHits            int
	checkpointsSaved     int
}

func newWorkflowContext(ctx context.Context, store JournalStore, state *RunState) *WorkflowContext {
	var recorded []JournalEntry
	for _, entry := range state.Entries {
		if entry.Type == JournalActivityCompleted {
			recorded = append(recorded, entry)
		}
	}
	return &WorkflowContext{
		ctx:                ctx,
		store:              store,
		state:              state,
		round:              0,
		recordedActivities: recorded,
	}
}

// Org returns the tenant identifier.
func (w *WorkflowContext) Org() string { return w.state.Org }

// RunID returns the workflow run identifier.
func (w *WorkflowContext) RunID() string { return w.state.RunID }

// Round returns the current workflow round index.
func (w *WorkflowContext) Round() int { return w.round }

// Sequence returns the latest persisted journal sequence number.
func (w *WorkflowContext) Sequence() int64 { return w.state.Sequence }

// NextRound advances the workflow round and persists a journal marker if advancing past state.Round.
func (w *WorkflowContext) NextRound() error {
	w.round++
	if w.round > w.state.Round {
		entry := JournalEntry{
			Sequence:         w.state.Sequence + 1,
			PreviousSequence: w.state.Sequence,
			Type:             JournalRoundAdvanced,
			Org:              w.state.Org,
			RunID:            w.state.RunID,
			Round:            w.round,
			Timestamp:        time.Now().UTC(),
		}
		w.state.applyEntry(entry)
		return w.store.SaveRunState(w.ctx, w.state, []JournalEntry{entry})
	}
	return nil
}

// Checkpoint durably persists named state between activations unless already recorded for (round, label).
func (w *WorkflowContext) Checkpoint(label string, statePayload any) error {
	if strings.TrimSpace(label) == "" {
		return errors.New("durable: checkpoint label cannot be empty")
	}
	for _, cp := range w.state.Checkpoints {
		if cp.Round == w.round && cp.Label == label {
			return nil
		}
	}

	raw, err := marshalOptionalJSON(statePayload)
	if err != nil {
		return fmt.Errorf("marshal checkpoint %s: %w", label, err)
	}

	entry := JournalEntry{
		Sequence:         w.state.Sequence + 1,
		PreviousSequence: w.state.Sequence,
		Type:             JournalCheckpointSaved,
		Org:              w.state.Org,
		RunID:            w.state.RunID,
		Round:            w.round,
		Label:            label,
		Payload:          raw,
		Timestamp:        time.Now().UTC(),
	}
	w.state.applyEntry(entry)
	if err := w.store.SaveRunState(w.ctx, w.state, []JournalEntry{entry}); err != nil {
		return err
	}
	w.checkpointsSaved++
	return nil
}

// RestoreCheckpoint loads the most recent persisted checkpoint with the given label
// and positions the workflow context at its round.
func (w *WorkflowContext) RestoreCheckpoint(label string, dest any) (bool, error) {
	for i := len(w.state.Checkpoints) - 1; i >= 0; i-- {
		cp := w.state.Checkpoints[i]
		if cp.Label != label {
			continue
		}
		w.round = cp.Round
		w.jumpedFromCheckpoint = true
		if dest != nil && len(cp.State) > 0 && string(cp.State) != "null" {
			if err := json.Unmarshal(cp.State, dest); err != nil {
				return false, fmt.Errorf("unmarshal checkpoint %s: %w", label, err)
			}
		}
		return true, nil
	}
	return false, nil
}

// SignalPayload checks whether signalName has been delivered to this run and decodes its payload.
func (w *WorkflowContext) SignalPayload(signalName string, dest any) (bool, error) {
	raw, ok := w.state.DeliveredSignals[strings.TrimSpace(signalName)]
	if !ok {
		return false, nil
	}
	if dest != nil && len(raw) > 0 && string(raw) != "null" {
		if err := json.Unmarshal(raw, dest); err != nil {
			return false, fmt.Errorf("unmarshal signal %s: %w", signalName, err)
		}
	}
	return true, nil
}

// AwaitEvent returns immediately if signalName has already been delivered; otherwise it
// persists a checkpoint and returns ErrSuspended so the event activation exits cleanly.
func (w *WorkflowContext) AwaitEvent(signalName string, checkpointState any) error {
	signalName = strings.TrimSpace(signalName)
	if signalName == "" {
		return errors.New("durable: signalName cannot be empty")
	}
	if _, delivered := w.state.DeliveredSignals[signalName]; delivered {
		return nil
	}

	if checkpointState != nil {
		if err := w.Checkpoint("await:"+signalName, checkpointState); err != nil {
			return err
		}
	}

	entry := JournalEntry{
		Sequence:         w.state.Sequence + 1,
		PreviousSequence: w.state.Sequence,
		Type:             JournalWorkflowSuspended,
		Org:              w.state.Org,
		RunID:            w.state.RunID,
		Round:            w.round,
		Label:            signalName,
		Timestamp:        time.Now().UTC(),
	}
	w.state.applyEntry(entry)
	if err := w.store.SaveRunState(w.ctx, w.state, []JournalEntry{entry}); err != nil {
		return err
	}
	return ErrSuspended
}

// ExecuteActivity runs fn at most once for sha256(org:runID:round:stepID).
// On replay or duplicate delivery, persisted output is returned without invoking fn.
func (w *WorkflowContext) ExecuteActivity(stepID string, payload any, fn ActivityFunc) (json.RawMessage, error) {
	if fn == nil {
		return nil, errors.New("durable: activity function cannot be nil")
	}
	if err := w.ctx.Err(); err != nil {
		return nil, err
	}

	key, err := IdempotencyKey(w.state.Org, w.state.RunID, w.round, stepID)
	if err != nil {
		return nil, err
	}

	rawPayload, err := marshalOptionalJSON(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal activity %s input: %w", stepID, err)
	}
	inputHash := hashRawPayload(rawPayload)

	ledgerEntry, ok := w.state.Ledger[key]
	if !ok {
		persisted, found, err := w.store.LookupActivity(w.ctx, w.state.Org, key)
		if err != nil {
			return nil, err
		}
		if found && persisted != nil {
			ledgerEntry = *persisted
			ok = true
		}
	}

	if ok && ledgerEntry.Status == ActivityStatusCompleted {
		if ledgerEntry.InputHash != "" && inputHash != "" && ledgerEntry.InputHash != inputHash {
			return nil, fmt.Errorf("%w: activity %s input payload mutated across replay", ErrNonDeterministicReplay, stepID)
		}
		if !w.jumpedFromCheckpoint && w.replayCursor < len(w.recordedActivities) {
			expected := w.recordedActivities[w.replayCursor]
			if expected.IdempotencyKey != key {
				return nil, fmt.Errorf("%w: expected step %q (round %d), got %q (round %d)",
					ErrNonDeterministicReplay, expected.StepID, expected.Round, stepID, w.round)
			}
			w.replayCursor++
		}
		w.replayedActivities++
		w.dedupHits++
		return cloneRaw(ledgerEntry.Output), nil
	}

	if !w.jumpedFromCheckpoint && w.replayCursor < len(w.recordedActivities) {
		expected := w.recordedActivities[w.replayCursor]
		return nil, fmt.Errorf("%w: expected replayed step %q (round %d) before new step %q (round %d)",
			ErrNonDeterministicReplay, expected.StepID, expected.Round, stepID, w.round)
	}

	out, actErr := fn(w.ctx, ActivityInput{
		Org:            w.state.Org,
		RunID:          w.state.RunID,
		Round:          w.round,
		StepID:         stepID,
		IdempotencyKey: key,
		Payload:        rawPayload,
	})
	if actErr != nil {
		failEntry := JournalEntry{
			Sequence:         w.state.Sequence + 1,
			PreviousSequence: w.state.Sequence,
			Type:             JournalActivityFailed,
			Org:              w.state.Org,
			RunID:            w.state.RunID,
			Round:            w.round,
			StepID:           stepID,
			IdempotencyKey:   key,
			InputHash:        inputHash,
			Error:            actErr.Error(),
			Timestamp:        time.Now().UTC(),
		}
		w.state.applyEntry(failEntry)
		if saveErr := w.store.SaveRunState(context.Background(), w.state, []JournalEntry{failEntry}); saveErr != nil {
			return nil, errors.Join(actErr, saveErr)
		}
		return nil, actErr
	}

	if len(out) == 0 {
		out = json.RawMessage("null")
	}

	doneEntry := JournalEntry{
		Sequence:         w.state.Sequence + 1,
		PreviousSequence: w.state.Sequence,
		Type:             JournalActivityCompleted,
		Org:              w.state.Org,
		RunID:            w.state.RunID,
		Round:            w.round,
		StepID:           stepID,
		IdempotencyKey:   key,
		InputHash:        inputHash,
		Payload:          cloneRaw(out),
		Timestamp:        time.Now().UTC(),
	}
	w.state.applyEntry(doneEntry)
	if err := w.store.SaveRunState(w.ctx, w.state, []JournalEntry{doneEntry}); err != nil {
		return nil, err
	}
	w.executedActivities++
	return cloneRaw(out), nil
}

func marshalOptionalJSON(v any) (json.RawMessage, error) {
	if v == nil {
		return nil, nil
	}
	if raw, ok := v.(json.RawMessage); ok {
		return cloneRaw(raw), nil
	}
	data, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	return data, nil
}

func cloneRaw(b json.RawMessage) json.RawMessage {
	if len(b) == 0 {
		return nil
	}
	out := make([]byte, len(b))
	copy(out, b)
	return out
}
