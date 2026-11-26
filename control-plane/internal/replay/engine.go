package replay

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"
)

// ReplayMode selects the execution and recovery strategy for an Engine.
type ReplayMode string

const (
	// ModeCheckpointedReplay is Hanzo's event-triggered durable replay mode:
	// periodic snapshot checkpoints + O(1) log-offset seek + delta event replay +
	// tenant-scoped idempotency_key ledger.
	ModeCheckpointedReplay ReplayMode = "checkpointed_replay"

	// ModeFullHistoryReplay is the matched Temporal-style baseline without
	// snapshot compaction: every recovery reads and verifies the full event log
	// from sequence 1 and replays every recorded step.
	ModeFullHistoryReplay ReplayMode = "full_history_replay"

	// ModeStatelessRetry is the naive stateless retry baseline: no durable
	// checkpoints or idempotency ledger; retries re-execute steps from step 0.
	ModeStatelessRetry ReplayMode = "stateless_retry"
)

// TriggerType identifies what external event woke a workflow activation.
type TriggerType string

const (
	TriggerStart            TriggerType = "start"
	TriggerResume           TriggerType = "resume"
	TriggerSignal           TriggerType = "signal"
	TriggerTimer            TriggerType = "timer"
	TriggerActivityCallback TriggerType = "activity_callback"
)

// TriggerEvent is a stateless external invocation that activates a workflow for
// a single synchronous pass with zero always-on daemon processes.
type TriggerEvent struct {
	TenantID     string          `json:"tenant_id"`
	WorkflowID   string          `json:"workflow_id"`
	WorkflowName string          `json:"workflow_name,omitempty"`
	Type         TriggerType     `json:"type"`
	SignalName   string          `json:"signal_name,omitempty"`
	TimerID      string          `json:"timer_id,omitempty"`
	Payload      json.RawMessage `json:"payload,omitempty"`
	// LogicalTime optionally pins the deterministic start/trigger timestamp.
	// When zero, time.Now().UTC() is recorded on first activation and replayed thereafter.
	LogicalTime time.Time `json:"logical_time,omitempty"`
}

// EngineConfig configures an event-triggered replay Engine.
type EngineConfig struct {
	Mode               ReplayMode
	Limits             Limits
	CheckpointInterval int
	// AutoFireTimers controls whether Sleep() immediately advances the virtual
	// clock and records TimerFired (true) or suspends with ErrSuspended until an
	// external TriggerTimer arrives (false, default).
	AutoFireTimers bool
}

// ActivityInput carries deterministic context and the scoped idempotency_key
// into an activity implementation.
type ActivityInput struct {
	TenantID       string          `json:"tenant_id"`
	WorkflowID     string          `json:"workflow_id"`
	ActivityName   string          `json:"activity_name"`
	IdempotencyKey string          `json:"idempotency_key"`
	LogicalTime    time.Time       `json:"logical_time"`
	Payload        json.RawMessage `json:"payload,omitempty"`
}

// ActivityFunc executes a single potentially side-effecting step (tool call,
// VFS edit, billing charge, external API call).
type ActivityFunc func(ctx context.Context, input ActivityInput) (json.RawMessage, error)

// WorkflowFunc implements deterministic event-driven workflow logic.
type WorkflowFunc func(ctx context.Context, wctx *WorkflowContext, trigger TriggerEvent) error

// ExecutionReport captures exact per-activation telemetry and status.
type ExecutionReport struct {
	TenantID           string         `json:"tenant_id"`
	WorkflowID         string         `json:"workflow_id"`
	Mode               ReplayMode     `json:"mode"`
	Status             WorkflowStatus `json:"status"`
	Sequence           int64          `json:"sequence"`
	HistoryDigest      string         `json:"history_digest"`
	RestoredCheckpoint bool           `json:"restored_checkpoint"`
	CheckpointSequence int64          `json:"checkpoint_sequence"`
	EventsLoaded       int            `json:"events_loaded"`
	BytesReadOnLoad    int64          `json:"bytes_read_on_load"`
	ExecutedActivities int            `json:"executed_activities"`
	ReplayedActivities int            `json:"replayed_activities"`
	DeduplicatedHits   int            `json:"deduplicated_hits"`
	CheckpointsSaved   int            `json:"checkpoints_saved"`
	AwaitingSignal     string         `json:"awaiting_signal,omitempty"`
	AwaitingTimer      string         `json:"awaiting_timer,omitempty"`
	Error              string         `json:"error,omitempty"`
}

type refMutex struct {
	mu   sync.Mutex
	refs int
}

// Engine orchestrates event-triggered durable workflow activations with zero
// background goroutines, zero polling loops, and ref-counted per-workflow locks
// that release all memory when workflows are idle.
type Engine struct {
	store  CheckpointStore
	cfg    EngineConfig
	lockMu sync.Mutex
	locks  map[string]*refMutex
}

// NewEngine creates a stateless, event-triggered Engine backed by store.
// In ModeStatelessRetry, store may be nil.
func NewEngine(store CheckpointStore, cfg EngineConfig) (*Engine, error) {
	if cfg.Mode == "" {
		cfg.Mode = ModeCheckpointedReplay
	}
	if cfg.Mode != ModeStatelessRetry && store == nil {
		return nil, errors.New("replay: CheckpointStore is required for durable modes")
	}
	cfg.Limits = cfg.Limits.EffectiveLimits()
	return &Engine{
		store: store,
		cfg:   cfg,
		locks: make(map[string]*refMutex),
	}, nil
}

// ActiveLockCount returns the number of currently held or contended workflow
// locks. When idle, this is strictly 0 (proving zero idle per-workflow memory).
func (e *Engine) ActiveLockCount() int {
	e.lockMu.Lock()
	defer e.lockMu.Unlock()
	return len(e.locks)
}

func (e *Engine) acquireLock(tenantID, workflowID string) func() {
	k := workflowScopeKey(tenantID, workflowID)
	e.lockMu.Lock()
	entry, ok := e.locks[k]
	if !ok {
		entry = &refMutex{}
		e.locks[k] = entry
	}
	entry.refs++
	e.lockMu.Unlock()

	entry.mu.Lock()
	return func() {
		entry.mu.Unlock()
		e.lockMu.Lock()
		entry.refs--
		if entry.refs == 0 {
			delete(e.locks, k)
		}
		e.lockMu.Unlock()
	}
}

// HandleEvent synchronously executes one event-triggered activation for
// (trigger.TenantID, trigger.WorkflowID), restoring persisted state if present,
// replaying completed activities deterministically, and durably committing any
// new events/checkpoints before returning.
func (e *Engine) HandleEvent(ctx context.Context, trigger TriggerEvent, fn WorkflowFunc) (*ExecutionReport, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := validateTenantWorkflow(trigger.TenantID, trigger.WorkflowID); err != nil {
		return nil, err
	}
	if fn == nil {
		return nil, fmt.Errorf("%w: workflow function is required", ErrInvalidTrigger)
	}

	release := e.acquireLock(trigger.TenantID, trigger.WorkflowID)
	defer release()

	if e.cfg.Mode == ModeStatelessRetry {
		return e.handleStateless(ctx, trigger, fn)
	}

	wfName := strings.TrimSpace(trigger.WorkflowName)
	if wfName == "" {
		wfName = "hanzo-durable-workflow"
	}

	useCP := e.cfg.Mode == ModeCheckpointedReplay
	loaded, err := e.store.LoadWorkflow(ctx, trigger.TenantID, trigger.WorkflowID, LoadOptions{
		UseCheckpoint: useCP,
	})
	if err != nil {
		return nil, err
	}

	var (
		meta            *WorkflowMeta
		cp              *Checkpoint
		loadedEvents    []HistoryEvent
		fromCheckpoint  bool
		bytesReadOnLoad int64
		logByteOffset   int64
	)

	now := trigger.LogicalTime.UTC()
	if now.IsZero() {
		now = time.Now().UTC()
	}

	if loaded == nil {
		meta = &WorkflowMeta{
			TenantID:         trigger.TenantID,
			WorkflowID:       trigger.WorkflowID,
			WorkflowName:     wfName,
			Status:           StatusPending,
			HistoryDigest:    GenesisDigest,
			StartTimeNS:      now.UnixNano(),
			LogicalTimeNS:    now.UnixNano(),
			DeliveredSignals: make(map[string]json.RawMessage),
			FiredTimers:      make(map[string]int64),
			CreatedAt:        now,
			UpdatedAt:        now,
		}
	} else {
		meta = loaded.Meta
		cp = loaded.Checkpoint
		loadedEvents = loaded.Events
		fromCheckpoint = loaded.FromCheckpoint
		bytesReadOnLoad = loaded.TotalBytesRead
		logByteOffset = loaded.LogByteOffset
		if meta.WorkflowName == "" {
			meta.WorkflowName = wfName
		}
	}

	// Idempotent no-op for already completed workflows (protects against duplicate webhook deliveries).
	if meta.Status == StatusCompleted {
		var cpSeq int64
		if cp != nil {
			cpSeq = cp.Sequence
		}
		return &ExecutionReport{
			TenantID:           meta.TenantID,
			WorkflowID:         meta.WorkflowID,
			Mode:               e.cfg.Mode,
			Status:             StatusCompleted,
			Sequence:           meta.Sequence,
			HistoryDigest:      meta.HistoryDigest,
			RestoredCheckpoint: fromCheckpoint,
			CheckpointSequence: cpSeq,
			EventsLoaded:       len(loadedEvents),
			BytesReadOnLoad:    bytesReadOnLoad,
			ReplayedActivities: meta.CompletedSteps,
			DeduplicatedHits:   meta.CompletedSteps,
		}, nil
	}

	// Build initial trigger events (WorkflowStarted, SignalReceived, TimerFired).
	var initEvents []HistoryEvent
	if meta.Sequence == 0 {
		startEv := HistoryEvent{
			Sequence:      1,
			Type:          EventWorkflowStarted,
			TenantID:      meta.TenantID,
			WorkflowID:    meta.WorkflowID,
			Label:         meta.WorkflowName,
			Payload:       cloneRaw(trigger.Payload),
			LogicalTimeNS: meta.StartTimeNS,
			RecordedAt:    now,
		}
		SealEvent(&startEv, meta.HistoryDigest)
		applyDeltaToMeta(meta, []HistoryEvent{startEv})
		initEvents = append(initEvents, startEv)
	}

	switch trigger.Type {
	case TriggerSignal:
		sigName := strings.TrimSpace(trigger.SignalName)
		if sigName == "" {
			return nil, fmt.Errorf("%w: signal_name required for signal trigger", ErrInvalidTrigger)
		}
		if _, exists := meta.DeliveredSignals[sigName]; !exists {
			logicalNS := maxInt64(meta.LogicalTimeNS+1, now.UnixNano())
			sigEv := HistoryEvent{
				Sequence:      meta.Sequence + 1,
				Type:          EventSignalReceived,
				TenantID:      meta.TenantID,
				WorkflowID:    meta.WorkflowID,
				SignalName:    sigName,
				Payload:       cloneRaw(trigger.Payload),
				LogicalTimeNS: logicalNS,
				RecordedAt:    time.Now().UTC(),
			}
			SealEvent(&sigEv, meta.HistoryDigest)
			applyDeltaToMeta(meta, []HistoryEvent{sigEv})
			initEvents = append(initEvents, sigEv)
		}
	case TriggerTimer:
		timerID := strings.TrimSpace(trigger.TimerID)
		if timerID == "" {
			return nil, fmt.Errorf("%w: timer_id required for timer trigger", ErrInvalidTrigger)
		}
		if _, exists := meta.FiredTimers[timerID]; !exists {
			logicalNS := maxInt64(meta.LogicalTimeNS+1, now.UnixNano())
			timerEv := HistoryEvent{
				Sequence:      meta.Sequence + 1,
				Type:          EventTimerFired,
				TenantID:      meta.TenantID,
				WorkflowID:    meta.WorkflowID,
				TimerID:       timerID,
				LogicalTimeNS: logicalNS,
				RecordedAt:    time.Now().UTC(),
			}
			SealEvent(&timerEv, meta.HistoryDigest)
			applyDeltaToMeta(meta, []HistoryEvent{timerEv})
			initEvents = append(initEvents, timerEv)
		}
	default:
		if meta.Status != StatusRunning {
			meta.Status = StatusRunning
			meta.UpdatedAt = time.Now().UTC()
		}
	}

	if len(initEvents) > 0 {
		newOffset, err := e.store.AppendEvents(ctx, meta.TenantID, meta.WorkflowID, meta, initEvents)
		if err != nil {
			return nil, err
		}
		logByteOffset = newOffset
		loadedEvents = append(loadedEvents, initEvents...)
	}

	wctx := newWorkflowContext(ctx, e.store, e.cfg, meta, cp, loadedEvents, fromCheckpoint, logByteOffset)
	runErr := fn(ctx, wctx, trigger)

	var cpSeq int64
	if cp != nil {
		cpSeq = cp.Sequence
	}
	report := &ExecutionReport{
		TenantID:           meta.TenantID,
		WorkflowID:         meta.WorkflowID,
		Mode:               e.cfg.Mode,
		Sequence:           meta.Sequence,
		HistoryDigest:      meta.HistoryDigest,
		RestoredCheckpoint: wctx.restoredCheckpoint,
		CheckpointSequence: cpSeq,
		EventsLoaded:       len(loadedEvents) - len(initEvents),
		BytesReadOnLoad:    bytesReadOnLoad,
		ExecutedActivities: wctx.executedActivities,
		ReplayedActivities: wctx.replayedActivities,
		DeduplicatedHits:   wctx.dedupHits,
		CheckpointsSaved:   wctx.checkpointsSaved,
		AwaitingSignal:     meta.AwaitingSignal,
		AwaitingTimer:      meta.AwaitingTimer,
	}

	if runErr == nil {
		doneEv := HistoryEvent{
			Sequence:      meta.Sequence + 1,
			Type:          EventWorkflowCompleted,
			TenantID:      meta.TenantID,
			WorkflowID:    meta.WorkflowID,
			LogicalTimeNS: wctx.clock.UnixNano(),
			RecordedAt:    time.Now().UTC(),
		}
		SealEvent(&doneEv, meta.HistoryDigest)
		applyDeltaToMeta(meta, []HistoryEvent{doneEv})
		if _, err := e.store.AppendEvents(ctx, meta.TenantID, meta.WorkflowID, meta, []HistoryEvent{doneEv}); err != nil {
			return nil, err
		}
		report.Status = StatusCompleted
		report.Sequence = meta.Sequence
		report.HistoryDigest = meta.HistoryDigest
		return report, nil
	}

	if errors.Is(runErr, ErrSuspended) {
		report.Status = StatusSuspended
		report.Sequence = meta.Sequence
		report.HistoryDigest = meta.HistoryDigest
		report.AwaitingSignal = meta.AwaitingSignal
		report.AwaitingTimer = meta.AwaitingTimer
		return report, nil
	}

	// Do not corrupt or append WorkflowFailed on non-determinism or security violations;
	// preserve the persisted history intact so operators can deploy fixed code and resume.
	if errors.Is(runErr, ErrNonDeterministicReplay) ||
		errors.Is(runErr, ErrTenantAccessDenied) ||
		errors.Is(runErr, ErrPayloadTooLarge) ||
		errors.Is(runErr, ErrHistoryDepthExceeded) {
		report.Status = meta.Status
		report.Error = runErr.Error()
		return report, runErr
	}

	failEv := HistoryEvent{
		Sequence:      meta.Sequence + 1,
		Type:          EventWorkflowFailed,
		TenantID:      meta.TenantID,
		WorkflowID:    meta.WorkflowID,
		Error:         runErr.Error(),
		LogicalTimeNS: wctx.clock.UnixNano(),
		RecordedAt:    time.Now().UTC(),
	}
	SealEvent(&failEv, meta.HistoryDigest)
	applyDeltaToMeta(meta, []HistoryEvent{failEv})
	if _, saveErr := e.store.AppendEvents(context.Background(), meta.TenantID, meta.WorkflowID, meta, []HistoryEvent{failEv}); saveErr != nil {
		return report, errors.Join(runErr, saveErr)
	}
	report.Status = StatusFailed
	report.Sequence = meta.Sequence
	report.HistoryDigest = meta.HistoryDigest
	report.Error = runErr.Error()
	return report, runErr
}

func (e *Engine) handleStateless(ctx context.Context, trigger TriggerEvent, fn WorkflowFunc) (*ExecutionReport, error) {
	now := trigger.LogicalTime.UTC()
	if now.IsZero() {
		now = time.Now().UTC()
	}
	meta := &WorkflowMeta{
		TenantID:         trigger.TenantID,
		WorkflowID:       trigger.WorkflowID,
		WorkflowName:     trigger.WorkflowName,
		Status:           StatusRunning,
		HistoryDigest:    GenesisDigest,
		StartTimeNS:      now.UnixNano(),
		LogicalTimeNS:    now.UnixNano(),
		DeliveredSignals: make(map[string]json.RawMessage),
		FiredTimers:      make(map[string]int64),
		CreatedAt:        now,
		UpdatedAt:        now,
	}
	if trigger.Type == TriggerSignal && trigger.SignalName != "" {
		meta.DeliveredSignals[trigger.SignalName] = cloneRaw(trigger.Payload)
	}
	if trigger.Type == TriggerTimer && trigger.TimerID != "" {
		meta.FiredTimers[trigger.TimerID] = now.UnixNano()
	}

	wctx := newWorkflowContext(ctx, nil, e.cfg, meta, nil, nil, false, 0)
	runErr := fn(ctx, wctx, trigger)
	report := &ExecutionReport{
		TenantID:           trigger.TenantID,
		WorkflowID:         trigger.WorkflowID,
		Mode:               ModeStatelessRetry,
		ExecutedActivities: wctx.executedActivities,
	}
	if runErr == nil {
		report.Status = StatusCompleted
		return report, nil
	}
	if errors.Is(runErr, ErrSuspended) {
		report.Status = StatusSuspended
		return report, nil
	}
	report.Status = StatusFailed
	report.Error = runErr.Error()
	return report, runErr
}

type commandKind string

const (
	cmdActivity commandKind = "activity"
	cmdTimer    commandKind = "timer"
)

type recordedCommand struct {
	kind           commandKind
	activityName   string
	idempotencyKey string
	inputDigest    string
	timerID        string
	logicalTimeNS  int64
	completed      bool
	output         json.RawMessage
}

// WorkflowContext exposes deterministic replay, virtual time, seeded RNG,
// checkpointing, signals, timers, and idempotent activity execution to workflows.
type WorkflowContext struct {
	ctx                context.Context
	store              CheckpointStore
	cfg                EngineConfig
	meta               *WorkflowMeta
	checkpoint         *Checkpoint
	fromCheckpoint     bool
	restoredCheckpoint bool
	logByteOffset      int64

	clock *WorkflowClock
	rng   *WorkflowRNG

	replayCommands     []recordedCommand
	replayCursor       int
	completedInHistory map[string]recordedCommand

	executedActivities int
	replayedActivities int
	dedupHits          int
	checkpointsSaved   int
	liveStepsSinceCP   int
	stepIndex          int
}

func newWorkflowContext(
	ctx context.Context,
	store CheckpointStore,
	cfg EngineConfig,
	meta *WorkflowMeta,
	cp *Checkpoint,
	events []HistoryEvent,
	fromCheckpoint bool,
	logByteOffset int64,
) *WorkflowContext {
	commands, completedMap := buildReplayCommands(events)
	return &WorkflowContext{
		ctx:                ctx,
		store:              store,
		cfg:                cfg,
		meta:               meta,
		checkpoint:         cp,
		fromCheckpoint:     fromCheckpoint,
		logByteOffset:      logByteOffset,
		clock:              NewWorkflowClock(meta.StartTimeNS),
		rng:                NewWorkflowRNG(meta.TenantID, meta.WorkflowID, meta.StartTimeNS),
		replayCommands:     commands,
		completedInHistory: completedMap,
	}
}

// buildReplayCommands extracts the ordered deterministic command stream from
// loaded history events. Consecutive failed attempts of an activity that later
// succeeded are collapsed onto the completed attempt so retries replay cleanly
// while still enforcing strict non-determinism checks on name, key, input, and order.
func buildReplayCommands(events []HistoryEvent) ([]recordedCommand, map[string]recordedCommand) {
	type actAgg struct {
		cmd     recordedCommand
		hasDone bool
		order   int
	}
	byKey := make(map[string]*actAgg)
	completedMap := make(map[string]recordedCommand)

	type rawSlot struct {
		isTimer bool
		key     string
		timer   recordedCommand
	}
	var slots []rawSlot

	for _, ev := range events {
		switch ev.Type {
		case EventActivityScheduled:
			agg, exists := byKey[ev.IdempotencyKey]
			if !exists {
				agg = &actAgg{
					cmd: recordedCommand{
						kind:           cmdActivity,
						activityName:   ev.ActivityName,
						idempotencyKey: ev.IdempotencyKey,
						inputDigest:    ev.InputDigest,
						logicalTimeNS:  ev.LogicalTimeNS,
					},
					order: len(slots),
				}
				byKey[ev.IdempotencyKey] = agg
				slots = append(slots, rawSlot{key: ev.IdempotencyKey})
			} else if !agg.hasDone {
				agg.cmd.inputDigest = ev.InputDigest
				agg.cmd.logicalTimeNS = ev.LogicalTimeNS
			}
		case EventActivityCompleted:
			agg, exists := byKey[ev.IdempotencyKey]
			if !exists {
				agg = &actAgg{
					cmd: recordedCommand{
						kind:           cmdActivity,
						activityName:   ev.ActivityName,
						idempotencyKey: ev.IdempotencyKey,
						inputDigest:    ev.InputDigest,
						logicalTimeNS:  ev.LogicalTimeNS,
						completed:      true,
						output:         cloneRaw(ev.Payload),
					},
					hasDone: true,
					order:   len(slots),
				}
				byKey[ev.IdempotencyKey] = agg
				slots = append(slots, rawSlot{key: ev.IdempotencyKey})
			} else {
				agg.hasDone = true
				agg.cmd.completed = true
				agg.cmd.output = cloneRaw(ev.Payload)
				agg.cmd.logicalTimeNS = ev.LogicalTimeNS
			}
			completedMap[ev.IdempotencyKey] = agg.cmd
		case EventTimerFired:
			slots = append(slots, rawSlot{
				isTimer: true,
				timer: recordedCommand{
					kind:          cmdTimer,
					timerID:       ev.TimerID,
					logicalTimeNS: ev.LogicalTimeNS,
					completed:     true,
				},
			})
		}
	}

	out := make([]recordedCommand, 0, len(slots))
	for _, s := range slots {
		if s.isTimer {
			out = append(out, s.timer)
			continue
		}
		out = append(out, byKey[s.key].cmd)
	}
	return out, completedMap
}

// TenantID returns the isolated tenant identifier for this workflow.
func (w *WorkflowContext) TenantID() string { return w.meta.TenantID }

// WorkflowID returns the workflow identifier.
func (w *WorkflowContext) WorkflowID() string { return w.meta.WorkflowID }

// Sequence returns the latest committed event sequence number.
func (w *WorkflowContext) Sequence() int64 { return w.meta.Sequence }

// HistoryDigest returns the current canonical SHA-256 history digest.
func (w *WorkflowContext) HistoryDigest() string { return w.meta.HistoryDigest }

// IsReplaying returns true while the context is still stepping through recorded history.
func (w *WorkflowContext) IsReplaying() bool { return w.replayCursor < len(w.replayCommands) }

// Now returns the current deterministic virtual UTC timestamp.
func (w *WorkflowContext) Now() time.Time { return w.clock.Now() }

// RNG returns the workflow's deterministic, checkpointable PRNG.
func (w *WorkflowContext) RNG() *WorkflowRNG { return w.rng }

// RestoreCheckpoint restores userState, deterministic clock, seeded PRNG, and
// stepIndex from the latest persisted checkpoint when running in
// ModeCheckpointedReplay. Returns (stepIndex, restored, error).
func (w *WorkflowContext) RestoreCheckpoint(dest any) (int, bool, error) {
	if w.cfg.Mode != ModeCheckpointedReplay || !w.fromCheckpoint || w.checkpoint == nil {
		return 0, false, nil
	}
	cp := w.checkpoint
	w.clock = NewWorkflowClock(cp.LogicalTimeNS)
	w.rng = RestoreWorkflowRNG(cp.RNGState, cp.RNGDraws)
	w.stepIndex = cp.StepIndex
	w.restoredCheckpoint = true
	if dest != nil && len(cp.UserState) > 0 && string(cp.UserState) != "null" {
		if err := json.Unmarshal(cp.UserState, dest); err != nil {
			return 0, false, fmt.Errorf("replay: unmarshal checkpoint user_state: %w", err)
		}
	}
	return cp.StepIndex, true, nil
}

// SaveCheckpoint atomically persists a snapshot of workflow state, virtual clock,
// seeded PRNG, and the current event log byte offset so future cold-start
// recoveries skip the compacted event prefix in O(1) I/O.
func (w *WorkflowContext) SaveCheckpoint(label string, stepIndex int, userState any) error {
	if w.cfg.Mode != ModeCheckpointedReplay {
		return nil
	}
	if strings.TrimSpace(label) == "" {
		return errors.New("replay: checkpoint label cannot be empty")
	}
	rawState, err := marshalOptional(userState)
	if err != nil {
		return fmt.Errorf("replay: marshal checkpoint state: %w", err)
	}
	if len(rawState) > w.cfg.Limits.MaxPayloadBytes {
		return fmt.Errorf("%w: checkpoint state %d bytes > limit %d", ErrPayloadTooLarge, len(rawState), w.cfg.Limits.MaxPayloadBytes)
	}

	cpEv := HistoryEvent{
		Sequence:      w.meta.Sequence + 1,
		Type:          EventCheckpointSaved,
		TenantID:      w.meta.TenantID,
		WorkflowID:    w.meta.WorkflowID,
		Label:         label,
		LogicalTimeNS: w.clock.UnixNano(),
		RecordedAt:    time.Now().UTC(),
	}
	SealEvent(&cpEv, w.meta.HistoryDigest)
	applyDeltaToMeta(w.meta, []HistoryEvent{cpEv})

	newOffset, err := w.store.AppendEvents(w.ctx, w.meta.TenantID, w.meta.WorkflowID, w.meta, []HistoryEvent{cpEv})
	if err != nil {
		return err
	}
	w.logByteOffset = newOffset

	rngState, rngDraws := w.rng.Snapshot()
	cp := &Checkpoint{
		TenantID:         w.meta.TenantID,
		WorkflowID:       w.meta.WorkflowID,
		WorkflowName:     w.meta.WorkflowName,
		Label:            label,
		Sequence:         w.meta.Sequence,
		StepIndex:        stepIndex,
		ReplayCursor:     w.replayCursor,
		LogByteOffset:    w.logByteOffset,
		HistoryDigest:    w.meta.HistoryDigest,
		StartTimeNS:      w.meta.StartTimeNS,
		LogicalTimeNS:    w.clock.UnixNano(),
		RNGState:         rngState,
		RNGDraws:         rngDraws,
		Status:           w.meta.Status,
		AwaitingSignal:   w.meta.AwaitingSignal,
		AwaitingTimer:    w.meta.AwaitingTimer,
		DeliveredSignals: cloneSignalMap(w.meta.DeliveredSignals),
		FiredTimers:      cloneTimerMap(w.meta.FiredTimers),
		UserState:        rawState,
		CreatedAt:        time.Now().UTC(),
	}
	if err := w.store.SaveCheckpoint(w.ctx, w.meta.TenantID, w.meta.WorkflowID, cp); err != nil {
		return err
	}
	w.checkpoint = cp
	w.stepIndex = stepIndex
	w.checkpointsSaved++
	w.liveStepsSinceCP = 0
	return nil
}

// Sleep executes a deterministic workflow timer. During replay it returns
// immediately and advances the virtual clock to the recorded TimerFired instant.
// When live, if AutoFireTimers is true it advances virtual time and records
// TimerFired; otherwise it checkpoints and returns ErrSuspended (0 goroutines).
func (w *WorkflowContext) Sleep(timerID string, d time.Duration, checkpointState any) error {
	timerID = strings.TrimSpace(timerID)
	if timerID == "" {
		return errors.New("replay: timer_id cannot be empty")
	}
	if w.cfg.Mode == ModeStatelessRetry {
		w.clock.AdvanceBy(d)
		return nil
	}

	if w.replayCursor < len(w.replayCommands) {
		expected := w.replayCommands[w.replayCursor]
		if expected.kind != cmdTimer {
			return fmt.Errorf("%w: expected activity %q (%s), got timer %q",
				ErrNonDeterministicReplay, expected.activityName, expected.idempotencyKey, timerID)
		}
		if expected.timerID != timerID {
			return fmt.Errorf("%w: expected timer %q, got %q",
				ErrNonDeterministicReplay, expected.timerID, timerID)
		}
		w.clock.AdvanceTo(expected.logicalTimeNS)
		w.replayCursor++
		return nil
	}

	if fireNS, fired := w.meta.FiredTimers[timerID]; fired {
		w.clock.AdvanceTo(fireNS)
		return nil
	}

	if w.cfg.AutoFireTimers {
		w.clock.AdvanceBy(d)
		ev := HistoryEvent{
			Sequence:      w.meta.Sequence + 1,
			Type:          EventTimerFired,
			TenantID:      w.meta.TenantID,
			WorkflowID:    w.meta.WorkflowID,
			TimerID:       timerID,
			LogicalTimeNS: w.clock.UnixNano(),
			RecordedAt:    time.Now().UTC(),
		}
		SealEvent(&ev, w.meta.HistoryDigest)
		applyDeltaToMeta(w.meta, []HistoryEvent{ev})
		newOffset, err := w.store.AppendEvents(w.ctx, w.meta.TenantID, w.meta.WorkflowID, w.meta, []HistoryEvent{ev})
		if err != nil {
			return err
		}
		w.logByteOffset = newOffset
		return nil
	}

	if checkpointState != nil {
		if err := w.SaveCheckpoint("await-timer:"+timerID, w.stepIndex, checkpointState); err != nil {
			return err
		}
	}
	suspEv := HistoryEvent{
		Sequence:      w.meta.Sequence + 1,
		Type:          EventWorkflowSuspended,
		TenantID:      w.meta.TenantID,
		WorkflowID:    w.meta.WorkflowID,
		TimerID:       timerID,
		LogicalTimeNS: w.clock.UnixNano(),
		RecordedAt:    time.Now().UTC(),
	}
	SealEvent(&suspEv, w.meta.HistoryDigest)
	applyDeltaToMeta(w.meta, []HistoryEvent{suspEv})
	newOffset, err := w.store.AppendEvents(w.ctx, w.meta.TenantID, w.meta.WorkflowID, w.meta, []HistoryEvent{suspEv})
	if err != nil {
		return err
	}
	w.logByteOffset = newOffset
	return ErrSuspended
}

// AwaitSignal returns immediately if signalName has already been delivered;
// otherwise it durably checkpoints and returns ErrSuspended (zero resident goroutines).
func (w *WorkflowContext) AwaitSignal(signalName string, checkpointState any) error {
	signalName = strings.TrimSpace(signalName)
	if signalName == "" {
		return errors.New("replay: signal_name cannot be empty")
	}
	if _, ok := w.meta.DeliveredSignals[signalName]; ok {
		return nil
	}
	if w.cfg.Mode == ModeStatelessRetry {
		return ErrSuspended
	}
	if checkpointState != nil {
		if err := w.SaveCheckpoint("await-signal:"+signalName, w.stepIndex, checkpointState); err != nil {
			return err
		}
	}
	suspEv := HistoryEvent{
		Sequence:      w.meta.Sequence + 1,
		Type:          EventWorkflowSuspended,
		TenantID:      w.meta.TenantID,
		WorkflowID:    w.meta.WorkflowID,
		SignalName:    signalName,
		LogicalTimeNS: w.clock.UnixNano(),
		RecordedAt:    time.Now().UTC(),
	}
	SealEvent(&suspEv, w.meta.HistoryDigest)
	applyDeltaToMeta(w.meta, []HistoryEvent{suspEv})
	newOffset, err := w.store.AppendEvents(w.ctx, w.meta.TenantID, w.meta.WorkflowID, w.meta, []HistoryEvent{suspEv})
	if err != nil {
		return err
	}
	w.logByteOffset = newOffset
	return ErrSuspended
}

// SignalPayload decodes a previously delivered signal payload into dest.
func (w *WorkflowContext) SignalPayload(signalName string, dest any) (bool, error) {
	raw, ok := w.meta.DeliveredSignals[strings.TrimSpace(signalName)]
	if !ok {
		return false, nil
	}
	if dest != nil && len(raw) > 0 && string(raw) != "null" {
		if err := json.Unmarshal(raw, dest); err != nil {
			return false, fmt.Errorf("replay: unmarshal signal %s: %w", signalName, err)
		}
	}
	return true, nil
}

// ExecuteActivity runs fn at most once for (tenant_id, workflow_id, idempotency_key).
// On replay or duplicate delivery, persisted output is returned without re-invoking fn.
// Divergent activity name, key, input digest, or command ordering fails fast with
// ErrNonDeterministicReplay.
func (w *WorkflowContext) ExecuteActivity(
	activityName string,
	idempotencyKey string,
	input any,
	fn ActivityFunc,
) (json.RawMessage, error) {
	if err := w.ctx.Err(); err != nil {
		return nil, err
	}
	if strings.TrimSpace(activityName) == "" {
		return nil, fmt.Errorf("%w: activity_name is required", ErrInvalidIdempotencyKey)
	}
	if strings.TrimSpace(idempotencyKey) == "" {
		return nil, fmt.Errorf("%w: idempotency_key is required", ErrInvalidIdempotencyKey)
	}
	if fn == nil {
		return nil, errors.New("replay: activity function cannot be nil")
	}

	rawInput, err := marshalOptional(input)
	if err != nil {
		return nil, fmt.Errorf("replay: marshal activity %s input: %w", activityName, err)
	}
	if len(rawInput) > w.cfg.Limits.MaxPayloadBytes {
		return nil, fmt.Errorf("%w: activity %s input %d bytes > limit %d", ErrPayloadTooLarge, activityName, len(rawInput), w.cfg.Limits.MaxPayloadBytes)
	}
	inputDigest := digestBytes(rawInput)

	// Stateless retry baseline: always executes fn live with zero persistence or deduplication.
	if w.cfg.Mode == ModeStatelessRetry {
		w.clock.AdvanceBy(time.Millisecond)
		out, actErr := fn(w.ctx, ActivityInput{
			TenantID:       w.meta.TenantID,
			WorkflowID:     w.meta.WorkflowID,
			ActivityName:   activityName,
			IdempotencyKey: idempotencyKey,
			LogicalTime:    w.clock.Now(),
			Payload:        cloneRaw(rawInput),
		})
		if actErr != nil {
			return nil, actErr
		}
		if len(out) == 0 {
			out = json.RawMessage("null")
		}
		w.executedActivities++
		w.stepIndex++
		return cloneRaw(out), nil
	}

	// If loaded from a checkpoint AND workflow did not call RestoreCheckpoint
	// (imperative top-to-bottom replay), check whether this activity already
	// completed prior to the checkpoint sequence.
	if w.fromCheckpoint && !w.restoredCheckpoint && w.checkpoint != nil {
		rec, found, err := w.store.LookupIdempotency(w.ctx, w.meta.TenantID, w.meta.WorkflowID, idempotencyKey)
		if err != nil {
			return nil, err
		}
		if found && rec.Status == ActivityStatusCompleted && rec.CompletedSeq <= w.checkpoint.Sequence {
			if rec.ActivityName != activityName {
				return nil, fmt.Errorf("%w: pre-checkpoint activity name mismatch: expected %q, got %q",
					ErrNonDeterministicReplay, rec.ActivityName, activityName)
			}
			if rec.InputDigest != inputDigest {
				return nil, fmt.Errorf("%w: activity %q input mutated across replay", ErrNonDeterministicReplay, activityName)
			}
			w.clock.AdvanceTo(rec.LogicalTimeNS)
			w.replayedActivities++
			w.dedupHits++
			w.stepIndex++
			return cloneRaw(rec.Output), nil
		}
	}

	// Strict deterministic command stream verification during replay.
	if w.replayCursor < len(w.replayCommands) {
		expected := w.replayCommands[w.replayCursor]
		if expected.kind != cmdActivity {
			return nil, fmt.Errorf("%w: expected timer %q at cursor %d, got activity %q (%s)",
				ErrNonDeterministicReplay, expected.timerID, w.replayCursor, activityName, idempotencyKey)
		}
		if expected.activityName != activityName || expected.idempotencyKey != idempotencyKey {
			return nil, fmt.Errorf("%w: expected activity %q (%s) at cursor %d, got %q (%s)",
				ErrNonDeterministicReplay, expected.activityName, expected.idempotencyKey, w.replayCursor, activityName, idempotencyKey)
		}
		if expected.inputDigest != inputDigest {
			return nil, fmt.Errorf("%w: activity %q (%s) input payload mutated across replay",
				ErrNonDeterministicReplay, activityName, idempotencyKey)
		}
		w.replayCursor++
		if expected.completed {
			w.clock.AdvanceTo(expected.logicalTimeNS)
			w.replayedActivities++
			w.dedupHits++
			w.stepIndex++
			return cloneRaw(expected.output), nil
		}
		// Previous attempt failed before completing; proceed to live retry below.
	}

	// Cross-check the durable idempotency ledger before running live side effects
	// (protects against duplicate calls or out-of-band retries even after checkpoint jump).
	if rec, found, err := w.store.LookupIdempotency(w.ctx, w.meta.TenantID, w.meta.WorkflowID, idempotencyKey); err != nil {
		return nil, err
	} else if found && rec.Status == ActivityStatusCompleted {
		if rec.ActivityName != activityName {
			return nil, fmt.Errorf("%w: ledger activity name mismatch: expected %q, got %q",
				ErrNonDeterministicReplay, rec.ActivityName, activityName)
		}
		if rec.InputDigest != inputDigest {
			return nil, fmt.Errorf("%w: activity %q (%s) input mutated across replay",
				ErrNonDeterministicReplay, activityName, idempotencyKey)
		}
		w.clock.AdvanceTo(rec.LogicalTimeNS)
		w.replayedActivities++
		w.dedupHits++
		w.stepIndex++
		return cloneRaw(rec.Output), nil
	}

	// Live execution path: append ActivityScheduled first (write-ahead).
	w.clock.AdvanceBy(time.Millisecond)
	schedTimeNS := w.clock.UnixNano()
	schedEv := HistoryEvent{
		Sequence:       w.meta.Sequence + 1,
		Type:           EventActivityScheduled,
		TenantID:       w.meta.TenantID,
		WorkflowID:     w.meta.WorkflowID,
		ActivityName:   activityName,
		IdempotencyKey: idempotencyKey,
		InputDigest:    inputDigest,
		Payload:        cloneRaw(rawInput),
		LogicalTimeNS:  schedTimeNS,
		RecordedAt:     time.Now().UTC(),
	}
	SealEvent(&schedEv, w.meta.HistoryDigest)
	applyDeltaToMeta(w.meta, []HistoryEvent{schedEv})
	newOffset, err := w.store.AppendEvents(w.ctx, w.meta.TenantID, w.meta.WorkflowID, w.meta, []HistoryEvent{schedEv})
	if err != nil {
		return nil, err
	}
	w.logByteOffset = newOffset

	out, actErr := fn(w.ctx, ActivityInput{
		TenantID:       w.meta.TenantID,
		WorkflowID:     w.meta.WorkflowID,
		ActivityName:   activityName,
		IdempotencyKey: idempotencyKey,
		LogicalTime:    w.clock.Now(),
		Payload:        cloneRaw(rawInput),
	})
	if actErr != nil {
		failEv := HistoryEvent{
			Sequence:       w.meta.Sequence + 1,
			Type:           EventActivityFailed,
			TenantID:       w.meta.TenantID,
			WorkflowID:     w.meta.WorkflowID,
			ActivityName:   activityName,
			IdempotencyKey: idempotencyKey,
			ScheduledSeq:   schedEv.Sequence,
			InputDigest:    inputDigest,
			Error:          actErr.Error(),
			LogicalTimeNS:  w.clock.UnixNano(),
			RecordedAt:     time.Now().UTC(),
		}
		SealEvent(&failEv, w.meta.HistoryDigest)
		applyDeltaToMeta(w.meta, []HistoryEvent{failEv})
		if off, saveErr := w.store.AppendEvents(context.Background(), w.meta.TenantID, w.meta.WorkflowID, w.meta, []HistoryEvent{failEv}); saveErr != nil {
			return nil, errors.Join(actErr, saveErr)
		} else {
			w.logByteOffset = off
		}
		return nil, actErr
	}

	if len(out) == 0 {
		out = json.RawMessage("null")
	}
	if len(out) > w.cfg.Limits.MaxPayloadBytes {
		return nil, fmt.Errorf("%w: activity %s output %d bytes > limit %d", ErrPayloadTooLarge, activityName, len(out), w.cfg.Limits.MaxPayloadBytes)
	}

	compEv := HistoryEvent{
		Sequence:       w.meta.Sequence + 1,
		Type:           EventActivityCompleted,
		TenantID:       w.meta.TenantID,
		WorkflowID:     w.meta.WorkflowID,
		ActivityName:   activityName,
		IdempotencyKey: idempotencyKey,
		ScheduledSeq:   schedEv.Sequence,
		InputDigest:    inputDigest,
		Payload:        cloneRaw(out),
		LogicalTimeNS:  w.clock.UnixNano(),
		RecordedAt:     time.Now().UTC(),
	}
	SealEvent(&compEv, w.meta.HistoryDigest)
	applyDeltaToMeta(w.meta, []HistoryEvent{compEv})

	// Record idempotency ledger entry BEFORE committing completion event so even
	// a crash mid-commit never allows duplicate execution of the completed side effect.
	ledgerRec := IdempotencyRecord{
		TenantID:       w.meta.TenantID,
		WorkflowID:     w.meta.WorkflowID,
		IdempotencyKey: idempotencyKey,
		ActivityName:   activityName,
		ScheduledSeq:   schedEv.Sequence,
		CompletedSeq:   compEv.Sequence,
		Status:         ActivityStatusCompleted,
		InputDigest:    inputDigest,
		Output:         cloneRaw(out),
		LogicalTimeNS:  compEv.LogicalTimeNS,
		RecordedAt:     compEv.RecordedAt,
	}
	if err := w.store.RecordIdempotency(w.ctx, w.meta.TenantID, w.meta.WorkflowID, ledgerRec); err != nil {
		return nil, err
	}
	newOffset, err = w.store.AppendEvents(w.ctx, w.meta.TenantID, w.meta.WorkflowID, w.meta, []HistoryEvent{compEv})
	if err != nil {
		return nil, err
	}
	w.logByteOffset = newOffset
	w.executedActivities++
	w.liveStepsSinceCP++
	w.stepIndex++
	return cloneRaw(out), nil
}

// ShouldCheckpoint returns true when automatic periodic checkpointing is enabled
// and at least CheckpointInterval live activities have completed since the last snapshot.
func (w *WorkflowContext) ShouldCheckpoint() bool {
	return w.cfg.Mode == ModeCheckpointedReplay &&
		w.cfg.CheckpointInterval > 0 &&
		w.liveStepsSinceCP >= w.cfg.CheckpointInterval
}

func marshalOptional(v any) (json.RawMessage, error) {
	if v == nil {
		return nil, nil
	}
	if raw, ok := v.(json.RawMessage); ok {
		return cloneRaw(raw), nil
	}
	b, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	return b, nil
}

func cloneRaw(b json.RawMessage) json.RawMessage {
	if len(b) == 0 {
		return nil
	}
	cp := make(json.RawMessage, len(b))
	copy(cp, b)
	return cp
}

func cloneSignalMap(src map[string]json.RawMessage) map[string]json.RawMessage {
	if len(src) == 0 {
		return nil
	}
	dst := make(map[string]json.RawMessage, len(src))
	for k, v := range src {
		dst[k] = cloneRaw(v)
	}
	return dst
}

func cloneTimerMap(src map[string]int64) map[string]int64 {
	if len(src) == 0 {
		return nil
	}
	dst := make(map[string]int64, len(src))
	for k, v := range src {
		dst[k] = v
	}
	return dst
}

func maxInt64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}
