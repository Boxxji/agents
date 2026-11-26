package replay

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// FileStoreOptions configures disk durability and safety limits for FileCheckpointStore.
type FileStoreOptions struct {
	Limits Limits
	// SyncWrites controls fsync durability on WAL appends and atomic checkpoint renames.
	// Defaults to true when constructed via NewFileCheckpointStore.
	SyncWrites bool
}

// FileCheckpointStore persists append-only event logs (events.jsonl), atomic
// snapshot checkpoints (checkpoint.json), workflow metadata (meta.json), and
// tenant-partitioned idempotency ledgers on local disk with zero background daemons.
type FileCheckpointStore struct {
	rootDir    string
	limits     Limits
	syncWrites bool
	mu         sync.Mutex
}

// NewFileCheckpointStore initializes a crash-safe disk-backed store at rootDir
// with fsync durability enabled by default.
func NewFileCheckpointStore(rootDir string, opts *FileStoreOptions) (*FileCheckpointStore, error) {
	if strings.TrimSpace(rootDir) == "" {
		return nil, errors.New("replay: rootDir cannot be empty")
	}
	if err := os.MkdirAll(rootDir, 0o750); err != nil {
		return nil, fmt.Errorf("replay: create store root: %w", err)
	}
	eff := FileStoreOptions{
		Limits:     Limits{}.EffectiveLimits(),
		SyncWrites: true,
	}
	if opts != nil {
		eff.Limits = opts.Limits.EffectiveLimits()
		eff.SyncWrites = opts.SyncWrites
	}
	return &FileCheckpointStore{
		rootDir:    rootDir,
		limits:     eff.Limits,
		syncWrites: eff.SyncWrites,
	}, nil
}

func stableHashDir(val string) string {
	sum := sha256.Sum256([]byte(val))
	return hex.EncodeToString(sum[:16])
}

func (f *FileCheckpointStore) tenantDir(tenantID string) string {
	return filepath.Join(f.rootDir, "tenants", stableHashDir(tenantID))
}

// WorkflowDir returns the isolated directory for (tenantID, workflowID).
func (f *FileCheckpointStore) WorkflowDir(tenantID, workflowID string) string {
	return filepath.Join(f.tenantDir(tenantID), "workflows", stableHashDir(workflowID))
}

func (f *FileCheckpointStore) ledgerDir(tenantID, workflowID string) string {
	return filepath.Join(f.WorkflowDir(tenantID, workflowID), "idempotency")
}

func (f *FileCheckpointStore) ledgerFilePath(tenantID, workflowID, scopedKey string) string {
	return filepath.Join(f.ledgerDir(tenantID, workflowID), scopedKey+".json")
}

func (f *FileCheckpointStore) LoadWorkflow(ctx context.Context, tenantID, workflowID string, opts LoadOptions) (*WorkflowLoadResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := validateTenantWorkflow(tenantID, workflowID); err != nil {
		return nil, err
	}

	f.mu.Lock()
	defer f.mu.Unlock()

	wfDir := f.WorkflowDir(tenantID, workflowID)
	metaPath := filepath.Join(wfDir, "meta.json")
	cpPath := filepath.Join(wfDir, "checkpoint.json")
	logPath := filepath.Join(wfDir, "events.jsonl")

	var meta *WorkflowMeta
	var metaBytes int64
	if raw, err := os.ReadFile(metaPath); err == nil {
		metaBytes = int64(len(raw))
		var parsed WorkflowMeta
		if err := json.Unmarshal(raw, &parsed); err != nil {
			return nil, fmt.Errorf("%w: decode meta.json: %v", ErrCorruptedHistory, err)
		}
		if parsed.TenantID != tenantID {
			return nil, ErrTenantAccessDenied
		}
		if parsed.WorkflowID != workflowID {
			return nil, fmt.Errorf("%w: workflow_id mismatch in meta.json", ErrCorruptedHistory)
		}
		meta = &parsed
	} else if !os.IsNotExist(err) {
		return nil, fmt.Errorf("replay: read meta.json: %w", err)
	}

	var cp *Checkpoint
	var cpBytes int64
	if opts.UseCheckpoint {
		if raw, err := os.ReadFile(cpPath); err == nil {
			cpBytes = int64(len(raw))
			var parsed Checkpoint
			if err := json.Unmarshal(raw, &parsed); err != nil {
				return nil, fmt.Errorf("%w: decode checkpoint.json: %v", ErrCorruptedHistory, err)
			}
			if parsed.TenantID != tenantID {
				return nil, ErrTenantAccessDenied
			}
			if parsed.WorkflowID != workflowID {
				return nil, fmt.Errorf("%w: workflow_id mismatch in checkpoint.json", ErrCorruptedHistory)
			}
			cp = &parsed
		} else if !os.IsNotExist(err) {
			return nil, fmt.Errorf("replay: read checkpoint.json: %w", err)
		}
	}

	// Fast checkpointed load: seek directly to cp.LogByteOffset in events.jsonl
	// and read + verify only delta events with Sequence > cp.Sequence.
	if cp != nil {
		deltaEvents, logReadBytes, endOffset, err := readEventsFromOffset(logPath, cp.LogByteOffset, f.limits.MaxPayloadBytes)
		if err != nil {
			return nil, err
		}
		if _, err := VerifyHistoryIntegrity(tenantID, workflowID, deltaEvents, cp.Sequence+1, cp.HistoryDigest); err != nil {
			return nil, err
		}
		if meta == nil {
			meta = rebuildMetaFromCheckpoint(cp, deltaEvents)
		} else if len(deltaEvents) > 0 && deltaEvents[len(deltaEvents)-1].Sequence > meta.Sequence {
			applyDeltaToMeta(meta, deltaEvents)
		}
		return &WorkflowLoadResult{
			Meta:              meta,
			Checkpoint:        cp,
			Events:            deltaEvents,
			FromCheckpoint:    true,
			DeltaEventsLoaded: len(deltaEvents),
			TotalBytesRead:    metaBytes + cpBytes + logReadBytes,
			LogByteOffset:     endOffset,
		}, nil
	}

	// Full-history load (or cold recovery when checkpoint.json is absent).
	events, logReadBytes, endOffset, err := readEventsFromOffset(logPath, 0, f.limits.MaxPayloadBytes)
	if err != nil {
		return nil, err
	}
	if len(events) == 0 && meta == nil {
		return nil, nil
	}
	if _, err := VerifyHistoryIntegrity(tenantID, workflowID, events, 1, GenesisDigest); err != nil {
		return nil, err
	}
	if meta == nil {
		meta = rebuildMetaFromEvents(tenantID, workflowID, events)
	} else if len(events) > 0 && events[len(events)-1].Sequence > meta.Sequence {
		var tail []HistoryEvent
		for _, ev := range events {
			if ev.Sequence > meta.Sequence {
				tail = append(tail, ev)
			}
		}
		applyDeltaToMeta(meta, tail)
	}

	return &WorkflowLoadResult{
		Meta:              meta,
		Checkpoint:        nil,
		Events:            events,
		FromCheckpoint:    false,
		DeltaEventsLoaded: len(events),
		TotalBytesRead:    metaBytes + logReadBytes,
		LogByteOffset:     endOffset,
	}, nil
}

func (f *FileCheckpointStore) AppendEvents(ctx context.Context, tenantID, workflowID string, meta *WorkflowMeta, newEvents []HistoryEvent) (int64, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	if err := validateTenantWorkflow(tenantID, workflowID); err != nil {
		return 0, err
	}
	if meta == nil || meta.TenantID != tenantID || meta.WorkflowID != workflowID {
		return 0, ErrTenantAccessDenied
	}
	if meta.Sequence > f.limits.MaxHistoryDepth {
		return 0, fmt.Errorf("%w: sequence %d > limit %d", ErrHistoryDepthExceeded, meta.Sequence, f.limits.MaxHistoryDepth)
	}
	for _, ev := range newEvents {
		if ev.TenantID != tenantID || ev.WorkflowID != workflowID {
			return 0, ErrTenantAccessDenied
		}
		if len(ev.Payload) > f.limits.MaxPayloadBytes {
			return 0, fmt.Errorf("%w: event seq %d payload %d bytes > limit %d", ErrPayloadTooLarge, ev.Sequence, len(ev.Payload), f.limits.MaxPayloadBytes)
		}
	}

	f.mu.Lock()
	defer f.mu.Unlock()

	wfDir := f.WorkflowDir(tenantID, workflowID)
	if err := os.MkdirAll(wfDir, 0o750); err != nil {
		return 0, fmt.Errorf("replay: create workflow dir: %w", err)
	}

	logPath := filepath.Join(wfDir, "events.jsonl")
	var endOffset int64
	if len(newEvents) > 0 {
		lf, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o640)
		if err != nil {
			return 0, fmt.Errorf("replay: open events.jsonl: %w", err)
		}
		for _, ev := range newEvents {
			line, err := json.Marshal(ev)
			if err != nil {
				_ = lf.Close()
				return 0, fmt.Errorf("replay: marshal event: %w", err)
			}
			line = append(line, '\n')
			if _, err := lf.Write(line); err != nil {
				_ = lf.Close()
				return 0, fmt.Errorf("replay: append event: %w", err)
			}
		}
		if f.syncWrites {
			if err := lf.Sync(); err != nil {
				_ = lf.Close()
				return 0, fmt.Errorf("replay: sync events.jsonl: %w", err)
			}
		}
		info, statErr := lf.Stat()
		if closeErr := lf.Close(); closeErr != nil {
			return 0, fmt.Errorf("replay: close events.jsonl: %w", closeErr)
		}
		if statErr != nil {
			return 0, fmt.Errorf("replay: stat events.jsonl: %w", statErr)
		}
		endOffset = info.Size()
	} else {
		if info, err := os.Stat(logPath); err == nil {
			endOffset = info.Size()
		}
	}

	metaRaw, err := json.Marshal(meta)
	if err != nil {
		return 0, fmt.Errorf("replay: marshal meta.json: %w", err)
	}
	// meta.json is a derived summary cache; events.jsonl above is the authoritative
	// fsynced WAL and LoadWorkflow deterministically recovers any tail past meta.Sequence.
	if err := writeAtomicFile(filepath.Join(wfDir, "meta.json"), metaRaw, false, false); err != nil {
		return 0, fmt.Errorf("replay: write meta.json: %w", err)
	}
	return endOffset, nil
}

func (f *FileCheckpointStore) SaveCheckpoint(ctx context.Context, tenantID, workflowID string, cp *Checkpoint) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := validateTenantWorkflow(tenantID, workflowID); err != nil {
		return err
	}
	if cp == nil || cp.TenantID != tenantID || cp.WorkflowID != workflowID {
		return ErrTenantAccessDenied
	}
	if len(cp.UserState) > f.limits.MaxPayloadBytes {
		return fmt.Errorf("%w: checkpoint payload %d bytes > limit %d", ErrPayloadTooLarge, len(cp.UserState), f.limits.MaxPayloadBytes)
	}

	f.mu.Lock()
	defer f.mu.Unlock()

	wfDir := f.WorkflowDir(tenantID, workflowID)
	if err := os.MkdirAll(wfDir, 0o750); err != nil {
		return fmt.Errorf("replay: create workflow dir: %w", err)
	}
	raw, err := json.Marshal(cp)
	if err != nil {
		return fmt.Errorf("replay: marshal checkpoint.json: %w", err)
	}
	return writeAtomicFile(filepath.Join(wfDir, "checkpoint.json"), raw, f.syncWrites, f.syncWrites)
}

func (f *FileCheckpointStore) LookupIdempotency(ctx context.Context, tenantID, workflowID, idempotencyKey string) (*IdempotencyRecord, bool, error) {
	if err := ctx.Err(); err != nil {
		return nil, false, err
	}
	scoped, err := ScopedLedgerKey(tenantID, workflowID, idempotencyKey)
	if err != nil {
		return nil, false, err
	}

	f.mu.Lock()
	defer f.mu.Unlock()

	data, err := os.ReadFile(f.ledgerFilePath(tenantID, workflowID, scoped))
	if err != nil {
		if os.IsNotExist(err) {
			return nil, false, nil
		}
		return nil, false, fmt.Errorf("replay: read idempotency record: %w", err)
	}
	var rec IdempotencyRecord
	if err := json.Unmarshal(data, &rec); err != nil {
		return nil, false, fmt.Errorf("%w: decode idempotency record: %v", ErrCorruptedHistory, err)
	}
	if rec.TenantID != tenantID || rec.WorkflowID != workflowID {
		return nil, false, ErrTenantAccessDenied
	}
	return &rec, true, nil
}

func (f *FileCheckpointStore) RecordIdempotency(ctx context.Context, tenantID, workflowID string, rec IdempotencyRecord) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if rec.TenantID != tenantID || rec.WorkflowID != workflowID {
		return ErrTenantAccessDenied
	}
	if len(rec.Output) > f.limits.MaxPayloadBytes {
		return fmt.Errorf("%w: idempotency output %d bytes > limit %d", ErrPayloadTooLarge, len(rec.Output), f.limits.MaxPayloadBytes)
	}
	scoped, err := ScopedLedgerKey(tenantID, workflowID, rec.IdempotencyKey)
	if err != nil {
		return err
	}

	f.mu.Lock()
	defer f.mu.Unlock()

	lDir := f.ledgerDir(tenantID, workflowID)
	if err := os.MkdirAll(lDir, 0o750); err != nil {
		return fmt.Errorf("replay: create idempotency dir: %w", err)
	}
	raw, err := json.Marshal(rec)
	if err != nil {
		return fmt.Errorf("replay: marshal idempotency record: %w", err)
	}
	return writeAtomicFile(f.ledgerFilePath(tenantID, workflowID, scoped), raw, f.syncWrites, false)
}

func (f *FileCheckpointStore) StorageBytes(ctx context.Context, tenantID, workflowID string) (int64, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	if err := validateTenantWorkflow(tenantID, workflowID); err != nil {
		return 0, err
	}

	f.mu.Lock()
	defer f.mu.Unlock()

	wfDir := f.WorkflowDir(tenantID, workflowID)
	var total int64
	err := filepath.Walk(wfDir, func(_ string, info os.FileInfo, walkErr error) error {
		if walkErr != nil {
			if os.IsNotExist(walkErr) {
				return nil
			}
			return walkErr
		}
		if !info.IsDir() {
			total += info.Size()
		}
		return nil
	})
	if err != nil && !os.IsNotExist(err) {
		return 0, err
	}
	return total, nil
}

func readEventsFromOffset(logPath string, offset int64, maxPayloadBytes int) ([]HistoryEvent, int64, int64, error) {
	lf, err := os.Open(logPath)
	if err != nil {
		if os.IsNotExist(err) {
			if offset > 0 {
				return nil, 0, 0, fmt.Errorf("%w: events.jsonl missing at checkpoint offset %d", ErrCorruptedHistory, offset)
			}
			return nil, 0, 0, nil
		}
		return nil, 0, 0, fmt.Errorf("replay: open events.jsonl: %w", err)
	}
	defer lf.Close()

	info, err := lf.Stat()
	if err != nil {
		return nil, 0, 0, fmt.Errorf("replay: stat events.jsonl: %w", err)
	}
	fileSize := info.Size()
	if offset < 0 || offset > fileSize {
		return nil, 0, 0, fmt.Errorf("%w: checkpoint log offset %d exceeds events.jsonl size %d", ErrCorruptedHistory, offset, fileSize)
	}
	if offset > 0 {
		if _, err := lf.Seek(offset, io.SeekStart); err != nil {
			return nil, 0, 0, fmt.Errorf("replay: seek events.jsonl: %w", err)
		}
	}

	maxLineBytes := maxPayloadBytes*2 + 64*1024
	if maxLineBytes < 256*1024 {
		maxLineBytes = 256 * 1024
	}
	scanner := bufio.NewScanner(lf)
	scanner.Buffer(make([]byte, 0, 64*1024), maxLineBytes)

	var events []HistoryEvent
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var ev HistoryEvent
		if err := json.Unmarshal([]byte(line), &ev); err != nil {
			return nil, 0, 0, fmt.Errorf("%w: decode event line: %v", ErrCorruptedHistory, err)
		}
		events = append(events, ev)
	}
	if err := scanner.Err(); err != nil {
		return nil, 0, 0, fmt.Errorf("replay: scan events.jsonl: %w", err)
	}
	return events, fileSize - offset, fileSize, nil
}

func rebuildMetaFromCheckpoint(cp *Checkpoint, delta []HistoryEvent) *WorkflowMeta {
	m := &WorkflowMeta{
		TenantID:         cp.TenantID,
		WorkflowID:       cp.WorkflowID,
		WorkflowName:     cp.WorkflowName,
		Status:           cp.Status,
		Sequence:         cp.Sequence,
		HistoryDigest:    cp.HistoryDigest,
		StartTimeNS:      cp.StartTimeNS,
		LogicalTimeNS:    cp.LogicalTimeNS,
		CheckpointsCount: 1,
		AwaitingSignal:   cp.AwaitingSignal,
		AwaitingTimer:    cp.AwaitingTimer,
		DeliveredSignals: make(map[string]json.RawMessage, len(cp.DeliveredSignals)),
		FiredTimers:      make(map[string]int64, len(cp.FiredTimers)),
		CreatedAt:        time.Unix(0, cp.StartTimeNS).UTC(),
		UpdatedAt:        cp.CreatedAt,
	}
	for k, v := range cp.DeliveredSignals {
		m.DeliveredSignals[k] = append(json.RawMessage(nil), v...)
	}
	for k, v := range cp.FiredTimers {
		m.FiredTimers[k] = v
	}
	applyDeltaToMeta(m, delta)
	return m
}

func rebuildMetaFromEvents(tenantID, workflowID string, events []HistoryEvent) *WorkflowMeta {
	m := &WorkflowMeta{
		TenantID:         tenantID,
		WorkflowID:       workflowID,
		Status:           StatusPending,
		HistoryDigest:    GenesisDigest,
		DeliveredSignals: make(map[string]json.RawMessage),
		FiredTimers:      make(map[string]int64),
	}
	applyDeltaToMeta(m, events)
	return m
}

func applyDeltaToMeta(m *WorkflowMeta, events []HistoryEvent) {
	if m.DeliveredSignals == nil {
		m.DeliveredSignals = make(map[string]json.RawMessage)
	}
	if m.FiredTimers == nil {
		m.FiredTimers = make(map[string]int64)
	}
	for _, ev := range events {
		if ev.Sequence > m.Sequence {
			m.Sequence = ev.Sequence
			m.HistoryDigest = ev.Digest
		}
		if ev.LogicalTimeNS > m.LogicalTimeNS {
			m.LogicalTimeNS = ev.LogicalTimeNS
		}
		if !ev.RecordedAt.IsZero() {
			if m.CreatedAt.IsZero() {
				m.CreatedAt = ev.RecordedAt
			}
			m.UpdatedAt = ev.RecordedAt
		}
		switch ev.Type {
		case EventWorkflowStarted:
			m.Status = StatusRunning
			m.StartTimeNS = ev.LogicalTimeNS
			if ev.Label != "" {
				m.WorkflowName = ev.Label
			}
		case EventActivityCompleted:
			m.CompletedSteps++
		case EventActivityFailed:
			m.FailedSteps++
		case EventCheckpointSaved:
			m.CheckpointsCount++
		case EventTimerFired:
			if ev.TimerID != "" {
				m.FiredTimers[ev.TimerID] = ev.LogicalTimeNS
				if m.AwaitingTimer == ev.TimerID {
					m.AwaitingTimer = ""
				}
			}
		case EventSignalReceived:
			if ev.SignalName != "" {
				m.DeliveredSignals[ev.SignalName] = append(json.RawMessage(nil), ev.Payload...)
				if m.AwaitingSignal == ev.SignalName {
					m.AwaitingSignal = ""
				}
			}
			if m.Status == StatusSuspended {
				m.Status = StatusRunning
			}
		case EventWorkflowSuspended:
			m.Status = StatusSuspended
			m.AwaitingSignal = ev.SignalName
			m.AwaitingTimer = ev.TimerID
		case EventWorkflowCompleted:
			m.Status = StatusCompleted
			t := ev.RecordedAt
			if t.IsZero() {
				t = time.Unix(0, ev.LogicalTimeNS).UTC()
			}
			m.CompletedAt = &t
		case EventWorkflowFailed:
			m.Status = StatusFailed
			m.LastError = ev.Error
		}
	}
}

func writeAtomicFile(targetPath string, data []byte, syncFile bool, syncDir bool) error {
	dir := filepath.Dir(targetPath)
	tmp, err := os.CreateTemp(dir, ".tmp-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	cleanup := true
	defer func() {
		if cleanup {
			_ = os.Remove(tmpName)
		}
	}()

	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		return err
	}
	if syncFile {
		if err := tmp.Sync(); err != nil {
			_ = tmp.Close()
			return err
		}
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmpName, targetPath); err != nil {
		return err
	}
	cleanup = false

	if syncDir {
		if df, err := os.Open(dir); err == nil {
			_ = df.Sync()
			_ = df.Close()
		}
	}
	return nil
}
