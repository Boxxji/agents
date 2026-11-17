package durable

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/hanzoai/agents/control-plane/pkg/types"
)

// JournalEventType enumerates append-only durable execution journal events.
type JournalEventType string

const (
	JournalWorkflowStarted   JournalEventType = "workflow_started"
	JournalActivityCompleted JournalEventType = "activity_completed"
	JournalActivityFailed    JournalEventType = "activity_failed"
	JournalCheckpointSaved   JournalEventType = "checkpoint_saved"
	JournalRoundAdvanced     JournalEventType = "round_advanced"
	JournalWorkflowSuspended JournalEventType = "workflow_suspended"
	JournalWorkflowResumed   JournalEventType = "workflow_resumed"
	JournalWorkflowCompleted JournalEventType = "workflow_completed"
	JournalWorkflowFailed    JournalEventType = "workflow_failed"
)

// RunStatus represents the persisted lifecycle state of a durable workflow run.
type RunStatus string

const (
	RunStatusPending   RunStatus = "pending"
	RunStatusRunning   RunStatus = "running"
	RunStatusSuspended RunStatus = "suspended"
	RunStatusCompleted RunStatus = "completed"
	RunStatusFailed    RunStatus = "failed"
)

// JournalEntry is a single immutable record in the execution journal.
type JournalEntry struct {
	Sequence         int64            `json:"sequence"`
	PreviousSequence int64            `json:"previous_sequence"`
	Type             JournalEventType `json:"type"`
	Org              string           `json:"org"`
	RunID            string           `json:"run_id"`
	Round            int              `json:"round"`
	StepID           string           `json:"step_id,omitempty"`
	IdempotencyKey   string           `json:"idempotency_key,omitempty"`
	InputHash        string           `json:"input_hash,omitempty"`
	Label            string           `json:"label,omitempty"`
	Payload          json.RawMessage  `json:"payload,omitempty"`
	Error            string           `json:"error,omitempty"`
	Timestamp        time.Time        `json:"timestamp"`
}

// ToWorkflowRunEvent converts a JournalEntry into control-plane's WorkflowRunEvent schema.
func (e JournalEntry) ToWorkflowRunEvent() types.WorkflowRunEvent {
	payload := e.Payload
	if len(payload) == 0 {
		payload = json.RawMessage(`{}`)
	}
	var statusReason *string
	if e.Error != "" {
		errCopy := e.Error
		statusReason = &errCopy
	}
	return types.WorkflowRunEvent{
		RunID:            e.RunID,
		Sequence:         e.Sequence,
		PreviousSequence: e.PreviousSequence,
		EventType:        string(e.Type),
		StatusReason:     statusReason,
		Payload:          payload,
		EmittedAt:        e.Timestamp,
		RecordedAt:       e.Timestamp,
	}
}

// CheckpointRecord stores serialized workflow state between event activations.
type CheckpointRecord struct {
	Org            string          `json:"org"`
	RunID          string          `json:"run_id"`
	Round          int             `json:"round"`
	Sequence       int64           `json:"sequence"`
	Label          string          `json:"label"`
	AwaitingSignal string          `json:"awaiting_signal,omitempty"`
	State          json.RawMessage `json:"state,omitempty"`
	CreatedAt      time.Time       `json:"created_at"`
}

// RunState holds the reconstructed durable workflow state for one (org, runID).
type RunState struct {
	Org              string                         `json:"org"`
	RunID            string                         `json:"run_id"`
	WorkflowName     string                         `json:"workflow_name"`
	Status           RunStatus                      `json:"status"`
	Round            int                            `json:"round"`
	Sequence         int64                          `json:"sequence"`
	AwaitingSignal   string                         `json:"awaiting_signal,omitempty"`
	Checkpoints      []CheckpointRecord             `json:"checkpoints,omitempty"`
	Entries          []JournalEntry                 `json:"entries,omitempty"`
	Ledger           map[string]ActivityLedgerEntry `json:"ledger,omitempty"`
	DeliveredSignals map[string]json.RawMessage     `json:"delivered_signals,omitempty"`
	CreatedAt        time.Time                      `json:"created_at"`
	UpdatedAt        time.Time                      `json:"updated_at"`
	CompletedAt      *time.Time                     `json:"completed_at,omitempty"`
	LastError        string                         `json:"last_error,omitempty"`
}

// ToWorkflowRun converts RunState into control-plane's canonical WorkflowRun record.
func (r *RunState) ToWorkflowRun() *types.WorkflowRun {
	if r == nil {
		return nil
	}
	metaBytes, _ := json.Marshal(map[string]any{
		"org":             r.Org,
		"round":           r.Round,
		"awaiting_signal": r.AwaitingSignal,
		"checkpoints":     len(r.Checkpoints),
	})
	completedSteps := 0
	failedSteps := 0
	for _, entry := range r.Ledger {
		switch entry.Status {
		case ActivityStatusCompleted:
			completedSteps++
		case ActivityStatusFailed:
			failedSteps++
		}
	}
	return &types.WorkflowRun{
		RunID:             r.RunID,
		RootWorkflowID:    r.WorkflowName,
		Status:            string(r.Status),
		TotalSteps:        completedSteps + failedSteps,
		CompletedSteps:    completedSteps,
		FailedSteps:       failedSteps,
		StateVersion:      r.Sequence,
		LastEventSequence: r.Sequence,
		Metadata:          metaBytes,
		CreatedAt:         r.CreatedAt,
		UpdatedAt:         r.UpdatedAt,
		CompletedAt:       r.CompletedAt,
	}
}

// clone creates a deep copy of RunState so callers cannot race on shared slices/maps.
func (r *RunState) clone() *RunState {
	if r == nil {
		return nil
	}
	cp := *r
	if r.CompletedAt != nil {
		t := *r.CompletedAt
		cp.CompletedAt = &t
	}
	cp.Checkpoints = make([]CheckpointRecord, len(r.Checkpoints))
	for i, c := range r.Checkpoints {
		c.State = append([]byte(nil), c.State...)
		cp.Checkpoints[i] = c
	}
	cp.Entries = make([]JournalEntry, len(r.Entries))
	for i, e := range r.Entries {
		e.Payload = append([]byte(nil), e.Payload...)
		cp.Entries[i] = e
	}
	cp.Ledger = make(map[string]ActivityLedgerEntry, len(r.Ledger))
	for k, v := range r.Ledger {
		v.Output = append([]byte(nil), v.Output...)
		cp.Ledger[k] = v
	}
	cp.DeliveredSignals = make(map[string]json.RawMessage, len(r.DeliveredSignals))
	for k, v := range r.DeliveredSignals {
		cp.DeliveredSignals[k] = append([]byte(nil), v...)
	}
	return &cp
}

func (r *RunState) applyEntry(entry JournalEntry) {
	r.Entries = append(r.Entries, entry)
	if entry.Sequence > r.Sequence {
		r.Sequence = entry.Sequence
	}
	if !entry.Timestamp.IsZero() {
		r.UpdatedAt = entry.Timestamp
	}
	switch entry.Type {
	case JournalWorkflowStarted:
		r.Status = RunStatusRunning
	case JournalActivityCompleted:
		if r.Ledger == nil {
			r.Ledger = make(map[string]ActivityLedgerEntry)
		}
		r.Ledger[entry.IdempotencyKey] = ActivityLedgerEntry{
			IdempotencyKey: entry.IdempotencyKey,
			Org:            entry.Org,
			RunID:          entry.RunID,
			Round:          entry.Round,
			StepID:         entry.StepID,
			Sequence:       entry.Sequence,
			Status:         ActivityStatusCompleted,
			InputHash:      entry.InputHash,
			Output:         append([]byte(nil), entry.Payload...),
			Attempt:        1,
			CompletedAt:    entry.Timestamp,
		}
	case JournalRoundAdvanced:
		r.Round = entry.Round
	case JournalCheckpointSaved:
		r.Checkpoints = append(r.Checkpoints, CheckpointRecord{
			Org:            entry.Org,
			RunID:          entry.RunID,
			Round:          entry.Round,
			Sequence:       entry.Sequence,
			Label:          entry.Label,
			AwaitingSignal: r.AwaitingSignal,
			State:          append([]byte(nil), entry.Payload...),
			CreatedAt:      entry.Timestamp,
		})
	case JournalWorkflowSuspended:
		r.Status = RunStatusSuspended
		r.AwaitingSignal = entry.Label
	case JournalWorkflowResumed:
		r.Status = RunStatusRunning
		if entry.Label != "" {
			if r.DeliveredSignals == nil {
				r.DeliveredSignals = make(map[string]json.RawMessage)
			}
			r.DeliveredSignals[entry.Label] = append([]byte(nil), entry.Payload...)
			if r.AwaitingSignal == entry.Label {
				r.AwaitingSignal = ""
			}
		}
	case JournalWorkflowCompleted:
		r.Status = RunStatusCompleted
		t := entry.Timestamp
		r.CompletedAt = &t
	case JournalWorkflowFailed:
		r.Status = RunStatusFailed
		r.LastError = entry.Error
	}
}

// JournalStore persists execution journals, checkpoints, and idempotency ledger records.
type JournalStore interface {
	LoadOrCreateRun(ctx context.Context, org, runID, workflowName string) (*RunState, error)
	GetRun(ctx context.Context, org, runID string) (*RunState, error)
	SaveRunState(ctx context.Context, state *RunState, newEntries []JournalEntry) error
	LookupActivity(ctx context.Context, org, idempotencyKey string) (*ActivityLedgerEntry, bool, error)
}

// MemoryJournalStore is an in-memory tenant-isolated JournalStore for unit testing.
type MemoryJournalStore struct {
	mu     sync.RWMutex
	runs   map[string]*RunState
	ledger map[string]map[string]ActivityLedgerEntry // org -> idempotency_key -> entry
}

// NewMemoryJournalStore constructs an empty MemoryJournalStore.
func NewMemoryJournalStore() *MemoryJournalStore {
	return &MemoryJournalStore{
		runs:   make(map[string]*RunState),
		ledger: make(map[string]map[string]ActivityLedgerEntry),
	}
}

func runMapKey(org, runID string) string {
	return org + "\x00" + runID
}

func (m *MemoryJournalStore) LoadOrCreateRun(ctx context.Context, org, runID, workflowName string) (*RunState, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	m.mu.Lock()
	defer m.mu.Unlock()

	k := runMapKey(org, runID)
	if existing, ok := m.runs[k]; ok {
		return existing.clone(), nil
	}

	now := time.Now().UTC()
	st := &RunState{
		Org:              org,
		RunID:            runID,
		WorkflowName:     workflowName,
		Status:           RunStatusPending,
		Ledger:           make(map[string]ActivityLedgerEntry),
		DeliveredSignals: make(map[string]json.RawMessage),
		CreatedAt:        now,
		UpdatedAt:        now,
	}
	m.runs[k] = st
	return st.clone(), nil
}

func (m *MemoryJournalStore) GetRun(ctx context.Context, org, runID string) (*RunState, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	st, ok := m.runs[runMapKey(org, runID)]
	if !ok {
		return nil, nil
	}
	return st.clone(), nil
}

func (m *MemoryJournalStore) SaveRunState(ctx context.Context, state *RunState, newEntries []JournalEntry) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if state == nil {
		return errors.New("durable: nil state")
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	cp := state.clone()
	if cp.Ledger == nil {
		cp.Ledger = make(map[string]ActivityLedgerEntry)
	}
	if m.ledger[cp.Org] == nil {
		m.ledger[cp.Org] = make(map[string]ActivityLedgerEntry)
	}
	for k, entry := range cp.Ledger {
		entry.Output = append([]byte(nil), entry.Output...)
		m.ledger[cp.Org][k] = entry
	}
	_ = newEntries
	m.runs[runMapKey(cp.Org, cp.RunID)] = cp
	return nil
}

func (m *MemoryJournalStore) LookupActivity(ctx context.Context, org, idempotencyKey string) (*ActivityLedgerEntry, bool, error) {
	if err := ctx.Err(); err != nil {
		return nil, false, err
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	orgLedger := m.ledger[org]
	if orgLedger == nil {
		return nil, false, nil
	}
	entry, ok := orgLedger[idempotencyKey]
	if !ok {
		return nil, false, nil
	}
	entry.Output = append([]byte(nil), entry.Output...)
	return &entry, true, nil
}

// FileJournalStore persists durable workflow journals, checkpoints, and activity
// deduplication ledgers to disk using fsync + atomic rename with zero resident daemons.
type FileJournalStore struct {
	rootDir string
	mu      sync.Mutex
}

// NewFileJournalStore initializes a disk-backed durable journal store at rootDir.
func NewFileJournalStore(rootDir string) (*FileJournalStore, error) {
	if strings.TrimSpace(rootDir) == "" {
		return nil, errors.New("durable: rootDir cannot be empty")
	}
	if err := os.MkdirAll(rootDir, 0o755); err != nil {
		return nil, fmt.Errorf("create journal root: %w", err)
	}
	return &FileJournalStore{rootDir: rootDir}, nil
}

func (f *FileJournalStore) orgDir(org string) string {
	return filepath.Join(f.rootDir, "orgs", stableDirHash(org))
}

func (f *FileJournalStore) runDir(org, runID string) string {
	return filepath.Join(f.orgDir(org), "runs", stableDirHash(runID))
}

func (f *FileJournalStore) ledgerPath(org, idempotencyKey string) string {
	return filepath.Join(f.orgDir(org), "ledger", idempotencyKey+".json")
}

func (f *FileJournalStore) LoadOrCreateRun(ctx context.Context, org, runID, workflowName string) (*RunState, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	f.mu.Lock()
	defer f.mu.Unlock()

	st, err := f.loadRunLocked(org, runID)
	if err != nil {
		return nil, err
	}
	if st != nil {
		return st, nil
	}

	now := time.Now().UTC()
	st = &RunState{
		Org:              org,
		RunID:            runID,
		WorkflowName:     workflowName,
		Status:           RunStatusPending,
		Ledger:           make(map[string]ActivityLedgerEntry),
		DeliveredSignals: make(map[string]json.RawMessage),
		CreatedAt:        now,
		UpdatedAt:        now,
	}
	if err := f.saveLocked(st, nil); err != nil {
		return nil, err
	}
	return st.clone(), nil
}

func (f *FileJournalStore) GetRun(ctx context.Context, org, runID string) (*RunState, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.loadRunLocked(org, runID)
}

func (f *FileJournalStore) SaveRunState(ctx context.Context, state *RunState, newEntries []JournalEntry) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if state == nil {
		return errors.New("durable: nil state")
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.saveLocked(state, newEntries)
}

func (f *FileJournalStore) LookupActivity(ctx context.Context, org, idempotencyKey string) (*ActivityLedgerEntry, bool, error) {
	if err := ctx.Err(); err != nil {
		return nil, false, err
	}
	f.mu.Lock()
	defer f.mu.Unlock()

	data, err := os.ReadFile(f.ledgerPath(org, idempotencyKey))
	if err != nil {
		if os.IsNotExist(err) {
			return nil, false, nil
		}
		return nil, false, fmt.Errorf("read ledger entry: %w", err)
	}
	var entry ActivityLedgerEntry
	if err := json.Unmarshal(data, &entry); err != nil {
		return nil, false, fmt.Errorf("decode ledger entry: %w", err)
	}
	return &entry, true, nil
}

func (f *FileJournalStore) loadRunLocked(org, runID string) (*RunState, error) {
	rDir := f.runDir(org, runID)
	statePath := filepath.Join(rDir, "state.json")
	journalPath := filepath.Join(rDir, "journal.jsonl")

	var st *RunState
	if data, err := os.ReadFile(statePath); err == nil {
		var parsed RunState
		if err := json.Unmarshal(data, &parsed); err != nil {
			return nil, fmt.Errorf("decode run state: %w", err)
		}
		if parsed.Org != org || parsed.RunID != runID {
			return nil, errors.New("durable: tenant/run state identity mismatch")
		}
		st = &parsed
	} else if !os.IsNotExist(err) {
		return nil, fmt.Errorf("read run state: %w", err)
	}

	journalEntries, err := readJournalLines(journalPath)
	if err != nil {
		return nil, err
	}
	if st == nil {
		if len(journalEntries) == 0 {
			return nil, nil
		}
		st = &RunState{
			Org:              org,
			RunID:            runID,
			Status:           RunStatusPending,
			Ledger:           make(map[string]ActivityLedgerEntry),
			DeliveredSignals: make(map[string]json.RawMessage),
			CreatedAt:        journalEntries[0].Timestamp,
			UpdatedAt:        journalEntries[0].Timestamp,
		}
	}
	if st.Ledger == nil {
		st.Ledger = make(map[string]ActivityLedgerEntry)
	}
	if st.DeliveredSignals == nil {
		st.DeliveredSignals = make(map[string]json.RawMessage)
	}

	// Recover any journal tail committed before state.json checkpoint write.
	for _, entry := range journalEntries {
		if entry.Sequence > st.Sequence {
			st.applyEntry(entry)
		}
	}
	return st.clone(), nil
}

func (f *FileJournalStore) saveLocked(state *RunState, newEntries []JournalEntry) error {
	rDir := f.runDir(state.Org, state.RunID)
	lDir := filepath.Join(f.orgDir(state.Org), "ledger")
	if err := os.MkdirAll(rDir, 0o755); err != nil {
		return fmt.Errorf("create run dir: %w", err)
	}
	if err := os.MkdirAll(lDir, 0o755); err != nil {
		return fmt.Errorf("create ledger dir: %w", err)
	}

	for _, entry := range newEntries {
		if entry.Type == JournalActivityCompleted {
			if rec, ok := state.Ledger[entry.IdempotencyKey]; ok {
				raw, err := json.Marshal(rec)
				if err != nil {
					return fmt.Errorf("marshal ledger entry: %w", err)
				}
				if err := writeFileAtomic(f.ledgerPath(state.Org, entry.IdempotencyKey), raw); err != nil {
					return fmt.Errorf("persist ledger entry: %w", err)
				}
			}
		}
	}

	if len(newEntries) > 0 {
		journalPath := filepath.Join(rDir, "journal.jsonl")
		jf, err := os.OpenFile(journalPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		if err != nil {
			return fmt.Errorf("open journal file: %w", err)
		}
		for _, entry := range newEntries {
			line, err := json.Marshal(entry)
			if err != nil {
				_ = jf.Close()
				return fmt.Errorf("marshal journal entry: %w", err)
			}
			if _, err := jf.Write(append(line, '\n')); err != nil {
				_ = jf.Close()
				return fmt.Errorf("append journal entry: %w", err)
			}
		}
		if err := jf.Sync(); err != nil {
			_ = jf.Close()
			return fmt.Errorf("sync journal file: %w", err)
		}
		if err := jf.Close(); err != nil {
			return fmt.Errorf("close journal file: %w", err)
		}
	}

	stateBytes, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal run state: %w", err)
	}
	return writeFileAtomic(filepath.Join(rDir, "state.json"), stateBytes)
}

func readJournalLines(path string) ([]JournalEntry, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("open journal: %w", err)
	}
	defer f.Close()

	var entries []JournalEntry
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var entry JournalEntry
		if err := json.Unmarshal([]byte(line), &entry); err != nil {
			return nil, fmt.Errorf("decode journal line: %w", err)
		}
		entries = append(entries, entry)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan journal: %w", err)
	}
	return entries, nil
}

func writeFileAtomic(targetPath string, data []byte) error {
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
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmpName, targetPath); err != nil {
		return err
	}
	cleanup = false
	return nil
}

func stableDirHash(val string) string {
	sum := sha256.Sum256([]byte(val))
	return hex.EncodeToString(sum[:16])
}
