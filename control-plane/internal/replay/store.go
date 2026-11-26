package replay

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
)

const (
	// DefaultMaxPayloadBytes bounds any single activity payload or checkpoint state (1 MiB).
	DefaultMaxPayloadBytes = 1 << 20
	// DefaultMaxHistoryDepth bounds total events per workflow against memory exhaustion.
	DefaultMaxHistoryDepth = 10_000
)

// Limits enforces security and memory-exhaustion bounds on workflow persistence.
type Limits struct {
	MaxPayloadBytes int
	MaxHistoryDepth int64
}

// EffectiveLimits returns l with safe defaults populated for zero values.
func (l Limits) EffectiveLimits() Limits {
	if l.MaxPayloadBytes <= 0 {
		l.MaxPayloadBytes = DefaultMaxPayloadBytes
	}
	if l.MaxHistoryDepth <= 0 {
		l.MaxHistoryDepth = DefaultMaxHistoryDepth
	}
	return l
}

// LoadOptions controls whether recovery uses the latest persisted checkpoint +
// delta event tail or replays the complete uncompacted history from event 1.
type LoadOptions struct {
	UseCheckpoint bool
}

// WorkflowLoadResult contains the persisted state loaded for one workflow activation.
type WorkflowLoadResult struct {
	Meta              *WorkflowMeta
	Checkpoint        *Checkpoint
	Events            []HistoryEvent
	FromCheckpoint    bool
	DeltaEventsLoaded int
	TotalBytesRead    int64
	LogByteOffset     int64
}

// CheckpointStore is the pluggable tenant-scoped persistence boundary for
// append-only event histories, atomic checkpoints, and idempotency ledgers.
type CheckpointStore interface {
	LoadWorkflow(ctx context.Context, tenantID, workflowID string, opts LoadOptions) (*WorkflowLoadResult, error)
	AppendEvents(ctx context.Context, tenantID, workflowID string, meta *WorkflowMeta, newEvents []HistoryEvent) (logByteOffset int64, err error)
	SaveCheckpoint(ctx context.Context, tenantID, workflowID string, cp *Checkpoint) error
	LookupIdempotency(ctx context.Context, tenantID, workflowID, idempotencyKey string) (*IdempotencyRecord, bool, error)
	RecordIdempotency(ctx context.Context, tenantID, workflowID string, rec IdempotencyRecord) error
	StorageBytes(ctx context.Context, tenantID, workflowID string) (int64, error)
}

func validateTenantWorkflow(tenantID, workflowID string) error {
	if strings.TrimSpace(tenantID) == "" {
		return fmt.Errorf("%w: tenant_id is required", ErrInvalidTenant)
	}
	if strings.TrimSpace(workflowID) == "" {
		return fmt.Errorf("%w: workflow_id is required", ErrInvalidWorkflowID)
	}
	return nil
}

func workflowScopeKey(tenantID, workflowID string) string {
	return fmt.Sprintf("%d:%s|%d:%s", len(tenantID), tenantID, len(workflowID), workflowID)
}

type memWorkflowBucket struct {
	meta           *WorkflowMeta
	checkpoint     *Checkpoint
	checkpointSize int64
	events         []HistoryEvent
	eventByteEnds  []int64 // cumulative serialized JSONL byte offsets
	ledger         map[string]IdempotencyRecord
	ledgerBytes    int64
}

// MemoryCheckpointStore is a thread-safe, tenant-isolated in-memory CheckpointStore
// with deep-copy isolation and accurate serialized byte accounting.
type MemoryCheckpointStore struct {
	mu      sync.RWMutex
	limits  Limits
	buckets map[string]*memWorkflowBucket
}

// NewMemoryCheckpointStore constructs a MemoryCheckpointStore with the given limits.
func NewMemoryCheckpointStore(limits Limits) *MemoryCheckpointStore {
	return &MemoryCheckpointStore{
		limits:  limits.EffectiveLimits(),
		buckets: make(map[string]*memWorkflowBucket),
	}
}

func (m *MemoryCheckpointStore) LoadWorkflow(ctx context.Context, tenantID, workflowID string, opts LoadOptions) (*WorkflowLoadResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := validateTenantWorkflow(tenantID, workflowID); err != nil {
		return nil, err
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	bucket, ok := m.buckets[workflowScopeKey(tenantID, workflowID)]
	if !ok {
		return nil, nil
	}
	if bucket.meta != nil && bucket.meta.TenantID != tenantID {
		return nil, ErrTenantAccessDenied
	}

	var totalOffset int64
	if len(bucket.eventByteEnds) > 0 {
		totalOffset = bucket.eventByteEnds[len(bucket.eventByteEnds)-1]
	}

	if opts.UseCheckpoint && bucket.checkpoint != nil {
		cp := bucket.checkpoint.clone()
		if cp.TenantID != tenantID {
			return nil, ErrTenantAccessDenied
		}
		var delta []HistoryEvent
		startIdx := len(bucket.events)
		for i, ev := range bucket.events {
			if ev.Sequence > cp.Sequence {
				startIdx = i
				break
			}
		}
		var deltaBytes int64
		if startIdx < len(bucket.events) {
			delta = make([]HistoryEvent, 0, len(bucket.events)-startIdx)
			for i := startIdx; i < len(bucket.events); i++ {
				delta = append(delta, bucket.events[i].clone())
			}
			var prevEnd int64
			if startIdx > 0 {
				prevEnd = bucket.eventByteEnds[startIdx-1]
			}
			deltaBytes = totalOffset - prevEnd
		}
		if _, err := VerifyHistoryIntegrity(tenantID, workflowID, delta, cp.Sequence+1, cp.HistoryDigest); err != nil {
			return nil, err
		}
		return &WorkflowLoadResult{
			Meta:              bucket.meta.clone(),
			Checkpoint:        cp,
			Events:            delta,
			FromCheckpoint:    true,
			DeltaEventsLoaded: len(delta),
			TotalBytesRead:    bucket.checkpointSize + deltaBytes,
			LogByteOffset:     totalOffset,
		}, nil
	}

	events := make([]HistoryEvent, len(bucket.events))
	for i, ev := range bucket.events {
		events[i] = ev.clone()
	}
	if _, err := VerifyHistoryIntegrity(tenantID, workflowID, events, 1, GenesisDigest); err != nil {
		return nil, err
	}
	return &WorkflowLoadResult{
		Meta:              bucket.meta.clone(),
		Checkpoint:        nil,
		Events:            events,
		FromCheckpoint:    false,
		DeltaEventsLoaded: len(events),
		TotalBytesRead:    totalOffset,
		LogByteOffset:     totalOffset,
	}, nil
}

func (m *MemoryCheckpointStore) AppendEvents(ctx context.Context, tenantID, workflowID string, meta *WorkflowMeta, newEvents []HistoryEvent) (int64, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	if err := validateTenantWorkflow(tenantID, workflowID); err != nil {
		return 0, err
	}
	if meta == nil || meta.TenantID != tenantID || meta.WorkflowID != workflowID {
		return 0, ErrTenantAccessDenied
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	k := workflowScopeKey(tenantID, workflowID)
	bucket, ok := m.buckets[k]
	if !ok {
		bucket = &memWorkflowBucket{
			ledger: make(map[string]IdempotencyRecord),
		}
		m.buckets[k] = bucket
	}

	if int64(len(bucket.events)+len(newEvents)) > m.limits.MaxHistoryDepth {
		return 0, fmt.Errorf("%w: depth %d > limit %d", ErrHistoryDepthExceeded, len(bucket.events)+len(newEvents), m.limits.MaxHistoryDepth)
	}

	var curOffset int64
	if len(bucket.eventByteEnds) > 0 {
		curOffset = bucket.eventByteEnds[len(bucket.eventByteEnds)-1]
	}

	startSeq := int64(len(bucket.events) + 1)
	prevDigest := GenesisDigest
	if len(bucket.events) > 0 {
		prevDigest = bucket.events[len(bucket.events)-1].Digest
	}
	if _, err := VerifyHistoryIntegrity(tenantID, workflowID, newEvents, startSeq, prevDigest); err != nil {
		return 0, err
	}

	for _, ev := range newEvents {
		if len(ev.Payload) > m.limits.MaxPayloadBytes {
			return 0, fmt.Errorf("%w: event seq %d payload %d bytes > limit %d", ErrPayloadTooLarge, ev.Sequence, len(ev.Payload), m.limits.MaxPayloadBytes)
		}
		raw, err := json.Marshal(ev)
		if err != nil {
			return 0, err
		}
		curOffset += int64(len(raw) + 1)
		bucket.events = append(bucket.events, ev.clone())
		bucket.eventByteEnds = append(bucket.eventByteEnds, curOffset)
	}
	bucket.meta = meta.clone()
	return curOffset, nil
}

func (m *MemoryCheckpointStore) SaveCheckpoint(ctx context.Context, tenantID, workflowID string, cp *Checkpoint) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := validateTenantWorkflow(tenantID, workflowID); err != nil {
		return err
	}
	if cp == nil || cp.TenantID != tenantID || cp.WorkflowID != workflowID {
		return ErrTenantAccessDenied
	}
	if len(cp.UserState) > m.limits.MaxPayloadBytes {
		return fmt.Errorf("%w: checkpoint payload %d bytes > limit %d", ErrPayloadTooLarge, len(cp.UserState), m.limits.MaxPayloadBytes)
	}
	raw, err := json.Marshal(cp)
	if err != nil {
		return err
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	k := workflowScopeKey(tenantID, workflowID)
	bucket, ok := m.buckets[k]
	if !ok {
		bucket = &memWorkflowBucket{
			ledger: make(map[string]IdempotencyRecord),
		}
		m.buckets[k] = bucket
	}
	bucket.checkpoint = cp.clone()
	bucket.checkpointSize = int64(len(raw))
	return nil
}

func (m *MemoryCheckpointStore) LookupIdempotency(ctx context.Context, tenantID, workflowID, idempotencyKey string) (*IdempotencyRecord, bool, error) {
	if err := ctx.Err(); err != nil {
		return nil, false, err
	}
	scoped, err := ScopedLedgerKey(tenantID, workflowID, idempotencyKey)
	if err != nil {
		return nil, false, err
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	bucket, ok := m.buckets[workflowScopeKey(tenantID, workflowID)]
	if !ok {
		return nil, false, nil
	}
	rec, found := bucket.ledger[scoped]
	if !found {
		return nil, false, nil
	}
	if rec.TenantID != tenantID || rec.WorkflowID != workflowID {
		return nil, false, ErrTenantAccessDenied
	}
	cp := rec.clone()
	return &cp, true, nil
}

func (m *MemoryCheckpointStore) RecordIdempotency(ctx context.Context, tenantID, workflowID string, rec IdempotencyRecord) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if rec.TenantID != tenantID || rec.WorkflowID != workflowID {
		return ErrTenantAccessDenied
	}
	if len(rec.Output) > m.limits.MaxPayloadBytes {
		return fmt.Errorf("%w: idempotency output %d bytes > limit %d", ErrPayloadTooLarge, len(rec.Output), m.limits.MaxPayloadBytes)
	}
	scoped, err := ScopedLedgerKey(tenantID, workflowID, rec.IdempotencyKey)
	if err != nil {
		return err
	}
	raw, err := json.Marshal(rec)
	if err != nil {
		return err
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	k := workflowScopeKey(tenantID, workflowID)
	bucket, ok := m.buckets[k]
	if !ok {
		bucket = &memWorkflowBucket{
			ledger: make(map[string]IdempotencyRecord),
		}
		m.buckets[k] = bucket
	}
	if _, exists := bucket.ledger[scoped]; !exists {
		bucket.ledgerBytes += int64(len(raw))
	}
	bucket.ledger[scoped] = rec.clone()
	return nil
}

func (m *MemoryCheckpointStore) StorageBytes(ctx context.Context, tenantID, workflowID string) (int64, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	if err := validateTenantWorkflow(tenantID, workflowID); err != nil {
		return 0, err
	}
	m.mu.RLock()
	defer m.mu.RUnlock()

	bucket, ok := m.buckets[workflowScopeKey(tenantID, workflowID)]
	if !ok {
		return 0, nil
	}
	var logBytes int64
	if len(bucket.eventByteEnds) > 0 {
		logBytes = bucket.eventByteEnds[len(bucket.eventByteEnds)-1]
	}
	var metaBytes int64
	if bucket.meta != nil {
		if b, err := json.Marshal(bucket.meta); err == nil {
			metaBytes = int64(len(b))
		}
	}
	return logBytes + bucket.checkpointSize + bucket.ledgerBytes + metaBytes, nil
}
