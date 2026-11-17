// Package membench evaluates Hanzo Agents memory policies on LongMemEval
// scenarios (HIP-1346 / #82) without introducing a competing memory store,
// and implements an edge-salience evidence packer and position-sweep benchmark
// for Lost-in-the-Middle mitigation (#83).
package membench

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"time"

	"github.com/hanzoai/agents/control-plane/pkg/types"
)

// TaskCategory identifies the LongMemEval reasoning and lifecycle dimension.
type TaskCategory string

const (
	CategoryTemporalUpdate   TaskCategory = "temporal_update"
	CategoryContradiction    TaskCategory = "contradiction"
	CategoryTemporalOrdering TaskCategory = "temporal_ordering"
	CategoryAbstention       TaskCategory = "abstention"
	CategoryCrossSession     TaskCategory = "cross_session"
)

// AllCategories lists all evaluated LongMemEval categories in canonical order.
var AllCategories = []TaskCategory{
	CategoryTemporalUpdate,
	CategoryContradiction,
	CategoryTemporalOrdering,
	CategoryAbstention,
	CategoryCrossSession,
}

// MemoryBackend matches the memory and event primitives of control-plane's
// storage.StorageProvider so existing memory policies are evaluated directly
// on Hanzo Agents' canonical memory primitives without a competing memory store.
type MemoryBackend interface {
	SetMemory(ctx context.Context, memory *types.Memory) error
	GetMemory(ctx context.Context, scope, scopeID, key string) (*types.Memory, error)
	DeleteMemory(ctx context.Context, scope, scopeID, key string) error
	ListMemory(ctx context.Context, scope, scopeID string) ([]*types.Memory, error)
	StoreEvent(ctx context.Context, event *types.MemoryChangeEvent) error
	GetEventHistory(ctx context.Context, filter types.EventFilter) ([]*types.MemoryChangeEvent, error)
}

// InMemoryBackend is a lightweight test adapter implementing MemoryBackend
// with the exact BoltDB+event semantics of LocalStorage for fast isolated runs.
type InMemoryBackend struct {
	mu     sync.RWMutex
	kv     map[string]*types.Memory // scope\x00scopeID\x00key -> Memory
	events []*types.MemoryChangeEvent
}

// NewInMemoryBackend initializes a clean evaluation backend over types.Memory.
func NewInMemoryBackend() *InMemoryBackend {
	return &InMemoryBackend{
		kv: make(map[string]*types.Memory),
	}
}

func memCompositeKey(scope, scopeID, key string) string {
	return scope + "\x00" + scopeID + "\x00" + key
}

func (b *InMemoryBackend) SetMemory(ctx context.Context, memory *types.Memory) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	cp := *memory
	cp.Data = append(json.RawMessage(nil), memory.Data...)
	b.kv[memCompositeKey(memory.Scope, memory.ScopeID, memory.Key)] = &cp
	return nil
}

func (b *InMemoryBackend) GetMemory(ctx context.Context, scope, scopeID, key string) (*types.Memory, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	b.mu.RLock()
	defer b.mu.RUnlock()
	mem, ok := b.kv[memCompositeKey(scope, scopeID, key)]
	if !ok {
		return nil, errMemoryNotFound
	}
	cp := *mem
	cp.Data = append(json.RawMessage(nil), mem.Data...)
	return &cp, nil
}

func (b *InMemoryBackend) DeleteMemory(ctx context.Context, scope, scopeID, key string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	k := memCompositeKey(scope, scopeID, key)
	if _, ok := b.kv[k]; !ok {
		return errMemoryNotFound
	}
	delete(b.kv, k)
	return nil
}

func (b *InMemoryBackend) ListMemory(ctx context.Context, scope, scopeID string) ([]*types.Memory, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	b.mu.RLock()
	defer b.mu.RUnlock()
	prefix := scope + "\x00" + scopeID + "\x00"
	var out []*types.Memory
	for k, mem := range b.kv {
		if strings.HasPrefix(k, prefix) {
			cp := *mem
			cp.Data = append(json.RawMessage(nil), mem.Data...)
			out = append(out, &cp)
		}
	}
	return out, nil
}

func (b *InMemoryBackend) StoreEvent(ctx context.Context, event *types.MemoryChangeEvent) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	cp := *event
	if cp.Timestamp.IsZero() {
		cp.Timestamp = time.Now().UTC()
	}
	cp.Data = append(json.RawMessage(nil), event.Data...)
	cp.PreviousData = append(json.RawMessage(nil), event.PreviousData...)
	b.events = append(b.events, &cp)
	return nil
}

func (b *InMemoryBackend) GetEventHistory(ctx context.Context, filter types.EventFilter) ([]*types.MemoryChangeEvent, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	b.mu.RLock()
	defer b.mu.RUnlock()
	var out []*types.MemoryChangeEvent
	for _, ev := range b.events {
		if filter.Scope != nil && ev.Scope != *filter.Scope {
			continue
		}
		if filter.ScopeID != nil && ev.ScopeID != *filter.ScopeID {
			continue
		}
		if filter.Since != nil && ev.Timestamp.Before(*filter.Since) {
			continue
		}
		cp := *ev
		cp.Data = append(json.RawMessage(nil), ev.Data...)
		out = append(out, &cp)
		if filter.Limit > 0 && len(out) >= filter.Limit {
			break
		}
	}
	return out, nil
}

var errMemoryNotFound = &notFoundError{}

type notFoundError struct{}

func (e *notFoundError) Error() string { return "memory entry not found" }

// estimateTokens provides a fast deterministic token estimate when callers do not supply a tokenizer.
func estimateTokens(text string) int {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return 0
	}
	words := len(strings.Fields(trimmed))
	charEst := (len(trimmed) + 3) / 4
	if words > charEst {
		return words
	}
	return charEst
}
