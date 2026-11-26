package longmemeval

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/hanzoai/agents/control-plane/pkg/types"
)

// MemoryStore is the exact canonical memory, vector, and event subset of
// control-plane/internal/storage.StorageProvider. Both storage.LocalStorage
// (BoltDB + SQLite) and InMemoryStore satisfy this interface directly so
// policies are evaluated over Hanzo Agents' canonical primitives without
// introducing a competing memory engine.
type MemoryStore interface {
	SetMemory(ctx context.Context, memory *types.Memory) error
	GetMemory(ctx context.Context, scope, scopeID, key string) (*types.Memory, error)
	DeleteMemory(ctx context.Context, scope, scopeID, key string) error
	ListMemory(ctx context.Context, scope, scopeID string) ([]*types.Memory, error)

	SetVector(ctx context.Context, record *types.VectorRecord) error
	GetVector(ctx context.Context, scope, scopeID, key string) (*types.VectorRecord, error)
	DeleteVector(ctx context.Context, scope, scopeID, key string) error
	DeleteVectorsByPrefix(ctx context.Context, scope, scopeID, prefix string) (int, error)
	SimilaritySearch(ctx context.Context, scope, scopeID string, queryEmbedding []float32, topK int, filters map[string]interface{}) ([]*types.VectorSearchResult, error)

	StoreEvent(ctx context.Context, event *types.MemoryChangeEvent) error
	GetEventHistory(ctx context.Context, filter types.EventFilter) ([]*types.MemoryChangeEvent, error)
}

// StoreFactory constructs a fresh isolated MemoryStore and an optional cleanup hook.
type StoreFactory func() (MemoryStore, func(), error)

// ErrNotFound signals a missing KV entry, mirroring LocalStorage.GetMemory behavior.
var ErrNotFound = errors.New("memory entry not found")

// InMemoryStore is a concurrency-safe, deep-copying reference implementation of
// MemoryStore that reproduces the exact KV, cosine-vector, and event-history
// semantics of control-plane/internal/storage.LocalStorage for fast, isolated,
// deterministic sweeps up to 200+ sessions.
type InMemoryStore struct {
	mu      sync.RWMutex
	kv      map[string]*types.Memory
	vectors map[string]*types.VectorRecord
	events  []*types.MemoryChangeEvent
}

// NewInMemoryStore allocates an empty canonical MemoryStore.
func NewInMemoryStore() *InMemoryStore {
	return &InMemoryStore{
		kv:      make(map[string]*types.Memory),
		vectors: make(map[string]*types.VectorRecord),
	}
}

func compositeKey(scope, scopeID, key string) string {
	return scope + "\x00" + scopeID + "\x00" + key
}

func compositePrefix(scope, scopeID string) string {
	return scope + "\x00" + scopeID + "\x00"
}

func cloneMemory(m *types.Memory) *types.Memory {
	if m == nil {
		return nil
	}
	cp := *m
	cp.Data = append(json.RawMessage(nil), m.Data...)
	if m.Metadata.Custom != nil {
		custom := make(map[string]interface{}, len(m.Metadata.Custom))
		for k, v := range m.Metadata.Custom {
			custom[k] = v
		}
		cp.Metadata.Custom = custom
	}
	return &cp
}

func cloneMetadata(meta map[string]interface{}) map[string]interface{} {
	if meta == nil {
		return map[string]interface{}{}
	}
	cp := make(map[string]interface{}, len(meta))
	for k, v := range meta {
		cp[k] = v
	}
	return cp
}

func cloneVector(v *types.VectorRecord) *types.VectorRecord {
	if v == nil {
		return nil
	}
	cp := *v
	cp.Embedding = append([]float32(nil), v.Embedding...)
	cp.Metadata = cloneMetadata(v.Metadata)
	return &cp
}

func (s *InMemoryStore) SetMemory(ctx context.Context, memory *types.Memory) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if memory == nil || memory.Scope == "" || memory.ScopeID == "" || memory.Key == "" {
		return errors.New("scope, scope_id, and key are required")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	cp := cloneMemory(memory)
	if cp.CreatedAt.IsZero() {
		cp.CreatedAt = time.Now().UTC()
	}
	if cp.UpdatedAt.IsZero() {
		cp.UpdatedAt = cp.CreatedAt
	}
	s.kv[compositeKey(cp.Scope, cp.ScopeID, cp.Key)] = cp
	return nil
}

func (s *InMemoryStore) GetMemory(ctx context.Context, scope, scopeID, key string) (*types.Memory, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	m, ok := s.kv[compositeKey(scope, scopeID, key)]
	if !ok {
		return nil, ErrNotFound
	}
	return cloneMemory(m), nil
}

func (s *InMemoryStore) DeleteMemory(ctx context.Context, scope, scopeID, key string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	k := compositeKey(scope, scopeID, key)
	if _, ok := s.kv[k]; !ok {
		return ErrNotFound
	}
	delete(s.kv, k)
	return nil
}

func (s *InMemoryStore) ListMemory(ctx context.Context, scope, scopeID string) ([]*types.Memory, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	prefix := compositePrefix(scope, scopeID)
	out := make([]*types.Memory, 0)
	for k, m := range s.kv {
		if strings.HasPrefix(k, prefix) {
			out = append(out, cloneMemory(m))
		}
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].Key < out[j].Key
	})
	return out, nil
}

func (s *InMemoryStore) SetVector(ctx context.Context, record *types.VectorRecord) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if record == nil || record.Scope == "" || record.ScopeID == "" || record.Key == "" {
		return errors.New("scope, scope_id, and key are required")
	}
	if len(record.Embedding) == 0 {
		return errors.New("embedding cannot be empty")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	cp := cloneVector(record)
	if cp.CreatedAt.IsZero() {
		cp.CreatedAt = time.Now().UTC()
	}
	if cp.UpdatedAt.IsZero() {
		cp.UpdatedAt = cp.CreatedAt
	}
	s.vectors[compositeKey(cp.Scope, cp.ScopeID, cp.Key)] = cp
	return nil
}

func (s *InMemoryStore) GetVector(ctx context.Context, scope, scopeID, key string) (*types.VectorRecord, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	v, ok := s.vectors[compositeKey(scope, scopeID, key)]
	if !ok {
		return nil, nil
	}
	return cloneVector(v), nil
}

func (s *InMemoryStore) DeleteVector(ctx context.Context, scope, scopeID, key string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.vectors, compositeKey(scope, scopeID, key))
	return nil
}

func (s *InMemoryStore) DeleteVectorsByPrefix(ctx context.Context, scope, scopeID, prefix string) (int, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	fullPrefix := compositePrefix(scope, scopeID) + prefix
	deleted := 0
	for k := range s.vectors {
		if strings.HasPrefix(k, fullPrefix) {
			delete(s.vectors, k)
			deleted++
		}
	}
	return deleted, nil
}

func (s *InMemoryStore) SimilaritySearch(ctx context.Context, scope, scopeID string, queryEmbedding []float32, topK int, filters map[string]interface{}) ([]*types.VectorSearchResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(queryEmbedding) == 0 {
		return nil, errors.New("query embedding cannot be empty")
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	prefix := compositePrefix(scope, scopeID)
	results := make([]*types.VectorSearchResult, 0)
	for k, rec := range s.vectors {
		if !strings.HasPrefix(k, prefix) {
			continue
		}
		if len(rec.Embedding) != len(queryEmbedding) {
			continue
		}
		if !metadataMatches(rec.Metadata, filters) {
			continue
		}
		cos := cosineSim32(queryEmbedding, rec.Embedding)
		results = append(results, &types.VectorSearchResult{
			Scope:     rec.Scope,
			ScopeID:   rec.ScopeID,
			Key:       rec.Key,
			Score:     cos,
			Distance:  1.0 - cos,
			Metadata:  cloneMetadata(rec.Metadata),
			CreatedAt: rec.CreatedAt,
			UpdatedAt: rec.UpdatedAt,
		})
	}
	sort.Slice(results, func(i, j int) bool {
		if results[i].Score == results[j].Score {
			return results[i].Key < results[j].Key
		}
		return results[i].Score > results[j].Score
	})
	if topK > 0 && len(results) > topK {
		results = results[:topK]
	}
	return results, nil
}

func (s *InMemoryStore) StoreEvent(ctx context.Context, event *types.MemoryChangeEvent) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if event == nil {
		return errors.New("event cannot be nil")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	cp := *event
	if cp.Timestamp.IsZero() {
		cp.Timestamp = time.Now().UTC()
	}
	cp.Data = append(json.RawMessage(nil), event.Data...)
	cp.PreviousData = append(json.RawMessage(nil), event.PreviousData...)
	s.events = append(s.events, &cp)
	return nil
}

func (s *InMemoryStore) GetEventHistory(ctx context.Context, filter types.EventFilter) ([]*types.MemoryChangeEvent, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]*types.MemoryChangeEvent, 0)
	for _, ev := range s.events {
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
		cp.PreviousData = append(json.RawMessage(nil), ev.PreviousData...)
		out = append(out, &cp)
		if filter.Limit > 0 && len(out) >= filter.Limit {
			break
		}
	}
	return out, nil
}

func metadataMatches(meta map[string]interface{}, filters map[string]interface{}) bool {
	if len(filters) == 0 {
		return true
	}
	for k, expected := range filters {
		actual, ok := meta[k]
		if !ok {
			return false
		}
		if fmt.Sprintf("%v", actual) != fmt.Sprintf("%v", expected) {
			return false
		}
	}
	return true
}

func cosineSim32(a, b []float32) float64 {
	var dot, normA, normB float64
	for i := range a {
		fa := float64(a[i])
		fb := float64(b[i])
		dot += fa * fb
		normA += fa * fa
		normB += fb * fb
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

// Embedder maps text into a deterministic dense float32 vector for canonical
// MemoryStore.SetVector and SimilaritySearch evaluation. Production deployments
// swap in Enso-routed neural embeddings without altering policy contracts.
type Embedder interface {
	Embed(text string) []float32
	Dims() int
}

// HashEmbedder is a zero-dependency, Apache-2.0 deterministic SHA-256 feature
// hasher over normalized unigrams, character trigrams (for morphological
// robustness), and word bigrams. It produces unit-norm float32 embeddings.
type HashEmbedder struct {
	dims int
}

// NewHashEmbedder creates a deterministic feature-hashing embedder (default 256 dims).
func NewHashEmbedder(dims int) *HashEmbedder {
	if dims <= 0 {
		dims = 256
	}
	return &HashEmbedder{dims: dims}
}

func (e *HashEmbedder) Dims() int { return e.dims }

var stopWords = map[string]struct{}{
	"the": {}, "a": {}, "an": {}, "is": {}, "are": {}, "was": {}, "were": {},
	"be": {}, "been": {}, "to": {}, "of": {}, "in": {}, "on": {}, "at": {},
	"for": {}, "with": {}, "by": {}, "from": {}, "that": {}, "this": {},
	"it": {}, "as": {}, "or": {}, "and": {}, "we": {}, "our": {}, "my": {},
	"what": {}, "which": {}, "where": {}, "who": {}, "when": {}, "how": {},
	"did": {}, "does": {}, "do": {}, "has": {}, "have": {}, "had": {},
	"currently": {}, "now": {}, "right": {}, "tell": {}, "me": {},
}

func tokenize(text string) []string {
	lower := strings.ToLower(text)
	var tokens []string
	var cur strings.Builder
	flush := func() {
		if cur.Len() > 0 {
			tok := cur.String()
			if len(tok) >= 2 {
				if _, isStop := stopWords[tok]; !isStop {
					tokens = append(tokens, tok)
				}
			}
			cur.Reset()
		}
	}
	for _, r := range lower {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' || r == '-' {
			cur.WriteRune(r)
		} else {
			flush()
		}
	}
	flush()
	return tokens
}

func hashFeature(feature string, dims int) (int, float64) {
	sum := sha256.Sum256([]byte(feature))
	idx := int(binary.LittleEndian.Uint64(sum[:8]) % uint64(dims))
	sign := 1.0
	if sum[8]&1 == 1 {
		sign = -1.0
	}
	return idx, sign
}

func (e *HashEmbedder) Embed(text string) []float32 {
	vec := make([]float64, e.dims)
	tokens := tokenize(text)
	for i, tok := range tokens {
		// Unigram feature (primary semantic weight).
		idx, sign := hashFeature("u:"+tok, e.dims)
		vec[idx] += sign * 1.5

		// Stem-like prefix for morphological matching (e.g. migrate/migrated/migration).
		if len(tok) >= 5 {
			pIdx, pSign := hashFeature("p:"+tok[:5], e.dims)
			vec[pIdx] += pSign * 0.75
		}

		// Adjacent bigram feature.
		if i+1 < len(tokens) {
			bIdx, bSign := hashFeature("b:"+tok+"_"+tokens[i+1], e.dims)
			vec[bIdx] += bSign * 1.0
		}
	}

	var normSq float64
	for _, v := range vec {
		normSq += v * v
	}
	out := make([]float32, e.dims)
	if normSq == 0 {
		return out
	}
	invNorm := 1.0 / math.Sqrt(normSq)
	for i, v := range vec {
		out[i] = float32(v * invNorm)
	}
	return out
}

// estimateTokens computes a deterministic token count approximation matching
// control-plane/internal/eval/membench.
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
