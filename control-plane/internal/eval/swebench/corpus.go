package swebench

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"time"
)

// CorpusTask is a self-contained, clean-room Apache-2.0 multi-file Go coding
// task aligned with SWE-bench Pro V2 (sep-2026 revision, v2).
//
// Each task defines a real Go module with zero external dependencies so it can
// be compiled and verified offline via `go test -json -count=1 ./...` under
// GOPROXY=off inside isolated Visor sandboxes.
type CorpusTask struct {
	InstanceID          string            `json:"instance_id"`
	Repo                string            `json:"repo"`
	Language            string            `json:"language"`
	Difficulty          string            `json:"difficulty"`
	ProblemStatement    string            `json:"problem_statement"`
	BaseTimestamp       time.Time         `json:"base_timestamp"`
	FixTimestamp        time.Time         `json:"fix_timestamp"`
	BaseFiles           map[string]string `json:"base_files"`
	TestFiles           map[string]string `json:"test_files"`
	ReferenceFixFiles   map[string]string `json:"reference_fix_files"`
	SingleShotFixFiles  map[string]string `json:"single_shot_fix_files,omitempty"`
	IterativeStep1Files map[string]string `json:"iterative_step1_files,omitempty"`
	FailToPassTests     []string          `json:"fail_to_pass_tests"`
	PassToPassTests     []string          `json:"pass_to_pass_tests"`
	ProtectedTestPaths  []string          `json:"protected_test_paths"`
	SolutionFingerprint string            `json:"solution_fingerprint"`
	RequiredIterations  int               `json:"required_iterations"`
}

// TestPatchHash computes a deterministic SHA-256 digest of all protected test files.
func (c CorpusTask) TestPatchHash() string {
	h := sha256.New()
	keys := make([]string, 0, len(c.TestFiles))
	for k := range c.TestFiles {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		_, _ = fmt.Fprintf(h, "%s:%d:%s\n", k, len(c.TestFiles[k]), c.TestFiles[k])
	}
	return hex.EncodeToString(h.Sum(nil))
}

// ToTaskSpec converts a CorpusTask into a canonical sep-2026 v2 TaskSpec.
func (c CorpusTask) ToTaskSpec(baseCommit string) TaskSpec {
	return TaskSpec{
		InstanceID:       c.InstanceID,
		Repo:             c.Repo,
		BaseCommit:       baseCommit,
		Revision:         RevisionSep2026,
		Version:          BenchmarkVersionV2,
		Language:         c.Language,
		Difficulty:       c.Difficulty,
		TestPatchHash:    c.TestPatchHash(),
		BaseTimestamp:    c.BaseTimestamp,
		FixTimestamp:     c.FixTimestamp,
		EnvironmentImage: "hanzo/visor-go-1.22:sep-2026",
		Metadata: map[string]string{
			"clean_room_license": "Apache-2.0",
			"corpus":             "hip-1347-go-v2",
		},
	}
}

// DefaultSep2026Cutoff returns the September 2026 SWE-bench Pro V2 cutoff date.
func DefaultSep2026Cutoff() time.Time {
	return time.Date(2026, 9, 22, 0, 0, 0, 0, time.UTC)
}

// BuildCleanRoomGoCorpus returns 8 deterministic multi-file Go coding tasks
// spanning routing, caching, DAG scheduling, rate limiting, circuit breaking,
// WAL compaction, log redaction, and layered config merging.
func BuildCleanRoomGoCorpus() []CorpusTask {
	base := time.Date(2026, 9, 10, 12, 0, 0, 0, time.UTC)
	fix := time.Date(2026, 9, 25, 12, 0, 0, 0, time.UTC)

	return []CorpusTask{
		taskRouterPathNorm(base, fix),
		taskTTLCacheEvict(base.Add(2*time.Hour), fix.Add(2*time.Hour)),
		taskDAGSchedCycle(base.Add(4*time.Hour), fix.Add(4*time.Hour)),
		taskTokenBucketBurst(base.Add(6*time.Hour), fix.Add(6*time.Hour)),
		taskCircuitBreakerHalfOpen(base.Add(8*time.Hour), fix.Add(8*time.Hour)),
		taskWALCompactCRC(base.Add(10*time.Hour), fix.Add(10*time.Hour)),
		taskLogRedactNested(base.Add(12*time.Hour), fix.Add(12*time.Hour)),
		taskConfigMergeOverlay(base.Add(14*time.Hour), fix.Add(14*time.Hour)),
	}
}

func taskRouterPathNorm(baseTime, fixTime time.Time) CorpusTask {
	goMod := "module swebench.local/router\n\ngo 1.22\n"
	baseRouter := `package router

import "strings"

type Route struct {
	Pattern string
	Handler string
}

type Match struct {
	Handler string
	Params  map[string]string
}

type Router struct {
	routes []Route
}

func New() *Router { return &Router{} }

func (r *Router) Add(pattern, handler string) {
	r.routes = append(r.routes, Route{Pattern: pattern, Handler: handler})
}

func (r *Router) MatchPath(rawPath string) (Match, bool) {
	p := NormalizePath(rawPath)
	parts := splitSegments(p)
	for _, rt := range r.routes {
		patParts := splitSegments(rt.Pattern)
		if len(patParts) != len(parts) {
			continue
		}
		params := make(map[string]string)
		ok := true
		for i := range patParts {
			if strings.HasPrefix(patParts[i], ":") {
				params[patParts[i][1:]] = parts[i]
				continue
			}
			if patParts[i] != parts[i] {
				ok = false
				break
			}
		}
		if ok {
			return Match{Handler: rt.Handler, Params: params}, true
		}
	}
	return Match{}, false
}

func splitSegments(p string) []string {
	trimmed := strings.Trim(p, "/")
	if trimmed == "" {
		return nil
	}
	return strings.Split(trimmed, "/")
}
`
	baseNorm := `package router

import "strings"

// NormalizePath collapses repeated slashes and strips trailing slashes except root.
func NormalizePath(raw string) string {
	if raw == "" {
		return "/"
	}
	// BUG: only replaces one pair of double slashes and does not trim trailing slash.
	out := strings.Replace(raw, "//", "/", 1)
	if !strings.HasPrefix(out, "/") {
		out = "/" + out
	}
	return out
}
`
	fixedRouter := `package router

import "strings"

type Route struct {
	Pattern string
	Handler string
}

type Match struct {
	Handler string
	Params  map[string]string
}

type Router struct {
	routes []Route
}

func New() *Router { return &Router{} }

func (r *Router) Add(pattern, handler string) {
	r.routes = append(r.routes, Route{Pattern: NormalizePath(pattern), Handler: handler})
}

func (r *Router) MatchPath(rawPath string) (Match, bool) {
	p := NormalizePath(rawPath)
	parts := splitSegments(p)
	for _, rt := range r.routes {
		patParts := splitSegments(rt.Pattern)
		if len(patParts) != len(parts) {
			continue
		}
		params := make(map[string]string)
		ok := true
		for i := range patParts {
			if strings.HasPrefix(patParts[i], ":") {
				if parts[i] == "" {
					ok = false
					break
				}
				params[patParts[i][1:]] = parts[i]
				continue
			}
			if patParts[i] != parts[i] {
				ok = false
				break
			}
		}
		if ok {
			return Match{Handler: rt.Handler, Params: params}, true
		}
	}
	return Match{}, false
}

func splitSegments(p string) []string {
	trimmed := strings.Trim(p, "/")
	if trimmed == "" {
		return nil
	}
	return strings.Split(trimmed, "/")
}
`
	fixedNorm := `package router

import "strings"

// NormalizePath collapses repeated slashes and strips trailing slashes except root.
func NormalizePath(raw string) string {
	if raw == "" {
		return "/"
	}
	var b strings.Builder
	b.Grow(len(raw) + 1)
	if raw[0] != '/' {
		b.WriteByte('/')
	}
	prevSlash := false
	for i := 0; i < len(raw); i++ {
		ch := raw[i]
		if ch == '/' {
			if !prevSlash {
				b.WriteByte('/')
			}
			prevSlash = true
			continue
		}
		prevSlash = false
		b.WriteByte(ch)
	}
	out := b.String()
	if len(out) > 1 && strings.HasSuffix(out, "/") {
		out = strings.TrimRight(out, "/")
		if out == "" {
			return "/"
		}
	}
	return out
}
`
	testCode := `package router

import "testing"

func TestBasicStaticAndParamMatch(t *testing.T) {
	r := New()
	r.Add("/v1/agents/:id", "get_agent")
	m, ok := r.MatchPath("/v1/agents/alpha")
	if !ok || m.Handler != "get_agent" || m.Params["id"] != "alpha" {
		t.Fatalf("unexpected match: %+v ok=%v", m, ok)
	}
}

func TestNormalizeRepeatedAndTrailingSlashes(t *testing.T) {
	if got := NormalizePath("///v1///agents///42///"); got != "/v1/agents/42" {
		t.Fatalf("got %q, want /v1/agents/42", got)
	}
	if got := NormalizePath("///"); got != "/" {
		t.Fatalf("got %q, want /", got)
	}
}

func TestMatchWithUnnormalizedPatternAndPath(t *testing.T) {
	r := New()
	r.Add("/v1//agents/:id/", "get_agent")
	m, ok := r.MatchPath("///v1////agents///beta///")
	if !ok || m.Handler != "get_agent" || m.Params["id"] != "beta" {
		t.Fatalf("expected normalized match, got %+v ok=%v", m, ok)
	}
}
`
	return CorpusTask{
		InstanceID:          "hanzo__router-path-norm-01",
		Repo:                "hanzoai/router-eval",
		Language:            "go",
		Difficulty:          "medium",
		ProblemStatement:    "NormalizePath fails to collapse more than two consecutive slashes or strip trailing slashes, and Router.Add does not normalize registered route patterns.",
		BaseTimestamp:       baseTime,
		FixTimestamp:        fixTime,
		BaseFiles:           map[string]string{"go.mod": goMod, "router.go": baseRouter, "normalize.go": baseNorm},
		TestFiles:           map[string]string{"router_test.go": testCode},
		ReferenceFixFiles:   map[string]string{"router.go": fixedRouter, "normalize.go": fixedNorm},
		SingleShotFixFiles:  map[string]string{"router.go": fixedRouter, "normalize.go": fixedNorm},
		FailToPassTests:     []string{"TestNormalizeRepeatedAndTrailingSlashes", "TestMatchWithUnnormalizedPatternAndPath"},
		PassToPassTests:     []string{"TestBasicStaticAndParamMatch"},
		ProtectedTestPaths:  []string{"router_test.go"},
		SolutionFingerprint: "HANZO_GOLD_FP_ROUTER_01_9f4b2c",
		RequiredIterations:  1,
	}
}

func taskTTLCacheEvict(baseTime, fixTime time.Time) CorpusTask {
	goMod := "module swebench.local/ttlcache\n\ngo 1.22\n"
	baseEntry := `package ttlcache

import "time"

type Entry struct {
	Key       string
	Value     string
	ExpiresAt time.Time
}

func (e Entry) IsExpired(now time.Time) bool {
	if e.ExpiresAt.IsZero() {
		return false
	}
	return now.After(e.ExpiresAt)
}
`
	baseCache := `package ttlcache

import (
	"sync"
	"time"
)

type Cache struct {
	mu      sync.Mutex
	cap     int
	items   map[string]*Entry
	order   []string
	nowFunc func() time.Time
}

func New(capacity int, nowFunc func() time.Time) *Cache {
	if capacity <= 0 {
		capacity = 1
	}
	if nowFunc == nil {
		nowFunc = time.Now
	}
	return &Cache{
		cap:     capacity,
		items:   make(map[string]*Entry, capacity),
		nowFunc: nowFunc,
	}
}

func (c *Cache) Set(key, val string, ttl time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	now := c.nowFunc()
	var exp time.Time
	if ttl > 0 {
		exp = now.Add(ttl)
	}
	if existing, ok := c.items[key]; ok {
		existing.Value = val
		existing.ExpiresAt = exp
		return
	}
	// BUG: does not purge expired entries before evicting live LRU keys.
	if len(c.items) >= c.cap && len(c.order) > 0 {
		oldest := c.order[0]
		c.order = c.order[1:]
		delete(c.items, oldest)
	}
	c.items[key] = &Entry{Key: key, Value: val, ExpiresAt: exp}
	c.order = append(c.order, key)
}

func (c *Cache) Get(key string) (string, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	e, ok := c.items[key]
	if !ok {
		return "", false
	}
	if e.IsExpired(c.nowFunc()) {
		delete(c.items, key)
		return "", false
	}
	return e.Value, true
}

func (c *Cache) Len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.items)
}
`
	// Partial fix misses updating recency on Get/Set and exact-boundary expiration (now >= ExpiresAt).
	partialCache := `package ttlcache

import (
	"sync"
	"time"
)

type Cache struct {
	mu      sync.Mutex
	cap     int
	items   map[string]*Entry
	order   []string
	nowFunc func() time.Time
}

func New(capacity int, nowFunc func() time.Time) *Cache {
	if capacity <= 0 {
		capacity = 1
	}
	if nowFunc == nil {
		nowFunc = time.Now
	}
	return &Cache{
		cap:     capacity,
		items:   make(map[string]*Entry, capacity),
		nowFunc: nowFunc,
	}
}

func (c *Cache) purgeExpiredLocked(now time.Time) {
	kept := c.order[:0]
	for _, k := range c.order {
		e, ok := c.items[k]
		if !ok || e.IsExpired(now) {
			delete(c.items, k)
			continue
		}
		kept = append(kept, k)
	}
	c.order = kept
}

func (c *Cache) Set(key, val string, ttl time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	now := c.nowFunc()
	c.purgeExpiredLocked(now)
	var exp time.Time
	if ttl > 0 {
		exp = now.Add(ttl)
	}
	if existing, ok := c.items[key]; ok {
		existing.Value = val
		existing.ExpiresAt = exp
		return
	}
	if len(c.items) >= c.cap && len(c.order) > 0 {
		oldest := c.order[0]
		c.order = c.order[1:]
		delete(c.items, oldest)
	}
	c.items[key] = &Entry{Key: key, Value: val, ExpiresAt: exp}
	c.order = append(c.order, key)
}

func (c *Cache) Get(key string) (string, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	e, ok := c.items[key]
	if !ok {
		return "", false
	}
	if e.IsExpired(c.nowFunc()) {
		delete(c.items, key)
		return "", false
	}
	return e.Value, true
}

func (c *Cache) Len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.purgeExpiredLocked(c.nowFunc())
	return len(c.items)
}
`
	fixedEntry := `package ttlcache

import "time"

type Entry struct {
	Key       string
	Value     string
	ExpiresAt time.Time
}

func (e Entry) IsExpired(now time.Time) bool {
	if e.ExpiresAt.IsZero() {
		return false
	}
	return !now.Before(e.ExpiresAt)
}
`
	fixedCache := `package ttlcache

import (
	"sync"
	"time"
)

type Cache struct {
	mu      sync.Mutex
	cap     int
	items   map[string]*Entry
	order   []string
	nowFunc func() time.Time
}

func New(capacity int, nowFunc func() time.Time) *Cache {
	if capacity <= 0 {
		capacity = 1
	}
	if nowFunc == nil {
		nowFunc = time.Now
	}
	return &Cache{
		cap:     capacity,
		items:   make(map[string]*Entry, capacity),
		nowFunc: nowFunc,
	}
}

func (c *Cache) touchLocked(key string) {
	for i, k := range c.order {
		if k == key {
			c.order = append(c.order[:i], c.order[i+1:]...)
			break
		}
	}
	c.order = append(c.order, key)
}

func (c *Cache) purgeExpiredLocked(now time.Time) {
	kept := c.order[:0]
	for _, k := range c.order {
		e, ok := c.items[k]
		if !ok || e.IsExpired(now) {
			delete(c.items, k)
			continue
		}
		kept = append(kept, k)
	}
	c.order = kept
}

func (c *Cache) Set(key, val string, ttl time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	now := c.nowFunc()
	c.purgeExpiredLocked(now)
	var exp time.Time
	if ttl > 0 {
		exp = now.Add(ttl)
	}
	if existing, ok := c.items[key]; ok {
		existing.Value = val
		existing.ExpiresAt = exp
		c.touchLocked(key)
		return
	}
	if len(c.items) >= c.cap && len(c.order) > 0 {
		oldest := c.order[0]
		c.order = c.order[1:]
		delete(c.items, oldest)
	}
	c.items[key] = &Entry{Key: key, Value: val, ExpiresAt: exp}
	c.order = append(c.order, key)
}

func (c *Cache) Get(key string) (string, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	now := c.nowFunc()
	e, ok := c.items[key]
	if !ok {
		return "", false
	}
	if e.IsExpired(now) {
		delete(c.items, key)
		c.purgeExpiredLocked(now)
		return "", false
	}
	c.touchLocked(key)
	return e.Value, true
}

func (c *Cache) Len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.purgeExpiredLocked(c.nowFunc())
	return len(c.items)
}
`
	testCode := `package ttlcache

import (
	"testing"
	"time"
)

func TestBasicSetAndGet(t *testing.T) {
	now := time.Unix(1000, 0)
	c := New(2, func() time.Time { return now })
	c.Set("a", "1", time.Minute)
	v, ok := c.Get("a")
	if !ok || v != "1" {
		t.Fatalf("got %q ok=%v", v, ok)
	}
}

func TestPurgeExpiredBeforeEvictingLiveEntries(t *testing.T) {
	now := time.Unix(1000, 0)
	c := New(2, func() time.Time { return now })
	c.Set("long", "l", 60*time.Second)
	c.Set("short", "s", 10*time.Second)
	now = now.Add(10 * time.Second) // exact expiration boundary
	c.Set("fresh", "f", 60*time.Second)
	if _, ok := c.Get("long"); !ok {
		t.Fatalf("live entry 'long' should not be evicted when 'short' already expired")
	}
	if _, ok := c.Get("short"); ok {
		t.Fatalf("expired entry 'short' must be purged at exact boundary")
	}
	if c.Len() != 2 {
		t.Fatalf("len=%d want 2", c.Len())
	}
}

func TestGetPromotesLRUOrder(t *testing.T) {
	now := time.Unix(1000, 0)
	c := New(2, func() time.Time { return now })
	c.Set("a", "1", time.Minute)
	c.Set("b", "2", time.Minute)
	if _, ok := c.Get("a"); !ok {
		t.Fatalf("expected 'a'")
	}
	c.Set("c", "3", time.Minute)
	if _, ok := c.Get("a"); !ok {
		t.Fatalf("'a' was touched and must not be evicted")
	}
	if _, ok := c.Get("b"); ok {
		t.Fatalf("'b' should have been evicted as LRU")
	}
}
`
	return CorpusTask{
		InstanceID:          "hanzo__ttlcache-evict-02",
		Repo:                "hanzoai/cache-eval",
		Language:            "go",
		Difficulty:          "hard",
		ProblemStatement:    "TTLCache evicts live entries instead of purging expired ones on capacity pressure, fails exact-boundary TTL expiration (now == ExpiresAt), and does not refresh LRU recency on Get/Set.",
		BaseTimestamp:       baseTime,
		FixTimestamp:        fixTime,
		BaseFiles:           map[string]string{"go.mod": goMod, "entry.go": baseEntry, "cache.go": baseCache},
		TestFiles:           map[string]string{"cache_test.go": testCode},
		ReferenceFixFiles:   map[string]string{"entry.go": fixedEntry, "cache.go": fixedCache},
		SingleShotFixFiles:  map[string]string{"cache.go": partialCache},
		IterativeStep1Files: map[string]string{"cache.go": partialCache},
		FailToPassTests:     []string{"TestPurgeExpiredBeforeEvictingLiveEntries", "TestGetPromotesLRUOrder"},
		PassToPassTests:     []string{"TestBasicSetAndGet"},
		ProtectedTestPaths:  []string{"cache_test.go"},
		SolutionFingerprint: "HANZO_GOLD_FP_TTLCACHE_02_7a1e8d",
		RequiredIterations:  2,
	}
}

func taskDAGSchedCycle(baseTime, fixTime time.Time) CorpusTask {
	goMod := "module swebench.local/dagsched\n\ngo 1.22\n"
	baseGraph := `package dagsched

type Graph struct {
	deps map[string][]string
}

func NewGraph() *Graph {
	return &Graph{deps: make(map[string][]string)}
}

func (g *Graph) AddNode(id string, dependsOn ...string) {
	if _, ok := g.deps[id]; !ok {
		g.deps[id] = nil
	}
	for _, d := range dependsOn {
		g.deps[id] = append(g.deps[id], d)
		if _, ok := g.deps[d]; !ok {
			g.deps[d] = nil
		}
	}
}
`
	baseSched := `package dagsched

import "errors"

var ErrCycle = errors.New("cycle detected")

// Plan returns a topological execution order.
func Plan(g *Graph) ([]string, error) {
	// BUG: non-deterministic map iteration and does not detect cycles properly.
	indeg := make(map[string]int, len(g.deps))
	adj := make(map[string][]string, len(g.deps))
	for node, deps := range g.deps {
		indeg[node] = len(deps)
		for _, d := range deps {
			adj[d] = append(adj[d], node)
		}
	}
	var q []string
	for node, deg := range indeg {
		if deg == 0 {
			q = append(q, node)
		}
	}
	var order []string
	for len(q) > 0 {
		cur := q[0]
		q = q[1:]
		order = append(order, cur)
		for _, nxt := range adj[cur] {
			indeg[nxt]--
			if indeg[nxt] == 0 {
				q = append(q, nxt)
			}
		}
	}
	return order, nil
}
`
	fixedGraph := `package dagsched

type Graph struct {
	deps map[string][]string
}

func NewGraph() *Graph {
	return &Graph{deps: make(map[string][]string)}
}

func (g *Graph) AddNode(id string, dependsOn ...string) {
	if _, ok := g.deps[id]; !ok {
		g.deps[id] = nil
	}
	seen := make(map[string]bool, len(g.deps[id]))
	for _, existing := range g.deps[id] {
		seen[existing] = true
	}
	for _, d := range dependsOn {
		if !seen[d] {
			g.deps[id] = append(g.deps[id], d)
			seen[d] = true
		}
		if _, ok := g.deps[d]; !ok {
			g.deps[d] = nil
		}
	}
}
`
	fixedSched := `package dagsched

import (
	"errors"
	"sort"
)

var ErrCycle = errors.New("cycle detected")

// Plan returns a deterministic lexicographical topological execution order.
func Plan(g *Graph) ([]string, error) {
	indeg := make(map[string]int, len(g.deps))
	adj := make(map[string][]string, len(g.deps))
	for node, deps := range g.deps {
		indeg[node] = len(deps)
		for _, d := range deps {
			adj[d] = append(adj[d], node)
		}
	}
	var ready []string
	for node, deg := range indeg {
		if deg == 0 {
			ready = append(ready, node)
		}
	}
	sort.Strings(ready)

	order := make([]string, 0, len(indeg))
	for len(ready) > 0 {
		cur := ready[0]
		ready = ready[1:]
		order = append(order, cur)
		for _, nxt := range adj[cur] {
			indeg[nxt]--
			if indeg[nxt] == 0 {
				ready = append(ready, nxt)
			}
		}
		sort.Strings(ready)
	}
	if len(order) != len(indeg) {
		return nil, ErrCycle
	}
	return order, nil
}
`
	testCode := `package dagsched

import (
	"errors"
	"reflect"
	"testing"
)

func TestLinearChainPlan(t *testing.T) {
	g := NewGraph()
	g.AddNode("c", "b")
	g.AddNode("b", "a")
	got, err := Plan(g)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	want := []string{"a", "b", "c"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}
}

func TestCycleDetectionAndDuplicateEdges(t *testing.T) {
	g := NewGraph()
	g.AddNode("a", "b")
	g.AddNode("b", "a")
	if _, err := Plan(g); !errors.Is(err, ErrCycle) {
		t.Fatalf("expected ErrCycle, got %v", err)
	}

	g2 := NewGraph()
	g2.AddNode("z", "a", "a")
	g2.AddNode("m", "a")
	got, err := Plan(g2)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	want := []string{"a", "m", "z"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}
}
`
	return CorpusTask{
		InstanceID:          "hanzo__dagsched-cycle-03",
		Repo:                "hanzoai/dagsched-eval",
		Language:            "go",
		Difficulty:          "medium",
		ProblemStatement:    "DAG Plan silently returns truncated schedules on cycles, fails on duplicate dependency edges, and produces non-deterministic order on ties.",
		BaseTimestamp:       baseTime,
		FixTimestamp:        fixTime,
		BaseFiles:           map[string]string{"go.mod": goMod, "graph.go": baseGraph, "scheduler.go": baseSched},
		TestFiles:           map[string]string{"scheduler_test.go": testCode},
		ReferenceFixFiles:   map[string]string{"graph.go": fixedGraph, "scheduler.go": fixedSched},
		SingleShotFixFiles:  map[string]string{"graph.go": fixedGraph, "scheduler.go": fixedSched},
		FailToPassTests:     []string{"TestCycleDetectionAndDuplicateEdges"},
		PassToPassTests:     []string{"TestLinearChainPlan"},
		ProtectedTestPaths:  []string{"scheduler_test.go"},
		SolutionFingerprint: "HANZO_GOLD_FP_DAGSCHED_03_4c8f11",
		RequiredIterations:  1,
	}
}

func taskTokenBucketBurst(baseTime, fixTime time.Time) CorpusTask {
	goMod := "module swebench.local/ratelimit\n\ngo 1.22\n"
	baseClock := `package ratelimit

import "time"

type Clock interface {
	Now() time.Time
}

type ManualClock struct {
	T time.Time
}

func (m *ManualClock) Now() time.Time { return m.T }
func (m *ManualClock) Advance(d time.Duration) {
	m.T = m.T.Add(d)
}
`
	baseLimiter := `package ratelimit

import (
	"sync"
	"time"
)

type bucket struct {
	tokens float64
	last   time.Time
}

type TenantLimiter struct {
	mu      sync.Mutex
	rate    float64
	burst   float64
	clock   Clock
	buckets map[string]*bucket
}

func NewTenantLimiter(ratePerSec float64, burst int, clk Clock) *TenantLimiter {
	return &TenantLimiter{
		rate:    ratePerSec,
		burst:   float64(burst),
		clock:   clk,
		buckets: make(map[string]*bucket),
	}
}

func (l *TenantLimiter) Allow(tenant string, cost int) bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	now := l.clock.Now()
	b, ok := l.buckets[tenant]
	if !ok {
		b = &bucket{tokens: l.burst, last: now}
		l.buckets[tenant] = b
	}
	// BUG: truncates elapsed seconds to integer, loses sub-second refills, does not clamp burst or reject negative/oversize cost.
	elapsedSec := int(now.Sub(b.last).Seconds())
	if elapsedSec > 0 {
		b.tokens += float64(elapsedSec) * l.rate
		b.last = now
	}
	if b.tokens < float64(cost) {
		return false
	}
	b.tokens -= float64(cost)
	return true
}
`
	partialLimiter := `package ratelimit

import (
	"sync"
	"time"
)

type bucket struct {
	tokens float64
	last   time.Time
}

type TenantLimiter struct {
	mu      sync.Mutex
	rate    float64
	burst   float64
	clock   Clock
	buckets map[string]*bucket
}

func NewTenantLimiter(ratePerSec float64, burst int, clk Clock) *TenantLimiter {
	return &TenantLimiter{
		rate:    ratePerSec,
		burst:   float64(burst),
		clock:   clk,
		buckets: make(map[string]*bucket),
	}
}

func (l *TenantLimiter) Allow(tenant string, cost int) bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	now := l.clock.Now()
	b, ok := l.buckets[tenant]
	if !ok {
		b = &bucket{tokens: l.burst, last: now}
		l.buckets[tenant] = b
	}
	elapsed := now.Sub(b.last).Seconds()
	if elapsed > 0 {
		b.tokens += elapsed * l.rate
		// Partial BUG: forgets burst clamp and backward clock guard.
		b.last = now
	}
	if b.tokens < float64(cost) {
		return false
	}
	b.tokens -= float64(cost)
	return true
}
`
	fixedClock := `package ratelimit

import "time"

type Clock interface {
	Now() time.Time
}

type ManualClock struct {
	T time.Time
}

func (m *ManualClock) Now() time.Time { return m.T }
func (m *ManualClock) Advance(d time.Duration) {
	m.T = m.T.Add(d)
}
`
	fixedLimiter := `package ratelimit

import (
	"sync"
	"time"
)

type bucket struct {
	tokens float64
	last   time.Time
}

type TenantLimiter struct {
	mu      sync.Mutex
	rate    float64
	burst   float64
	clock   Clock
	buckets map[string]*bucket
}

func NewTenantLimiter(ratePerSec float64, burst int, clk Clock) *TenantLimiter {
	return &TenantLimiter{
		rate:    ratePerSec,
		burst:   float64(burst),
		clock:   clk,
		buckets: make(map[string]*bucket),
	}
}

func (l *TenantLimiter) Allow(tenant string, cost int) bool {
	if cost <= 0 {
		return false
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	now := l.clock.Now()
	b, ok := l.buckets[tenant]
	if !ok {
		b = &bucket{tokens: l.burst, last: now}
		l.buckets[tenant] = b
	}
	if now.After(b.last) {
		elapsed := now.Sub(b.last).Seconds()
		b.tokens += elapsed * l.rate
		if b.tokens > l.burst {
			b.tokens = l.burst
		}
		b.last = now
	}
	if float64(cost) > l.burst || b.tokens+1e-9 < float64(cost) {
		return false
	}
	b.tokens -= float64(cost)
	if b.tokens < 0 {
		b.tokens = 0
	}
	return true
}
`
	testCode := `package ratelimit

import (
	"testing"
	"time"
)

func TestTenantIsolation(t *testing.T) {
	clk := &ManualClock{T: time.Unix(1000, 0)}
	lim := NewTenantLimiter(2, 2, clk)
	if !lim.Allow("t1", 2) {
		t.Fatalf("t1 initial burst should pass")
	}
	if lim.Allow("t1", 1) {
		t.Fatalf("t1 should be exhausted")
	}
	if !lim.Allow("t2", 2) {
		t.Fatalf("t2 must be isolated from t1")
	}
}

func TestFractionalRefillAndBurstClamping(t *testing.T) {
	clk := &ManualClock{T: time.Unix(1000, 0)}
	lim := NewTenantLimiter(4, 2, clk)
	if !lim.Allow("t1", 2) {
		t.Fatalf("drain burst failed")
	}
	clk.Advance(250 * time.Millisecond) // +1.0 token
	if !lim.Allow("t1", 1) {
		t.Fatalf("expected fractional refill to allow 1 token")
	}
	clk.Advance(10 * time.Second) // would exceed burst if unclamped
	if !lim.Allow("t1", 2) {
		t.Fatalf("expected full burst 2")
	}
	if lim.Allow("t1", 1) {
		t.Fatalf("tokens must be clamped to burst=2")
	}
	if lim.Allow("t1", 0) || lim.Allow("t1", -1) {
		t.Fatalf("non-positive costs must be rejected")
	}
}
`
	return CorpusTask{
		InstanceID:          "hanzo__tokenbucket-burst-04",
		Repo:                "hanzoai/ratelimit-eval",
		Language:            "go",
		Difficulty:          "hard",
		ProblemStatement:    "TenantLimiter truncates refill durations to integer seconds, fails to clamp refilled tokens to burst capacity, and accepts non-positive costs.",
		BaseTimestamp:       baseTime,
		FixTimestamp:        fixTime,
		BaseFiles:           map[string]string{"go.mod": goMod, "clock.go": baseClock, "limiter.go": baseLimiter},
		TestFiles:           map[string]string{"limiter_test.go": testCode},
		ReferenceFixFiles:   map[string]string{"clock.go": fixedClock, "limiter.go": fixedLimiter},
		SingleShotFixFiles:  map[string]string{"limiter.go": partialLimiter},
		IterativeStep1Files: map[string]string{"limiter.go": partialLimiter},
		FailToPassTests:     []string{"TestFractionalRefillAndBurstClamping"},
		PassToPassTests:     []string{"TestTenantIsolation"},
		ProtectedTestPaths:  []string{"limiter_test.go"},
		SolutionFingerprint: "HANZO_GOLD_FP_RATE_04_b21a9e",
		RequiredIterations:  2,
	}
}

func taskCircuitBreakerHalfOpen(baseTime, fixTime time.Time) CorpusTask {
	goMod := "module swebench.local/breaker\n\ngo 1.22\n"
	basePolicy := `package breaker

import "time"

type Policy struct {
	FailureThreshold int
	OpenTimeout      time.Duration
	HalfOpenMaxProbe int
}
`
	baseBreaker := `package breaker

import (
	"sync"
	"time"
)

type State string

const (
	StateClosed   State = "closed"
	StateOpen     State = "open"
	StateHalfOpen State = "half_open"
)

type Breaker struct {
	mu       sync.Mutex
	policy   Policy
	state    State
	failures int
	openedAt time.Time
	nowFunc  func() time.Time
}

func New(p Policy, nowFunc func() time.Time) *Breaker {
	if nowFunc == nil {
		nowFunc = time.Now
	}
	return &Breaker{policy: p, state: StateClosed, nowFunc: nowFunc}
}

func (b *Breaker) Allow() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.state == StateOpen {
		if b.nowFunc().Sub(b.openedAt) >= b.policy.OpenTimeout {
			b.state = StateHalfOpen
			return true
		}
		return false
	}
	return true
}

func (b *Breaker) RecordFailure() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.failures++
	if b.failures >= b.policy.FailureThreshold {
		b.state = StateOpen
		b.openedAt = b.nowFunc()
	}
}

func (b *Breaker) RecordSuccess() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.failures = 0
	b.state = StateClosed
}

func (b *Breaker) State() State {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.state
}
`
	partialBreaker := `package breaker

import (
	"sync"
	"time"
)

type State string

const (
	StateClosed   State = "closed"
	StateOpen     State = "open"
	StateHalfOpen State = "half_open"
)

type Breaker struct {
	mu          sync.Mutex
	policy      Policy
	state       State
	failures    int
	halfOpenInF int
	openedAt    time.Time
	nowFunc     func() time.Time
}

func New(p Policy, nowFunc func() time.Time) *Breaker {
	if nowFunc == nil {
		nowFunc = time.Now
	}
	return &Breaker{policy: p, state: StateClosed, nowFunc: nowFunc}
}

func (b *Breaker) Allow() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.state == StateOpen {
		if b.nowFunc().Sub(b.openedAt) >= b.policy.OpenTimeout {
			b.state = StateHalfOpen
			b.halfOpenInF = 1
			return true
		}
		return false
	}
	if b.state == StateHalfOpen {
		if b.halfOpenInF >= b.policy.HalfOpenMaxProbe {
			return false
		}
		b.halfOpenInF++
		return true
	}
	return true
}

func (b *Breaker) RecordFailure() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.failures++
	// Partial BUG: forgets immediate re-open on HalfOpen probe failure and halfOpenInF reset.
	if b.failures >= b.policy.FailureThreshold {
		b.state = StateOpen
		b.openedAt = b.nowFunc()
	}
}

func (b *Breaker) RecordSuccess() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.failures = 0
	b.halfOpenInF = 0
	b.state = StateClosed
}

func (b *Breaker) State() State {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.state
}
`
	fixedBreaker := `package breaker

import (
	"sync"
	"time"
)

type State string

const (
	StateClosed   State = "closed"
	StateOpen     State = "open"
	StateHalfOpen State = "half_open"
)

type Breaker struct {
	mu              sync.Mutex
	policy          Policy
	state           State
	failures        int
	halfOpenProbes  int
	openedAt        time.Time
	nowFunc         func() time.Time
}

func New(p Policy, nowFunc func() time.Time) *Breaker {
	if nowFunc == nil {
		nowFunc = time.Now
	}
	if p.HalfOpenMaxProbe <= 0 {
		p.HalfOpenMaxProbe = 1
	}
	return &Breaker{policy: p, state: StateClosed, nowFunc: nowFunc}
}

func (b *Breaker) Allow() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.state == StateOpen {
		if b.nowFunc().Sub(b.openedAt) >= b.policy.OpenTimeout {
			b.state = StateHalfOpen
			b.halfOpenProbes = 1
			return true
		}
		return false
	}
	if b.state == StateHalfOpen {
		if b.halfOpenProbes >= b.policy.HalfOpenMaxProbe {
			return false
		}
		b.halfOpenProbes++
		return true
	}
	return true
}

func (b *Breaker) RecordFailure() {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.state == StateHalfOpen {
		b.state = StateOpen
		b.halfOpenProbes = 0
		b.failures = b.policy.FailureThreshold
		b.openedAt = b.nowFunc()
		return
	}
	b.failures++
	if b.failures >= b.policy.FailureThreshold {
		b.state = StateOpen
		b.openedAt = b.nowFunc()
	}
}

func (b *Breaker) RecordSuccess() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.failures = 0
	b.halfOpenProbes = 0
	b.state = StateClosed
}

func (b *Breaker) State() State {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.state
}
`
	testCode := `package breaker

import (
	"testing"
	"time"
)

func TestOpenAfterFailureThreshold(t *testing.T) {
	now := time.Unix(100, 0)
	b := New(Policy{FailureThreshold: 2, OpenTimeout: 5 * time.Second, HalfOpenMaxProbe: 1}, func() time.Time { return now })
	b.RecordFailure()
	if b.State() != StateClosed {
		t.Fatalf("want closed after 1 failure")
	}
	b.RecordFailure()
	if b.State() != StateOpen || b.Allow() {
		t.Fatalf("want open and denied after threshold")
	}
}

func TestHalfOpenProbeCapAndImmediateReopen(t *testing.T) {
	now := time.Unix(100, 0)
	b := New(Policy{FailureThreshold: 3, OpenTimeout: 5 * time.Second, HalfOpenMaxProbe: 1}, func() time.Time { return now })
	b.RecordFailure()
	b.RecordFailure()
	b.RecordFailure()
	now = now.Add(5 * time.Second)
	if !b.Allow() {
		t.Fatalf("first half-open probe should be allowed")
	}
	if b.Allow() {
		t.Fatalf("second concurrent half-open probe must be denied when HalfOpenMaxProbe=1")
	}
	// Reset failures via success then re-enter half-open and fail a single probe:
	b.RecordSuccess()
	b.RecordFailure()
	b.RecordFailure()
	b.RecordFailure()
	now = now.Add(5 * time.Second)
	if !b.Allow() {
		t.Fatalf("probe should be allowed after open timeout")
	}
	b.RecordSuccess() // recover
	b.RecordFailure()
	b.RecordFailure()
	b.RecordFailure()
	now = now.Add(5 * time.Second)
	b.Allow()
	// Fail half-open probe when failures counter was reset to 0 after prior recovery:
	b.failures = 0
	b.RecordFailure()
	if b.State() != StateOpen {
		t.Fatalf("single failure in half-open state must immediately reopen breaker, got %s", b.State())
	}
}
`
	return CorpusTask{
		InstanceID:          "hanzo__circuitbreaker-halfopen-05",
		Repo:                "hanzoai/breaker-eval",
		Language:            "go",
		Difficulty:          "hard",
		ProblemStatement:    "Circuit breaker allows unbounded concurrent probes in HalfOpen state and fails to immediately re-open on a single HalfOpen probe failure.",
		BaseTimestamp:       baseTime,
		FixTimestamp:        fixTime,
		BaseFiles:           map[string]string{"go.mod": goMod, "policy.go": basePolicy, "breaker.go": baseBreaker},
		TestFiles:           map[string]string{"breaker_test.go": testCode},
		ReferenceFixFiles:   map[string]string{"breaker.go": fixedBreaker},
		SingleShotFixFiles:  map[string]string{"breaker.go": partialBreaker},
		IterativeStep1Files: map[string]string{"breaker.go": partialBreaker},
		FailToPassTests:     []string{"TestHalfOpenProbeCapAndImmediateReopen"},
		PassToPassTests:     []string{"TestOpenAfterFailureThreshold"},
		ProtectedTestPaths:  []string{"breaker_test.go"},
		SolutionFingerprint: "HANZO_GOLD_FP_BREAKER_05_e33d70",
		RequiredIterations:  2,
	}
}

func taskWALCompactCRC(baseTime, fixTime time.Time) CorpusTask {
	goMod := "module swebench.local/wal\n\ngo 1.22\n"
	baseWAL := `package wal

import (
	"fmt"
	"hash/crc32"
	"strconv"
	"strings"
)

type Record struct {
	Op    string
	Key   string
	Value string
}

func EncodeLine(r Record) string {
	payload := r.Op + "|" + r.Key + "|" + r.Value
	sum := crc32.ChecksumIEEE([]byte(payload))
	return fmt.Sprintf("%08x|%s", sum, payload)
}

func DecodeLines(raw string) ([]Record, error) {
	var out []Record
	for _, line := range strings.Split(strings.TrimSpace(raw), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		parts := strings.SplitN(line, "|", 4)
		if len(parts) != 4 {
			return nil, fmt.Errorf("corrupt line: %q", line)
		}
		want, err := strconv.ParseUint(parts[0], 16, 32)
		if err != nil {
			return nil, err
		}
		payload := parts[1] + "|" + parts[2] + "|" + parts[3]
		if uint32(want) != crc32.ChecksumIEEE([]byte(payload)) {
			return nil, fmt.Errorf("crc mismatch")
		}
		out = append(out, Record{Op: parts[1], Key: parts[2], Value: parts[3]})
	}
	return out, nil
}
`
	baseCompact := `package wal

// Compact replays records and returns the surviving key-value state.
func Compact(records []Record) map[string]string {
	state := make(map[string]string)
	for _, r := range records {
		// BUG: ignores DEL tombstones.
		if r.Op == "PUT" {
			state[r.Key] = r.Value
		}
	}
	return state
}
`
	partialWAL := baseWAL
	partialCompact := `package wal

// Compact replays records and returns the surviving key-value state.
func Compact(records []Record) map[string]string {
	state := make(map[string]string)
	for _, r := range records {
		switch r.Op {
		case "PUT":
			state[r.Key] = r.Value
		case "DEL":
			delete(state, r.Key)
		}
	}
	return state
}
`
	fixedWAL := `package wal

import (
	"fmt"
	"hash/crc32"
	"strconv"
	"strings"
)

type Record struct {
	Op    string
	Key   string
	Value string
}

func EncodeLine(r Record) string {
	payload := r.Op + "|" + r.Key + "|" + r.Value
	sum := crc32.ChecksumIEEE([]byte(payload))
	return fmt.Sprintf("%08x|%s", sum, payload)
}

// DecodeLines tolerates a single torn/corrupt tail line (crash during append)
// while failing on mid-log corruption.
func DecodeLines(raw string) ([]Record, error) {
	lines := strings.Split(strings.TrimRight(raw, "\n"), "\n")
	var out []Record
	for idx, line := range lines {
		if strings.TrimSpace(line) == "" {
			continue
		}
		isLast := idx == len(lines)-1
		parts := strings.SplitN(line, "|", 4)
		if len(parts) != 4 {
			if isLast {
				break
			}
			return nil, fmt.Errorf("corrupt mid-log line %d: %q", idx, line)
		}
		want, err := strconv.ParseUint(parts[0], 16, 32)
		if err != nil {
			if isLast {
				break
			}
			return nil, err
		}
		payload := parts[1] + "|" + parts[2] + "|" + parts[3]
		if uint32(want) != crc32.ChecksumIEEE([]byte(payload)) {
			if isLast {
				break
			}
			return nil, fmt.Errorf("crc mismatch at line %d", idx)
		}
		out = append(out, Record{Op: parts[1], Key: parts[2], Value: parts[3]})
	}
	return out, nil
}
`
	testCode := `package wal

import "testing"

func TestEncodeDecodeRoundtrip(t *testing.T) {
	raw := EncodeLine(Record{Op: "PUT", Key: "k1", Value: "v1"}) + "\n"
	recs, err := DecodeLines(raw)
	if err != nil || len(recs) != 1 || recs[0].Value != "v1" {
		t.Fatalf("unexpected decode: %+v err=%v", recs, err)
	}
}

func TestTornTailRecoveryAndTombstoneCompaction(t *testing.T) {
	raw := EncodeLine(Record{Op: "PUT", Key: "k1", Value: "v1"}) + "\n" +
		EncodeLine(Record{Op: "PUT", Key: "k2", Value: "v2"}) + "\n" +
		EncodeLine(Record{Op: "DEL", Key: "k1", Value: ""}) + "\n" +
		"deadbeef|PUT|k3|torn-tail-write"
	recs, err := DecodeLines(raw)
	if err != nil {
		t.Fatalf("torn tail should recover cleanly, got err: %v", err)
	}
	state := Compact(recs)
	if _, exists := state["k1"]; exists {
		t.Fatalf("deleted key k1 must be removed during compaction")
	}
	if state["k2"] != "v2" {
		t.Fatalf("expected k2=v2, got %+v", state)
	}
	// Mid-log corruption must still fail.
	badMid := EncodeLine(Record{Op: "PUT", Key: "k1", Value: "v1"}) + "\n" +
		"00000000|PUT|bad|corrupt\n" +
		EncodeLine(Record{Op: "PUT", Key: "k2", Value: "v2"}) + "\n"
	if _, err := DecodeLines(badMid); err == nil {
		t.Fatalf("mid-log corruption must return error")
	}
}
`
	return CorpusTask{
		InstanceID:          "hanzo__walcompact-crc-06",
		Repo:                "hanzoai/wal-eval",
		Language:            "go",
		Difficulty:          "hard",
		ProblemStatement:    "WAL DecodeLines fails on crash-induced torn tail records instead of truncating only the trailing frame, and Compact ignores DEL tombstones.",
		BaseTimestamp:       baseTime,
		FixTimestamp:        fixTime,
		BaseFiles:           map[string]string{"go.mod": goMod, "wal.go": baseWAL, "compact.go": baseCompact},
		TestFiles:           map[string]string{"wal_test.go": testCode},
		ReferenceFixFiles:   map[string]string{"wal.go": fixedWAL, "compact.go": partialCompact},
		SingleShotFixFiles:  map[string]string{"wal.go": partialWAL, "compact.go": partialCompact},
		IterativeStep1Files: map[string]string{"compact.go": partialCompact},
		FailToPassTests:     []string{"TestTornTailRecoveryAndTombstoneCompaction"},
		PassToPassTests:     []string{"TestEncodeDecodeRoundtrip"},
		ProtectedTestPaths:  []string{"wal_test.go"},
		SolutionFingerprint: "HANZO_GOLD_FP_WAL_06_55c0a1",
		RequiredIterations:  2,
	}
}

func taskLogRedactNested(baseTime, fixTime time.Time) CorpusTask {
	goMod := "module swebench.local/logredact\n\ngo 1.22\n"
	baseWalker := `package logredact

import "strings"

func IsSensitiveKey(k string) bool {
	// BUG: case-sensitive check misses API_KEY / Authorization / Token.
	return k == "password" || k == "secret" || k == "api_key"
}

func hasSensitiveToken(k string) bool {
	low := strings.ToLower(k)
	for _, needle := range []string{"password", "secret", "token", "api_key", "authorization"} {
		if strings.Contains(low, needle) {
			return true
		}
	}
	return false
}
`
	baseRedactor := `package logredact

const Mask = "[REDACTED]"

// ScrubMap returns a redacted copy of m.
func ScrubMap(m map[string]any) map[string]any {
	out := make(map[string]any, len(m))
	for k, v := range m {
		if IsSensitiveKey(k) {
			out[k] = Mask
			continue
		}
		// BUG: does not recurse into nested maps or slices.
		out[k] = v
	}
	return out
}
`
	fixedWalker := `package logredact

import "strings"

func IsSensitiveKey(k string) bool {
	return hasSensitiveToken(k)
}

func hasSensitiveToken(k string) bool {
	low := strings.ToLower(k)
	for _, needle := range []string{"password", "secret", "token", "api_key", "authorization"} {
		if strings.Contains(low, needle) {
			return true
		}
	}
	return false
}
`
	fixedRedactor := `package logredact

const Mask = "[REDACTED]"

// ScrubMap returns a deep-redacted copy of m without mutating the caller map.
func ScrubMap(m map[string]any) map[string]any {
	out := make(map[string]any, len(m))
	for k, v := range m {
		if IsSensitiveKey(k) {
			out[k] = Mask
			continue
		}
		out[k] = scrubValue(v)
	}
	return out
}

func scrubValue(v any) any {
	switch tv := v.(type) {
	case map[string]any:
		return ScrubMap(tv)
	case []any:
		cp := make([]any, len(tv))
		for i, elem := range tv {
			cp[i] = scrubValue(elem)
		}
		return cp
	default:
		return v
	}
}
`
	testCode := `package logredact

import "testing"

func TestTopLevelRedaction(t *testing.T) {
	got := ScrubMap(map[string]any{"user": "alice", "api_key": "hz_secret"})
	if got["user"] != "alice" || got["api_key"] != Mask {
		t.Fatalf("unexpected top-level scrub: %+v", got)
	}
}

func TestNestedAndCaseInsensitiveRedaction(t *testing.T) {
	in := map[string]any{
		"Headers": map[string]any{
			"Authorization": "Bearer top-secret",
			"X-Request-Id":  "req-1",
		},
		"Items": []any{
			map[string]any{"HANZO_API_KEY": "hz_live_999", "name": "job"},
		},
	}
	got := ScrubMap(in)
	hdrs := got["Headers"].(map[string]any)
	if hdrs["Authorization"] != Mask || hdrs["X-Request-Id"] != "req-1" {
		t.Fatalf("nested headers not scrubbed: %+v", hdrs)
	}
	item := got["Items"].([]any)[0].(map[string]any)
	if item["HANZO_API_KEY"] != Mask || item["name"] != "job" {
		t.Fatalf("nested slice map not scrubbed: %+v", item)
	}
	// Caller input must remain unmutated.
	if in["Headers"].(map[string]any)["Authorization"] != "Bearer top-secret" {
		t.Fatalf("caller map was mutated")
	}
}
`
	return CorpusTask{
		InstanceID:          "hanzo__logredact-nested-07",
		Repo:                "hanzoai/logredact-eval",
		Language:            "go",
		Difficulty:          "medium",
		ProblemStatement:    "ScrubMap fails to match secret keys case-insensitively (e.g., Authorization, HANZO_API_KEY) and fails to recurse into nested maps and slices.",
		BaseTimestamp:       baseTime,
		FixTimestamp:        fixTime,
		BaseFiles:           map[string]string{"go.mod": goMod, "walker.go": baseWalker, "redactor.go": baseRedactor},
		TestFiles:           map[string]string{"redactor_test.go": testCode},
		ReferenceFixFiles:   map[string]string{"walker.go": fixedWalker, "redactor.go": fixedRedactor},
		SingleShotFixFiles:  map[string]string{"walker.go": fixedWalker, "redactor.go": fixedRedactor},
		FailToPassTests:     []string{"TestNestedAndCaseInsensitiveRedaction"},
		PassToPassTests:     []string{"TestTopLevelRedaction"},
		ProtectedTestPaths:  []string{"redactor_test.go"},
		SolutionFingerprint: "HANZO_GOLD_FP_REDACT_07_c09b33",
		RequiredIterations:  1,
	}
}

func taskConfigMergeOverlay(baseTime, fixTime time.Time) CorpusTask {
	goMod := "module swebench.local/cfgmerge\n\ngo 1.22\n"
	baseMerge := `package cfgmerge

// Merge overlays override onto base.
func Merge(base, override map[string]any) map[string]any {
	out := make(map[string]any, len(base))
	for k, v := range base {
		out[k] = v
	}
	// BUG: shallow overwrite replaces entire nested maps instead of deep-merging.
	for k, v := range override {
		out[k] = v
	}
	return out
}
`
	baseValidate := `package cfgmerge

import "fmt"

// ValidateAllowedKeys ensures every key in cfg appears in allowed.
func ValidateAllowedKeys(cfg map[string]any, allowed map[string]bool) error {
	for k := range cfg {
		if !allowed[k] {
			return fmt.Errorf("unknown key %q", k)
		}
	}
	return nil
}
`
	fixedMerge := `package cfgmerge

// Merge deep-merges override onto base without mutating either input map.
func Merge(base, override map[string]any) map[string]any {
	out := deepCopyMap(base)
	for k, ov := range override {
		if ovMap, ok := ov.(map[string]any); ok {
			if baseMap, ok2 := out[k].(map[string]any); ok2 {
				out[k] = Merge(baseMap, ovMap)
				continue
			}
			out[k] = deepCopyMap(ovMap)
			continue
		}
		out[k] = ov
	}
	return out
}

func deepCopyMap(m map[string]any) map[string]any {
	cp := make(map[string]any, len(m))
	for k, v := range m {
		if sub, ok := v.(map[string]any); ok {
			cp[k] = deepCopyMap(sub)
		} else {
			cp[k] = v
		}
	}
	return cp
}
`
	fixedValidate := `package cfgmerge

import (
	"fmt"
	"sort"
)

// ValidateAllowedKeys ensures every top-level key in cfg appears in allowed (deterministic order).
func ValidateAllowedKeys(cfg map[string]any, allowed map[string]bool) error {
	keys := make([]string, 0, len(cfg))
	for k := range cfg {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		if !allowed[k] {
			return fmt.Errorf("unknown key %q", k)
		}
	}
	return nil
}
`
	testCode := `package cfgmerge

import "testing"

func TestShallowOverrideAndValidation(t *testing.T) {
	merged := Merge(map[string]any{"host": "127.0.0.1"}, map[string]any{"port": 8080})
	if merged["host"] != "127.0.0.1" || merged["port"] != 8080 {
		t.Fatalf("unexpected merged: %+v", merged)
	}
	if err := ValidateAllowedKeys(merged, map[string]bool{"host": true, "port": true}); err != nil {
		t.Fatalf("unexpected validation err: %v", err)
	}
}

func TestDeepMergePreservesSiblingsAndImmutability(t *testing.T) {
	base := map[string]any{
		"sandbox": map[string]any{
			"memory_mb": 2048,
			"network":   "disabled",
		},
	}
	override := map[string]any{
		"sandbox": map[string]any{
			"memory_mb": 4096,
		},
	}
	merged := Merge(base, override)
	sb := merged["sandbox"].(map[string]any)
	if sb["memory_mb"] != 4096 || sb["network"] != "disabled" {
		t.Fatalf("deep merge lost sibling key: %+v", sb)
	}
	sb["memory_mb"] = 1
	if base["sandbox"].(map[string]any)["memory_mb"] != 2048 {
		t.Fatalf("base map was mutated through merged output")
	}
}
`
	return CorpusTask{
		InstanceID:          "hanzo__cfgmerge-overlay-08",
		Repo:                "hanzoai/cfgmerge-eval",
		Language:            "go",
		Difficulty:          "medium",
		ProblemStatement:    "Merge performs shallow overwrites on nested config maps, dropping sibling keys and aliasing caller maps.",
		BaseTimestamp:       baseTime,
		FixTimestamp:        fixTime,
		BaseFiles:           map[string]string{"go.mod": goMod, "merge.go": baseMerge, "validate.go": baseValidate},
		TestFiles:           map[string]string{"merge_test.go": testCode},
		ReferenceFixFiles:   map[string]string{"merge.go": fixedMerge, "validate.go": fixedValidate},
		SingleShotFixFiles:  map[string]string{"merge.go": fixedMerge, "validate.go": fixedValidate},
		FailToPassTests:     []string{"TestDeepMergePreservesSiblingsAndImmutability"},
		PassToPassTests:     []string{"TestShallowOverrideAndValidation"},
		ProtectedTestPaths:  []string{"merge_test.go"},
		SolutionFingerprint: "HANZO_GOLD_FP_CFG_08_f7219a",
		RequiredIterations:  1,
	}
}
