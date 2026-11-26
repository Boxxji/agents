package replay

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"time"
)

// WorkflowClock provides deterministic virtual time anchored at the workflow's
// recorded start timestamp. Replaying a workflow from either event 0 or a
// mid-workflow checkpoint yields bit-identical timestamps without reading wall time.
type WorkflowClock struct {
	logicalNS int64
}

// NewWorkflowClock initializes a deterministic clock at startNS (nanoseconds since Unix epoch).
func NewWorkflowClock(startNS int64) *WorkflowClock {
	return &WorkflowClock{logicalNS: startNS}
}

// Now returns the current deterministic virtual UTC timestamp.
func (c *WorkflowClock) Now() time.Time {
	return time.Unix(0, c.logicalNS).UTC()
}

// UnixNano returns the current deterministic virtual timestamp in nanoseconds.
func (c *WorkflowClock) UnixNano() int64 {
	return c.logicalNS
}

// AdvanceTo monotonically advances the virtual clock to at least targetNS.
func (c *WorkflowClock) AdvanceTo(targetNS int64) {
	if targetNS > c.logicalNS {
		c.logicalNS = targetNS
	}
}

// AdvanceBy advances the virtual clock by d (ignoring non-positive durations).
func (c *WorkflowClock) AdvanceBy(d time.Duration) {
	if d > 0 {
		c.logicalNS += d.Nanoseconds()
	}
}

// WorkflowRNG is a deterministic, checkpointable SplitMix64 PRNG seeded strictly
// from workflow identity and start logical time. Restoring (State, Draws) from a
// persisted checkpoint reproduces the exact bit sequence of replaying from event 0.
type WorkflowRNG struct {
	state uint64
	draws uint64
}

// NewWorkflowRNG derives a deterministic PRNG seed from (tenantID, workflowID, startNS).
func NewWorkflowRNG(tenantID, workflowID string, startNS int64) *WorkflowRNG {
	h := sha256.New()
	writeLenPrefixed(h, "hanzo.replay.rng.v1")
	writeLenPrefixed(h, tenantID)
	writeLenPrefixed(h, workflowID)
	var buf [8]byte
	binary.BigEndian.PutUint64(buf[:], uint64(startNS))
	_, _ = h.Write(buf[:])
	sum := h.Sum(nil)
	seed := binary.BigEndian.Uint64(sum[:8])
	if seed == 0 {
		seed = 0x9e3779b97f4a7c15
	}
	return &WorkflowRNG{state: seed, draws: 0}
}

// RestoreWorkflowRNG restores a WorkflowRNG from checkpointed state.
func RestoreWorkflowRNG(state, draws uint64) *WorkflowRNG {
	return &WorkflowRNG{state: state, draws: draws}
}

// Snapshot returns the serializable PRNG state and draw counter for checkpointing.
func (r *WorkflowRNG) Snapshot() (state uint64, draws uint64) {
	return r.state, r.draws
}

// Uint64 returns the next deterministic 64-bit unsigned integer (SplitMix64).
func (r *WorkflowRNG) Uint64() uint64 {
	r.state += 0x9e3779b97f4a7c15
	r.draws++
	z := r.state
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

// Int63n returns a deterministic integer in [0, n) using unbiased rejection sampling.
func (r *WorkflowRNG) Int63n(n int64) int64 {
	if n <= 0 {
		panic("replay: invalid argument to Int63n")
	}
	un := uint64(n)
	limit := ^uint64(0) - (^uint64(0) % un)
	for {
		v := r.Uint64()
		if v < limit {
			return int64(v % un)
		}
	}
}

// Float64 returns a deterministic float64 in [0.0, 1.0) with 53 bits of precision.
func (r *WorkflowRNG) Float64() float64 {
	return float64(r.Uint64()>>11) / (1 << 53)
}

// DeterministicUUID generates a deterministic RFC-4122 v4-formatted UUID derived
// exclusively from the workflow's seeded PRNG.
func (r *WorkflowRNG) DeterministicUUID() string {
	var b [16]byte
	binary.BigEndian.PutUint64(b[0:8], r.Uint64())
	binary.BigEndian.PutUint64(b[8:16], r.Uint64())
	b[6] = (b[6] & 0x0f) | 0x40
	b[8] = (b[8] & 0x3f) | 0x80
	return fmt.Sprintf("%08x-%04x-%04x-%04x-%012x",
		b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}
