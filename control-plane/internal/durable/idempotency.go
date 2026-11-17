// Package durable implements event-triggered durable execution replay with
// persisted checkpoints and deterministic idempotency_key activity deduplication
// without always-on daemon or polling processes (HIP-1347).
package durable

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"
)

var (
	// ErrInvalidKeyComponent indicates an empty tenant, run, step, or negative round.
	ErrInvalidKeyComponent = errors.New("durable: invalid idempotency key component")
)

// IdempotencyKey computes the deterministic activity deduplication key
// sha256(org:runID:round:stepID) encoded as lowercase hexadecimal.
func IdempotencyKey(org, runID string, round int, stepID string) (string, error) {
	if strings.TrimSpace(org) == "" {
		return "", fmt.Errorf("%w: org cannot be empty", ErrInvalidKeyComponent)
	}
	if strings.TrimSpace(runID) == "" {
		return "", fmt.Errorf("%w: runID cannot be empty", ErrInvalidKeyComponent)
	}
	if round < 0 {
		return "", fmt.Errorf("%w: round %d cannot be negative", ErrInvalidKeyComponent, round)
	}
	if strings.TrimSpace(stepID) == "" {
		return "", fmt.Errorf("%w: stepID cannot be empty", ErrInvalidKeyComponent)
	}

	material := fmt.Sprintf("%s:%s:%d:%s", org, runID, round, stepID)
	sum := sha256.Sum256([]byte(material))
	return hex.EncodeToString(sum[:]), nil
}

// MustIdempotencyKey computes IdempotencyKey and panics on invalid components.
func MustIdempotencyKey(org, runID string, round int, stepID string) string {
	key, err := IdempotencyKey(org, runID, round, stepID)
	if err != nil {
		panic(err)
	}
	return key
}

// ActivityStatus represents the persisted outcome state of an idempotent activity.
type ActivityStatus string

const (
	ActivityStatusCompleted ActivityStatus = "completed"
	ActivityStatusFailed    ActivityStatus = "failed"
)

// ActivityLedgerEntry records a completed activity indexed by (org, idempotency_key).
type ActivityLedgerEntry struct {
	IdempotencyKey string          `json:"idempotency_key"`
	Org            string          `json:"org"`
	RunID          string          `json:"run_id"`
	Round          int             `json:"round"`
	StepID         string          `json:"step_id"`
	Sequence       int64           `json:"sequence"`
	Status         ActivityStatus  `json:"status"`
	InputHash      string          `json:"input_hash,omitempty"`
	Output         json.RawMessage `json:"output,omitempty"`
	Error          string          `json:"error,omitempty"`
	Attempt        int             `json:"attempt"`
	CompletedAt    time.Time       `json:"completed_at"`
}

func hashRawPayload(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:])
}
