package replay

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// DeriveIdempotencyKey computes a collision-resistant, deterministic lowercase
// hex SHA-256 idempotency key using length-prefixed framing across
// (tenantID, workflowID, activityName, stepID).
//
// Length-prefixed encoding prevents delimiter-injection collisions such as
// ("a:b", "c", ...) vs ("a", "b:c", ...).
func DeriveIdempotencyKey(tenantID, workflowID, activityName, stepID string) (string, error) {
	if strings.TrimSpace(tenantID) == "" {
		return "", fmt.Errorf("%w: tenant_id cannot be empty", ErrInvalidTenant)
	}
	if strings.TrimSpace(workflowID) == "" {
		return "", fmt.Errorf("%w: workflow_id cannot be empty", ErrInvalidWorkflowID)
	}
	if strings.TrimSpace(activityName) == "" {
		return "", fmt.Errorf("%w: activity_name cannot be empty", ErrInvalidIdempotencyKey)
	}
	if strings.TrimSpace(stepID) == "" {
		return "", fmt.Errorf("%w: step_id cannot be empty", ErrInvalidIdempotencyKey)
	}

	h := sha256.New()
	writeLenPrefixed(h, "hanzo.replay.idem.v1")
	writeLenPrefixed(h, tenantID)
	writeLenPrefixed(h, workflowID)
	writeLenPrefixed(h, activityName)
	writeLenPrefixed(h, stepID)
	return hex.EncodeToString(h.Sum(nil)), nil
}

// ScopedLedgerKey computes the storage lookup key for (tenantID, workflowID, idempotencyKey).
func ScopedLedgerKey(tenantID, workflowID, idempotencyKey string) (string, error) {
	if strings.TrimSpace(tenantID) == "" {
		return "", fmt.Errorf("%w: tenant_id cannot be empty", ErrInvalidTenant)
	}
	if strings.TrimSpace(workflowID) == "" {
		return "", fmt.Errorf("%w: workflow_id cannot be empty", ErrInvalidWorkflowID)
	}
	if strings.TrimSpace(idempotencyKey) == "" {
		return "", fmt.Errorf("%w: idempotency_key cannot be empty", ErrInvalidIdempotencyKey)
	}

	h := sha256.New()
	writeLenPrefixed(h, "hanzo.replay.ledger.v1")
	writeLenPrefixed(h, tenantID)
	writeLenPrefixed(h, workflowID)
	writeLenPrefixed(h, idempotencyKey)
	return hex.EncodeToString(h.Sum(nil)), nil
}

// VFSContentRef formats a canonical content-addressed reference URI for large
// binary payloads stored in Hanzo VFS/S3 rather than inlined into workflow
// checkpoints (canonical boundary: vfs -> hanzoai/s3 -> visor).
func VFSContentRef(payload []byte) string {
	sum := sha256.Sum256(payload)
	return "vfs://sha256/" + hex.EncodeToString(sum[:])
}

// ActivityStatus records the terminal persisted state of a single activity execution.
type ActivityStatus string

const (
	ActivityStatusCompleted ActivityStatus = "completed"
	ActivityStatusFailed    ActivityStatus = "failed"
)

// IdempotencyRecord is persisted in the tenant-scoped deduplication ledger keyed
// by (tenant_id, workflow_id, idempotency_key).
type IdempotencyRecord struct {
	TenantID       string          `json:"tenant_id"`
	WorkflowID     string          `json:"workflow_id"`
	IdempotencyKey string          `json:"idempotency_key"`
	ActivityName   string          `json:"activity_name"`
	ScheduledSeq   int64           `json:"scheduled_seq"`
	CompletedSeq   int64           `json:"completed_seq"`
	Status         ActivityStatus  `json:"status"`
	InputDigest    string          `json:"input_digest"`
	Output         json.RawMessage `json:"output,omitempty"`
	OutputURI      string          `json:"output_uri,omitempty"`
	Error          string          `json:"error,omitempty"`
	LogicalTimeNS  int64           `json:"logical_time_ns"`
	RecordedAt     time.Time       `json:"recorded_at"`
}

func (r IdempotencyRecord) clone() IdempotencyRecord {
	cp := r
	if len(r.Output) > 0 {
		cp.Output = make(json.RawMessage, len(r.Output))
		copy(cp.Output, r.Output)
	}
	return cp
}

func digestBytes(data []byte) string {
	if len(data) == 0 {
		return ""
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func writeLenPrefixed(w interface{ Write([]byte) (int, error) }, s string) {
	_, _ = fmt.Fprintf(w, "%d:%s|", len(s), s)
}
