// Package replay implements clean-room, event-triggered durable workflow replay
// with persisted checkpoints, deterministic virtual time/RNG, SHA-256 chained event
// histories, and tenant-scoped idempotency_key deduplication — requiring zero
// always-on daemon or background polling processes (HIP-1347).
package replay

import "errors"

var (
	// ErrNonDeterministicReplay indicates that workflow replay diverged from the
	// persisted event history (different activity name, idempotency key, input
	// digest, command type, or execution ordering).
	ErrNonDeterministicReplay = errors.New("replay: non-deterministic workflow replay")

	// ErrTenantAccessDenied indicates an attempted cross-tenant read, write, or
	// tampered tenant identity on a persisted workflow or idempotency record.
	ErrTenantAccessDenied = errors.New("replay: tenant access denied")

	// ErrInvalidTenant indicates an empty or whitespace-only tenant_id.
	ErrInvalidTenant = errors.New("replay: invalid tenant_id")

	// ErrInvalidWorkflowID indicates an empty or whitespace-only workflow_id.
	ErrInvalidWorkflowID = errors.New("replay: invalid workflow_id")

	// ErrInvalidIdempotencyKey indicates an empty or invalid idempotency key component.
	ErrInvalidIdempotencyKey = errors.New("replay: invalid idempotency_key")

	// ErrInvalidTrigger indicates a malformed trigger event or missing workflow handler.
	ErrInvalidTrigger = errors.New("replay: invalid trigger event")

	// ErrPayloadTooLarge indicates an activity input/output or checkpoint payload
	// exceeded the configured MaxPayloadBytes limit.
	ErrPayloadTooLarge = errors.New("replay: payload exceeds max allowed size")

	// ErrHistoryDepthExceeded indicates the workflow event log exceeded the
	// configured MaxHistoryDepth bound.
	ErrHistoryDepthExceeded = errors.New("replay: event history depth limit exceeded")

	// ErrCorruptedHistory indicates broken monotonic sequences or a mismatched
	// SHA-256 history digest chain.
	ErrCorruptedHistory = errors.New("replay: corrupted event history")

	// ErrSuspended signals that a workflow durably checkpointed and paused waiting
	// for an external signal, timer, or callback event without holding any process
	// resources or background goroutines.
	ErrSuspended = errors.New("replay: workflow suspended awaiting external event")

	// ErrWorkflowNotFound indicates no persisted workflow exists for (tenant_id, workflow_id).
	ErrWorkflowNotFound = errors.New("replay: workflow not found")
)
