package storage

import (
	"errors"
	"testing"
)

func TestGetAgentMissingNodeWrapsSentinel(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	_, err := ls.GetAgent(ctx, "does-not-exist")
	if !errors.Is(err, ErrAgentNodeNotFound) {
		t.Fatalf("err = %v, want ErrAgentNodeNotFound", err)
	}
}
