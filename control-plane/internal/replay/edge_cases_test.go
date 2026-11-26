package replay

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestTransientActivityFailure_RetryAndSubsequentCrashReplay(t *testing.T) {
	ctx := context.Background()
	store := NewMemoryCheckpointStore(Limits{})
	eng, err := NewEngine(store, EngineConfig{Mode: ModeFullHistoryReplay})
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}

	var step0Runs, step1Attempts, step2Runs int
	step1FailOnce := true
	step2CrashOnce := true
	errTransient := errors.New("503 upstream timeout")
	errCrash := errors.New("worker preempted")

	workflow := func(_ context.Context, wctx *WorkflowContext, _ TriggerEvent) error {
		if _, err := wctx.ExecuteActivity("read_context", "k0", "c0", func(_ context.Context, _ ActivityInput) (json.RawMessage, error) {
			step0Runs++
			return json.RawMessage(`{"ctx":true}`), nil
		}); err != nil {
			return err
		}
		if _, err := wctx.ExecuteActivity("call_model", "k1", "p1", func(_ context.Context, _ ActivityInput) (json.RawMessage, error) {
			step1Attempts++
			if step1FailOnce {
				step1FailOnce = false
				return nil, errTransient
			}
			return json.RawMessage(`{"patch":"diff"}`), nil
		}); err != nil {
			return err
		}
		if step2CrashOnce {
			step2CrashOnce = false
			return errCrash
		}
		_, err := wctx.ExecuteActivity("write_vfs", "k2", "p2", func(_ context.Context, _ ActivityInput) (json.RawMessage, error) {
			step2Runs++
			return json.RawMessage(`{"written":true}`), nil
		})
		return err
	}

	// Attempt 1: step0 succeeds, step1 fails transiently.
	if _, err := eng.HandleEvent(ctx, TriggerEvent{TenantID: "t1", WorkflowID: "wf-retry", Type: TriggerStart}, workflow); !errors.Is(err, errTransient) {
		t.Fatalf("expected errTransient, got %v", err)
	}
	// Attempt 2: step0 replays, step1 succeeds on retry, worker preempted before step2.
	if _, err := eng.HandleEvent(ctx, TriggerEvent{TenantID: "t1", WorkflowID: "wf-retry", Type: TriggerResume}, workflow); !errors.Is(err, errCrash) {
		t.Fatalf("expected errCrash, got %v", err)
	}
	// Attempt 3: step0 and step1 both replay (collapsing earlier failed attempt), step2 runs once.
	rep, err := eng.HandleEvent(ctx, TriggerEvent{TenantID: "t1", WorkflowID: "wf-retry", Type: TriggerResume}, workflow)
	if err != nil {
		t.Fatalf("final resume failed: %v", err)
	}
	if rep.Status != StatusCompleted || step0Runs != 1 || step1Attempts != 2 || step2Runs != 1 {
		t.Fatalf("unexpected counts: status=%s s0=%d s1=%d s2=%d", rep.Status, step0Runs, step1Attempts, step2Runs)
	}
	if rep.ReplayedActivities != 2 || rep.ExecutedActivities != 1 {
		t.Fatalf("expected 2 replayed + 1 executed, got %+v", rep)
	}
}

func TestTamperedDiskTenantMetadata_RejectedWithTenantAccessDenied(t *testing.T) {
	ctx := context.Background()
	store, err := NewFileCheckpointStore(t.TempDir(), &FileStoreOptions{SyncWrites: false})
	if err != nil {
		t.Fatalf("NewFileCheckpointStore: %v", err)
	}
	eng, err := NewEngine(store, EngineConfig{Mode: ModeCheckpointedReplay, CheckpointInterval: 1})
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}

	_, err = eng.HandleEvent(ctx, TriggerEvent{
		TenantID:   "tenant-honest",
		WorkflowID: "wf-1",
		Type:       TriggerStart,
	}, func(_ context.Context, wctx *WorkflowContext, _ TriggerEvent) error {
		if _, err := wctx.ExecuteActivity("op", "k1", "x", func(_ context.Context, _ ActivityInput) (json.RawMessage, error) {
			return json.RawMessage(`"ok"`), nil
		}); err != nil {
			return err
		}
		return wctx.SaveCheckpoint("cp1", 1, map[string]int{"s": 1})
	})
	if err != nil {
		t.Fatalf("HandleEvent: %v", err)
	}

	// Tamper tenant_id inside meta.json on disk.
	metaPath := filepath.Join(store.WorkflowDir("tenant-honest", "wf-1"), "meta.json")
	raw, err := os.ReadFile(metaPath)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	tampered := strings.Replace(string(raw), "tenant-honest", "tenant-forged", 1)
	if err := os.WriteFile(metaPath, []byte(tampered), 0o640); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	if _, err := store.LoadWorkflow(ctx, "tenant-honest", "wf-1", LoadOptions{UseCheckpoint: true}); !errors.Is(err, ErrTenantAccessDenied) {
		t.Fatalf("expected ErrTenantAccessDenied on forged meta.json, got %v", err)
	}
}

func TestImperativeTopToBottomWorkflow_WithExistingCheckpoint_DeduplicatesPreCheckpointSteps(t *testing.T) {
	ctx := context.Background()
	store := NewMemoryCheckpointStore(Limits{})
	eng, err := NewEngine(store, EngineConfig{Mode: ModeCheckpointedReplay, CheckpointInterval: 1})
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}

	var step1Runs, step2Runs int
	crash := true
	errCrash := errors.New("crash before step 2")

	// Imperative workflow that saves a checkpoint after step 1 but does NOT call RestoreCheckpoint.
	workflow := func(_ context.Context, wctx *WorkflowContext, _ TriggerEvent) error {
		if _, err := wctx.ExecuteActivity("step1", "k1", "in1", func(_ context.Context, _ ActivityInput) (json.RawMessage, error) {
			step1Runs++
			return json.RawMessage(`"r1"`), nil
		}); err != nil {
			return err
		}
		if wctx.ShouldCheckpoint() {
			if err := wctx.SaveCheckpoint("cp1", 1, map[string]int{"done": 1}); err != nil {
				return err
			}
		}
		if crash {
			crash = false
			return errCrash
		}
		_, err := wctx.ExecuteActivity("step2", "k2", "in2", func(_ context.Context, _ ActivityInput) (json.RawMessage, error) {
			step2Runs++
			return json.RawMessage(`"r2"`), nil
		})
		return err
	}

	if _, err := eng.HandleEvent(ctx, TriggerEvent{TenantID: "t", WorkflowID: "wf-imp", Type: TriggerStart}, workflow); !errors.Is(err, errCrash) {
		t.Fatalf("expected errCrash, got %v", err)
	}
	rep, err := eng.HandleEvent(ctx, TriggerEvent{TenantID: "t", WorkflowID: "wf-imp", Type: TriggerResume}, workflow)
	if err != nil {
		t.Fatalf("resume failed: %v", err)
	}
	if rep.Status != StatusCompleted || step1Runs != 1 || step2Runs != 1 {
		t.Fatalf("expected strictly once execution, got s1=%d s2=%d rep=%+v", step1Runs, step2Runs, rep)
	}

	bytesUsed, err := store.StorageBytes(ctx, "t", "wf-imp")
	if err != nil || bytesUsed <= 0 {
		t.Fatalf("expected positive StorageBytes, got %d err=%v", bytesUsed, err)
	}
}

func TestStatelessRetryBaseline_DuplicatesSideEffectsOnCrash(t *testing.T) {
	ctx := context.Background()
	eng, err := NewEngine(nil, EngineConfig{Mode: ModeStatelessRetry})
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}

	var billed int
	crashOnce := true
	errCrash := errors.New("crash after charge")

	workflow := func(_ context.Context, wctx *WorkflowContext, _ TriggerEvent) error {
		if _, err := wctx.ExecuteActivity("billing_charge", "k-bill", 100, func(_ context.Context, _ ActivityInput) (json.RawMessage, error) {
			billed++
			return json.RawMessage(`{"ok":true}`), nil
		}); err != nil {
			return err
		}
		if crashOnce {
			crashOnce = false
			return errCrash
		}
		return nil
	}

	if _, err := eng.HandleEvent(ctx, TriggerEvent{
		TenantID:    "t-stateless",
		WorkflowID:  "wf-s",
		Type:        TriggerStart,
		LogicalTime: time.Unix(1700000000, 0).UTC(),
	}, workflow); !errors.Is(err, errCrash) {
		t.Fatalf("expected errCrash, got %v", err)
	}
	rep, err := eng.HandleEvent(ctx, TriggerEvent{
		TenantID:    "t-stateless",
		WorkflowID:  "wf-s",
		Type:        TriggerResume,
		LogicalTime: time.Unix(1700000001, 0).UTC(),
	}, workflow)
	if err != nil || rep.Status != StatusCompleted {
		t.Fatalf("expected completion on retry, got rep=%+v err=%v", rep, err)
	}
	if billed != 2 {
		t.Fatalf("expected naive stateless retry to duplicate side effect (billed=2), got %d", billed)
	}
}
