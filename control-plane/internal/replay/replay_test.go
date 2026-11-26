package replay

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestIdempotencyKey_DeterministicAndCollisionResistant(t *testing.T) {
	k1, err := DeriveIdempotencyKey("tenant-a", "wf-1", "charge_usage", "step-0")
	if err != nil {
		t.Fatalf("DeriveIdempotencyKey failed: %v", err)
	}
	if len(k1) != 64 {
		t.Fatalf("expected 64-char sha256 hex, got len=%d", len(k1))
	}

	k1Repeat, err := DeriveIdempotencyKey("tenant-a", "wf-1", "charge_usage", "step-0")
	if err != nil || k1 != k1Repeat {
		t.Fatalf("expected deterministic key, got %q vs %q (err=%v)", k1, k1Repeat, err)
	}

	// Delimiter-injection collision resistance: ("a:b", "c") must differ from ("a", "b:c").
	inj1, err := DeriveIdempotencyKey("a:b", "c", "act", "s")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	inj2, err := DeriveIdempotencyKey("a", "b:c", "act", "s")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if inj1 == inj2 {
		t.Fatalf("delimiter injection collision detected: %s", inj1)
	}

	// Invalid inputs must fail with sentinel errors.
	if _, err := DeriveIdempotencyKey("", "wf", "act", "s"); !errors.Is(err, ErrInvalidTenant) {
		t.Fatalf("expected ErrInvalidTenant, got %v", err)
	}
	if _, err := DeriveIdempotencyKey("t", "", "act", "s"); !errors.Is(err, ErrInvalidWorkflowID) {
		t.Fatalf("expected ErrInvalidWorkflowID, got %v", err)
	}
	if _, err := DeriveIdempotencyKey("t", "wf", "", "s"); !errors.Is(err, ErrInvalidIdempotencyKey) {
		t.Fatalf("expected ErrInvalidIdempotencyKey, got %v", err)
	}
	if _, err := DeriveIdempotencyKey("t", "wf", "act", ""); !errors.Is(err, ErrInvalidIdempotencyKey) {
		t.Fatalf("expected ErrInvalidIdempotencyKey, got %v", err)
	}
}

func TestHistoryDigestChain_AndTamperDetection(t *testing.T) {
	ctx := context.Background()
	store, err := NewFileCheckpointStore(t.TempDir(), &FileStoreOptions{SyncWrites: true})
	if err != nil {
		t.Fatalf("NewFileCheckpointStore: %v", err)
	}
	eng, err := NewEngine(store, EngineConfig{
		Mode:               ModeCheckpointedReplay,
		CheckpointInterval: 2,
	})
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}

	fixedStart := time.Date(2026, 11, 26, 12, 0, 0, 0, time.UTC)
	report, err := eng.HandleEvent(ctx, TriggerEvent{
		TenantID:     "tenant-sec",
		WorkflowID:   "wf-digest-1",
		WorkflowName: "agent-pipeline",
		Type:         TriggerStart,
		LogicalTime:  fixedStart,
	}, func(_ context.Context, wctx *WorkflowContext, _ TriggerEvent) error {
		for i := 0; i < 3; i++ {
			key, err := DeriveIdempotencyKey(wctx.TenantID(), wctx.WorkflowID(), "tool_call", fmt.Sprintf("step-%d", i))
			if err != nil {
				return err
			}
			if _, err := wctx.ExecuteActivity("tool_call", key, map[string]int{"i": i}, func(_ context.Context, _ ActivityInput) (json.RawMessage, error) {
				return json.RawMessage(fmt.Sprintf(`{"ok":%d}`, i)), nil
			}); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		t.Fatalf("HandleEvent failed: %v", err)
	}
	if report.Status != StatusCompleted || len(report.HistoryDigest) != 64 {
		t.Fatalf("unexpected report: %+v", report)
	}

	// Full load verifies SHA-256 chain.
	loaded, err := store.LoadWorkflow(ctx, "tenant-sec", "wf-digest-1", LoadOptions{UseCheckpoint: false})
	if err != nil {
		t.Fatalf("LoadWorkflow failed: %v", err)
	}
	if got := ComputeHistoryDigest(loaded.Events); got != report.HistoryDigest {
		t.Fatalf("history digest mismatch: got %s want %s", got, report.HistoryDigest)
	}

	// Verify canonical pkg/types conversions.
	wfRun := loaded.Meta.ToWorkflowRun()
	if wfRun.RunID != "wf-digest-1" || wfRun.CompletedSteps != 3 || wfRun.Status != string(StatusCompleted) {
		t.Fatalf("unexpected WorkflowRun conversion: %+v", wfRun)
	}
	exec := loaded.Meta.ToExecution("node-1", "coding-agent")
	if exec.ExecutionID != "wf-digest-1" || exec.ActorID == nil || *exec.ActorID != "tenant-sec" {
		t.Fatalf("unexpected Execution conversion: %+v", exec)
	}
	steps := ToWorkflowSteps(loaded.Events)
	if len(steps) != 3 {
		t.Fatalf("expected 3 WorkflowSteps, got %d", len(steps))
	}
	if ev := loaded.Events[0].ToWorkflowRunEvent(); ev.RunID != "wf-digest-1" || ev.Sequence != 1 {
		t.Fatalf("unexpected WorkflowRunEvent conversion: %+v", ev)
	}

	// Tamper with one byte in events.jsonl and verify ErrCorruptedHistory.
	logPath := filepath.Join(store.WorkflowDir("tenant-sec", "wf-digest-1"), "events.jsonl")
	raw, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	tampered := []byte(string(raw))
	for i := range tampered {
		if tampered[i] == '1' && i > 20 {
			tampered[i] = '2'
			break
		}
	}
	if err := os.WriteFile(logPath, tampered, 0o640); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	if _, err := store.LoadWorkflow(ctx, "tenant-sec", "wf-digest-1", LoadOptions{UseCheckpoint: false}); !errors.Is(err, ErrCorruptedHistory) && !errors.Is(err, ErrTenantAccessDenied) {
		t.Fatalf("expected ErrCorruptedHistory on tampered log, got %v", err)
	}
}

func TestDeterminism_VirtualClockAndSeededRNG_BitExactAcrossCheckpointAndFullReplay(t *testing.T) {
	ctx := context.Background()
	startTime := time.Date(2026, 11, 26, 0, 0, 0, 0, time.UTC)

	type wfState struct {
		NextStep int      `json:"next_step"`
		Tokens   []string `json:"tokens"`
		Times    []int64  `json:"times"`
	}

	runWorkflow := func(mode ReplayMode, injectCrashAt int) (*wfState, string) {
		store, err := NewFileCheckpointStore(t.TempDir(), &FileStoreOptions{SyncWrites: false})
		if err != nil {
			t.Fatalf("NewFileCheckpointStore: %v", err)
		}
		eng, err := NewEngine(store, EngineConfig{
			Mode:               mode,
			CheckpointInterval: 3,
			AutoFireTimers:     true,
		})
		if err != nil {
			t.Fatalf("NewEngine: %v", err)
		}

		crashed := false
		var finalState wfState
		handler := func(_ context.Context, wctx *WorkflowContext, _ TriggerEvent) error {
			var st wfState
			if step, restored, err := wctx.RestoreCheckpoint(&st); err != nil {
				return err
			} else if restored {
				st.NextStep = step
			}

			for i := st.NextStep; i < 8; i++ {
				if injectCrashAt == i && !crashed {
					crashed = true
					return errors.New("injected mid-workflow crash")
				}
				// Draw deterministic RNG values & virtual time inside workflow logic.
				uuid := wctx.RNG().DeterministicUUID()
				jitter := time.Duration(wctx.RNG().Int63n(50)+10) * time.Millisecond
				if err := wctx.Sleep(fmt.Sprintf("timer-%d", i), jitter, &st); err != nil {
					return err
				}
				key, _ := DeriveIdempotencyKey(wctx.TenantID(), wctx.WorkflowID(), "vfs_edit", fmt.Sprintf("step-%d", i))
				out, err := wctx.ExecuteActivity("vfs_edit", key, map[string]any{"uuid": uuid, "step": i}, func(_ context.Context, in ActivityInput) (json.RawMessage, error) {
					return json.Marshal(map[string]any{"uuid": uuid, "at": in.LogicalTime.UnixNano()})
				})
				if err != nil {
					return err
				}
				var decoded map[string]any
				_ = json.Unmarshal(out, &decoded)
				st.Tokens = append(st.Tokens, uuid)
				st.Times = append(st.Times, wctx.Now().UnixNano())
				st.NextStep = i + 1
				if wctx.ShouldCheckpoint() {
					if err := wctx.SaveCheckpoint(fmt.Sprintf("cp-%d", st.NextStep), st.NextStep, &st); err != nil {
						return err
					}
				}
			}
			finalState = st
			return nil
		}

		rep, err := eng.HandleEvent(ctx, TriggerEvent{
			TenantID:    "tenant-det",
			WorkflowID:  "wf-bitexact",
			Type:        TriggerStart,
			LogicalTime: startTime,
		}, handler)
		if injectCrashAt >= 0 {
			if err == nil {
				t.Fatalf("expected crash at step %d", injectCrashAt)
			}
			// Cold restart: brand-new store & engine reading persisted files.
			rep, err = eng.HandleEvent(ctx, TriggerEvent{
				TenantID:    "tenant-det",
				WorkflowID:  "wf-bitexact",
				Type:        TriggerResume,
				LogicalTime: startTime.Add(time.Hour), // different wall/trigger time must NOT affect deterministic replay!
			}, handler)
		}
		if err != nil {
			t.Fatalf("workflow completion failed (mode=%s): %v", mode, err)
		}
		return &finalState, rep.HistoryDigest
	}

	baselineState, _ := runWorkflow(ModeFullHistoryReplay, -1)
	fullCrashState, _ := runWorkflow(ModeFullHistoryReplay, 5)
	cpCrashState, _ := runWorkflow(ModeCheckpointedReplay, 5)

	// Bit-exact RNG UUIDs and virtual timestamps across un-crashed, full-history crash recovery,
	// and checkpoint-restored crash recovery!
	for i := 0; i < 8; i++ {
		if baselineState.Tokens[i] != fullCrashState.Tokens[i] || baselineState.Tokens[i] != cpCrashState.Tokens[i] {
			t.Fatalf("RNG token mismatch at step %d: base=%s full=%s cp=%s",
				i, baselineState.Tokens[i], fullCrashState.Tokens[i], cpCrashState.Tokens[i])
		}
		if baselineState.Times[i] != fullCrashState.Times[i] || baselineState.Times[i] != cpCrashState.Times[i] {
			t.Fatalf("virtual clock mismatch at step %d: base=%d full=%d cp=%d",
				i, baselineState.Times[i], fullCrashState.Times[i], cpCrashState.Times[i])
		}
	}
}

func TestCrashRecovery_AtEveryStep_ColdStartDiskRestart_ZeroDuplicateSideEffects(t *testing.T) {
	ctx := context.Background()
	const totalSteps = 8
	errCrash := errors.New("process SIGKILL simulated")

	type agentState struct {
		NextStep int `json:"next_step"`
	}

	for crashAt := 0; crashAt < totalSteps; crashAt++ {
		t.Run(fmt.Sprintf("crash_at_step_%d", crashAt), func(t *testing.T) {
			dir := t.TempDir()
			sideEffects := make([]int, totalSteps)
			shouldCrash := true

			workflow := func(_ context.Context, wctx *WorkflowContext, _ TriggerEvent) error {
				var st agentState
				if step, ok, err := wctx.RestoreCheckpoint(&st); err != nil {
					return err
				} else if ok {
					st.NextStep = step
				}

				for i := st.NextStep; i < totalSteps; i++ {
					if shouldCrash && i == crashAt {
						return errCrash
					}
					actName := fmt.Sprintf("step_act_%d", i)
					key, err := DeriveIdempotencyKey(wctx.TenantID(), wctx.WorkflowID(), actName, fmt.Sprintf("s%d", i))
					if err != nil {
						return err
					}
					if _, err := wctx.ExecuteActivity(actName, key, map[string]int{"step": i}, func(_ context.Context, _ ActivityInput) (json.RawMessage, error) {
						sideEffects[i]++
						return json.RawMessage(fmt.Sprintf(`{"done":%d}`, i)), nil
					}); err != nil {
						return err
					}
					st.NextStep = i + 1
					if wctx.ShouldCheckpoint() {
						if err := wctx.SaveCheckpoint(fmt.Sprintf("cp-%d", st.NextStep), st.NextStep, &st); err != nil {
							return err
						}
					}
				}
				return nil
			}

			// First process runs until crash.
			store1, err := NewFileCheckpointStore(dir, &FileStoreOptions{SyncWrites: true})
			if err != nil {
				t.Fatalf("store1: %v", err)
			}
			eng1, err := NewEngine(store1, EngineConfig{Mode: ModeCheckpointedReplay, CheckpointInterval: 3})
			if err != nil {
				t.Fatalf("eng1: %v", err)
			}
			rep1, err := eng1.HandleEvent(ctx, TriggerEvent{
				TenantID:   "tenant-recovery",
				WorkflowID: "wf-crash",
				Type:       TriggerStart,
			}, workflow)
			if !errors.Is(err, errCrash) {
				t.Fatalf("expected errCrash, got %v", err)
			}
			if rep1.ExecutedActivities != crashAt {
				t.Fatalf("expected %d executed before crash, got %d", crashAt, rep1.ExecutedActivities)
			}
			if eng1.ActiveLockCount() != 0 {
				t.Fatalf("expected 0 active locks after crash, got %d", eng1.ActiveLockCount())
			}

			// Cold restart: fresh FileCheckpointStore + fresh Engine.
			shouldCrash = false
			store2, err := NewFileCheckpointStore(dir, &FileStoreOptions{SyncWrites: true})
			if err != nil {
				t.Fatalf("store2: %v", err)
			}
			eng2, err := NewEngine(store2, EngineConfig{Mode: ModeCheckpointedReplay, CheckpointInterval: 3})
			if err != nil {
				t.Fatalf("eng2: %v", err)
			}
			rep2, err := eng2.HandleEvent(ctx, TriggerEvent{
				TenantID:   "tenant-recovery",
				WorkflowID: "wf-crash",
				Type:       TriggerResume,
			}, workflow)
			if err != nil {
				t.Fatalf("resume failed: %v", err)
			}
			if rep2.Status != StatusCompleted {
				t.Fatalf("expected completed, got %s", rep2.Status)
			}
			if crashAt >= 3 && !rep2.RestoredCheckpoint {
				t.Fatalf("expected checkpoint restore when crashAt=%d >= CheckpointInterval(3)", crashAt)
			}
			for i, count := range sideEffects {
				if count != 1 {
					t.Fatalf("step %d executed %d times (want strictly 1)", i, count)
				}
			}

			// Duplicate completion webhook delivery is a safe 0-side-effect no-op.
			rep3, err := eng2.HandleEvent(ctx, TriggerEvent{
				TenantID:   "tenant-recovery",
				WorkflowID: "wf-crash",
				Type:       TriggerResume,
			}, workflow)
			if err != nil || rep3.ExecutedActivities != 0 || rep3.Status != StatusCompleted {
				t.Fatalf("duplicate webhook should be no-op, got rep=%+v err=%v", rep3, err)
			}
			if eng2.ActiveLockCount() != 0 {
				t.Fatalf("expected 0 idle locks, got %d", eng2.ActiveLockCount())
			}
		})
	}
}

func TestCrashTailRecovery_MissingCheckpointAndMetaFiles(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	store, err := NewFileCheckpointStore(dir, &FileStoreOptions{SyncWrites: true})
	if err != nil {
		t.Fatalf("NewFileCheckpointStore: %v", err)
	}
	eng, err := NewEngine(store, EngineConfig{Mode: ModeCheckpointedReplay, CheckpointInterval: 1})
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}

	var step1Runs int
	errCrash := errors.New("power loss before checkpoint rename")
	_, err = eng.HandleEvent(ctx, TriggerEvent{
		TenantID:   "tenant-tail",
		WorkflowID: "wf-tail",
		Type:       TriggerStart,
	}, func(_ context.Context, wctx *WorkflowContext, _ TriggerEvent) error {
		key, _ := DeriveIdempotencyKey("tenant-tail", "wf-tail", "charge", "s1")
		if _, err := wctx.ExecuteActivity("charge", key, "p1", func(_ context.Context, _ ActivityInput) (json.RawMessage, error) {
			step1Runs++
			return json.RawMessage(`{"charged":true}`), nil
		}); err != nil {
			return err
		}
		return errCrash
	})
	if !errors.Is(err, errCrash) {
		t.Fatalf("expected errCrash, got %v", err)
	}

	// Simulate crash wiping meta.json and checkpoint.json while WAL events.jsonl survived.
	wfDir := store.WorkflowDir("tenant-tail", "wf-tail")
	_ = os.Remove(filepath.Join(wfDir, "meta.json"))
	_ = os.Remove(filepath.Join(wfDir, "checkpoint.json"))

	store2, _ := NewFileCheckpointStore(dir, &FileStoreOptions{SyncWrites: true})
	eng2, _ := NewEngine(store2, EngineConfig{Mode: ModeCheckpointedReplay, CheckpointInterval: 1})
	rep, err := eng2.HandleEvent(ctx, TriggerEvent{
		TenantID:   "tenant-tail",
		WorkflowID: "wf-tail",
		Type:       TriggerResume,
	}, func(_ context.Context, wctx *WorkflowContext, _ TriggerEvent) error {
		k1, _ := DeriveIdempotencyKey("tenant-tail", "wf-tail", "charge", "s1")
		if _, err := wctx.ExecuteActivity("charge", k1, "p1", func(_ context.Context, _ ActivityInput) (json.RawMessage, error) {
			step1Runs++
			return json.RawMessage(`{"charged":true}`), nil
		}); err != nil {
			return err
		}
		k2, _ := DeriveIdempotencyKey("tenant-tail", "wf-tail", "finalize", "s2")
		_, err := wctx.ExecuteActivity("finalize", k2, "p2", func(_ context.Context, _ ActivityInput) (json.RawMessage, error) {
			return json.RawMessage(`{"ok":true}`), nil
		})
		return err
	})
	if err != nil {
		t.Fatalf("tail recovery failed: %v", err)
	}
	if rep.Status != StatusCompleted || step1Runs != 1 || rep.ReplayedActivities != 1 || rep.ExecutedActivities != 1 {
		t.Fatalf("unexpected tail recovery result: step1Runs=%d rep=%+v", step1Runs, rep)
	}
}

func TestNonDeterminismDetection_AllDivergenceClasses(t *testing.T) {
	ctx := context.Background()

	seedCrash := func(t *testing.T) *Engine {
		t.Helper()
		store := NewMemoryCheckpointStore(Limits{})
		eng, err := NewEngine(store, EngineConfig{Mode: ModeFullHistoryReplay})
		if err != nil {
			t.Fatalf("NewEngine: %v", err)
		}
		_, err = eng.HandleEvent(ctx, TriggerEvent{
			TenantID:   "t-nd",
			WorkflowID: "wf-nd",
			Type:       TriggerStart,
		}, func(_ context.Context, wctx *WorkflowContext, _ TriggerEvent) error {
			if _, err := wctx.ExecuteActivity("read_repo", "k1", map[string]string{"path": "a.go"}, func(_ context.Context, _ ActivityInput) (json.RawMessage, error) {
				return json.RawMessage(`"ok"`), nil
			}); err != nil {
				return err
			}
			return errors.New("crash after step 1")
		})
		if err == nil {
			t.Fatalf("expected crash")
		}
		return eng
	}

	t.Run("different_activity_name", func(t *testing.T) {
		eng := seedCrash(t)
		_, err := eng.HandleEvent(ctx, TriggerEvent{TenantID: "t-nd", WorkflowID: "wf-nd", Type: TriggerResume},
			func(_ context.Context, wctx *WorkflowContext, _ TriggerEvent) error {
				_, err := wctx.ExecuteActivity("delete_repo", "k1", map[string]string{"path": "a.go"}, func(_ context.Context, _ ActivityInput) (json.RawMessage, error) {
					return json.RawMessage(`"bad"`), nil
				})
				return err
			})
		if !errors.Is(err, ErrNonDeterministicReplay) {
			t.Fatalf("expected ErrNonDeterministicReplay, got %v", err)
		}
	})

	t.Run("different_idempotency_key", func(t *testing.T) {
		eng := seedCrash(t)
		_, err := eng.HandleEvent(ctx, TriggerEvent{TenantID: "t-nd", WorkflowID: "wf-nd", Type: TriggerResume},
			func(_ context.Context, wctx *WorkflowContext, _ TriggerEvent) error {
				_, err := wctx.ExecuteActivity("read_repo", "k-mutated", map[string]string{"path": "a.go"}, func(_ context.Context, _ ActivityInput) (json.RawMessage, error) {
					return json.RawMessage(`"bad"`), nil
				})
				return err
			})
		if !errors.Is(err, ErrNonDeterministicReplay) {
			t.Fatalf("expected ErrNonDeterministicReplay, got %v", err)
		}
	})

	t.Run("mutated_input_payload", func(t *testing.T) {
		eng := seedCrash(t)
		_, err := eng.HandleEvent(ctx, TriggerEvent{TenantID: "t-nd", WorkflowID: "wf-nd", Type: TriggerResume},
			func(_ context.Context, wctx *WorkflowContext, _ TriggerEvent) error {
				_, err := wctx.ExecuteActivity("read_repo", "k1", map[string]string{"path": "DIFFERENT.go"}, func(_ context.Context, _ ActivityInput) (json.RawMessage, error) {
					return json.RawMessage(`"bad"`), nil
				})
				return err
			})
		if !errors.Is(err, ErrNonDeterministicReplay) {
			t.Fatalf("expected ErrNonDeterministicReplay, got %v", err)
		}
	})

	t.Run("timer_instead_of_activity", func(t *testing.T) {
		eng := seedCrash(t)
		_, err := eng.HandleEvent(ctx, TriggerEvent{TenantID: "t-nd", WorkflowID: "wf-nd", Type: TriggerResume},
			func(_ context.Context, wctx *WorkflowContext, _ TriggerEvent) error {
				return wctx.Sleep("unexpected-timer", time.Second, nil)
			})
		if !errors.Is(err, ErrNonDeterministicReplay) {
			t.Fatalf("expected ErrNonDeterministicReplay, got %v", err)
		}
	})
}

func TestEventTriggeredSuspendResume_SignalAndTimer_ZeroBackgroundGoroutines(t *testing.T) {
	ctx := context.Background()
	store, err := NewFileCheckpointStore(t.TempDir(), &FileStoreOptions{SyncWrites: true})
	if err != nil {
		t.Fatalf("NewFileCheckpointStore: %v", err)
	}
	eng, err := NewEngine(store, EngineConfig{
		Mode:           ModeCheckpointedReplay,
		AutoFireTimers: false,
	})
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}

	var step1, step2, step3 int
	type state struct {
		Stage int `json:"stage"`
	}

	workflow := func(_ context.Context, wctx *WorkflowContext, _ TriggerEvent) error {
		var st state
		if step, ok, err := wctx.RestoreCheckpoint(&st); err != nil {
			return err
		} else if ok {
			st.Stage = step
		}

		if st.Stage < 1 {
			if _, err := wctx.ExecuteActivity("draft_pr", "k-draft", "in", func(_ context.Context, _ ActivityInput) (json.RawMessage, error) {
				step1++
				return json.RawMessage(`{"pr":85}`), nil
			}); err != nil {
				return err
			}
			st.Stage = 1
			if err := wctx.SaveCheckpoint("after-draft", 1, &st); err != nil {
				return err
			}
		}

		if err := wctx.AwaitSignal("review_approved", &st); err != nil {
			return err
		}

		var approver map[string]string
		ok, err := wctx.SignalPayload("review_approved", &approver)
		if err != nil || !ok || approver["by"] != "kai" {
			return fmt.Errorf("missing signal payload: ok=%v err=%v", ok, err)
		}

		if st.Stage < 2 {
			if err := wctx.Sleep("cooldown-timer", 5*time.Second, &st); err != nil {
				return err
			}
			if _, err := wctx.ExecuteActivity("deploy", "k-deploy", approver, func(_ context.Context, _ ActivityInput) (json.RawMessage, error) {
				step2++
				return json.RawMessage(`{"deployed":true}`), nil
			}); err != nil {
				return err
			}
			st.Stage = 2
		}

		if _, err := wctx.ExecuteActivity("bill_usage", "k-bill", "1", func(_ context.Context, _ ActivityInput) (json.RawMessage, error) {
			step3++
			return json.RawMessage(`{"billed":true}`), nil
		}); err != nil {
			return err
		}
		return nil
	}

	runtime.GC()
	baseGoroutines := runtime.NumGoroutine()

	// Activation 1: suspends on signal with 0 background goroutines.
	rep1, err := eng.HandleEvent(ctx, TriggerEvent{TenantID: "t1", WorkflowID: "wf-susp", Type: TriggerStart}, workflow)
	if err != nil || rep1.Status != StatusSuspended || rep1.AwaitingSignal != "review_approved" {
		t.Fatalf("expected suspended awaiting signal, got rep=%+v err=%v", rep1, err)
	}
	if runtime.NumGoroutine() > baseGoroutines {
		t.Fatalf("goroutine leak detected after signal suspend: base=%d now=%d", baseGoroutines, runtime.NumGoroutine())
	}

	// Activation 2 (fresh Engine): delivers signal -> suspends on timer with 0 background goroutines.
	eng2, _ := NewEngine(store, EngineConfig{Mode: ModeCheckpointedReplay, AutoFireTimers: false})
	rep2, err := eng2.HandleEvent(ctx, TriggerEvent{
		TenantID:   "t1",
		WorkflowID: "wf-susp",
		Type:       TriggerSignal,
		SignalName: "review_approved",
		Payload:    json.RawMessage(`{"by":"kai"}`),
	}, workflow)
	if err != nil || rep2.Status != StatusSuspended || rep2.AwaitingTimer != "cooldown-timer" {
		t.Fatalf("expected suspended awaiting timer, got rep=%+v err=%v", rep2, err)
	}
	if runtime.NumGoroutine() > baseGoroutines {
		t.Fatalf("goroutine leak detected after timer suspend: base=%d now=%d", baseGoroutines, runtime.NumGoroutine())
	}

	// Activation 3 (fresh Engine): timer fires -> completes cleanly.
	eng3, _ := NewEngine(store, EngineConfig{Mode: ModeCheckpointedReplay, AutoFireTimers: false})
	rep3, err := eng3.HandleEvent(ctx, TriggerEvent{
		TenantID:   "t1",
		WorkflowID: "wf-susp",
		Type:       TriggerTimer,
		TimerID:    "cooldown-timer",
	}, workflow)
	if err != nil || rep3.Status != StatusCompleted {
		t.Fatalf("expected completed after timer, got rep=%+v err=%v", rep3, err)
	}
	if step1 != 1 || step2 != 1 || step3 != 1 {
		t.Fatalf("expected strictly once execution (1,1,1), got (%d,%d,%d)", step1, step2, step3)
	}
	if eng3.ActiveLockCount() != 0 || runtime.NumGoroutine() > baseGoroutines {
		t.Fatalf("expected 0 idle locks and no goroutine leak: locks=%d goroutines=%d/%d",
			eng3.ActiveLockCount(), runtime.NumGoroutine(), baseGoroutines)
	}
}

func TestMultiTenantIsolation_SecurityAndConcurrentWebhookStorm(t *testing.T) {
	ctx := context.Background()
	store, err := NewFileCheckpointStore(t.TempDir(), &FileStoreOptions{SyncWrites: false})
	if err != nil {
		t.Fatalf("NewFileCheckpointStore: %v", err)
	}
	eng, err := NewEngine(store, EngineConfig{Mode: ModeCheckpointedReplay, CheckpointInterval: 2})
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}

	const numTenants = 6
	const webhooksPerTenant = 10
	counters := make([]atomic.Int64, numTenants)

	var wg sync.WaitGroup
	errs := make(chan error, numTenants*webhooksPerTenant)

	for ti := 0; ti < numTenants; ti++ {
		tenant := fmt.Sprintf("tenant-%d", ti)
		counter := &counters[ti]
		for w := 0; w < webhooksPerTenant; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				// Every tenant uses the EXACT SAME workflow_id and idempotency_key!
				_, e := eng.HandleEvent(ctx, TriggerEvent{
					TenantID:   tenant,
					WorkflowID: "shared-wf-id",
					Type:       TriggerStart,
				}, func(_ context.Context, wctx *WorkflowContext, _ TriggerEvent) error {
					_, err := wctx.ExecuteActivity("billing_charge", "shared-idem-key", map[string]int{"cents": 500}, func(_ context.Context, _ ActivityInput) (json.RawMessage, error) {
						time.Sleep(2 * time.Millisecond)
						counter.Add(1)
						return json.RawMessage(`{"charged":500}`), nil
					})
					return err
				})
				if e != nil {
					errs <- e
				}
			}()
		}
	}
	wg.Wait()
	close(errs)
	for e := range errs {
		t.Fatalf("concurrent activation error: %v", e)
	}

	for ti := 0; ti < numTenants; ti++ {
		if got := counters[ti].Load(); got != 1 {
			t.Fatalf("tenant-%d billing_charge executed %d times (want strictly 1)", ti, got)
		}
	}

	// Cross-tenant lookup must never see another tenant's workflow or idempotency record.
	otherLoaded, err := store.LoadWorkflow(ctx, "tenant-intruder", "shared-wf-id", LoadOptions{UseCheckpoint: true})
	if err != nil || otherLoaded != nil {
		t.Fatalf("intruder tenant saw foreign workflow: loaded=%+v err=%v", otherLoaded, err)
	}
	if _, found, err := store.LookupIdempotency(ctx, "tenant-intruder", "shared-wf-id", "shared-idem-key"); err != nil || found {
		t.Fatalf("intruder tenant saw foreign idempotency key: found=%v err=%v", found, err)
	}

	// Direct cross-tenant mismatch in SaveCheckpoint / RecordIdempotency must return ErrTenantAccessDenied.
	if err := store.SaveCheckpoint(ctx, "tenant-0", "shared-wf-id", &Checkpoint{
		TenantID:   "tenant-1",
		WorkflowID: "shared-wf-id",
	}); !errors.Is(err, ErrTenantAccessDenied) {
		t.Fatalf("expected ErrTenantAccessDenied, got %v", err)
	}
	if err := store.RecordIdempotency(ctx, "tenant-0", "shared-wf-id", IdempotencyRecord{
		TenantID:       "tenant-1",
		WorkflowID:     "shared-wf-id",
		IdempotencyKey: "k",
	}); !errors.Is(err, ErrTenantAccessDenied) {
		t.Fatalf("expected ErrTenantAccessDenied, got %v", err)
	}
	if eng.ActiveLockCount() != 0 {
		t.Fatalf("expected 0 active locks after storm, got %d", eng.ActiveLockCount())
	}
}

func TestSecurityBounds_PayloadSizeAndHistoryDepthLimits(t *testing.T) {
	ctx := context.Background()
	limits := Limits{
		MaxPayloadBytes: 256,
		MaxHistoryDepth: 6,
	}
	store, err := NewFileCheckpointStore(t.TempDir(), &FileStoreOptions{
		Limits:     limits,
		SyncWrites: false,
	})
	if err != nil {
		t.Fatalf("NewFileCheckpointStore: %v", err)
	}
	eng, err := NewEngine(store, EngineConfig{
		Mode:   ModeCheckpointedReplay,
		Limits: limits,
	})
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}

	// Oversized input payload rejected.
	big := make([]byte, 512)
	for i := range big {
		big[i] = 'x'
	}
	_, err = eng.HandleEvent(ctx, TriggerEvent{
		TenantID:   "t-bounds",
		WorkflowID: "wf-big-payload",
		Type:       TriggerStart,
	}, func(_ context.Context, wctx *WorkflowContext, _ TriggerEvent) error {
		_, err := wctx.ExecuteActivity("upload", "k1", string(big), func(_ context.Context, _ ActivityInput) (json.RawMessage, error) {
			return json.RawMessage(`"ok"`), nil
		})
		return err
	})
	if !errors.Is(err, ErrPayloadTooLarge) {
		t.Fatalf("expected ErrPayloadTooLarge, got %v", err)
	}

	// Unbounded history depth rejected.
	_, err = eng.HandleEvent(ctx, TriggerEvent{
		TenantID:   "t-bounds",
		WorkflowID: "wf-deep",
		Type:       TriggerStart,
	}, func(_ context.Context, wctx *WorkflowContext, _ TriggerEvent) error {
		for i := 0; i < 10; i++ {
			if _, err := wctx.ExecuteActivity("step", fmt.Sprintf("k-%d", i), i, func(_ context.Context, _ ActivityInput) (json.RawMessage, error) {
				return json.RawMessage(`1`), nil
			}); err != nil {
				return err
			}
		}
		return nil
	})
	if !errors.Is(err, ErrHistoryDepthExceeded) {
		t.Fatalf("expected ErrHistoryDepthExceeded, got %v", err)
	}
}

func TestVFSContentRefFormat(t *testing.T) {
	ref := VFSContentRef([]byte("binary-patch-artifact"))
	if len(ref) != len("vfs://sha256/")+64 {
		t.Fatalf("unexpected VFS ref format: %s", ref)
	}
	if ref != VFSContentRef([]byte("binary-patch-artifact")) {
		t.Fatalf("VFSContentRef must be deterministic")
	}
}
