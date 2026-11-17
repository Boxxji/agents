package durable

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
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

	"github.com/stretchr/testify/require"
)

func TestIdempotencyKey_DeterministicSHA256Format(t *testing.T) {
	key, err := IdempotencyKey("hanzo", "run-42", 3, "apply-patch")
	require.NoError(t, err)

	sum := sha256.Sum256([]byte("hanzo:run-42:3:apply-patch"))
	require.Equal(t, hex.EncodeToString(sum[:]), key)
	require.Len(t, key, 64)

	// Changing any component produces a distinct key.
	kOrg, err := IdempotencyKey("other-org", "run-42", 3, "apply-patch")
	require.NoError(t, err)
	require.NotEqual(t, key, kOrg)

	kRun, err := IdempotencyKey("hanzo", "run-43", 3, "apply-patch")
	require.NoError(t, err)
	require.NotEqual(t, key, kRun)

	kRound, err := IdempotencyKey("hanzo", "run-42", 4, "apply-patch")
	require.NoError(t, err)
	require.NotEqual(t, key, kRound)

	kStep, err := IdempotencyKey("hanzo", "run-42", 3, "verify-tests")
	require.NoError(t, err)
	require.NotEqual(t, key, kStep)

	// Invalid components must fail explicitly.
	_, err = IdempotencyKey("", "run-42", 0, "step")
	require.ErrorIs(t, err, ErrInvalidKeyComponent)
	_, err = IdempotencyKey("hanzo", "", 0, "step")
	require.ErrorIs(t, err, ErrInvalidKeyComponent)
	_, err = IdempotencyKey("hanzo", "run-42", -1, "step")
	require.ErrorIs(t, err, ErrInvalidKeyComponent)
	_, err = IdempotencyKey("hanzo", "run-42", 0, "")
	require.ErrorIs(t, err, ErrInvalidKeyComponent)
}

func TestEngine_InterruptAtEveryStepResumesExactlyOnce_FileJournal(t *testing.T) {
	ctx := context.Background()
	errCrash := errors.New("simulated process crash")

	steps := []string{"fetch-context", "plan-edit", "apply-diff", "run-tests", "publish-report"}

	// Interrupt at every step index (0..len(steps)-1), create a brand-new Engine over
	// the persisted FileJournalStore after each crash, and verify each side-effect runs once.
	for crashAt := 0; crashAt < len(steps); crashAt++ {
		t.Run(fmt.Sprintf("crash_before_step_%d", crashAt), func(t *testing.T) {
			journalDir := t.TempDir()
			sideEffects := make([]int, len(steps))
			seenKeys := make([]string, len(steps))
			shouldCrash := true

			workflow := func(_ context.Context, wctx *WorkflowContext, _ ActivationEvent) error {
				for idx, stepID := range steps {
					if idx > 0 {
						if err := wctx.NextRound(); err != nil {
							return err
						}
					}
					if shouldCrash && idx == crashAt {
						return errCrash
					}
					out, err := wctx.ExecuteActivity(stepID, map[string]any{"idx": idx}, func(_ context.Context, in ActivityInput) (json.RawMessage, error) {
						sideEffects[idx]++
						seenKeys[idx] = in.IdempotencyKey
						return json.Marshal(map[string]any{"step": in.StepID, "round": in.Round})
					})
					if err != nil {
						return err
					}
					var decoded map[string]any
					if err := json.Unmarshal(out, &decoded); err != nil {
						return err
					}
					if err := wctx.Checkpoint("after-"+stepID, decoded); err != nil {
						return err
					}
				}
				return nil
			}

			// First activation hits simulated crash.
			store1, err := NewFileJournalStore(journalDir)
			require.NoError(t, err)
			engine1, err := NewEngine(store1)
			require.NoError(t, err)

			res1, err := engine1.Activate(ctx, ActivationEvent{
				Org:          "org-wave1",
				RunID:        "run-crash-recovery",
				WorkflowName: "coding-agent",
				Type:         EventStart,
			}, workflow)
			require.ErrorIs(t, err, errCrash)
			require.Equal(t, RunStatusFailed, res1.Status)
			require.Equal(t, crashAt, res1.ExecutedActivities)

			// Simulate process restart: new FileJournalStore + new Engine reading disk state.
			shouldCrash = false
			store2, err := NewFileJournalStore(journalDir)
			require.NoError(t, err)
			engine2, err := NewEngine(store2)
			require.NoError(t, err)

			res2, err := engine2.Activate(ctx, ActivationEvent{
				Org:          "org-wave1",
				RunID:        "run-crash-recovery",
				WorkflowName: "coding-agent",
				Type:         EventResume,
			}, workflow)
			require.NoError(t, err)
			require.Equal(t, RunStatusCompleted, res2.Status)
			require.Equal(t, crashAt, res2.ReplayedActivities)
			require.Equal(t, len(steps)-crashAt, res2.ExecutedActivities)

			for idx, count := range sideEffects {
				require.Equal(t, 1, count, "step %s must execute strictly once", steps[idx])
				require.Equal(t, MustIdempotencyKey("org-wave1", "run-crash-recovery", idx, steps[idx]), seenKeys[idx])
			}

			// Re-delivering completion/resume event after completion is a safe no-op.
			res3, err := engine2.Activate(ctx, ActivationEvent{
				Org:   "org-wave1",
				RunID: "run-crash-recovery",
				Type:  EventResume,
			}, workflow)
			require.NoError(t, err)
			require.Equal(t, RunStatusCompleted, res3.Status)
			require.Equal(t, 0, res3.ExecutedActivities)
			require.Equal(t, len(steps), res3.ReplayedActivities)
		})
	}
}

func TestEngine_EventTriggeredSuspendAndCheckpointResume_ZeroDaemons(t *testing.T) {
	ctx := context.Background()
	store, err := NewFileJournalStore(t.TempDir())
	require.NoError(t, err)

	engine, err := NewEngine(store)
	require.NoError(t, err)

	var step1Count, step2Count int
	type approvalPayload struct {
		ApprovedBy string `json:"approved_by"`
	}

	workflow := func(_ context.Context, wctx *WorkflowContext, _ ActivationEvent) error {
		var savedStage string
		if ok, err := wctx.RestoreCheckpoint("before-approval", &savedStage); err != nil {
			return err
		} else if !ok {
			if _, err := wctx.ExecuteActivity("prepare-pr", "draft-1", func(_ context.Context, _ ActivityInput) (json.RawMessage, error) {
				step1Count++
				return json.RawMessage(`{"pr":101}`), nil
			}); err != nil {
				return err
			}
			if err := wctx.Checkpoint("before-approval", "awaiting-review"); err != nil {
				return err
			}
		}

		if err := wctx.AwaitEvent("human-approval", "waiting"); err != nil {
			return err
		}

		var sig approvalPayload
		ok, err := wctx.SignalPayload("human-approval", &sig)
		if err != nil {
			return err
		}
		if !ok || sig.ApprovedBy == "" {
			return errors.New("missing approval payload")
		}

		if err := wctx.NextRound(); err != nil {
			return err
		}
		_, err = wctx.ExecuteActivity("merge-pr", sig.ApprovedBy, func(_ context.Context, _ ActivityInput) (json.RawMessage, error) {
			step2Count++
			return json.RawMessage(`{"merged":true}`), nil
		})
		return err
	}

	baselineGoroutines := runtime.NumGoroutine()

	// Activation 1: runs step 1, persists checkpoint, and suspends without keeping a daemon alive.
	res1, err := engine.Activate(ctx, ActivationEvent{
		Org:   "org-a",
		RunID: "run-approval-1",
		Type:  EventStart,
	}, workflow)
	require.NoError(t, err)
	require.Equal(t, RunStatusSuspended, res1.Status)
	require.Equal(t, "human-approval", res1.AwaitingSignal)
	require.Equal(t, 1, step1Count)
	require.Equal(t, 0, step2Count)
	require.LessOrEqual(t, runtime.NumGoroutine(), baselineGoroutines+1)

	// Activation 2 (from fresh Engine after restart): signal wakes run, restores checkpoint, runs step 2 once.
	engineAfterRestart, err := NewEngine(store)
	require.NoError(t, err)
	sigJSON, _ := json.Marshal(approvalPayload{ApprovedBy: "kai-reviewer"})
	res2, err := engineAfterRestart.Activate(ctx, ActivationEvent{
		Org:        "org-a",
		RunID:      "run-approval-1",
		Type:       EventSignal,
		SignalName: "human-approval",
		Payload:    sigJSON,
	}, workflow)
	require.NoError(t, err)
	require.Equal(t, RunStatusCompleted, res2.Status)
	require.Equal(t, 1, step1Count)
	require.Equal(t, 1, step2Count)

	persisted, err := store.GetRun(ctx, "org-a", "run-approval-1")
	require.NoError(t, err)
	require.NotNil(t, persisted)
	wfRun := persisted.ToWorkflowRun()
	require.Equal(t, string(RunStatusCompleted), wfRun.Status)
	require.Equal(t, 2, wfRun.CompletedSteps)
	require.GreaterOrEqual(t, len(persisted.Entries), 4)
	require.Equal(t, "run-approval-1", persisted.Entries[0].ToWorkflowRunEvent().RunID)
}

func TestEngine_ConcurrentDuplicateEventsAndTenantIsolation(t *testing.T) {
	ctx := context.Background()
	store := NewMemoryJournalStore()
	engine, err := NewEngine(store)
	require.NoError(t, err)

	var orgACount, orgBCount atomic.Int64

	makeWorkflow := func(counter *atomic.Int64) WorkflowHandler {
		return func(_ context.Context, wctx *WorkflowContext, _ ActivationEvent) error {
			_, err := wctx.ExecuteActivity("charge-step", map[string]int{"amount": 10}, func(_ context.Context, _ ActivityInput) (json.RawMessage, error) {
				time.Sleep(5 * time.Millisecond)
				counter.Add(1)
				return json.RawMessage(`{"ok":true}`), nil
			})
			return err
		}
	}

	const goroutines = 12
	var wg sync.WaitGroup
	errs := make(chan error, goroutines*2)
	for i := 0; i < goroutines; i++ {
		wg.Add(2)
		go func() {
			defer wg.Done()
			_, e := engine.Activate(ctx, ActivationEvent{
				Org:   "tenant-alpha",
				RunID: "shared-run-id",
				Type:  EventStart,
			}, makeWorkflow(&orgACount))
			if e != nil {
				errs <- e
			}
		}()
		go func() {
			defer wg.Done()
			_, e := engine.Activate(ctx, ActivationEvent{
				Org:   "tenant-beta",
				RunID: "shared-run-id",
				Type:  EventStart,
			}, makeWorkflow(&orgBCount))
			if e != nil {
				errs <- e
			}
		}()
	}
	wg.Wait()
	close(errs)
	for e := range errs {
		require.NoError(t, e)
	}

	// Same RunID and StepID across two orgs must run once per org, never colliding or leaking.
	require.Equal(t, int64(1), orgACount.Load())
	require.Equal(t, int64(1), orgBCount.Load())
}

func TestEngine_DetectsNonDeterministicReplay(t *testing.T) {
	ctx := context.Background()
	store := NewMemoryJournalStore()
	engine, err := NewEngine(store)
	require.NoError(t, err)

	crashErr := errors.New("stop after step 1")
	_, err = engine.Activate(ctx, ActivationEvent{
		Org:   "org-1",
		RunID: "run-nd-1",
		Type:  EventStart,
	}, func(_ context.Context, wctx *WorkflowContext, _ ActivationEvent) error {
		if _, err := wctx.ExecuteActivity("step-a", "v1", func(_ context.Context, _ ActivityInput) (json.RawMessage, error) {
			return json.RawMessage(`1`), nil
		}); err != nil {
			return err
		}
		return crashErr
	})
	require.ErrorIs(t, err, crashErr)

	// Replaying with altered step name must fail with ErrNonDeterministicReplay.
	_, err = engine.Activate(ctx, ActivationEvent{
		Org:   "org-1",
		RunID: "run-nd-1",
		Type:  EventResume,
	}, func(_ context.Context, wctx *WorkflowContext, _ ActivationEvent) error {
		_, err := wctx.ExecuteActivity("step-b-mutated", "v1", func(_ context.Context, _ ActivityInput) (json.RawMessage, error) {
			return json.RawMessage(`2`), nil
		})
		return err
	})
	require.ErrorIs(t, err, ErrNonDeterministicReplay)
}

func TestFileJournalStore_RecoversJournalTailWhenStateMissing(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	store, err := NewFileJournalStore(dir)
	require.NoError(t, err)

	engine, err := NewEngine(store)
	require.NoError(t, err)

	var step1Execs int
	crashErr := errors.New("crash after step 1")
	_, err = engine.Activate(ctx, ActivationEvent{
		Org:   "org-tail",
		RunID: "run-tail",
		Type:  EventStart,
	}, func(_ context.Context, wctx *WorkflowContext, _ ActivationEvent) error {
		if _, err := wctx.ExecuteActivity("step-1", "in", func(_ context.Context, _ ActivityInput) (json.RawMessage, error) {
			step1Execs++
			return json.RawMessage(`"done-1"`), nil
		}); err != nil {
			return err
		}
		return crashErr
	})
	require.ErrorIs(t, err, crashErr)

	// Remove state.json to simulate crash right before snapshot write while journal.jsonl survived.
	stateFile := filepath.Join(store.runDir("org-tail", "run-tail"), "state.json")
	require.NoError(t, os.Remove(stateFile))

	recoveredStore, err := NewFileJournalStore(dir)
	require.NoError(t, err)
	recoveredEngine, err := NewEngine(recoveredStore)
	require.NoError(t, err)

	res, err := recoveredEngine.Activate(ctx, ActivationEvent{
		Org:   "org-tail",
		RunID: "run-tail",
		Type:  EventResume,
	}, func(_ context.Context, wctx *WorkflowContext, _ ActivationEvent) error {
		if _, err := wctx.ExecuteActivity("step-1", "in", func(_ context.Context, _ ActivityInput) (json.RawMessage, error) {
			step1Execs++
			return json.RawMessage(`"done-1"`), nil
		}); err != nil {
			return err
		}
		_, err := wctx.ExecuteActivity("step-2", "in2", func(_ context.Context, _ ActivityInput) (json.RawMessage, error) {
			return json.RawMessage(`"done-2"`), nil
		})
		return err
	})
	require.NoError(t, err)
	require.Equal(t, RunStatusCompleted, res.Status)
	require.Equal(t, 1, step1Execs)
	require.Equal(t, 1, res.ReplayedActivities)
	require.Equal(t, 1, res.ExecutedActivities)
}
