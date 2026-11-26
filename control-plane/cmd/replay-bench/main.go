// Command replay-bench runs reproducible matched-baseline benchmarks comparing:
//  1. stateless_retry           (naive stateless retry — re-executes completed steps)
//  2. full_history_replay       (Temporal-style uncompacted full-history replay from event 1)
//  3. hanzo_checkpointed_replay (Hanzo event-triggered periodic checkpoint + O(1) log seek + idempotency ledger)
//
// All timings are real wall-clock measurements against local disk (FileCheckpointStore)
// with zero synthetic latency formulas and zero always-on background goroutines (HIP-1347).
package main

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/hanzoai/agents/control-plane/internal/replay"
)

// BenchmarkReport is the top-level machine-readable JSON output emitted by replay-bench.
type BenchmarkReport struct {
	Benchmark          string                 `json:"benchmark"`
	HIP                string                 `json:"hip"`
	Issue              string                 `json:"issue"`
	GeneratedAtUTC     string                 `json:"generated_at_utc"`
	HostEnvironment    HostEnvironment        `json:"host_environment"`
	HonestyNotice      string                 `json:"honesty_notice"`
	Config             BenchConfig            `json:"config"`
	ZeroDaemonCheck    ZeroDaemonVerification `json:"zero_daemon_verification"`
	CrashRecoveryFsync []CrashRecoveryCell    `json:"crash_recovery_fsync"`
	RecoveryScaling    []CrashRecoveryCell    `json:"recovery_scaling_pagecache"`
	WebhookStorm       []WebhookStormCell     `json:"webhook_storm_multitenant"`
	NegativeRegimes    NegativeRegimesReport  `json:"negative_regimes"`
}

type HostEnvironment struct {
	GOOS      string `json:"goos"`
	GOARCH    string `json:"goarch"`
	GoVersion string `json:"go_version"`
	NumCPU    int    `json:"num_cpu"`
}

type BenchConfig struct {
	Seed               int64 `json:"seed"`
	CheckpointInterval int   `json:"checkpoint_interval"`
	FsyncTrials        int   `json:"fsync_trials"`
	ScalingTrials      int   `json:"scaling_trials"`
	Tenants            int   `json:"tenants"`
	DuplicateWebhooks  int   `json:"duplicate_webhooks_per_tenant"`
	TrivialTrials      int   `json:"trivial_trials"`
	NegativeTrials     int   `json:"negative_trials"`
	FsyncSteps         []int `json:"fsync_steps"`
	ScalingSteps       []int `json:"scaling_steps"`
	NR3Steps           int   `json:"nr3_steps"`
}

type ZeroDaemonVerification struct {
	BaselineGoroutines int  `json:"baseline_goroutines"`
	PostRunGoroutines  int  `json:"post_run_goroutines"`
	GoroutineDelta     int  `json:"goroutine_delta"`
	IdleActiveLocks    int  `json:"idle_active_locks"`
	ZeroDaemonsPassed  bool `json:"zero_daemons_passed"`
}

type LatencyStatsMS struct {
	P50  float64 `json:"p50_ms"`
	P95  float64 `json:"p95_ms"`
	P99  float64 `json:"p99_ms"`
	Mean float64 `json:"mean_ms"`
}

type CrashRecoveryCell struct {
	Mode                   string         `json:"mode"`
	Steps                  int            `json:"steps"`
	CrashStep              int            `json:"crash_step"`
	Fsync                  bool           `json:"fsync"`
	CheckpointInterval     int            `json:"checkpoint_interval"`
	Trials                 int            `json:"trials"`
	TimeToResumeLatency    LatencyStatsMS `json:"time_to_resume_latency"`
	RecoveryLatency        LatencyStatsMS `json:"recovery_latency"`
	TotalEndToEndLatency   LatencyStatsMS `json:"total_e2e_latency"`
	DuplicateSideEffects   int            `json:"duplicate_side_effects"`
	TotalUniqueSideEffects int            `json:"total_unique_side_effects"`
	DuplicateSideEffPct    float64        `json:"duplicate_side_effect_pct"`
	MeanEventsLoaded       float64        `json:"mean_events_loaded"`
	MeanBytesReadOnLoad    float64        `json:"mean_bytes_read_on_load"`
	StorageBytesPerStep    float64        `json:"storage_bytes_per_step"`
	RestoredCheckpointRate float64        `json:"restored_checkpoint_rate_pct"`
}

type WebhookStormCell struct {
	Mode                  string         `json:"mode"`
	Tenants               int            `json:"tenants"`
	DeliveriesPerTenant   int            `json:"deliveries_per_tenant"`
	StepsPerWorkflow      int            `json:"steps_per_workflow"`
	ActivationLatency     LatencyStatsMS `json:"activation_latency"`
	ExpectedSideEffects   int            `json:"expected_side_effects"`
	ObservedSideEffects   int            `json:"observed_side_effects"`
	DuplicateSideEffects  int            `json:"duplicate_side_effects"`
	DuplicateSideEffPct   float64        `json:"duplicate_side_effect_pct"`
	CrossTenantViolations int            `json:"cross_tenant_violations"`
	IdleActiveLocks       int            `json:"idle_active_locks"`
}

type NegativeRegimesReport struct {
	TrivialOneStep  []TrivialOneStepCell  `json:"nr1_trivial_one_step"`
	LargeBinaryBlob []LargeBinaryBlobCell `json:"nr2_large_binary_payload"`
	CheckpointFreq  []CheckpointFreqCell  `json:"nr3_checkpoint_frequency"`
}

type TrivialOneStepCell struct {
	Mode                string         `json:"mode"`
	Steps               int            `json:"steps"`
	Fsync               bool           `json:"fsync"`
	Trials              int            `json:"trials"`
	HappyPathLatency    LatencyStatsMS `json:"happy_path_latency"`
	StorageBytesPerStep float64        `json:"storage_bytes_per_step"`
}

type LargeBinaryBlobCell struct {
	PayloadStrategy     string         `json:"payload_strategy"`
	Steps               int            `json:"steps"`
	BlobBytesPerStep    int            `json:"blob_bytes_per_step"`
	Fsync               bool           `json:"fsync"`
	Trials              int            `json:"trials"`
	HappyPathLatency    LatencyStatsMS `json:"happy_path_latency"`
	RecoveryLatency     LatencyStatsMS `json:"recovery_latency"`
	StorageBytesPerStep float64        `json:"storage_bytes_per_step"`
	MeanBytesReadOnLoad float64        `json:"mean_bytes_read_on_load"`
}

type CheckpointFreqCell struct {
	CheckpointInterval  int            `json:"checkpoint_interval"`
	Steps               int            `json:"steps"`
	Fsync               bool           `json:"fsync"`
	Trials              int            `json:"trials"`
	HappyPathLatency    LatencyStatsMS `json:"happy_path_latency"`
	CrashRecoveryLat    LatencyStatsMS `json:"crash_recovery_latency"`
	CheckpointsSaved    int            `json:"checkpoints_saved"`
	StorageBytesPerStep float64        `json:"storage_bytes_per_step"`
}

type agentStepSpec struct {
	Index        int
	ActivityName string
	Kind         string // "tool_call", "vfs_edit", "billing_charge", "model_infer"
	Payload      map[string]any
}

func generateTrace(seed int64, steps int) []agentStepSpec {
	rng := rand.New(rand.NewSource(seed + int64(steps)*7919))
	kinds := []string{"tool_call", "model_infer", "vfs_edit", "billing_charge"}
	out := make([]agentStepSpec, steps)
	for i := 0; i < steps; i++ {
		kind := kinds[i%len(kinds)]
		var buf [48]byte
		for j := range buf {
			buf[j] = byte('a' + rng.Intn(26))
		}
		out[i] = agentStepSpec{
			Index:        i,
			ActivityName: fmt.Sprintf("%s_%03d", kind, i),
			Kind:         kind,
			Payload: map[string]any{
				"step":       i,
				"kind":       kind,
				"file":       fmt.Sprintf("internal/pkg_%d/module_%d.go", i%7, i),
				"token_cost": 120 + rng.Intn(880),
				"snippet":    string(buf[:]),
			},
		}
	}
	return out
}

type agentCheckpointState struct {
	NextStep      int    `json:"next_step"`
	RollingDigest string `json:"rolling_digest"`
}

func runCrashRecoveryCell(
	ctx context.Context,
	baseDir string,
	mode replay.ReplayMode,
	steps int,
	crashStep int,
	cpInterval int,
	syncWrites bool,
	trials int,
	seed int64,
) (CrashRecoveryCell, int, error) {
	trace := generateTrace(seed, steps)
	errCrash := errors.New("injected mid-workflow crash")

	ttrLatencies := make([]float64, 0, trials)
	recLatencies := make([]float64, 0, trials)
	e2eLatencies := make([]float64, 0, trials)
	var totalDupes, totalUnique int
	var sumEventsLoaded float64
	var sumBytesRead float64
	var sumStorageBytes int64
	var restoredCount int
	var maxIdleLocks int

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	for trial := 0; trial < trials; trial++ {
		trialDir := filepath.Join(baseDir, fmt.Sprintf("%s_n%d_t%d", mode, steps, trial))
		var store *replay.FileCheckpointStore
		var err error
		if mode != replay.ModeStatelessRetry {
			store, err = replay.NewFileCheckpointStore(trialDir, &replay.FileStoreOptions{
				SyncWrites: syncWrites,
				Limits: replay.Limits{
					MaxPayloadBytes: 2 << 20,
					MaxHistoryDepth: int64(steps*4 + 64),
				},
			})
			if err != nil {
				return CrashRecoveryCell{}, 0, err
			}
		}

		eng1, err := replay.NewEngine(store, replay.EngineConfig{
			Mode:               mode,
			CheckpointInterval: cpInterval,
			Limits: replay.Limits{
				MaxPayloadBytes: 2 << 20,
				MaxHistoryDepth: int64(steps*4 + 64),
			},
		})
		if err != nil {
			return CrashRecoveryCell{}, 0, err
		}

		execCounts := make([]int, steps)
		shouldCrash := true
		var recStart time.Time
		var ttrMS float64
		var ttrRecorded bool
		tenantID := fmt.Sprintf("tenant-bench-%d", trial%4)
		wfID := fmt.Sprintf("wf-%s-%d-%d", mode, steps, trial)
		startTime := time.Date(2026, 11, 26, 12, 0, 0, 0, time.UTC).Add(time.Duration(trial) * time.Minute)

		workflow := func(_ context.Context, wctx *replay.WorkflowContext, _ replay.TriggerEvent) error {
			var st agentCheckpointState
			if step, ok, err := wctx.RestoreCheckpoint(&st); err != nil {
				return err
			} else if ok {
				st.NextStep = step
			}

			for i := st.NextStep; i < len(trace); i++ {
				if shouldCrash && i == crashStep {
					return errCrash
				}
				if !shouldCrash && !ttrRecorded && i == crashStep {
					ttrMS = elapsedMS(recStart)
					ttrRecorded = true
				}
				spec := trace[i]
				key, err := replay.DeriveIdempotencyKey(wctx.TenantID(), wctx.WorkflowID(), spec.ActivityName, fmt.Sprintf("s%d", i))
				if err != nil {
					return err
				}
				out, err := wctx.ExecuteActivity(spec.ActivityName, key, spec.Payload, func(_ context.Context, in replay.ActivityInput) (json.RawMessage, error) {
					execCounts[i]++
					h := sha256.Sum256(in.Payload)
					return json.Marshal(map[string]any{
						"step":   i,
						"kind":   spec.Kind,
						"digest": hex.EncodeToString(h[:16]),
					})
				})
				if err != nil {
					return err
				}
				h := sha256.Sum256(append([]byte(st.RollingDigest), out...))
				st.RollingDigest = hex.EncodeToString(h[:16])
				st.NextStep = i + 1
				if wctx.ShouldCheckpoint() {
					if err := wctx.SaveCheckpoint(fmt.Sprintf("cp-%d", st.NextStep), st.NextStep, &st); err != nil {
						return err
					}
				}
			}
			return nil
		}

		e2eStart := time.Now()
		_, err = eng1.HandleEvent(ctx, replay.TriggerEvent{
			TenantID:     tenantID,
			WorkflowID:   wfID,
			WorkflowName: "coding-agent-bench",
			Type:         replay.TriggerStart,
			LogicalTime:  startTime,
		}, workflow)
		if !errors.Is(err, errCrash) {
			return CrashRecoveryCell{}, 0, fmt.Errorf("expected injected crash, got %v", err)
		}
		if locks := eng1.ActiveLockCount(); locks > maxIdleLocks {
			maxIdleLocks = locks
		}

		// Cold restart: fresh FileCheckpointStore + fresh Engine reading disk state.
		shouldCrash = false
		recStart = time.Now()
		var store2 *replay.FileCheckpointStore
		if mode != replay.ModeStatelessRetry {
			store2, err = replay.NewFileCheckpointStore(trialDir, &replay.FileStoreOptions{
				SyncWrites: syncWrites,
				Limits: replay.Limits{
					MaxPayloadBytes: 2 << 20,
					MaxHistoryDepth: int64(steps*4 + 64),
				},
			})
			if err != nil {
				return CrashRecoveryCell{}, 0, err
			}
		}
		eng2, err := replay.NewEngine(store2, replay.EngineConfig{
			Mode:               mode,
			CheckpointInterval: cpInterval,
			Limits: replay.Limits{
				MaxPayloadBytes: 2 << 20,
				MaxHistoryDepth: int64(steps*4 + 64),
			},
		})
		if err != nil {
			return CrashRecoveryCell{}, 0, err
		}
		rep, err := eng2.HandleEvent(ctx, replay.TriggerEvent{
			TenantID:     tenantID,
			WorkflowID:   wfID,
			WorkflowName: "coding-agent-bench",
			Type:         replay.TriggerResume,
			LogicalTime:  startTime.Add(5 * time.Second),
		}, workflow)
		if err != nil {
			return CrashRecoveryCell{}, 0, fmt.Errorf("recovery failed (mode=%s): %w", mode, err)
		}
		recMS := elapsedMS(recStart)
		e2eMS := elapsedMS(e2eStart)

		if locks := eng2.ActiveLockCount(); locks > maxIdleLocks {
			maxIdleLocks = locks
		}
		ttrLatencies = append(ttrLatencies, ttrMS)
		recLatencies = append(recLatencies, recMS)
		e2eLatencies = append(e2eLatencies, e2eMS)
		sumEventsLoaded += float64(rep.EventsLoaded)
		sumBytesRead += float64(rep.BytesReadOnLoad)
		if rep.RestoredCheckpoint {
			restoredCount++
		}

		for _, c := range execCounts {
			totalUnique++
			if c > 1 {
				totalDupes += c - 1
			}
		}
		if store2 != nil {
			used, err := store2.StorageBytes(ctx, tenantID, wfID)
			if err != nil {
				return CrashRecoveryCell{}, 0, err
			}
			sumStorageBytes += used
		}
	}

	dupePct := 0.0
	if totalUnique > 0 {
		dupePct = float64(totalDupes) / float64(totalUnique) * 100.0
	}
	storagePerStep := 0.0
	if trials > 0 && steps > 0 {
		storagePerStep = float64(sumStorageBytes) / float64(trials*steps)
	}

	return CrashRecoveryCell{
		Mode:                   string(mode),
		Steps:                  steps,
		CrashStep:              crashStep,
		Fsync:                  syncWrites,
		CheckpointInterval:     cpInterval,
		Trials:                 trials,
		TimeToResumeLatency:    computeLatencyStats(ttrLatencies),
		RecoveryLatency:        computeLatencyStats(recLatencies),
		TotalEndToEndLatency:   computeLatencyStats(e2eLatencies),
		DuplicateSideEffects:   totalDupes,
		TotalUniqueSideEffects: totalUnique,
		DuplicateSideEffPct:    round4(dupePct),
		MeanEventsLoaded:       round2(sumEventsLoaded / float64(trials)),
		MeanBytesReadOnLoad:    round2(sumBytesRead / float64(trials)),
		StorageBytesPerStep:    round2(storagePerStep),
		RestoredCheckpointRate: round2(float64(restoredCount) / float64(trials) * 100.0),
	}, maxIdleLocks, nil
}

func runWebhookStormCell(
	ctx context.Context,
	baseDir string,
	mode replay.ReplayMode,
	tenants int,
	deliveriesPerTenant int,
	steps int,
	cpInterval int,
	syncWrites bool,
	seed int64,
) (WebhookStormCell, error) {
	trace := generateTrace(seed+101, steps)
	var store *replay.FileCheckpointStore
	var err error
	if mode != replay.ModeStatelessRetry {
		store, err = replay.NewFileCheckpointStore(filepath.Join(baseDir, "storm_"+string(mode)), &replay.FileStoreOptions{
			SyncWrites: syncWrites,
		})
		if err != nil {
			return WebhookStormCell{}, err
		}
	}
	eng, err := replay.NewEngine(store, replay.EngineConfig{
		Mode:               mode,
		CheckpointInterval: cpInterval,
	})
	if err != nil {
		return WebhookStormCell{}, err
	}

	sideEffects := make([]atomic.Int64, tenants)
	var latMu sync.Mutex
	latencies := make([]float64, 0, tenants*deliveriesPerTenant)
	var crossTenantErrs atomic.Int64

	var wg sync.WaitGroup
	errCh := make(chan error, tenants*deliveriesPerTenant)

	for ti := 0; ti < tenants; ti++ {
		tenantID := fmt.Sprintf("tenant-storm-%d", ti)
		counter := &sideEffects[ti]
		for d := 0; d < deliveriesPerTenant; d++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				t0 := time.Now()
				_, e := eng.HandleEvent(ctx, replay.TriggerEvent{
					TenantID:     tenantID,
					WorkflowID:   "shared-webhook-wf",
					WorkflowName: "webhook-agent",
					Type:         replay.TriggerStart,
				}, func(_ context.Context, wctx *replay.WorkflowContext, _ replay.TriggerEvent) error {
					if wctx.TenantID() != tenantID {
						crossTenantErrs.Add(1)
					}
					for i, spec := range trace {
						key, err := replay.DeriveIdempotencyKey(wctx.TenantID(), wctx.WorkflowID(), spec.ActivityName, fmt.Sprintf("s%d", i))
						if err != nil {
							return err
						}
						if _, err := wctx.ExecuteActivity(spec.ActivityName, key, spec.Payload, func(_ context.Context, _ replay.ActivityInput) (json.RawMessage, error) {
							counter.Add(1)
							return json.RawMessage(`{"ok":true}`), nil
						}); err != nil {
							return err
						}
					}
					return nil
				})
				ms := elapsedMS(t0)
				latMu.Lock()
				latencies = append(latencies, ms)
				latMu.Unlock()
				if e != nil {
					errCh <- e
				}
			}()
		}
	}
	wg.Wait()
	close(errCh)
	for e := range errCh {
		return WebhookStormCell{}, e
	}

	expected := tenants * steps
	observed := 0
	for ti := 0; ti < tenants; ti++ {
		observed += int(sideEffects[ti].Load())
	}
	dupes := observed - expected
	if dupes < 0 {
		dupes = 0
	}
	dupePct := float64(dupes) / float64(expected) * 100.0

	return WebhookStormCell{
		Mode:                  string(mode),
		Tenants:               tenants,
		DeliveriesPerTenant:   deliveriesPerTenant,
		StepsPerWorkflow:      steps,
		ActivationLatency:     computeLatencyStats(latencies),
		ExpectedSideEffects:   expected,
		ObservedSideEffects:   observed,
		DuplicateSideEffects:  dupes,
		DuplicateSideEffPct:   round4(dupePct),
		CrossTenantViolations: int(crossTenantErrs.Load()),
		IdleActiveLocks:       eng.ActiveLockCount(),
	}, nil
}

func runNegativeRegimes(ctx context.Context, baseDir string, cfg BenchConfig) (NegativeRegimesReport, error) {
	var out NegativeRegimesReport

	// NR1: Trivial 1-step workflow, 0 crashes, fsync=true.
	for _, mode := range []replay.ReplayMode{replay.ModeStatelessRetry, replay.ModeCheckpointedReplay} {
		lats := make([]float64, 0, cfg.TrivialTrials)
		var sumStorage int64
		for tr := 0; tr < cfg.TrivialTrials; tr++ {
			var store *replay.FileCheckpointStore
			var err error
			if mode != replay.ModeStatelessRetry {
				store, err = replay.NewFileCheckpointStore(filepath.Join(baseDir, fmt.Sprintf("nr1_%s_%d", mode, tr)), &replay.FileStoreOptions{SyncWrites: true})
				if err != nil {
					return out, err
				}
			}
			eng, err := replay.NewEngine(store, replay.EngineConfig{Mode: mode, CheckpointInterval: 1})
			if err != nil {
				return out, err
			}
			t0 := time.Now()
			_, err = eng.HandleEvent(ctx, replay.TriggerEvent{
				TenantID:   "tenant-nr1",
				WorkflowID: fmt.Sprintf("wf-nr1-%d", tr),
				Type:       replay.TriggerStart,
			}, func(_ context.Context, wctx *replay.WorkflowContext, _ replay.TriggerEvent) error {
				_, err := wctx.ExecuteActivity("single_call", "k0", map[string]int{"x": tr}, func(_ context.Context, _ replay.ActivityInput) (json.RawMessage, error) {
					return json.RawMessage(`{"ok":true}`), nil
				})
				if err != nil {
					return err
				}
				if wctx.ShouldCheckpoint() {
					return wctx.SaveCheckpoint("cp-1", 1, map[string]int{"done": 1})
				}
				return nil
			})
			if err != nil {
				return out, err
			}
			lats = append(lats, elapsedMS(t0))
			if store != nil {
				used, _ := store.StorageBytes(ctx, "tenant-nr1", fmt.Sprintf("wf-nr1-%d", tr))
				sumStorage += used
			}
		}
		out.TrivialOneStep = append(out.TrivialOneStep, TrivialOneStepCell{
			Mode:                string(mode),
			Steps:               1,
			Fsync:               true,
			Trials:              cfg.TrivialTrials,
			HappyPathLatency:    computeLatencyStats(lats),
			StorageBytesPerStep: round2(float64(sumStorage) / float64(cfg.TrivialTrials)),
		})
	}

	// NR2: Large binary step payloads (256 KiB inline vs VFS content hash reference).
	const blobBytes = 256 * 1024
	const nr2Steps = 10
	rng := rand.New(rand.NewSource(cfg.Seed + 999))
	rawBlob := make([]byte, blobBytes)
	for i := range rawBlob {
		rawBlob[i] = byte(rng.Intn(256))
	}
	encodedBlob := base64.StdEncoding.EncodeToString(rawBlob)
	vfsRef := replay.VFSContentRef(rawBlob)

	for _, strategy := range []string{"inline_256kb_blob", "vfs_content_hash_ref"} {
		happyLats := make([]float64, 0, cfg.NegativeTrials)
		recLats := make([]float64, 0, cfg.NegativeTrials)
		var sumStorage int64
		var sumBytesRead float64
		errCrash := errors.New("nr2 crash")

		for tr := 0; tr < cfg.NegativeTrials; tr++ {
			dir := filepath.Join(baseDir, fmt.Sprintf("nr2_%s_%d", strategy, tr))
			limits := replay.Limits{MaxPayloadBytes: 4 << 20, MaxHistoryDepth: 1000}
			store, err := replay.NewFileCheckpointStore(dir, &replay.FileStoreOptions{
				SyncWrites: true,
				Limits:     limits,
			})
			if err != nil {
				return out, err
			}
			eng, err := replay.NewEngine(store, replay.EngineConfig{
				Mode:               replay.ModeCheckpointedReplay,
				CheckpointInterval: 4,
				Limits:             limits,
			})
			if err != nil {
				return out, err
			}

			// First measure happy path.
			wfHappy := fmt.Sprintf("wf-nr2-happy-%d", tr)
			type blobState struct {
				NextStep int    `json:"next_step"`
				LastBlob string `json:"last_blob"`
			}
			t0 := time.Now()
			_, err = eng.HandleEvent(ctx, replay.TriggerEvent{
				TenantID:   "tenant-nr2",
				WorkflowID: wfHappy,
				Type:       replay.TriggerStart,
			}, func(_ context.Context, wctx *replay.WorkflowContext, _ replay.TriggerEvent) error {
				var st blobState
				for i := 0; i < nr2Steps; i++ {
					payloadVal := vfsRef
					if strategy == "inline_256kb_blob" {
						payloadVal = encodedBlob
					}
					_, err := wctx.ExecuteActivity(fmt.Sprintf("artifact_%d", i), fmt.Sprintf("k-%d", i), map[string]string{"ref": payloadVal}, func(_ context.Context, _ replay.ActivityInput) (json.RawMessage, error) {
						return json.Marshal(map[string]string{"artifact": payloadVal})
					})
					if err != nil {
						return err
					}
					st.NextStep = i + 1
					st.LastBlob = payloadVal
					if wctx.ShouldCheckpoint() {
						if err := wctx.SaveCheckpoint(fmt.Sprintf("cp-%d", st.NextStep), st.NextStep, &st); err != nil {
							return err
						}
					}
				}
				return nil
			})
			if err != nil {
				return out, err
			}
			happyLats = append(happyLats, elapsedMS(t0))
			used, _ := store.StorageBytes(ctx, "tenant-nr2", wfHappy)
			sumStorage += used

			// Next measure crash recovery at step 7.
			wfRec := fmt.Sprintf("wf-nr2-rec-%d", tr)
			shouldCrash := true
			recHandler := func(_ context.Context, wctx *replay.WorkflowContext, _ replay.TriggerEvent) error {
				var st blobState
				if step, ok, err := wctx.RestoreCheckpoint(&st); err != nil {
					return err
				} else if ok {
					st.NextStep = step
				}
				for i := st.NextStep; i < nr2Steps; i++ {
					if shouldCrash && i == 7 {
						return errCrash
					}
					payloadVal := vfsRef
					if strategy == "inline_256kb_blob" {
						payloadVal = encodedBlob
					}
					_, err := wctx.ExecuteActivity(fmt.Sprintf("artifact_%d", i), fmt.Sprintf("k-%d", i), map[string]string{"ref": payloadVal}, func(_ context.Context, _ replay.ActivityInput) (json.RawMessage, error) {
						return json.Marshal(map[string]string{"artifact": payloadVal})
					})
					if err != nil {
						return err
					}
					st.NextStep = i + 1
					st.LastBlob = payloadVal
					if wctx.ShouldCheckpoint() {
						if err := wctx.SaveCheckpoint(fmt.Sprintf("cp-%d", st.NextStep), st.NextStep, &st); err != nil {
							return err
						}
					}
				}
				return nil
			}
			_, _ = eng.HandleEvent(ctx, replay.TriggerEvent{TenantID: "tenant-nr2", WorkflowID: wfRec, Type: TriggerStart}, recHandler)
			shouldCrash = false
			tRec := time.Now()
			store2, _ := replay.NewFileCheckpointStore(dir, &replay.FileStoreOptions{SyncWrites: true, Limits: limits})
			eng2, _ := replay.NewEngine(store2, replay.EngineConfig{Mode: replay.ModeCheckpointedReplay, CheckpointInterval: 4, Limits: limits})
			rep, err := eng2.HandleEvent(ctx, replay.TriggerEvent{TenantID: "tenant-nr2", WorkflowID: wfRec, Type: TriggerResume}, recHandler)
			if err != nil {
				return out, err
			}
			recLats = append(recLats, elapsedMS(tRec))
			sumBytesRead += float64(rep.BytesReadOnLoad)
		}

		out.LargeBinaryBlob = append(out.LargeBinaryBlob, LargeBinaryBlobCell{
			PayloadStrategy:     strategy,
			Steps:               nr2Steps,
			BlobBytesPerStep:    blobBytes,
			Fsync:               true,
			Trials:              cfg.NegativeTrials,
			HappyPathLatency:    computeLatencyStats(happyLats),
			RecoveryLatency:     computeLatencyStats(recLats),
			StorageBytesPerStep: round2(float64(sumStorage) / float64(cfg.NegativeTrials*nr2Steps)),
			MeanBytesReadOnLoad: round2(sumBytesRead / float64(cfg.NegativeTrials)),
		})
	}

	// NR3: Checkpoint frequency tradeoff (K=1 vs K=8 vs K=20, fsync=true).
	nr3Steps := cfg.NR3Steps
	if nr3Steps <= 0 {
		nr3Steps = 40
	}
	traceNR3 := generateTrace(cfg.Seed+303, nr3Steps)
	for _, k := range []int{1, 8, 20} {
		happyLats := make([]float64, 0, cfg.NegativeTrials)
		var sumStorage int64
		var cpSaved int
		for tr := 0; tr < cfg.NegativeTrials; tr++ {
			dir := filepath.Join(baseDir, fmt.Sprintf("nr3_k%d_t%d", k, tr))
			store, err := replay.NewFileCheckpointStore(dir, &replay.FileStoreOptions{SyncWrites: true})
			if err != nil {
				return out, err
			}
			eng, err := replay.NewEngine(store, replay.EngineConfig{Mode: replay.ModeCheckpointedReplay, CheckpointInterval: k})
			if err != nil {
				return out, err
			}
			t0 := time.Now()
			rep, err := eng.HandleEvent(ctx, replay.TriggerEvent{
				TenantID:   "tenant-nr3",
				WorkflowID: fmt.Sprintf("wf-nr3-%d-%d", k, tr),
				Type:       replay.TriggerStart,
			}, func(_ context.Context, wctx *replay.WorkflowContext, _ replay.TriggerEvent) error {
				var st agentCheckpointState
				for i, spec := range traceNR3 {
					key, _ := replay.DeriveIdempotencyKey(wctx.TenantID(), wctx.WorkflowID(), spec.ActivityName, fmt.Sprintf("s%d", i))
					if _, err := wctx.ExecuteActivity(spec.ActivityName, key, spec.Payload, func(_ context.Context, _ replay.ActivityInput) (json.RawMessage, error) {
						return json.RawMessage(`{"ok":true}`), nil
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
			})
			if err != nil {
				return out, err
			}
			happyLats = append(happyLats, elapsedMS(t0))
			cpSaved = rep.CheckpointsSaved
			used, _ := store.StorageBytes(ctx, "tenant-nr3", fmt.Sprintf("wf-nr3-%d-%d", k, tr))
			sumStorage += used
		}

		// Also measure crash recovery at ~85% step for this K.
		crashAt := (nr3Steps * 7) / 8
		if crashAt < 1 {
			crashAt = 1
		}
		recCell, _, err := runCrashRecoveryCell(ctx, filepath.Join(baseDir, fmt.Sprintf("nr3_rec_k%d", k)), replay.ModeCheckpointedReplay, nr3Steps, crashAt, k, true, cfg.NegativeTrials, cfg.Seed+303)
		if err != nil {
			return out, err
		}
		out.CheckpointFreq = append(out.CheckpointFreq, CheckpointFreqCell{
			CheckpointInterval:  k,
			Steps:               nr3Steps,
			Fsync:               true,
			Trials:              cfg.NegativeTrials,
			HappyPathLatency:    computeLatencyStats(happyLats),
			CrashRecoveryLat:    recCell.RecoveryLatency,
			CheckpointsSaved:    cpSaved,
			StorageBytesPerStep: round2(float64(sumStorage) / float64(cfg.NegativeTrials*nr3Steps)),
		})
	}

	return out, nil
}

const (
	TriggerStart  = replay.TriggerStart
	TriggerResume = replay.TriggerResume
)

func runAllBenchmarks(cfg BenchConfig) (*BenchmarkReport, error) {
	if len(cfg.FsyncSteps) == 0 {
		cfg.FsyncSteps = []int{10, 40, 100}
	}
	if len(cfg.ScalingSteps) == 0 {
		cfg.ScalingSteps = []int{50, 200, 500}
	}
	if cfg.NR3Steps <= 0 {
		cfg.NR3Steps = 40
	}

	ctx := context.Background()
	tmpRoot, err := os.MkdirTemp("", "hanzo-replay-bench-*")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(tmpRoot)

	runtime.GC()
	baseGoroutines := runtime.NumGoroutine()
	maxIdleLocks := 0

	report := &BenchmarkReport{
		Benchmark:      "hanzo-durable-workflow-replay-bench",
		HIP:            "HIP-1347",
		Issue:          "hanzoai/agents#85",
		GeneratedAtUTC: time.Now().UTC().Format(time.RFC3339),
		HostEnvironment: HostEnvironment{
			GOOS:      runtime.GOOS,
			GOARCH:    runtime.GOARCH,
			GoVersion: runtime.Version(),
			NumCPU:    runtime.NumCPU(),
		},
		HonestyNotice: "All latency, storage, and I/O measurements are real local-disk wall-clock measurements on the authoring host running control-plane/internal/replay with FileCheckpointStore. No external LLM/Temporal cluster is invoked; figures must not be extrapolated to production cloud hardware without reproducing on target Hanzo infrastructure.",
		Config:        cfg,
	}

	modes := []replay.ReplayMode{
		replay.ModeStatelessRetry,
		replay.ModeFullHistoryReplay,
		replay.ModeCheckpointedReplay,
	}

	// Scenario A1: Real fsync=true crash recovery sweep (crash at 75%).
	for _, n := range cfg.FsyncSteps {
		crashAt := (n * 3) / 4
		for _, mode := range modes {
			cell, idleLocks, err := runCrashRecoveryCell(ctx, filepath.Join(tmpRoot, "a1_fsync"), mode, n, crashAt, cfg.CheckpointInterval, true, cfg.FsyncTrials, cfg.Seed)
			if err != nil {
				return nil, err
			}
			if idleLocks > maxIdleLocks {
				maxIdleLocks = idleLocks
			}
			report.CrashRecoveryFsync = append(report.CrashRecoveryFsync, cell)
		}
	}

	// Scenario A2: High-depth recovery scaling (crash at 90%, fsync=false page-cache).
	for _, n := range cfg.ScalingSteps {
		crashAt := (n * 9) / 10
		for _, mode := range modes {
			cell, idleLocks, err := runCrashRecoveryCell(ctx, filepath.Join(tmpRoot, "a2_scale"), mode, n, crashAt, cfg.CheckpointInterval, false, cfg.ScalingTrials, cfg.Seed+55)
			if err != nil {
				return nil, err
			}
			if idleLocks > maxIdleLocks {
				maxIdleLocks = idleLocks
			}
			report.RecoveryScaling = append(report.RecoveryScaling, cell)
		}
	}

	// Scenario B: Concurrent multi-tenant duplicate webhook storm (fsync=true).
	for _, mode := range modes {
		cell, err := runWebhookStormCell(ctx, filepath.Join(tmpRoot, "b_storm"), mode, cfg.Tenants, cfg.DuplicateWebhooks, 10, cfg.CheckpointInterval, true, cfg.Seed)
		if err != nil {
			return nil, err
		}
		if cell.IdleActiveLocks > maxIdleLocks {
			maxIdleLocks = cell.IdleActiveLocks
		}
		report.WebhookStorm = append(report.WebhookStorm, cell)
	}

	// Scenario C: Honest negative regimes.
	neg, err := runNegativeRegimes(ctx, filepath.Join(tmpRoot, "c_neg"), cfg)
	if err != nil {
		return nil, err
	}
	report.NegativeRegimes = neg

	runtime.GC()
	postGoroutines := runtime.NumGoroutine()
	delta := postGoroutines - baseGoroutines
	report.ZeroDaemonCheck = ZeroDaemonVerification{
		BaselineGoroutines: baseGoroutines,
		PostRunGoroutines:  postGoroutines,
		GoroutineDelta:     delta,
		IdleActiveLocks:    maxIdleLocks,
		ZeroDaemonsPassed:  delta <= 0 && maxIdleLocks == 0,
	}
	return report, nil
}

func renderMarkdownReport(r *BenchmarkReport) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# HIP-1347 Durable Event-Driven Workflow Replay — Measured Score Tables\n\n")
	fmt.Fprintf(&b, "- **Generated (UTC)**: `%s`\n", r.GeneratedAtUTC)
	fmt.Fprintf(&b, "- **Host**: `%s/%s`, `%s`, `%d` vCPUs\n", r.HostEnvironment.GOOS, r.HostEnvironment.GOARCH, r.HostEnvironment.GoVersion, r.HostEnvironment.NumCPU)
	fmt.Fprintf(&b, "- **Seed**: `%d` | **Checkpoint Interval (`K`)**: `%d`\n", r.Config.Seed, r.Config.CheckpointInterval)
	fmt.Fprintf(&b, "- **Zero-Daemon Verification**: baseline goroutines=`%d`, post-run=`%d` (delta=`%d`), idle active locks=`%d`, passed=`%v`\n\n",
		r.ZeroDaemonCheck.BaselineGoroutines, r.ZeroDaemonCheck.PostRunGoroutines, r.ZeroDaemonCheck.GoroutineDelta,
		r.ZeroDaemonCheck.IdleActiveLocks, r.ZeroDaemonCheck.ZeroDaemonsPassed)
	fmt.Fprintf(&b, "> **Measurement Attribution**: %s\n\n", r.HonestyNotice)

	fmt.Fprintf(&b, "## 1. Multi-Step Coding-Agent Crash Recovery (`fsync=true`, Crash at 75%% of Workflow)\n\n")
	fmt.Fprintf(&b, "| Steps (`N`) | Crash Step | Mode | Time-to-Resume p50 / p95 / p99 (ms) | Full Recovery p50 (ms) | Events Loaded | Bytes Read on Load | Dup Side Effects | Storage (B/step) | Ckpt Restored |\n")
	fmt.Fprintf(&b, "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|\n")
	for _, c := range r.CrashRecoveryFsync {
		fmt.Fprintf(&b, "| %d | %d | `%s` | %.3f / %.3f / %.3f | %.3f | %.1f | %.0f | %d / %d (%.2f%%) | %.1f | %.0f%% |\n",
			c.Steps, c.CrashStep, c.Mode,
			c.TimeToResumeLatency.P50, c.TimeToResumeLatency.P95, c.TimeToResumeLatency.P99,
			c.RecoveryLatency.P50,
			c.MeanEventsLoaded, c.MeanBytesReadOnLoad,
			c.DuplicateSideEffects, c.TotalUniqueSideEffects, c.DuplicateSideEffPct,
			c.StorageBytesPerStep, c.RestoredCheckpointRate)
	}

	fmt.Fprintf(&b, "\n## 2. Deep-History Recovery Scaling (`fsync=false` Page-Cache Isolated, Crash at 90%%)\n\n")
	fmt.Fprintf(&b, "| Steps (`N`) | Crash Step | Mode | Time-to-Resume p50 / p95 / p99 (ms) | Full Recovery p50 (ms) | Events Loaded | Bytes Read on Load | Dup Side Effects | Storage (B/step) |\n")
	fmt.Fprintf(&b, "|---:|---:|---|---:|---:|---:|---:|---:|---:|\n")
	for _, c := range r.RecoveryScaling {
		fmt.Fprintf(&b, "| %d | %d | `%s` | %.3f / %.3f / %.3f | %.3f | %.1f | %.0f | %d / %d (%.2f%%) | %.1f |\n",
			c.Steps, c.CrashStep, c.Mode,
			c.TimeToResumeLatency.P50, c.TimeToResumeLatency.P95, c.TimeToResumeLatency.P99,
			c.RecoveryLatency.P50,
			c.MeanEventsLoaded, c.MeanBytesReadOnLoad,
			c.DuplicateSideEffects, c.TotalUniqueSideEffects, c.DuplicateSideEffPct,
			c.StorageBytesPerStep)
	}

	fmt.Fprintf(&b, "\n## 3. Concurrent Multi-Tenant Duplicate Webhook Storm (`fsync=true`, %d Tenants × %d Deliveries)\n\n",
		r.Config.Tenants, r.Config.DuplicateWebhooks)
	fmt.Fprintf(&b, "| Mode | Activation p50 / p95 (ms) | Expected Side Effects | Observed Side Effects | Duplicate Rate | Cross-Tenant Violations | Idle Locks |\n")
	fmt.Fprintf(&b, "|---|---:|---:|---:|---:|---:|---:|\n")
	for _, c := range r.WebhookStorm {
		fmt.Fprintf(&b, "| `%s` | %.3f / %.3f | %d | %d | %.2f%% | %d | %d |\n",
			c.Mode, c.ActivationLatency.P50, c.ActivationLatency.P95,
			c.ExpectedSideEffects, c.ObservedSideEffects, c.DuplicateSideEffPct,
			c.CrossTenantViolations, c.IdleActiveLocks)
	}

	fmt.Fprintf(&b, "\n## 4. Honest Negative Regimes\n\n")
	fmt.Fprintf(&b, "### NR1: Trivial 1-Step Workflow (`N=1`, 0 Crashes, `fsync=true`)\n\n")
	fmt.Fprintf(&b, "| Mode | Trials | Happy-Path p50 / p95 / p99 (ms) | Storage (B/step) |\n")
	fmt.Fprintf(&b, "|---|---:|---:|---:|\n")
	for _, c := range r.NegativeRegimes.TrivialOneStep {
		fmt.Fprintf(&b, "| `%s` | %d | %.3f / %.3f / %.3f | %.1f |\n",
			c.Mode, c.Trials, c.HappyPathLatency.P50, c.HappyPathLatency.P95, c.HappyPathLatency.P99, c.StorageBytesPerStep)
	}

	fmt.Fprintf(&b, "\n### NR2: Large Binary Step Payloads (`N=10`, 256 KiB Blob/Step, `fsync=true`)\n\n")
	fmt.Fprintf(&b, "| Payload Strategy | Happy-Path p50 / p95 (ms) | Recovery p50 / p95 (ms) | Storage (B/step) | Bytes Read on Recovery |\n")
	fmt.Fprintf(&b, "|---|---:|---:|---:|---:|\n")
	for _, c := range r.NegativeRegimes.LargeBinaryBlob {
		fmt.Fprintf(&b, "| `%s` | %.3f / %.3f | %.3f / %.3f | %.1f | %.0f |\n",
			c.PayloadStrategy,
			c.HappyPathLatency.P50, c.HappyPathLatency.P95,
			c.RecoveryLatency.P50, c.RecoveryLatency.P95,
			c.StorageBytesPerStep, c.MeanBytesReadOnLoad)
	}

	fmt.Fprintf(&b, "\n### NR3: Over-Frequent Checkpointing Tradeoff (`N=40`, `fsync=true`)\n\n")
	fmt.Fprintf(&b, "| Checkpoint Interval (`K`) | Checkpoints Saved | Happy-Path p50 / p95 (ms) | Crash Recovery p50 / p95 (ms) | Storage (B/step) |\n")
	fmt.Fprintf(&b, "|---:|---:|---:|---:|---:|\n")
	for _, c := range r.NegativeRegimes.CheckpointFreq {
		fmt.Fprintf(&b, "| %d | %d | %.3f / %.3f | %.3f / %.3f | %.1f |\n",
			c.CheckpointInterval, c.CheckpointsSaved,
			c.HappyPathLatency.P50, c.HappyPathLatency.P95,
			c.CrashRecoveryLat.P50, c.CrashRecoveryLat.P95,
			c.StorageBytesPerStep)
	}

	return b.String()
}

func computeLatencyStats(samples []float64) LatencyStatsMS {
	if len(samples) == 0 {
		return LatencyStatsMS{}
	}
	cp := append([]float64(nil), samples...)
	sort.Float64s(cp)
	var sum float64
	for _, v := range cp {
		sum += v
	}
	return LatencyStatsMS{
		P50:  round3(percentile(cp, 0.50)),
		P95:  round3(percentile(cp, 0.95)),
		P99:  round3(percentile(cp, 0.99)),
		Mean: round3(sum / float64(len(cp))),
	}
}

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	if len(sorted) == 1 {
		return sorted[0]
	}
	rank := p * float64(len(sorted)-1)
	lo := int(math.Floor(rank))
	hi := int(math.Ceil(rank))
	if lo == hi {
		return sorted[lo]
	}
	frac := rank - float64(lo)
	return sorted[lo]*(1-frac) + sorted[hi]*frac
}

func elapsedMS(start time.Time) float64 {
	ns := time.Since(start).Nanoseconds()
	if ns < 0 {
		return 0
	}
	return float64(ns) / 1e6
}

func round2(v float64) float64 { return math.Round(v*100) / 100 }
func round3(v float64) float64 { return math.Round(v*1000) / 1000 }
func round4(v float64) float64 { return math.Round(v*10000) / 10000 }

func main() {
	var (
		seed          = flag.Int64("seed", 42, "Deterministic PRNG seed")
		cpInterval    = flag.Int("checkpoint-interval", 8, "Periodic checkpoint interval K")
		fsyncTrials   = flag.Int("fsync-trials", 5, "Trials per cell for fsync=true crash recovery sweep")
		scalingTrials = flag.Int("scaling-trials", 15, "Trials per cell for deep-history scaling sweep")
		tenants       = flag.Int("tenants", 8, "Concurrent tenants in webhook storm")
		dupWebhooks   = flag.Int("duplicate-webhooks", 5, "Duplicate webhook deliveries per tenant")
		trivialTrials = flag.Int("trivial-trials", 20, "Trials for NR1 1-step workflow")
		negTrials     = flag.Int("negative-trials", 5, "Trials for NR2/NR3 negative regimes")
		outJSON       = flag.String("output-json", "", "Optional output path for raw JSON results")
		outMD         = flag.String("output-markdown", "", "Optional output path for Markdown score tables")
	)
	flag.Parse()

	cfg := BenchConfig{
		Seed:               *seed,
		CheckpointInterval: *cpInterval,
		FsyncTrials:        *fsyncTrials,
		ScalingTrials:      *scalingTrials,
		Tenants:            *tenants,
		DuplicateWebhooks:  *dupWebhooks,
		TrivialTrials:      *trivialTrials,
		NegativeTrials:     *negTrials,
	}

	report, err := runAllBenchmarks(cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "replay-bench failed: %v\n", err)
		os.Exit(1)
	}

	md := renderMarkdownReport(report)
	if *outJSON != "" {
		if err := os.MkdirAll(filepath.Dir(*outJSON), 0o755); err != nil {
			fmt.Fprintf(os.Stderr, "mkdir for json: %v\n", err)
			os.Exit(1)
		}
		raw, err := json.MarshalIndent(report, "", "  ")
		if err != nil {
			fmt.Fprintf(os.Stderr, "marshal json: %v\n", err)
			os.Exit(1)
		}
		if err := os.WriteFile(*outJSON, append(raw, '\n'), 0o644); err != nil {
			fmt.Fprintf(os.Stderr, "write json: %v\n", err)
			os.Exit(1)
		}
	}
	if *outMD != "" {
		if err := os.MkdirAll(filepath.Dir(*outMD), 0o755); err != nil {
			fmt.Fprintf(os.Stderr, "mkdir for md: %v\n", err)
			os.Exit(1)
		}
		if err := os.WriteFile(*outMD, []byte(md), 0o644); err != nil {
			fmt.Fprintf(os.Stderr, "write md: %v\n", err)
			os.Exit(1)
		}
	}

	fmt.Print(md)
}
