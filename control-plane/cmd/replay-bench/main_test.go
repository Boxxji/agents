package main

import (
	"context"
	"strings"
	"testing"

	"github.com/hanzoai/agents/control-plane/internal/replay"
)

func TestGenerateTrace_Deterministic(t *testing.T) {
	t1 := generateTrace(42, 16)
	t2 := generateTrace(42, 16)
	if len(t1) != 16 || len(t2) != 16 {
		t.Fatalf("unexpected trace lengths: %d %d", len(t1), len(t2))
	}
	for i := range t1 {
		if t1[i].ActivityName != t2[i].ActivityName || t1[i].Payload["snippet"] != t2[i].Payload["snippet"] {
			t.Fatalf("non-deterministic trace at index %d", i)
		}
	}
}

func TestReplayBench_InvariantsAndBaselines(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()

	stateless, locks0, err := runCrashRecoveryCell(ctx, dir, replay.ModeStatelessRetry, 20, 15, 5, false, 3, 42)
	if err != nil {
		t.Fatalf("stateless cell failed: %v", err)
	}
	fullHist, locks1, err := runCrashRecoveryCell(ctx, dir, replay.ModeFullHistoryReplay, 20, 15, 5, false, 3, 42)
	if err != nil {
		t.Fatalf("full_history cell failed: %v", err)
	}
	ckpt, locks2, err := runCrashRecoveryCell(ctx, dir, replay.ModeCheckpointedReplay, 20, 15, 5, false, 3, 42)
	if err != nil {
		t.Fatalf("checkpointed cell failed: %v", err)
	}

	if locks0 != 0 || locks1 != 0 || locks2 != 0 {
		t.Fatalf("expected 0 idle active locks, got %d/%d/%d", locks0, locks1, locks2)
	}
	if stateless.DuplicateSideEffects == 0 {
		t.Fatalf("expected naive stateless baseline to incur duplicate side effects")
	}
	if fullHist.DuplicateSideEffects != 0 || ckpt.DuplicateSideEffects != 0 {
		t.Fatalf("expected 0 duplicate side effects for durable modes, got full=%d ckpt=%d",
			fullHist.DuplicateSideEffects, ckpt.DuplicateSideEffects)
	}
	if ckpt.MeanEventsLoaded >= fullHist.MeanEventsLoaded {
		t.Fatalf("expected checkpointed replay to load fewer events than full-history: ckpt=%.1f full=%.1f",
			ckpt.MeanEventsLoaded, fullHist.MeanEventsLoaded)
	}
	if ckpt.MeanBytesReadOnLoad >= fullHist.MeanBytesReadOnLoad {
		t.Fatalf("expected checkpointed replay to read fewer bytes than full-history: ckpt=%.0f full=%.0f",
			ckpt.MeanBytesReadOnLoad, fullHist.MeanBytesReadOnLoad)
	}

	// Verify webhook storm invariants.
	stormStateless, err := runWebhookStormCell(ctx, dir, replay.ModeStatelessRetry, 4, 3, 4, 2, false, 42)
	if err != nil {
		t.Fatalf("storm stateless: %v", err)
	}
	stormCkpt, err := runWebhookStormCell(ctx, dir, replay.ModeCheckpointedReplay, 4, 3, 4, 2, false, 42)
	if err != nil {
		t.Fatalf("storm ckpt: %v", err)
	}
	if stormStateless.DuplicateSideEffects == 0 {
		t.Fatalf("expected stateless webhook storm duplicates > 0")
	}
	if stormCkpt.DuplicateSideEffects != 0 || stormCkpt.CrossTenantViolations != 0 || stormCkpt.IdleActiveLocks != 0 {
		t.Fatalf("checkpointed webhook storm violated invariants: %+v", stormCkpt)
	}
}

func TestRunAllBenchmarks_MinimalSmokeAndMarkdownRender(t *testing.T) {
	cfg := BenchConfig{
		Seed:               42,
		CheckpointInterval: 4,
		FsyncTrials:        1,
		ScalingTrials:      1,
		Tenants:            2,
		DuplicateWebhooks:  2,
		TrivialTrials:      2,
		NegativeTrials:     1,
		FsyncSteps:         []int{8},
		ScalingSteps:       []int{16},
		NR3Steps:           8,
	}
	rep, err := runAllBenchmarks(cfg)
	if err != nil {
		t.Fatalf("runAllBenchmarks failed: %v", err)
	}
	if !rep.ZeroDaemonCheck.ZeroDaemonsPassed {
		t.Fatalf("zero daemon check failed: %+v", rep.ZeroDaemonCheck)
	}
	md := renderMarkdownReport(rep)
	for _, want := range []string{
		"HIP-1347 Durable Event-Driven Workflow Replay",
		"NR1: Trivial 1-Step Workflow",
		"NR2: Large Binary Step Payloads",
		"NR3: Over-Frequent Checkpointing Tradeoff",
		"vfs_content_hash_ref",
	} {
		if !strings.Contains(md, want) {
			t.Fatalf("markdown report missing expected section %q", want)
		}
	}
}
