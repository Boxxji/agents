// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

package contextpack

import (
	"context"
	"encoding/json"
	"sync"
	"testing"
)

func TestPositionAttentionWeight_TACL24UShapeAndLengthScaling(t *testing.T) {
	for _, k := range []int{5, 10, 20, 30} {
		tokens := k * 220
		startW := PositionAttentionWeight(0.0, k, tokens)
		midW := PositionAttentionWeight(0.5, k, tokens)
		endW := PositionAttentionWeight(1.0, k, tokens)

		if startW <= midW {
			t.Errorf("K=%d: expected primacy start (%.3f) > middle (%.3f)", k, startW, midW)
		}
		if endW <= midW {
			t.Errorf("K=%d: expected recency end (%.3f) > middle (%.3f)", k, endW, midW)
		}
		if startW < endW {
			t.Errorf("K=%d: expected primacy (%.3f) >= recency (%.3f)", k, startW, endW)
		}
	}

	// Middle degradation must worsen monotonically as K increases 5 -> 10 -> 20 -> 30.
	m5 := PositionAttentionWeight(0.5, 5, 1100)
	m10 := PositionAttentionWeight(0.5, 10, 2200)
	m20 := PositionAttentionWeight(0.5, 20, 4400)
	m30 := PositionAttentionWeight(0.5, 30, 6600)
	if !(m5 > m10 && m10 > m20 && m20 > m30) {
		t.Fatalf("expected monotonic middle degradation with K: K5=%.3f K10=%.3f K20=%.3f K30=%.3f", m5, m10, m20, m30)
	}
}

func TestModelEvaluator_CustomLiveAdapterHook(t *testing.T) {
	calls := 0
	liveHook := FuncEvaluator{
		EvaluatorName: "mock-live-enso-zen-adapter",
		Simulated:     false,
		Fn: func(ctx context.Context, evalCase EvalCase, packed PackedContext) (EvalScore, error) {
			calls++
			return EvalScore{
				ExactCorrect:        packed.AllGoldRetained,
				CalibratedAccuracy:  packed.GoldRecall,
				AttentionRecallProb: packed.GoldRecall,
				ExtractiveGrounded:  packed.AllGoldRetained,
				CitationGrounded:    packed.AllGoldRetained,
			}, nil
		},
	}

	corpus := GenerateCorpus(1346)
	out, err := evaluateOne(context.Background(), corpus.PrimaryCases[0], PolicyKaiEvidenceEdge, DefaultPackConfig(), liveHook)
	if err != nil {
		t.Fatalf("live hook evaluation failed: %v", err)
	}
	if calls != 1 {
		t.Fatalf("expected live adapter hook to be invoked once, got %d", calls)
	}
	if liveHook.IsSimulated() {
		t.Fatalf("expected live adapter hook IsSimulated() == false")
	}
	if !out.ExactCorrect {
		t.Fatalf("expected custom live adapter to pass on retained gold evidence")
	}
}

func TestRunBenchmark_PrimaryAndAllThreeHonestNegativeRegimes(t *testing.T) {
	report, err := RunBenchmark(context.Background(), DefaultRunConfig())
	if err != nil {
		t.Fatalf("RunBenchmark failed: %v", err)
	}

	if !report.Metadata.EvaluatorSimulated {
		t.Fatalf("offline CI evaluator must transparently report EvaluatorSimulated=true")
	}

	byPolicy := func(rows []PolicySummary) map[PolicyID]PolicySummary {
		m := make(map[PolicyID]PolicySummary, len(rows))
		for _, r := range rows {
			m[r.PolicyID] = r
		}
		return m
	}

	// Primary checks:
	// 1. Zero cross-tenant leaks, zero injection rank hijacks, zero Kai generative output tokens across all policies.
	primary := byPolicy(report.PrimaryOverall)
	for _, p := range AllPolicies {
		s := primary[p]
		if s.CrossTenantLeaks != 0 {
			t.Errorf("policy %s had %d cross-tenant leaks", p, s.CrossTenantLeaks)
		}
		if s.InjectionRankHijacks != 0 {
			t.Errorf("policy %s had %d injection rank hijacks", p, s.InjectionRankHijacks)
		}
		if s.MaxKaiGenOutputTokens != 0 {
			t.Errorf("policy %s violated non-generative Kai invariant: %d", p, s.MaxKaiGenOutputTokens)
		}
	}

	// 2. Kai outperforms raw_concat_baseline on accuracy AND token-cost efficiency, and beats u_shaped_edge_packing on cost efficiency.
	raw := primary[PolicyRawConcatBaseline]
	uEdge := primary[PolicyUShapedEdgePacking]
	kai := primary[PolicyKaiEvidenceEdge]
	if kai.MeanCalibratedAccuracy <= raw.MeanCalibratedAccuracy {
		t.Errorf("expected kai calibrated accuracy (%.4f) > raw_concat_baseline (%.4f)",
			kai.MeanCalibratedAccuracy, raw.MeanCalibratedAccuracy)
	}
	if kai.CostUSDPer1kQueries >= uEdge.CostUSDPer1kQueries {
		t.Errorf("expected kai cost/1k ($%.4f) < u_shaped_edge_packing ($%.4f)",
			kai.CostUSDPer1kQueries, uEdge.CostUSDPer1kQueries)
	}
	if kai.AccuracyPer1kInputTokens <= uEdge.AccuracyPer1kInputTokens {
		t.Errorf("expected kai accuracy/1k tokens (%.3f) > u_shaped_edge_packing (%.3f)",
			kai.AccuracyPer1kInputTokens, uEdge.AccuracyPer1kInputTokens)
	}

	// Honest Negative Regime #1: Severe reranker score inversion causes budgeted pruning (budgeted_top_k_prune & kai)
	// to drop true gold evidence completely and perform WORSE than raw_concat_baseline!
	var invRaw, invPrune, invKai RerankerRegimeRow
	for _, r := range report.NegativeNoisyReranker {
		if r.NoiseLevel != "severe_inversion" {
			continue
		}
		switch r.PolicyID {
		case PolicyRawConcatBaseline:
			invRaw = r
		case PolicyBudgetedTopKPrune:
			invPrune = r
		case PolicyKaiEvidenceEdge:
			invKai = r
		}
	}
	if invPrune.MeanCalibratedAccuracy >= invRaw.MeanCalibratedAccuracy || invKai.MeanCalibratedAccuracy >= invRaw.MeanCalibratedAccuracy {
		t.Fatalf("Negative Regime #1 invariant failed: under severe reranker inversion, pruned policies (prune=%.3f, kai=%.3f) must perform WORSE than raw_concat_baseline (%.3f)",
			invPrune.MeanCalibratedAccuracy, invKai.MeanCalibratedAccuracy, invRaw.MeanCalibratedAccuracy)
	}
	if invKai.AllGoldRetainedRate != 0 {
		t.Errorf("expected severe inversion to drop all gold docs under tight budget (got AllGoldRetainedRate=%.3f)", invKai.AllGoldRetainedRate)
	}

	// Honest Negative Regime #2: Strictly ordered sequential logs — unguarded u_shaped_edge_packing and unguarded Kai
	// suffer chronological order violations, whereas guarded Kai preserves 0 violations and high accuracy!
	var chronoUEdge, chronoKaiUnguarded, chronoKaiGuarded ChronologicalRegimeRow
	for _, r := range report.NegativeChronologicalLog {
		if r.PolicyID == PolicyUShapedEdgePacking {
			chronoUEdge = r
		}
		if r.PolicyID == PolicyKaiEvidenceEdge && !r.GuardEnabled {
			chronoKaiUnguarded = r
		}
		if r.PolicyID == PolicyKaiEvidenceEdge && r.GuardEnabled {
			chronoKaiGuarded = r
		}
	}
	if chronoUEdge.OrderViolationRate <= 0 || chronoKaiUnguarded.OrderViolationRate <= 0 {
		t.Fatalf("Negative Regime #2 invariant failed: unguarded reordering must exhibit order violations (uEdge=%.2f, unguardedKai=%.2f)",
			chronoUEdge.OrderViolationRate, chronoKaiUnguarded.OrderViolationRate)
	}
	if chronoKaiGuarded.OrderViolationRate != 0 || chronoKaiGuarded.ExactPassRate <= chronoUEdge.ExactPassRate {
		t.Fatalf("Negative Regime #2 guard failed: guarded Kai must achieve 0 violations (got %.2f) and outperform unguarded uEdge (%.2f vs %.2f)",
			chronoKaiGuarded.OrderViolationRate, chronoKaiGuarded.ExactPassRate, chronoUEdge.ExactPassRate)
	}

	// Honest Negative Regime #3: M=3 multi-hop synthesis under tight token budget (400 tokens < ~705 gold tokens):
	// budgeted pruning saves tokens but collapses to 0% exact accuracy, losing to unpruned u_shaped_edge_packing.
	var tightUEdge, tightPrune, tightKai TightBudgetSweepRow
	for _, r := range report.NegativeTightBudget {
		if r.TokenBudget != 400 {
			continue
		}
		switch r.PolicyID {
		case PolicyUShapedEdgePacking:
			tightUEdge = r
		case PolicyBudgetedTopKPrune:
			tightPrune = r
		case PolicyKaiEvidenceEdge:
			tightKai = r
		}
	}
	if tightPrune.ExactPassRate != 0 || tightKai.ExactPassRate != 0 {
		t.Fatalf("Negative Regime #3 invariant failed: at budget=400 (< M=3 gold docs), budgeted policies must collapse to 0%% (prune=%.2f, kai=%.2f)",
			tightPrune.ExactPassRate, tightKai.ExactPassRate)
	}
	if tightUEdge.MeanCalibratedAccuracy <= tightKai.MeanCalibratedAccuracy {
		t.Fatalf("Negative Regime #3 invariant failed: unpruned u_shaped_edge_packing (%.3f) must beat budget-starved Kai (%.3f) at budget=400",
			tightUEdge.MeanCalibratedAccuracy, tightKai.MeanCalibratedAccuracy)
	}
}

func TestConcurrency_16GoroutinesRaceFreeAndDeterministic(t *testing.T) {
	const workers = 16
	corpus := GenerateCorpus(1346)
	cfg := DefaultPackConfig()
	eval := NewCalibratedEvaluator()

	// Reference sequential run on first 20 primary cases across all 5 policies.
	subset := corpus.PrimaryCases[:20]
	runOnce := func() ([]byte, error) {
		outs := make([]CaseOutcome, 0, len(subset)*len(AllPolicies))
		for _, ec := range subset {
			for _, policy := range AllPolicies {
				out, err := evaluateOne(context.Background(), ec, policy, cfg, eval)
				if err != nil {
					return nil, err
				}
				// Zero non-deterministic wall-clock latency field for bit-for-bit equality check.
				out.DecisionLatencyMicros = 0
				outs = append(outs, out)
			}
		}
		return json.Marshal(outs)
	}

	expected, err := runOnce()
	if err != nil {
		t.Fatalf("reference run failed: %v", err)
	}

	var wg sync.WaitGroup
	errCh := make(chan error, workers)
	resCh := make(chan []byte, workers)

	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got, err := runOnce()
			if err != nil {
				errCh <- err
				return
			}
			resCh <- got
		}()
	}

	wg.Wait()
	close(errCh)
	close(resCh)

	for err := range errCh {
		t.Fatalf("concurrent worker failed: %v", err)
	}
	for got := range resCh {
		if string(got) != string(expected) {
			t.Fatalf("concurrent worker produced non-deterministic output mismatch")
		}
	}
}
