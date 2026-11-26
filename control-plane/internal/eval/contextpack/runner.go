// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

package contextpack

import (
	"context"
	"fmt"
	"math"
	"runtime"
	"sort"
	"strings"
)

// RunConfig configures an end-to-end HIP-1346 context-packing benchmark run.
type RunConfig struct {
	Seed       int64
	PackConfig PackConfig
	Evaluator  ModelEvaluator
}

// DefaultRunConfig returns the canonical reproducible benchmark configuration (--seed 1346).
func DefaultRunConfig() RunConfig {
	return RunConfig{
		Seed:       1346,
		PackConfig: DefaultPackConfig(),
		Evaluator:  NewCalibratedEvaluator(),
	}
}

// RunBenchmark executes the complete primary suite (100 cases) and all 3 honest negative-regime
// suites across all 5 matched context-packing policies, returning a fully populated BenchmarkReport.
func RunBenchmark(ctx context.Context, cfg RunConfig) (*BenchmarkReport, error) {
	if cfg.Evaluator == nil {
		cfg.Evaluator = NewCalibratedEvaluator()
	}
	if cfg.PackConfig.TokenBudget <= 0 {
		cfg.PackConfig = DefaultPackConfig()
	}

	corpus := GenerateCorpus(cfg.Seed)

	// 1. Primary Suite (100 cases x 5 policies = 500 evaluations).
	primaryOutcomes := make([]CaseOutcome, 0, len(corpus.PrimaryCases)*len(AllPolicies))
	for _, ec := range corpus.PrimaryCases {
		for _, policy := range AllPolicies {
			out, err := evaluateOne(ctx, ec, policy, cfg.PackConfig, cfg.Evaluator)
			if err != nil {
				return nil, fmt.Errorf("primary case %s policy %s: %w", ec.ID, policy, err)
			}
			primaryOutcomes = append(primaryOutcomes, out)
		}
	}

	// Filter slices for primary sub-reports.
	posSweepOutcomes := filterOutcomesByCategory(primaryOutcomes, "position_sweep")
	multiOutcomes := filterOutcomesByCategory(primaryOutcomes, "multi_evidence_synthesis")
	secOutcomes := filterOutcomesByCategory(primaryOutcomes, "security_acl_injection")

	// 2. Negative Regime #1: Noisy / Adversarially Inverted Reranker.
	rerankerRows, err := runNoisyRerankerRegime(ctx, corpus.NoisyRerankerCases, cfg.PackConfig, cfg.Evaluator)
	if err != nil {
		return nil, err
	}

	// 3. Negative Regime #2: Strictly Ordered Sequential Logs / Diffs (guarded vs unguarded).
	chronoRows, err := runChronologicalLogRegime(ctx, corpus.ChronologicalCases, cfg.PackConfig, cfg.Evaluator)
	if err != nil {
		return nil, err
	}

	// 4. Negative Regime #3: Multi-Evidence M=3 Synthesis under Tight Token Budget Sweep.
	tightBudgetRows, err := runTightBudgetRegime(ctx, corpus.TightBudgetCases, cfg.PackConfig, cfg.Evaluator)
	if err != nil {
		return nil, err
	}

	simNote := "Live model evaluation via pluggable ModelEvaluator adapter."
	if cfg.Evaluator.IsSimulated() {
		simNote = "Offline CI uses a deterministic, empirically calibrated TACL '24 (Liu et al., 2024) U-shaped position-sensitivity proxy paired with exact model-independent structural/extractive/citation/cost metrics. Live Zen/Enso runs plug into the same ModelEvaluator interface."
	}

	negCount := len(corpus.NoisyRerankerCases) + len(corpus.ChronologicalCases) + len(corpus.TightBudgetCases)
	report := &BenchmarkReport{
		Metadata: BenchmarkMetadata{
			BenchmarkName:       "HIP-1346 Lost-in-the-Middle Context Packing Benchmark",
			HIP:                 "HIP-1346",
			Issue:               "hanzoai/agents#83",
			Seed:                cfg.Seed,
			GoVersion:           runtime.Version(),
			GOOS:                runtime.GOOS,
			GOARCH:              runtime.GOARCH,
			NumCPU:              runtime.NumCPU(),
			EvaluatorName:       cfg.Evaluator.Name(),
			EvaluatorSimulated:  cfg.Evaluator.IsSimulated(),
			SimulationNote:      simNote,
			PackConfig:          cfg.PackConfig,
			PrimaryCaseCount:    len(corpus.PrimaryCases),
			NegativeRegimeCases: negCount,
		},
		PrimaryOverall:           summarizeByPolicy(primaryOutcomes),
		PositionSweepOverall:     summarizeByPolicy(posSweepOutcomes),
		PositionCurve:            buildPositionCurve(posSweepOutcomes),
		MultiEvidenceSummary:     summarizeByPolicy(multiOutcomes),
		SecurityAuditSummary:     summarizeByPolicy(secOutcomes),
		NegativeNoisyReranker:    rerankerRows,
		NegativeChronologicalLog: chronoRows,
		NegativeTightBudget:      tightBudgetRows,
		PrimaryOutcomes:          primaryOutcomes,
	}

	return report, nil
}

func evaluateOne(
	ctx context.Context,
	evalCase EvalCase,
	policy PolicyID,
	packCfg PackConfig,
	eval ModelEvaluator,
) (CaseOutcome, error) {
	packed, err := PackContext(evalCase, policy, packCfg)
	if err != nil {
		return CaseOutcome{}, err
	}
	score, err := eval.Evaluate(ctx, evalCase, packed)
	if err != nil {
		return CaseOutcome{}, err
	}
	return CaseOutcome{
		CaseID:                    evalCase.ID,
		Suite:                     evalCase.Suite,
		Category:                  evalCase.Category,
		ContextLength:             evalCase.ContextLength,
		GoldSlotLabel:             evalCase.GoldSlotLabel,
		PolicyID:                  policy,
		ExactCorrect:              score.ExactCorrect,
		CalibratedAccuracy:        score.CalibratedAccuracy,
		AttentionRecallProb:       score.AttentionRecallProb,
		ExtractiveGrounded:        score.ExtractiveGrounded,
		CitationGrounded:          score.CitationGrounded,
		AllGoldRetained:           packed.AllGoldRetained,
		GoldRecall:                packed.GoldRecall,
		GoldInEdgeWindow:          packed.GoldInEdgeWindow,
		InputTokens:               packed.TotalInputTokens,
		RawAuthorizedTokens:       packed.RawAuthorizedTokens,
		TokenSavingsRatio:         packed.TokenSavingsRatio,
		EstimatedInputCostUSD:     packed.EstimatedInputCostUSD,
		CrossTenantLeaks:          packed.CrossTenantLeaks,
		InjectionRankHijacks:      packed.InjectionRankHijacks,
		OrderViolations:           packed.OrderViolations,
		KaiGenerativeOutputTokens: packed.KaiDecision.GenerativeOutputTokens,
		DecisionLatencyMicros:     packed.KaiDecision.DecisionLatencyMicros,
		FailureMode:               score.FailureMode,
	}, nil
}

func runNoisyRerankerRegime(
	ctx context.Context,
	cases []EvalCase,
	packCfg PackConfig,
	eval ModelEvaluator,
) ([]RerankerRegimeRow, error) {
	levels := []string{"clean", "moderate_noise", "severe_inversion"}
	rows := make([]RerankerRegimeRow, 0, len(levels)*len(AllPolicies))

	for _, level := range levels {
		var levelCases []EvalCase
		for _, c := range cases {
			if c.RerankerNoiseLevel == level {
				levelCases = append(levelCases, c)
			}
		}
		if len(levelCases) == 0 {
			continue
		}

		// Measure upstream reranker Recall@6 (matching ~1400-token budget window of ~6 docs).
		topKHits := 0
		for _, c := range levelCases {
			if goldInTopKByBaseRelevance(c, 6) {
				topKHits++
			}
		}
		recallAt6 := float64(topKHits) / float64(len(levelCases))

		for _, policy := range AllPolicies {
			var outs []CaseOutcome
			for _, c := range levelCases {
				out, err := evaluateOne(ctx, c, policy, packCfg, eval)
				if err != nil {
					return nil, fmt.Errorf("noisy reranker %s/%s: %w", c.ID, policy, err)
				}
				outs = append(outs, out)
			}
			sum := aggregateOutcomes(policy, outs)
			rows = append(rows, RerankerRegimeRow{
				NoiseLevel:             level,
				RerankerTopKRecall:     round4(recallAt6),
				PolicyID:               policy,
				CaseCount:              sum.CaseCount,
				ExactPassRate:          sum.ExactPassRate,
				MeanCalibratedAccuracy: sum.MeanCalibratedAccuracy,
				AllGoldRetainedRate:    sum.AllGoldRetainedRate,
				AvgInputTokens:         sum.AvgInputTokens,
			})
		}
	}
	return rows, nil
}

func goldInTopKByBaseRelevance(evalCase EvalCase, k int) bool {
	type pair struct {
		score  float64
		isGold bool
		idx    int
	}
	pairs := make([]pair, 0, len(evalCase.Documents))
	for i, d := range evalCase.Documents {
		if !evalCase.Scope.CanAccess(d.TenantID, d.WorkspaceID, d.RequiredRole) {
			continue
		}
		pairs = append(pairs, pair{score: d.BaseRelevance, isGold: d.IsGold, idx: i})
	}
	sort.SliceStable(pairs, func(i, j int) bool {
		if pairs[i].score != pairs[j].score {
			return pairs[i].score > pairs[j].score
		}
		return pairs[i].idx < pairs[j].idx
	})
	limit := k
	if limit > len(pairs) {
		limit = len(pairs)
	}
	for i := 0; i < limit; i++ {
		if pairs[i].isGold {
			return true
		}
	}
	return false
}

func runChronologicalLogRegime(
	ctx context.Context,
	cases []EvalCase,
	packCfg PackConfig,
	eval ModelEvaluator,
) ([]ChronologicalRegimeRow, error) {
	type runSpec struct {
		policy       PolicyID
		guardEnabled bool
	}
	specs := []runSpec{
		{PolicyRawConcatBaseline, false},
		{PolicyRelevanceDesc, false},
		{PolicyUShapedEdgePacking, false},
		{PolicyBudgetedTopKPrune, false},
		{PolicyKaiEvidenceEdge, false}, // Unguarded Kai ablation (proves why chronological lock is required!)
		{PolicyKaiEvidenceEdge, true},  // Guarded Kai default (PreserveChronologicalOrder enforced)
	}

	rows := make([]ChronologicalRegimeRow, 0, len(specs))
	for _, sp := range specs {
		cfg := packCfg
		cfg.EnforceChronologicalOrder = sp.guardEnabled
		var outs []CaseOutcome
		violCases := 0
		for _, c := range cases {
			out, err := evaluateOne(ctx, c, sp.policy, cfg, eval)
			if err != nil {
				return nil, fmt.Errorf("chronological regime %s/%s: %w", c.ID, sp.policy, err)
			}
			if out.OrderViolations > 0 {
				violCases++
			}
			outs = append(outs, out)
		}
		sum := aggregateOutcomes(sp.policy, outs)
		rows = append(rows, ChronologicalRegimeRow{
			PolicyID:               sp.policy,
			GuardEnabled:           sp.guardEnabled,
			CaseCount:              sum.CaseCount,
			ExactPassRate:          sum.ExactPassRate,
			MeanCalibratedAccuracy: sum.MeanCalibratedAccuracy,
			OrderViolationRate:     round4(float64(violCases) / float64(len(cases))),
			TotalOrderViolations:   sum.OrderViolations,
			AvgInputTokens:         sum.AvgInputTokens,
		})
	}
	return rows, nil
}

func runTightBudgetRegime(
	ctx context.Context,
	cases []EvalCase,
	packCfg PackConfig,
	eval ModelEvaluator,
) ([]TightBudgetSweepRow, error) {
	budgets := []int{2400, 1400, 900, 600, 400}
	rows := make([]TightBudgetSweepRow, 0, len(budgets)*len(AllPolicies))

	for _, budget := range budgets {
		cfg := packCfg
		cfg.TokenBudget = budget
		for _, policy := range AllPolicies {
			var outs []CaseOutcome
			for _, c := range cases {
				out, err := evaluateOne(ctx, c, policy, cfg, eval)
				if err != nil {
					return nil, fmt.Errorf("tight budget %d %s/%s: %w", budget, c.ID, policy, err)
				}
				outs = append(outs, out)
			}
			sum := aggregateOutcomes(policy, outs)
			rows = append(rows, TightBudgetSweepRow{
				TokenBudget:            budget,
				PolicyID:               policy,
				CaseCount:              sum.CaseCount,
				ExactPassRate:          sum.ExactPassRate,
				MeanCalibratedAccuracy: sum.MeanCalibratedAccuracy,
				AllGoldRetainedRate:    sum.AllGoldRetainedRate,
				GoldRecallRate:         sum.GoldRecallRate,
				AvgInputTokens:         sum.AvgInputTokens,
				CostUSDPer1kQueries:    sum.CostUSDPer1kQueries,
			})
		}
	}
	return rows, nil
}

func filterOutcomesByCategory(all []CaseOutcome, category string) []CaseOutcome {
	out := make([]CaseOutcome, 0)
	for _, o := range all {
		if o.Category == category {
			out = append(out, o)
		}
	}
	return out
}

func summarizeByPolicy(outcomes []CaseOutcome) []PolicySummary {
	summaries := make([]PolicySummary, 0, len(AllPolicies))
	for _, policy := range AllPolicies {
		var subset []CaseOutcome
		for _, o := range outcomes {
			if o.PolicyID == policy {
				subset = append(subset, o)
			}
		}
		summaries = append(summaries, aggregateOutcomes(policy, subset))
	}
	return summaries
}

func aggregateOutcomes(policy PolicyID, outs []CaseOutcome) PolicySummary {
	n := len(outs)
	if n == 0 {
		return PolicySummary{PolicyID: policy}
	}
	var exactCount, allGoldCount, edgeCount, citeCount int
	var sumAcc, sumRecall, sumCost float64
	var sumTokens, sumRawTokens int
	var sumLatency int64
	var leaks, hijacks, orderViols, maxGen int

	for _, o := range outs {
		if o.ExactCorrect {
			exactCount++
		}
		if o.AllGoldRetained {
			allGoldCount++
		}
		if o.GoldInEdgeWindow {
			edgeCount++
		}
		if o.CitationGrounded {
			citeCount++
		}
		sumAcc += o.CalibratedAccuracy
		sumRecall += o.GoldRecall
		sumCost += o.EstimatedInputCostUSD
		sumTokens += o.InputTokens
		sumRawTokens += o.RawAuthorizedTokens
		sumLatency += o.DecisionLatencyMicros
		leaks += o.CrossTenantLeaks
		hijacks += o.InjectionRankHijacks
		orderViols += o.OrderViolations
		if o.KaiGenerativeOutputTokens > maxGen {
			maxGen = o.KaiGenerativeOutputTokens
		}
	}

	nf := float64(n)
	exactRate := float64(exactCount) / nf
	meanAcc := sumAcc / nf
	avgTokens := float64(sumTokens) / nf
	avgRaw := float64(sumRawTokens) / nf
	savingsPct := 0.0
	if avgRaw > 0 {
		savingsPct = (1.0 - avgTokens/avgRaw) * 100.0
		if savingsPct < 0 {
			savingsPct = 0
		}
	}
	costPer1k := (sumCost / nf) * 1000.0
	accPer1kTok := 0.0
	exactPer1kTok := 0.0
	if avgTokens > 0 {
		accPer1kTok = (meanAcc * 1000.0) / avgTokens
		exactPer1kTok = (exactRate * 1000.0) / avgTokens
	}

	return PolicySummary{
		PolicyID:                  policy,
		CaseCount:                 n,
		ExactPassRate:             round4(exactRate),
		MeanCalibratedAccuracy:    round4(meanAcc),
		GoldRecallRate:            round4(sumRecall / nf),
		AllGoldRetainedRate:       round4(float64(allGoldCount) / nf),
		GoldInEdgeWindowRate:      round4(float64(edgeCount) / nf),
		CitationGroundedRate:      round4(float64(citeCount) / nf),
		AvgInputTokens:            round2(avgTokens),
		AvgRawTokens:              round2(avgRaw),
		TokenSavingsPct:           round2(savingsPct),
		CostUSDPer1kQueries:       round4(costPer1k),
		AccuracyPer1kInputTokens:  round4(accPer1kTok),
		ExactPassPer1kInputTokens: round4(exactPer1kTok),
		CrossTenantLeaks:          leaks,
		InjectionRankHijacks:      hijacks,
		OrderViolations:           orderViols,
		MaxKaiGenOutputTokens:     maxGen,
		AvgDecisionLatencyMicros:  round2(float64(sumLatency) / nf),
	}
}

func buildPositionCurve(posOutcomes []CaseOutcome) []PositionCurveCell {
	lengths := []int{5, 10, 20, 30}
	cells := make([]PositionCurveCell, 0, len(AllPolicies)*len(lengths)*len(AllSingleSlots))

	for _, policy := range AllPolicies {
		for _, k := range lengths {
			for _, slot := range AllSingleSlots {
				var subset []CaseOutcome
				for _, o := range posOutcomes {
					if o.PolicyID == policy && o.ContextLength == k && o.GoldSlotLabel == slot {
						subset = append(subset, o)
					}
				}
				if len(subset) == 0 {
					continue
				}
				sum := aggregateOutcomes(policy, subset)
				cells = append(cells, PositionCurveCell{
					PolicyID:               policy,
					ContextLength:          k,
					GoldSlotLabel:          slot,
					CaseCount:              sum.CaseCount,
					ExactPassRate:          sum.ExactPassRate,
					MeanCalibratedAccuracy: sum.MeanCalibratedAccuracy,
					GoldInEdgeWindowRate:   sum.GoldInEdgeWindowRate,
					AvgInputTokens:         sum.AvgInputTokens,
				})
			}
		}
	}
	return cells
}

// RenderMarkdownScoreTable formats a complete, auditable Markdown score table directly
// from a BenchmarkReport with zero hand-copied numbers.
func RenderMarkdownScoreTable(r *BenchmarkReport) string {
	var b strings.Builder

	fmt.Fprintf(&b, "# HIP-1346 Score Table: Lost-in-the-Middle Context Packing (`hanzoai/agents#83`)\n\n")
	fmt.Fprintf(&b, "- **Seed**: `%d` | **Go**: `%s` (`%s/%s`, `NumCPU=%d`)\n",
		r.Metadata.Seed, r.Metadata.GoVersion, r.Metadata.GOOS, r.Metadata.GOARCH, r.Metadata.NumCPU)
	fmt.Fprintf(&b, "- **Evaluator**: `%s` (`simulated=%v`)\n", r.Metadata.EvaluatorName, r.Metadata.EvaluatorSimulated)
	fmt.Fprintf(&b, "- **Token Budget (Budgeted Policies)**: `%d` tokens | **Enso Pricing**: `$%.2f / 1M` input tokens | **Kai Overhead**: `$%.2f / 1k` calls (`GenerativeOutputTokens = 0`)\n",
		r.Metadata.PackConfig.TokenBudget, r.Metadata.PackConfig.EnsoInputUSDPer1MTokens, r.Metadata.PackConfig.KaiDecisionUSDPer1kCalls)
	fmt.Fprintf(&b, "- **Corpus**: `%d` primary cases (`60` position sweep + `24` multi-evidence synthesis + `16` security/ACL/injection) + `%d` negative-regime cases\n\n",
		r.Metadata.PrimaryCaseCount, r.Metadata.NegativeRegimeCases)

	b.WriteString("## 1. Overall Primary Suite Comparison (100 Cases)\n\n")
	writePolicySummaryTable(&b, r.PrimaryOverall)

	b.WriteString("\n## 2. Single-Evidence Position Curve (`K ∈ {5, 10, 20, 30}` x 5 Canonical Slots)\n\n")
	b.WriteString("Calibrated accuracy (`ExactPass%`) across gold evidence slots `start (0)`, `q1 (K/4)`, `middle (K/2)`, `q3 (3K/4)`, `end (K-1)`:\n\n")

	cellIdx := make(map[string]PositionCurveCell, len(r.PositionCurve))
	for _, c := range r.PositionCurve {
		key := fmt.Sprintf("%s|%d|%s", c.PolicyID, c.ContextLength, c.GoldSlotLabel)
		cellIdx[key] = c
	}

	b.WriteString("| Policy | K | Start (`0`) | Q1 (`K/4`) | Middle (`K/2`) | Q3 (`3K/4`) | End (`K-1`) | Middle Drop vs Start | Avg Input Tokens |\n")
	b.WriteString("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
	for _, policy := range AllPolicies {
		for _, k := range []int{5, 10, 20, 30} {
			cStart := cellIdx[fmt.Sprintf("%s|%d|%s", policy, k, SlotStart)]
			cQ1 := cellIdx[fmt.Sprintf("%s|%d|%s", policy, k, SlotQ1)]
			cMid := cellIdx[fmt.Sprintf("%s|%d|%s", policy, k, SlotMiddle)]
			cQ3 := cellIdx[fmt.Sprintf("%s|%d|%s", policy, k, SlotQ3)]
			cEnd := cellIdx[fmt.Sprintf("%s|%d|%s", policy, k, SlotEnd)]
			dropPP := (cStart.MeanCalibratedAccuracy - cMid.MeanCalibratedAccuracy) * 100.0
			avgTok := (cStart.AvgInputTokens + cQ1.AvgInputTokens + cMid.AvgInputTokens + cQ3.AvgInputTokens + cEnd.AvgInputTokens) / 5.0
			fmt.Fprintf(&b, "| `%s` | %d | %.1f%% (%.0f%%) | %.1f%% (%.0f%%) | %.1f%% (%.0f%%) | %.1f%% (%.0f%%) | %.1f%% (%.0f%%) | %+.1f pp | %.0f |\n",
				policy, k,
				cStart.MeanCalibratedAccuracy*100, cStart.ExactPassRate*100,
				cQ1.MeanCalibratedAccuracy*100, cQ1.ExactPassRate*100,
				cMid.MeanCalibratedAccuracy*100, cMid.ExactPassRate*100,
				cQ3.MeanCalibratedAccuracy*100, cQ3.ExactPassRate*100,
				cEnd.MeanCalibratedAccuracy*100, cEnd.ExactPassRate*100,
				-dropPP, avgTok,
			)
		}
	}

	b.WriteString("\n## 3. Multi-Evidence Synthesis (`M ∈ {2, 3}` Required Docs with Near-Duplicate Noise, 24 Cases)\n\n")
	writePolicySummaryTable(&b, r.MultiEvidenceSummary)

	b.WriteString("\n## 4. Security, Multi-Tenant ACL & Prompt-Injection Audit (16 Stress Cases)\n\n")
	b.WriteString("| Policy | Exact Pass | Calibrated Acc | Gold Edge % | Cross-Tenant Leaks | Injection Rank Hijacks | Kai Gen Output Tokens |\n")
	b.WriteString("|---|---:|---:|---:|---:|---:|---:|\n")
	for _, s := range r.SecurityAuditSummary {
		fmt.Fprintf(&b, "| `%s` | %.1f%% | %.1f%% | %.1f%% | %d | %d | %d |\n",
			s.PolicyID, s.ExactPassRate*100, s.MeanCalibratedAccuracy*100, s.GoldInEdgeWindowRate*100,
			s.CrossTenantLeaks, s.InjectionRankHijacks, s.MaxKaiGenOutputTokens)
	}

	b.WriteString("\n## 5. Honest Negative Regime #1: Noisy / Adversarially Inverted Reranker (36 Cases, `K=20`)\n\n")
	b.WriteString("| Reranker Regime | Reranker Recall@6 | Policy | Exact Pass | Calibrated Acc | All-Gold Retained | Avg Input Tokens |\n")
	b.WriteString("|---|---:|---|---:|---:|---:|---:|\n")
	for _, row := range r.NegativeNoisyReranker {
		fmt.Fprintf(&b, "| `%s` | %.1f%% | `%s` | %.1f%% | %.1f%% | %.1f%% | %.0f |\n",
			row.NoiseLevel, row.RerankerTopKRecall*100, row.PolicyID,
			row.ExactPassRate*100, row.MeanCalibratedAccuracy*100, row.AllGoldRetainedRate*100, row.AvgInputTokens)
	}

	b.WriteString("\n## 6. Honest Negative Regime #2: Strictly Ordered Sequential Logs / Diffs (18 Cases)\n\n")
	b.WriteString("| Policy | Chrono Guard | Exact Pass | Calibrated Acc | Order Violation Rate | Total Order Violations | Avg Input Tokens |\n")
	b.WriteString("|---|---|---:|---:|---:|---:|---:|\n")
	for _, row := range r.NegativeChronologicalLog {
		fmt.Fprintf(&b, "| `%s` | `%v` | %.1f%% | %.1f%% | %.1f%% | %d | %.0f |\n",
			row.PolicyID, row.GuardEnabled, row.ExactPassRate*100, row.MeanCalibratedAccuracy*100,
			row.OrderViolationRate*100, row.TotalOrderViolations, row.AvgInputTokens)
	}

	b.WriteString("\n## 7. Honest Negative Regime #3: Tight Token Budget on `M=3` Multi-Hop Synthesis (15 Cases, `K=16`)\n\n")
	b.WriteString("| Token Budget | Policy | Exact Pass | Calibrated Acc | All-Gold Retained | Gold Recall | Avg Input Tokens | Cost USD / 1k Queries |\n")
	b.WriteString("|---:|---|---:|---:|---:|---:|---:|---:|\n")
	for _, row := range r.NegativeTightBudget {
		fmt.Fprintf(&b, "| %d | `%s` | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.0f | $%.4f |\n",
			row.TokenBudget, row.PolicyID, row.ExactPassRate*100, row.MeanCalibratedAccuracy*100,
			row.AllGoldRetainedRate*100, row.GoldRecallRate*100, row.AvgInputTokens, row.CostUSDPer1kQueries)
	}

	return b.String()
}

func writePolicySummaryTable(b *strings.Builder, rows []PolicySummary) {
	b.WriteString("| Policy | Exact Pass | Calibrated Acc | All-Gold Retained | Gold in Edge % | Avg Tokens | Token Savings | Cost USD/1k | Acc / 1k Tok | Leaks | Hijacks | Kai Gen Tok |\n")
	b.WriteString("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
	for _, s := range rows {
		fmt.Fprintf(b, "| `%s` | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.0f | %.1f%% | $%.4f | %.3f | %d | %d | %d |\n",
			s.PolicyID,
			s.ExactPassRate*100,
			s.MeanCalibratedAccuracy*100,
			s.AllGoldRetainedRate*100,
			s.GoldInEdgeWindowRate*100,
			s.AvgInputTokens,
			s.TokenSavingsPct,
			s.CostUSDPer1kQueries,
			s.AccuracyPer1kInputTokens,
			s.CrossTenantLeaks,
			s.InjectionRankHijacks,
			s.MaxKaiGenOutputTokens,
		)
	}
}

func round2(v float64) float64 { return math.Round(v*100) / 100 }
func round4(v float64) float64 { return math.Round(v*10000) / 10000 }
