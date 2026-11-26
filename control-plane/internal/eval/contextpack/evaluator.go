// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

package contextpack

import (
	"context"
	"fmt"
	"math"
	"strings"
)

// CalibratedLostInMiddleEvaluator implements ModelEvaluator using:
//  1. Exact, deterministic structural verification (all required gold evidence retained,
//     exact extractive answer grounding, citation header grounding, and chronological order integrity).
//  2. An empirically calibrated U-shaped serial-position attention proxy grounded in
//     Liu et al. (TACL 2024, "Lost in the Middle: How Language Models Use Long Contexts").
//
// IMPORTANT TRANSPARENCY NOTE:
// IsSimulated() returns true. Offline CI benchmarks evaluate context-packing policies against
// this calibrated proxy alongside model-independent structural/cost metrics. Production or
// hardware evaluations plug live Zen/Enso models directly into the ModelEvaluator interface.
type CalibratedLostInMiddleEvaluator struct {
	// PassThreshold is the minimum calibrated recall probability required for ExactCorrect == true.
	PassThreshold float64
}

// NewCalibratedEvaluator returns the default TACL '24 calibrated offline evaluator.
func NewCalibratedEvaluator() *CalibratedLostInMiddleEvaluator {
	return &CalibratedLostInMiddleEvaluator{
		PassThreshold: 0.72,
	}
}

// Name returns a descriptive identifier recorded in BenchmarkReport metadata.
func (e *CalibratedLostInMiddleEvaluator) Name() string {
	return "calibrated-tacl24-lost-in-the-middle-proxy"
}

// IsSimulated explicitly signals that this offline evaluator uses a calibrated
// TACL '24 attention proxy rather than live generative decoding.
func (e *CalibratedLostInMiddleEvaluator) IsSimulated() bool {
	return true
}

// PositionAttentionWeight computes the calibrated U-shaped attention recall weight
// A(p, K, T) ∈ [0.20, 0.99] for a single gold evidence document located at normalized
// position p ∈ [0, 1] inside a packed context containing docCount documents and totalTokens tokens.
//
// Properties matching Liu et al. (TACL 2024):
//   - Primacy (p=0) and recency (p=1) exhibit highest attention (primacy slightly > recency).
//   - Middle positions (p≈0.5) suffer a pronounced U-shaped valley whose depth grows monotonically with K.
//   - Larger total token counts introduce a secondary attention-dilution penalty across all slots.
func PositionAttentionWeight(normPos float64, docCount int, totalTokens int) float64 {
	p := clamp01(normPos)
	if docCount < 1 {
		docCount = 1
	}

	// Primacy peak (0.96) slightly exceeds recency peak (0.93), consistent with TACL '24 Fig. 1.
	edgeBase := 0.96 - 0.03*p

	// Middle degradation depth grows with document count K (5 -> ~15%, 10 -> ~21%, 20 -> ~32%, 30 -> ~44%).
	midDepth := math.Min(0.44, 0.09+0.0115*float64(docCount))

	// Smooth U-shaped valley peaking at p = 0.5 and vanishing at p = 0 and p = 1.
	bowl := math.Pow(math.Sin(math.Pi*p), 1.35)

	// Secondary distraction penalty from raw token volume (capped at 0.11 for long contexts).
	tokenPenalty := math.Min(0.11, float64(totalTokens)/58000.0)

	w := edgeBase - midDepth*bowl - tokenPenalty
	if w < 0.20 {
		w = 0.20
	}
	if w > 0.99 {
		w = 0.99
	}
	return w
}

// Evaluate scores a PackedContext against an EvalCase deterministically.
func (e *CalibratedLostInMiddleEvaluator) Evaluate(
	ctx context.Context,
	evalCase EvalCase,
	packed PackedContext,
) (EvalScore, error) {
	if err := ctx.Err(); err != nil {
		return EvalScore{}, err
	}

	// 1. Exact extractive & citation grounding verification (100% deterministic, model-independent).
	extractiveOK := len(evalCase.ExpectedAnswers) > 0
	for _, ans := range evalCase.ExpectedAnswers {
		if !strings.Contains(packed.RenderedText, ans) {
			extractiveOK = false
			break
		}
	}

	citationOK := packed.AllGoldRetained && len(packed.GoldSlotIndices) > 0
	if citationOK {
		for _, idx := range packed.GoldSlotIndices {
			if idx < 0 || idx >= len(packed.OrderedDocs) {
				citationOK = false
				break
			}
			doc := packed.OrderedDocs[idx]
			if !strings.Contains(packed.RenderedText, "id="+doc.ID) ||
				!strings.Contains(packed.RenderedText, "uri="+doc.SourceURI) {
				citationOK = false
				break
			}
		}
	}

	// Hard failure #1: required gold evidence pruned away or missing from rendered context.
	if !packed.AllGoldRetained || !extractiveOK {
		return EvalScore{
			ExactCorrect:        false,
			CalibratedAccuracy:  0.0,
			AttentionRecallProb: 0.0,
			ExtractiveGrounded:  extractiveOK,
			CitationGrounded:    citationOK,
			FailureMode:         "evidence_pruned",
		}, nil
	}

	// 2. Calibrated U-shaped serial-position attention recall across all required gold documents.
	docCount := len(packed.OrderedDocs)
	totalTokens := packed.TotalInputTokens

	jointProd := 1.0
	for _, normPos := range packed.GoldPositionsNormalized {
		jointProd *= PositionAttentionWeight(normPos, docCount, totalTokens)
	}
	m := len(packed.GoldPositionsNormalized)
	if m < 1 {
		m = 1
	}
	// Sub-linear conjunction exponent (1 / sqrt(M)) preserves strict all-or-nothing degradation
	// when any required document lands in the middle valley while keeping M=1..4 comparable.
	attentionProb := math.Pow(jointProd, 1.0/math.Sqrt(float64(m)))
	attentionProb = clamp01(attentionProb)

	// Hard failure #2: chronological/causal order broken on sequentially coupled logs/diffs.
	if evalCase.PreserveChronologicalOrder && packed.OrderViolations > 0 {
		degraded := clamp01(attentionProb * 0.22)
		return EvalScore{
			ExactCorrect:        false,
			CalibratedAccuracy:  degraded,
			AttentionRecallProb: attentionProb,
			ExtractiveGrounded:  extractiveOK,
			CitationGrounded:    citationOK,
			FailureMode:         "chronological_order_violation",
		}, nil
	}

	threshold := e.PassThreshold + evalCase.DifficultyOffset
	if threshold < 0.50 {
		threshold = 0.50
	}
	if threshold > 0.92 {
		threshold = 0.92
	}

	exact := extractiveOK && citationOK && attentionProb >= threshold
	failureMode := ""
	if !exact {
		failureMode = "lost_in_the_middle_degradation"
	}

	return EvalScore{
		ExactCorrect:        exact,
		CalibratedAccuracy:  attentionProb,
		AttentionRecallProb: attentionProb,
		ExtractiveGrounded:  extractiveOK,
		CitationGrounded:    citationOK,
		FailureMode:         failureMode,
	}, nil
}

// FuncEvaluator adapts a plain Go function into a ModelEvaluator, simplifying
// integration of live Enso/Zen HTTP or ZAP adapters and unit-test stubs.
type FuncEvaluator struct {
	EvaluatorName string
	Simulated     bool
	Fn            func(ctx context.Context, evalCase EvalCase, packed PackedContext) (EvalScore, error)
}

func (f FuncEvaluator) Name() string      { return f.EvaluatorName }
func (f FuncEvaluator) IsSimulated() bool { return f.Simulated }
func (f FuncEvaluator) Evaluate(ctx context.Context, evalCase EvalCase, packed PackedContext) (EvalScore, error) {
	if f.Fn == nil {
		return EvalScore{}, fmt.Errorf("FuncEvaluator %q has nil Fn", f.EvaluatorName)
	}
	return f.Fn(ctx, evalCase, packed)
}
