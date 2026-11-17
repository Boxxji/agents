package membench

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"
)

// ReaderOutput captures an answer and token/cost accounting over packed documents.
type ReaderOutput struct {
	Answer      string  `json:"answer"`
	InputTokens int     `json:"input_tokens"`
	CostUSD     float64 `json:"cost_usd"`
	GoldFound   bool    `json:"gold_found"`
	GoldPos     float64 `json:"gold_pos"`
}

// ContextReader models an answer generator consuming packed documents in order.
type ContextReader interface {
	ReadAndAnswer(ctx context.Context, query string, ordered []PackedDoc) (ReaderOutput, error)
}

// UShapedAttentionReader deterministically models TACL '24 middle-position degradation:
// gold evidence placed near the start or end edges passes RecallThreshold, whereas
// gold evidence buried in the middle falls below RecallThreshold.
type UShapedAttentionReader struct {
	RecallThreshold float64
	CostPer1KTokens float64
}

func (r UShapedAttentionReader) threshold() float64 {
	if r.RecallThreshold <= 0 {
		return 0.55
	}
	return r.RecallThreshold
}

func (r UShapedAttentionReader) ReadAndAnswer(ctx context.Context, _ string, ordered []PackedDoc) (ReaderOutput, error) {
	if err := ctx.Err(); err != nil {
		return ReaderOutput{}, err
	}
	var out ReaderOutput
	for _, item := range ordered {
		out.InputTokens += item.Tokens
		if item.Doc.IsGold && !out.GoldFound {
			out.GoldFound = true
			out.GoldPos = item.PositionRatio
			if edgeAttentionWeight(item.PositionRatio) >= r.threshold() {
				out.Answer = item.Doc.Metadata["answer"]
				if out.Answer == "" {
					out.Answer = item.Doc.Content
				}
			}
		}
	}
	if r.CostPer1KTokens > 0 {
		out.CostUSD = (float64(out.InputTokens) / 1000.0) * r.CostPer1KTokens
	}
	return out, nil
}

// PositionTrial evaluates answer accuracy when gold evidence starts at InputGoldPos.
type PositionTrial struct {
	ID             string        `json:"id"`
	TenantID       string        `json:"tenant_id"`
	Query          string        `json:"query"`
	ExpectedAnswer string        `json:"expected_answer"`
	InputGoldPos   float64       `json:"input_gold_pos"`
	Docs           []EvidenceDoc `json:"docs"`
}

// PositionSweepConfig configures a matched comparison across packing strategies.
type PositionSweepConfig struct {
	Strategies    []PackStrategy `json:"strategies"`
	TokenBudget   int            `json:"token_budget"`
	ReserveTokens int            `json:"reserve_tokens,omitempty"`
	Deduplicate   bool           `json:"deduplicate"`
}

// StrategySweepMetrics summarizes accuracy by position bucket and token efficiency.
type StrategySweepMetrics struct {
	Strategy           PackStrategy `json:"strategy"`
	TotalTrials        int          `json:"total_trials"`
	CorrectTrials      int          `json:"correct_trials"`
	OverallAccuracy    float64      `json:"overall_accuracy"`
	StartAccuracy      float64      `json:"start_accuracy"`
	MiddleAccuracy     float64      `json:"middle_accuracy"`
	EndAccuracy        float64      `json:"end_accuracy"`
	MiddleGap          float64      `json:"middle_degradation_gap"`
	AvgInputTokens     float64      `json:"avg_input_tokens"`
	TokensPerCorrect   float64      `json:"tokens_per_correct_answer"`
	TotalCostUSD       float64      `json:"total_cost_usd"`
	MeanEdgeSalience   float64      `json:"mean_edge_salience_score"`
	BudgetExceededRuns int          `json:"budget_exceeded_runs"`
}

// PositionSweepReport collects matched measurements across all evaluated strategies.
type PositionSweepReport struct {
	TokenBudget int                    `json:"token_budget"`
	Results     []StrategySweepMetrics `json:"results"`
}

// RunPositionSweep executes all trials identically across each configured PackStrategy.
func RunPositionSweep(ctx context.Context, cfg PositionSweepConfig, trials []PositionTrial, reader ContextReader) (*PositionSweepReport, error) {
	if len(trials) == 0 {
		return nil, errors.New("membench: trials cannot be empty")
	}
	if reader == nil {
		reader = UShapedAttentionReader{RecallThreshold: 0.55, CostPer1KTokens: 0.002}
	}
	strategies := cfg.Strategies
	if len(strategies) == 0 {
		strategies = []PackStrategy{
			StrategyChronological,
			StrategyMiddleBuried,
			StrategyRelevanceDesc,
			StrategyEdgeSalience,
		}
	}

	report := &PositionSweepReport{
		TokenBudget: cfg.TokenBudget,
		Results:     make([]StrategySweepMetrics, 0, len(strategies)),
	}

	availableBudget := cfg.TokenBudget - cfg.ReserveTokens
	if availableBudget < 0 {
		availableBudget = 0
	}

	for _, strategy := range strategies {
		metrics := StrategySweepMetrics{
			Strategy:    strategy,
			TotalTrials: len(trials),
		}
		var (
			startTotal, startCorrect int
			midTotal, midCorrect     int
			endTotal, endCorrect     int
			totalTokens              int
			edgeSalSum               float64
		)

		for _, trial := range trials {
			packRes, err := PackEvidence(trial.Docs, PackOptions{
				Strategy:      strategy,
				TokenBudget:   cfg.TokenBudget,
				ReserveTokens: cfg.ReserveTokens,
				TenantID:      trial.TenantID,
				Query:         trial.Query,
				Deduplicate:   cfg.Deduplicate,
			})
			if err != nil {
				return nil, fmt.Errorf("pack trial %s (%s): %w", trial.ID, strategy, err)
			}
			if packRes.TotalTokens > availableBudget {
				metrics.BudgetExceededRuns++
			}
			edgeSalSum += packRes.EdgeSalienceScore

			readOut, err := reader.ReadAndAnswer(ctx, trial.Query, packRes.Packed)
			if err != nil {
				return nil, fmt.Errorf("read trial %s (%s): %w", trial.ID, strategy, err)
			}
			totalTokens += readOut.InputTokens
			metrics.TotalCostUSD += readOut.CostUSD

			correct := normalizeAnswer(readOut.Answer) == normalizeAnswer(trial.ExpectedAnswer)
			if correct {
				metrics.CorrectTrials++
			}

			switch {
			case trial.InputGoldPos <= 0.2:
				startTotal++
				if correct {
					startCorrect++
				}
			case trial.InputGoldPos >= 0.8:
				endTotal++
				if correct {
					endCorrect++
				}
			default:
				midTotal++
				if correct {
					midCorrect++
				}
			}
		}

		metrics.OverallAccuracy = float64(metrics.CorrectTrials) / float64(metrics.TotalTrials)
		metrics.StartAccuracy = ratioInt(startCorrect, startTotal)
		metrics.MiddleAccuracy = ratioInt(midCorrect, midTotal)
		metrics.EndAccuracy = ratioInt(endCorrect, endTotal)
		edgeAcc := (metrics.StartAccuracy + metrics.EndAccuracy) / 2.0
		metrics.MiddleGap = edgeAcc - metrics.MiddleAccuracy
		metrics.AvgInputTokens = float64(totalTokens) / float64(metrics.TotalTrials)
		metrics.MeanEdgeSalience = edgeSalSum / float64(metrics.TotalTrials)
		if metrics.CorrectTrials > 0 {
			metrics.TokensPerCorrect = float64(totalTokens) / float64(metrics.CorrectTrials)
		}
		report.Results = append(report.Results, metrics)
	}

	return report, nil
}

// FormatPositionSweepMarkdown formats a reproducible position-sweep comparison table.
func FormatPositionSweepMarkdown(report *PositionSweepReport) string {
	if report == nil {
		return ""
	}
	var b strings.Builder
	_, _ = fmt.Fprintf(&b, "### Lost-in-the-Middle Position Sweep (Budget: %d tokens)\n\n", report.TokenBudget)
	b.WriteString("| Strategy | Overall | Start | Middle | End | Mid Gap | Avg Tokens | Tokens / Correct | Cost (USD) |\n")
	b.WriteString("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
	for _, r := range report.Results {
		_, _ = fmt.Fprintf(&b, "| `%s` | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %+.1f%% | %.1f | %.1f | $%.5f |\n",
			r.Strategy,
			r.OverallAccuracy*100,
			r.StartAccuracy*100,
			r.MiddleAccuracy*100,
			r.EndAccuracy*100,
			r.MiddleGap*100,
			r.AvgInputTokens,
			r.TokensPerCorrect,
			r.TotalCostUSD,
		)
	}
	return b.String()
}

// DefaultLostInTheMiddleTrials constructs multi-document trials sweeping gold evidence
// across start (0%), early-mid (25%), middle (50%), late-mid (75%), and end (100%) positions.
func DefaultLostInTheMiddleTrials(numDocs, tokensPerDoc int) []PositionTrial {
	if numDocs < 5 {
		numDocs = 5
	}
	if tokensPerDoc <= 0 {
		tokensPerDoc = 100
	}
	positions := []float64{0.0, 0.25, 0.50, 0.75, 1.0}
	baseTime := time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC)

	trials := make([]PositionTrial, 0, len(positions))
	for pIdx, pos := range positions {
		goldIdx := int(pos * float64(numDocs-1))
		docs := make([]EvidenceDoc, 0, numDocs)
		expected := fmt.Sprintf("sentinel-answer-%d", pIdx)
		for i := 0; i < numDocs; i++ {
			ts := baseTime.Add(time.Duration(i) * time.Minute)
			if i == goldIdx {
				docs = append(docs, EvidenceDoc{
					ID:         fmt.Sprintf("gold-%d", i),
					TenantID:   "tenant-a",
					Content:    fmt.Sprintf("Primary service failover runbook key=%s quorum=3.", expected),
					Salience:   0.95,
					Timestamp:  ts,
					TokenCount: tokensPerDoc,
					IsGold:     true,
					Metadata:   map[string]string{"answer": expected},
				})
			} else {
				docs = append(docs, EvidenceDoc{
					ID:         fmt.Sprintf("dist-%d", i),
					TenantID:   "tenant-a",
					Content:    fmt.Sprintf("Unrelated historical build log artifact %d.", i),
					Salience:   0.20 + 0.01*float64(i%5),
					Timestamp:  ts,
					TokenCount: tokensPerDoc,
					IsGold:     false,
				})
			}
		}
		trials = append(trials, PositionTrial{
			ID:             fmt.Sprintf("pos-%02d", int(pos*100)),
			TenantID:       "tenant-a",
			Query:          "What is the primary service failover runbook key?",
			ExpectedAnswer: expected,
			InputGoldPos:   pos,
			Docs:           docs,
		})
	}
	return trials
}

func ratioInt(num, den int) float64 {
	if den == 0 {
		return 0
	}
	return float64(num) / float64(den)
}
