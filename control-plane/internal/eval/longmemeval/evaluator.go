package longmemeval

import (
	"context"
	"errors"
	"fmt"
	"math"
	"runtime"
	"sort"
	"strings"
	"time"
)

// BenchmarkConfig controls a reproducible LongMemEval benchmark execution.
type BenchmarkConfig struct {
	Seed              int64
	Categories        []TaskCategory
	Horizons          []int
	ParetoThresholds  []float64
	IncludeOutcomes   bool
	IncludeNegRegimes bool
	Policies          []MemoryPolicy
	Factory           StoreFactory
}

// DefaultHorizons defines the distractor session scales for Negative Regime 2.
var DefaultHorizons = []int{5, 25, 50, 100, 200}

// DefaultParetoThresholds defines the Kai evidence thresholds for Negative Regime 3.
var DefaultParetoThresholds = []float64{0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90}

// EvaluateSuite evaluates each MemoryPolicy across identical EvalCases using
// isolated canonical MemoryStore instances produced by factory.
func EvaluateSuite(ctx context.Context, policies []MemoryPolicy, cases []EvalCase, factory StoreFactory, includeOutcomes bool) ([]PolicyMetrics, error) {
	if len(policies) == 0 {
		return nil, errors.New("longmemeval: at least one policy is required")
	}
	if len(cases) == 0 {
		return nil, errors.New("longmemeval: evaluation cases cannot be empty")
	}
	if factory == nil {
		factory = func() (MemoryStore, func(), error) {
			return NewInMemoryStore(), nil, nil
		}
	}

	results := make([]PolicyMetrics, 0, len(policies))
	for _, pol := range policies {
		pm := PolicyMetrics{
			PolicyName:  pol.Name(),
			Description: pol.Description(),
			TotalCases:  len(cases),
			ByCategory:  make(map[TaskCategory]CategoryMetrics, len(AllCategories)),
		}
		for _, cat := range AllCategories {
			pm.ByCategory[cat] = CategoryMetrics{}
		}
		if includeOutcomes {
			pm.Outcomes = make([]CaseOutcome, 0, len(cases))
		}

		var (
			totalTurns       int
			totalTokens      int
			totalLatUS       int64
			latencies        = make([]int64, 0, len(cases))
			staleErrors      int
			unansExpected    int
			unansFP          int
			validExpected    int
			overAbstentions  int
			abstainPredicted int
			abstainTP        int
			contraExpected   int
			contraDetected   int
		)

		for _, tc := range cases {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			store, cleanup, err := factory()
			if err != nil {
				return nil, fmt.Errorf("create store for %s/%s: %w", pol.Name(), tc.ID, err)
			}

			runOne := func() error {
				if cleanup != nil {
					defer cleanup()
				}
				for _, turn := range tc.Turns {
					totalTurns++
					stats, err := pol.Ingest(ctx, store, turn)
					if err != nil {
						return fmt.Errorf("policy %s ingest %s/%s: %w", pol.Name(), tc.ID, turn.TurnID, err)
					}
					pm.TotalWrites += stats.TotalWrites()
					pm.TotalSkippedNoops += stats.SkippedNoops
				}

				start := time.Now()
				dec, err := pol.Recall(ctx, store, tc.Query)
				latUS := time.Since(start).Microseconds()
				if err != nil {
					return fmt.Errorf("policy %s recall %s: %w", pol.Name(), tc.ID, err)
				}
				totalLatUS += latUS
				latencies = append(latencies, latUS)
				totalTokens += dec.RetrievalTokens
				pm.GenerativeTokensTotal += dec.GenerativeOutputTokens

				// Audit cross-tenant leakage across every surfaced evidence hit.
				leaked := false
				for _, evTenant := range dec.EvidenceTenants {
					if evTenant != "" && tc.Query.TenantID != "" && evTenant != tc.Query.TenantID {
						pm.CrossTenantLeaks++
						leaked = true
					}
				}

				cm := pm.ByCategory[tc.Category]
				cm.Total++

				if tc.ShouldAbstain {
					unansExpected++
				} else {
					validExpected++
				}
				if dec.Abstained {
					abstainPredicted++
				}

				if tc.ExpectContradiction {
					contraExpected++
					cm.ContradictionsExpected++
					if dec.ContradictionDetected {
						contraDetected++
						cm.ContradictionsDetected++
					}
				}

				passed := false
				staleErr := false
				if tc.ShouldAbstain {
					if dec.Abstained {
						abstainTP++
						passed = true
					} else {
						unansFP++
						cm.FalseMemories++
					}
				} else {
					if dec.Abstained {
						overAbstentions++
						cm.OverAbstentions++
					} else if len(tc.ExpectedOrder) > 0 {
						passed = equalNormalizedSlice(dec.OrderedAnswers, tc.ExpectedOrder)
					} else if len(tc.ExpectedSet) > 0 {
						passed = equalNormalizedSet(dec.AggregatedValues, tc.ExpectedSet)
					} else {
						passed = normalizeAnswer(dec.Answer) == normalizeAnswer(tc.ExpectedAnswer)
					}
				}

				if !dec.Abstained {
					for _, bad := range tc.ForbiddenStale {
						if normalizeAnswer(dec.Answer) == normalizeAnswer(bad) || containsNormalized(dec.AggregatedValues, bad) {
							staleErr = true
							staleErrors++
							cm.StaleErrors++
							break
						}
					}
				}

				if passed && !leaked {
					pm.CorrectCases++
					cm.Correct++
				}
				pm.ByCategory[tc.Category] = cm

				if includeOutcomes {
					expStr := tc.ExpectedAnswer
					if tc.ShouldAbstain {
						expStr = "INSUFFICIENT_EVIDENCE"
					} else if len(tc.ExpectedOrder) > 0 {
						expStr = strings.Join(tc.ExpectedOrder, " -> ")
					} else if len(tc.ExpectedSet) > 0 {
						expStr = strings.Join(tc.ExpectedSet, ", ")
					}
					pm.Outcomes = append(pm.Outcomes, CaseOutcome{
						CaseID:                tc.ID,
						Category:              tc.Category,
						Subcategory:           tc.Subcategory,
						Passed:                passed && !leaked,
						Abstained:             dec.Abstained,
						ShouldAbstain:         tc.ShouldAbstain,
						StaleError:            staleErr,
						ContradictionDetected: dec.ContradictionDetected,
						CrossTenantLeak:       leaked,
						PredictedAnswer:       dec.Answer,
						ExpectedAnswer:        expStr,
						EvidenceScore:         round4(dec.EvidenceScore),
						RetrievalTokens:       dec.RetrievalTokens,
						LatencyUS:             latUS,
					})
				}
				return nil
			}

			if err := runOne(); err != nil {
				return nil, err
			}
		}

		for cat, cm := range pm.ByCategory {
			if cm.Total > 0 {
				cm.Accuracy = round4(float64(cm.Correct) / float64(cm.Total))
				pm.ByCategory[cat] = cm
			}
		}
		pm.OverallAccuracy = round4(float64(pm.CorrectCases) / float64(pm.TotalCases))
		pm.StaleFactErrorRate = round4(float64(staleErrors) / float64(pm.TotalCases))
		if contraExpected > 0 {
			pm.ContradictionDetRate = round4(float64(contraDetected) / float64(contraExpected))
		}
		if unansExpected > 0 {
			pm.AbstentionRecall = round4(float64(abstainTP) / float64(unansExpected))
			pm.UnanswerableFPR = round4(float64(unansFP) / float64(unansExpected))
		}
		if abstainPredicted > 0 {
			pm.AbstentionPrecision = round4(float64(abstainTP) / float64(abstainPredicted))
		}
		if pm.AbstentionPrecision+pm.AbstentionRecall > 0 {
			pm.AbstentionF1 = round4(2 * pm.AbstentionPrecision * pm.AbstentionRecall / (pm.AbstentionPrecision + pm.AbstentionRecall))
		}
		if validExpected > 0 {
			pm.OverAbstentionRate = round4(float64(overAbstentions) / float64(validExpected))
		}
		if totalTurns > 0 {
			pm.WriteAmplification = round4(float64(pm.TotalWrites) / float64(totalTurns))
		}
		pm.AvgRecallTokens = round2(float64(totalTokens) / float64(pm.TotalCases))
		pm.MeanLatencyUS = round2(float64(totalLatUS) / float64(pm.TotalCases))
		pm.P50LatencyUS = percentileInt64(latencies, 0.50)
		pm.P95LatencyUS = percentileInt64(latencies, 0.95)

		results = append(results, pm)
	}
	return results, nil
}

// RunHistoricalRegime executes Negative Regime 1: recency-bias failure on
// historical point-in-time queries ("what was X BEFORE the migration on day 4?").
func RunHistoricalRegime(ctx context.Context, seed int64, factory StoreFactory) (*HistoricalRegimeResult, error) {
	cases := BuildHistoricalQuerySuite(seed)
	emb := NewHashEmbedder(256)
	policies := []MemoryPolicy{
		FlatKVLatestPolicy{},
		HierarchicalScopeKVPolicy{},
		VectorLexicalRecencyPolicy{Embedder: emb, TopK: 8, RecencyWeight: 0.25, MinScore: 0.0},
		AtomicReflectivePolicy{MinConfidence: 0.55, Embedder: emb},
		KaiEvidenceGatedPolicy{MinConfidence: 0.55, MinEvidenceScore: 0.35, DisableHistoricalAsOf: true, Embedder: emb, CustomName: "kai-evidence-gated (no-asof ablation)"},
		KaiEvidenceGatedPolicy{MinConfidence: 0.55, MinEvidenceScore: 0.35, DisableHistoricalAsOf: false, Embedder: emb, CustomName: "kai-evidence-gated (asof time-travel)"},
	}

	metrics, err := EvaluateSuite(ctx, policies, cases, factory, true)
	if err != nil {
		return nil, err
	}

	rows := make([]HistoricalPolicyRow, 0, len(metrics))
	for _, m := range metrics {
		recencyFails := 0
		abstained := 0
		for _, o := range m.Outcomes {
			if o.StaleError {
				recencyFails++
			}
			if o.Abstained {
				abstained++
			}
		}
		rows = append(rows, HistoricalPolicyRow{
			PolicyName:          m.PolicyName,
			Accuracy:            m.OverallAccuracy,
			RecencyBiasRate:     round4(float64(recencyFails) / float64(m.TotalCases)),
			AbstentionRate:      round4(float64(abstained) / float64(m.TotalCases)),
			CorrectCases:        m.CorrectCases,
			RecencyBiasFailures: recencyFails,
			TotalCases:          m.TotalCases,
		})
	}
	return &HistoricalRegimeResult{
		RegimeName: "Recency-Bias Failure on Historical Point-in-Time Queries",
		CaseCount:  len(cases),
		Policies:   rows,
	}, nil
}

// RunLongHorizonRegime executes Negative Regime 2: high-distractor long-horizon
// scaling across 5..200 distractor sessions on semantic needle & abstention probes.
func RunLongHorizonRegime(ctx context.Context, seed int64, horizons []int, factory StoreFactory) (*LongHorizonRegimeResult, error) {
	if len(horizons) == 0 {
		horizons = DefaultHorizons
	}
	emb := NewHashEmbedder(256)
	policies := []MemoryPolicy{
		VectorLexicalRecencyPolicy{Embedder: emb, TopK: 5, RecencyWeight: 0.25, MinScore: 0.0, CustomName: "vector-lexical-recency (unthresholded)"},
		VectorLexicalRecencyPolicy{Embedder: emb, TopK: 5, RecencyWeight: 0.25, MinScore: 0.35, CoordinatedDelete: true, CustomName: "vector-lexical-recency (min_score=0.35)"},
		KaiEvidenceGatedPolicy{MinConfidence: 0.55, MinEvidenceScore: 0.35, TopK: 5, Embedder: emb, CustomName: "kai-evidence-gated (semantic, theta=0.35)"},
	}

	points := make([]HorizonScalingPoint, 0, len(horizons)*len(policies))
	for _, h := range horizons {
		cases := BuildLongHorizonSuite(seed, h)
		metrics, err := EvaluateSuite(ctx, policies, cases, factory, false)
		if err != nil {
			return nil, err
		}
		for _, m := range metrics {
			needleAcc := m.ByCategory[CategoryCrossSession].Accuracy
			points = append(points, HorizonScalingPoint{
				DistractorSessions: h,
				PolicyName:         m.PolicyName,
				NeedleRecall:       needleAcc,
				UnanswerableFPR:    m.UnanswerableFPR,
				AbstentionF1:       m.AbstentionF1,
				OverallAccuracy:    m.OverallAccuracy,
				AvgRecallTokens:    m.AvgRecallTokens,
				MeanLatencyUS:      m.MeanLatencyUS,
				CrossTenantLeaks:   m.CrossTenantLeaks,
			})
		}
	}

	return &LongHorizonRegimeResult{
		RegimeName: "High-Distractor Long-Horizon Scaling (5 to 200 Sessions)",
		Horizons:   append([]int(nil), horizons...),
		Points:     points,
	}, nil
}

// RunAbstentionParetoRegime executes Negative Regime 3: sweeping Kai's evidence
// threshold theta on mixed paraphrased valid queries and close-distractor
// unanswerable probes to expose the exact recall vs FPR Pareto tradeoff.
func RunAbstentionParetoRegime(ctx context.Context, seed int64, thresholds []float64, factory StoreFactory) (*AbstentionParetoResult, error) {
	if len(thresholds) == 0 {
		thresholds = DefaultParetoThresholds
	}
	cases := BuildParaphraseAbstentionSuite(seed)
	emb := NewHashEmbedder(256)

	validCount := 0
	unansCount := 0
	for _, tc := range cases {
		if tc.ShouldAbstain {
			unansCount++
		} else {
			validCount++
		}
	}

	points := make([]ParetoPoint, 0, len(thresholds))
	bestTheta := thresholds[0]
	bestAcc := -1.0

	for _, th := range thresholds {
		pol := KaiEvidenceGatedPolicy{
			MinConfidence:    0.55,
			MinEvidenceScore: th,
			TopK:             8,
			Embedder:         emb,
			CustomName:       fmt.Sprintf("kai-evidence-gated@%.2f", th),
		}
		metrics, err := EvaluateSuite(ctx, []MemoryPolicy{pol}, cases, factory, false)
		if err != nil {
			return nil, err
		}
		m := metrics[0]
		paraRecall := m.ByCategory[CategoryCrossSession].Accuracy
		pt := ParetoPoint{
			Threshold:          round4(th),
			ParaphraseRecall:   paraRecall,
			UnanswerableFPR:    m.UnanswerableFPR,
			OverAbstentionRate: m.OverAbstentionRate,
			AbstentionPrec:     m.AbstentionPrecision,
			AbstentionRecall:   m.AbstentionRecall,
			AbstentionF1:       m.AbstentionF1,
			OverallAccuracy:    m.OverallAccuracy,
			AvgRecallTokens:    m.AvgRecallTokens,
		}
		points = append(points, pt)
		if pt.OverallAccuracy > bestAcc {
			bestAcc = pt.OverallAccuracy
			bestTheta = pt.Threshold
		}
	}

	return &AbstentionParetoResult{
		RegimeName:       "Abstention Threshold Over-Conservatism (Paraphrase Recall vs Unanswerable FPR Pareto)",
		PolicyName:       "kai-evidence-gated",
		ValidCases:       validCount,
		UnanswerableCase: unansCount,
		BestThreshold:    bestTheta,
		BestAccuracy:     bestAcc,
		Points:           points,
	}, nil
}

// RunFullBenchmark executes the complete reproducible HIP-1346 LongMemEval
// benchmark suite including all 5 categories and all 3 negative regimes.
func RunFullBenchmark(ctx context.Context, cfg BenchmarkConfig) (*BenchmarkReport, error) {
	seed := cfg.Seed
	if seed == 0 {
		seed = DefaultSeed
	}
	policies := cfg.Policies
	if len(policies) == 0 {
		policies = DefaultPolicyMatrix()
	}
	allCases := BuildDefaultSuite(seed)
	cases := FilterCasesByCategory(allCases, cfg.Categories)
	if len(cases) == 0 {
		return nil, errors.New("longmemeval: no cases matched the requested category filter")
	}

	policyResults, err := EvaluateSuite(ctx, policies, cases, cfg.Factory, cfg.IncludeOutcomes)
	if err != nil {
		return nil, err
	}

	cats := cfg.Categories
	if len(cats) == 0 {
		cats = AllCategories
	}

	report := &BenchmarkReport{
		SchemaVersion:      "1.0.0",
		HIP:                "HIP-1346",
		Issue:              "hanzoai/agents#82",
		Seed:               seed,
		GoVersion:          runtime.Version(),
		GOOS:               runtime.GOOS,
		GOARCH:             runtime.GOARCH,
		NumCPU:             runtime.NumCPU(),
		DatasetCases:       len(cases),
		Categories:         cats,
		PolicyResults:      policyResults,
		CanonicalStackNote: "Evaluates existing hanzoai/agents memory policies (#62-#66) over canonical storage.StorageProvider KV/vector/event primitives with zero competing memory engine; stack boundaries intact (vfs -> hanzoai/s3 -> visor, Cloud lifecycle/billing, Enso routing, Zen generation, Kai non-generative decisions with OutputTokens=0).",
	}

	if cfg.IncludeNegRegimes {
		hist, err := RunHistoricalRegime(ctx, seed, cfg.Factory)
		if err != nil {
			return nil, fmt.Errorf("negative regime 1 (historical): %w", err)
		}
		report.HistoricalRegime = hist

		horizon, err := RunLongHorizonRegime(ctx, seed, cfg.Horizons, cfg.Factory)
		if err != nil {
			return nil, fmt.Errorf("negative regime 2 (long horizon): %w", err)
		}
		report.LongHorizonRegime = horizon

		pareto, err := RunAbstentionParetoRegime(ctx, seed, cfg.ParetoThresholds, cfg.Factory)
		if err != nil {
			return nil, fmt.Errorf("negative regime 3 (abstention pareto): %w", err)
		}
		report.AbstentionPareto = pareto
	}

	return report, nil
}

// RenderScoreTableMarkdown emits clean GitHub-Flavored Markdown score tables
// directly from measured BenchmarkReport numbers.
func RenderScoreTableMarkdown(rep *BenchmarkReport) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# HIP-1346 LongMemEval Score Table (`--seed %d`, %d Cases)\n\n", rep.Seed, rep.DatasetCases)

	b.WriteString("## 1. Overall Policy Comparison (5 LongMemEval Dimensions)\n\n")
	b.WriteString("| Policy | Overall | Update | Contradiction | Contra. Det. | Temporal | Abstention | Cross-Session | Stale Err | Unans. FPR | Abst. F1 | Tenant Leaks | Kai Gen Tok | Write Amp | Avg Tok |\n")
	b.WriteString("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
	for _, r := range rep.PolicyResults {
		fmt.Fprintf(&b, "| `%s` | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %d | %d | %.2fx | %.1f |\n",
			r.PolicyName,
			r.OverallAccuracy*100,
			r.ByCategory[CategoryKnowledgeUpdate].Accuracy*100,
			r.ByCategory[CategoryContradiction].Accuracy*100,
			r.ContradictionDetRate*100,
			r.ByCategory[CategoryTemporalOrdering].Accuracy*100,
			r.ByCategory[CategoryAbstention].Accuracy*100,
			r.ByCategory[CategoryCrossSession].Accuracy*100,
			r.StaleFactErrorRate*100,
			r.UnanswerableFPR*100,
			r.AbstentionF1*100,
			r.CrossTenantLeaks,
			r.GenerativeTokensTotal,
			r.WriteAmplification,
			r.AvgRecallTokens,
		)
	}

	if rep.HistoricalRegime != nil {
		b.WriteString("\n## 2. Negative Regime 1 — Recency-Bias Failure on Historical `AsOf` Queries (16 Cases)\n\n")
		b.WriteString("| Policy | Historical Accuracy | Recency-Bias Error Rate | Abstention Rate | Correct / Total |\n")
		b.WriteString("|---|---:|---:|---:|---:|\n")
		for _, row := range rep.HistoricalRegime.Policies {
			fmt.Fprintf(&b, "| `%s` | %.2f%% | %.2f%% | %.2f%% | %d / %d |\n",
				row.PolicyName,
				row.Accuracy*100,
				row.RecencyBiasRate*100,
				row.AbstentionRate*100,
				row.CorrectCases,
				row.TotalCases,
			)
		}
	}

	if rep.LongHorizonRegime != nil {
		b.WriteString("\n## 3. Negative Regime 2 — High-Distractor Long-Horizon Scaling (5 to 200 Sessions)\n\n")
		b.WriteString("| Distractor Sessions | Policy | Needle Recall | Unanswerable FPR | Abstention F1 | Overall Acc | Avg Recall Tok | Mean Latency (us) | Tenant Leaks |\n")
		b.WriteString("|---:|---|---:|---:|---:|---:|---:|---:|---:|\n")
		for _, pt := range rep.LongHorizonRegime.Points {
			fmt.Fprintf(&b, "| %d | `%s` | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %.1f | %.1f | %d |\n",
				pt.DistractorSessions,
				pt.PolicyName,
				pt.NeedleRecall*100,
				pt.UnanswerableFPR*100,
				pt.AbstentionF1*100,
				pt.OverallAccuracy*100,
				pt.AvgRecallTokens,
				pt.MeanLatencyUS,
				pt.CrossTenantLeaks,
			)
		}
	}

	if rep.AbstentionPareto != nil {
		b.WriteString("\n## 4. Negative Regime 3 — Abstention Threshold Pareto Tradeoff (`kai-evidence-gated`, 40 Semantic Cases)\n\n")
		b.WriteString("| Threshold (`theta`) | Paraphrase Recall | Over-Abstention Rate | Unanswerable FPR | Abstention Prec | Abstention Recall | Abstention F1 | Overall Acc |\n")
		b.WriteString("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
		for _, pt := range rep.AbstentionPareto.Points {
			fmt.Fprintf(&b, "| `%.2f` | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %.2f%% |\n",
				pt.Threshold,
				pt.ParaphraseRecall*100,
				pt.OverAbstentionRate*100,
				pt.UnanswerableFPR*100,
				pt.AbstentionPrec*100,
				pt.AbstentionRecall*100,
				pt.AbstentionF1*100,
				pt.OverallAccuracy*100,
			)
		}
	}

	return b.String()
}

func equalNormalizedSlice(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if normalizeAnswer(a[i]) != normalizeAnswer(b[i]) {
			return false
		}
	}
	return true
}

func equalNormalizedSet(actual, expected []string) bool {
	if len(expected) == 0 {
		return len(actual) == 0
	}
	normA := make(map[string]struct{}, len(actual))
	for _, v := range actual {
		if n := normalizeAnswer(v); n != "" {
			normA[n] = struct{}{}
		}
	}
	normE := make(map[string]struct{}, len(expected))
	for _, v := range expected {
		if n := normalizeAnswer(v); n != "" {
			normE[n] = struct{}{}
		}
	}
	if len(normA) != len(normE) {
		return false
	}
	for k := range normE {
		if _, ok := normA[k]; !ok {
			return false
		}
	}
	return true
}

func containsNormalized(slice []string, target string) bool {
	nt := normalizeAnswer(target)
	for _, s := range slice {
		if normalizeAnswer(s) == nt {
			return true
		}
	}
	return false
}

func round4(v float64) float64 {
	return math.Round(v*10000) / 10000
}

func round2(v float64) float64 {
	return math.Round(v*100) / 100
}

func percentileInt64(vals []int64, p float64) int64 {
	if len(vals) == 0 {
		return 0
	}
	cp := append([]int64(nil), vals...)
	sort.Slice(cp, func(i, j int) bool { return cp[i] < cp[j] })
	idx := int(math.Ceil(p*float64(len(cp)))) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= len(cp) {
		idx = len(cp) - 1
	}
	return cp[idx]
}
