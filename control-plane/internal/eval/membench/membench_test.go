package membench

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestLongMemEval_EvaluatesAllCategoriesWithoutCompetingStore(t *testing.T) {
	ctx := context.Background()
	cases := DefaultLongMemEvalSuite()

	policies := []MemoryPolicy{
		BaselineAppendPolicy{},
		AtomicFactPolicy{},
		TemporalBeliefPolicy{MinConfidence: 0.55},
	}

	reports, err := EvaluatePolicies(ctx, policies, cases, nil)
	require.NoError(t, err)
	require.Len(t, reports, 3)

	baseline := reports[0]
	atomicPol := reports[1]
	temporal := reports[2]

	// Baseline fails temporal update, contradiction, ordering, and abstention.
	require.Less(t, baseline.OverallAccuracy, 0.5)
	require.Greater(t, baseline.StaleFactErrorRate, 0.0)
	require.Greater(t, baseline.FalseMemoryRate, 0.0)

	// AtomicFactPolicy handles updates and abstention, while missing multi-step ordering / reflection separation.
	require.Equal(t, 1.0, atomicPol.ByCategory[CategoryTemporalUpdate].Accuracy)
	require.Equal(t, 1.0, atomicPol.ByCategory[CategoryAbstention].Accuracy)
	require.Equal(t, 0, atomicPol.CrossTenantLeaks)

	// TemporalBeliefPolicy solves all five dimensions with zero cross-tenant leaks.
	require.Equal(t, 1.0, temporal.OverallAccuracy)
	require.Equal(t, 1.0, temporal.ByCategory[CategoryTemporalUpdate].Accuracy)
	require.Equal(t, 1.0, temporal.ByCategory[CategoryContradiction].Accuracy)
	require.Equal(t, 1.0, temporal.ByCategory[CategoryTemporalOrdering].Accuracy)
	require.Equal(t, 1.0, temporal.ByCategory[CategoryAbstention].Accuracy)
	require.Equal(t, 1.0, temporal.ByCategory[CategoryCrossSession].Accuracy)
	require.Zero(t, temporal.StaleFactErrorRate)
	require.Zero(t, temporal.FalseMemoryRate)
	require.Zero(t, temporal.CrossTenantLeaks)

	md := FormatLongMemEvalMarkdown(reports)
	require.Contains(t, md, "temporal-belief-calibrated")
	require.Contains(t, md, "baseline-append")
}

func TestPackEvidence_EnforcesBudgetTenantACLAndEdgePlacement(t *testing.T) {
	now := time.Date(2026, 9, 10, 0, 0, 0, 0, time.UTC)
	docs := []EvidenceDoc{
		{ID: "d-low-1", TenantID: "t1", Content: "minor background note alpha", Salience: 0.20, TokenCount: 50, Timestamp: now},
		{ID: "d-top-1", TenantID: "t1", Content: "critical root cause patch note", Salience: 0.98, TokenCount: 80, Timestamp: now.Add(time.Minute), IsGold: true},
		{ID: "d-top-1-dup", TenantID: "t1", Content: "Critical root cause patch note", Salience: 0.70, TokenCount: 80, Timestamp: now.Add(2 * time.Minute)},
		{ID: "d-tenant-leak", TenantID: "t2", Content: "cross tenant secret must never surface", Salience: 1.00, TokenCount: 40, Timestamp: now.Add(3 * time.Minute)},
		{ID: "d-top-2", TenantID: "t1", Content: "second critical evidence verification", Salience: 0.88, TokenCount: 70, Timestamp: now.Add(4 * time.Minute)},
		{ID: "d-mid-1", TenantID: "t1", Content: "supporting config observation", Salience: 0.60, TokenCount: 60, Timestamp: now.Add(5 * time.Minute)},
		{ID: "d-low-2", TenantID: "t1", Content: "peripheral build output", Salience: 0.30, TokenCount: 50, Timestamp: now.Add(6 * time.Minute)},
		{ID: "d-too-big", TenantID: "t1", Content: "huge transcript exceeding budget", Salience: 0.85, TokenCount: 500, Timestamp: now.Add(7 * time.Minute)},
	}

	res, err := PackEvidence(docs, PackOptions{
		Strategy:      StrategyEdgeSalience,
		TokenBudget:   350,
		ReserveTokens: 50, // available = 300
		TenantID:      "t1",
		Query:         "critical root cause evidence",
		Deduplicate:   true,
	})
	require.NoError(t, err)
	require.Equal(t, 1, res.TenantFilteredCount)
	require.Equal(t, 1, res.DeduplicatedCount)
	require.LessOrEqual(t, res.TotalTokens, 300)
	require.Equal(t, 300-res.TotalTokens, res.BudgetRemaining)

	// Verify tenant-2 doc is excluded and huge doc is dropped without breaking budget.
	for _, p := range res.Packed {
		require.Equal(t, "t1", p.Doc.TenantID)
		require.NotEqual(t, "d-too-big", p.Doc.ID)
	}

	// Highest salience docs must sit at front (0) and back (n-1) edges.
	require.GreaterOrEqual(t, len(res.Packed), 4)
	require.Equal(t, "d-top-1", res.Packed[0].Doc.ID)
	require.Equal(t, "d-top-2", res.Packed[len(res.Packed)-1].Doc.ID)

	// Compare EdgeSalienceScore against StrategyMiddleBuried and StrategyRelevanceDesc.
	midRes, err := PackEvidence(docs, PackOptions{
		Strategy:      StrategyMiddleBuried,
		TokenBudget:   350,
		ReserveTokens: 50,
		TenantID:      "t1",
		Query:         "critical root cause evidence",
		Deduplicate:   true,
	})
	require.NoError(t, err)
	require.Greater(t, res.EdgeSalienceScore, midRes.EdgeSalienceScore)
	require.Less(t, res.MiddleRiskScore, midRes.MiddleRiskScore)
}

func TestPackEvidence_MultiEvidenceEdgeSalienceBeatsRelevanceDesc(t *testing.T) {
	now := time.Date(2026, 9, 10, 0, 0, 0, 0, time.UTC)
	docs := make([]EvidenceDoc, 0, 9)
	for i := 0; i < 4; i++ {
		docs = append(docs, EvidenceDoc{
			ID:         fmt.Sprintf("gold-%d", i),
			TenantID:   "t1",
			Content:    fmt.Sprintf("critical evidence span %d", i),
			Salience:   0.95 - 0.02*float64(i),
			TokenCount: 50,
			IsGold:     true,
			Timestamp:  now.Add(time.Duration(i) * time.Minute),
		})
	}
	for i := 0; i < 5; i++ {
		docs = append(docs, EvidenceDoc{
			ID:         fmt.Sprintf("dist-%d", i),
			TenantID:   "t1",
			Content:    fmt.Sprintf("background log entry %d", i),
			Salience:   0.20 + 0.01*float64(i),
			TokenCount: 50,
			IsGold:     false,
			Timestamp:  now.Add(time.Duration(i+10) * time.Minute),
		})
	}

	edgeRes, err := PackEvidence(docs, PackOptions{
		Strategy:    StrategyEdgeSalience,
		TokenBudget: 450,
		TenantID:    "t1",
	})
	require.NoError(t, err)

	relRes, err := PackEvidence(docs, PackOptions{
		Strategy:    StrategyRelevanceDesc,
		TokenBudget: 450,
		TenantID:    "t1",
	})
	require.NoError(t, err)

	require.Greater(t, edgeRes.EdgeSalienceScore, relRes.EdgeSalienceScore)
	for _, p := range edgeRes.Packed {
		if p.Doc.IsGold {
			require.GreaterOrEqual(t, edgeAttentionWeight(p.PositionRatio), 0.55,
				"gold doc %s placed at ratio %.3f fell into middle trough", p.Doc.ID, p.PositionRatio)
		}
	}
}

func TestRunPositionSweep_MitigatesLostInTheMiddleWithinTokenBudget(t *testing.T) {
	ctx := context.Background()
	trials := DefaultLostInTheMiddleTrials(21, 100) // 2,100 raw tokens per trial

	reader := UShapedAttentionReader{
		RecallThreshold: 0.55,
		CostPer1KTokens: 0.002,
	}

	report, err := RunPositionSweep(ctx, PositionSweepConfig{
		TokenBudget:   800, // Tighter than 2,100 raw tokens -> requires selective packing
		ReserveTokens: 100, // Available = 700 tokens (max 7 docs)
		Deduplicate:   true,
	}, trials, reader)
	require.NoError(t, err)
	require.Len(t, report.Results, 4)

	byStrat := make(map[PackStrategy]StrategySweepMetrics, len(report.Results))
	for _, m := range report.Results {
		byStrat[m.Strategy] = m
		require.Zero(t, m.BudgetExceededRuns, "strategy %s exceeded token budget", m.Strategy)
		require.LessOrEqual(t, m.AvgInputTokens, 700.0)
	}

	chrono := byStrat[StrategyChronological]
	buried := byStrat[StrategyMiddleBuried]
	edge := byStrat[StrategyEdgeSalience]

	// Middle-buried drops all middle trials; chronological drops late docs under tight budget.
	require.Equal(t, 0.0, buried.OverallAccuracy)
	require.Less(t, chrono.OverallAccuracy, 1.0)

	// Edge-salience places gold evidence on the primacy/recency edge inside the 700-token cap.
	require.Equal(t, 1.0, edge.OverallAccuracy)
	require.Equal(t, 1.0, edge.MiddleAccuracy)
	require.Equal(t, 0.0, edge.MiddleGap)
	require.LessOrEqual(t, edge.TokensPerCorrect, chrono.TokensPerCorrect)

	md := FormatPositionSweepMarkdown(report)
	require.Contains(t, md, "edge_salience")
	require.Contains(t, md, "middle_buried")
}

func TestMembench_ConcurrentSafety(t *testing.T) {
	ctx := context.Background()
	cases := DefaultLongMemEvalSuite()
	trials := DefaultLostInTheMiddleTrials(11, 80)

	var wg sync.WaitGroup
	errs := make(chan error, 8)
	for i := 0; i < 4; i++ {
		wg.Add(2)
		go func() {
			defer wg.Done()
			_, err := EvaluatePolicies(ctx, []MemoryPolicy{TemporalBeliefPolicy{}}, cases, nil)
			if err != nil {
				errs <- err
			}
		}()
		go func() {
			defer wg.Done()
			_, err := RunPositionSweep(ctx, PositionSweepConfig{TokenBudget: 500, ReserveTokens: 50}, trials, nil)
			if err != nil {
				errs <- err
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		require.NoError(t, err)
	}
}
