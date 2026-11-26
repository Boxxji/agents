package longmemeval

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/hanzoai/agents/control-plane/internal/storage"
	"github.com/hanzoai/agents/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

// Compile-time proof that Hanzo Agents' canonical storage.LocalStorage
// satisfies MemoryStore directly — zero competing memory engine introduced.
var _ MemoryStore = (*storage.LocalStorage)(nil)

func TestFullBenchmark_AllFiveCategoriesAndZeroLeaks(t *testing.T) {
	ctx := context.Background()
	rep, err := RunFullBenchmark(ctx, BenchmarkConfig{
		Seed:              DefaultSeed,
		IncludeOutcomes:   true,
		IncludeNegRegimes: false,
	})
	require.NoError(t, err)
	require.Equal(t, 60, rep.DatasetCases)
	require.Len(t, rep.PolicyResults, 6)

	byName := make(map[string]PolicyMetrics, len(rep.PolicyResults))
	for _, r := range rep.PolicyResults {
		byName[r.PolicyName] = r
		// Security & canonical Kai boundary invariants across EVERY policy.
		require.Equal(t, 0, r.CrossTenantLeaks, "policy %s leaked cross-tenant memory", r.PolicyName)
		require.Equal(t, 0, r.GenerativeTokensTotal, "policy %s violated Kai non-generative boundary", r.PolicyName)
		require.Equal(t, 60, r.TotalCases)
		for _, cat := range AllCategories {
			require.Equal(t, 12, r.ByCategory[cat].Total, "policy %s category %s expected 12 cases", r.PolicyName, cat)
		}
	}

	// Append-only fails updates, abstention, and cross-session retrieval.
	appendPol := byName["append-only-history"]
	require.Equal(t, 0.0, appendPol.ByCategory[CategoryKnowledgeUpdate].Accuracy)
	require.Equal(t, 0.0, appendPol.ByCategory[CategoryAbstention].Accuracy)
	require.Equal(t, 0.0, appendPol.ByCategory[CategoryCrossSession].Accuracy)

	// Hierarchical scope KV suffers cross-scope stale shadowing (4/12 updates shadowed).
	hierPol := byName["hierarchical-scope-kv"]
	require.Less(t, hierPol.ByCategory[CategoryKnowledgeUpdate].Accuracy, 0.75)
	require.Greater(t, hierPol.StaleFactErrorRate, 0.20)

	// Unthresholded vector-lexical-recency suffers high unanswerable FPR & stale errors.
	vecPol := byName["vector-lexical-recency"]
	require.GreaterOrEqual(t, vecPol.UnanswerableFPR, 0.50)
	require.Greater(t, vecPol.StaleFactErrorRate, 0.40)

	// Flat KV handles updates and abstention but fails temporal ordering & multi-session aggregation.
	flatPol := byName["flat-kv-latest"]
	require.Equal(t, 1.0, flatPol.ByCategory[CategoryKnowledgeUpdate].Accuracy)
	require.Equal(t, 1.0, flatPol.ByCategory[CategoryAbstention].Accuracy)
	require.Less(t, flatPol.ByCategory[CategoryTemporalOrdering].Accuracy, 0.25)
	require.Equal(t, 0.5, flatPol.ByCategory[CategoryCrossSession].Accuracy)

	// Atomic-reflective solves updates, contradictions, abstention, cross-session,
	// while dropping the 2 historical AsOf cases in temporal_ordering.
	atomicPol := byName["atomic-reflective"]
	require.Equal(t, 1.0, atomicPol.ByCategory[CategoryKnowledgeUpdate].Accuracy)
	require.Equal(t, 1.0, atomicPol.ByCategory[CategoryContradiction].Accuracy)
	require.Equal(t, 1.0, atomicPol.ContradictionDetRate)
	require.Less(t, atomicPol.ByCategory[CategoryTemporalOrdering].Accuracy, 1.0)
	require.Equal(t, 1.0, atomicPol.ByCategory[CategoryAbstention].Accuracy)
	require.Equal(t, 1.0, atomicPol.ByCategory[CategoryCrossSession].Accuracy)

	// Kai evidence-gated solves all 5 structured dimensions with 0 stale errors & 100% contradiction detection.
	kaiPol := byName["kai-evidence-gated"]
	require.Equal(t, 1.0, kaiPol.OverallAccuracy)
	require.Equal(t, 1.0, kaiPol.ContradictionDetRate)
	require.Equal(t, 0.0, kaiPol.StaleFactErrorRate)
	require.Equal(t, 0.0, kaiPol.UnanswerableFPR)
	require.Equal(t, 1.0, kaiPol.AbstentionF1)

	md := RenderScoreTableMarkdown(rep)
	require.Contains(t, md, "kai-evidence-gated")
	require.Contains(t, md, "hierarchical-scope-kv")
	require.Contains(t, md, "vector-lexical-recency")
}

func TestNegativeRegime1_HistoricalRecencyBias(t *testing.T) {
	ctx := context.Background()
	res, err := RunHistoricalRegime(ctx, DefaultSeed, nil)
	require.NoError(t, err)
	require.Equal(t, 16, res.CaseCount)

	byName := make(map[string]HistoricalPolicyRow, len(res.Policies))
	for _, r := range res.Policies {
		byName[r.PolicyName] = r
	}

	// All recency-default policies + no-asof ablation fail historical pre-migration queries.
	for _, name := range []string{
		"flat-kv-latest",
		"hierarchical-scope-kv",
		"vector-lexical-recency",
		"atomic-reflective",
		"kai-evidence-gated (no-asof ablation)",
	} {
		row, ok := byName[name]
		require.True(t, ok, "missing historical row for %s", name)
		require.Equal(t, 0.0, row.Accuracy, "expected 0%% accuracy for recency-biased policy %s", name)
		require.Equal(t, 1.0, row.RecencyBiasRate, "expected 100%% recency-bias error rate for %s", name)
	}

	// Enabling explicit AsOf time-travel over the canonical timeline recovers 100%.
	asofRow := byName["kai-evidence-gated (asof time-travel)"]
	require.Equal(t, 1.0, asofRow.Accuracy)
	require.Equal(t, 0.0, asofRow.RecencyBiasRate)
}

func TestNegativeRegime2_LongHorizonDistractorDegradation(t *testing.T) {
	ctx := context.Background()
	horizons := []int{5, 50, 200}
	res, err := RunLongHorizonRegime(ctx, DefaultSeed, horizons, nil)
	require.NoError(t, err)
	require.Len(t, res.Points, len(horizons)*3)

	lookup := func(h int, pol string) HorizonScalingPoint {
		for _, pt := range res.Points {
			if pt.DistractorSessions == h && pt.PolicyName == pol {
				return pt
			}
		}
		t.Fatalf("missing point for horizon=%d policy=%s", h, pol)
		return HorizonScalingPoint{}
	}

	// Unthresholded vector-lexical-recency degrades sharply in needle recall as distractors grow 5 -> 200.
	vec5 := lookup(5, "vector-lexical-recency (unthresholded)")
	vec200 := lookup(200, "vector-lexical-recency (unthresholded)")
	require.Greater(t, vec5.NeedleRecall, vec200.NeedleRecall)
	require.Equal(t, 1.0, vec5.UnanswerableFPR)
	require.Equal(t, 1.0, vec200.UnanswerableFPR)

	// Even with Kai static lexical threshold theta=0.35, Unanswerable FPR rises from 5 -> 200 sessions.
	kai5 := lookup(5, "kai-evidence-gated (semantic, theta=0.35)")
	kai200 := lookup(200, "kai-evidence-gated (semantic, theta=0.35)")
	require.Greater(t, kai200.UnanswerableFPR, kai5.UnanswerableFPR)
	require.Less(t, kai200.AbstentionF1, kai5.AbstentionF1)

	// Zero cross-tenant leaks at every horizon.
	for _, pt := range res.Points {
		require.Equal(t, 0, pt.CrossTenantLeaks)
	}
}

func TestNegativeRegime3_AbstentionParetoTradeoff(t *testing.T) {
	ctx := context.Background()
	res, err := RunAbstentionParetoRegime(ctx, DefaultSeed, DefaultParetoThresholds, nil)
	require.NoError(t, err)
	require.Equal(t, 20, res.ValidCases)
	require.Equal(t, 20, res.UnanswerableCase)
	require.Len(t, res.Points, len(DefaultParetoThresholds))

	first := res.Points[0]                // theta = 0.10
	last := res.Points[len(res.Points)-1] // theta = 0.90

	// Low threshold: high paraphrase recall, catastrophic unanswerable FPR.
	require.GreaterOrEqual(t, first.ParaphraseRecall, 0.70)
	require.GreaterOrEqual(t, first.UnanswerableFPR, 0.80)

	// High threshold: 0% unanswerable FPR at the cost of complete paraphrase recall collapse.
	require.Equal(t, 0.0, last.UnanswerableFPR)
	require.Less(t, last.ParaphraseRecall, first.ParaphraseRecall)
	require.Greater(t, last.OverAbstentionRate, first.OverAbstentionRate)

	// Verify monotonic non-increasing UnanswerableFPR and ParaphraseRecall as theta increases.
	for i := 1; i < len(res.Points); i++ {
		require.LessOrEqual(t, res.Points[i].UnanswerableFPR, res.Points[i-1].UnanswerableFPR)
		require.LessOrEqual(t, res.Points[i].ParaphraseRecall, res.Points[i-1].ParaphraseRecall)
		require.GreaterOrEqual(t, res.Points[i].OverAbstentionRate, res.Points[i-1].OverAbstentionRate)
	}
}

func TestMultiTenantAndACLRoleIsolation(t *testing.T) {
	ctx := context.Background()
	now := time.Date(2026, 10, 1, 12, 0, 0, 0, time.UTC)

	secretTurn := SessionTurn{
		TurnID:     "iso-1",
		TenantID:   "tenant-alpha",
		ActorID:    "shared-actor",
		SessionID:  "shared-session",
		WorkflowID: "shared-wf",
		Scope:      "actor",
		ACLRoles:   []string{"secops-admin"},
		Timestamp:  now,
		Entity:     "vault",
		Attribute:  "master_key_ref",
		Value:      "kms-secret-8821",
		Statement:  "Tenant alpha vault master_key_ref is kms-secret-8821.",
		Action:     ActionAdd,
		Kind:       KindObservation,
		Confidence: 0.99,
	}

	for _, pol := range DefaultPolicyMatrix() {
		store := NewInMemoryStore()
		_, err := pol.Ingest(ctx, store, secretTurn)
		require.NoError(t, err)

		// 1. Cross-tenant query with identical actor/session/workflow AND admin role MUST NOT leak.
		crossTenantDec, err := pol.Recall(ctx, store, RecallQuery{
			CaseID:       "iso-cross-tenant",
			Category:     CategoryAbstention,
			TenantID:     "tenant-beta",
			ActorID:      "shared-actor",
			SessionIDs:   []string{"shared-session"},
			WorkflowID:   "shared-wf",
			CallerRoles:  []string{"secops-admin"},
			Question:     "What is vault master_key_ref?",
			Entity:       "vault",
			Attribute:    "master_key_ref",
			QueryMode:    QueryStructured,
			TemporalMode: TemporalLatest,
			QueryTime:    now.Add(time.Hour),
		})
		require.NoError(t, err)
		require.NotEqual(t, "kms-secret-8821", crossTenantDec.Answer, "policy %s leaked cross-tenant secret", pol.Name())
		for _, evT := range crossTenantDec.EvidenceTenants {
			require.NotEqual(t, "tenant-alpha", evT, "policy %s surfaced tenant-alpha evidence to tenant-beta", pol.Name())
		}

		// 2. Same-tenant unauthorized role MUST NOT leak.
		unauthDec, err := pol.Recall(ctx, store, RecallQuery{
			CaseID:       "iso-unauth-role",
			Category:     CategoryAbstention,
			TenantID:     "tenant-alpha",
			ActorID:      "shared-actor",
			SessionIDs:   []string{"shared-session"},
			WorkflowID:   "shared-wf",
			CallerRoles:  []string{"guest-viewer"},
			Question:     "What is vault master_key_ref?",
			Entity:       "vault",
			Attribute:    "master_key_ref",
			QueryMode:    QueryStructured,
			TemporalMode: TemporalLatest,
			QueryTime:    now.Add(time.Hour),
		})
		require.NoError(t, err)
		require.NotEqual(t, "kms-secret-8821", unauthDec.Answer, "policy %s leaked ACL-restricted secret to guest-viewer", pol.Name())

		// 3. Same-tenant authorized role CAN retrieve (for policies supporting actor scope).
		if pol.Name() != "append-only-history" {
			authDec, err := pol.Recall(ctx, store, RecallQuery{
				CaseID:       "iso-auth-role",
				Category:     CategoryCrossSession,
				TenantID:     "tenant-alpha",
				ActorID:      "shared-actor",
				SessionIDs:   []string{"shared-session"},
				WorkflowID:   "shared-wf",
				CallerRoles:  []string{"secops-admin"},
				Question:     "What is vault master_key_ref?",
				Entity:       "vault",
				Attribute:    "master_key_ref",
				QueryMode:    QueryStructured,
				TemporalMode: TemporalLatest,
				QueryTime:    now.Add(time.Hour),
			})
			require.NoError(t, err)
			require.Equal(t, "kms-secret-8821", authDec.Answer, "policy %s failed authorized recall", pol.Name())
		}
	}
}

func TestDeterminism_BitForBitIdenticalSeededRuns(t *testing.T) {
	ctx := context.Background()
	cfg := BenchmarkConfig{
		Seed:              DefaultSeed,
		Horizons:          []int{5, 25},
		ParetoThresholds:  []float64{0.20, 0.50, 0.80},
		IncludeOutcomes:   true,
		IncludeNegRegimes: true,
	}

	r1, err := RunFullBenchmark(ctx, cfg)
	require.NoError(t, err)
	r2, err := RunFullBenchmark(ctx, cfg)
	require.NoError(t, err)

	stripTiming := func(r *BenchmarkReport) {
		for i := range r.PolicyResults {
			r.PolicyResults[i].MeanLatencyUS = 0
			r.PolicyResults[i].P50LatencyUS = 0
			r.PolicyResults[i].P95LatencyUS = 0
			for j := range r.PolicyResults[i].Outcomes {
				r.PolicyResults[i].Outcomes[j].LatencyUS = 0
			}
		}
		if r.LongHorizonRegime != nil {
			for i := range r.LongHorizonRegime.Points {
				r.LongHorizonRegime.Points[i].MeanLatencyUS = 0
			}
		}
	}
	stripTiming(r1)
	stripTiming(r2)

	b1, err := json.Marshal(r1)
	require.NoError(t, err)
	b2, err := json.Marshal(r2)
	require.NoError(t, err)
	require.Equal(t, string(b1), string(b2), "seeded benchmark output must be bit-for-bit deterministic")
}

func TestConcurrencySafety_RaceFreeMultiTenantSharedStore(t *testing.T) {
	ctx := context.Background()
	sharedStore := NewInMemoryStore()
	pol := KaiEvidenceGatedPolicy{MinConfidence: 0.55, MinEvidenceScore: 0.35, Embedder: NewHashEmbedder(256)}
	base := time.Date(2026, 11, 1, 9, 0, 0, 0, time.UTC)

	const workers = 16
	var wg sync.WaitGroup
	errs := make(chan error, workers*2)

	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			tenant := fmt.Sprintf("tenant-%d", id)
			actor := "shared-agent"
			expected := fmt.Sprintf("region-v3-%d", id)

			for step := 1; step <= 3; step++ {
				val := fmt.Sprintf("region-v%d-%d", step, id)
				_, err := pol.Ingest(ctx, sharedStore, SessionTurn{
					TurnID:     fmt.Sprintf("w%d-t%d", id, step),
					TenantID:   tenant,
					ActorID:    actor,
					SessionID:  fmt.Sprintf("s-%d", step),
					Scope:      "actor",
					Timestamp:  base.Add(time.Duration(step*12) * time.Hour),
					Entity:     "service",
					Attribute:  "region",
					Value:      val,
					Statement:  fmt.Sprintf("Tenant %d service region updated to %s.", id, val),
					Action:     ActionUpdate,
					Kind:       KindObservation,
					Confidence: 0.96,
				})
				if err != nil {
					errs <- err
					return
				}
			}

			dec, err := pol.Recall(ctx, sharedStore, RecallQuery{
				CaseID:       fmt.Sprintf("conc-%d", id),
				Category:     CategoryKnowledgeUpdate,
				TenantID:     tenant,
				ActorID:      actor,
				Question:     "What is the service region?",
				Entity:       "service",
				Attribute:    "region",
				QueryMode:    QueryStructured,
				TemporalMode: TemporalLatest,
				QueryTime:    base.Add(72 * time.Hour),
			})
			if err != nil {
				errs <- err
				return
			}
			if dec.Answer != expected {
				errs <- fmt.Errorf("worker %d got %q, want %q", id, dec.Answer, expected)
				return
			}
			for _, evT := range dec.EvidenceTenants {
				if evT != tenant {
					errs <- fmt.Errorf("worker %d cross-tenant leak: got %q", id, evT)
					return
				}
			}
		}(w)
	}

	wg.Wait()
	close(errs)
	for err := range errs {
		require.NoError(t, err)
	}

	// Verify event history recorded cleanly without race.
	events, err := sharedStore.GetEventHistory(ctx, types.EventFilter{})
	require.NoError(t, err)
	require.Equal(t, workers*3, len(events))
}

func TestCategoryFiltering_PreservesSeedInvariance(t *testing.T) {
	all := BuildDefaultSuite(DefaultSeed)
	onlyAbst := FilterCasesByCategory(all, []TaskCategory{CategoryAbstention})
	require.Len(t, onlyAbst, 12)
	for _, tc := range onlyAbst {
		require.Equal(t, CategoryAbstention, tc.Category)
	}

	ctx := context.Background()
	rep, err := RunFullBenchmark(ctx, BenchmarkConfig{
		Seed:              DefaultSeed,
		Categories:        []TaskCategory{CategoryAbstention, CategoryContradiction},
		IncludeNegRegimes: false,
	})
	require.NoError(t, err)
	require.Equal(t, 24, rep.DatasetCases)
}
