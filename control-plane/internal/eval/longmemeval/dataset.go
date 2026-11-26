package longmemeval

import (
	"fmt"
	"math/rand"
	"time"
)

// DefaultSeed is the canonical reproducible seed pinned to HIP-1346.
const DefaultSeed int64 = 1346

// BuildDefaultSuite generates the canonical 60-case clean-room Apache-2.0
// LongMemEval multi-session suite (12 cases per each of the 5 dimensions)
// deterministically from seed.
func BuildDefaultSuite(seed int64) []EvalCase {
	rng := rand.New(rand.NewSource(seed))
	baseTime := time.Date(2026, 6, 1, 9, 0, 0, 0, time.UTC)

	var cases []EvalCase
	cases = append(cases, buildKnowledgeUpdateCases(rng, baseTime)...)
	cases = append(cases, buildContradictionCases(rng, baseTime)...)
	cases = append(cases, buildTemporalOrderingCases(rng, baseTime)...)
	cases = append(cases, buildAbstentionCases(rng, baseTime)...)
	cases = append(cases, buildCrossSessionCases(rng, baseTime)...)
	return cases
}

// FilterCasesByCategory filters cases while preserving canonical order.
// An empty filter returns all cases unchanged.
func FilterCasesByCategory(cases []EvalCase, allowed []TaskCategory) []EvalCase {
	if len(allowed) == 0 {
		out := make([]EvalCase, len(cases))
		copy(out, cases)
		return out
	}
	set := make(map[TaskCategory]struct{}, len(allowed))
	for _, c := range allowed {
		set[c] = struct{}{}
	}
	out := make([]EvalCase, 0, len(cases))
	for _, tc := range cases {
		if _, ok := set[tc.Category]; ok {
			out = append(out, tc)
		}
	}
	return out
}

func buildKnowledgeUpdateCases(rng *rand.Rand, base time.Time) []EvalCase {
	type updateSpec struct {
		entity   string
		attr     string
		values   []string
		question string
		stmtFmt  string
		crossScp bool
	}
	specs := []updateSpec{
		{"api_gateway", "endpoint", []string{"https://v1.api.hanzo.internal", "https://v2.api.hanzo.internal", "https://edge-v3.api.hanzo.internal"}, "What is the current API gateway endpoint?", "API gateway endpoint migrated to %s.", false},
		{"billing_svc", "deploy_region", []string{"us-east-1", "eu-west-1", "eu-central-1"}, "Which region currently hosts billing_svc?", "Deployment region for billing_svc updated to %s.", false},
		{"search_index", "primary_db", []string{"sqlite-local", "postgres-15", "postgres-17-ha"}, "What primary database backs search_index now?", "search_index primary database switched to %s.", false},
		{"alert_router", "pager_channel", []string{"#ops-legacy", "#sre-oncall", "#sev1-warroom"}, "Which channel receives alert_router pages?", "Alert routing channel updated to %s.", false},
		{"enso_router", "default_model", []string{"zen-mini-v1", "zen-pro-v2", "zen-4-turbo"}, "Which default model is configured on enso_router?", "Default routing model set to %s.", false},
		{"visor_pool", "sandbox_image", []string{"ubuntu-22-base", "ubuntu-24-hardened", "ubuntu-24-fips"}, "Which sandbox image does visor_pool launch?", "Visor sandbox image upgraded to %s.", false},
		{"vfs_mount", "storage_tier", []string{"s3-standard", "s3-express-onezone", "s3-nvme-tier"}, "What storage tier backs vfs_mount?", "VFS storage tier migrated to %s.", false},
		{"ci_runner", "concurrency_cap", []string{"8", "16", "32"}, "What is the active concurrency cap for ci_runner?", "CI runner concurrency cap raised to %s.", false},
		// Cross-scope shadowing updates (older workflow/session vs newer actor update)
		{"checkout_api", "rate_limit_rps", []string{"120", "450"}, "What is the authoritative rate limit for checkout_api?", "Checkout API rate limit configured at %s rps.", true},
		{"auth_broker", "token_ttl", []string{"15m", "60m"}, "What is the current token TTL on auth_broker?", "Auth broker token TTL changed to %s.", true},
		{"metrics_pipe", "retention_days", []string{"14d", "90d"}, "What is the active retention window for metrics_pipe?", "Metrics pipeline retention updated to %s.", true},
		{"deploy_bot", "approval_mode", []string{"manual-single", "dual-control"}, "What approval mode does deploy_bot enforce?", "Deploy bot approval mode switched to %s.", true},
	}

	out := make([]EvalCase, 0, len(specs))
	for i, sp := range specs {
		tenant := fmt.Sprintf("tenant-%d", (i%3)+1)
		actor := fmt.Sprintf("ops-agent-%02d", i+1)
		wfID := fmt.Sprintf("wf-ku-%02d", i+1)
		tStart := base.Add(time.Duration(i*72+rng.Intn(6)) * time.Hour)

		var turns []SessionTurn
		var sessionIDs []string
		for step, val := range sp.values {
			sid := fmt.Sprintf("s-ku-%02d-%d", i+1, step+1)
			sessionIDs = append(sessionIDs, sid)
			act := ActionUpdate
			if step == 0 {
				act = ActionAdd
			}
			scope := "actor"
			if sp.crossScp && step == 0 {
				if i%2 == 0 {
					scope = "session"
				} else {
					scope = "workflow"
				}
			}
			turns = append(turns, SessionTurn{
				TurnID:     fmt.Sprintf("ku-%02d-t%d", i+1, step+1),
				TenantID:   tenant,
				ActorID:    actor,
				SessionID:  sid,
				WorkflowID: wfID,
				Scope:      scope,
				Timestamp:  tStart.Add(time.Duration(step*24) * time.Hour),
				Entity:     sp.entity,
				Attribute:  sp.attr,
				Value:      val,
				Statement:  fmt.Sprintf(sp.stmtFmt, val),
				Action:     act,
				Kind:       KindObservation,
				Confidence: 0.94 + 0.02*float64(step%3),
				SourceID:   "ops-cli",
			})
			// Inject a benign redundant NOOP / distractor turn in between sessions.
			if step == 0 {
				turns = append(turns, SessionTurn{
					TurnID:       fmt.Sprintf("ku-%02d-noop", i+1),
					TenantID:     tenant,
					ActorID:      actor,
					SessionID:    sid,
					WorkflowID:   wfID,
					Scope:        "actor",
					Timestamp:    tStart.Add(2 * time.Hour),
					Entity:       "heartbeat",
					Attribute:    "status",
					Value:        "healthy",
					Statement:    "Routine health check passed with zero alerts.",
					Action:       ActionNoop,
					Kind:         KindObservation,
					Confidence:   0.90,
					IsDistractor: true,
				})
			}
		}

		// Interleave another tenant's turn on the exact same entity/attribute to assert 0 leaks.
		otherTenant := fmt.Sprintf("tenant-%d", ((i+1)%3)+1)
		turns = append(turns, SessionTurn{
			TurnID:     fmt.Sprintf("ku-%02d-other-tenant", i+1),
			TenantID:   otherTenant,
			ActorID:    actor,
			SessionID:  sessionIDs[len(sessionIDs)-1],
			WorkflowID: wfID,
			Scope:      "actor",
			Timestamp:  tStart.Add(time.Duration(len(sp.values)*24+1) * time.Hour),
			Entity:     sp.entity,
			Attribute:  sp.attr,
			Value:      "leaked-cross-tenant-value",
			Statement:  fmt.Sprintf("Cross-tenant %s %s set to leaked-cross-tenant-value.", sp.entity, sp.attr),
			Action:     ActionAdd,
			Kind:       KindObservation,
			Confidence: 0.99,
		})

		latest := sp.values[len(sp.values)-1]
		stale := append([]string(nil), sp.values[:len(sp.values)-1]...)
		stale = append(stale, "leaked-cross-tenant-value")

		sub := "multi_session_update"
		if sp.crossScp {
			sub = "cross_scope_update"
		}
		out = append(out, EvalCase{
			ID:           fmt.Sprintf("lme-ku-%02d", i+1),
			Category:     CategoryKnowledgeUpdate,
			Subcategory:  sub,
			TenantID:     tenant,
			ActorID:      actor,
			SessionCount: len(sessionIDs),
			Turns:        turns,
			Query: RecallQuery{
				CaseID:       fmt.Sprintf("lme-ku-%02d", i+1),
				Category:     CategoryKnowledgeUpdate,
				TenantID:     tenant,
				ActorID:      actor,
				WorkflowID:   wfID,
				SessionIDs:   sessionIDs,
				Question:     sp.question,
				Entity:       sp.entity,
				Attribute:    sp.attr,
				QueryMode:    QueryStructured,
				TemporalMode: TemporalLatest,
				QueryTime:    tStart.Add(time.Duration(len(sp.values)*24+12) * time.Hour),
			},
			ExpectedAnswer: latest,
			ForbiddenStale: stale,
		})
	}
	return out
}

func buildContradictionCases(rng *rand.Rand, base time.Time) []EvalCase {
	type contraSpec struct {
		entity      string
		attr        string
		verifiedVal string
		conflictVal string
		obsStmt     string
		hypStmt     string
		question    string
		hypFirst    bool
		crossSource bool
	}
	specs := []contraSpec{
		{"incident_402", "root_cause", "connection_pool_exhaustion", "transient_dns_flake", "Verified telemetry confirmed connection_pool_exhaustion on DB shard 3.", "Speculation: maybe transient_dns_flake caused the latency spike.", "What is the verified root cause of incident_402?", false, false},
		{"checkout_db", "replica_lag", "420ms", "5ms", "Prometheus gauge directly observed replica_lag at 420ms.", "Guess in chat: replica_lag probably dropped to 5ms.", "What is the verified replica lag for checkout_db?", false, false},
		{"release_24", "rollback_status", "rolled_back", "still_canary", "Audit log confirmed release_24 was rolled_back at 14:10 UTC.", "Hypothesis: release_24 might be still_canary in staging.", "What is the verified rollback status of release_24?", false, false},
		{"kms_signer", "key_algorithm", "ed25519", "rsa-2048", "HSM attestation verified kms_signer key_algorithm is ed25519.", "Unverified rumor: kms_signer may still use rsa-2048.", "Which key algorithm does kms_signer actually use?", false, false},
		{"gpu_cluster", "driver_version", "560.35.03", "535.129.03", "nvidia-smi output recorded driver_version 560.35.03 across all nodes.", "Guess: cluster might be stuck on 535.129.03.", "What GPU driver version is verified on gpu_cluster?", false, false},
		{"edge_waf", "tls_policy", "tls1.3-strict", "tls1.2-compat", "TLS scanner verified edge_waf enforces tls1.3-strict.", "Assumption: edge_waf probably runs tls1.2-compat.", "What TLS policy is verified on edge_waf?", false, false},
		// Hypothesis stated first, then overturned by direct observation
		{"queue_broker", "bottleneck", "disk_fsync_iops", "cpu_throttling", "iostat direct measurement proved bottleneck is disk_fsync_iops.", "Initial hypothesis blamed cpu_throttling on broker-0.", "What is the confirmed bottleneck on queue_broker?", true, false},
		{"cache_tier", "eviction_policy", "allkeys-lfu", "volatile-lru", "Redis CONFIG GET confirmed eviction_policy is allkeys-lfu.", "Early conjecture assumed cache_tier used volatile-lru.", "What eviction policy is confirmed on cache_tier?", true, false},
		{"build_farm", "oom_culprit", "linker_lto", "test_fixture", "Kernel dmesg confirmed OOM kill target was linker_lto.", "Initial guess suspected test_fixture memory leak.", "What is the confirmed OOM culprit in build_farm?", true, false},
		// Cross-source conflicting observations (newer authoritative source supersedes older source + flags contradiction)
		{"s3_gateway", "quorum_mode", "strict-3of5", "relaxed-2of5", "Storage control-plane audit updated quorum_mode to strict-3of5.", "Legacy runbook claimed quorum_mode was relaxed-2of5.", "What is the authoritative quorum mode for s3_gateway?", false, true},
		{"iam_policy", "mfa_requirement", "hardware-fido2", "totp-optional", "Security compliance scan enforced mfa_requirement hardware-fido2.", "Old onboarding doc listed mfa_requirement as totp-optional.", "What MFA requirement is enforced by iam_policy?", false, true},
		{"billing_ledger", "settlement_currency", "USD", "EUR", "Finance reconciliation verified settlement_currency is USD.", "Regional draft sheet listed settlement_currency as EUR.", "What settlement currency is authoritative for billing_ledger?", false, true},
	}

	out := make([]EvalCase, 0, len(specs))
	for i, sp := range specs {
		tenant := fmt.Sprintf("tenant-%d", (i%3)+1)
		actor := fmt.Sprintf("auditor-%02d", i+1)
		t0 := base.Add(time.Duration(1000+i*48+rng.Intn(4)) * time.Hour)
		s1 := fmt.Sprintf("s-ch-%02d-1", i+1)
		s2 := fmt.Sprintf("s-ch-%02d-2", i+1)

		var turns []SessionTurn
		if sp.crossSource {
			// Earlier stale observation from legacy source in session scope, followed by newer authoritative observation from audit source in actor scope.
			turns = []SessionTurn{
				{
					TurnID: fmt.Sprintf("ch-%02d-t1", i+1), TenantID: tenant, ActorID: actor, SessionID: s1,
					Scope: "session", Timestamp: t0, Entity: sp.entity, Attribute: sp.attr, Value: sp.conflictVal,
					Statement: sp.hypStmt, Action: ActionAdd, Kind: KindObservation, Confidence: 0.72, SourceID: "legacy-doc",
				},
				{
					TurnID: fmt.Sprintf("ch-%02d-t2", i+1), TenantID: tenant, ActorID: actor, SessionID: s2,
					Scope: "actor", Timestamp: t0.Add(18 * time.Hour), Entity: sp.entity, Attribute: sp.attr, Value: sp.verifiedVal,
					Statement: sp.obsStmt, Action: ActionReflect, Kind: KindObservation, Confidence: 0.98, SourceID: "verified-audit",
				},
			}
		} else if sp.hypFirst {
			turns = []SessionTurn{
				{
					TurnID: fmt.Sprintf("ch-%02d-t1", i+1), TenantID: tenant, ActorID: actor, SessionID: s1,
					Scope: "session", Timestamp: t0, Entity: sp.entity, Attribute: sp.attr, Value: sp.conflictVal,
					Statement: sp.hypStmt, Action: ActionReflect, Kind: KindHypothesis, Confidence: 0.38, SourceID: "chat-guess",
				},
				{
					TurnID: fmt.Sprintf("ch-%02d-t2", i+1), TenantID: tenant, ActorID: actor, SessionID: s2,
					Scope: "actor", Timestamp: t0.Add(6 * time.Hour), Entity: sp.entity, Attribute: sp.attr, Value: sp.verifiedVal,
					Statement: sp.obsStmt, Action: ActionAdd, Kind: KindObservation, Confidence: 0.97, SourceID: "telemetry-probe",
				},
			}
		} else {
			turns = []SessionTurn{
				{
					TurnID: fmt.Sprintf("ch-%02d-t1", i+1), TenantID: tenant, ActorID: actor, SessionID: s1,
					Scope: "actor", Timestamp: t0, Entity: sp.entity, Attribute: sp.attr, Value: sp.verifiedVal,
					Statement: sp.obsStmt, Action: ActionAdd, Kind: KindObservation, Confidence: 0.97, SourceID: "telemetry-probe",
				},
				{
					TurnID: fmt.Sprintf("ch-%02d-t2", i+1), TenantID: tenant, ActorID: actor, SessionID: s2,
					Scope: "session", Timestamp: t0.Add(8 * time.Hour), Entity: sp.entity, Attribute: sp.attr, Value: sp.conflictVal,
					Statement: sp.hypStmt, Action: ActionReflect, Kind: KindHypothesis, Confidence: 0.36, SourceID: "chat-guess",
				},
			}
		}

		sub := "obs_vs_hypothesis"
		if sp.crossSource {
			sub = "cross_source_conflict"
		} else if sp.hypFirst {
			sub = "hypothesis_overturned"
		}

		out = append(out, EvalCase{
			ID:           fmt.Sprintf("lme-ch-%02d", i+1),
			Category:     CategoryContradiction,
			Subcategory:  sub,
			TenantID:     tenant,
			ActorID:      actor,
			SessionCount: 2,
			Turns:        turns,
			Query: RecallQuery{
				CaseID:       fmt.Sprintf("lme-ch-%02d", i+1),
				Category:     CategoryContradiction,
				TenantID:     tenant,
				ActorID:      actor,
				SessionIDs:   []string{s1, s2},
				Question:     sp.question,
				Entity:       sp.entity,
				Attribute:    sp.attr,
				QueryMode:    QueryStructured,
				TemporalMode: TemporalLatest,
				QueryTime:    t0.Add(24 * time.Hour),
			},
			ExpectedAnswer:      sp.verifiedVal,
			ExpectContradiction: true,
			ForbiddenStale:      []string{sp.conflictVal},
		})
	}
	return out
}

func buildTemporalOrderingCases(rng *rand.Rand, base time.Time) []EvalCase {
	out := make([]EvalCase, 0, 12)

	// 3 chronological sequence cases (with out-of-order ingestion to test event-time sorting)
	seqSpecs := []struct {
		entity string
		attr   string
		stages []string
		hours  []int
		order  []int // ingestion permutation
	}{
		{"release_v3", "stage", []string{"dev", "canary", "staging", "prod"}, []int{0, 12, 36, 60}, []int{2, 0, 3, 1}},
		{"incident_88", "milestone", []string{"detected", "triaged", "mitigated", "resolved"}, []int{0, 2, 7, 15}, []int{1, 3, 0, 2}},
		{"schema_mig", "phase", []string{"shadow_write", "backfill", "dual_read", "cutover"}, []int{0, 24, 72, 96}, []int{3, 1, 0, 2}},
	}
	for i, sp := range seqSpecs {
		tenant := fmt.Sprintf("tenant-%d", (i%2)+1)
		actor := fmt.Sprintf("timeline-agent-%02d", i+1)
		t0 := base.Add(time.Duration(2000+i*120+rng.Intn(5)) * time.Hour)
		var turns []SessionTurn
		var sids []string
		for _, idx := range sp.order {
			sid := fmt.Sprintf("s-to-seq-%02d-%d", i+1, idx+1)
			sids = append(sids, sid)
			turns = append(turns, SessionTurn{
				TurnID: fmt.Sprintf("to-seq-%02d-t%d", i+1, idx+1), TenantID: tenant, ActorID: actor, SessionID: sid,
				Scope: "actor", Timestamp: t0.Add(time.Duration(sp.hours[idx]) * time.Hour),
				Entity: sp.entity, Attribute: sp.attr, Value: sp.stages[idx],
				Statement: fmt.Sprintf("%s reached %s %s.", sp.entity, sp.attr, sp.stages[idx]),
				Action:    ActionUpdate, Kind: KindObservation, Confidence: 0.96,
			})
		}
		// Only pass the final session ID in query.SessionIDs to verify cross-session timeline durability.
		out = append(out, EvalCase{
			ID:           fmt.Sprintf("lme-to-%02d", i+1),
			Category:     CategoryTemporalOrdering,
			Subcategory:  "chronological_sequence_out_of_order",
			TenantID:     tenant,
			ActorID:      actor,
			SessionCount: len(sp.stages),
			Turns:        turns,
			Query: RecallQuery{
				CaseID:       fmt.Sprintf("lme-to-%02d", i+1),
				Category:     CategoryTemporalOrdering,
				TenantID:     tenant,
				ActorID:      actor,
				SessionIDs:   []string{sids[len(sids)-1]},
				Question:     fmt.Sprintf("List all %s %s values in chronological order.", sp.entity, sp.attr),
				Entity:       sp.entity,
				Attribute:    sp.attr,
				QueryMode:    QueryStructured,
				TemporalMode: TemporalSequence,
				QueryTime:    t0.Add(120 * time.Hour),
			},
			ExpectedOrder: sp.stages,
		})
	}

	// 2 First / Last ordering cases ("which happened first / last")
	firstLastSpecs := []struct {
		entity   string
		attr     string
		values   []string
		hours    []int
		mode     TemporalMode
		expected string
		q        string
	}{
		{"cluster_a", "region_failover", []string{"us-west-2", "us-east-1", "eu-central-1"}, []int{0, 18, 42}, TemporalFirst, "us-west-2", "Which region did cluster_a fail over to FIRST?"},
		{"canary_roll", "wave", []string{"wave-1pct", "wave-10pct", "wave-50pct", "wave-100pct"}, []int{0, 4, 10, 22}, TemporalLast, "wave-100pct", "Which wave completed LAST in canary_roll?"},
	}
	for i, sp := range firstLastSpecs {
		idx := 4 + i
		tenant := fmt.Sprintf("tenant-%d", (i%2)+1)
		actor := fmt.Sprintf("timeline-agent-%02d", idx)
		t0 := base.Add(time.Duration(2600+i*80) * time.Hour)
		var turns []SessionTurn
		// Ingest in reverse chronological order so arrival order != event order!
		for j := len(sp.values) - 1; j >= 0; j-- {
			sid := fmt.Sprintf("s-to-fl-%02d-%d", idx, j+1)
			turns = append(turns, SessionTurn{
				TurnID: fmt.Sprintf("to-fl-%02d-t%d", idx, j+1), TenantID: tenant, ActorID: actor, SessionID: sid,
				Scope: "actor", Timestamp: t0.Add(time.Duration(sp.hours[j]) * time.Hour),
				Entity: sp.entity, Attribute: sp.attr, Value: sp.values[j],
				Statement: fmt.Sprintf("%s recorded %s %s.", sp.entity, sp.attr, sp.values[j]),
				Action:    ActionUpdate, Kind: KindObservation, Confidence: 0.95,
			})
		}
		out = append(out, EvalCase{
			ID:           fmt.Sprintf("lme-to-%02d", idx),
			Category:     CategoryTemporalOrdering,
			Subcategory:  string(sp.mode),
			TenantID:     tenant,
			ActorID:      actor,
			SessionCount: len(sp.values),
			Turns:        turns,
			Query: RecallQuery{
				CaseID:       fmt.Sprintf("lme-to-%02d", idx),
				Category:     CategoryTemporalOrdering,
				TenantID:     tenant,
				ActorID:      actor,
				SessionIDs:   []string{fmt.Sprintf("s-to-fl-%02d-%d", idx, len(sp.values))},
				Question:     sp.q,
				Entity:       sp.entity,
				Attribute:    sp.attr,
				QueryMode:    QueryStructured,
				TemporalMode: sp.mode,
				QueryTime:    t0.Add(72 * time.Hour),
			},
			ExpectedAnswer: sp.expected,
		})
	}

	// 3 Duration-between-events cases
	durSpecs := []struct {
		entity   string
		attr     string
		values   []string
		hours    []int
		startVal string
		endVal   string
		expected string
	}{
		{"outage_71", "state", []string{"opened", "escalated", "resolved"}, []int{0, 6, 30}, "opened", "resolved", "30h"},
		{"backup_job", "state", []string{"queued", "snapshot_done", "verified_s3"}, []int{0, 12, 48}, "queued", "verified_s3", "48h"},
		{"index_build", "state", []string{"started", "sharded", "published"}, []int{0, 18, 72}, "started", "published", "72h"},
	}
	for i, sp := range durSpecs {
		idx := 6 + i
		tenant := "tenant-1"
		actor := fmt.Sprintf("timeline-agent-%02d", idx)
		t0 := base.Add(time.Duration(3000+i*96) * time.Hour)
		var turns []SessionTurn
		for j, val := range sp.values {
			turns = append(turns, SessionTurn{
				TurnID: fmt.Sprintf("to-dur-%02d-t%d", idx, j+1), TenantID: tenant, ActorID: actor,
				SessionID: fmt.Sprintf("s-to-dur-%02d-%d", idx, j+1),
				Scope:     "actor", Timestamp: t0.Add(time.Duration(sp.hours[j]) * time.Hour),
				Entity: sp.entity, Attribute: sp.attr, Value: val,
				Statement: fmt.Sprintf("%s transitioned to %s.", sp.entity, val),
				Action:    ActionUpdate, Kind: KindObservation, Confidence: 0.97,
			})
		}
		out = append(out, EvalCase{
			ID:           fmt.Sprintf("lme-to-%02d", idx),
			Category:     CategoryTemporalOrdering,
			Subcategory:  "duration_between_events",
			TenantID:     tenant,
			ActorID:      actor,
			SessionCount: len(sp.values),
			Turns:        turns,
			Query: RecallQuery{
				CaseID:             fmt.Sprintf("lme-to-%02d", idx),
				Category:           CategoryTemporalOrdering,
				TenantID:           tenant,
				ActorID:            actor,
				SessionIDs:         []string{fmt.Sprintf("s-to-dur-%02d-3", idx)},
				Question:           fmt.Sprintf("How many hours elapsed for %s between %s and %s?", sp.entity, sp.startVal, sp.endVal),
				Entity:             sp.entity,
				Attribute:          sp.attr,
				QueryMode:          QueryStructured,
				TemporalMode:       TemporalDuration,
				DurationStartValue: sp.startVal,
				DurationEndValue:   sp.endVal,
				QueryTime:          t0.Add(96 * time.Hour),
			},
			ExpectedAnswer: sp.expected,
		})
	}

	// 2 Time-windowed [t_start, t_end] query cases
	winSpecs := []struct {
		entity   string
		attr     string
		values   []string
		hours    []int
		winStart int
		winEnd   int
		expected []string
	}{
		{"deploy_log", "tag", []string{"v1.0", "v1.1", "v1.2", "v1.3", "v1.4"}, []int{0, 24, 48, 72, 120}, 20, 80, []string{"v1.1", "v1.2", "v1.3"}},
		{"audit_trail", "policy_rev", []string{"rev-1", "rev-2", "rev-3", "rev-4"}, []int{0, 12, 36, 96}, 10, 40, []string{"rev-2", "rev-3"}},
	}
	for i, sp := range winSpecs {
		idx := 9 + i
		tenant := "tenant-2"
		actor := fmt.Sprintf("timeline-agent-%02d", idx)
		t0 := base.Add(time.Duration(3500+i*144) * time.Hour)
		var turns []SessionTurn
		for j, val := range sp.values {
			turns = append(turns, SessionTurn{
				TurnID: fmt.Sprintf("to-win-%02d-t%d", idx, j+1), TenantID: tenant, ActorID: actor,
				SessionID: fmt.Sprintf("s-to-win-%02d-%d", idx, j+1),
				Scope:     "actor", Timestamp: t0.Add(time.Duration(sp.hours[j]) * time.Hour),
				Entity: sp.entity, Attribute: sp.attr, Value: val,
				Statement: fmt.Sprintf("%s recorded %s %s.", sp.entity, sp.attr, val),
				Action:    ActionUpdate, Kind: KindObservation, Confidence: 0.96,
			})
		}
		out = append(out, EvalCase{
			ID:           fmt.Sprintf("lme-to-%02d", idx),
			Category:     CategoryTemporalOrdering,
			Subcategory:  "time_windowed_range",
			TenantID:     tenant,
			ActorID:      actor,
			SessionCount: len(sp.values),
			Turns:        turns,
			Query: RecallQuery{
				CaseID:       fmt.Sprintf("lme-to-%02d", idx),
				Category:     CategoryTemporalOrdering,
				TenantID:     tenant,
				ActorID:      actor,
				SessionIDs:   []string{fmt.Sprintf("s-to-win-%02d-%d", idx, len(sp.values))},
				Question:     fmt.Sprintf("Which %s %s values occurred between hour %d and hour %d?", sp.entity, sp.attr, sp.winStart, sp.winEnd),
				Entity:       sp.entity,
				Attribute:    sp.attr,
				QueryMode:    QueryStructured,
				TemporalMode: TemporalWindow,
				WindowStart:  t0.Add(time.Duration(sp.winStart) * time.Hour),
				WindowEnd:    t0.Add(time.Duration(sp.winEnd) * time.Hour),
				QueryTime:    t0.Add(160 * time.Hour),
			},
			ExpectedOrder: sp.expected,
		})
	}

	// 2 Historical point-in-time AsOf cases ("before the day-3 migration")
	asOfSpecs := []struct {
		entity string
		attr   string
		vals   []string
		days   []int
		asOfD  int
		expect string
		newer  string
	}{
		{"edge_proxy", "region", []string{"us-east-1", "eu-west-1", "ap-northeast-1"}, []int{1, 2, 4}, 3, "eu-west-1", "ap-northeast-1"},
		{"billing_db", "version", []string{"pg-14", "pg-16", "pg-17"}, []int{1, 2, 5}, 3, "pg-16", "pg-17"},
	}
	for i, sp := range asOfSpecs {
		idx := 11 + i
		tenant := "tenant-1"
		actor := fmt.Sprintf("timeline-agent-%02d", idx)
		t0 := base.Add(time.Duration(3800+i*144) * time.Hour)
		var turns []SessionTurn
		var sids []string
		for j, val := range sp.vals {
			sid := fmt.Sprintf("s-to-asof-%02d-%d", idx, j+1)
			sids = append(sids, sid)
			turns = append(turns, SessionTurn{
				TurnID: fmt.Sprintf("to-asof-%02d-t%d", idx, j+1), TenantID: tenant, ActorID: actor, SessionID: sid,
				Scope: "actor", Timestamp: t0.Add(time.Duration(sp.days[j]*24) * time.Hour),
				Entity: sp.entity, Attribute: sp.attr, Value: val,
				Statement: fmt.Sprintf("Day %d: %s %s set to %s.", sp.days[j], sp.entity, sp.attr, val),
				Action:    ActionUpdate, Kind: KindObservation, Confidence: 0.96,
			})
		}
		out = append(out, EvalCase{
			ID:           fmt.Sprintf("lme-to-%02d", idx),
			Category:     CategoryTemporalOrdering,
			Subcategory:  "historical_as_of",
			TenantID:     tenant,
			ActorID:      actor,
			SessionCount: len(sp.vals),
			Turns:        turns,
			Query: RecallQuery{
				CaseID:       fmt.Sprintf("lme-to-%02d", idx),
				Category:     CategoryTemporalOrdering,
				TenantID:     tenant,
				ActorID:      actor,
				SessionIDs:   sids,
				Question:     fmt.Sprintf("What was %s %s BEFORE the migration (as of day %d)?", sp.entity, sp.attr, sp.asOfD),
				Entity:       sp.entity,
				Attribute:    sp.attr,
				QueryMode:    QueryStructured,
				TemporalMode: TemporalAsOf,
				AsOf:         t0.Add(time.Duration(sp.asOfD*24) * time.Hour),
				QueryTime:    t0.Add(168 * time.Hour),
			},
			ExpectedAnswer: sp.expect,
			ForbiddenStale: []string{sp.newer},
		})
	}

	return out
}

func buildAbstentionCases(rng *rand.Rand, base time.Time) []EvalCase {
	out := make([]EvalCase, 0, 12)

	// 4 Never-stated / false-premise queries with in-scope topical distractors
	unmentioned := []struct {
		entity   string
		attr     string
		question string
		distEnt  string
		distAttr string
		distVal  string
		distStmt string
	}{
		{"payment_svc", "iban_account", "What IBAN account does payment_svc settle into?", "payment_svc", "currency", "USD", "payment_svc processes invoices in USD."},
		{"user_profile", "passport_number", "What is the user's passport number?", "user_profile", "timezone", "America/Los_Angeles", "User timezone is America/Los_Angeles."},
		{"cluster_eu", "bgp_asn", "What BGP ASN is configured on cluster_eu?", "cluster_eu", "cidr_block", "10.44.0.0/16", "cluster_eu VPC uses CIDR 10.44.0.0/16."},
		{"zen_endpoint", "custom_lora_id", "Which custom LoRA adapter is loaded on zen_endpoint?", "zen_endpoint", "max_tokens", "32768", "zen_endpoint max_tokens is configured to 32768."},
	}
	for i, sp := range unmentioned {
		tenant := "tenant-1"
		actor := fmt.Sprintf("abstain-agent-%02d", i+1)
		t0 := base.Add(time.Duration(4000+i*24+rng.Intn(4)) * time.Hour)
		s1 := fmt.Sprintf("s-ab-ns-%02d-1", i+1)
		turns := []SessionTurn{
			{
				TurnID: fmt.Sprintf("ab-ns-%02d-d1", i+1), TenantID: tenant, ActorID: actor, SessionID: s1,
				Scope: "actor", Timestamp: t0, Entity: sp.distEnt, Attribute: sp.distAttr, Value: sp.distVal,
				Statement: sp.distStmt, Action: ActionAdd, Kind: KindObservation, Confidence: 0.95, IsDistractor: true,
			},
			{
				TurnID: fmt.Sprintf("ab-ns-%02d-d2", i+1), TenantID: tenant, ActorID: actor, SessionID: s1,
				Scope: "actor", Timestamp: t0.Add(time.Hour), Entity: "ops_note", Attribute: "owner", Value: "platform-team",
				Statement: "Platform team maintains internal service configuration notes.", Action: ActionAdd, Kind: KindObservation, Confidence: 0.92, IsDistractor: true,
			},
		}
		out = append(out, EvalCase{
			ID:           fmt.Sprintf("lme-ab-%02d", i+1),
			Category:     CategoryAbstention,
			Subcategory:  "never_stated_false_premise",
			TenantID:     tenant,
			ActorID:      actor,
			SessionCount: 1,
			Turns:        turns,
			Query: RecallQuery{
				CaseID:       fmt.Sprintf("lme-ab-%02d", i+1),
				Category:     CategoryAbstention,
				TenantID:     tenant,
				ActorID:      actor,
				SessionIDs:   []string{s1},
				Question:     sp.question,
				Entity:       sp.entity,
				Attribute:    sp.attr,
				QueryMode:    QueryStructured,
				TemporalMode: TemporalLatest,
				QueryTime:    t0.Add(12 * time.Hour),
			},
			ShouldAbstain:  true,
			ForbiddenStale: []string{sp.distVal, "platform-team"},
		})
	}

	// 4 Explicitly deleted / forgotten facts (#66 revocation)
	revoked := []struct {
		entity string
		attr   string
		val    string
		stmt   string
		act    MemoryAction
		q      string
	}{
		{"user", "phone_number", "555-0199", "User phone number is 555-0199.", ActionForget, "What is the user's phone number?"},
		{"ci_secret", "webhook_token", "whsec_9911aa", "CI webhook token stored as whsec_9911aa.", ActionDelete, "What is the CI webhook token?"},
		{"billing", "backup_card_last4", "4242", "Backup billing card ends in 4242.", ActionForget, "What are the last 4 digits of the backup billing card?"},
		{"ssh_bastion", "temp_passcode", "delta-7741", "Temporary bastion passcode is delta-7741.", ActionDelete, "What is the temporary bastion passcode?"},
	}
	for i, sp := range revoked {
		idx := 5 + i
		tenant := "tenant-1"
		actor := fmt.Sprintf("abstain-agent-%02d", idx)
		t0 := base.Add(time.Duration(4200+i*24) * time.Hour)
		s1 := fmt.Sprintf("s-ab-rev-%02d-1", idx)
		s2 := fmt.Sprintf("s-ab-rev-%02d-2", idx)
		turns := []SessionTurn{
			{
				TurnID: fmt.Sprintf("ab-rev-%02d-t1", idx), TenantID: tenant, ActorID: actor, SessionID: s1,
				Scope: "actor", Timestamp: t0, Entity: sp.entity, Attribute: sp.attr, Value: sp.val,
				Statement: sp.stmt, Action: ActionAdd, Kind: KindObservation, Confidence: 0.96,
			},
			{
				TurnID: fmt.Sprintf("ab-rev-%02d-t2", idx), TenantID: tenant, ActorID: actor, SessionID: s2,
				Scope: "actor", Timestamp: t0.Add(8 * time.Hour), Entity: sp.entity, Attribute: sp.attr, Value: "",
				Statement: fmt.Sprintf("Revoke and forget %s %s immediately.", sp.entity, sp.attr), Action: sp.act, Kind: KindObservation, Confidence: 1.0,
			},
		}
		out = append(out, EvalCase{
			ID:           fmt.Sprintf("lme-ab-%02d", idx),
			Category:     CategoryAbstention,
			Subcategory:  "deleted_or_forgotten",
			TenantID:     tenant,
			ActorID:      actor,
			SessionCount: 2,
			Turns:        turns,
			Query: RecallQuery{
				CaseID:       fmt.Sprintf("lme-ab-%02d", idx),
				Category:     CategoryAbstention,
				TenantID:     tenant,
				ActorID:      actor,
				SessionIDs:   []string{s1, s2},
				Question:     sp.q,
				Entity:       sp.entity,
				Attribute:    sp.attr,
				QueryMode:    QueryStructured,
				TemporalMode: TemporalLatest,
				QueryTime:    t0.Add(16 * time.Hour),
			},
			ShouldAbstain:  true,
			ForbiddenStale: []string{sp.val},
		})
	}

	// 4 Cross-tenant & Role-ACL isolation abstention probes (0 leaks allowed!)
	isoSpecs := []struct {
		writeTenant string
		queryTenant string
		aclRoles    []string
		callerRoles []string
		entity      string
		attr        string
		secretVal   string
		sub         string
	}{
		{"tenant-alpha", "tenant-beta", nil, nil, "vault", "api_quota", "95000", "cross_tenant_isolation"},
		{"tenant-alpha", "tenant-gamma", nil, nil, "billing", "enterprise_rate", "$0.0012", "cross_tenant_isolation"},
		{"tenant-alpha", "tenant-alpha", []string{"secops-admin"}, []string{"viewer"}, "kms", "rotation_seed_id", "seed-9981", "acl_role_isolation"},
		{"tenant-beta", "tenant-beta", []string{"finance-approver"}, []string{"dev-intern"}, "payroll", "wire_limit", "250000", "acl_role_isolation"},
	}
	for i, sp := range isoSpecs {
		idx := 9 + i
		actor := fmt.Sprintf("shared-actor-%02d", idx)
		t0 := base.Add(time.Duration(4500+i*24) * time.Hour)
		s1 := fmt.Sprintf("s-ab-iso-%02d-1", idx)
		turns := []SessionTurn{
			{
				TurnID: fmt.Sprintf("ab-iso-%02d-t1", idx), TenantID: sp.writeTenant, ActorID: actor, SessionID: s1,
				Scope: "actor", ACLRoles: sp.aclRoles, Timestamp: t0,
				Entity: sp.entity, Attribute: sp.attr, Value: sp.secretVal,
				Statement: fmt.Sprintf("Restricted %s %s is %s.", sp.entity, sp.attr, sp.secretVal),
				Action:    ActionAdd, Kind: KindObservation, Confidence: 0.99,
			},
		}
		out = append(out, EvalCase{
			ID:           fmt.Sprintf("lme-ab-%02d", idx),
			Category:     CategoryAbstention,
			Subcategory:  sp.sub,
			TenantID:     sp.queryTenant,
			ActorID:      actor,
			SessionCount: 1,
			Turns:        turns,
			Query: RecallQuery{
				CaseID:       fmt.Sprintf("lme-ab-%02d", idx),
				Category:     CategoryAbstention,
				TenantID:     sp.queryTenant,
				ActorID:      actor,
				SessionIDs:   []string{s1},
				CallerRoles:  sp.callerRoles,
				Question:     fmt.Sprintf("What is %s %s?", sp.entity, sp.attr),
				Entity:       sp.entity,
				Attribute:    sp.attr,
				QueryMode:    QueryStructured,
				TemporalMode: TemporalLatest,
				QueryTime:    t0.Add(6 * time.Hour),
			},
			ShouldAbstain:  true,
			ForbiddenStale: []string{sp.secretVal},
		})
	}

	return out
}

func buildCrossSessionCases(rng *rand.Rand, base time.Time) []EvalCase {
	out := make([]EvalCase, 0, 12)

	// 6 Multi-session aggregation cases scattered across 6 to 18 distinct sessions
	aggSpecs := []struct {
		entity   string
		attr     string
		values   []string
		sessions int
		question string
	}{
		{"release_q3", "affected_service", []string{"auth-api", "billing-worker", "search-index"}, 8, "Which services were recorded as affected in release_q3 across sessions?"},
		{"dr_drill", "verified_region", []string{"us-east-1", "eu-central-1", "ap-northeast-1"}, 10, "Which regions passed the DR drill across sessions?"},
		{"security_audit", "patched_cve", []string{"CVE-2026-1011", "CVE-2026-2044", "CVE-2026-3099", "CVE-2026-4120"}, 14, "List all CVEs patched during security_audit across sessions."},
		{"enso_fleet", "enabled_provider", []string{"zen-native", "vertex-eu", "bedrock-us"}, 12, "Which providers were enabled in enso_fleet across sessions?"},
		{"visor_images", "approved_base", []string{"alpine-3.21", "debian-12-slim", "ubuntu-24.04"}, 16, "Which base images were approved for visor_images across sessions?"},
		{"s3_buckets", "replicated_zone", []string{"iad-1", "fra-1", "nrt-1", "sin-1"}, 20, "Which zones were added to s3_buckets replication across all 20 sessions?"},
	}
	for i, sp := range aggSpecs {
		tenant := fmt.Sprintf("tenant-%d", (i%2)+1)
		actor := fmt.Sprintf("cross-agent-%02d", i+1)
		t0 := base.Add(time.Duration(5000+i*240) * time.Hour)

		var turns []SessionTurn
		needleSessions := make(map[int]string)
		stride := sp.sessions / len(sp.values)
		if stride < 1 {
			stride = 1
		}
		for vIdx, val := range sp.values {
			sIdx := vIdx * stride
			if sIdx >= sp.sessions-1 {
				sIdx = sp.sessions - 2
			}
			needleSessions[sIdx] = val
		}

		for s := 0; s < sp.sessions; s++ {
			sid := fmt.Sprintf("s-cs-%02d-%02d", i+1, s+1)
			ts := t0.Add(time.Duration(s*6) * time.Hour)
			if val, ok := needleSessions[s]; ok {
				turns = append(turns, SessionTurn{
					TurnID: fmt.Sprintf("cs-%02d-s%02d-needle", i+1, s+1), TenantID: tenant, ActorID: actor, SessionID: sid,
					Scope: "actor", Timestamp: ts, Entity: sp.entity, Attribute: sp.attr, Value: val,
					Statement: fmt.Sprintf("Session %d added %s %s = %s.", s+1, sp.entity, sp.attr, val),
					Action:    ActionAdd, Kind: KindObservation, Confidence: 0.96,
				})
			} else {
				turns = append(turns, SessionTurn{
					TurnID: fmt.Sprintf("cs-%02d-s%02d-bg", i+1, s+1), TenantID: tenant, ActorID: actor, SessionID: sid,
					Scope: "actor", Timestamp: ts, Entity: fmt.Sprintf("bg_task_%d", s), Attribute: "status", Value: "ok",
					Statement: fmt.Sprintf("Background session %d telemetry sync ok (check %d).", s+1, rng.Intn(1000)),
					Action:    ActionAdd, Kind: KindObservation, Confidence: 0.90, IsDistractor: true,
				})
			}
			// Add a cross-tenant turn on the same entity/attr mid-way to verify 0 leaks.
			if s == sp.sessions/2 {
				otherTenant := "tenant-intruder"
				turns = append(turns, SessionTurn{
					TurnID: fmt.Sprintf("cs-%02d-intruder", i+1), TenantID: otherTenant, ActorID: actor, SessionID: sid,
					Scope: "actor", Timestamp: ts.Add(time.Minute), Entity: sp.entity, Attribute: sp.attr, Value: "intruder-val",
					Statement: fmt.Sprintf("Intruder tenant wrote %s %s = intruder-val.", sp.entity, sp.attr),
					Action:    ActionAdd, Kind: KindObservation, Confidence: 0.99,
				})
			}
		}

		finalSID := fmt.Sprintf("s-cs-%02d-%02d", i+1, sp.sessions)
		out = append(out, EvalCase{
			ID:           fmt.Sprintf("lme-cs-%02d", i+1),
			Category:     CategoryCrossSession,
			Subcategory:  "multi_session_aggregation",
			TenantID:     tenant,
			ActorID:      actor,
			SessionCount: sp.sessions,
			Turns:        turns,
			Query: RecallQuery{
				CaseID:       fmt.Sprintf("lme-cs-%02d", i+1),
				Category:     CategoryCrossSession,
				TenantID:     tenant,
				ActorID:      actor,
				SessionIDs:   []string{finalSID},
				Question:     sp.question,
				Entity:       sp.entity,
				Attribute:    sp.attr,
				QueryMode:    QueryStructured,
				TemporalMode: TemporalAggregate,
				QueryTime:    t0.Add(time.Duration(sp.sessions*6+12) * time.Hour),
			},
			ExpectedSet:    sp.values,
			ForbiddenStale: []string{"intruder-val"},
		})
	}

	// 6 Long-range single-fact cross-session recall cases (fact in session 1, queried in session 8..16)
	recallSpecs := []struct {
		entity   string
		attr     string
		value    string
		sessions int
		stmt     string
		q        string
	}{
		{"project_hydra", "target_db", "postgres-17", 8, "Standardized project_hydra target_db on postgres-17.", "Which target database did we pick for project_hydra?"},
		{"cloud_billing", "invoice_cadence", "monthly-net30", 10, "Set cloud_billing invoice_cadence to monthly-net30.", "What is the invoice cadence for cloud_billing?"},
		{"kai_gate", "fallback_action", "abstain_and_escalate", 12, "Configured kai_gate fallback_action to abstain_and_escalate.", "What fallback action is configured for kai_gate?"},
		{"vfs_cache", "eviction_ttl", "6h", 14, "Set vfs_cache eviction_ttl to 6h.", "What eviction TTL was configured for vfs_cache?"},
		{"visor_net", "egress_policy", "allowlist-only", 16, "Enforced visor_net egress_policy as allowlist-only.", "What egress policy was set for visor_net?"},
		{"zen_batch", "max_batch_size", "64", 18, "Tuned zen_batch max_batch_size to 64.", "What max batch size was tuned for zen_batch?"},
	}
	for i, sp := range recallSpecs {
		idx := 7 + i
		tenant := fmt.Sprintf("tenant-%d", (i%2)+1)
		actor := fmt.Sprintf("cross-agent-%02d", idx)
		t0 := base.Add(time.Duration(6500+i*240) * time.Hour)

		turns := []SessionTurn{
			{
				TurnID: fmt.Sprintf("cs-%02d-s01-needle", idx), TenantID: tenant, ActorID: actor,
				SessionID: fmt.Sprintf("s-cs-%02d-01", idx),
				Scope:     "actor", Timestamp: t0, Entity: sp.entity, Attribute: sp.attr, Value: sp.value,
				Statement: sp.stmt, Action: ActionAdd, Kind: KindObservation, Confidence: 0.96,
			},
		}
		for s := 1; s < sp.sessions; s++ {
			turns = append(turns, SessionTurn{
				TurnID: fmt.Sprintf("cs-%02d-s%02d-bg", idx, s+1), TenantID: tenant, ActorID: actor,
				SessionID: fmt.Sprintf("s-cs-%02d-%02d", idx, s+1),
				Scope:     "actor", Timestamp: t0.Add(time.Duration(s*5) * time.Hour),
				Entity: fmt.Sprintf("session_note_%d", s), Attribute: "summary", Value: fmt.Sprintf("routine-check-%d", s),
				Statement: fmt.Sprintf("Session %d routine status check completed without errors.", s+1),
				Action:    ActionAdd, Kind: KindObservation, Confidence: 0.90, IsDistractor: true,
			})
		}
		finalSID := fmt.Sprintf("s-cs-%02d-%02d", idx, sp.sessions)
		out = append(out, EvalCase{
			ID:           fmt.Sprintf("lme-cs-%02d", idx),
			Category:     CategoryCrossSession,
			Subcategory:  "long_range_single_fact_recall",
			TenantID:     tenant,
			ActorID:      actor,
			SessionCount: sp.sessions,
			Turns:        turns,
			Query: RecallQuery{
				CaseID:       fmt.Sprintf("lme-cs-%02d", idx),
				Category:     CategoryCrossSession,
				TenantID:     tenant,
				ActorID:      actor,
				SessionIDs:   []string{finalSID},
				Question:     sp.q,
				Entity:       sp.entity,
				Attribute:    sp.attr,
				QueryMode:    QueryStructured,
				TemporalMode: TemporalLatest,
				QueryTime:    t0.Add(time.Duration(sp.sessions*5+12) * time.Hour),
			},
			ExpectedAnswer: sp.value,
		})
	}

	return out
}

// ============================================================================
// Negative Regime 1 Dataset: Recency-Bias Failure on Historical AsOf Queries
// ============================================================================

// BuildHistoricalQuerySuite generates 16 deterministic cases asking for a fact's
// value BEFORE a later migration ("what was X as of day 3?"), exposing policies
// that blindly prefer the newest timestamp.
func BuildHistoricalQuerySuite(seed int64) []EvalCase {
	rng := rand.New(rand.NewSource(seed + 101))
	base := time.Date(2026, 7, 1, 9, 0, 0, 0, time.UTC)

	specs := []struct {
		entity string
		attr   string
		d1     string
		d2     string
		d4     string
		d5     string
	}{
		{"gateway", "region", "us-east-1", "eu-west-1", "ap-northeast-1", "ap-southeast-1"},
		{"ledger_db", "engine", "postgres-14", "postgres-16", "cockroach-24", "cockroach-25"},
		{"enso_edge", "default_tier", "zen-small", "zen-medium", "zen-pro", "zen-ultra"},
		{"vfs_bucket", "storage_class", "standard", "ia-archive", "express-nvme", "glacier-instant"},
		{"visor_vm", "kernel_profile", "linux-6.6", "linux-6.8", "linux-6.12", "linux-6.14"},
		{"auth_mesh", "cipher_suite", "aes-128-gcm", "aes-256-gcm", "chacha20-poly1305", "ml-kem-hybrid"},
		{"search_svc", "shard_count", "4", "8", "16", "32"},
		{"billing_api", "webhook_v", "v1", "v2", "v3", "v4"},
		{"ci_pool", "runner_arch", "x86_64", "arm64-graviton3", "arm64-graviton4", "riscv-preview"},
		{"cache_mesh", "cluster_mode", "standalone", "sentinel", "cluster-3node", "cluster-6node"},
		{"alert_bus", "escalation_min", "30m", "15m", "5m", "2m"},
		{"telemetry", "sample_rate", "10pct", "25pct", "50pct", "100pct"},
		{"cdn_origin", "pop_region", "iad", "fra", "nrt", "syd"},
		{"dns_zone", "ttl_sec", "3600", "600", "120", "30"},
		{"queue_core", "partition_cnt", "12", "24", "48", "96"},
		{"deploy_pipe", "canary_pct", "5pct", "10pct", "25pct", "50pct"},
	}

	out := make([]EvalCase, 0, len(specs))
	for i, sp := range specs {
		tenant := fmt.Sprintf("tenant-hist-%d", (i%2)+1)
		actor := fmt.Sprintf("hist-actor-%02d", i+1)
		t0 := base.Add(time.Duration(i*168+rng.Intn(4)) * time.Hour)

		vals := []string{sp.d1, sp.d2, sp.d4, sp.d5}
		days := []int{1, 2, 4, 5}
		var turns []SessionTurn
		var sids []string
		for j, val := range vals {
			sid := fmt.Sprintf("s-hist-%02d-d%d", i+1, days[j])
			sids = append(sids, sid)
			act := ActionUpdate
			if j == 0 {
				act = ActionAdd
			}
			turns = append(turns, SessionTurn{
				TurnID: fmt.Sprintf("hist-%02d-d%d", i+1, days[j]), TenantID: tenant, ActorID: actor, SessionID: sid,
				Scope: "actor", Timestamp: t0.Add(time.Duration(days[j]*24) * time.Hour),
				Entity: sp.entity, Attribute: sp.attr, Value: val,
				Statement: fmt.Sprintf("Day %d: %s %s set to %s.", days[j], sp.entity, sp.attr, val),
				Action:    act, Kind: KindObservation, Confidence: 0.96,
			})
		}

		// Ask explicitly for state BEFORE day-4 migration (AsOf = Day 3).
		asOf := t0.Add(3 * 24 * time.Hour)
		out = append(out, EvalCase{
			ID:           fmt.Sprintf("neg1-hist-%02d", i+1),
			Category:     CategoryTemporalOrdering,
			Subcategory:  "historical_as_of_before_migration",
			TenantID:     tenant,
			ActorID:      actor,
			SessionCount: 4,
			Turns:        turns,
			Query: RecallQuery{
				CaseID:       fmt.Sprintf("neg1-hist-%02d", i+1),
				Category:     CategoryTemporalOrdering,
				TenantID:     tenant,
				ActorID:      actor,
				SessionIDs:   sids,
				Question:     fmt.Sprintf("What was %s %s BEFORE the migration on day 4 (as of day 3)?", sp.entity, sp.attr),
				Entity:       sp.entity,
				Attribute:    sp.attr,
				QueryMode:    QueryStructured,
				TemporalMode: TemporalAsOf,
				AsOf:         asOf,
				QueryTime:    t0.Add(7 * 24 * time.Hour),
			},
			ExpectedAnswer: sp.d2,
			ForbiddenStale: []string{sp.d4, sp.d5}, // Post-migration newer values = recency-bias error!
		})
	}
	return out
}

// ============================================================================
// Negative Regime 2 Dataset: High-Distractor Long Horizon (5..200 sessions)
// ============================================================================

// BuildLongHorizonSuite creates 20 semantic-only cases (10 needles + 10
// unanswerable probes) embedded inside distractorSessions overlapping ops
// sessions to empirically measure retrieval degradation & FPR scaling.
func BuildLongHorizonSuite(seed int64, distractorSessions int) []EvalCase {
	base := time.Date(2026, 8, 1, 9, 0, 0, 0, time.UTC)

	distractorTopics := []struct {
		entity string
		attr   string
		valFmt string
		tmpl   string
	}{
		{"gateway", "region", "region-check-%d", "Operations check %d reviewed gateway region cluster endpoint latency and deployment policy."},
		{"billing", "database", "db-audit-%d", "Telemetry audit %d inspected billing database replication cluster quota and failover tier."},
		{"visor", "sandbox", "visor-run-%d", "Sandbox verification %d checked visor container image runtime egress policy and memory limit."},
		{"enso", "routing", "route-trace-%d", "Model router trace %d profiled enso inference gateway latency batch size and fallback model."},
		{"vfs", "storage", "s3-sync-%d", "Storage sync %d benchmarked vfs s3 bucket replication cache ttl and throughput quota."},
		{"auth", "token", "iam-check-%d", "Security audit %d verified auth broker token ttl mfa policy and rotation schedule."},
		{"search", "index", "idx-scan-%d", "Index maintenance %d refreshed search shard count vector threshold and query cache."},
		{"deploy", "pipeline", "pipe-step-%d", "Release pipeline %d validated canary wave rollout approval mode and rollback trigger."},
		{"metrics", "collector", "otel-batch-%d", "Observability collector %d flushed metrics pipeline retention window and sampling ratio."},
		{"queue", "broker", "mq-flush-%d", "Message broker %d balanced queue partitions disk fsync throughput and consumer lag."},
	}

	needles := []struct {
		entity string
		attr   string
		val    string
		stmt   string
		q      string
	}{
		{"gateway", "canary_region", "eu-north-1", "Confirmed gateway canary_region deployment runs in eu-north-1.", "Which region runs the gateway canary deployment?"},
		{"billing", "settlement_bank", "mercury-treasury", "Verified billing settlement_bank account routes via mercury-treasury.", "Which settlement bank handles billing treasury routing?"},
		{"visor", "microvm_runtime", "firecracker-1.10", "Configured visor microvm_runtime engine to firecracker-1.10.", "What microVM runtime engine is configured for visor?"},
		{"enso", "fallback_model", "zen-4-mini", "Enso router fallback_model switches to zen-4-mini during quota spikes.", "Which fallback model does enso switch to during spikes?"},
		{"vfs", "archive_bucket", "hanzo-cold-fra01", "VFS archive_bucket points to hanzo-cold-fra01 for cold snapshots.", "Which archive bucket stores VFS cold snapshots?"},
		{"auth", "jwks_issuer", "iam.hanzo.id", "Auth broker jwks_issuer validates signatures from iam.hanzo.id.", "Which JWKS issuer validates auth broker signatures?"},
		{"search", "hnsw_ef_search", "128", "Search index hnsw_ef_search parameter tuned to 128 for recall.", "What HNSW ef_search parameter was tuned on the search index?"},
		{"deploy", "freeze_window", "friday-20utc", "Deployment freeze_window begins every friday-20utc across prod.", "When does the production deployment freeze window begin?"},
		{"metrics", "alert_webhook", "pagerduty-eu-v2", "Metrics alert_webhook dispatches critical pages to pagerduty-eu-v2.", "Which alert webhook receives critical metrics pages?"},
		{"queue", "deadletter_topic", "dlq-orders-v3", "Queue broker deadletter_topic captures poisoned jobs in dlq-orders-v3.", "Which deadletter topic captures poisoned queue jobs?"},
	}

	unanswerables := []struct {
		q string
	}{
		{"What satellite uplink frequency does the gateway region cluster use?"},
		{"Which Swiss notary holds the billing database replication paper keys?"},
		{"What submarine cable carries visor sandbox egress container traffic?"},
		{"Which quantum annealing chip backs enso router fallback model inference?"},
		{"What glacial cavern stores physical tapes for vfs s3 bucket replication?"},
		{"Which biometric iris scanner authorizes auth broker token ttl policy?"},
		{"What acoustic modem syncs search index shard count vector thresholds?"},
		{"Which maritime lighthouse monitors release pipeline canary wave rollouts?"},
		{"What seismic sensor triggers observability metrics pipeline retention sweeps?"},
		{"Which carrier pigeon loft backs message broker queue partition failover?"},
	}

	// Prefix-nested distractor builder: seeded strictly by (seed, caseIdx) so
	// D=5 is an exact prefix of D=25, D=50, D=100, and D=200.
	buildDistractors := func(tenant, actor string, t0 time.Time, caseIdx int) []SessionTurn {
		caseRng := rand.New(rand.NewSource(seed + 202 + int64(caseIdx)*97))
		turns := make([]SessionTurn, 0, distractorSessions*2)
		for s := 0; s < distractorSessions; s++ {
			sid := fmt.Sprintf("s-hz-%02d-%03d", caseIdx, s+1)
			for sub := 0; sub < 2; sub++ {
				tp := distractorTopics[(s*2+sub+caseIdx+caseRng.Intn(len(distractorTopics)))%len(distractorTopics)]
				seq := s*2 + sub + 1
				turns = append(turns, SessionTurn{
					TurnID:       fmt.Sprintf("hz-%02d-d%04d", caseIdx, seq),
					TenantID:     tenant,
					ActorID:      actor,
					SessionID:    sid,
					Scope:        "actor",
					Timestamp:    t0.Add(time.Duration(seq) * time.Hour),
					Entity:       fmt.Sprintf("%s_%d", tp.entity, seq),
					Attribute:    tp.attr,
					Value:        fmt.Sprintf(tp.valFmt, seq),
					Statement:    fmt.Sprintf(tp.tmpl, seq),
					Action:       ActionAdd,
					Kind:         KindObservation,
					Confidence:   0.88,
					IsDistractor: true,
				})
			}
		}
		return turns
	}

	out := make([]EvalCase, 0, len(needles)+len(unanswerables))
	for i, n := range needles {
		tenant := "tenant-horizon"
		actor := fmt.Sprintf("hz-needle-%d-%02d", distractorSessions, i+1)
		t0 := base.Add(time.Duration(i*500) * time.Hour)

		turns := buildDistractors(tenant, actor, t0, i+1)
		// Place needle at fixed early position (index 2, hour 3) so larger horizons
		// strictly append newer distractors after the needle.
		needleIdx := 2
		if needleIdx > len(turns) {
			needleIdx = len(turns)
		}
		needleTurn := SessionTurn{
			TurnID: fmt.Sprintf("hz-%02d-needle", i+1), TenantID: tenant, ActorID: actor,
			SessionID: fmt.Sprintf("s-hz-%02d-needle", i+1),
			Scope:     "actor", Timestamp: t0.Add(time.Duration(needleIdx+1) * time.Hour),
			Entity: n.entity, Attribute: n.attr, Value: n.val,
			Statement: n.stmt, Action: ActionAdd, Kind: KindObservation, Confidence: 0.96,
		}
		turns = append(turns[:needleIdx], append([]SessionTurn{needleTurn}, turns[needleIdx:]...)...)

		out = append(out, EvalCase{
			ID:           fmt.Sprintf("neg2-hz%d-needle-%02d", distractorSessions, i+1),
			Category:     CategoryCrossSession,
			Subcategory:  fmt.Sprintf("long_horizon_%d_sessions", distractorSessions),
			TenantID:     tenant,
			ActorID:      actor,
			SessionCount: distractorSessions + 1,
			Turns:        turns,
			Query: RecallQuery{
				CaseID:       fmt.Sprintf("neg2-hz%d-needle-%02d", distractorSessions, i+1),
				Category:     CategoryCrossSession,
				TenantID:     tenant,
				ActorID:      actor,
				Question:     n.q,
				QueryMode:    QuerySemantic,
				TemporalMode: TemporalLatest,
				QueryTime:    t0.Add(time.Duration(distractorSessions*2+24) * time.Hour),
			},
			ExpectedAnswer: n.val,
		})
	}

	for i, u := range unanswerables {
		tenant := "tenant-horizon"
		actor := fmt.Sprintf("hz-unans-%d-%02d", distractorSessions, i+1)
		t0 := base.Add(time.Duration((i+10)*500) * time.Hour)
		turns := buildDistractors(tenant, actor, t0, i+11)

		out = append(out, EvalCase{
			ID:           fmt.Sprintf("neg2-hz%d-unans-%02d", distractorSessions, i+1),
			Category:     CategoryAbstention,
			Subcategory:  fmt.Sprintf("long_horizon_unanswerable_%d", distractorSessions),
			TenantID:     tenant,
			ActorID:      actor,
			SessionCount: distractorSessions,
			Turns:        turns,
			Query: RecallQuery{
				CaseID:       fmt.Sprintf("neg2-hz%d-unans-%02d", distractorSessions, i+1),
				Category:     CategoryAbstention,
				TenantID:     tenant,
				ActorID:      actor,
				Question:     u.q,
				QueryMode:    QuerySemantic,
				TemporalMode: TemporalLatest,
				QueryTime:    t0.Add(time.Duration(distractorSessions*2+24) * time.Hour),
			},
			ShouldAbstain: true,
		})
	}

	return out
}

// ============================================================================
// Negative Regime 3 Dataset: Abstention Threshold Over-Conservatism (Pareto)
// ============================================================================

// BuildParaphraseAbstentionSuite generates 40 semantic-only cases:
//   - 20 valid paraphrased multi-session queries (varying lexical overlap from
//     near-exact to heavy paraphrase, creating realistic intermediate scores)
//   - 20 unanswerable false-premise queries with topically close distractors
//
// Sweeping Kai MinEvidenceScore across [0.10 .. 0.90] exposes the exact
// Pareto tradeoff between Unanswerable FPR and Paraphrased Valid Recall.
func BuildParaphraseAbstentionSuite(seed int64) []EvalCase {
	rng := rand.New(rand.NewSource(seed + 303))
	base := time.Date(2026, 9, 1, 9, 0, 0, 0, time.UTC)

	bgPool := []struct {
		ent  string
		attr string
		val  string
		stmt string
	}{
		{"telemetry_bus", "collector_region", "eu-west-1", "Telemetry collector pipeline aggregates application metrics in eu-west-1."},
		{"edge_ingress", "tls_profile", "strict-1.3", "Edge ingress load balancer enforces TLS 1.3 strict cipher profile."},
		{"ci_cluster", "build_cache", "s3-nvme-cache", "Continuous integration workers mount s3-nvme-cache for build artifacts."},
		{"auth_service", "session_store", "redis-sentinel", "Authentication service keeps active session tokens in redis-sentinel."},
		{"billing_core", "ledger_engine", "postgres-17", "Billing core records immutable customer invoices inside postgres-17."},
		{"visor_runner", "isolation_mode", "microvm-strict", "Visor runner executes untrusted workloads under microvm-strict isolation."},
		{"enso_gateway", "fallback_tier", "zen-mini", "Enso model gateway routes overflow traffic to zen-mini fallback tier."},
		{"vfs_layer", "object_backend", "hanzo-s3", "Virtual filesystem layer persists workspace blobs into hanzo-s3 storage."},
	}

	paraphrases := []struct {
		ent  string
		attr string
		val  string
		stmt string
		q    string // Ranges from high overlap down to heavy paraphrase
	}{
		{"analytics_db", "engine", "clickhouse-eu", "Primary analytics database runs on clickhouse-eu in Frankfurt.", "Which engine runs the primary analytics database in Frankfurt?"},
		{"alert_pager", "escalation_team", "sre-platform", "Critical production alert pager escalates directly to sre-platform.", "Which team receives critical production alert pager escalations?"},
		{"deploy_bot", "canary_duration", "45m", "Automated deploy_bot holds every canary rollout for 45m.", "How long does automated deploy_bot hold each canary rollout?"},
		{"search_vec", "distance_metric", "cosine", "Vector search index computes nearest neighbors using cosine metric.", "Which distance metric does the vector search index compute?"},
		{"kms_vault", "rotation_cycle", "30d", "KMS vault rotates envelope encryption keys on a 30d cycle.", "How frequently does KMS vault rotate envelope encryption keys?"},
		{"worker_pool", "max_retries", "5", "Background worker pool retries transient webhook failures up to 5 times.", "How many times does the worker pool retry webhook failures?"},
		{"cdn_cache", "purge_mode", "surrogate-key", "CDN cache invalidation uses surrogate-key tags for instant purges.", "What invalidation tag mechanism does CDN cache use for purges?"},
		{"audit_sink", "export_format", "parquet-zstd", "Compliance audit sink archives daily logs as parquet-zstd files.", "Which file format does the compliance audit sink archive daily logs in?"},
		{"rate_limiter", "algorithm", "token-bucket", "API rate limiter enforces per-tenant quotas via token-bucket algorithm.", "Which algorithm enforces per-tenant quotas in the API rate limiter?"},
		{"trace_pipe", "sampling_policy", "tail-based-error", "Distributed tracing pipeline applies tail-based-error sampling policy.", "What sampling policy does the distributed tracing pipeline apply?"},
		// Moderate to heavy paraphrases (lower lexical overlap -> tested by high theta!)
		{"ledger_sync", "region", "zurich-ch1", "Financial ledger replicas reside in zurich-ch1 for Swiss residency.", "Where are the Swiss financial records hosted?"},
		{"gpu_sched", "preemption", "checkpoint-resume", "GPU scheduler handles spot eviction via checkpoint-resume snapshots.", "How does training recover from spot node eviction?"},
		{"secret_scan", "hook", "pre-receive", "Git secret scanner blocks credential leaks at the pre-receive hook.", "At what repository stage are leaked credentials blocked?"},
		{"dns_failover", "probe_interval", "10s", "Global DNS health checker probes origin endpoints every 10s.", "How often are global origin health checks performed?"},
		{"backup_vault", "immutability", "worm-90d", "Disaster recovery snapshots carry worm-90d object lock retention.", "What object lock durability protects disaster recovery snapshots?"},
		{"mesh_proxy", "mtls_ca", "spiffe-spire", "Service mesh sidecars obtain workload certificates from spiffe-spire.", "Which certificate authority issues sidecar workload identities?"},
		{"queue_dlq", "retention", "14d", "Poisoned event messages stay inspectable in the dead-letter queue for 14d.", "How long can failed queue messages be inspected before expiry?"},
		{"feature_flag", "eval_mode", "local-sdk-cache", "Feature flags evaluate offline using local-sdk-cache polling.", "How are rollout toggles resolved without network round trips?"},
		{"schema_reg", "compat_rule", "backward-transitive", "Event schema registry enforces backward-transitive compatibility checks.", "Which evolution compatibility rule governs published event schemas?"},
		{"cost_guard", "hard_cap", "$5000-daily", "Cloud FinOps budget guard trips hard_cap breaker at $5000-daily.", "At what daily spend limit does the FinOps breaker trip?"},
	}

	unanswerableProbes := []string{
		"What encryption key size does the telemetry collector pipeline use in eu-west-1?",
		"Which SSL certificate vendor issued the edge ingress load balancer wildcard?",
		"What NVMe RAID level backs the continuous integration s3-nvme-cache workers?",
		"What eviction memory cap is configured on authentication service redis-sentinel?",
		"Which tax jurisdiction audits billing core customer invoices in postgres-17?",
		"What hypervisor firmware patch level runs visor runner microvm-strict isolation?",
		"What prompt caching TTL does enso model gateway apply on zen-mini overflow?",
		"Which erasure coding stripe width backs virtual filesystem hanzo-s3 blobs?",
		"What SNMP community string monitors edge ingress load balancer metrics?",
		"Which pager rotation calendar owns telemetry collector pipeline outages?",
		"What JVM heap size is assigned to continuous integration build artifacts?",
		"What Redis ACL password authenticates active session tokens in redis-sentinel?",
		"Which accounting ERP imports billing core postgres-17 invoices?",
		"What seccomp syscall whitelist restricts visor runner untrusted workloads?",
		"Which fine-tuned checkpoint powers enso model gateway overflow traffic?",
		"What glacier transition delay archives virtual filesystem workspace blobs?",
		"Which office subnet is allowlisted on edge ingress load balancer TLS?",
		"What Prometheus scrape port exposes telemetry collector pipeline metrics?",
		"Which hardware HSM signs authentication service active session tokens?",
		"What Postgres tablespace stores billing core immutable customer invoices?",
	}

	out := make([]EvalCase, 0, len(paraphrases)+len(unanswerableProbes))
	for i, pp := range paraphrases {
		tenant := "tenant-pareto"
		actor := fmt.Sprintf("pareto-valid-%02d", i+1)
		t0 := base.Add(time.Duration(i*48) * time.Hour)

		turns := make([]SessionTurn, 0, len(bgPool)+1)
		for j, bg := range bgPool {
			turns = append(turns, SessionTurn{
				TurnID: fmt.Sprintf("pv-%02d-bg%d", i+1, j+1), TenantID: tenant, ActorID: actor,
				SessionID: fmt.Sprintf("s-pv-%02d-%d", i+1, j+1), Scope: "actor",
				Timestamp: t0.Add(time.Duration(j*2) * time.Hour),
				Entity:    bg.ent, Attribute: bg.attr, Value: bg.val, Statement: bg.stmt,
				Action: ActionAdd, Kind: KindObservation, Confidence: 0.92, IsDistractor: true,
			})
		}
		// Insert the valid target fact across sessions.
		targetPos := 1 + rng.Intn(len(bgPool)-1)
		targetTurn := SessionTurn{
			TurnID: fmt.Sprintf("pv-%02d-target", i+1), TenantID: tenant, ActorID: actor,
			SessionID: fmt.Sprintf("s-pv-%02d-target", i+1), Scope: "actor",
			Timestamp: t0.Add(time.Duration(targetPos*2+1) * time.Hour),
			Entity:    pp.ent, Attribute: pp.attr, Value: pp.val, Statement: pp.stmt,
			Action: ActionAdd, Kind: KindObservation, Confidence: 0.96,
		}
		turns = append(turns[:targetPos], append([]SessionTurn{targetTurn}, turns[targetPos:]...)...)

		out = append(out, EvalCase{
			ID:           fmt.Sprintf("neg3-valid-%02d", i+1),
			Category:     CategoryCrossSession,
			Subcategory:  "paraphrased_valid_query",
			TenantID:     tenant,
			ActorID:      actor,
			SessionCount: len(bgPool) + 1,
			Turns:        turns,
			Query: RecallQuery{
				CaseID:       fmt.Sprintf("neg3-valid-%02d", i+1),
				Category:     CategoryCrossSession,
				TenantID:     tenant,
				ActorID:      actor,
				Question:     pp.q,
				QueryMode:    QuerySemantic,
				TemporalMode: TemporalLatest,
				QueryTime:    t0.Add(48 * time.Hour),
			},
			ExpectedAnswer: pp.val,
			IsParaphrase:   true,
		})
	}

	for i, q := range unanswerableProbes {
		tenant := "tenant-pareto"
		actor := fmt.Sprintf("pareto-unans-%02d", i+1)
		t0 := base.Add(time.Duration((i+25)*48) * time.Hour)

		turns := make([]SessionTurn, 0, len(bgPool))
		for j, bg := range bgPool {
			turns = append(turns, SessionTurn{
				TurnID: fmt.Sprintf("pu-%02d-bg%d", i+1, j+1), TenantID: tenant, ActorID: actor,
				SessionID: fmt.Sprintf("s-pu-%02d-%d", i+1, j+1), Scope: "actor",
				Timestamp: t0.Add(time.Duration(j*2) * time.Hour),
				Entity:    bg.ent, Attribute: bg.attr, Value: bg.val, Statement: bg.stmt,
				Action: ActionAdd, Kind: KindObservation, Confidence: 0.92, IsDistractor: true,
			})
		}

		out = append(out, EvalCase{
			ID:           fmt.Sprintf("neg3-unans-%02d", i+1),
			Category:     CategoryAbstention,
			Subcategory:  "false_premise_close_distractor",
			TenantID:     tenant,
			ActorID:      actor,
			SessionCount: len(bgPool),
			Turns:        turns,
			Query: RecallQuery{
				CaseID:       fmt.Sprintf("neg3-unans-%02d", i+1),
				Category:     CategoryAbstention,
				TenantID:     tenant,
				ActorID:      actor,
				Question:     q,
				QueryMode:    QuerySemantic,
				TemporalMode: TemporalLatest,
				QueryTime:    t0.Add(48 * time.Hour),
			},
			ShouldAbstain: true,
		})
	}

	return out
}
