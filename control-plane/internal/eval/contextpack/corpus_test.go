// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

package contextpack

import (
	"encoding/json"
	"testing"
)

func TestGenerateCorpus_DeterministicSeed(t *testing.T) {
	a := GenerateCorpus(1346)
	b := GenerateCorpus(1346)

	ja, err := json.Marshal(a)
	if err != nil {
		t.Fatalf("marshal corpus a: %v", err)
	}
	jb, err := json.Marshal(b)
	if err != nil {
		t.Fatalf("marshal corpus b: %v", err)
	}
	if string(ja) != string(jb) {
		t.Fatalf("GenerateCorpus(1346) is not deterministic across repeated invocations")
	}
}

func TestGenerateCorpus_CoverageAndInvariants(t *testing.T) {
	bundle := GenerateCorpus(1346)

	if len(bundle.PrimaryCases) != 100 {
		t.Fatalf("expected 100 primary cases, got %d", len(bundle.PrimaryCases))
	}
	if len(bundle.NoisyRerankerCases) != 36 {
		t.Fatalf("expected 36 noisy reranker cases, got %d", len(bundle.NoisyRerankerCases))
	}
	if len(bundle.ChronologicalCases) != 18 {
		t.Fatalf("expected 18 chronological log cases, got %d", len(bundle.ChronologicalCases))
	}
	if len(bundle.TightBudgetCases) != 15 {
		t.Fatalf("expected 15 tight-budget multi-hop cases, got %d", len(bundle.TightBudgetCases))
	}

	seenLengths := make(map[int]bool)
	seenSlots := make(map[SlotLabel]bool)
	seenSources := make(map[ToolSourceKind]bool)
	catCounts := make(map[string]int)

	for _, c := range bundle.PrimaryCases {
		catCounts[c.Category]++
		if c.Category == "position_sweep" {
			seenLengths[c.ContextLength] = true
			seenSlots[c.GoldSlotLabel] = true
		}
		if c.Scope.TenantID == "" || c.Scope.WorkspaceID == "" {
			t.Errorf("case %s missing tenant/workspace scope", c.ID)
		}
		for _, d := range c.Documents {
			seenSources[d.SourceKind] = true
			if d.TenantID == "" {
				t.Errorf("doc %s in case %s missing tenant tag", d.ID, c.ID)
			}
			if d.Tokens <= 0 {
				t.Errorf("doc %s in case %s has non-positive token count %d", d.ID, c.ID, d.Tokens)
			}
		}
	}

	for _, k := range []int{5, 10, 20, 30} {
		if !seenLengths[k] {
			t.Errorf("position sweep missing context length K=%d", k)
		}
	}
	for _, slot := range AllSingleSlots {
		if !seenSlots[slot] {
			t.Errorf("position sweep missing canonical slot %q", slot)
		}
	}
	for _, src := range AllSourceKinds {
		if !seenSources[src] {
			t.Errorf("corpus missing tool source kind %q", src)
		}
	}

	if catCounts["position_sweep"] != 60 || catCounts["multi_evidence_synthesis"] != 24 || catCounts["security_acl_injection"] != 16 {
		t.Errorf("unexpected primary category counts: %+v", catCounts)
	}
}
