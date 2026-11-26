// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

package contextpack

import (
	"fmt"
	"sort"
	"strings"
	"time"
	"unicode"
)

// injectionMarkers lists untrusted directive patterns that embedded tool outputs
// may attempt to use to hijack packing priority, relevance ranking, or ACL checks.
var injectionMarkers = []string{
	"ignore previous instructions",
	"system_override",
	"priority_score=",
	"rank_first=",
	"admin system prompt",
	"<|im_start|>system",
	"tool_meta_inject",
	"force_top_rank=",
	"bypass_budget=",
	"ignore tenant acl",
}

type indexedDoc struct {
	doc        ToolDocument
	origIndex  int
	cleanScore float64
	hadInject  bool
}

// PackContext executes one of the 5 matched context-packing policies on an EvalCase.
// Selection and ordering NEVER read doc.IsGold or doc.EvidenceFactID (zero oracle leakage);
// those ground-truth labels are inspected solely post-hoc to compute structural audit metrics.
func PackContext(evalCase EvalCase, policy PolicyID, cfg PackConfig) (PackedContext, error) {
	startTime := time.Now()

	// 1. Pre-packing tenant & ACL boundary enforcement (0 cross-tenant leaks invariant).
	authorized := make([]indexedDoc, 0, len(evalCase.Documents))
	var filteredUnauth []string
	var sanitizedIDs []string
	rawAuthorizedTokens := 0

	for idx, rawDoc := range evalCase.Documents {
		if !evalCase.Scope.CanAccess(rawDoc.TenantID, rawDoc.WorkspaceID, rawDoc.RequiredRole) {
			filteredUnauth = append(filteredUnauth, rawDoc.ID)
			continue
		}
		doc := cloneDoc(rawDoc)
		hadInject := containsInjectionDirective(doc.Content) || doc.InjectionPayload
		if hadInject && cfg.SanitizeInjections {
			doc.Content = sanitizeContent(doc.Content)
			doc.Tokens = EstimateTokens(doc.Content)
			sanitizedIDs = append(sanitizedIDs, doc.ID)
		}
		if doc.Tokens <= 0 {
			doc.Tokens = EstimateTokens(doc.Content)
		}
		rawAuthorizedTokens += doc.Tokens
		authorized = append(authorized, indexedDoc{
			doc:        doc,
			origIndex:  idx,
			cleanScore: clamp01(doc.BaseRelevance),
			hadInject:  hadInject,
		})
	}

	var selected []indexedDoc
	var dedupedIDs []string
	chronoLockApplied := false

	switch policy {
	case PolicyRawConcatBaseline:
		selected = append([]indexedDoc(nil), authorized...)

	case PolicyRelevanceDesc:
		selected = sortByScoreDesc(authorized)

	case PolicyUShapedEdgePacking:
		ranked := sortByScoreDesc(authorized)
		selected = applyUShapedEdges(ranked)

	case PolicyBudgetedTopKPrune:
		ranked := sortByScoreDesc(authorized)
		selected = greedyBudgetPack(ranked, cfg.TokenBudget)

	case PolicyKaiEvidenceEdge:
		selected, dedupedIDs, chronoLockApplied = packKaiEvidenceEdge(evalCase, authorized, cfg)

	default:
		return PackedContext{}, fmt.Errorf("unknown context packing policy: %q", policy)
	}

	latencyMicros := time.Since(startTime).Microseconds()
	if latencyMicros < 0 {
		latencyMicros = 0
	}

	return finalizePackedContext(
		evalCase,
		policy,
		cfg,
		authorized,
		selected,
		rawAuthorizedTokens,
		filteredUnauth,
		sanitizedIDs,
		dedupedIDs,
		chronoLockApplied,
		latencyMicros,
	), nil
}

// packKaiEvidenceEdge implements Kai's structured, non-generative context-packing policy
// (GenerativeOutputTokens == 0):
//  1. Deduplicates near-identical tool chunks via token-set Jaccard similarity.
//  2. Respects chronological order groups when EnforceChronologicalOrder is active.
//  3. Applies diversity-preserving budgeted selection across tool sources.
//  4. Places non-order-locked evidence at U-shaped start/end edges to mitigate middle degradation.
func packKaiEvidenceEdge(evalCase EvalCase, authorized []indexedDoc, cfg PackConfig) ([]indexedDoc, []string, bool) {
	ranked := sortByScoreDesc(authorized)

	// Step A: Near-duplicate elimination (skip collapsing members of the same chronological order group).
	dedupThreshold := cfg.DedupeJaccardThreshold
	if dedupThreshold <= 0 {
		dedupThreshold = 0.72
	}
	unique := make([]indexedDoc, 0, len(ranked))
	uniqueTokenSets := make([]map[string]struct{}, 0, len(ranked))
	var dedupedIDs []string

	for _, cand := range ranked {
		tset := contentTokenSet(cand.doc.Content)
		isDup := false
		for i, kept := range unique {
			// Never collapse distinct sequence steps of an ordered log/diff group.
			if cand.doc.RequiresOrder && kept.doc.RequiresOrder &&
				cand.doc.OrderGroupID != "" && cand.doc.OrderGroupID == kept.doc.OrderGroupID &&
				cand.doc.SequenceIndex != kept.doc.SequenceIndex {
				continue
			}
			if jaccardSimilarity(tset, uniqueTokenSets[i]) >= dedupThreshold {
				isDup = true
				break
			}
		}
		if isDup {
			dedupedIDs = append(dedupedIDs, cand.doc.ID)
			continue
		}
		unique = append(unique, cand)
		uniqueTokenSets = append(uniqueTokenSets, tset)
	}

	// Step B: Check whether chronological ordering lock should be applied.
	needsChrono := false
	if cfg.EnforceChronologicalOrder {
		if evalCase.PreserveChronologicalOrder {
			needsChrono = true
		} else {
			for _, c := range unique {
				if c.doc.RequiresOrder {
					needsChrono = true
					break
				}
			}
		}
	}

	if needsChrono {
		return packWithChronologicalLock(unique, cfg.TokenBudget), dedupedIDs, true
	}

	// Step C: Diversity-preserving budgeted selection.
	chosen := make([]indexedDoc, 0, len(unique))
	chosenSet := make(map[string]bool, len(unique))
	usedTokens := 0

	if cfg.DiversityMinPerSource > 0 {
		perSource := make(map[ToolSourceKind]int)
		// Pass 1: reserve high-signal evidence (cleanScore >= 0.70) across distinct tool sources first.
		for _, cand := range unique {
			if cand.cleanScore < 0.70 {
				continue
			}
			if perSource[cand.doc.SourceKind] >= cfg.DiversityMinPerSource {
				continue
			}
			if cfg.TokenBudget > 0 && usedTokens+cand.doc.Tokens > cfg.TokenBudget {
				continue
			}
			chosen = append(chosen, cand)
			chosenSet[cand.doc.ID] = true
			perSource[cand.doc.SourceKind]++
			usedTokens += cand.doc.Tokens
		}
	}

	// Pass 2: greedily fill remaining token budget in descending relevance order.
	for _, cand := range unique {
		if chosenSet[cand.doc.ID] {
			continue
		}
		if cfg.TokenBudget > 0 && usedTokens+cand.doc.Tokens > cfg.TokenBudget {
			continue
		}
		chosen = append(chosen, cand)
		chosenSet[cand.doc.ID] = true
		usedTokens += cand.doc.Tokens
	}

	// Re-rank chosen set descending before U-shaped edge interleaving so rank 1 -> start, rank 2 -> end.
	chosenRanked := sortByScoreDesc(chosen)
	return applyUShapedEdges(chosenRanked), dedupedIDs, false
}

// packWithChronologicalLock keeps causally ordered log/diff groups intact in strictly
// ascending SequenceIndex order, prioritizing the highest-relevance ordered group under budget.
func packWithChronologicalLock(unique []indexedDoc, tokenBudget int) []indexedDoc {
	groups := make(map[string][]indexedDoc)
	var nonGrouped []indexedDoc
	for _, c := range unique {
		if c.doc.RequiresOrder && c.doc.OrderGroupID != "" {
			groups[c.doc.OrderGroupID] = append(groups[c.doc.OrderGroupID], c)
		} else {
			nonGrouped = append(nonGrouped, c)
		}
	}

	type scoredGroup struct {
		id       string
		maxScore float64
		members  []indexedDoc
	}
	var orderedGroups []scoredGroup
	for gid, members := range groups {
		sort.SliceStable(members, func(i, j int) bool {
			if members[i].doc.SequenceIndex != members[j].doc.SequenceIndex {
				return members[i].doc.SequenceIndex < members[j].doc.SequenceIndex
			}
			return members[i].origIndex < members[j].origIndex
		})
		maxS := 0.0
		for _, m := range members {
			if m.cleanScore > maxS {
				maxS = m.cleanScore
			}
		}
		orderedGroups = append(orderedGroups, scoredGroup{
			id:       gid,
			maxScore: maxS,
			members:  members,
		})
	}
	sort.SliceStable(orderedGroups, func(i, j int) bool {
		if orderedGroups[i].maxScore != orderedGroups[j].maxScore {
			return orderedGroups[i].maxScore > orderedGroups[j].maxScore
		}
		return orderedGroups[i].id < orderedGroups[j].id
	})

	usedTokens := 0
	var lockedPrefix []indexedDoc
	for _, grp := range orderedGroups {
		for _, m := range grp.members {
			if tokenBudget > 0 && usedTokens+m.Tokens() > tokenBudget {
				continue
			}
			lockedPrefix = append(lockedPrefix, m)
			usedTokens += m.Tokens()
		}
	}

	// Fill any leftover token budget only with high-signal non-grouped context (>= 0.70)
	// so irrelevant distractors do not dilute or push the locked sequential trace into the middle valley.
	nonGroupedRanked := sortByScoreDesc(nonGrouped)
	var trailing []indexedDoc
	for _, c := range nonGroupedRanked {
		if c.cleanScore < 0.70 {
			continue
		}
		if tokenBudget > 0 && usedTokens+c.doc.Tokens > tokenBudget {
			continue
		}
		trailing = append(trailing, c)
		usedTokens += c.doc.Tokens
	}

	out := make([]indexedDoc, 0, len(lockedPrefix)+len(trailing))
	out = append(out, lockedPrefix...)
	out = append(out, trailing...)
	return out
}

func (d indexedDoc) Tokens() int {
	return d.doc.Tokens
}

func sortByScoreDesc(docs []indexedDoc) []indexedDoc {
	out := append([]indexedDoc(nil), docs...)
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].cleanScore != out[j].cleanScore {
			return out[i].cleanScore > out[j].cleanScore
		}
		return out[i].origIndex < out[j].origIndex
	})
	return out
}

// applyUShapedEdges places highest-ranked documents alternately at the start and end edges:
// rank 0 -> slot[0], rank 1 -> slot[n-1], rank 2 -> slot[1], rank 3 -> slot[n-2], ...
// pushing lowest-relevance distractors into the middle valley (Liu et al., TACL 2024).
func applyUShapedEdges(ranked []indexedDoc) []indexedDoc {
	n := len(ranked)
	if n <= 1 {
		return append([]indexedDoc(nil), ranked...)
	}
	placed := make([]indexedDoc, n)
	left := 0
	right := n - 1
	for i, item := range ranked {
		if i%2 == 0 {
			placed[left] = item
			left++
		} else {
			placed[right] = item
			right--
		}
	}
	return placed
}

func greedyBudgetPack(ranked []indexedDoc, tokenBudget int) []indexedDoc {
	if tokenBudget <= 0 {
		return append([]indexedDoc(nil), ranked...)
	}
	selected := make([]indexedDoc, 0, len(ranked))
	used := 0
	for _, item := range ranked {
		if used+item.doc.Tokens > tokenBudget {
			continue
		}
		selected = append(selected, item)
		used += item.doc.Tokens
	}
	return selected
}

func finalizePackedContext(
	evalCase EvalCase,
	policy PolicyID,
	cfg PackConfig,
	authorized []indexedDoc,
	selected []indexedDoc,
	rawAuthorizedTokens int,
	filteredUnauth []string,
	sanitizedIDs []string,
	dedupedIDs []string,
	chronoLockApplied bool,
	latencyMicros int64,
) PackedContext {
	orderedDocs := make([]ToolDocument, len(selected))
	selectedSet := make(map[string]bool, len(selected))
	selectedIDs := make([]string, len(selected))
	totalTokens := 0
	crossTenantLeaks := 0

	var rendered strings.Builder
	for i, item := range selected {
		orderedDocs[i] = item.doc
		selectedSet[item.doc.ID] = true
		selectedIDs[i] = item.doc.ID
		totalTokens += item.doc.Tokens
		if !evalCase.Scope.CanAccess(item.doc.TenantID, item.doc.WorkspaceID, item.doc.RequiredRole) {
			crossTenantLeaks++
		}
		fmt.Fprintf(&rendered, "[DOC %d | id=%s | source=%s | uri=%s]\n%s\n\n",
			i+1, item.doc.ID, item.doc.SourceKind, item.doc.SourceURI, item.doc.Content)
	}

	var droppedIDs []string
	for _, auth := range authorized {
		if !selectedSet[auth.doc.ID] {
			droppedIDs = append(droppedIDs, auth.doc.ID)
		}
	}

	// Post-hoc structural ground-truth audit (never used during packing decisions!).
	reqFactSet := make(map[string]bool, len(evalCase.RequiredFactIDs))
	for _, fid := range evalCase.RequiredFactIDs {
		reqFactSet[fid] = true
	}
	retainedFacts := make(map[string]bool, len(evalCase.RequiredFactIDs))
	var goldSlotIndices []int
	var goldPositionsNorm []float64
	n := len(orderedDocs)
	edgeFrac := cfg.EdgeWindowFraction
	if edgeFrac <= 0 || edgeFrac >= 0.5 {
		edgeFrac = 0.25
	}

	allGoldInEdge := true
	for idx, doc := range orderedDocs {
		if doc.IsGold && (len(reqFactSet) == 0 || reqFactSet[doc.EvidenceFactID]) {
			if doc.EvidenceFactID != "" {
				retainedFacts[doc.EvidenceFactID] = true
			}
			goldSlotIndices = append(goldSlotIndices, idx)
			normPos := normalizedPosition(idx, n)
			goldPositionsNorm = append(goldPositionsNorm, normPos)
			if !isInEdgeWindow(normPos, edgeFrac) {
				allGoldInEdge = false
			}
		}
	}

	reqCount := len(evalCase.RequiredFactIDs)
	if reqCount == 0 {
		reqCount = evalCase.RequiredEvidenceCount
	}
	if reqCount < 1 {
		reqCount = 1
	}
	foundCount := len(retainedFacts)
	if len(evalCase.RequiredFactIDs) == 0 {
		foundCount = len(goldSlotIndices)
	}
	goldRecall := float64(foundCount) / float64(reqCount)
	if goldRecall > 1.0 {
		goldRecall = 1.0
	}
	allGoldRetained := foundCount >= reqCount
	goldInEdgeWindow := allGoldRetained && len(goldSlotIndices) > 0 && allGoldInEdge

	// Audit chronological order violations for required sequential groups.
	orderViolations := countOrderViolations(evalCase, orderedDocs)

	// Audit injection-induced rank hijacks:
	// An injection hijack occurs if any injected document beats a higher-BaseRelevance non-injected
	// document in rank-priority ordering (for ranked/pruned policies).
	injectionHijacks := countInjectionRankHijacks(policy, selected)

	tokenSavings := 0.0
	if rawAuthorizedTokens > 0 {
		tokenSavings = 1.0 - (float64(totalTokens) / float64(rawAuthorizedTokens))
		if tokenSavings < 0 {
			tokenSavings = 0
		}
	}

	// Enso USD input token cost + Kai local structured compute overhead (for Kai policy only).
	inputCostUSD := (float64(totalTokens) / 1_000_000.0) * cfg.EnsoInputUSDPer1MTokens
	if policy == PolicyKaiEvidenceEdge {
		inputCostUSD += cfg.KaiDecisionUSDPer1kCalls / 1000.0
	}

	return PackedContext{
		PolicyID:                policy,
		OrderedDocs:             orderedDocs,
		RenderedText:            rendered.String(),
		TotalInputTokens:        totalTokens,
		RawAuthorizedTokens:     rawAuthorizedTokens,
		TokenSavingsRatio:       tokenSavings,
		GoldSlotIndices:         goldSlotIndices,
		GoldPositionsNormalized: goldPositionsNorm,
		GoldInEdgeWindow:        goldInEdgeWindow,
		AllGoldRetained:         allGoldRetained,
		GoldRecall:              goldRecall,
		CrossTenantLeaks:        crossTenantLeaks,
		InjectionRankHijacks:    injectionHijacks,
		OrderViolations:         orderViolations,
		EstimatedInputCostUSD:   inputCostUSD,
		KaiDecision: KaiDecisionRecord{
			PolicyID:                 policy,
			SelectedDocIDs:           selectedIDs,
			DroppedDocIDs:            droppedIDs,
			FilteredUnauthorizedIDs:  filteredUnauth,
			SanitizedInjectionIDs:    sanitizedIDs,
			DeduplicatedIDs:          dedupedIDs,
			ChronologicalLockApplied: chronoLockApplied,
			GenerativeOutputTokens:   0, // Strict architectural invariant: Kai is non-generative.
			DecisionLatencyMicros:    latencyMicros,
		},
	}
}

func countOrderViolations(evalCase EvalCase, orderedDocs []ToolDocument) int {
	if !evalCase.PreserveChronologicalOrder {
		return 0
	}
	lastSeqByGroup := make(map[string]int)
	seenByGroup := make(map[string]bool)
	violations := 0
	for _, doc := range orderedDocs {
		if !doc.RequiresOrder || doc.OrderGroupID == "" {
			continue
		}
		if seenByGroup[doc.OrderGroupID] && doc.SequenceIndex < lastSeqByGroup[doc.OrderGroupID] {
			violations++
		}
		lastSeqByGroup[doc.OrderGroupID] = doc.SequenceIndex
		seenByGroup[doc.OrderGroupID] = true
	}
	return violations
}

func countInjectionRankHijacks(policy PolicyID, selected []indexedDoc) int {
	if policy == PolicyRawConcatBaseline || len(selected) <= 1 {
		return 0
	}
	// Verify that no prompt-injected document received an artificial boost to slot 0 or slot N-1
	// ahead of strictly higher-BaseRelevance benign documents.
	n := len(selected)
	hijacks := 0
	for i, item := range selected {
		if !item.hadInject {
			continue
		}
		// Compute edge priority distance: 0 at edges (0, n-1), larger toward middle.
		dist := i
		if policy == PolicyUShapedEdgePacking || policy == PolicyKaiEvidenceEdge {
			if rightDist := n - 1 - i; rightDist < dist {
				dist = rightDist
			}
		}
		if dist == 0 {
			// Check if any benign doc has strictly higher cleanScore (> +0.05 margin) yet sits further inside.
			for j, other := range selected {
				if other.hadInject || other.cleanScore <= item.cleanScore+0.05 {
					continue
				}
				otherDist := j
				if policy == PolicyUShapedEdgePacking || policy == PolicyKaiEvidenceEdge {
					if rd := n - 1 - j; rd < otherDist {
						otherDist = rd
					}
				}
				if otherDist > dist {
					hijacks++
					break
				}
			}
		}
	}
	return hijacks
}

func normalizedPosition(idx, total int) float64 {
	if total <= 1 {
		return 0.0
	}
	return float64(idx) / float64(total-1)
}

func isInEdgeWindow(normPos, edgeFrac float64) bool {
	return normPos <= edgeFrac+1e-9 || normPos >= (1.0-edgeFrac)-1e-9
}

func containsInjectionDirective(content string) bool {
	lower := strings.ToLower(content)
	for _, marker := range injectionMarkers {
		if strings.Contains(lower, marker) {
			return true
		}
	}
	return false
}

func sanitizeContent(content string) string {
	lines := strings.Split(content, "\n")
	cleaned := make([]string, 0, len(lines))
	for _, line := range lines {
		if containsInjectionDirective(line) {
			cleaned = append(cleaned, "[REDACTED_INJECTION_DIRECTIVE]")
		} else {
			cleaned = append(cleaned, line)
		}
	}
	return strings.Join(cleaned, "\n")
}

func contentTokenSet(text string) map[string]struct{} {
	set := make(map[string]struct{})
	var cur strings.Builder
	flush := func() {
		if cur.Len() >= 3 {
			set[cur.String()] = struct{}{}
		}
		cur.Reset()
	}
	for _, r := range strings.ToLower(text) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' {
			cur.WriteRune(r)
		} else {
			flush()
		}
	}
	flush()
	return set
}

func jaccardSimilarity(a, b map[string]struct{}) float64 {
	if len(a) == 0 && len(b) == 0 {
		return 1.0
	}
	if len(a) == 0 || len(b) == 0 {
		return 0.0
	}
	inter := 0
	for k := range a {
		if _, ok := b[k]; ok {
			inter++
		}
	}
	union := len(a) + len(b) - inter
	if union == 0 {
		return 0.0
	}
	return float64(inter) / float64(union)
}

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func cloneDoc(d ToolDocument) ToolDocument {
	cp := d
	if d.Metadata != nil {
		cp.Metadata = make(map[string]string, len(d.Metadata))
		for k, v := range d.Metadata {
			cp.Metadata[k] = v
		}
	}
	return cp
}
