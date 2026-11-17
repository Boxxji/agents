package membench

import (
	"errors"
	"math"
	"sort"
	"strings"
	"time"
)

// PackStrategy controls how candidate evidence is selected and positioned within a prompt budget.
type PackStrategy string

const (
	StrategyEdgeSalience  PackStrategy = "edge_salience"
	StrategyChronological PackStrategy = "chronological"
	StrategyRelevanceDesc PackStrategy = "relevance_desc"
	StrategyMiddleBuried  PackStrategy = "middle_buried"
)

// EvidenceDoc represents a candidate context document or memory snippet to pack.
type EvidenceDoc struct {
	ID         string            `json:"id"`
	TenantID   string            `json:"tenant_id,omitempty"`
	SessionID  string            `json:"session_id,omitempty"`
	SourceID   string            `json:"source_id,omitempty"`
	Content    string            `json:"content"`
	Salience   float64           `json:"salience"`
	Timestamp  time.Time         `json:"timestamp,omitempty"`
	TokenCount int               `json:"token_count,omitempty"`
	IsGold     bool              `json:"is_gold,omitempty"`
	Metadata   map[string]string `json:"metadata,omitempty"`
}

// PackOptions configures tenant ACL filtering, token budget, deduplication, and ordering strategy.
type PackOptions struct {
	Strategy      PackStrategy     `json:"strategy"`
	TokenBudget   int              `json:"token_budget"`
	ReserveTokens int              `json:"reserve_tokens,omitempty"`
	TenantID      string           `json:"tenant_id,omitempty"`
	Query         string           `json:"query,omitempty"`
	Deduplicate   bool             `json:"deduplicate"`
	TokenCounter  func(string) int `json:"-"`
}

// PackedDoc records a document's packed index, position ratio, and effective salience.
type PackedDoc struct {
	Doc               EvidenceDoc `json:"doc"`
	EffectiveSalience float64     `json:"effective_salience"`
	Tokens            int         `json:"tokens"`
	OriginalIndex     int         `json:"original_index"`
	PackedIndex       int         `json:"packed_index"`
	PositionRatio     float64     `json:"position_ratio"`
}

// PackResult summarizes packed documents, strict budget accounting, and U-curve metrics.
type PackResult struct {
	Strategy            PackStrategy  `json:"strategy"`
	Packed              []PackedDoc   `json:"packed"`
	Dropped             []EvidenceDoc `json:"dropped,omitempty"`
	TotalTokens         int           `json:"total_tokens"`
	TokenBudget         int           `json:"token_budget"`
	AvailableBudget     int           `json:"available_budget"`
	BudgetRemaining     int           `json:"budget_remaining"`
	EdgeSalienceScore   float64       `json:"edge_salience_score"`
	MiddleRiskScore     float64       `json:"middle_risk_score"`
	TenantFilteredCount int           `json:"tenant_filtered_count"`
	DeduplicatedCount   int           `json:"deduplicated_count"`
}

type scoredCandidate struct {
	doc        EvidenceDoc
	origIdx    int
	tokens     int
	salience   float64
	dedupeHash string
}

// PackEvidence filters by tenant ACL, deduplicates, fits within TokenBudget - ReserveTokens,
// and positions documents according to opts.Strategy.
func PackEvidence(docs []EvidenceDoc, opts PackOptions) (*PackResult, error) {
	if opts.TokenBudget < 0 {
		return nil, errors.New("membench: token_budget cannot be negative")
	}
	if opts.ReserveTokens < 0 {
		return nil, errors.New("membench: reserve_tokens cannot be negative")
	}
	strategy := opts.Strategy
	if strategy == "" {
		strategy = StrategyEdgeSalience
	}
	switch strategy {
	case StrategyEdgeSalience, StrategyChronological, StrategyRelevanceDesc, StrategyMiddleBuried:
	default:
		return nil, errors.New("membench: unsupported pack strategy")
	}

	counter := opts.TokenCounter
	if counter == nil {
		counter = estimateTokens
	}

	available := opts.TokenBudget - opts.ReserveTokens
	if available < 0 {
		available = 0
	}

	queryTokens := tokenizeWords(opts.Query)
	res := &PackResult{
		Strategy:        strategy,
		TokenBudget:     opts.TokenBudget,
		AvailableBudget: available,
		BudgetRemaining: available,
	}

	candidates := make([]scoredCandidate, 0, len(docs))
	for i, doc := range docs {
		// Apply tenant isolation BEFORE scoring or ranking.
		if opts.TenantID != "" && doc.TenantID != opts.TenantID {
			res.TenantFilteredCount++
			continue
		}
		toks := doc.TokenCount
		if toks <= 0 {
			toks = counter(doc.Content)
		}
		sal := computeEffectiveSalience(doc, queryTokens)
		candidates = append(candidates, scoredCandidate{
			doc:        doc,
			origIdx:    i,
			tokens:     toks,
			salience:   sal,
			dedupeHash: normalizeContentKey(doc.Content),
		})
	}

	if opts.Deduplicate && len(candidates) > 1 {
		// Keep highest-salience copy per normalized content.
		bestByKey := make(map[string]int, len(candidates))
		deduped := make([]scoredCandidate, 0, len(candidates))
		for _, c := range candidates {
			if idx, exists := bestByKey[c.dedupeHash]; exists {
				res.DeduplicatedCount++
				if c.salience > deduped[idx].salience {
					deduped[idx] = c
				}
				continue
			}
			bestByKey[c.dedupeHash] = len(deduped)
			deduped = append(deduped, c)
		}
		candidates = deduped
	}

	if strategy == StrategyChronological {
		sort.SliceStable(candidates, func(i, j int) bool {
			if !candidates[i].doc.Timestamp.Equal(candidates[j].doc.Timestamp) {
				return candidates[i].doc.Timestamp.Before(candidates[j].doc.Timestamp)
			}
			return candidates[i].origIdx < candidates[j].origIdx
		})
	} else {
		sort.SliceStable(candidates, func(i, j int) bool {
			if candidates[i].salience != candidates[j].salience {
				return candidates[i].salience > candidates[j].salience
			}
			if !candidates[i].doc.Timestamp.Equal(candidates[j].doc.Timestamp) {
				return candidates[i].doc.Timestamp.After(candidates[j].doc.Timestamp)
			}
			return candidates[i].doc.ID < candidates[j].doc.ID
		})
	}

	selected := make([]scoredCandidate, 0, len(candidates))
	usedTokens := 0
	for _, c := range candidates {
		if c.tokens <= available-usedTokens {
			selected = append(selected, c)
			usedTokens += c.tokens
		} else {
			res.Dropped = append(res.Dropped, c.doc)
		}
	}

	ordered := reorderSelected(selected, strategy)
	res.Packed = make([]PackedDoc, len(ordered))
	var weightedSal, totalSal float64
	for idx, c := range ordered {
		ratio := 0.0
		if len(ordered) > 1 {
			ratio = float64(idx) / float64(len(ordered)-1)
		}
		res.Packed[idx] = PackedDoc{
			Doc:               c.doc,
			EffectiveSalience: c.salience,
			Tokens:            c.tokens,
			OriginalIndex:     c.origIdx,
			PackedIndex:       idx,
			PositionRatio:     ratio,
		}
		att := edgeAttentionWeight(ratio)
		salWeight := c.salience
		if salWeight < 0 {
			salWeight = 0
		}
		weightedSal += salWeight * att
		totalSal += salWeight
	}

	res.TotalTokens = usedTokens
	res.BudgetRemaining = available - usedTokens
	if totalSal > 0 {
		res.EdgeSalienceScore = weightedSal / totalSal
		res.MiddleRiskScore = 1.0 - res.EdgeSalienceScore
	}
	return res, nil
}

func reorderSelected(selected []scoredCandidate, strategy PackStrategy) []scoredCandidate {
	n := len(selected)
	if n <= 1 {
		return selected
	}
	switch strategy {
	case StrategyChronological, StrategyRelevanceDesc:
		return selected
	case StrategyEdgeSalience:
		// Interleave highest-salience items onto primacy (0, 1, ...) and recency (n-1, n-2, ...) edges.
		out := make([]scoredCandidate, n)
		left, right := 0, n-1
		for i, item := range selected {
			if i%2 == 0 {
				out[left] = item
				left++
			} else {
				out[right] = item
				right--
			}
		}
		return out
	case StrategyMiddleBuried:
		// Place highest-salience items at the center to model worst-case middle degradation.
		out := make([]scoredCandidate, n)
		mid := n / 2
		out[mid] = selected[0]
		left, right := mid-1, mid+1
		for i := 1; i < n; i++ {
			if i%2 == 1 && left >= 0 {
				out[left] = selected[i]
				left--
			} else if right < n {
				out[right] = selected[i]
				right++
			} else if left >= 0 {
				out[left] = selected[i]
				left--
			}
		}
		return out
	default:
		return selected
	}
}

func computeEffectiveSalience(doc EvidenceDoc, queryTokens map[string]struct{}) float64 {
	sal := doc.Salience
	if len(queryTokens) == 0 {
		return sal
	}
	docTokens := tokenizeWords(doc.Content)
	if len(docTokens) == 0 {
		return sal
	}
	overlap := 0
	for tok := range queryTokens {
		if _, ok := docTokens[tok]; ok {
			overlap++
		}
	}
	lexical := float64(overlap) / float64(len(queryTokens))
	if sal == 0 {
		return lexical
	}
	return sal + 0.35*lexical
}

func tokenizeWords(text string) map[string]struct{} {
	fields := strings.Fields(strings.ToLower(text))
	if len(fields) == 0 {
		return nil
	}
	out := make(map[string]struct{}, len(fields))
	for _, f := range fields {
		cleaned := strings.Trim(f, ".,:;!?\"'()[]{}")
		if len(cleaned) >= 2 {
			out[cleaned] = struct{}{}
		}
	}
	return out
}

func normalizeContentKey(content string) string {
	return strings.Join(strings.Fields(strings.ToLower(content)), " ")
}

// edgeAttentionWeight models the TACL '24 U-shaped attention profile:
// 1.0 at edges (ratio 0.0 and 1.0) and 0.35 at the middle (ratio 0.5).
func edgeAttentionWeight(posRatio float64) float64 {
	if posRatio < 0 {
		posRatio = 0
	}
	if posRatio > 1 {
		posRatio = 1
	}
	return 1.0 - 0.65*math.Sin(math.Pi*posRatio)
}
