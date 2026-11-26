package swebench

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"time"
)

// BackendSummary aggregates task resolution, timing, cost, and IO metrics for one backend.
type BackendSummary struct {
	Backend            StorageBackend `json:"backend"`
	TotalTasks         int            `json:"total_tasks"`
	PassedTasks        int            `json:"passed_tasks"`
	BlockedTasks       int            `json:"blocked_tasks"`
	TamperedTasks      int            `json:"tampered_tasks,omitempty"`
	FailedTasks        int            `json:"failed_tasks"`
	PassRate           float64        `json:"pass_rate"`
	TotalWallTimeMS    int64          `json:"total_wall_time_ms"`
	MeanWallTimeMS     float64        `json:"mean_wall_time_ms"`
	P50WallTimeMS      int64          `json:"p50_wall_time_ms"`
	P95WallTimeMS      int64          `json:"p95_wall_time_ms"`
	P99WallTimeMS      int64          `json:"p99_wall_time_ms"`
	TotalTokensIn      int64          `json:"total_tokens_in"`
	TotalTokensOut     int64          `json:"total_tokens_out"`
	TotalTokensCached  int64          `json:"total_tokens_cached"`
	TotalModelCostUSD  float64        `json:"total_model_cost_usd"`
	MeanModelCostUSD   float64        `json:"mean_model_cost_usd"`
	CostPerResolvedUSD float64        `json:"cost_per_resolved_usd,omitempty"`
	AggregateIO        IOCounters     `json:"aggregate_io"`
	ReadThroughputMBs  float64        `json:"read_throughput_mb_s"`
	WriteThroughputMB  float64        `json:"write_throughput_mb_s"`
}

// VFSvsNVMeDelta captures comparative ratios and deltas of vfs against plain NVMe.
type VFSvsNVMeDelta struct {
	PassRateDelta     float64 `json:"pass_rate_delta"`
	WallTimeRatio     float64 `json:"wall_time_ratio"`
	CostDeltaUSD      float64 `json:"cost_delta_usd"`
	ReadOpsDelta      int64   `json:"read_ops_delta"`
	WriteOpsDelta     int64   `json:"write_ops_delta"`
	MetadataOpsDelta  int64   `json:"metadata_ops_delta"`
	BytesReadDelta    int64   `json:"bytes_read_delta"`
	BytesWrittenDelta int64   `json:"bytes_written_delta"`
	ReadLatencyRatio  float64 `json:"read_latency_ratio"`
	WriteLatencyRatio float64 `json:"write_latency_ratio"`
}

// ComparisonReport captures a reproducible SWE-bench Pro V2 evaluation across storage backends.
type ComparisonReport struct {
	SubsetName       string                            `json:"subset_name"`
	Revision         string                            `json:"revision"`
	BenchmarkVersion string                            `json:"benchmark_version"`
	ManifestDigest   string                            `json:"manifest_digest"`
	GeneratedAt      time.Time                         `json:"generated_at"`
	Summaries        map[StorageBackend]BackendSummary `json:"summaries"`
	VFSvsNVMe        *VFSvsNVMeDelta                   `json:"vfs_vs_nvme,omitempty"`
	Outcomes         []TaskOutcome                     `json:"outcomes"`
}

// BuildComparisonReport aggregates raw task outcomes into a backend comparison report.
func BuildComparisonReport(spec *SubsetSpec, outcomes []TaskOutcome) (*ComparisonReport, error) {
	if err := spec.Validate(); err != nil {
		return nil, err
	}

	grouped := make(map[StorageBackend][]TaskOutcome)
	for _, out := range outcomes {
		grouped[out.Backend] = append(grouped[out.Backend], out)
	}

	summaries := make(map[StorageBackend]BackendSummary, len(grouped))
	for backend, list := range grouped {
		summaries[backend] = summarizeBackend(backend, list)
	}

	report := &ComparisonReport{
		SubsetName:       spec.Name,
		Revision:         spec.Revision,
		BenchmarkVersion: spec.BenchmarkVersion,
		ManifestDigest:   spec.ManifestDigest,
		GeneratedAt:      time.Now().UTC(),
		Summaries:        summaries,
		Outcomes:         append([]TaskOutcome(nil), outcomes...),
	}

	vfsSummary, hasVFS := summaries[BackendVFS]
	nvmeSummary, hasNVMe := summaries[BackendNVMe]
	if hasVFS && hasNVMe {
		report.VFSvsNVMe = &VFSvsNVMeDelta{
			PassRateDelta:     vfsSummary.PassRate - nvmeSummary.PassRate,
			WallTimeRatio:     safeRatio(float64(vfsSummary.TotalWallTimeMS), float64(nvmeSummary.TotalWallTimeMS)),
			CostDeltaUSD:      vfsSummary.TotalModelCostUSD - nvmeSummary.TotalModelCostUSD,
			ReadOpsDelta:      vfsSummary.AggregateIO.ReadOps - nvmeSummary.AggregateIO.ReadOps,
			WriteOpsDelta:     vfsSummary.AggregateIO.WriteOps - nvmeSummary.AggregateIO.WriteOps,
			MetadataOpsDelta:  vfsSummary.AggregateIO.MetadataOps - nvmeSummary.AggregateIO.MetadataOps,
			BytesReadDelta:    vfsSummary.AggregateIO.BytesRead - nvmeSummary.AggregateIO.BytesRead,
			BytesWrittenDelta: vfsSummary.AggregateIO.BytesWritten - nvmeSummary.AggregateIO.BytesWritten,
			ReadLatencyRatio:  safeRatio(float64(vfsSummary.AggregateIO.ReadLatencyNS), float64(nvmeSummary.AggregateIO.ReadLatencyNS)),
			WriteLatencyRatio: safeRatio(float64(vfsSummary.AggregateIO.WriteLatencyNS), float64(nvmeSummary.AggregateIO.WriteLatencyNS)),
		}
	}

	return report, nil
}

// FormatMarkdown renders a reproducible Markdown comparison summary for HIP/issue reporting.
func (r *ComparisonReport) FormatMarkdown() string {
	if r == nil {
		return ""
	}
	backends := make([]StorageBackend, 0, len(r.Summaries))
	for b := range r.Summaries {
		backends = append(backends, b)
	}
	sort.Slice(backends, func(i, j int) bool { return backends[i] < backends[j] })

	var b strings.Builder
	_, _ = fmt.Fprintf(&b, "### SWE-bench Pro V2 (%s / %s) — %s\n", r.Revision, r.BenchmarkVersion, r.SubsetName)
	_, _ = fmt.Fprintf(&b, "- Manifest Digest: `%s`\n\n", r.ManifestDigest)
	b.WriteString("| Backend | Passed / Total | Pass Rate | Wall Time p50 / p95 (ms) | Total Cost (USD) | Read / Write / Meta Ops | Bytes Read / Written |\n")
	b.WriteString("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
	for _, backend := range backends {
		s := r.Summaries[backend]
		_, _ = fmt.Fprintf(&b, "| `%s` | %d / %d | %.2f%% | %d / %d | $%.4f | %d / %d / %d | %d / %d |\n",
			s.Backend,
			s.PassedTasks, s.TotalTasks, s.PassRate*100,
			s.P50WallTimeMS, s.P95WallTimeMS,
			s.TotalModelCostUSD,
			s.AggregateIO.ReadOps, s.AggregateIO.WriteOps, s.AggregateIO.MetadataOps,
			s.AggregateIO.BytesRead, s.AggregateIO.BytesWritten,
		)
	}
	if r.VFSvsNVMe != nil {
		d := r.VFSvsNVMe
		_, _ = fmt.Fprintf(&b, "\n- `vfs` vs `nvme`: pass_rate_delta=%+.2f%%, wall_time_ratio=%.3fx, cost_delta=$%+.4f, metadata_ops_delta=%+d\n",
			d.PassRateDelta*100, d.WallTimeRatio, d.CostDeltaUSD, d.MetadataOpsDelta)
	}
	return b.String()
}

func summarizeBackend(backend StorageBackend, outcomes []TaskOutcome) BackendSummary {
	s := BackendSummary{
		Backend:     backend,
		TotalTasks:  len(outcomes),
		AggregateIO: IOCounters{Backend: backend},
	}
	if len(outcomes) == 0 {
		return s
	}

	wallTimes := make([]int64, 0, len(outcomes))
	for _, o := range outcomes {
		if o.Passed {
			s.PassedTasks++
		} else if o.Status == StatusTampered || o.Tampered {
			s.TamperedTasks++
			s.BlockedTasks++
		} else if o.Status == StatusLeakageBlocked || o.Status == StatusPolicyBlocked {
			s.BlockedTasks++
		} else {
			s.FailedTasks++
		}
		s.TotalWallTimeMS += o.WallTimeMS
		wallTimes = append(wallTimes, o.WallTimeMS)
		s.TotalTokensIn += o.ModelTokensIn
		s.TotalTokensOut += o.ModelTokensOut
		s.TotalTokensCached += o.ModelTokensCached
		s.TotalModelCostUSD += o.ModelCostUSD
		s.AggregateIO.Add(o.IO)
	}

	sort.Slice(wallTimes, func(i, j int) bool { return wallTimes[i] < wallTimes[j] })
	n := float64(len(outcomes))
	s.PassRate = float64(s.PassedTasks) / n
	s.MeanWallTimeMS = float64(s.TotalWallTimeMS) / n
	s.P50WallTimeMS = percentileInt64(wallTimes, 0.50)
	s.P95WallTimeMS = percentileInt64(wallTimes, 0.95)
	s.P99WallTimeMS = percentileInt64(wallTimes, 0.99)
	s.MeanModelCostUSD = s.TotalModelCostUSD / n
	if s.PassedTasks > 0 {
		s.CostPerResolvedUSD = s.TotalModelCostUSD / float64(s.PassedTasks)
	}

	if s.AggregateIO.ReadLatencyNS > 0 {
		sec := float64(s.AggregateIO.ReadLatencyNS) / 1e9
		s.ReadThroughputMBs = (float64(s.AggregateIO.BytesRead) / (1024 * 1024)) / sec
	}
	if s.AggregateIO.WriteLatencyNS > 0 {
		sec := float64(s.AggregateIO.WriteLatencyNS) / 1e9
		s.WriteThroughputMB = (float64(s.AggregateIO.BytesWritten) / (1024 * 1024)) / sec
	}
	return s
}

func percentileInt64(sorted []int64, p float64) int64 {
	if len(sorted) == 0 {
		return 0
	}
	if p <= 0 {
		return sorted[0]
	}
	if p >= 1 {
		return sorted[len(sorted)-1]
	}
	idx := int(math.Ceil(p*float64(len(sorted)))) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

func safeRatio(num, den float64) float64 {
	if den == 0 {
		return 0
	}
	return num / den
}
