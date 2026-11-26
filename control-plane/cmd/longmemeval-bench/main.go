package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/hanzoai/agents/control-plane/internal/eval/longmemeval"
)

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintf(os.Stderr, "longmemeval-bench: %v\n", err)
		os.Exit(1)
	}
}

func run(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("longmemeval-bench", flag.ContinueOnError)
	fs.SetOutput(stderr)

	var (
		seed            int64
		categoriesFlag  string
		horizonsFlag    string
		thresholdsFlag  string
		outJSON         string
		outMD           string
		includeOutcomes bool
		skipNegRegimes  bool
		quiet           bool
	)

	fs.Int64Var(&seed, "seed", longmemeval.DefaultSeed, "Deterministic benchmark PRNG seed")
	fs.StringVar(&categoriesFlag, "categories", "", "Optional comma-separated TaskCategory filter (knowledge_update,contradiction,temporal_ordering,abstention,cross_session)")
	fs.StringVar(&horizonsFlag, "horizons", "5,25,50,100,200", "Comma-separated distractor session horizons for Negative Regime 2")
	fs.StringVar(&thresholdsFlag, "thresholds", "0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90", "Comma-separated Kai evidence thresholds for Negative Regime 3")
	fs.StringVar(&outJSON, "out-json", "", "Optional path to write reproducible JSON benchmark report")
	fs.StringVar(&outMD, "out-md", "", "Optional path to write Markdown score table")
	fs.BoolVar(&includeOutcomes, "include-outcomes", true, "Include per-case grading diagnostics in JSON output")
	fs.BoolVar(&skipNegRegimes, "skip-negative-regimes", false, "Skip the 3 negative-regime experiment suites")
	fs.BoolVar(&quiet, "quiet", false, "Suppress printing the Markdown summary to stdout")

	if err := fs.Parse(args); err != nil {
		return err
	}

	categories, err := parseCategories(categoriesFlag)
	if err != nil {
		return err
	}
	horizons, err := parseInts(horizonsFlag)
	if err != nil {
		return fmt.Errorf("invalid --horizons: %w", err)
	}
	thresholds, err := parseFloats(thresholdsFlag)
	if err != nil {
		return fmt.Errorf("invalid --thresholds: %w", err)
	}

	report, err := longmemeval.RunFullBenchmark(context.Background(), longmemeval.BenchmarkConfig{
		Seed:              seed,
		Categories:        categories,
		Horizons:          horizons,
		ParetoThresholds:  thresholds,
		IncludeOutcomes:   includeOutcomes,
		IncludeNegRegimes: !skipNegRegimes,
	})
	if err != nil {
		return err
	}

	md := longmemeval.RenderScoreTableMarkdown(report)

	if outJSON != "" {
		raw, err := json.MarshalIndent(report, "", "  ")
		if err != nil {
			return fmt.Errorf("marshal report JSON: %w", err)
		}
		raw = append(raw, '\n')
		if err := writeFileCreatingDirs(outJSON, raw); err != nil {
			return fmt.Errorf("write --out-json: %w", err)
		}
	}

	if outMD != "" {
		if err := writeFileCreatingDirs(outMD, []byte(md)); err != nil {
			return fmt.Errorf("write --out-md: %w", err)
		}
	}

	if !quiet {
		_, _ = io.WriteString(stdout, md)
	}
	return nil
}

func parseCategories(raw string) ([]longmemeval.TaskCategory, error) {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return nil, nil
	}
	valid := make(map[longmemeval.TaskCategory]struct{}, len(longmemeval.AllCategories))
	for _, c := range longmemeval.AllCategories {
		valid[c] = struct{}{}
	}
	parts := strings.Split(trimmed, ",")
	out := make([]longmemeval.TaskCategory, 0, len(parts))
	seen := make(map[longmemeval.TaskCategory]struct{}, len(parts))
	for _, p := range parts {
		cat := longmemeval.TaskCategory(strings.TrimSpace(p))
		if cat == "" {
			continue
		}
		if _, ok := valid[cat]; !ok {
			return nil, fmt.Errorf("unknown category %q (valid: knowledge_update, contradiction, temporal_ordering, abstention, cross_session)", cat)
		}
		if _, dup := seen[cat]; !dup {
			seen[cat] = struct{}{}
			out = append(out, cat)
		}
	}
	return out, nil
}

func parseInts(raw string) ([]int, error) {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return nil, nil
	}
	parts := strings.Split(trimmed, ",")
	out := make([]int, 0, len(parts))
	for _, p := range parts {
		s := strings.TrimSpace(p)
		if s == "" {
			continue
		}
		v, err := strconv.Atoi(s)
		if err != nil || v <= 0 {
			return nil, fmt.Errorf("expected positive integer, got %q", s)
		}
		out = append(out, v)
	}
	return out, nil
}

func parseFloats(raw string) ([]float64, error) {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return nil, nil
	}
	parts := strings.Split(trimmed, ",")
	out := make([]float64, 0, len(parts))
	for _, p := range parts {
		s := strings.TrimSpace(p)
		if s == "" {
			continue
		}
		v, err := strconv.ParseFloat(s, 64)
		if err != nil || v < 0 || v > 1 {
			return nil, fmt.Errorf("expected float in [0, 1], got %q", s)
		}
		out = append(out, v)
	}
	return out, nil
}

func writeFileCreatingDirs(path string, data []byte) error {
	dir := filepath.Dir(path)
	if dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	return os.WriteFile(path, data, 0o644)
}
