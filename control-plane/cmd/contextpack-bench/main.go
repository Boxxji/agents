// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

// Command contextpack-bench executes the deterministic HIP-1346 Lost-in-the-Middle
// context-packing benchmark (hanzoai/agents#83) and emits machine-readable JSON
// results and an auditable Markdown score table.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/hanzoai/agents/control-plane/internal/eval/contextpack"
)

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintf(os.Stderr, "contextpack-bench error: %v\n", err)
		os.Exit(1)
	}
}

func run(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("contextpack-bench", flag.ContinueOnError)
	fs.SetOutput(stderr)

	seed := fs.Int64("seed", 1346, "deterministic PRNG seed for corpus generation")
	tokenBudget := fs.Int("token-budget", 1400, "token budget for budgeted context-packing policies")
	outJSON := fs.String("out-json", "", "optional path to write BenchmarkReport JSON")
	outMD := fs.String("out-md", "", "optional path to write rendered Markdown score table")
	quiet := fs.Bool("quiet", false, "suppress summary table output to stdout")

	if err := fs.Parse(args); err != nil {
		return err
	}
	if *tokenBudget <= 0 {
		return fmt.Errorf("--token-budget must be > 0, got %d", *tokenBudget)
	}

	runCfg := contextpack.DefaultRunConfig()
	runCfg.Seed = *seed
	runCfg.PackConfig.TokenBudget = *tokenBudget

	report, err := contextpack.RunBenchmark(context.Background(), runCfg)
	if err != nil {
		return fmt.Errorf("benchmark execution failed: %w", err)
	}

	if *outJSON != "" {
		payload, err := json.MarshalIndent(report, "", "  ")
		if err != nil {
			return fmt.Errorf("marshal report json: %w", err)
		}
		payload = append(payload, '\n')
		if err := writeFileAtomic(*outJSON, payload); err != nil {
			return fmt.Errorf("write json %s: %w", *outJSON, err)
		}
	}

	md := contextpack.RenderMarkdownScoreTable(report)
	if *outMD != "" {
		if err := writeFileAtomic(*outMD, []byte(md)); err != nil {
			return fmt.Errorf("write markdown %s: %w", *outMD, err)
		}
	}

	if !*quiet {
		fmt.Fprintf(stdout, "HIP-1346 contextpack-bench completed (seed=%d, budget=%d, primary=%d cases, negative=%d cases)\n",
			report.Metadata.Seed,
			report.Metadata.PackConfig.TokenBudget,
			report.Metadata.PrimaryCaseCount,
			report.Metadata.NegativeRegimeCases,
		)
		for _, s := range report.PrimaryOverall {
			fmt.Fprintf(stdout, "  %-24s exact=%5.1f%%  cal_acc=%5.1f%%  edge=%5.1f%%  tokens=%6.0f  savings=%5.1f%%  usd/1k=$%.4f  acc/1k=%.3f  leaks=%d  hijacks=%d\n",
				s.PolicyID,
				s.ExactPassRate*100,
				s.MeanCalibratedAccuracy*100,
				s.GoldInEdgeWindowRate*100,
				s.AvgInputTokens,
				s.TokenSavingsPct,
				s.CostUSDPer1kQueries,
				s.AccuracyPer1kInputTokens,
				s.CrossTenantLeaks,
				s.InjectionRankHijacks,
			)
		}
	}

	return nil
}

func writeFileAtomic(target string, data []byte) error {
	dir := filepath.Dir(target)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".contextpack-tmp-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		_ = os.Remove(tmpName)
		return err
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpName)
		return err
	}
	return os.Rename(tmpName, target)
}
