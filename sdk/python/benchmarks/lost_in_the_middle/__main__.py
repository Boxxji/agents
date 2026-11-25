"""CLI entry point for the Lost-in-the-Middle context-packing benchmark.

Examples:
  # Run all offline physical + calibrated stress-test modes and write artifacts:
  python3 -m benchmarks.lost_in_the_middle \
      --mode all \
      --output-json ../../docs/research/hip-1347-lost-in-the-middle/raw_results.json \
      --output-markdown ../../docs/research/hip-1347-lost-in-the-middle/score_table.md

  # Run live against Hanzo Zen/Enso or OpenAI-compatible endpoint:
  HANZO_API_KEY=... python3 -m benchmarks.lost_in_the_middle \
      --mode live_llm --model zen/zen-coder --doc-counts 5,10 --cases-per-position 1
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Optional, Sequence, Tuple

from .packers import POLICY_NAMES
from .runner import BenchmarkConfig, render_score_table_markdown, run_benchmark


def _parse_int_tuple(raw: str) -> Tuple[int, ...]:
    vals = [int(tok.strip()) for tok in raw.split(",") if tok.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("Expected at least one positive integer")
    return tuple(vals)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m benchmarks.lost_in_the_middle",
        description="Evaluate evidence-ordering & context-packing policies for Hanzo Agents (HIP-1347).",
    )
    parser.add_argument(
        "--mode",
        choices=("all", "budget_truncation", "distractor_interference", "calibrated_tacl2024", "live_llm"),
        default="all",
        help="Evaluation mode. 'all' runs physical budget_truncation (middle_trim + tail_clip), "
        "distractor_interference, and calibrated_tacl2024 stress-test.",
    )
    parser.add_argument(
        "--truncation-mode",
        choices=("hanzo_middle_trim", "tail_clip"),
        default="hanzo_middle_trim",
        help="Downstream truncation mechanism when a budget-unaware policy exceeds max_tokens.",
    )
    parser.add_argument(
        "--doc-counts",
        type=_parse_int_tuple,
        default=(5, 10, 20),
        help="Comma-separated document counts N to sweep across positions 1..N (default: 5,10,20).",
    )
    parser.add_argument(
        "--budgets",
        type=_parse_int_tuple,
        default=(512, 1024, 2048, 4096),
        help="Comma-separated strict token budgets (default: 512,1024,2048,4096).",
    )
    parser.add_argument(
        "--cases-per-position",
        type=int,
        default=2,
        help="Number of independent QA cases per (N, gold_position) slot (default: 2).",
    )
    parser.add_argument(
        "--negative-cases",
        type=int,
        default=10,
        help="Number of cases per negative/adversarial regime (default: 10; 0 to skip).",
    )
    parser.add_argument(
        "--policies",
        type=str,
        default=",".join(POLICY_NAMES),
        help="Comma-separated packing policies to evaluate.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed.")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Max concurrent async case evaluations via asyncio.gather.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="zen/zen-coder",
        help="Model identifier for AIConfig / live_llm mode.",
    )
    parser.add_argument(
        "--calibrated-profile",
        choices=("gpt-3.5-turbo-16k", "claude-1.3", "mpt-30b-instruct"),
        default="gpt-3.5-turbo-16k",
        help="Published TACL 2024 sensitivity profile for calibrated_tacl2024 mode.",
    )
    parser.add_argument(
        "--external-jsonl",
        type=Path,
        default=None,
        help="Optional path to upstream-compatible Lost-in-the-Middle .jsonl or .jsonl.gz file.",
    )
    parser.add_argument(
        "--external-limit",
        type=int,
        default=None,
        help="Maximum number of cases to load from --external-jsonl.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output path for full raw_results.json.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=None,
        help="Optional output path for generated score_table.md.",
    )
    parser.add_argument(
        "--no-raw-records",
        action="store_true",
        help="Omit per-case raw_records array from JSON output to keep artifact compact.",
    )
    parser.add_argument(
        "--compact-json",
        action="store_true",
        help="Write compact single-line JSON (no indentation) to minimize file size.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    policies = tuple(p.strip() for p in args.policies.split(",") if p.strip())
    config = BenchmarkConfig(
        mode=args.mode,
        truncation_mode=args.truncation_mode,
        doc_counts=args.doc_counts,
        budgets=args.budgets,
        cases_per_position=args.cases_per_position,
        negative_cases_per_regime=args.negative_cases,
        policies=policies,
        seed=args.seed,
        concurrency=args.concurrency,
        model=args.model,
        calibrated_profile=args.calibrated_profile,
        external_jsonl=args.external_jsonl,
        external_limit=args.external_limit,
        include_raw_records=not args.no_raw_records,
    )
    results = asyncio.run(run_benchmark(config))
    markdown = render_score_table_markdown(results)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        if args.compact_json:
            payload = json.dumps(results, separators=(",", ":"), sort_keys=False) + "\n"
        else:
            payload = json.dumps(results, indent=2, sort_keys=False) + "\n"
        args.output_json.write_text(payload, encoding="utf-8")
    if args.output_markdown is not None:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(markdown, encoding="utf-8")

    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
