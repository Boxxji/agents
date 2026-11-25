"""CLI entry point for the HIP-1348 durable workflow replay benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .runner import render_score_table_markdown, run_full_benchmark


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m benchmarks.durable_replay",
        description=(
            "Run reproducible cross-process crash-recovery durable workflow replay "
            "benchmarks (HIP-1348, hanzoai/agents#78 & #85)."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=6,
        help="Number of workflow steps for crash-at-every-step matrix (default: 6).",
    )
    parser.add_argument(
        "--trivial-runs",
        type=int,
        default=15,
        help="Number of no-crash 1-step runs for overhead crossover (default: 15).",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Optional working directory for SQLite/JSONL scratch state (default: temp dir).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write raw_results.json and score_table.md.",
    )
    parser.add_argument(
        "--json-stdout",
        action="store_true",
        help="Emit full raw_results JSON to stdout instead of Markdown summary.",
    )
    args = parser.parse_args(argv)

    report = run_full_benchmark(
        num_steps=args.steps,
        trivial_runs=args.trivial_runs,
        work_dir=args.work_dir,
        output_dir=args.output_dir,
    )

    if args.json_stdout:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_score_table_markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
