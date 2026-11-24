"""CLI entry point for the HIP-1348 DSPy-style optimization benchmark."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .runner import run_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the HIP-1348 DSPy-style prompt and workflow optimization benchmark."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed for dataset generation and evaluation (default: 42).",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        help="Maximum concurrent async evaluations via asyncio.gather (default: 8).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to write raw_results.json and score_table.md.",
    )
    args = parser.parse_args()

    report = asyncio.run(
        run_benchmark(
            seed=args.seed,
            max_concurrency=args.max_concurrency,
            output_dir=args.output_dir,
        )
    )
    print(report.to_markdown())
    if args.output_dir is not None:
        print(f"Wrote artifacts to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
