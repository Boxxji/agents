"""
Reproducible CLI and programmatic runner for the Hanzo Agents LongMemEval
memory policy benchmark (`hanzoai/agents#75`, HIP-1347).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .dataset import ALL_CATEGORIES, BenchmarkCase, build_longmemeval_dataset
from .embedder import DeterministicHashEmbedder
from .metrics import (
    CaseOutcome,
    PolicyScore,
    aggregate_policy_score,
    format_score_table_markdown,
    score_case,
)
from .policies import BaseMemoryPolicy, build_default_policies
from .transport import ControlPlaneMemoryTransport, InMemoryHanzoAgentsClient


def _detect_git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


async def evaluate_policy(
    policy: BaseMemoryPolicy,
    cases: Sequence[BenchmarkCase],
) -> PolicyScore:
    """
    Evaluate a single memory policy across all benchmark cases using a clean
    isolated `ControlPlaneMemoryTransport` per case.
    """
    outcomes: List[CaseOutcome] = []
    total_api_calls: Dict[str, int] = {
        "set": 0,
        "get": 0,
        "delete": 0,
        "list": 0,
        "set_vector": 0,
        "delete_vector": 0,
        "similarity_search": 0,
    }

    for case in cases:
        transport = ControlPlaneMemoryTransport()
        client = InMemoryHanzoAgentsClient(transport=transport)

        for turn in case.turns:
            await policy.ingest(client, turn)

        t0 = time.perf_counter()
        result = await policy.recall(client, case.query)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        outcomes.append(score_case(case, result, elapsed_ms))
        for op, count in transport.api_calls.items():
            total_api_calls[op] = total_api_calls.get(op, 0) + count

    return aggregate_policy_score(
        policy_name=policy.name,
        description=policy.description,
        outcomes=outcomes,
        api_calls=total_api_calls,
    )


async def run_benchmark(
    *,
    seed: int = 1347,
    min_score: float = 0.45,
    top_k: int = 5,
    embedding_dim: int = 128,
    policy_names: Optional[Sequence[str]] = None,
    categories: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Run the complete LongMemEval policy evaluation suite and return a
    serializable report dictionary containing environment metadata, summary
    scorecards, and per-case raw outcomes.
    """
    cases = build_longmemeval_dataset(seed=seed, categories=categories)
    embedder = DeterministicHashEmbedder(dim=embedding_dim, seed=seed)
    policies = build_default_policies(embedder, min_score=min_score, top_k=top_k)

    if policy_names:
        wanted = set(policy_names)
        available = {p.name for p in policies}
        unknown = sorted(wanted - available)
        if unknown:
            raise ValueError(
                f"Unknown policy name(s): {unknown}. Available: {sorted(available)}"
            )
        policies = [p for p in policies if p.name in wanted]

    scores: List[PolicyScore] = []
    for policy in policies:
        scores.append(await evaluate_policy(policy, cases))

    markdown = format_score_table_markdown(scores)
    return {
        "benchmark": "hanzo-agents-longmemeval",
        "issue": "hanzoai/agents#75",
        "hip": "HIP-1347",
        "reference": {
            "paper": "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory (Wu et al., ICLR 2025)",
            "paper_url": "https://proceedings.iclr.cc/paper_files/paper/2025/hash/d813d324dbf0598bbdc9c8e79740ed01-Abstract-Conference.html",
            "code_url": "https://github.com/xiaowu0162/LongMemEval",
            "license": "MIT (zero upstream code or dataset imported; clean-room benchmark over Hanzo SDK)",
        },
        "config": {
            "seed": seed,
            "min_score": min_score,
            "top_k": top_k,
            "embedding_dim": embedding_dim,
            "categories": list(categories) if categories else list(ALL_CATEGORIES),
            "total_cases": len(cases),
        },
        "environment": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "git_sha": _detect_git_sha(),
        },
        "scores": [s.to_dict(include_outcomes=True) for s in scores],
        "markdown_table": markdown,
    }


async def run_threshold_sweep(
    *,
    seed: int = 1347,
    thresholds: Sequence[float] = (0.0, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75),
    top_k: int = 8,
    embedding_dim: int = 128,
) -> List[Dict[str, Any]]:
    """
    Evaluate `vector_thresholded` across multiple cosine similarity cutoffs to
    quantify the abstention vs recall trade-off curve.
    """
    from .policies import VectorThresholdedPolicy

    cases = build_longmemeval_dataset(seed=seed)
    embedder = DeterministicHashEmbedder(dim=embedding_dim, seed=seed)
    rows: List[Dict[str, Any]] = []
    for thr in thresholds:
        pol = VectorThresholdedPolicy(embedder=embedder, min_score=thr, top_k=top_k)
        sc = await evaluate_policy(pol, cases)
        rows.append(
            {
                "min_score": thr,
                "exact_match": sc.exact_match,
                "recall_at_k": sc.recall_at_k,
                "abstention_precision": sc.abstention_precision,
                "abstention_recall": sc.abstention_recall,
                "abstention_f1": sc.abstention_f1,
                "false_positive_rate": sc.false_positive_rate,
                "avg_retrieval_tokens": sc.avg_retrieval_tokens,
            }
        )
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m benchmarks.longmemeval",
        description=(
            "Run the reproducible LongMemEval temporal memory benchmark "
            "against existing Hanzo Agents Python SDK memory policies."
        ),
    )
    parser.add_argument("--seed", type=int, default=1347, help="Deterministic RNG seed (default: 1347)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.45,
        help="Cosine similarity threshold for thresholded vector policies (default: 0.45)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K parameter for vector similarity_search (default: 5)",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
        help="Deterministic hash embedding dimension (default: 128)",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=None,
        help="Subset of policies to evaluate (default: all 5 policies)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Subset of LongMemEval categories to evaluate (default: all 5 categories)",
    )
    parser.add_argument(
        "--sweep-thresholds",
        action="store_true",
        help="Include cosine threshold sensitivity sweep for vector_thresholded in output JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write full raw JSON results",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Optional path to write the rendered Markdown score table",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress printing the Markdown table to stdout",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    report = asyncio.run(
        run_benchmark(
            seed=args.seed,
            min_score=args.threshold,
            top_k=args.top_k,
            embedding_dim=args.embedding_dim,
            policy_names=args.policies,
            categories=args.categories,
        )
    )
    if args.sweep_thresholds:
        report["threshold_sweep"] = asyncio.run(
            run_threshold_sweep(
                seed=args.seed,
                top_k=max(args.top_k, 8),
                embedding_dim=args.embedding_dim,
            )
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(report["markdown_table"], encoding="utf-8")

    if not args.quiet:
        sys.stdout.write(report["markdown_table"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
