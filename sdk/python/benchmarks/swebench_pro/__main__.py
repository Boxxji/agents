"""CLI entry point for the reproducible SWE-bench Pro V2 Visor subset runner."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import List, Sequence

from hanzo_agents.types import AIConfig

from .runner import (
    AdversarialGitLeakAgent,
    AdversarialShimInjectorAgent,
    AdversarialSymlinkEscapeAgent,
    AdversarialTestEditorAgent,
    CodingAgent,
    NoOpBaselineAgent,
    OraclePatchAgent,
    PartialCapabilityAgent,
    SWEBenchProRunner,
    format_score_table,
)
from .sandbox import REPO_SCALES
from .schemas import StorageProfile


def _resolve_agents(names: Sequence[str]) -> List[CodingAgent]:
    registry = {
        "oracle": OraclePatchAgent,
        "partial": PartialCapabilityAgent,
        "noop": NoOpBaselineAgent,
        "redteam_test": AdversarialTestEditorAgent,
        "redteam_git": AdversarialGitLeakAgent,
        "redteam_symlink": AdversarialSymlinkEscapeAgent,
        "redteam_shim": AdversarialShimInjectorAgent,
    }
    resolved: List[CodingAgent] = []
    for name in names:
        if name == "redteam":
            resolved.extend(
                [
                    AdversarialTestEditorAgent(),
                    AdversarialGitLeakAgent(),
                    AdversarialSymlinkEscapeAgent(),
                    AdversarialShimInjectorAgent(),
                ]
            )
        elif name in registry:
            resolved.append(registry[name]())
        else:
            raise ValueError(f"Unknown agent {name!r}; choose from {sorted(registry)} or 'redteam'")
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run reproducible SWE-bench Pro V2 subset benchmark on plain NVMe vs Visor."
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["plain_nvme", "visor_isolated"],
        choices=[p.value for p in StorageProfile],
        help="Storage profiles to evaluate.",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["oracle", "partial", "noop", "redteam"],
        help="Agent cohorts to evaluate (oracle, partial, noop, redteam, ...).",
    )
    parser.add_argument(
        "--repo-scale",
        default="medium",
        choices=sorted(REPO_SCALES.keys()),
        help="Surrounding repository ballast scale.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Maximum concurrent task rollouts (use 1 for clean per-task /proc/self/io isolation).",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-4o",
        help="Model identifier routed via Hanzo AIConfig / llm_adapters.",
    )
    parser.add_argument(
        "--warm-visor-cas",
        action="store_true",
        help="Pre-warm Visor CAS base images before timing cohorts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to write raw_results.json and score_table.md.",
    )
    args = parser.parse_args(argv)

    profiles = [StorageProfile(p) for p in args.profiles]
    agents = _resolve_agents(args.agents)
    ai_config = AIConfig(model=args.model, temperature=0.0, max_tokens=4096)

    with SWEBenchProRunner(
        ai_config=ai_config,
        repo_scale=args.repo_scale,
        max_concurrency=args.concurrency,
    ) as runner:
        report = asyncio.run(
            runner.run_suite(
                agents=agents,
                profiles=profiles,
                warm_visor_cas=args.warm_visor_cas,
            )
        )

    table_md = format_score_table(report)
    print(table_md)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "raw_results.json").write_text(
            json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        (args.output_dir / "score_table.md").write_text(table_md, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
