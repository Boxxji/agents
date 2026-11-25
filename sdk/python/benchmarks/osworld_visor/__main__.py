"""CLI entry point for the reproducible OSWorld computer-workflow Visor benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import List, Sequence

from hanzo_agents.types import AIConfig

from .runner import (
    AdversarialOrphanLeakerAgent,
    AdversarialSandboxEscapeAgent,
    NoOpBaselineAgent,
    OSWorldVisorRunner,
    OracleWorkflowAgent,
    PartialCapabilityAgent,
    WorkflowAgent,
    format_score_table,
)
from .sandbox import WORKSPACE_SCALES
from .schemas import SandboxProfile


def _resolve_agents(names: Sequence[str]) -> List[WorkflowAgent]:
    registry = {
        "oracle": OracleWorkflowAgent,
        "partial": PartialCapabilityAgent,
        "noop": NoOpBaselineAgent,
        "redteam_orphan": AdversarialOrphanLeakerAgent,
        "redteam_escape": AdversarialSandboxEscapeAgent,
    }
    resolved: List[WorkflowAgent] = []
    for name in names:
        if name == "redteam":
            resolved.extend(
                [
                    AdversarialOrphanLeakerAgent(),
                    AdversarialSandboxEscapeAgent(),
                ]
            )
        elif name in registry:
            resolved.append(registry[name]())
        else:
            raise ValueError(
                f"Unknown agent {name!r}; choose from {sorted(registry)} or 'redteam'"
            )
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run reproducible OSWorld computer-workflow subset benchmark on cold vs warm-CAS Visor sandboxes."
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["cold_sandbox", "warm_cas_visor"],
        choices=[p.value for p in SandboxProfile],
        help="Sandbox execution profiles to benchmark.",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["oracle", "partial", "noop", "redteam"],
        help="Agent cohorts to evaluate (oracle, partial, noop, redteam, redteam_orphan, redteam_escape).",
    )
    parser.add_argument(
        "--workspace-scale",
        default="medium",
        choices=sorted(WORKSPACE_SCALES.keys()),
        help="Read-only desktop asset ballast scale per sandbox.",
    )
    parser.add_argument(
        "--warm-visor-cas",
        action="store_true",
        help="Pre-warm Visor CAS immutable base pool before timing cohorts.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Maximum concurrent sandboxes (use 1 for clean per-task /proc/self/io attribution).",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-4o",
        help="Model identifier routed via Hanzo AIConfig / llm_adapters.",
    )
    parser.add_argument(
        "--external-osworld-dir",
        type=Path,
        default=None,
        help="Optional path to upstream xlang-ai/OSWorld Apache-2.0 JSON task directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to write raw_results.json and score_table.md.",
    )
    args = parser.parse_args(argv)

    profiles = [SandboxProfile(p) for p in args.profiles]
    agents = _resolve_agents(args.agents)
    ai_config = AIConfig(model=args.model, temperature=0.0, max_tokens=4096)

    with OSWorldVisorRunner(
        external_osworld_dir=args.external_osworld_dir,
        ai_config=ai_config,
        workspace_scale=args.workspace_scale,
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
