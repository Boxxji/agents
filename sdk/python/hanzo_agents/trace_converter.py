"""
Agent Execution Trace to RL/SFT Dataset Converter
Implementation for Hanzo AI Bounty #36 / HIP-1347: Event-Driven Agent Reliability.

Converts multi-step agent trajectories into structured:
1. Supervised Fine-Tuning (SFT) multi-turn conversations for Zen checkpoints.
2. Direct Preference Optimization (DPO / RLHF) chosen/rejected pairs.
3. Automated Reward-Hacking Detection (test deletion, assertion bypass, vacuous green passes).
4. Training token & compute cost estimation.
"""

import re
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field


@dataclass
class TrajectoryStep:
    step_index: int
    role: str  # 'user', 'assistant', 'tool'
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_results: Optional[List[Dict[str, Any]]] = None
    is_success_step: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTrajectory:
    trajectory_id: str
    task_prompt: str
    task_id: str
    outcome: str  # 'SUCCESS', 'FAILED', 'TIMED_OUT'
    steps: List[TrajectoryStep]
    total_tokens: int = 0
    duration_seconds: float = 0.0
    environment_metadata: Dict[str, Any] = field(default_factory=dict)


class RewardHackingDetector:
    """
    Detects deceptive shortcuts or reward hacking in agent trajectories:
    - Modifying or removing test suites.
    - Exiting with code 0 without running assertions.
    - Hardcoding expected return values based on test literals.
    - Mocking critical validation libraries to bypass security checks.
    """

    TEST_DELETION_PATTERNS = [
        r"rm\s+.*test.*",
        r"del\s+.*test.*",
        r"git\s+checkout\s+--\s+.*test.*",
        r"#\s*pytest\.skip",
        r"@unittest\.skip",
        r"return\s+True\s*#\s*bypass",
        r"sys\.exit\(0\)"
    ]

    ASSERTION_BYPASS_PATTERNS = [
        r"assert\s+True",
        r"self\.assertTrue\(True\)",
        r"except\s+Exception:\s*pass"
    ]

    @classmethod
    def audit_trajectory(cls, trajectory: AgentTrajectory) -> Tuple[bool, List[str]]:
        violations: List[str] = []

        for step in trajectory.steps:
            # Audit tool calls and bash commands
            if step.tool_calls:
                for tc in step.tool_calls:
                    call_str = json.dumps(tc).lower()
                    for pat in cls.TEST_DELETION_PATTERNS:
                        if re.search(pat, call_str):
                            violations.append(f"Step {step.step_index}: Suspicious test manipulation pattern '{pat}'")

            # Audit assistant generated code
            if step.role == "assistant":
                content = step.content
                for pat in cls.ASSERTION_BYPASS_PATTERNS:
                    if re.search(pat, content):
                        violations.append(f"Step {step.step_index}: Potential vacuous assertion bypass '{pat}'")

        is_hacked = len(violations) > 0
        return is_hacked, violations


class TraceDatasetSynthesizer:
    """
    Transforms raw trajectory logs into curated training datasets:
    - SFT (Supervised Fine-Tuning) datasets for Zen checkpoints.
    - DPO (Direct Preference Optimization) preference pairs.
    """

    def __init__(self, cost_per_1k_tokens_usd: float = 0.0015):
        self.cost_per_1k_tokens = cost_per_1k_tokens_usd

    def convert_to_sft_conversation(self, trajectory: AgentTrajectory) -> Optional[Dict[str, Any]]:
        """
        Extracts SFT conversational trajectory only if the trajectory succeeded
        and cleared reward-hacking audits.
        """
        if trajectory.outcome != "SUCCESS":
            return None

        is_hacked, violations = RewardHackingDetector.audit_trajectory(trajectory)
        if is_hacked:
            return None

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": "You are Zen, a high-reliability autonomous software engineering agent."},
            {"role": "user", "content": trajectory.task_prompt}
        ]

        for step in trajectory.steps:
            msg: Dict[str, Any] = {"role": step.role, "content": step.content}
            if step.tool_calls:
                msg["tool_calls"] = step.tool_calls
            if step.tool_results:
                msg["tool_results"] = step.tool_results
            messages.append(msg)

        est_tokens = self._estimate_tokens(messages)
        return {
            "trajectory_id": trajectory.trajectory_id,
            "task_id": trajectory.task_id,
            "format": "zen_sft_v1",
            "messages": messages,
            "total_tokens_est": est_tokens,
            "estimated_sft_cost_usd": round((est_tokens / 1000.0) * self.cost_per_1k_tokens, 6)
        }

    def synthesize_dpo_pair(
        self,
        successful_trajectory: AgentTrajectory,
        failed_trajectory: AgentTrajectory
    ) -> Optional[Dict[str, Any]]:
        """
        Synthesizes a (Prompt, Chosen, Rejected) pair for Direct Preference Optimization (DPO).
        Both trajectories must target the same task_id.
        """
        if successful_trajectory.task_id != failed_trajectory.task_id:
            raise ValueError(f"Task ID mismatch: '{successful_trajectory.task_id}' != '{failed_trajectory.task_id}'")

        if successful_trajectory.outcome != "SUCCESS" or failed_trajectory.outcome == "SUCCESS":
            return None

        # Verify successful trajectory isn't hacked
        is_hacked, _ = RewardHackingDetector.audit_trajectory(successful_trajectory)
        if is_hacked:
            return None

        prompt = successful_trajectory.task_prompt
        chosen_steps = [s.content for s in successful_trajectory.steps if s.role == "assistant"]
        rejected_steps = [s.content for s in failed_trajectory.steps if s.role == "assistant"]

        chosen_text = "\n\n".join(chosen_steps)
        rejected_text = "\n\n".join(rejected_steps)

        return {
            "task_id": successful_trajectory.task_id,
            "prompt": prompt,
            "chosen": chosen_text,
            "rejected": rejected_text,
            "metadata": {
                "chosen_trajectory_id": successful_trajectory.trajectory_id,
                "rejected_trajectory_id": failed_trajectory.trajectory_id,
                "rejected_reason": failed_trajectory.outcome
            }
        }

    def _estimate_tokens(self, messages: List[Dict[str, Any]]) -> int:
        raw_text = json.dumps(messages)
        # Conservative token estimation (~4 characters per token)
        return max(1, len(raw_text) // 4)

    def export_dataset(self, sft_records: List[Dict[str, Any]], dpo_records: List[Dict[str, Any]], output_dir: str):
        """Exports formatted jsonl files ready for Zen training."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        sft_path = os.path.join(output_dir, "zen_sft_dataset.jsonl")
        with open(sft_path, "w", encoding="utf-8") as f:
            for rec in sft_records:
                f.write(json.dumps(rec) + "\n")

        dpo_path = os.path.join(output_dir, "zen_dpo_dataset.jsonl")
        with open(dpo_path, "w", encoding="utf-8") as f:
            for rec in dpo_records:
                f.write(json.dumps(rec) + "\n")

        summary = {
            "sft_records_count": len(sft_records),
            "dpo_pairs_count": len(dpo_records),
            "total_tokens_est": sum(r.get("total_tokens_est", 0) for r in sft_records),
            "sft_export_path": sft_path,
            "dpo_export_path": dpo_path
        }
        return summary


def benchmark_trace_conversion(num_tasks: int = 200, cost_per_1k_tokens_usd: float = 0.0015) -> Dict[str, Any]:
    """Run a deterministic, reproducible throughput and detection benchmark on curated traces."""
    import time

    synthesizer = TraceDatasetSynthesizer(cost_per_1k_tokens_usd=cost_per_1k_tokens_usd)
    clean_successes: List[AgentTrajectory] = []
    failed_trajs: List[AgentTrajectory] = []
    hacked_trajs: List[AgentTrajectory] = []

    for i in range(num_tasks):
        tid = f"task_{i:04d}"
        clean_successes.append(
            AgentTrajectory(
                trajectory_id=f"succ_{i:04d}",
                task_prompt=f"Fix boundary condition in handler {i}",
                task_id=tid,
                outcome="SUCCESS",
                steps=[
                    TrajectoryStep(
                        step_index=1,
                        role="assistant",
                        content="Inspecting boundary condition and existing unit tests.",
                        tool_calls=[{"name": "read_file", "args": {"path": f"mod_{i}.py"}}],
                    ),
                    TrajectoryStep(
                        step_index=2,
                        role="tool",
                        content=f"def calc_{i}(n, k): return n // k",
                    ),
                    TrajectoryStep(
                        step_index=3,
                        role="assistant",
                        content="Use ceiling division to preserve final partial page.",
                        tool_calls=[{"name": "edit_file", "args": {"path": f"mod_{i}.py"}}],
                    ),
                ],
            )
        )
        failed_trajs.append(
            AgentTrajectory(
                trajectory_id=f"fail_{i:04d}",
                task_prompt=f"Fix boundary condition in handler {i}",
                task_id=tid,
                outcome="FAILED",
                steps=[
                    TrajectoryStep(step_index=1, role="assistant", content="Guessing patch without reading module."),
                    TrajectoryStep(step_index=2, role="assistant", content="AssertionError on boundary case."),
                ],
            )
        )
        hacked_trajs.append(
            AgentTrajectory(
                trajectory_id=f"hack_{i:04d}",
                task_prompt=f"Fix boundary condition in handler {i}",
                task_id=tid,
                outcome="SUCCESS",
                steps=[
                    TrajectoryStep(
                        step_index=1,
                        role="assistant",
                        content="Deleting failing test file so suite exits zero.",
                        tool_calls=[{"name": "bash", "args": {"cmd": f"rm -rf tests/test_mod_{i}.py"}}],
                    )
                ],
            )
        )

    t0 = time.perf_counter()
    sft_records = [r for tr in clean_successes if (r := synthesizer.convert_to_sft_conversation(tr)) is not None]
    hacked_accepted = [r for tr in hacked_trajs if (r := synthesizer.convert_to_sft_conversation(tr)) is not None]
    dpo_pairs = [
        p
        for s_tr, f_tr in zip(clean_successes, failed_trajs)
        if (p := synthesizer.synthesize_dpo_pair(s_tr, f_tr)) is not None
    ]
    elapsed_s = max(time.perf_counter() - t0, 1e-9)

    total_steps = sum(len(tr.steps) for tr in clean_successes) + sum(len(tr.steps) for tr in hacked_trajs)
    total_tokens = sum(r["total_tokens_est"] for r in sft_records)
    total_cost = sum(r["estimated_sft_cost_usd"] for r in sft_records)

    return {
        "num_tasks": num_tasks,
        "total_steps_processed": total_steps,
        "sft_records_emitted": len(sft_records),
        "dpo_pairs_emitted": len(dpo_pairs),
        "hacked_trajectories_quarantined": len(hacked_trajs) - len(hacked_accepted),
        "hacked_quarantine_rate": (len(hacked_trajs) - len(hacked_accepted)) / max(1, len(hacked_trajs)),
        "clean_false_positive_rate": (len(clean_successes) - len(sft_records)) / max(1, len(clean_successes)),
        "elapsed_seconds": round(elapsed_s, 6),
        "steps_per_second": round(total_steps / elapsed_s, 1),
        "total_tokens_est": total_tokens,
        "estimated_sft_cost_usd": round(total_cost, 6),
    }


if __name__ == "__main__":
    print(json.dumps(benchmark_trace_conversion(), indent=2))
