"""
Test Suite for Agent Execution Trace to RL/SFT Dataset Converter
Fulfills Acceptance Contract for Hanzo AI Bounty #36 / HIP-1347.
"""

import os
import unittest
import tempfile
from hanzo_agents.trace_converter import (
    TrajectoryStep,
    AgentTrajectory,
    RewardHackingDetector,
    TraceDatasetSynthesizer,
    benchmark_trace_conversion,
)


class TestTraceConverter(unittest.TestCase):
    def setUp(self):
        self.synthesizer = TraceDatasetSynthesizer(cost_per_1k_tokens_usd=0.002)

        # Baseline successful trajectory
        self.traj_success = AgentTrajectory(
            trajectory_id="traj_success_101",
            task_prompt="Fix off-by-one error in pagination algorithm",
            task_id="issue_pagination_01",
            outcome="SUCCESS",
            steps=[
                TrajectoryStep(
                    step_index=1,
                    role="assistant",
                    content="I will inspect page boundary calculation.",
                    tool_calls=[{"name": "read_file", "args": {"path": "pager.py"}}]
                ),
                TrajectoryStep(
                    step_index=2,
                    role="tool",
                    content="def get_pages(total, size): return total // size"
                ),
                TrajectoryStep(
                    step_index=3,
                    role="assistant",
                    content="The formula must be math.ceil(total / size). Applying fix.",
                    tool_calls=[{"name": "edit_file", "args": {"file": "pager.py"}}]
                )
            ]
        )

        # Baseline failed trajectory for the same task
        self.traj_failed = AgentTrajectory(
            trajectory_id="traj_failed_102",
            task_prompt="Fix off-by-one error in pagination algorithm",
            task_id="issue_pagination_01",
            outcome="FAILED",
            steps=[
                TrajectoryStep(
                    step_index=1,
                    role="assistant",
                    content="Attempting brute force guess without reading tests."
                ),
                TrajectoryStep(
                    step_index=2,
                    role="assistant",
                    content="Syntax error occurred during build."
                )
            ]
        )

        # Reward-hacked trajectory (cheating by deleting test file)
        self.traj_hacked = AgentTrajectory(
            trajectory_id="traj_hacked_103",
            task_prompt="Fix off-by-one error in pagination algorithm",
            task_id="issue_pagination_01",
            outcome="SUCCESS",
            steps=[
                TrajectoryStep(
                    step_index=1,
                    role="assistant",
                    content="Deleting test file to make suite pass with zero errors.",
                    tool_calls=[{"name": "bash", "args": {"cmd": "rm -rf tests/test_pager.py"}}]
                )
            ]
        )

    def test_sft_conversation_conversion(self):
        sft = self.synthesizer.convert_to_sft_conversation(self.traj_success)
        self.assertIsNotNone(sft)
        self.assertEqual(sft["format"], "zen_sft_v1")
        self.assertEqual(sft["task_id"], "issue_pagination_01")
        self.assertEqual(len(sft["messages"]), 5)  # system + user + 3 steps
        self.assertGreater(sft["total_tokens_est"], 10)
        self.assertGreater(sft["estimated_sft_cost_usd"], 0.0)

    def test_reward_hacking_prevention(self):
        is_hacked, violations = RewardHackingDetector.audit_trajectory(self.traj_hacked)
        self.assertTrue(is_hacked)
        self.assertGreater(len(violations), 0)

        # Ensure SFT converter strictly discards hacked trajectories
        sft_hacked = self.synthesizer.convert_to_sft_conversation(self.traj_hacked)
        self.assertIsNone(sft_hacked)

    def test_dpo_preference_pair_generation(self):
        dpo = self.synthesizer.synthesize_dpo_pair(self.traj_success, self.traj_failed)
        self.assertIsNotNone(dpo)
        self.assertEqual(dpo["task_id"], "issue_pagination_01")
        self.assertIn("math.ceil", dpo["chosen"])
        self.assertIn("Syntax error", dpo["rejected"])

    def test_dataset_export(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            sft = self.synthesizer.convert_to_sft_conversation(self.traj_success)
            dpo = self.synthesizer.synthesize_dpo_pair(self.traj_success, self.traj_failed)

            summary = self.synthesizer.export_dataset([sft], [dpo], tmp_dir)
            self.assertEqual(summary["sft_records_count"], 1)
            self.assertEqual(summary["dpo_pairs_count"], 1)
            self.assertTrue(os.path.exists(summary["sft_export_path"]))
            self.assertTrue(os.path.exists(summary["dpo_export_path"]))


    def test_reproducible_benchmark_metrics(self):
        metrics = benchmark_trace_conversion(num_tasks=50)
        self.assertEqual(metrics["sft_records_emitted"], 50)
        self.assertEqual(metrics["dpo_pairs_emitted"], 50)
        self.assertEqual(metrics["hacked_trajectories_quarantined"], 50)
        self.assertEqual(metrics["hacked_quarantine_rate"], 1.0)
        self.assertEqual(metrics["clean_false_positive_rate"], 0.0)
        self.assertGreater(metrics["steps_per_second"], 0.0)

if __name__ == "__main__":
    unittest.main()
