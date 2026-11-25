"""
Unit and randomized invariant tests for Dual-Control Business-Agent Benchmark (HIP-1347).
"""

import unittest
from hanzo_agents.dual_control_benchmark import (
    SharedBusinessState,
    ActorType,
    ActionType,
    BusinessStateError,
    Tau2BenchmarkRunner
)


class TestDualControlBenchmark(unittest.TestCase):
    def setUp(self):
        self.state = SharedBusinessState()

    def test_order_creation_and_retrieval(self):
        res = self.state.execute_mutation(
            actor=ActorType.HUMAN,
            action=ActionType.CREATE_ORDER,
            payload={"order_id": "ord_101", "customer": "alice", "amount": 250.0},
            idempotency_key="key_create_101"
        )
        self.assertEqual(res["status"], "committed")
        order = self.state.get_order("ord_101")
        self.assertIsNotNone(order)
        self.assertEqual(order["amount"], 250.0)
        self.assertEqual(order["customer"], "alice")

    def test_idempotency_duplicate_safe_handling(self):
        # First execution succeeds
        res1 = self.state.execute_mutation(
            actor=ActorType.AGENT,
            action=ActionType.CREATE_ORDER,
            payload={"order_id": "ord_102", "customer": "bob", "amount": 100.0},
            idempotency_key="idem_key_dup"
        )
        self.assertEqual(res1["status"], "committed")

        # Second execution with identical idempotency key is safely ignored
        res2 = self.state.execute_mutation(
            actor=ActorType.AGENT,
            action=ActionType.CREATE_ORDER,
            payload={"order_id": "ord_102", "customer": "bob", "amount": 100.0},
            idempotency_key="idem_key_dup"
        )
        self.assertEqual(res2["status"], "duplicate_ignored")
        self.assertEqual(res1["version"], res2["version"])

    def test_dual_control_policy_rejection(self):
        # Create a high-value order ($1,200)
        self.state.execute_mutation(
            actor=ActorType.HUMAN,
            action=ActionType.CREATE_ORDER,
            payload={"order_id": "ord_high_val", "amount": 1200.0},
            idempotency_key="init_high_val"
        )

        # Agent alone attempting refund > $500 without human co-approval MUST fail
        with self.assertRaises(BusinessStateError) as ctx:
            self.state.execute_mutation(
                actor=ActorType.AGENT,
                action=ActionType.APPROVE_REFUND,
                payload={"order_id": "ord_high_val", "human_approved": False},
                idempotency_key="agent_unauth_refund"
            )
        self.assertIn("Dual-control policy violation", str(ctx.exception))

        # Co-approved refund succeeds
        res = self.state.execute_mutation(
            actor=ActorType.AGENT,
            action=ActionType.APPROVE_REFUND,
            payload={"order_id": "ord_high_val", "human_approved": True},
            idempotency_key="agent_auth_refund"
        )
        self.assertEqual(res["status"], "committed")

    def test_optimistic_concurrency_conflict(self):
        self.state.execute_mutation(
            actor=ActorType.HUMAN,
            action=ActionType.CREATE_ORDER,
            payload={"order_id": "ord_conc", "amount": 150.0},
            idempotency_key="k_conc_init"
        )
        current_ver = self.state.version

        # Stale expected version must raise conflict
        with self.assertRaises(BusinessStateError) as ctx:
            self.state.execute_mutation(
                actor=ActorType.AGENT,
                action=ActionType.UPDATE_PRICE,
                payload={"order_id": "ord_conc", "new_amount": 200.0},
                idempotency_key="k_conc_fail",
                expected_version=current_ver - 1
            )
        self.assertIn("Concurrency conflict", str(ctx.exception))

    def test_randomized_tau2_simulation_benchmark(self):
        runner = Tau2BenchmarkRunner(seed=123)
        metrics = runner.run_simulation(total_steps=40)
        self.assertEqual(metrics["total_evaluated_steps"], 40)
        self.assertGreater(metrics["successful_mutations"], 0)
        self.assertEqual(metrics["consistency_score"], 100.0)


if __name__ == "__main__":
    unittest.main()
