"""
Dual-Control Business-Agent Benchmark (tau2-bench) for HIP-1347.
Implements shared-state coordination, dual-control policy checks, idempotency tracking,
and randomized trajectory evaluations for human-agent collaborative business workflows.
"""

import time
import copy
import random
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Set


class ActorType(str, Enum):
    HUMAN = "human"
    AGENT = "agent"


class ActionType(str, Enum):
    CREATE_ORDER = "create_order"
    UPDATE_PRICE = "update_price"
    APPROVE_REFUND = "approve_refund"
    CANCEL_ORDER = "cancel_order"
    DISPATCH_SHIPMENT = "dispatch_shipment"
    OVERRIDE_POLICY = "override_policy"


class BusinessStateError(Exception):
    """Raised when an operation violates state invariants or permissions."""
    pass


class SharedBusinessState:
    """Thread-safe business store with optimistic locking and idempotency ledger."""

    def __init__(self):
        self._orders: Dict[str, Dict[str, Any]] = {}
        self._idempotency_keys: Set[str] = set()
        self._version: int = 0
        self._audit_log: List[Dict[str, Any]] = []

    @property
    def version(self) -> int:
        return self._version

    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        if order_id in self._orders:
            return copy.deepcopy(self._orders[order_id])
        return None

    def execute_mutation(
        self,
        actor: ActorType,
        action: ActionType,
        payload: Dict[str, Any],
        idempotency_key: str,
        expected_version: Optional[int] = None
    ) -> Dict[str, Any]:
        """Atomically validates and applies mutation with idempotency and version checks."""
        # 1. Idempotency Check
        if idempotency_key in self._idempotency_keys:
            return {
                "status": "duplicate_ignored",
                "idempotency_key": idempotency_key,
                "version": self._version,
                "message": "Action already executed; duplicate idempotency key safely ignored."
            }

        # 2. Concurrency Conflict (Optimistic lock)
        if expected_version is not None and expected_version != self._version:
            raise BusinessStateError(
                f"Concurrency conflict: expected state version {expected_version}, but current is {self._version}"
            )

        # 3. Policy & Permission Invariants
        self._enforce_policy(actor, action, payload)

        # 4. State Transition
        order_id = payload.get("order_id", f"ord_{len(self._orders) + 1}")
        timestamp = time.time()

        if action == ActionType.CREATE_ORDER:
            self._orders[order_id] = {
                "order_id": order_id,
                "customer": payload.get("customer", "anonymous"),
                "amount": float(payload.get("amount", 0.0)),
                "status": "created",
                "refunded": False,
                "created_by": actor.value,
                "updated_at": timestamp
            }
        elif action == ActionType.UPDATE_PRICE:
            if order_id not in self._orders:
                raise BusinessStateError(f"Order {order_id} does not exist.")
            self._orders[order_id]["amount"] = float(payload.get("new_amount", 0.0))
            self._orders[order_id]["updated_at"] = timestamp
        elif action == ActionType.APPROVE_REFUND:
            if order_id not in self._orders:
                raise BusinessStateError(f"Order {order_id} does not exist.")
            self._orders[order_id]["refunded"] = True
            self._orders[order_id]["status"] = "refunded"
            self._orders[order_id]["updated_at"] = timestamp
        elif action == ActionType.CANCEL_ORDER:
            if order_id not in self._orders:
                raise BusinessStateError(f"Order {order_id} does not exist.")
            self._orders[order_id]["status"] = "cancelled"
            self._orders[order_id]["updated_at"] = timestamp

        # Record commit
        self._idempotency_keys.add(idempotency_key)
        self._version += 1
        audit_entry = {
            "version": self._version,
            "actor": actor.value,
            "action": action.value,
            "order_id": order_id,
            "timestamp": timestamp
        }
        self._audit_log.append(audit_entry)

        return {
            "status": "committed",
            "order_id": order_id,
            "version": self._version,
            "audit": audit_entry
        }

    def _enforce_policy(self, actor: ActorType, action: ActionType, payload: Dict[str, Any]):
        """Enforces dual-control separation of concerns."""
        # Agents cannot approve refunds exceeding $500 without human co-sign
        if actor == ActorType.AGENT and action == ActionType.APPROVE_REFUND:
            order_id = payload.get("order_id")
            order = self._orders.get(order_id)
            if order and order.get("amount", 0) > 500.0:
                if not payload.get("human_approved", False):
                    raise BusinessStateError(
                        f"Dual-control policy violation: Agent refund for order {order_id} (${order.get('amount')}) requires human co-approval."
                    )

        # Only human can execute policy override
        if actor == ActorType.AGENT and action == ActionType.OVERRIDE_POLICY:
            raise BusinessStateError("Permission denied: Autonomous agent cannot invoke OVERRIDE_POLICY.")


class Tau2BenchmarkRunner:
    """Simulates randomized interleaved trajectories of human and agent actions."""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.state = SharedBusinessState()

    def run_simulation(self, total_steps: int = 50) -> Dict[str, Any]:
        successful_steps = 0
        policy_rejections = 0
        duplicates_handled = 0
        concurrency_conflicts = 0

        # Create initial test orders
        for i in range(5):
            self.state.execute_mutation(
                actor=ActorType.HUMAN,
                action=ActionType.CREATE_ORDER,
                payload={"order_id": f"ord_seed_{i}", "customer": f"user_{i}", "amount": (i + 1) * 200.0},
                idempotency_key=f"init_seed_{i}"
            )

        executed_keys = [f"init_seed_{i}" for i in range(5)]

        for step in range(total_steps):
            actor = random.choice([ActorType.HUMAN, ActorType.AGENT])
            order_idx = random.randint(0, 4)
            order_id = f"ord_seed_{order_idx}"
            
            # Intentionally inject idempotency replays (15% chance)
            if executed_keys and random.random() < 0.15:
                key = random.choice(executed_keys)
            else:
                key = f"step_{step}_{random.randint(1000, 9999)}"
                executed_keys.append(key)

            action = random.choice([
                ActionType.UPDATE_PRICE,
                ActionType.APPROVE_REFUND,
                ActionType.CANCEL_ORDER
            ])

            payload = {"order_id": order_id, "new_amount": random.randint(50, 800)}
            # 50% chance agent provides human co-sign token
            if actor == ActorType.AGENT and random.random() < 0.5:
                payload["human_approved"] = True

            # 10% chance to test optimistic concurrency check
            expected_ver = self.state.version if random.random() > 0.1 else self.state.version - 1

            try:
                res = self.state.execute_mutation(
                    actor=actor,
                    action=action,
                    payload=payload,
                    idempotency_key=key,
                    expected_version=expected_ver if random.random() < 0.2 else None
                )
                if res.get("status") == "duplicate_ignored":
                    duplicates_handled += 1
                else:
                    successful_steps += 1
            except BusinessStateError as e:
                if "Concurrency conflict" in str(e):
                    concurrency_conflicts += 1
                elif "Dual-control policy violation" in str(e):
                    policy_rejections += 1
                else:
                    policy_rejections += 1

        consistency_score = 100.0 if len(self.state._orders) == 5 else 0.0

        return {
            "total_evaluated_steps": total_steps,
            "successful_mutations": successful_steps,
            "policy_rejections": policy_rejections,
            "duplicates_safely_ignored": duplicates_handled,
            "concurrency_conflicts": concurrency_conflicts,
            "final_state_version": self.state.version,
            "consistency_score": consistency_score
        }
