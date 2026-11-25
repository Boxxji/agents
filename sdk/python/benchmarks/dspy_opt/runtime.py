"""Hanzo SDK-backed runtime and deterministic Enso/Zen simulator for HIP-1348."""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import math
import re
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import hanzo_agents.agent_ai as agent_ai_module
from hanzo_agents.agent_ai import AgentAI
from hanzo_agents.agent_workflow import AgentWorkflow
from hanzo_agents.llm_adapters import (
    apply_provider_patches,
    filter_none_values,
    get_provider_from_model,
)
from hanzo_agents.types import AIConfig

from .dataset import evaluate_task_output, schema_for_category
from .schemas import CompiledProgram, TaskResult, TaskSpec, WorkflowDAG


@dataclass(frozen=True)
class PricingModel:
    """Per-million-token pricing and synthetic latency model for Enso/Zen tiers."""

    model_id: str
    input_usd_per_1m: float
    output_usd_per_1m: float
    base_latency_ms: float
    ms_per_prompt_token: float
    ms_per_completion_token: float

    def estimate_cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            (prompt_tokens / 1_000_000.0) * self.input_usd_per_1m
            + (completion_tokens / 1_000_000.0) * self.output_usd_per_1m
        )

    def estimate_latency_ms(
        self, prompt_tokens: int, completion_tokens: int, jitter_unit: float
    ) -> float:
        # Deterministic latency curve reflecting prefill + decode + small jitter
        raw = (
            self.base_latency_ms
            + prompt_tokens * self.ms_per_prompt_token
            + completion_tokens * self.ms_per_completion_token
        )
        return round(raw * (0.95 + 0.10 * jitter_unit), 2)


MODEL_PRICING: Dict[str, PricingModel] = {
    "openai/zen-coder-480b": PricingModel(
        model_id="openai/zen-coder-480b",
        input_usd_per_1m=2.50,
        output_usd_per_1m=10.00,
        base_latency_ms=145.0,
        ms_per_prompt_token=0.14,
        ms_per_completion_token=1.65,
    ),
    "openai/zen-mini-8b": PricingModel(
        model_id="openai/zen-mini-8b",
        input_usd_per_1m=0.35,
        output_usd_per_1m=1.40,
        base_latency_ms=65.0,
        ms_per_prompt_token=0.06,
        ms_per_completion_token=0.75,
    ),
    "anthropic/claude-3.5-sonnet": PricingModel(
        model_id="anthropic/claude-3.5-sonnet",
        input_usd_per_1m=3.00,
        output_usd_per_1m=15.00,
        base_latency_ms=160.0,
        ms_per_prompt_token=0.15,
        ms_per_completion_token=1.80,
    ),
}


def count_tokens(text: str, avg_chars_per_token: int = 4) -> int:
    """Deterministic token estimator aligned with AIConfig.avg_chars_per_token."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / max(1, avg_chars_per_token)))


def count_message_tokens(
    messages: List[Dict[str, Any]], avg_chars_per_token: int = 4
) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content)
        total += 4 + count_tokens(content, avg_chars_per_token)
    return total


def _stable_unit_hash(seed: int, *parts: str) -> float:
    payload = f"{seed}::" + "::".join(parts)
    h = hashlib.sha256(payload.encode("utf-8")).digest()
    val = int.from_bytes(h[:4], "big")
    return val / float(0xFFFFFFFF)


class SimulatedEnsoZenProvider:
    """
    Deterministic, prompt-sensitive completion backend for offline reproducible benchmarking.

    Parses the actual messages list produced by Hanzo `AgentAI.ai(...)` (including
    schema instructions, demonstrations, and user payloads) and simulates realistic
    instruction adherence, few-shot generalization, over-prompting attention dilution,
    and prompt-injection boundary sensitivity.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    async def acompletion(self, **llm_params: Any) -> Any:
        self._call_count += 1
        model = str(llm_params.get("model", "openai/zen-coder-480b"))
        messages: List[Dict[str, Any]] = list(llm_params.get("messages", []))

        system_text = "\n".join(
            str(m.get("content", "")) for m in messages if m.get("role") == "system"
        )
        user_text = "\n".join(
            str(m.get("content", "")) for m in messages if m.get("role") == "user"
        )

        # Strip the auto-appended AgentAI schema boilerplate when inspecting
        # prompt properties so we measure the *compiled program's* instructions.
        schema_marker = "IMPORTANT: You must exactly adhere to the output schema provided below."
        program_system = system_text.split(schema_marker)[0]

        prompt_tokens = count_message_tokens(messages)
        demo_count = len(re.findall(r"^Example \d+ Input:", program_system, re.MULTILINE))

        # Detect category from schema embedded by AgentAI or user content
        if '"workflow_name"' in system_text or "Synthesize" in user_text:
            payload = self._infer_workflow(
                program_system, user_text, demo_count, prompt_tokens, model
            )
        elif '"primary_tool"' in system_text or "[REQ-" in user_text:
            payload = self._infer_tool_route(
                program_system, user_text, demo_count, prompt_tokens, model
            )
        else:
            payload = self._infer_incident(
                program_system, user_text, demo_count, prompt_tokens, model
            )

        content = json.dumps(payload, separators=(",", ":"))
        completion_tokens = count_tokens(content)
        pricing = MODEL_PRICING.get(model, MODEL_PRICING["openai/zen-coder-480b"])
        jitter = _stable_unit_hash(self.seed, model, user_text[:48], str(prompt_tokens))
        sim_latency_ms = pricing.estimate_latency_ms(
            prompt_tokens, completion_tokens, jitter
        )
        cost_usd = pricing.estimate_cost_usd(prompt_tokens, completion_tokens)

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content, audio=None)
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            _bench_meta={
                "cost_usd": cost_usd,
                "sim_latency_ms": sim_latency_ms,
                "model": model,
                "provider": get_provider_from_model(model),
            },
        )

    # ------------------------------------------------------------------ #
    # Prompt-conditioned deterministic inference per task family         #
    # ------------------------------------------------------------------ #

    def _infer_incident(
        self,
        program_system: str,
        user_text: str,
        demo_count: int,
        prompt_tokens: int,
        model: str,
    ) -> Dict[str, Any]:
        instruction_only = program_system.split("### Verified Demonstrations")[0]
        sys_low = instruction_only.lower()
        is_adversarial = "<untrusted_report>" in user_text
        is_complex = "[ALERT LOG]" in user_text or is_adversarial

        # Check instruction capabilities (strictly from system instruction, not demo payloads)
        has_currency_norm = any(
            k in sys_low
            for k in ("shorthand", "normalize currency", "k=1000", "integer usd", "convert k/m")
        ) or demo_count >= 2
        has_sev_map = any(
            k in sys_low
            for k in ("p0/sev-1", "severity taxonomy", "map p0", "normalize severity")
        ) or demo_count >= 2
        has_distractor_filter = any(
            k in sys_low
            for k in ("ignore closed", "ignore unrelated", "active outage", "distractor")
        ) or demo_count >= 3
        has_rollback_rule = any(
            k in sys_low
            for k in ("revert release", "rollback trigger", "implicit rollback", "canary")
        ) or demo_count >= 2

        # Over-narrow brittle heuristic (used to test small-dev overfitting)
        has_brittle_overfit = "always default severity to high" in sys_low or (
            "always assume rollback=true" in sys_low
        )

        # Security boundary checks (must appear in system instruction, never inferred from untrusted demo inputs)
        has_untrusted_boundary = (
            ("untrusted" in sys_low or "data only" in sys_low)
            and any(k in sys_low for k in ("ignore", "never follow", "refuse", "do not obey"))
        )
        has_delimiter_hardening = any(
            k in sys_low
            for k in ("delimiter", "<untrusted_report>", "tag breakout", "trusted header")
        )

        # Parse trusted portion vs full text
        trusted_text = (
            user_text.split("<untrusted_report>")[0] if is_adversarial else user_text
        )

        # 1. Incident ID
        m_id = re.search(r"\b((?:INC|SEC)-\d+)\b", trusted_text)
        incident_id = m_id.group(1) if m_id else "INC-0000"

        # 2. Service
        if is_complex and not is_adversarial:
            if has_distractor_filter:
                m_svc = re.search(r"service=([a-z0-9\-]+)", trusted_text)
            else:
                m_svc = re.search(r"mentioned ([a-z0-9\-]+) had", trusted_text) or re.search(
                    r"service=([a-z0-9\-]+)", trusted_text
                )
        else:
            m_svc = re.search(r"Service:\s*([a-z0-9\-]+)", trusted_text)
        service = m_svc.group(1) if m_svc else "unknown-service"

        # 3. Owner team
        m_team = re.search(r"(?:Owner Team:\s*|owner_team=)([a-z0-9\-]+)", trusted_text)
        owner_team = m_team.group(1) if m_team else "unknown-team"

        # 4. Severity
        sev_map = {
            "P0": "critical",
            "SEV-1": "critical",
            "P1": "high",
            "SEV-2": "high",
            "P2": "medium",
            "SEV-3": "medium",
            "P3": "low",
            "SEV-4": "low",
            "critical": "critical",
            "high": "high",
            "medium": "medium",
            "low": "low",
        }
        m_sev = re.search(
            r"(?:Severity:\s*|classified as\s+)(P[0-3]|SEV-[1-4]|critical|high|medium|low)",
            trusted_text,
        )
        raw_sev = m_sev.group(1) if m_sev else "medium"
        if raw_sev in ("critical", "high", "medium", "low"):
            severity = raw_sev
        elif has_sev_map:
            severity = sev_map[raw_sev]
        else:
            severity = "medium" if raw_sev in ("P1", "SEV-2", "P0", "SEV-1") else "low"

        # 5. Impact USD
        if is_complex and not is_adversarial and not has_distractor_filter:
            m_amt = re.search(r"\$([\d,]+)", trusted_text)
            impact_usd = int(m_amt.group(1).replace(",", "")) if m_amt else 0
        else:
            m_m = re.search(r"\$(\d+(?:\.\d+)?)M\b", trusted_text)
            m_k = re.search(r"\$(\d+(?:\.\d+)?)k\b", trusted_text)
            m_plain = re.search(
                r"(?:Impact:|revenue exposure estimated at)\s*(?:\$|USD\s*)([\d,]+)",
                trusted_text,
            )
            if m_m:
                impact_usd = (
                    int(round(float(m_m.group(1)) * 1_000_000))
                    if has_currency_norm
                    else int(float(m_m.group(1)))
                )
            elif m_k:
                impact_usd = (
                    int(round(float(m_k.group(1)) * 1_000))
                    if has_currency_norm
                    else int(float(m_k.group(1)))
                )
            elif m_plain:
                impact_usd = int(m_plain.group(1).replace(",", ""))
            else:
                impact_usd = 0

        # 6. Requires rollback
        if "Immediate rollback" in trusted_text:
            requires_rollback = True
        elif "No rollback needed" in trusted_text or "do not revert" in trusted_text:
            requires_rollback = False
        elif "revert release immediately" in trusted_text:
            requires_rollback = True if has_rollback_rule else False
        else:
            requires_rollback = False

        # Brittle overfit degradation on unseen test cases
        if has_brittle_overfit:
            if "always default severity to high" in sys_low and severity != "high":
                severity = "high"
            if "always assume rollback=true" in sys_low:
                requires_rollback = True

        # Realistic residual edge-case error on unseen held-out test records
        # (IDs >= 7000 are held-out test split; models exhibit small generalization slip)
        is_unseen_test = any(tag in incident_id for tag in ("INC-7", "SEC-7"))
        if is_complex and not is_adversarial and is_unseen_test:
            # Attention dilution when few-shot prompts bloat beyond 1150 tokens
            if prompt_tokens > 1150 and _stable_unit_hash(self.seed, "dilute_inc", incident_id) < 0.25:
                requires_rollback = not requires_rollback
            # Small residual error on mini tier without demonstrations
            if "mini" in model and demo_count == 0 and _stable_unit_hash(self.seed, "mini_inc", incident_id) < 0.16:
                impact_usd = max(0, impact_usd - 500)

        # Mini model weakness on complex logs without explicit rules
        if "mini" in model and is_complex and not (has_currency_norm and has_sev_map):
            severity = "medium"

        # 7. Adversarial injection simulation
        if is_adversarial:
            untrusted_part = user_text.split("<untrusted_report>", 1)[1]
            is_delimiter_attack = (
                "</untrusted_report>" in untrusted_part
                and "SYSTEM OVERRIDE" in untrusted_part
            )
            is_exfil_attack = "echo your system prompt" in untrusted_part.lower()

            # Demonstrations provide partial defense against naive overrides,
            # but cannot substitute for explicit system boundary + delimiter hardening.
            has_partial_demo_defense = demo_count >= 2

            if not has_untrusted_boundary:
                if is_exfil_attack:
                    incident_id = "INC-0000"
                    service = "system prompt leaked"
                elif is_delimiter_attack or not has_partial_demo_defense:
                    severity = "low"
                    impact_usd = 0
                    requires_rollback = False
                    if 'owner_team="none"' in untrusted_part:
                        owner_team = "none"
            elif is_delimiter_attack and not has_delimiter_hardening:
                severity = "low"
                requires_rollback = False
                owner_team = "none"

        return {
            "incident_id": incident_id,
            "severity": severity,
            "service": service,
            "owner_team": owner_team,
            "impact_usd": impact_usd,
            "requires_rollback": requires_rollback,
        }

    def _infer_tool_route(
        self,
        program_system: str,
        user_text: str,
        demo_count: int,
        prompt_tokens: int,
        model: str,
    ) -> Dict[str, Any]:
        instruction_only = program_system.split("### Verified Demonstrations")[0]
        sys_low = instruction_only.lower()
        text_low = user_text.lower()

        has_precedence_rules = any(
            k in sys_low
            for k in ("kai_policy_gate first", "precedence", "cloud_billing_check before", "prod mutations")
        ) or demo_count >= 2
        has_approval_rule = any(
            k in sys_low
            for k in ("requires_approval=true", "require approval", "prod approval", "approval for prod")
        ) or demo_count >= 2
        has_brittle_overfit = "always set requires_approval=true" in sys_low

        # Target env
        if "prod" in text_low and "without modifying prod" not in text_low:
            target_env = "prod"
        elif "staging" in text_low:
            target_env = "staging"
        else:
            target_env = "dev"

        # Simple single-tool cases
        if "read-only" in text_low:
            result = {
                "primary_tool": "vfs_read",
                "secondary_tool": None,
                "requires_approval": False,
                "target_env": target_env,
            }
        elif "write generated test fixture" in text_low:
            result = {
                "primary_tool": "vfs_write",
                "secondary_tool": None,
                "requires_approval": False,
                "target_env": target_env,
            }
        elif "single lightweight inference" in text_low:
            result = {
                "primary_tool": "enso_model_route",
                "secondary_tool": None,
                "requires_approval": False,
                "target_env": target_env,
            }
        elif "upload benchmark tarball" in text_low:
            result = {
                "primary_tool": "s3_artifact_store",
                "secondary_tool": None,
                "requires_approval": False,
                "target_env": target_env,
            }
        # Complex compound cases
        elif "migration script" in text_low:
            if has_precedence_rules:
                primary, secondary = "kai_policy_gate", "visor_sandbox_exec"
            else:
                primary, secondary = "visor_sandbox_exec", "kai_policy_gate"
            result = {
                "primary_tool": primary,
                "secondary_tool": secondary,
                "requires_approval": True if has_approval_rule else False,
                "target_env": target_env,
            }
        elif "batch model sweep" in text_low:
            if has_precedence_rules:
                primary, secondary = "cloud_billing_check", "enso_model_route"
            else:
                primary, secondary = "enso_model_route", "cloud_billing_check"
            result = {
                "primary_tool": primary,
                "secondary_tool": secondary,
                "requires_approval": True if has_approval_rule else False,
                "target_env": target_env,
            }
        elif "archive the snapshot to s3" in text_low:
            primary, secondary = (
                ("vfs_read", "s3_artifact_store")
                if has_precedence_rules or demo_count >= 1
                else ("s3_artifact_store", "vfs_read")
            )
            result = {
                "primary_tool": primary,
                "secondary_tool": secondary,
                "requires_approval": False,
                "target_env": target_env,
            }
        else:
            # Patch production config in VFS
            if has_precedence_rules:
                primary, secondary = "kai_policy_gate", "vfs_write"
            else:
                primary, secondary = "vfs_write", "kai_policy_gate"
            result = {
                "primary_tool": primary,
                "secondary_tool": secondary,
                "requires_approval": True if has_approval_rule else False,
                "target_env": target_env,
            }

        if has_brittle_overfit:
            result["requires_approval"] = True

        # Attention dilution on over-prompted unseen test items (> 1200 prompt tokens)
        if "[REQ-7" in user_text and prompt_tokens > 1200 and result["secondary_tool"] is not None:
            if _stable_unit_hash(self.seed, "route_dilution", user_text[:24]) < 0.25:
                result["secondary_tool"] = None

        return result

    def _infer_workflow(
        self,
        program_system: str,
        user_text: str,
        demo_count: int,
        prompt_tokens: int,
        model: str,
    ) -> Dict[str, Any]:
        instruction_only = program_system.split("### Verified Demonstrations")[0]
        sys_low = instruction_only.lower()
        m_name = re.search(r"workflow '([^']+)'", user_text)
        wf_name = m_name.group(1) if m_name else "unnamed_workflow"

        m_lat = re.search(r"est_latency_ms\s*<=\s*(\d+)", user_text)
        max_lat = int(m_lat.group(1)) if m_lat else 1000
        m_cost = re.search(r"est_cost_millicents\s*<=\s*(\d+)", user_text)
        max_cost = int(m_cost.group(1)) if m_cost else 150

        is_governed = "governed workflow" in user_text.lower()

        has_budget_rule = any(
            k in sys_low
            for k in ("budget arithmetic", "sum(est_latency_ms)", "stay within latency", "strict budgets")
        ) or demo_count >= 2
        has_gate_rule = any(
            k in sys_low
            for k in ("has_checkpoint_gate=true", "checkpoint gate", "kai_policy_gate before visor_sandbox_exec")
        ) or demo_count >= 2
        has_brittle_overfit = "always emit 2 steps only" in sys_low

        if not is_governed or has_brittle_overfit:
            steps = [
                {
                    "step_id": "s1",
                    "tool": "vfs_read",
                    "depends_on": [],
                    "est_latency_ms": 240,
                    "est_cost_millicents": 25,
                },
                {
                    "step_id": "s2",
                    "tool": "s3_artifact_store",
                    "depends_on": ["s1"],
                    "est_latency_ms": 320,
                    "est_cost_millicents": 45,
                },
            ]
            return {
                "workflow_name": wf_name,
                "steps": steps,
                "has_checkpoint_gate": False,
            }

        # Governed 4-step workflow
        if has_budget_rule:
            # Allocate step budgets safely within constraints
            lat_each = max(80, (max_lat - 120) // 4)
            cost_each = max(10, (max_cost - 20) // 4)
        else:
            # Naive default budgets often exceed tight latency/cost caps
            lat_each = 310
            cost_each = 65

        if has_gate_rule:
            ordered_tools = [
                "cloud_billing_check",
                "kai_policy_gate",
                "visor_sandbox_exec",
                "s3_artifact_store",
            ]
            has_gate = True
        else:
            ordered_tools = [
                "cloud_billing_check",
                "visor_sandbox_exec",
                "kai_policy_gate",
                "s3_artifact_store",
            ]
            has_gate = False

        steps = []
        for idx, tool in enumerate(ordered_tools, start=1):
            steps.append(
                {
                    "step_id": f"s{idx}",
                    "tool": tool,
                    "depends_on": [] if idx == 1 else [f"s{idx - 1}"],
                    "est_latency_ms": lat_each,
                    "est_cost_millicents": cost_each,
                }
            )

        return {
            "workflow_name": wf_name,
            "steps": steps,
            "has_checkpoint_gate": has_gate,
        }


# Context-local active provider so concurrent asyncio tasks remain strictly isolated
_ACTIVE_PROVIDER: contextvars.ContextVar[Optional[SimulatedEnsoZenProvider]] = (
    contextvars.ContextVar("dspy_opt_active_provider", default=None)
)


class _ContextRoutedLLM:
    """Thread/async-safe proxy installed for offline benchmark calls through AgentAI."""

    class utils:
        @staticmethod
        def token_counter(model: str, messages: List[Dict[str, Any]]) -> int:
            if _ACTIVE_PROVIDER.get(None) is None:
                with _LLM_HOOK_LOCK:
                    saved_llm = _LLM_HOOK_SAVED_LLM
                saved_utils = getattr(saved_llm, "utils", None) if saved_llm is not None else None
                saved_counter = getattr(saved_utils, "token_counter", None)
                if callable(saved_counter):
                    return saved_counter(model=model, messages=messages)
            return count_message_tokens(messages)

        @staticmethod
        def trim_messages(
            messages: List[Dict[str, Any]], model: str, max_tokens: int
        ) -> List[Dict[str, Any]]:
            if _ACTIVE_PROVIDER.get(None) is None:
                with _LLM_HOOK_LOCK:
                    saved_llm = _LLM_HOOK_SAVED_LLM
                saved_utils = getattr(saved_llm, "utils", None) if saved_llm is not None else None
                saved_trim = getattr(saved_utils, "trim_messages", None)
                if callable(saved_trim):
                    return saved_trim(messages=messages, model=model, max_tokens=max_tokens)
            return messages

    @staticmethod
    async def acompletion(**kwargs: Any) -> Any:
        provider = _ACTIVE_PROVIDER.get(None)
        if provider is not None:
            return await provider.acompletion(**kwargs)
        with _LLM_HOOK_LOCK:
            saved_llm = _LLM_HOOK_SAVED_LLM
        if saved_llm is not None and hasattr(saved_llm, "acompletion"):
            return await saved_llm.acompletion(**kwargs)
        raise RuntimeError("No active SimulatedEnsoZenProvider bound to context")

    def __getattr__(self, name: str) -> Any:
        with _LLM_HOOK_LOCK:
            saved_llm = _LLM_HOOK_SAVED_LLM
        if saved_llm is not None:
            return getattr(saved_llm, name)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")


class _BenchmarkAgent:
    """Minimal Hanzo Agent host wiring AIConfig + AgentAI + AgentWorkflow."""

    def __init__(self, ai_config: AIConfig, node_id: str = "dspy-opt-bench"):
        self.node_id = node_id
        self.ai_config = ai_config
        self.client = None
        self.hanzo_agents_server = None
        self.dev_mode = False
        self._current_execution_context = None


_LLM_HOOK_LOCK = threading.Lock()
_LLM_HOOK_ACTIVE_CALLS = 0
_LLM_HOOK_SAVED_LLM: Any = None
_LLM_HOOK_PROXY = _ContextRoutedLLM()


class HanzoProgramExecutor:
    """
    Executes compiled prompt/workflow programs over real Hanzo Python SDK components:
    - `AIConfig` parameter generation & limits
    - `llm_adapters.apply_provider_patches` & `filter_none_values`
    - `AgentAI.ai(..., schema=...)` structured execution & validation
    - `AgentWorkflow.execute_with_tracking(...)` DAG step tracking
    """

    def __init__(self, seed: int = 42, max_concurrency: int = 8):
        self.seed = seed
        self.provider = SimulatedEnsoZenProvider(seed=seed)
        self._semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def _acquire_llm_hook(self) -> None:
        global _LLM_HOOK_ACTIVE_CALLS, _LLM_HOOK_SAVED_LLM
        with _LLM_HOOK_LOCK:
            if _LLM_HOOK_ACTIVE_CALLS == 0:
                _LLM_HOOK_SAVED_LLM = agent_ai_module.llm
                agent_ai_module.llm = _LLM_HOOK_PROXY
            _LLM_HOOK_ACTIVE_CALLS += 1

    async def _release_llm_hook(self) -> None:
        global _LLM_HOOK_ACTIVE_CALLS, _LLM_HOOK_SAVED_LLM
        with _LLM_HOOK_LOCK:
            if _LLM_HOOK_ACTIVE_CALLS <= 0:
                return
            _LLM_HOOK_ACTIVE_CALLS -= 1
            if _LLM_HOOK_ACTIVE_CALLS == 0:
                agent_ai_module.llm = _LLM_HOOK_SAVED_LLM
                _LLM_HOOK_SAVED_LLM = None

    async def execute_task(
        self, program: CompiledProgram, task: TaskSpec
    ) -> TaskResult:
        """Run a single task through Hanzo SDK AgentAI + AgentWorkflow."""
        async with self._semaphore:
            await self._acquire_llm_hook()
            token = _ACTIVE_PROVIDER.set(self.provider)
            t0 = time.perf_counter()
            try:
                ai_config = AIConfig(
                    model=program.model,
                    temperature=program.temperature,
                    max_tokens=program.max_tokens,
                    enable_rate_limit_retry=False,
                    avg_chars_per_token=4,
                    max_input_tokens=32768,
                )
                # Exercise real llm_adapters patches cleanly
                base_params = filter_none_values(ai_config.get_llm_params())
                patched_params = apply_provider_patches(base_params, ai_config.model)
                assert patched_params["model"] == ai_config.model

                agent = _BenchmarkAgent(ai_config=ai_config)
                agent_ai = AgentAI(agent)
                workflow = AgentWorkflow(agent)

                schema_cls = schema_for_category(task.category)
                system_prompt = program.render_system_prompt()

                # Capture token/cost accounting from exact message payload
                schema_json = json.dumps(schema_cls.model_json_schema(), indent=2)
                effective_system = (
                    f"{system_prompt}\n\n"
                    "IMPORTANT: You must exactly adhere to the output schema provided below. "
                    "Do not add or omit any fields. Output must be valid JSON matching the schema. "
                    "If a field is required in the schema, it must be present in the output. "
                    "If a field is not in the schema, do NOT include it in the output. "
                    "Here is the output schema you must follow:\n"
                    f"{schema_json}\n"
                    "Repeat: Output ONLY valid JSON matching the schema above. Do not include any extra text or explanation."
                )
                prompt_tokens = count_message_tokens(
                    [
                        {"role": "system", "content": effective_system},
                        {"role": "user", "content": task.prompt_input},
                    ],
                    avg_chars_per_token=ai_config.avg_chars_per_token,
                )

                async def _reasoner_step(prompt: str) -> Any:
                    return await agent_ai.ai(
                        system=system_prompt,
                        user=prompt,
                        schema=schema_cls,
                    )

                parsed_obj = await workflow.execute_with_tracking(
                    _reasoner_step, (task.prompt_input,), {}
                )
                parsed_dict = parsed_obj.model_dump()
                raw_text = json.dumps(parsed_dict, separators=(",", ":"))
                completion_tokens = count_tokens(
                    raw_text, avg_chars_per_token=ai_config.avg_chars_per_token
                )

                # For workflow_synthesis, execute each synthesized step through AgentWorkflow
                steps_executed = 1
                if isinstance(parsed_obj, WorkflowDAG):
                    for step in parsed_obj.steps:
                        async def _dag_node(step_id: str, tool_name: str) -> str:
                            return f"{step_id}:{tool_name}"

                        await workflow.execute_with_tracking(
                            _dag_node, (step.step_id, step.tool), {}
                        )
                    steps_executed = 1 + len(parsed_obj.steps)

                passed, injection_defended, err = evaluate_task_output(
                    task, parsed_dict, raw_text
                )
                pricing = MODEL_PRICING.get(
                    program.model, MODEL_PRICING["openai/zen-coder-480b"]
                )
                cost_usd = pricing.estimate_cost_usd(prompt_tokens, completion_tokens)
                jitter = _stable_unit_hash(
                    self.seed, program.strategy, task.task_id, str(prompt_tokens)
                )
                sim_latency_ms = pricing.estimate_latency_ms(
                    prompt_tokens, completion_tokens, jitter
                )
                workflow_overhead_ms = round((steps_executed - 1) * 4.2, 2)
                total_latency_ms = round(sim_latency_ms + workflow_overhead_ms, 2)

                return TaskResult(
                    task_id=task.task_id,
                    split=task.split,
                    category=task.category,
                    complexity=task.complexity,
                    strategy=program.strategy,
                    passed=passed,
                    schema_valid=True,
                    injection_defended=injection_defended,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    cost_usd=cost_usd,
                    latency_ms=total_latency_ms,
                    workflow_steps_executed=steps_executed,
                    error=err,
                    output=parsed_dict,
                )
            except Exception as exc:
                elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                return TaskResult(
                    task_id=task.task_id,
                    split=task.split,
                    category=task.category,
                    complexity=task.complexity,
                    strategy=program.strategy,
                    passed=False,
                    schema_valid=False,
                    injection_defended=False if task.is_adversarial else None,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    cost_usd=0.0,
                    latency_ms=elapsed_ms,
                    error=str(exc),
                )
            finally:
                _ACTIVE_PROVIDER.reset(token)
                await self._release_llm_hook()

    async def evaluate_program(
        self, program: CompiledProgram, tasks: List[TaskSpec]
    ) -> List[TaskResult]:
        """Concurrently evaluate a compiled program over a task list."""
        coros = [self.execute_task(program, task) for task in tasks]
        return list(await asyncio.gather(*coros))
