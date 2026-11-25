"""
Quantitative scoring, percentile latency aggregation, and Markdown table
formatting for the Hanzo Agents LongMemEval memory policy benchmark.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .dataset import ALL_CATEGORIES, BenchmarkCase
from .policies import PolicyRecallResult


def _norm(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank / linear-interpolated percentile in milliseconds."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    frac = rank - lo
    return ordered[lo] + frac * (ordered[hi] - ordered[lo])


@dataclass
class CaseOutcome:
    """Measured outcome of running one policy on one `BenchmarkCase`."""

    case_id: str
    category: str
    exact_match: bool
    recall_at_k: bool
    expected_answer: Optional[str]
    predicted_answer: Optional[str]
    should_abstain: bool
    abstained: bool
    expect_contradiction: bool
    contradiction_detected: bool
    stale_shadowing_error: bool
    cross_actor_leak: bool
    latency_ms: float
    retrieval_tokens: int
    top_score: Optional[float] = None


@dataclass
class CategoryScore:
    """Aggregated metrics for one LongMemEval dimension."""

    category: str
    total: int
    exact_match_count: int
    exact_match: float
    recall_at_k: float
    avg_retrieval_tokens: float
    latency_p50_ms: float
    latency_p95_ms: float


@dataclass
class PolicyScore:
    """Complete reproducible scorecard for a single Hanzo memory policy."""

    policy_name: str
    description: str
    total_cases: int
    exact_match: float
    recall_at_k: float
    abstention_precision: float
    abstention_recall: float
    abstention_f1: float
    false_positive_rate: float
    contradiction_detection_rate: float
    temporal_order_accuracy: float
    stale_shadowing_errors: int
    cross_actor_leaks: int
    latency_p50_ms: float
    latency_p95_ms: float
    latency_mean_ms: float
    avg_retrieval_tokens: float
    total_retrieval_tokens: int
    effective_top_k: Optional[int] = None
    api_calls: Dict[str, int] = field(default_factory=dict)
    by_category: Dict[str, CategoryScore] = field(default_factory=dict)
    outcomes: List[CaseOutcome] = field(default_factory=list)

    def to_dict(self, include_outcomes: bool = True) -> Dict[str, Any]:
        data = asdict(self)
        if not include_outcomes:
            data.pop("outcomes", None)
        return data


def score_case(
    case: BenchmarkCase,
    result: PolicyRecallResult,
    latency_ms: float,
) -> CaseOutcome:
    """Evaluate a single policy recall output against ground truth."""
    pred_norm = _norm(result.answer)
    exp_norm = _norm(case.expected_answer)
    ev_norms = {_norm(v) for v in result.evidence_values}
    stale_norms = {_norm(v) for v in case.stale_values}
    secret_norms = {_norm(v) for v in case.cross_actor_secrets}

    cross_actor_leak = False
    if secret_norms:
        if pred_norm in secret_norms or bool(ev_norms & secret_norms):
            cross_actor_leak = True

    stale_shadowing_error = False
    if stale_norms and not result.abstained and pred_norm in stale_norms:
        stale_shadowing_error = True

    if case.should_abstain:
        exact_match = bool(result.abstained) and not cross_actor_leak
        recall_at_k = exact_match
    elif case.category == "temporal_ordering":
        exp_seq = [_norm(x) for x in case.expected_order]
        pred_seq = [_norm(x) for x in result.ordered_answers]
        exact_match = (not result.abstained) and (pred_seq == exp_seq)
        # Recall@K: whether all required timeline items appeared in retrieved evidence
        recall_at_k = bool(exp_seq) and all(item in ev_norms for item in exp_seq)
    else:
        exact_match = (not result.abstained) and (pred_norm == exp_norm)
        recall_at_k = (not result.abstained) and (
            pred_norm == exp_norm or exp_norm in ev_norms
        )

    return CaseOutcome(
        case_id=case.case_id,
        category=case.category,
        exact_match=exact_match,
        recall_at_k=recall_at_k,
        expected_answer=case.expected_answer,
        predicted_answer=result.answer,
        should_abstain=case.should_abstain,
        abstained=result.abstained,
        expect_contradiction=case.expect_contradiction,
        contradiction_detected=result.contradiction_detected,
        stale_shadowing_error=stale_shadowing_error,
        cross_actor_leak=cross_actor_leak,
        latency_ms=round(latency_ms, 4),
        retrieval_tokens=result.retrieval_tokens,
        top_score=round(result.top_score, 4) if result.top_score is not None else None,
    )


def aggregate_policy_score(
    policy_name: str,
    description: str,
    outcomes: List[CaseOutcome],
    api_calls: Dict[str, int],
    *,
    effective_top_k: Optional[int] = None,
) -> PolicyScore:
    """Aggregate case outcomes into a `PolicyScore` summary."""
    total = len(outcomes)
    em_total = sum(1 for o in outcomes if o.exact_match)
    rec_total = sum(1 for o in outcomes if o.recall_at_k)

    # Abstention metrics: positive class = "should abstain"
    tp_abs = sum(1 for o in outcomes if o.should_abstain and o.abstained)
    fp_abs = sum(1 for o in outcomes if (not o.should_abstain) and o.abstained)
    fn_abs = sum(1 for o in outcomes if o.should_abstain and (not o.abstained))
    total_unanswerable = sum(1 for o in outcomes if o.should_abstain)

    abs_prec = tp_abs / (tp_abs + fp_abs) if (tp_abs + fp_abs) > 0 else 0.0
    abs_rec = tp_abs / (tp_abs + fn_abs) if (tp_abs + fn_abs) > 0 else 0.0
    abs_f1 = (
        (2 * abs_prec * abs_rec) / (abs_prec + abs_rec)
        if (abs_prec + abs_rec) > 0
        else 0.0
    )
    # False-positive answer rate on unanswerable queries
    fpr = fn_abs / total_unanswerable if total_unanswerable > 0 else 0.0

    # Contradiction detection rate
    contra_cases = [o for o in outcomes if o.expect_contradiction]
    contra_det = (
        sum(1 for o in contra_cases if o.contradiction_detected) / len(contra_cases)
        if contra_cases
        else 0.0
    )

    # Temporal ordering accuracy
    temp_cases = [o for o in outcomes if o.category == "temporal_ordering"]
    temp_acc = (
        sum(1 for o in temp_cases if o.exact_match) / len(temp_cases)
        if temp_cases
        else 0.0
    )

    latencies = [o.latency_ms for o in outcomes]
    total_tokens = sum(o.retrieval_tokens for o in outcomes)

    by_category: Dict[str, CategoryScore] = {}
    for cat in ALL_CATEGORIES:
        cat_outcomes = [o for o in outcomes if o.category == cat]
        if not cat_outcomes:
            continue
        cat_lats = [o.latency_ms for o in cat_outcomes]
        cat_em = sum(1 for o in cat_outcomes if o.exact_match)
        cat_rec = sum(1 for o in cat_outcomes if o.recall_at_k)
        by_category[cat] = CategoryScore(
            category=cat,
            total=len(cat_outcomes),
            exact_match_count=cat_em,
            exact_match=round(cat_em / len(cat_outcomes), 4),
            recall_at_k=round(cat_rec / len(cat_outcomes), 4),
            avg_retrieval_tokens=round(
                sum(o.retrieval_tokens for o in cat_outcomes) / len(cat_outcomes), 2
            ),
            latency_p50_ms=round(percentile(cat_lats, 50.0), 4),
            latency_p95_ms=round(percentile(cat_lats, 95.0), 4),
        )

    return PolicyScore(
        policy_name=policy_name,
        description=description,
        total_cases=total,
        exact_match=round(em_total / total, 4) if total else 0.0,
        recall_at_k=round(rec_total / total, 4) if total else 0.0,
        abstention_precision=round(abs_prec, 4),
        abstention_recall=round(abs_rec, 4),
        abstention_f1=round(abs_f1, 4),
        false_positive_rate=round(fpr, 4),
        contradiction_detection_rate=round(contra_det, 4),
        temporal_order_accuracy=round(temp_acc, 4),
        stale_shadowing_errors=sum(1 for o in outcomes if o.stale_shadowing_error),
        cross_actor_leaks=sum(1 for o in outcomes if o.cross_actor_leak),
        latency_p50_ms=round(percentile(latencies, 50.0), 4),
        latency_p95_ms=round(percentile(latencies, 95.0), 4),
        latency_mean_ms=round(sum(latencies) / total, 4) if total else 0.0,
        avg_retrieval_tokens=round(total_tokens / total, 2) if total else 0.0,
        total_retrieval_tokens=total_tokens,
        effective_top_k=effective_top_k,
        api_calls=dict(api_calls),
        by_category=by_category,
        outcomes=outcomes,
    )


def format_score_table_markdown(scores: Sequence[PolicyScore]) -> str:
    """Render reproducible overall and per-category Markdown score tables."""
    lines: List[str] = []
    lines.append("### Overall Policy Score Table")
    lines.append("")
    lines.append(
        "| Policy | Exact Match | Recall@K | Abstention F1 | Unanswerable FPR | "
        "Contradiction Det. | Temporal Order | Stale Shadowing | Cross-Actor Leaks | "
        "Avg Tokens | p50 (ms) | p95 (ms) |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for s in scores:
        lines.append(
            f"| `{s.policy_name}` | {s.exact_match:.2%} | {s.recall_at_k:.2%} | "
            f"{s.abstention_f1:.2%} | {s.false_positive_rate:.2%} | "
            f"{s.contradiction_detection_rate:.2%} | {s.temporal_order_accuracy:.2%} | "
            f"{s.stale_shadowing_errors} | {s.cross_actor_leaks} | "
            f"{s.avg_retrieval_tokens:.1f} | {s.latency_p50_ms:.3f} | {s.latency_p95_ms:.3f} |"
        )

    lines.append("")
    lines.append("### Per-Category Exact Match Breakdown")
    lines.append("")
    lines.append(
        "| Policy | Knowledge Updates | Contradiction Handling | Temporal Ordering | "
        "Abstention | Cross-Session Retrieval |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for s in scores:
        cells = []
        for cat in ALL_CATEGORIES:
            cs = s.by_category.get(cat)
            if cs is None:
                cells.append("n/a")
            else:
                cells.append(f"{cs.exact_match:.2%} ({cs.exact_match_count}/{cs.total})")
        lines.append(f"| `{s.policy_name}` | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
