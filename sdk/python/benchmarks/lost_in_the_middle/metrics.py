"""Evaluation metrics, statistical bootstrap intervals, and score aggregation."""

from __future__ import annotations

import math
import random
import re
import string
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """Standard QA normalization: lowercase, strip punctuation/articles, collapse spaces."""
    lowered = text.lower()
    no_punct = lowered.translate(_PUNCT_TABLE)
    no_articles = _ARTICLES_RE.sub(" ", no_punct)
    return _WHITESPACE_RE.sub(" ", no_articles).strip()


def exact_match(prediction: str, gold: str) -> float:
    """Return 1.0 if normalized prediction exactly matches normalized gold, else 0.0."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def substring_match(prediction: str, gold: str) -> float:
    """Return 1.0 if normalized gold appears in normalized prediction (Liu et al. TACL 2024 metric)."""
    norm_gold = normalize_answer(gold)
    if not norm_gold:
        return 0.0
    return 1.0 if norm_gold in normalize_answer(prediction) else 0.0


def token_f1(prediction: str, gold: str) -> float:
    """Token-level F1 score between normalized prediction and gold answer."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2.0 * precision * recall) / (precision + recall)


def percentile(values: Sequence[float], pct: float) -> float:
    """Compute linear-interpolated percentile in [0, 100]."""
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


@dataclass(frozen=True)
class BootstrapCI:
    """Paired bootstrap confidence interval for score_a - score_b."""

    mean_diff: float
    ci_low: float
    ci_high: float
    n_pairs: int
    n_resamples: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def paired_bootstrap_ci(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> BootstrapCI:
    """Compute paired bootstrap 95% confidence interval for mean(scores_a - scores_b)."""
    if len(scores_a) != len(scores_b):
        raise ValueError("scores_a and scores_b must have identical length")
    n = len(scores_a)
    if n == 0:
        return BootstrapCI(0.0, 0.0, 0.0, 0, n_resamples)

    diffs = [float(a) - float(b) for a, b in zip(scores_a, scores_b)]
    mean_diff = sum(diffs) / n
    rng = random.Random(seed)
    boot_means: List[float] = []
    for _ in range(n_resamples):
        sample_sum = 0.0
        for _ in range(n):
            sample_sum += diffs[rng.randrange(n)]
        boot_means.append(sample_sum / n)
    boot_means.sort()
    lo_pct = (alpha / 2.0) * 100.0
    hi_pct = (1.0 - alpha / 2.0) * 100.0
    return BootstrapCI(
        mean_diff=round(mean_diff, 6),
        ci_low=round(percentile(boot_means, lo_pct), 6),
        ci_high=round(percentile(boot_means, hi_pct), 6),
        n_pairs=n,
        n_resamples=n_resamples,
    )


@dataclass(frozen=True)
class PolicyAggregate:
    """Aggregated benchmark metrics for a slice of CaseEvaluation records."""

    policy: str
    count: int
    answer_correctness: float
    evidence_retention_rate: float
    truncation_loss_rate: float
    mean_tokens: float
    p95_tokens: float
    token_efficiency_per_1k: float
    boundary_integrity_rate: float
    p50_latency_ms: float
    p95_latency_ms: float
    primacy_correctness: Optional[float] = None
    middle_correctness: Optional[float] = None
    recency_correctness: Optional[float] = None
    edge_minus_middle_gap: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def aggregate_evaluations(
    policy: str,
    records: Sequence[Dict[str, Any]],
) -> PolicyAggregate:
    """Aggregate per-case evaluation dicts into a PolicyAggregate summary."""
    n = len(records)
    if n == 0:
        return PolicyAggregate(
            policy=policy,
            count=0,
            answer_correctness=0.0,
            evidence_retention_rate=0.0,
            truncation_loss_rate=0.0,
            mean_tokens=0.0,
            p95_tokens=0.0,
            token_efficiency_per_1k=0.0,
            boundary_integrity_rate=0.0,
            p50_latency_ms=0.0,
            p95_latency_ms=0.0,
        )

    correctness = sum(float(r["correctness"]) for r in records) / n
    retention = sum(1.0 if r["evidence_retained"] else 0.0 for r in records) / n
    trunc_loss = sum(1.0 if r["truncation_lost_gold"] else 0.0 for r in records) / n
    tokens = [float(r["estimated_tokens"]) for r in records]
    mean_tok = sum(tokens) / n
    p95_tok = percentile(tokens, 95.0)
    efficiency = (correctness / (mean_tok / 1000.0)) if mean_tok > 0 else 0.0
    integrity = sum(1.0 if r["boundary_integrity_ok"] else 0.0 for r in records) / n
    latencies = [float(r["packing_latency_ms"]) for r in records]
    p50_lat = percentile(latencies, 50.0)
    p95_lat = percentile(latencies, 95.0)

    # Primacy (pos == 1), Middle (pos == N // 2 or (N+1)//2), Recency (pos == N)
    primacy_vals = [
        float(r["correctness"]) for r in records if int(r["gold_position"]) == 1
    ]
    middle_vals = [
        float(r["correctness"])
        for r in records
        if int(r["gold_position"]) == max(1, int(r["num_docs"]) // 2)
    ]
    recency_vals = [
        float(r["correctness"])
        for r in records
        if int(r["gold_position"]) == int(r["num_docs"])
    ]
    prim = (sum(primacy_vals) / len(primacy_vals)) if primacy_vals else None
    mid = (sum(middle_vals) / len(middle_vals)) if middle_vals else None
    rec = (sum(recency_vals) / len(recency_vals)) if recency_vals else None
    gap: Optional[float] = None
    if prim is not None and mid is not None and rec is not None:
        gap = round(((prim + rec) / 2.0) - mid, 6)

    return PolicyAggregate(
        policy=policy,
        count=n,
        answer_correctness=round(correctness, 6),
        evidence_retention_rate=round(retention, 6),
        truncation_loss_rate=round(trunc_loss, 6),
        mean_tokens=round(mean_tok, 2),
        p95_tokens=round(p95_tok, 2),
        token_efficiency_per_1k=round(efficiency, 6),
        boundary_integrity_rate=round(integrity, 6),
        p50_latency_ms=round(p50_lat, 4),
        p95_latency_ms=round(p95_lat, 4),
        primacy_correctness=round(prim, 6) if prim is not None else None,
        middle_correctness=round(mid, 6) if mid is not None else None,
        recency_correctness=round(rec, 6) if rec is not None else None,
        edge_minus_middle_gap=gap,
    )
