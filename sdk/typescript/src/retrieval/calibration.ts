// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

import { AdaptiveRetrievePolicy } from './kai-gate.js';
import {
  type AdaptiveRetrievePolicyConfig,
  CalibrationConstraintError,
  type KaiRawDecision,
} from './types.js';

/**
 * Computes exact order-statistic percentile (`0 <= p <= 100`) on a numeric array.
 */
export function computePercentile(values: readonly number[], percentile: number): number {
  if (values.length === 0) {
    return 0;
  }
  if (!Number.isFinite(percentile) || percentile < 0 || percentile > 100) {
    throw new CalibrationConstraintError(
      `Percentile must lie in [0, 100], got ${String(percentile)}.`
    );
  }
  const sorted = [...values].sort((a, b) => a - b);
  if (percentile === 0) {
    return sorted[0];
  }
  if (percentile === 100) {
    return sorted[sorted.length - 1];
  }
  const rank = (percentile / 100) * (sorted.length - 1);
  const lower = Math.floor(rank);
  const upper = Math.ceil(rank);
  const weight = rank - lower;
  return Number((sorted[lower] * (1 - weight) + sorted[upper] * weight).toFixed(4));
}

/**
 * Single labeled query sample used to calibrate threshold `theta`.
 */
export interface CalibrationSample {
  readonly sampleId: string;
  readonly kaiDecision: KaiRawDecision;
  /** True when live index evidence is required to answer accurately. */
  readonly requiresRetrieval: boolean;
  /** True when the prompt is underspecified or unauthorized and must clarify or abstain. */
  readonly shouldClarifyOrAbstain?: boolean;
  /** Whether direct parametric Zen synthesis produces a factually accurate answer. */
  readonly directAnswerAccurate: boolean;
  /** Whether RAG synthesis produces an accurate answer (or safe post-retrieval abstention). */
  readonly retrievedAnswerAccurate: boolean;
  /** Whether `hanzoai/index` returns authorized grounded chunks for this sample. */
  readonly retrievalGrounded?: boolean;
  readonly directLatencyMs?: number;
  readonly retrievedLatencyMs?: number;
  readonly clarifyLatencyMs?: number;
  readonly directCostUsd?: number;
  readonly retrievedCostUsd?: number;
  readonly clarifyCostUsd?: number;
}

export interface SweepThresholdsOptions {
  readonly thresholds?: readonly number[];
  readonly policyConfig?: Omit<AdaptiveRetrievePolicyConfig, 'theta'>;
  readonly defaultDirectLatencyMs?: number;
  readonly defaultRetrievedLatencyMs?: number;
  readonly defaultClarifyLatencyMs?: number;
  readonly defaultDirectCostUsd?: number;
  readonly defaultRetrievedCostUsd?: number;
  readonly defaultClarifyCostUsd?: number;
}

/**
 * Aggregated accuracy, grounding, latency, and cost metrics at one candidate `theta`.
 */
export interface ThresholdSweepPoint {
  readonly theta: number;
  readonly totalSamples: number;
  readonly retrievalCount: number;
  readonly directCount: number;
  readonly clarifyCount: number;
  readonly retrievalRate: number;
  readonly unnecessaryRetrievals: number;
  readonly unnecessaryRetrievalsAvoided: number;
  readonly unnecessaryRetrievalsAvoidedPct: number;
  readonly factualAccuracy: number;
  readonly groundingRate: number;
  readonly p50LatencyMs: number;
  readonly p95LatencyMs: number;
  readonly meanLatencyMs: number;
  readonly costPer1kQueriesUsd: number;
}

const DEFAULT_THRESHOLD_GRID: readonly number[] = Object.freeze([
  0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
  0.95,
]);

/**
 * Evaluates candidate `theta` values across a labeled calibration set using the exact
 * `AdaptiveRetrievePolicy` decision rules (including clarify bounds and risk floors).
 */
export function sweepThresholds(
  samples: readonly CalibrationSample[],
  options: SweepThresholdsOptions = {}
): readonly ThresholdSweepPoint[] {
  if (!Array.isArray(samples) || samples.length === 0) {
    throw new CalibrationConstraintError(
      'sweepThresholds requires at least one CalibrationSample.'
    );
  }

  const thresholds = options.thresholds ?? DEFAULT_THRESHOLD_GRID;
  if (thresholds.length === 0) {
    throw new CalibrationConstraintError('sweepThresholds thresholds array cannot be empty.');
  }

  for (const theta of thresholds) {
    if (typeof theta !== 'number' || !Number.isFinite(theta) || theta < 0 || theta > 1) {
      throw new CalibrationConstraintError(
        `Candidate theta must be a finite number in [0, 1], got ${String(theta)}.`
      );
    }
  }

  const basePolicy = new AdaptiveRetrievePolicy({
    ...options.policyConfig,
    theta: 0.5,
  });

  const defaultDirectLatencyMs = options.defaultDirectLatencyMs ?? 96;
  const defaultRetrievedLatencyMs = options.defaultRetrievedLatencyMs ?? 218;
  const defaultClarifyLatencyMs = options.defaultClarifyLatencyMs ?? 6;
  const defaultDirectCostUsd = options.defaultDirectCostUsd ?? 0.0005;
  const defaultRetrievedCostUsd = options.defaultRetrievedCostUsd ?? 0.00232;
  const defaultClarifyCostUsd = options.defaultClarifyCostUsd ?? 0.00002;

  const sortedThetas = [...thresholds].sort((a, b) => a - b);
  const results: ThresholdSweepPoint[] = [];

  for (const theta of sortedThetas) {
    let retrievalCount = 0;
    let directCount = 0;
    let clarifyCount = 0;
    let unnecessaryPool = 0;
    let unnecessaryRetrievals = 0;
    let requiredPool = 0;
    let groundedHits = 0;
    let accurateCount = 0;
    let totalCostUsd = 0;
    const latencies: number[] = [];

    for (const sample of samples) {
      const decision = basePolicy.evaluate(sample.kaiDecision, theta);
      const shouldAbstain = sample.shouldClarifyOrAbstain === true;

      if (!sample.requiresRetrieval) {
        unnecessaryPool += 1;
      } else {
        requiredPool += 1;
      }

      if (decision.action === 'clarify_or_abstain') {
        clarifyCount += 1;
        if (shouldAbstain) {
          accurateCount += 1;
        }
        latencies.push(sample.clarifyLatencyMs ?? defaultClarifyLatencyMs);
        totalCostUsd += sample.clarifyCostUsd ?? defaultClarifyCostUsd;
      } else if (decision.action === 'answer_direct') {
        directCount += 1;
        if (!shouldAbstain && sample.directAnswerAccurate) {
          accurateCount += 1;
        }
        latencies.push(sample.directLatencyMs ?? defaultDirectLatencyMs);
        totalCostUsd += sample.directCostUsd ?? defaultDirectCostUsd;
      } else {
        retrievalCount += 1;
        if (!sample.requiresRetrieval) {
          unnecessaryRetrievals += 1;
        }
        const isGrounded = sample.retrievalGrounded ?? sample.retrievedAnswerAccurate;
        if (sample.requiresRetrieval && isGrounded) {
          groundedHits += 1;
        }
        if (sample.retrievedAnswerAccurate) {
          accurateCount += 1;
        }
        latencies.push(sample.retrievedLatencyMs ?? defaultRetrievedLatencyMs);
        totalCostUsd += sample.retrievedCostUsd ?? defaultRetrievedCostUsd;
      }
    }

    const unnecessaryAvoided = unnecessaryPool - unnecessaryRetrievals;
    const unnecessaryRetrievalsAvoidedPct =
      unnecessaryPool > 0 ? Number(((unnecessaryAvoided / unnecessaryPool) * 100).toFixed(4)) : 0;
    const meanLatencyMs = Number(
      (latencies.reduce((acc, v) => acc + v, 0) / latencies.length).toFixed(4)
    );

    results.push(
      Object.freeze({
        theta: Number(theta.toFixed(6)),
        totalSamples: samples.length,
        retrievalCount,
        directCount,
        clarifyCount,
        retrievalRate: Number((retrievalCount / samples.length).toFixed(6)),
        unnecessaryRetrievals,
        unnecessaryRetrievalsAvoided: unnecessaryAvoided,
        unnecessaryRetrievalsAvoidedPct,
        factualAccuracy: Number((accurateCount / samples.length).toFixed(6)),
        groundingRate: requiredPool > 0 ? Number((groundedHits / requiredPool).toFixed(6)) : 1,
        p50LatencyMs: computePercentile(latencies, 50),
        p95LatencyMs: computePercentile(latencies, 95),
        meanLatencyMs,
        costPer1kQueriesUsd: Number(((totalCostUsd / samples.length) * 1000).toFixed(4)),
      })
    );
  }

  return Object.freeze(results);
}

export interface ThetaSelectionConstraints {
  /** Hard minimum factual accuracy floor in `[0, 1]`. */
  readonly minFactualAccuracy: number;
  /** Optional upper bound on USD cost per 1,000 queries. */
  readonly maxCostPer1kQueriesUsd?: number;
  /** Optional upper bound on p95 latency in milliseconds. */
  readonly maxP95LatencyMs?: number;
  /** Optional minimum grounding rate on retrieval-mandatory samples in `[0, 1]`. */
  readonly minGroundingRate?: number;
  /** Relative weight on normalized cost reduction among accuracy-compliant candidates. Default: `0.04`. */
  readonly costTradeoffWeight?: number;
  /** Throw `CalibrationConstraintError` if no candidate satisfies all bounds. Default: `false`. */
  readonly strict?: boolean;
  /** Optional sweep options used when passing `CalibrationSample[]` directly. */
  readonly sweepOptions?: SweepThresholdsOptions;
}

export interface OptimalThetaSelection {
  readonly theta: number;
  readonly selected: ThresholdSweepPoint;
  readonly accuracyFloorSatisfied: boolean;
  readonly budgetSatisfied: boolean;
  readonly feasibleCount: number;
  readonly evaluatedPoints: readonly ThresholdSweepPoint[];
}

function isSweepPointArray(
  input: readonly ThresholdSweepPoint[] | readonly CalibrationSample[]
): input is readonly ThresholdSweepPoint[] {
  return input.length > 0 && 'costPer1kQueriesUsd' in input[0] && 'retrievalRate' in input[0];
}

/**
 * Selects optimal threshold `theta` maximizing accuracy-cost utility under budget/latency bounds
 * without regressing factual accuracy below `minFactualAccuracy`.
 */
export function selectOptimalTheta(
  sweepOrSamples: readonly ThresholdSweepPoint[] | readonly CalibrationSample[],
  constraints: ThetaSelectionConstraints
): OptimalThetaSelection {
  if (!Array.isArray(sweepOrSamples) || sweepOrSamples.length === 0) {
    throw new CalibrationConstraintError('selectOptimalTheta input array cannot be empty.');
  }
  if (
    typeof constraints?.minFactualAccuracy !== 'number' ||
    !Number.isFinite(constraints.minFactualAccuracy) ||
    constraints.minFactualAccuracy < 0 ||
    constraints.minFactualAccuracy > 1
  ) {
    throw new CalibrationConstraintError(
      `minFactualAccuracy must be a finite number in [0, 1], got ${String(constraints?.minFactualAccuracy)}.`
    );
  }

  const points: readonly ThresholdSweepPoint[] = isSweepPointArray(sweepOrSamples)
    ? sweepOrSamples
    : sweepThresholds(sweepOrSamples, constraints.sweepOptions);

  const eps = 1e-9;
  const accuracyCompliant = points.filter(
    (p) => p.factualAccuracy + eps >= constraints.minFactualAccuracy
  );

  if (accuracyCompliant.length === 0) {
    if (constraints.strict) {
      throw new CalibrationConstraintError(
        `No candidate theta achieved minFactualAccuracy=${constraints.minFactualAccuracy}.`
      );
    }
    const fallback = [...points].sort((a, b) => {
      if (b.factualAccuracy !== a.factualAccuracy) {
        return b.factualAccuracy - a.factualAccuracy;
      }
      return a.costPer1kQueriesUsd - b.costPer1kQueriesUsd;
    })[0];

    return Object.freeze({
      theta: fallback.theta,
      selected: fallback,
      accuracyFloorSatisfied: false,
      budgetSatisfied: false,
      feasibleCount: 0,
      evaluatedPoints: points,
    });
  }

  const fullyFeasible = accuracyCompliant.filter((p) => {
    if (
      constraints.maxCostPer1kQueriesUsd !== undefined &&
      p.costPer1kQueriesUsd > constraints.maxCostPer1kQueriesUsd + eps
    ) {
      return false;
    }
    if (
      constraints.maxP95LatencyMs !== undefined &&
      p.p95LatencyMs > constraints.maxP95LatencyMs + eps
    ) {
      return false;
    }
    if (
      constraints.minGroundingRate !== undefined &&
      p.groundingRate + eps < constraints.minGroundingRate
    ) {
      return false;
    }
    return true;
  });

  if (fullyFeasible.length === 0) {
    if (constraints.strict) {
      throw new CalibrationConstraintError(
        'No accuracy-compliant theta satisfied cost, latency, or grounding constraints.'
      );
    }
    const lowestCostAccurate = [...accuracyCompliant].sort((a, b) => {
      if (a.costPer1kQueriesUsd !== b.costPer1kQueriesUsd) {
        return a.costPer1kQueriesUsd - b.costPer1kQueriesUsd;
      }
      if (b.factualAccuracy !== a.factualAccuracy) {
        return b.factualAccuracy - a.factualAccuracy;
      }
      return b.theta - a.theta;
    })[0];

    return Object.freeze({
      theta: lowestCostAccurate.theta,
      selected: lowestCostAccurate,
      accuracyFloorSatisfied: true,
      budgetSatisfied: false,
      feasibleCount: 0,
      evaluatedPoints: points,
    });
  }

  const maxCostObserved = Math.max(...points.map((p) => p.costPer1kQueriesUsd), 1e-6);
  const costWeight = constraints.costTradeoffWeight ?? 0.04;

  const ranked = [...fullyFeasible].sort((a, b) => {
    const utilA = a.factualAccuracy - costWeight * (a.costPer1kQueriesUsd / maxCostObserved);
    const utilB = b.factualAccuracy - costWeight * (b.costPer1kQueriesUsd / maxCostObserved);
    if (Math.abs(utilB - utilA) > 1e-9) {
      return utilB - utilA;
    }
    if (b.factualAccuracy !== a.factualAccuracy) {
      return b.factualAccuracy - a.factualAccuracy;
    }
    if (a.costPer1kQueriesUsd !== b.costPer1kQueriesUsd) {
      return a.costPer1kQueriesUsd - b.costPer1kQueriesUsd;
    }
    const marginDistA = Math.abs(a.theta - 0.5);
    const marginDistB = Math.abs(b.theta - 0.5);
    if (Math.abs(marginDistA - marginDistB) > 1e-9) {
      return marginDistA - marginDistB;
    }
    return b.theta - a.theta;
  });

  const best = ranked[0];
  return Object.freeze({
    theta: best.theta,
    selected: best,
    accuracyFloorSatisfied: true,
    budgetSatisfied: true,
    feasibleCount: fullyFeasible.length,
    evaluatedPoints: points,
  });
}
