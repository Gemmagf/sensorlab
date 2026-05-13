"""Threshold optimisation under an explicit cost model.

The detector outputs a scalar score per sample; the threshold decides whether
to alarm. Each operational decision has a cost:

* **False alarm** (FA): the operator investigates a non-fault — process
  downtime, lost batch yield.
* **Missed fault** (MF): the fault propagates undetected — off-spec product,
  rework, possibly safety review.
* **Late detection** (LD): per-minute penalty for each minute between fault
  onset and first true alarm.

The optimal threshold minimises expected total cost over a representative
operating period.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CostModel:
    """Operational cost parameters (in CHF or any consistent currency)."""

    false_alarm_cost: float = 100.0
    missed_fault_cost: float = 5000.0
    delay_cost_per_min: float = 50.0
    name: str = "default"


@dataclass
class DecisionResult:
    threshold: float
    expected_cost: float
    false_alarms: int
    missed_faults: int
    mean_delay_min: float
    n_normal_samples: int
    n_faulty_runs: int


def evaluate_threshold(
    scores: np.ndarray,
    run_id: np.ndarray,
    run_onsets: np.ndarray,
    run_fault_id: np.ndarray,
    threshold: float,
    cost: CostModel,
    samples_to_minutes: float = 3.0,
    consecutive: int = 3,
) -> DecisionResult:
    """Compute expected total cost for a given threshold."""
    n_normal_samples = 0
    n_false_alarms = 0
    n_faulty_runs = 0
    n_missed = 0
    delays: list[float] = []

    for rid in np.unique(run_id):
        rid_int = int(rid)
        idx = np.where(run_id == rid)[0]
        fault = int(run_fault_id[rid_int])
        s = scores[idx]
        if fault == 0:
            n_normal_samples += len(s)
            n_false_alarms += int((s >= threshold).sum())
            continue
        n_faulty_runs += 1
        onset = int(run_onsets[rid_int])
        if onset < 0 or onset >= len(s):
            continue
        # FA on pre-onset segment
        n_normal_samples += onset
        n_false_alarms += int((s[:onset] >= threshold).sum())
        # detection on post-onset segment
        post_above = s[onset:] >= threshold
        if consecutive <= 1:
            hits = np.where(post_above)[0]
        else:
            kernel = np.ones(consecutive, dtype=int)
            conv = np.convolve(post_above.astype(int), kernel, mode="valid")
            hits = np.where(conv == consecutive)[0]
        if hits.size:
            delays.append(float(hits[0]) * samples_to_minutes)
        else:
            n_missed += 1

    mean_delay = float(np.mean(delays)) if delays else 0.0
    total = (
        cost.false_alarm_cost * n_false_alarms
        + cost.missed_fault_cost * n_missed
        + cost.delay_cost_per_min * sum(delays)
    )
    return DecisionResult(
        threshold=float(threshold),
        expected_cost=float(total),
        false_alarms=int(n_false_alarms),
        missed_faults=int(n_missed),
        mean_delay_min=mean_delay,
        n_normal_samples=int(n_normal_samples),
        n_faulty_runs=int(n_faulty_runs),
    )


def cost_curve(
    scores: np.ndarray,
    run_id: np.ndarray,
    run_onsets: np.ndarray,
    run_fault_id: np.ndarray,
    cost: CostModel,
    n_grid: int = 60,
    samples_to_minutes: float = 3.0,
    consecutive: int = 3,
) -> tuple[np.ndarray, list[DecisionResult]]:
    """Sweep thresholds and evaluate each — for plotting the cost-vs-threshold curve."""
    lo, hi = float(np.quantile(scores, 0.01)), float(np.quantile(scores, 0.999))
    grid = np.linspace(lo, hi, n_grid)
    results = [
        evaluate_threshold(
            scores,
            run_id,
            run_onsets,
            run_fault_id,
            t,
            cost,
            samples_to_minutes=samples_to_minutes,
            consecutive=consecutive,
        )
        for t in grid
    ]
    return grid, results


def optimal_threshold(
    scores: np.ndarray,
    run_id: np.ndarray,
    run_onsets: np.ndarray,
    run_fault_id: np.ndarray,
    cost: CostModel,
    n_grid: int = 60,
    samples_to_minutes: float = 3.0,
    consecutive: int = 3,
) -> DecisionResult:
    """Threshold minimising expected total cost on the supplied data."""
    grid, results = cost_curve(
        scores,
        run_id,
        run_onsets,
        run_fault_id,
        cost,
        n_grid=n_grid,
        samples_to_minutes=samples_to_minutes,
        consecutive=consecutive,
    )
    costs = np.array([r.expected_cost for r in results])
    best = int(np.argmin(costs))
    return results[best]
