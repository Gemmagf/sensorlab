"""Consistent, leakage-free evaluation of detectors and downstream heads.

Rules enforced here (and nowhere else, so they cannot drift apart):

* thresholds are calibrated on **validation-normal** samples only;
* every metric — AUROC, TPR, detection delay, fraction detected, cost — is
  reported on the **held-out evaluation runs** only;
* per-run metrics operate on whole runs, so masks must never split a run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from sensorlab.data.loader import TEPDataset
from sensorlab.decision.cost import CostModel, DecisionResult, optimal_threshold
from sensorlab.detection.metrics import (
    auroc,
    detection_delay,
    false_alarm_rate,
    threshold_at_far,
    true_positive_rate,
)


@dataclass
class DetectorReport:
    """Metrics of one detector on the evaluation runs."""

    name: str
    threshold: float
    far_target: float
    auroc: float
    tpr_at_far: float
    far_observed: float
    fraction_detected: float
    median_delay_min: float
    mean_delay_min: float
    p90_delay_min: float
    n_eval_runs: int
    n_eval_faulty_runs: int

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def _check_whole_runs(mask: np.ndarray, run_id: np.ndarray) -> None:
    for rid in np.unique(run_id[mask]):
        in_run = run_id == rid
        if not mask[in_run].all():
            raise ValueError(
                f"evaluation mask splits run {int(rid)} — per-run metrics need whole runs"
            )


def evaluate_detector(
    name: str,
    scores: np.ndarray,
    ds: TEPDataset,
    calib_mask: np.ndarray,
    eval_mask: np.ndarray,
    far_target: float = 0.01,
    consecutive: int = 3,
) -> DetectorReport:
    """Score a detector with a threshold calibrated on ``calib_mask`` (validation-normal)."""
    if scores.shape[0] != ds.n_samples:
        raise ValueError("scores must be per-sample (lift window scores first)")
    _check_whole_runs(eval_mask, ds.run_id)
    thr = threshold_at_far(scores[calib_mask], far=far_target)

    s_eval = scores[eval_mask]
    y_eval = ds.is_anomaly[eval_mask]
    delay = detection_delay(
        s_eval,
        ds.run_id[eval_mask],
        ds.run_onsets,
        ds.run_fault_id,
        thr,
        samples_to_minutes=ds.sample_minutes,
        consecutive=consecutive,
    )
    eval_runs = np.unique(ds.run_id[eval_mask])
    return DetectorReport(
        name=name,
        threshold=float(thr),
        far_target=float(far_target),
        auroc=auroc(s_eval, y_eval),
        tpr_at_far=true_positive_rate(s_eval, y_eval, thr),
        far_observed=false_alarm_rate(s_eval, y_eval, thr),
        fraction_detected=float(delay["fraction_detected"]),
        median_delay_min=float(delay["median_min"]),
        mean_delay_min=float(delay["mean_min"]),
        p90_delay_min=float(delay["p90_min"]),
        n_eval_runs=int(eval_runs.size),
        n_eval_faulty_runs=int(delay["n_faulty_runs"]),
    )


def evaluate_decision(
    scores: np.ndarray,
    ds: TEPDataset,
    eval_mask: np.ndarray,
    cost: CostModel,
    n_grid: int = 80,
    consecutive: int = 3,
) -> DecisionResult:
    """Cost-optimal threshold **and** its cost, both on the evaluation runs."""
    _check_whole_runs(eval_mask, ds.run_id)
    return optimal_threshold(
        scores[eval_mask],
        ds.run_id[eval_mask],
        ds.run_onsets,
        ds.run_fault_id,
        cost=cost,
        n_grid=n_grid,
        samples_to_minutes=ds.sample_minutes,
        consecutive=consecutive,
    )


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import balanced_accuracy_score, f1_score

    return {
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "n_samples": int(y_true.shape[0]),
    }


def aggregate_seeds(per_seed: list[dict]) -> dict:
    """Mean ± std of every numeric leaf across per-seed result dicts.

    Non-numeric leaves (names, nested lists) are taken from the first seed.
    """
    if not per_seed:
        return {}

    def _agg(values: list):
        first = values[0]
        if isinstance(first, dict):
            return {k: _agg([v[k] for v in values if k in v]) for k in first}
        if isinstance(first, bool) or not isinstance(first, int | float):
            return first
        arr = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(np.nanmean(arr)),
            "std": float(np.nanstd(arr)) if arr.size > 1 else 0.0,
            "n": int(arr.size),
        }

    return _agg(per_seed)
