"""Detection metrics: detection delay, false-alarm rate, TPR, AUROC.

All metrics expect *per-sample* (or per-window-end) anomaly scores and a
boolean ``is_anomaly`` mask. ``run_id`` and ``onset_index`` are required for
per-run detection delay so that pre-fault samples don't pollute the average.
"""

from __future__ import annotations

import numpy as np


def threshold_at_far(scores_normal: np.ndarray, far: float = 0.01) -> float:
    """Empirical threshold on normal-only scores for the requested FAR."""
    if not 0.0 < far < 1.0:
        raise ValueError("far must be in (0, 1)")
    return float(np.quantile(scores_normal, 1.0 - far))


def false_alarm_rate(scores: np.ndarray, is_anomaly: np.ndarray, threshold: float) -> float:
    """Fraction of *normal* samples flagged."""
    mask = ~np.asarray(is_anomaly, dtype=bool)
    if not mask.any():
        return float("nan")
    return float((scores[mask] >= threshold).mean())


def true_positive_rate(scores: np.ndarray, is_anomaly: np.ndarray, threshold: float) -> float:
    """Fraction of *anomalous* samples flagged."""
    mask = np.asarray(is_anomaly, dtype=bool)
    if not mask.any():
        return float("nan")
    return float((scores[mask] >= threshold).mean())


def auroc(scores: np.ndarray, is_anomaly: np.ndarray) -> float:
    """Area under the ROC curve, anomaly = positive class.

    Uses the Mann-Whitney U formulation with average ranks for ties so the
    output matches :func:`sklearn.metrics.roc_auc_score` exactly.
    """
    y = np.asarray(is_anomaly, dtype=bool)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Average ranks (higher score → higher rank); SciPy is already a project dep
    from scipy.stats import rankdata

    ranks = rankdata(s, method="average")
    u = ranks[y].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def detection_delay(
    scores: np.ndarray,
    run_id: np.ndarray,
    run_onsets: np.ndarray,
    run_fault_id: np.ndarray,
    threshold: float,
    samples_to_minutes: float = 1.0,
    consecutive: int = 3,
) -> dict[str, float]:
    """Per-faulty-run detection delay.

    A run is considered "detected" the first time ``consecutive`` consecutive
    samples (after the onset) exceed the threshold. Returns mean / median /
    p90 detection delay in *minutes* and the fraction of faulty runs detected.
    """
    delays: list[float] = []
    detected = 0
    n_faulty = 0
    for rid in np.unique(run_id):
        rid_int = int(rid)
        fault = int(run_fault_id[rid_int])
        if fault == 0:
            continue
        n_faulty += 1
        idx = np.where(run_id == rid)[0]
        onset = int(run_onsets[rid_int])
        if onset < 0 or onset >= len(idx):
            continue
        sub = scores[idx][onset:]
        above = sub >= threshold
        # Detect first run of `consecutive` True values
        if consecutive <= 1:
            hits = np.where(above)[0]
        else:
            # convolution-based run detection
            kernel = np.ones(consecutive, dtype=int)
            conv = np.convolve(above.astype(int), kernel, mode="valid")
            hit_starts = np.where(conv == consecutive)[0]
            hits = hit_starts
        if hits.size > 0:
            delay_samples = int(hits[0])
            delays.append(delay_samples * samples_to_minutes)
            detected += 1
    if not delays:
        return {
            "mean_min": float("nan"),
            "median_min": float("nan"),
            "p90_min": float("nan"),
            "fraction_detected": 0.0,
            "n_faulty_runs": n_faulty,
        }
    arr = np.asarray(delays)
    return {
        "mean_min": float(arr.mean()),
        "median_min": float(np.median(arr)),
        "p90_min": float(np.quantile(arr, 0.9)),
        "fraction_detected": detected / max(n_faulty, 1),
        "n_faulty_runs": n_faulty,
    }
