"""Quantile-regression RUL with calibrated lower/upper intervals.

We train three gradient-boosted regressors at q=0.1, 0.5, 0.9 — giving a
point estimate (median) and an 80% prediction interval at every sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


def build_rul_targets(
    run_id: np.ndarray,
    is_anomaly: np.ndarray,
    run_onsets: np.ndarray,
    run_fault_id: np.ndarray,
    samples_to_minutes: float = 3.0,
    cap_minutes: float = 600.0,
) -> tuple[np.ndarray, np.ndarray]:
    """For every faulty sample, ``time_until_end_of_run`` (in minutes).

    Nominal samples and pre-onset samples are masked out (returned mask=False).
    The horizon is capped so that very long runs don't dominate the loss.
    """
    n = run_id.shape[0]
    rul = np.full(n, np.nan, dtype=np.float32)
    mask = np.zeros(n, dtype=bool)

    for rid in np.unique(run_id):
        rid_int = int(rid)
        if int(run_fault_id[rid_int]) == 0:
            continue
        idx = np.where(run_id == rid)[0]
        onset = int(run_onsets[rid_int])
        if onset < 0:
            continue
        post = idx[onset:]
        n_post = post.size
        # samples_remaining (inclusive of current → minimum 0 at last sample)
        rem = (n_post - 1 - np.arange(n_post)) * samples_to_minutes
        rem = np.minimum(rem, cap_minutes)
        rul[post] = rem.astype(np.float32)
        mask[post] = True
    return rul, mask


@dataclass
class QuantileRUL:
    """Three GBM regressors at quantiles (low, median, high)."""

    quantiles: tuple[float, float, float] = (0.1, 0.5, 0.9)
    n_estimators: int = 200
    max_depth: int = 4
    learning_rate: float = 0.05
    random_state: int = 0
    name: str = "QuantileRUL"
    models_: dict[float, GradientBoostingRegressor] = field(default_factory=dict)

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        for q in self.quantiles:
            m = GradientBoostingRegressor(
                loss="quantile",
                alpha=q,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
            )
            m.fit(X, y)
            self.models_[q] = m
        return self

    def predict(self, X: np.ndarray) -> dict[float, np.ndarray]:
        if not self.models_:
            raise RuntimeError("QuantileRUL must be fit before predicting.")
        return {q: m.predict(X) for q, m in self.models_.items()}

    def predict_median(self, X: np.ndarray) -> np.ndarray:
        return self.predict(X)[self.quantiles[1]]

    def predict_interval(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        out = self.predict(X)
        return out[self.quantiles[0]], out[self.quantiles[1]], out[self.quantiles[2]]

    @staticmethod
    def coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
        return float(((y_true >= lo) & (y_true <= hi)).mean())

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
        e = y_true - y_pred
        return float(np.mean(np.maximum(q * e, (q - 1.0) * e)))
