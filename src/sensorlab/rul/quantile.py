"""Quantile-regression RUL with conformally calibrated lower/upper intervals.

Three histogram gradient-boosted regressors at q=0.1, 0.5, 0.9 give a point
estimate (median) and a nominal 80 % prediction interval. Vanilla quantile
GBMs are systematically over-confident on held-out data, so the interval is
then widened by a **split-conformal margin** (Conformalized Quantile
Regression, Romano et al. 2019) fitted on a calibration set the models never
saw. After :meth:`QuantileRUL.calibrate` the interval carries a
finite-sample coverage guarantee of ``1 - alpha`` under exchangeability —
which is what an operator needs to trust a "you have 90 to 240 minutes".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


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

    Note that on the synthetic benchmark (fixed run length, fixed onset) this
    target is a deterministic function of time since onset — it measures how
    well the *state* of the process encodes elapsed fault time, i.e. an
    urgency ranking, not a calibrated time-to-failure.
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
    """Three quantile GBMs (low, median, high) plus an optional conformal margin."""

    quantiles: tuple[float, float, float] = (0.1, 0.5, 0.9)
    n_estimators: int = 200
    max_depth: int = 4
    learning_rate: float = 0.05
    random_state: int = 0
    name: str = "QuantileRUL"
    models_: dict[float, HistGradientBoostingRegressor] = field(default_factory=dict)
    conformal_margin_: float | None = None
    n_calibration_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        for q in self.quantiles:
            m = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=q,
                max_iter=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
            )
            m.fit(X, y)
            self.models_[q] = m
        self.conformal_margin_ = None
        self.n_calibration_ = 0
        return self

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> Self:
        """Split-conformal (CQR) calibration of the interval on held-out data.

        The nominal level is taken from the outer quantiles, e.g. (0.1, 0.9)
        gives ``alpha = 0.2``. With fewer than ~20 calibration points the
        margin is left unset (interval stays raw) and a warning is logged.
        """
        import logging

        n = int(np.asarray(y_cal).shape[0])
        if n < 20:
            logging.getLogger(__name__).warning(
                "conformal calibration skipped: only %d calibration samples", n
            )
            self.conformal_margin_ = None
            self.n_calibration_ = n
            return self
        lo, _, hi = self._raw_interval(X_cal)
        alpha = 1.0 - (self.quantiles[2] - self.quantiles[0])
        conformity = np.maximum(lo - y_cal, y_cal - hi)
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        self.conformal_margin_ = float(np.quantile(conformity, level))
        self.n_calibration_ = n
        return self

    def predict(self, X: np.ndarray) -> dict[float, np.ndarray]:
        if not self.models_:
            raise RuntimeError("QuantileRUL must be fit before predicting.")
        return {q: m.predict(X) for q, m in self.models_.items()}

    def predict_median(self, X: np.ndarray) -> np.ndarray:
        return self.predict(X)[self.quantiles[1]]

    def _raw_interval(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        out = self.predict(X)
        return out[self.quantiles[0]], out[self.quantiles[1]], out[self.quantiles[2]]

    def predict_interval(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(low, median, high); low/high include the conformal margin once calibrated."""
        lo, med, hi = self._raw_interval(X)
        if self.conformal_margin_ is not None:
            lo = lo - self.conformal_margin_
            hi = hi + self.conformal_margin_
        return lo, med, hi

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
