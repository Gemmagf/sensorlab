"""Post-deployment monitoring: is the data still the data the model was trained on?

An anomaly detector that fires because a sensor was recalibrated is a
false alarm the operator cannot distinguish from a fault. The
:class:`DriftMonitor` keeps a compact reference of the training-normal
distribution and reports per-sensor **Population Stability Index** (PSI) on
new batches — the standard, explainable check used in production model
governance.

Textbook rule of thumb (Siddiqi, 2006): PSI < 0.10 stable, 0.10 to 0.25
investigate, > 0.25 the model should not be trusted on that sensor. Process
data is strongly autocorrelated, so a batch of a few runs can sit well above
those cut-offs while being perfectly normal. When the reference is fitted
with run ids, the monitor therefore measures its own **noise floor** by
leave-one-run-out PSI and raises the thresholds per sensor to sit above it
(``watch = max(0.10, 1.5 x floor)``, ``alert = max(0.25, 3 x floor)``). A real
recalibration of one standard deviation scores an order of magnitude higher.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PSI_WATCH = 0.10
PSI_ALERT = 0.25
_EPS = 1e-4


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, edges: np.ndarray, eps: float = _EPS
) -> float:
    """PSI between two 1-D samples on a fixed set of bin ``edges``."""
    ref_hist, _ = np.histogram(reference, bins=edges)
    cur_hist, _ = np.histogram(current, bins=edges)
    return _psi_from_hist(ref_hist, cur_hist, eps)


def _psi_from_hist(ref_hist: np.ndarray, cur_hist: np.ndarray, eps: float = _EPS) -> float:
    ref_p = ref_hist / max(ref_hist.sum(), 1) + eps
    cur_p = cur_hist / max(cur_hist.sum(), 1) + eps
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


@dataclass
class DriftReport:
    psi: dict[str, float]
    n_current: int
    watch: list[str] = field(default_factory=list)
    alert: list[str] = field(default_factory=list)
    alert_thresholds: dict[str, float] = field(default_factory=dict)

    @property
    def status(self) -> str:
        if self.alert:
            return "alert"
        if self.watch:
            return "watch"
        return "stable"

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "n_current": self.n_current,
            "alert": self.alert,
            "watch": self.watch,
            "psi": self.psi,
            "alert_thresholds": self.alert_thresholds,
        }


@dataclass
class DriftMonitor:
    """Per-sensor PSI against a reference (training-normal) sample.

    The reference is stored as bin edges + counts only, so a saved pipeline
    carries a few KB of monitoring state rather than the training data.
    """

    n_bins: int = 10
    watch_threshold: float = PSI_WATCH
    alert_threshold: float = PSI_ALERT
    watch_floor_factor: float = 1.5
    alert_floor_factor: float = 3.0
    sensor_names: list[str] = field(default_factory=list)
    edges_: dict[str, np.ndarray] = field(default_factory=dict)
    ref_hist_: dict[str, np.ndarray] = field(default_factory=dict)
    noise_floor_: dict[str, float] = field(default_factory=dict)
    watch_thresholds_: dict[str, float] = field(default_factory=dict)
    alert_thresholds_: dict[str, float] = field(default_factory=dict)

    def fit(
        self,
        X_reference: np.ndarray,
        sensor_names: list[str],
        run_id: np.ndarray | None = None,
    ) -> DriftMonitor:
        """Store the reference histograms; with ``run_id`` also learn the noise floor."""
        X = np.asarray(X_reference, dtype=np.float64)
        if X.shape[1] != len(sensor_names):
            raise ValueError("sensor_names must match the number of columns")
        self.sensor_names = list(sensor_names)
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        for j, name in enumerate(sensor_names):
            col = X[:, j]
            inner = np.unique(np.quantile(col, qs))
            edges = np.concatenate([[-np.inf], inner, [np.inf]])
            self.edges_[name] = edges
            self.ref_hist_[name], _ = np.histogram(col, bins=edges)

        self.noise_floor_ = dict.fromkeys(sensor_names, 0.0)
        if run_id is not None:
            run_id = np.asarray(run_id)
            runs = np.unique(run_id)
            if runs.size >= 2:
                for j, name in enumerate(sensor_names):
                    edges = self.edges_[name]
                    total = self.ref_hist_[name]
                    worst = 0.0
                    for r in runs:
                        held, _ = np.histogram(X[run_id == r, j], bins=edges)
                        worst = max(worst, _psi_from_hist(total - held, held))
                    self.noise_floor_[name] = worst
        self.watch_thresholds_ = {
            n: max(self.watch_threshold, self.watch_floor_factor * f)
            for n, f in self.noise_floor_.items()
        }
        self.alert_thresholds_ = {
            n: max(self.alert_threshold, self.alert_floor_factor * f)
            for n, f in self.noise_floor_.items()
        }
        return self

    def check(self, X_current: np.ndarray) -> DriftReport:
        if not self.edges_:
            raise RuntimeError("DriftMonitor must be fit first")
        X = np.asarray(X_current, dtype=np.float64)
        if X.shape[1] != len(self.sensor_names):
            raise ValueError(
                f"expected {len(self.sensor_names)} sensors, got {X.shape[1]} — "
                "column order/count must match the training layout"
            )
        psi: dict[str, float] = {}
        for j, name in enumerate(self.sensor_names):
            cur_hist, _ = np.histogram(X[:, j], bins=self.edges_[name])
            psi[name] = _psi_from_hist(self.ref_hist_[name], cur_hist)
        alert = [n for n, v in psi.items() if v >= self.alert_thresholds_[n]]
        watch = [
            n for n, v in psi.items() if self.watch_thresholds_[n] <= v < self.alert_thresholds_[n]
        ]
        return DriftReport(
            psi=psi,
            n_current=int(X.shape[0]),
            watch=watch,
            alert=alert,
            alert_thresholds=dict(self.alert_thresholds_),
        )
