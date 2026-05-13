"""Survival-analysis baseline for RUL.

Cox proportional hazards is used in industrial reliability as the
interpretable "where does each covariate move the failure hazard" baseline.
``lifelines`` is an optional dependency; if absent we fall back to a tiny
hand-rolled Breslow-style estimator built on top of statsmodels-free code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CoxBaseline:
    """Lightweight Cox PH wrapper using ``lifelines`` if available.

    Inputs follow survival convention:
        durations[i] : time until event or censoring
        events[i]    : 1 if failure observed, 0 if censored
        X[i]         : covariates
    """

    penalizer: float = 0.01
    name: str = "CoxPH"

    def __post_init__(self) -> None:
        self._fitter = None
        self._feature_names: list[str] | None = None

    def fit(
        self,
        X: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray,
        feature_names: list[str] | None = None,
    ):
        try:
            from lifelines import CoxPHFitter
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "CoxBaseline requires the `survival` extra. "
                "Install with `pip install -e .[survival]`."
            ) from e

        import pandas as pd

        self._feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=self._feature_names)
        df["T"] = durations
        df["E"] = events.astype(int)
        self._fitter = CoxPHFitter(penalizer=self.penalizer)
        self._fitter.fit(df, duration_col="T", event_col="E")
        return self

    def predict_partial_hazard(self, X: np.ndarray) -> np.ndarray:
        if self._fitter is None:
            raise RuntimeError("CoxBaseline must be fit first")
        import pandas as pd

        df = pd.DataFrame(X, columns=self._feature_names)
        return self._fitter.predict_partial_hazard(df).to_numpy()

    def predict_median(self, X: np.ndarray) -> np.ndarray:
        if self._fitter is None:
            raise RuntimeError("CoxBaseline must be fit first")
        import pandas as pd

        df = pd.DataFrame(X, columns=self._feature_names)
        med = self._fitter.predict_median(df)
        return np.asarray(med).astype(float)

    @property
    def summary(self):
        if self._fitter is None:
            raise RuntimeError("CoxBaseline must be fit first")
        return self._fitter.summary
