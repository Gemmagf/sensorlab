"""Multi-class fault diagnosis with XGBoost on per-sensor summary features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
import xgboost as xgb


def window_features(windows: np.ndarray, sensor_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Per-sensor summary features over each window.

    For each sensor we extract: mean, std, last value, and linear slope.
    This keeps the dimensionality at ``4 * n_sensors`` (132 for TEP) and gives
    SHAP attributions that map back to physically meaningful quantities like
    *"the slope of reactor temperature"*.
    """
    if windows.ndim != 3:
        raise ValueError(f"expected (n, T, p) windows, got {windows.shape}")
    n, T, p = windows.shape

    means = windows.mean(axis=1)
    stds = windows.std(axis=1)
    lasts = windows[:, -1, :]
    t_idx = (np.arange(T) - (T - 1) / 2.0).astype(np.float32)
    denom = float((t_idx**2).sum()) + 1e-9
    slopes = (windows * t_idx[None, :, None]).sum(axis=1) / denom

    feats = np.concatenate([means, stds, lasts, slopes], axis=1).astype(np.float32)
    names = (
        [f"mean[{s}]" for s in sensor_names]
        + [f"std[{s}]" for s in sensor_names]
        + [f"last[{s}]" for s in sensor_names]
        + [f"slope[{s}]" for s in sensor_names]
    )
    return feats, names


@dataclass
class FaultClassifier:
    """XGBoost wrapper that predicts the active fault id from window features."""

    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_lambda: float = 1.0
    random_state: int = 0
    # n_jobs=1 by default — multi-threaded XGBoost segfaults on macOS arm64
    # when libomp is loaded simultaneously by PyTorch. Single-threaded is
    # fast enough for our scale and reliable across platforms.
    n_jobs: int = 1
    name: str = "XGB-fault"

    def __post_init__(self) -> None:
        self._model: xgb.XGBClassifier | None = None
        self.classes_: np.ndarray | None = None
        self.feature_names_: list[str] | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> Self:
        # XGBoost is picky about label dtype — coerce to int32 / float32 to
        # avoid segfaults seen with int16 inputs.
        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.ascontiguousarray(y, dtype=np.int32)
        if sample_weight is not None:
            sample_weight = np.ascontiguousarray(sample_weight, dtype=np.float32)
        self.classes_ = np.unique(y)
        self.feature_names_ = feature_names
        self._model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            objective="multi:softprob",
            tree_method="hist",
            verbosity=0,
        )
        self._model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.ascontiguousarray(X, dtype=np.float32)
        return self._model.predict(X)  # type: ignore[union-attr]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.ascontiguousarray(X, dtype=np.float32)
        return self._model.predict_proba(X)  # type: ignore[union-attr]

    @property
    def feature_importances_(self) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("classifier must be fit first")
        return self._model.feature_importances_

    @property
    def booster(self):
        if self._model is None:
            raise RuntimeError("classifier must be fit first")
        return self._model
