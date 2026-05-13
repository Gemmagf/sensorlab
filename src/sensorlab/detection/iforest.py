"""Isolation Forest wrapper conforming to :class:`BaseDetector`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
from sklearn.ensemble import IsolationForest

from sensorlab.detection.base import BaseDetector


@dataclass
class IForestDetector(BaseDetector):
    """Tree-based unsupervised detector. Fast, interpretable feature importances."""

    n_estimators: int = 200
    max_samples: str | int = "auto"
    contamination: float = 0.01
    random_state: int = 0
    name: str = "IForest"

    def __post_init__(self) -> None:
        self._model: IsolationForest | None = None

    def fit(self, X_normal: np.ndarray) -> Self:
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        ).fit(np.asarray(X_normal))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("IForestDetector must be fit before scoring.")
        # sklearn returns higher = more normal; invert so higher = more anomalous
        return -self._model.score_samples(np.asarray(X))
