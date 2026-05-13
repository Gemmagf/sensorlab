"""Common detector interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import numpy as np


class BaseDetector(ABC):
    """Abstract base — all detectors return *higher = more anomalous*.

    Detectors are fit on **normal-only** training data so the threshold
    interpretation (false-alarm rate on nominal operation) is meaningful.
    """

    name: str = "base"

    @abstractmethod
    def fit(self, X_normal: np.ndarray) -> Self:
        """Fit on a 2D array of normal samples (n_samples, n_features)."""

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Return per-sample anomaly scores (higher = more anomalous)."""

    def fit_score(self, X_normal: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        return self.fit(X_normal).score(X_test)

    def threshold(self, scores_normal: np.ndarray, far: float = 0.01) -> float:
        """Empirical threshold giving the requested false-alarm rate on normal data."""
        q = float(np.quantile(scores_normal, 1.0 - far))
        return q
