"""Hotelling's T² and the SPE/Q residual statistic on a PCA model.

Reference: Jackson & Mudholkar (1979), MacGregor & Kourti (1995). The combined
T² + Q monitor is the de-facto baseline in chemical-process anomaly detection
and is what every classical paper compares against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
from sklearn.decomposition import PCA

from sensorlab.detection.base import BaseDetector


@dataclass
class PCAMonitor(BaseDetector):
    """T² + Q residual monitor on a PCA decomposition of the normal sensor space.

    Parameters
    ----------
    var_explained : keep enough principal components to explain this fraction
        of variance (typical: 0.85-0.95). Falls back to ``n_components`` if set.
    n_components : optional fixed component count (overrides ``var_explained``).
    combined : if True, ``score`` returns a normalised sum of T² and Q. If
        False, returns T² alone (Q is still available via ``score_components``).
    """

    var_explained: float = 0.90
    n_components: int | None = None
    combined: bool = True
    name: str = "PCA-T2Q"

    def __post_init__(self) -> None:
        self._pca: PCA | None = None
        self._mean: np.ndarray | None = None
        self._eigvals: np.ndarray | None = None
        self._t2_scale: float = 1.0
        self._q_scale: float = 1.0

    def fit(self, X_normal: np.ndarray) -> Self:
        X = np.asarray(X_normal, dtype=np.float64)
        self._mean = X.mean(axis=0)
        Xc = X - self._mean

        if self.n_components is not None:
            k = int(self.n_components)
        else:
            full = PCA().fit(Xc)
            cum = np.cumsum(full.explained_variance_ratio_)
            k = int(np.searchsorted(cum, self.var_explained) + 1)
            k = max(min(k, Xc.shape[1] - 1), 1)

        self._pca = PCA(n_components=k).fit(Xc)
        self._eigvals = np.maximum(self._pca.explained_variance_, 1e-12)

        # Normalisation constants so T² and Q live on roughly the same scale
        t2_train, q_train = self._raw_stats(Xc)
        self._t2_scale = float(np.median(t2_train) + 1e-9)
        self._q_scale = float(np.median(q_train) + 1e-9)
        return self

    def _raw_stats(self, Xc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self._pca is not None and self._eigvals is not None
        scores = Xc @ self._pca.components_.T  # (n, k)
        t2 = np.sum((scores**2) / self._eigvals, axis=1)
        reconstructed = scores @ self._pca.components_
        residual = Xc - reconstructed
        q = np.sum(residual**2, axis=1)
        return t2, q

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._pca is None or self._mean is None:
            raise RuntimeError("PCAMonitor must be fit before scoring.")
        Xc = np.asarray(X, dtype=np.float64) - self._mean
        t2, q = self._raw_stats(Xc)
        if self.combined:
            return (t2 / self._t2_scale) + (q / self._q_scale)
        return t2

    def score_components(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return raw (T², Q) stats — useful for diagnostics."""
        if self._pca is None or self._mean is None:
            raise RuntimeError("PCAMonitor must be fit before scoring.")
        Xc = np.asarray(X, dtype=np.float64) - self._mean
        return self._raw_stats(Xc)

    @property
    def n_components_(self) -> int:
        if self._pca is None:
            return 0
        return int(self._pca.n_components_)
