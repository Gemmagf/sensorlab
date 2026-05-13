"""Scaling, sliding-window construction and run-level dataset splits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sensorlab.data.loader import TEPDataset


@dataclass
class Standardizer:
    """Mean/variance scaler fit on training data only (avoid leakage)."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, eps: float = 1e-8) -> "Standardizer":
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd = np.where(sd < eps, 1.0, sd)
        return cls(mean=mu.astype(np.float32), std=sd.astype(np.float32))

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        return Z * self.std + self.mean


def train_val_test_split_by_run(
    dataset: TEPDataset,
    frac_train: float = 0.6,
    frac_val: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified-by-fault run-level split.

    Returns three boolean masks over samples (train, val, test). Splitting at
    the run level guarantees no temporal leakage across the boundary.
    """
    rng = np.random.default_rng(seed)
    train_runs: list[int] = []
    val_runs: list[int] = []
    test_runs: list[int] = []

    for fid in np.unique(dataset.run_fault_id):
        runs = np.where(dataset.run_fault_id == fid)[0]
        runs = rng.permutation(runs)
        n = len(runs)
        n_tr = max(int(n * frac_train), 1 if n > 1 else 0)
        n_va = max(int(n * frac_val), 1 if n - n_tr > 1 else 0)
        train_runs.extend(runs[:n_tr].tolist())
        val_runs.extend(runs[n_tr : n_tr + n_va].tolist())
        test_runs.extend(runs[n_tr + n_va :].tolist())

    train_set, val_set, test_set = set(train_runs), set(val_runs), set(test_runs)
    train_mask = np.array([r in train_set for r in dataset.run_id])
    val_mask = np.array([r in val_set for r in dataset.run_id])
    test_mask = np.array([r in test_set for r in dataset.run_id])
    return train_mask, val_mask, test_mask


def sliding_windows(
    X: np.ndarray,
    run_id: np.ndarray,
    window: int = 20,
    stride: int = 1,
    labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Construct overlapping windows that never cross a ``run_id`` boundary.

    Returns
    -------
    windows : (n_win, window, n_sensors) float32
    window_labels : (n_win,) or None — label of the **last** sample in the window
    end_idx : (n_win,) — index into the original arrays of the last sample
    """
    parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    end_idx_parts: list[np.ndarray] = []

    for r in np.unique(run_id):
        mask = run_id == r
        idx = np.where(mask)[0]
        Xr = X[idx]
        n = Xr.shape[0]
        if n < window:
            continue
        starts = np.arange(0, n - window + 1, stride)
        ends = starts + window  # exclusive
        # Build with stride_tricks for speed
        from numpy.lib.stride_tricks import sliding_window_view

        view = sliding_window_view(Xr, window_shape=window, axis=0)[::stride]
        # view shape: (n_win, n_sensors, window) — transpose to (n_win, window, n_sensors)
        view = view.transpose(0, 2, 1)
        parts.append(view.astype(np.float32, copy=False))
        end_idx_parts.append(idx[ends - 1])
        if labels is not None:
            label_parts.append(labels[idx[ends - 1]])

    if not parts:
        empty = np.empty((0, window, X.shape[1]), dtype=np.float32)
        return empty, (None if labels is None else np.empty(0, dtype=labels.dtype)), np.empty(0, dtype=int)

    windows = np.concatenate(parts, axis=0)
    end_idx = np.concatenate(end_idx_parts, axis=0)
    window_labels = np.concatenate(label_parts, axis=0) if label_parts else None
    return windows, window_labels, end_idx
