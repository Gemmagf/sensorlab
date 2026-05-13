"""Shared pytest fixtures.

Builds a deliberately tiny synthetic dataset once per session so the full
test suite runs in a few seconds. Each test that needs altered settings
should construct its own ``SyntheticTEPConfig``.
"""

from __future__ import annotations

# Must run before torch / xgboost are imported. PyTorch and XGBoost each ship
# their own libomp on macOS arm64 — loading both into the same process and
# allowing OpenMP threads to interleave causes XGBoost's DMatrix construction
# to segfault. Set the env var AND force xgboost to load first so its libomp
# wins. Both safeguards are needed; one alone is not enough.
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest
import torch  # noqa: F401
import xgboost  # noqa: F401  — must precede torch

from sensorlab.data import (
    Standardizer,
    SyntheticTEPConfig,
    load_dataset,
    sliding_windows,
    train_val_test_split_by_run,
)


@pytest.fixture(scope="session")
def tiny_cfg() -> SyntheticTEPConfig:
    """Minimal config — fast enough that every test can use it."""
    return SyntheticTEPConfig(
        n_normal_runs=4,
        n_runs_per_fault=2,
        fault_run_minutes=180,  # 60 samples per run @ 3 min
        sensor_noise=0.08,
        seed=42,
    )


@pytest.fixture(scope="session")
def tiny_dataset(tiny_cfg):
    return load_dataset("synthetic", cfg=tiny_cfg)


@pytest.fixture(scope="session")
def tiny_splits(tiny_dataset):
    train, val, test = train_val_test_split_by_run(tiny_dataset, seed=1)
    return {"train": train, "val": val, "test": test}


@pytest.fixture(scope="session")
def tiny_standardized(tiny_dataset, tiny_splits):
    normal_train = tiny_splits["train"] & (tiny_dataset.fault_id == 0)
    sc = Standardizer.fit(tiny_dataset.X[normal_train])
    Xz = sc.transform(tiny_dataset.X)
    return {"scaler": sc, "X": Xz, "normal_train": normal_train}


@pytest.fixture(scope="session")
def tiny_windows(tiny_dataset, tiny_standardized):
    windows, _, end_idx = sliding_windows(
        tiny_standardized["X"], tiny_dataset.run_id, window=12, stride=2
    )
    return {"windows": windows, "end_idx": end_idx}


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
