"""Shared pytest fixtures.

Builds a deliberately tiny synthetic dataset once per session so the full
test suite runs in a few seconds. Each test that needs altered settings
should construct its own ``SyntheticTEPConfig``.
"""

from __future__ import annotations

import numpy as np
import pytest

# Importing the package first applies the platform shims (macOS libomp
# ordering, thread caps) before torch / xgboost are loaded anywhere else.
import sensorlab  # noqa: F401
from sensorlab.data import (
    Standardizer,
    SyntheticTEPConfig,
    load_dataset,
    sliding_windows,
    train_val_test_split_by_run,
)
from sensorlab.pipeline import MonitoringPipeline, PipelineConfig


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


@pytest.fixture(scope="session")
def fast_pipeline_config() -> PipelineConfig:
    return PipelineConfig(
        window=12,
        stride=2,
        ae_epochs=2,
        ae_hidden=8,
        ae_latent=4,
        iforest_estimators=30,
        xgb_estimators=20,
        xgb_max_depth=3,
        rul_estimators=20,
        rul_max_depth=2,
        seed=0,
    )


@pytest.fixture(scope="session")
def fitted_pipeline(tiny_dataset, tiny_splits, fast_pipeline_config):
    return MonitoringPipeline(fast_pipeline_config).fit(
        tiny_dataset, tiny_splits["train"], tiny_splits["val"]
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
