"""Data acquisition, simulation and preprocessing."""

from sensorlab.data.loader import TEPDataset, load_dataset
from sensorlab.data.preprocess import (
    Standardizer,
    sliding_windows,
    train_val_test_split_by_run,
)
from sensorlab.data.synthetic import (
    FAULT_CATALOGUE,
    SyntheticTEPConfig,
    generate_synthetic_dataset,
)

__all__ = [
    "FAULT_CATALOGUE",
    "Standardizer",
    "SyntheticTEPConfig",
    "TEPDataset",
    "generate_synthetic_dataset",
    "load_dataset",
    "sliding_windows",
    "train_val_test_split_by_run",
]
