import numpy as np
import pytest

from sensorlab.data import TEPDataset, load_dataset


def test_load_synthetic_returns_dataset(tiny_dataset):
    assert isinstance(tiny_dataset, TEPDataset)
    assert tiny_dataset.X.ndim == 2
    assert tiny_dataset.X.shape[0] == tiny_dataset.fault_id.shape[0]


def test_to_dataframe_columns(tiny_dataset):
    df = tiny_dataset.to_dataframe()
    for col in ("fault_id", "fault_name", "is_anomaly", "run_id", "t_minutes"):
        assert col in df.columns
    for s in tiny_dataset.sensor_names:
        assert s in df.columns


def test_runs_iterator(tiny_dataset):
    runs = list(tiny_dataset.runs())
    assert len(runs) == tiny_dataset.n_runs
    rid, fid, X, mask, onset = runs[0]
    assert X.shape[1] == tiny_dataset.n_sensors


def test_load_real_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset("real", real_root=tmp_path / "does-not-exist")


def test_load_invalid_source_raises():
    with pytest.raises(ValueError):
        load_dataset("not-a-source")  # type: ignore[arg-type]


def test_time_axis_resets_per_run(tiny_dataset):
    df = tiny_dataset.to_dataframe()
    for rid in np.unique(df["run_id"]):
        sub = df[df["run_id"] == rid].sort_index()
        # t starts at 0 each run, increments uniformly
        assert sub["t_minutes"].iloc[0] == 0
        assert sub["t_minutes"].is_monotonic_increasing
