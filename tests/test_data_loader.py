import numpy as np
import pandas as pd
import pytest

from sensorlab.config import TEP
from sensorlab.data import TEPDataset, dataset_from_rieth_frame, load_dataset


def test_load_synthetic_returns_dataset(tiny_dataset):
    assert isinstance(tiny_dataset, TEPDataset)
    assert tiny_dataset.X.ndim == 2
    assert tiny_dataset.X.shape[0] == tiny_dataset.fault_id.shape[0]


def test_to_dataframe_columns(tiny_dataset):
    df = tiny_dataset.to_dataframe()
    for col in ("fault_id", "fault_name", "active_fault_id", "is_anomaly", "run_id", "t_minutes"):
        assert col in df.columns
    for s in tiny_dataset.sensor_names:
        assert s in df.columns


def test_runs_iterator(tiny_dataset):
    runs = list(tiny_dataset.runs())
    assert len(runs) == tiny_dataset.n_runs
    rid, fid, X, mask, onset = runs[0]
    assert X.shape[1] == tiny_dataset.n_sensors


def test_active_fault_id_is_zero_before_onset(tiny_dataset):
    active = tiny_dataset.active_fault_id
    # pre-onset samples of faulty runs are nominal
    pre_onset = (tiny_dataset.fault_id > 0) & ~tiny_dataset.is_anomaly
    assert pre_onset.any()
    assert (active[pre_onset] == 0).all()
    # post-onset samples keep the scenario id
    np.testing.assert_array_equal(
        active[tiny_dataset.is_anomaly], tiny_dataset.fault_id[tiny_dataset.is_anomaly]
    )


def test_run_mask_selects_whole_runs(tiny_dataset):
    m = tiny_dataset.run_mask([0, 3])
    assert set(np.unique(tiny_dataset.run_id[m]).tolist()) == {0, 3}
    assert m.sum() == (tiny_dataset.run_id == 0).sum() + (tiny_dataset.run_id == 3).sum()


def test_validate_rejects_inconsistent_container(tiny_dataset):
    with pytest.raises(ValueError):
        TEPDataset(
            X=tiny_dataset.X[:10],
            fault_id=tiny_dataset.fault_id[:9],
            is_anomaly=tiny_dataset.is_anomaly[:10],
            run_id=tiny_dataset.run_id[:10],
            run_onsets=tiny_dataset.run_onsets,
            run_fault_id=tiny_dataset.run_fault_id,
            sensor_names=tiny_dataset.sensor_names,
            fault_names=tiny_dataset.fault_names,
            sample_minutes=3.0,
        )


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


# ---------------------------------------------------------------------------
# Rieth et al. 2017 layout
# ---------------------------------------------------------------------------


def _rieth_frame(split: str, faults: list[int], sim_runs: list[int], n_samples: int, seed=0):
    """Fake Rieth frame with lowercase columns: xmeas_1..3, xmv_1..2."""
    rng = np.random.default_rng(seed)
    rows = []
    for f in faults:
        for r in sim_runs:
            for s in range(1, n_samples + 1):
                rows.append({"faultNumber": f, "simulationRun": r, "sample": s, "split": split})
    df = pd.DataFrame(rows)
    for c in ("xmeas_1", "xmeas_2", "xmeas_3", "xmv_1", "xmv_2"):
        df[c] = rng.standard_normal(len(df))
    return df


def test_rieth_frame_builds_unique_runs_and_onsets():
    train = _rieth_frame("train", faults=[0, 1, 2], sim_runs=[1, 2], n_samples=30)
    test = _rieth_frame("test", faults=[0, 1], sim_runs=[1], n_samples=200)
    ds = dataset_from_rieth_frame(pd.concat([train, test], ignore_index=True))

    # (split, fault, simulationRun) → 6 train runs + 2 test runs, all unique
    assert ds.n_runs == 8
    assert sorted(np.unique(ds.run_id).tolist()) == list(range(8))
    assert ds.n_sensors == 5
    # column order canonicalised: XMEAS first then XMV, numeric order
    assert ds.sensor_names == ["XMEAS(1)", "XMEAS(2)", "XMEAS(3)", "XMV(1)", "XMV(2)"]

    # run-level fault labels survived the simulationRun collision across faults
    assert sorted(ds.run_fault_id.tolist()) == [0, 0, 0, 1, 1, 1, 2, 2]

    for rid, fid, X_run, mask_run, onset in ds.runs():
        n = X_run.shape[0]
        if fid == 0:
            assert onset == -1 and not mask_run.any()
        elif n == 30:  # train file → onset after sample 20
            assert onset == TEP.fault_onset_sample_train
            assert not mask_run[:20].any() and mask_run[20:].all()
        else:  # test file → onset after sample 160
            assert onset == TEP.fault_onset_sample_test
            assert not mask_run[:160].any() and mask_run[160:].all()


def test_rieth_frame_requires_columns():
    df = _rieth_frame("train", [0], [1], 5).drop(columns=["sample"])
    with pytest.raises(ValueError, match="missing columns"):
        dataset_from_rieth_frame(df)


def test_load_real_from_parquet_infers_split_from_filename(tmp_path):
    pytest.importorskip("pyarrow")
    root = tmp_path / "tep"
    root.mkdir()
    tr = _rieth_frame("train", [0, 3], [1], 25).drop(columns=["split"])
    te = _rieth_frame("test", [3], [1], 170).drop(columns=["split"])
    tr.to_parquet(root / "TEP_Faulty_Training__x.parquet", index=False)
    te.to_parquet(root / "TEP_Faulty_Testing__x.parquet", index=False)
    ds = load_dataset("real", real_root=root)
    assert ds.n_runs == 3
    onsets = {int(o) for o in ds.run_onsets if o >= 0}
    assert onsets == {TEP.fault_onset_sample_train, TEP.fault_onset_sample_test}
