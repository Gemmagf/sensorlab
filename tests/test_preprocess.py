import numpy as np

from sensorlab.data import (
    Standardizer,
    sliding_windows,
    train_val_test_split_by_run,
)


def test_standardizer_roundtrip(rng):
    X = rng.standard_normal((100, 5)).astype(np.float32)
    sc = Standardizer.fit(X)
    Z = sc.transform(X)
    X_back = sc.inverse_transform(Z)
    np.testing.assert_allclose(X, X_back, atol=1e-5)


def test_standardizer_handles_zero_std():
    X = np.zeros((50, 3), dtype=np.float32)
    X[:, 1] = 1.0
    sc = Standardizer.fit(X)
    Z = sc.transform(X)
    assert np.isfinite(Z).all()


def test_split_masks_are_disjoint_and_cover(tiny_dataset):
    tr, va, te = train_val_test_split_by_run(tiny_dataset, seed=3)
    assert not (tr & va).any()
    assert not (tr & te).any()
    assert not (va & te).any()
    assert int((tr | va | te).sum()) == tiny_dataset.n_samples


def test_split_is_stratified_by_fault(tiny_dataset):
    tr, va, te = train_val_test_split_by_run(tiny_dataset, seed=11)
    fault_train = set(tiny_dataset.fault_id[tr].tolist())
    # Every fault id (including 0) must appear in train
    for fid in range(22):
        assert fid in fault_train


def test_sliding_windows_shape(tiny_dataset, tiny_standardized):
    W, _, end_idx = sliding_windows(
        tiny_standardized["X"], tiny_dataset.run_id, window=10, stride=2
    )
    assert W.ndim == 3
    assert W.shape[1] == 10
    assert W.shape[2] == tiny_dataset.n_sensors
    assert end_idx.shape[0] == W.shape[0]


def test_sliding_windows_does_not_cross_runs(tiny_dataset, tiny_standardized):
    W, _, end_idx = sliding_windows(
        tiny_standardized["X"], tiny_dataset.run_id, window=10, stride=1
    )
    # The first sample in each window's range must share run_id with the last
    for ei in end_idx:
        start = ei - 10 + 1
        assert tiny_dataset.run_id[start] == tiny_dataset.run_id[ei]


def test_sliding_windows_with_labels(tiny_dataset, tiny_standardized):
    W, labels, end_idx = sliding_windows(
        tiny_standardized["X"],
        tiny_dataset.run_id,
        window=10,
        stride=2,
        labels=tiny_dataset.fault_id,
    )
    assert labels is not None
    assert labels.shape == (W.shape[0],)
    np.testing.assert_array_equal(labels, tiny_dataset.fault_id[end_idx])
