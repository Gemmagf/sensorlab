import numpy as np

from sensorlab.diagnosis import window_features
from sensorlab.rul import QuantileRUL, build_rul_targets


def test_build_rul_targets_zero_at_run_end(tiny_dataset):
    rul, mask = build_rul_targets(
        tiny_dataset.run_id,
        tiny_dataset.is_anomaly,
        tiny_dataset.run_onsets,
        tiny_dataset.run_fault_id,
        samples_to_minutes=tiny_dataset.sample_minutes,
        cap_minutes=600,
    )
    # Each faulty run's last sample should have RUL=0
    for rid in range(tiny_dataset.n_runs):
        if tiny_dataset.run_fault_id[rid] == 0:
            continue
        sub = np.where(tiny_dataset.run_id == rid)[0]
        last = sub[-1]
        assert mask[last]
        assert rul[last] == 0.0


def test_build_rul_targets_mask_only_faulty(tiny_dataset):
    rul, mask = build_rul_targets(
        tiny_dataset.run_id,
        tiny_dataset.is_anomaly,
        tiny_dataset.run_onsets,
        tiny_dataset.run_fault_id,
    )
    # Mask True only where fault_id > 0 and post-onset
    assert mask[tiny_dataset.fault_id == 0].sum() == 0
    assert not mask[~tiny_dataset.is_anomaly].any()


def test_quantile_rul_fit_predict(tiny_dataset, tiny_windows, tiny_splits):
    W = tiny_windows["windows"]
    end_idx = tiny_windows["end_idx"]
    feats, _ = window_features(W, tiny_dataset.sensor_names)
    rul, mask = build_rul_targets(
        tiny_dataset.run_id,
        tiny_dataset.is_anomaly,
        tiny_dataset.run_onsets,
        tiny_dataset.run_fault_id,
        samples_to_minutes=tiny_dataset.sample_minutes,
    )
    rul_w = rul[end_idx]
    mask_w = mask[end_idx]
    train = tiny_splits["train"][end_idx] & mask_w
    test = tiny_splits["test"][end_idx] & mask_w

    qr = QuantileRUL(n_estimators=30, max_depth=2).fit(feats[train], rul_w[train])
    lo, med, hi = qr.predict_interval(feats[test])
    assert lo.shape == med.shape == hi.shape == (test.sum(),)
    # quantile order should hold for the large majority of samples
    assert ((lo <= med + 1e-6) & (med <= hi + 1e-6)).mean() > 0.9


def test_quantile_rul_metric_helpers():
    y = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 22.0, 28.0])
    assert abs(QuantileRUL.mae(y, y_pred) - 2.0) < 1e-9
    assert QuantileRUL.coverage(y, y - 1, y + 1) == 1.0
    assert QuantileRUL.coverage(y, y + 1, y + 2) == 0.0
    pb = QuantileRUL.pinball_loss(y, y_pred, q=0.5)
    assert pb >= 0


def test_conformal_calibration_lifts_coverage_to_nominal(rng):
    # heteroscedastic toy problem where raw quantile GBMs under-cover
    n = 3000
    X = rng.uniform(0, 10, (n, 2))
    y = X[:, 0] * 10 + rng.standard_normal(n) * (2 + X[:, 1] * 3)
    tr, cal, te = slice(0, 1500), slice(1500, 2200), slice(2200, n)
    qr = QuantileRUL(n_estimators=50, max_depth=3).fit(X[tr], y[tr])
    lo_raw, _, hi_raw = qr.predict_interval(X[te])
    qr.calibrate(X[cal], y[cal])
    assert qr.conformal_margin_ is not None and qr.n_calibration_ == 700
    lo, _, hi = qr.predict_interval(X[te])
    cov = QuantileRUL.coverage(y[te], lo, hi)
    assert 0.74 <= cov <= 0.88
    # the margin (positive or negative) is applied symmetrically
    np.testing.assert_allclose((hi - lo) - (hi_raw - lo_raw), 2 * qr.conformal_margin_)


def test_conformal_calibration_skipped_on_tiny_sets(rng):
    X = rng.standard_normal((60, 2))
    y = X[:, 0]
    qr = QuantileRUL(n_estimators=10, max_depth=2).fit(X, y)
    qr.calibrate(X[:5], y[:5])
    assert qr.conformal_margin_ is None
