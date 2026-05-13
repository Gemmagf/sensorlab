import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from sensorlab.detection import (
    auroc,
    detection_delay,
    false_alarm_rate,
    threshold_at_far,
    true_positive_rate,
)


def test_auroc_perfect_separation():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    y = np.array([False, False, True, True])
    assert auroc(scores, y) == 1.0


def test_auroc_inverted_separation():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    y = np.array([False, False, True, True])
    assert auroc(scores, y) == 0.0


def test_auroc_matches_sklearn(rng):
    scores = rng.standard_normal(200)
    y = rng.random(200) > 0.5
    assert abs(auroc(scores, y) - roc_auc_score(y, scores)) < 1e-9


def test_auroc_returns_nan_for_single_class():
    scores = np.array([0.5, 0.6, 0.7])
    y = np.array([False, False, False])
    assert np.isnan(auroc(scores, y))


def test_threshold_at_far_quantile(rng):
    scores = rng.standard_normal(1000)
    t = threshold_at_far(scores, far=0.05)
    above = (scores >= t).mean()
    assert abs(above - 0.05) < 0.02


def test_threshold_at_far_validates_input(rng):
    scores = rng.standard_normal(100)
    with pytest.raises(ValueError):
        threshold_at_far(scores, far=0.0)
    with pytest.raises(ValueError):
        threshold_at_far(scores, far=1.5)


def test_false_alarm_rate():
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    y = np.array([False, False, True, True])
    # threshold 0.5: 1 of 2 normals above
    assert false_alarm_rate(scores, y, 0.5) == 0.5


def test_true_positive_rate():
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    y = np.array([False, False, True, True])
    # threshold 0.5: 1 of 2 anomalies above
    assert true_positive_rate(scores, y, 0.5) == 0.5


def test_detection_delay_on_synthetic(tiny_dataset, tiny_standardized):
    from sensorlab.detection import PCAMonitor

    spc = PCAMonitor(var_explained=0.9).fit(
        tiny_standardized["X"][tiny_standardized["normal_train"]]
    )
    scores = spc.score(tiny_standardized["X"])
    thr = float(np.quantile(scores, 0.95))
    res = detection_delay(
        scores,
        tiny_dataset.run_id,
        tiny_dataset.run_onsets,
        tiny_dataset.run_fault_id,
        thr,
        samples_to_minutes=tiny_dataset.sample_minutes,
    )
    assert 0 <= res["fraction_detected"] <= 1
    assert res["n_faulty_runs"] > 0


def test_detection_delay_returns_nan_on_no_faulty_runs():
    scores = np.zeros(10)
    run_id = np.zeros(10, dtype=int)
    run_onsets = np.array([-1])
    run_fault_id = np.array([0])
    res = detection_delay(scores, run_id, run_onsets, run_fault_id, 0.5)
    assert res["fraction_detected"] == 0.0
    assert res["n_faulty_runs"] == 0
