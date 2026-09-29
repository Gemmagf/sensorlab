import numpy as np
import pytest

from sensorlab.decision import CostModel
from sensorlab.detection import PCAMonitor
from sensorlab.evaluation import (
    aggregate_seeds,
    classification_report,
    evaluate_decision,
    evaluate_detector,
)


def _scores(tiny_standardized):
    normal = tiny_standardized["normal_train"]
    return (
        PCAMonitor(var_explained=0.9)
        .fit(tiny_standardized["X"][normal])
        .score(tiny_standardized["X"])
    )


def test_evaluate_detector_reports_on_eval_runs_only(tiny_dataset, tiny_splits, tiny_standardized):
    s = _scores(tiny_standardized)
    calib = tiny_splits["val"] & (tiny_dataset.fault_id == 0)
    rep = evaluate_detector("PCA", s, tiny_dataset, calib, tiny_splits["test"], far_target=0.05)
    n_test_runs = np.unique(tiny_dataset.run_id[tiny_splits["test"]]).size
    assert rep.n_eval_runs == n_test_runs
    assert 0.0 <= rep.auroc <= 1.0
    assert 0.0 <= rep.fraction_detected <= 1.0
    assert rep.to_dict()["name"] == "PCA"


def test_evaluate_detector_rejects_split_runs(tiny_dataset, tiny_splits, tiny_standardized):
    s = _scores(tiny_standardized)
    calib = tiny_splits["val"] & (tiny_dataset.fault_id == 0)
    bad = tiny_splits["test"].copy()
    first = np.where(bad)[0][0]
    bad[first] = False  # chop one sample off a test run
    with pytest.raises(ValueError, match="splits run"):
        evaluate_detector("PCA", s, tiny_dataset, calib, bad)


def test_evaluate_decision_returns_result(tiny_dataset, tiny_splits, tiny_standardized):
    s = _scores(tiny_standardized)
    r = evaluate_decision(s, tiny_dataset, tiny_splits["test"], CostModel(), n_grid=15)
    assert r.expected_cost >= 0
    assert r.n_faulty_runs > 0


def test_classification_report_keys():
    y = np.array([0, 1, 1, 2])
    rep = classification_report(y, y)
    assert rep["accuracy"] == 1.0 and rep["macro_f1"] == 1.0 and rep["n_samples"] == 4


def test_aggregate_seeds_mean_std_over_nested_dicts():
    a = {"det": {"auroc": 0.8, "name": "x"}, "n": 3}
    b = {"det": {"auroc": 0.6, "name": "x"}, "n": 5}
    agg = aggregate_seeds([a, b])
    assert agg["det"]["auroc"] == {"mean": 0.7, "std": pytest.approx(0.1), "n": 2}
    assert agg["det"]["name"] == "x"
    assert agg["n"]["mean"] == 4.0
    assert aggregate_seeds([]) == {}
