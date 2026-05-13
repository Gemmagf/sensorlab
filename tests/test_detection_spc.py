import numpy as np
import pytest

from sensorlab.detection import PCAMonitor


def test_spc_fit_and_score(tiny_dataset, tiny_standardized):
    normal = tiny_standardized["normal_train"]
    spc = PCAMonitor(var_explained=0.9).fit(tiny_standardized["X"][normal])
    scores = spc.score(tiny_standardized["X"])
    assert scores.shape == (tiny_dataset.n_samples,)
    assert (scores >= 0).all()


def test_spc_requires_fit_first(tiny_standardized):
    spc = PCAMonitor()
    with pytest.raises(RuntimeError):
        spc.score(tiny_standardized["X"][:10])


def test_spc_combined_differs_from_t2(tiny_standardized):
    normal = tiny_standardized["normal_train"]
    X = tiny_standardized["X"]
    a = PCAMonitor(var_explained=0.9, combined=True).fit(X[normal]).score(X[:50])
    b = PCAMonitor(var_explained=0.9, combined=False).fit(X[normal]).score(X[:50])
    assert not np.allclose(a, b)


def test_spc_n_components_increases_with_more_variance(tiny_standardized):
    normal = tiny_standardized["normal_train"]
    X = tiny_standardized["X"]
    low = PCAMonitor(var_explained=0.5).fit(X[normal]).n_components_
    high = PCAMonitor(var_explained=0.99).fit(X[normal]).n_components_
    assert low <= high


def test_spc_score_components_returns_t2_and_q(tiny_standardized):
    normal = tiny_standardized["normal_train"]
    X = tiny_standardized["X"]
    spc = PCAMonitor(var_explained=0.9).fit(X[normal])
    t2, q = spc.score_components(X[:30])
    assert t2.shape == q.shape == (30,)
    assert (t2 >= 0).all() and (q >= 0).all()


def test_spc_anomalous_samples_score_higher(tiny_dataset, tiny_standardized):
    normal = tiny_standardized["normal_train"]
    X = tiny_standardized["X"]
    spc = PCAMonitor(var_explained=0.9).fit(X[normal])
    scores = spc.score(X)
    anom_mean = scores[tiny_dataset.is_anomaly].mean()
    norm_mean = scores[~tiny_dataset.is_anomaly].mean()
    # In the synthetic data faults shift mean scores up
    assert anom_mean > norm_mean
