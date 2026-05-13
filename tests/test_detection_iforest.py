import numpy as np
import pytest

from sensorlab.detection import IForestDetector


def test_iforest_fit_and_score(tiny_dataset, tiny_standardized):
    normal = tiny_standardized["normal_train"]
    det = IForestDetector(n_estimators=50).fit(tiny_standardized["X"][normal])
    scores = det.score(tiny_standardized["X"])
    assert scores.shape == (tiny_dataset.n_samples,)
    assert np.isfinite(scores).all()


def test_iforest_requires_fit(tiny_standardized):
    det = IForestDetector()
    with pytest.raises(RuntimeError):
        det.score(tiny_standardized["X"][:5])


def test_iforest_score_orientation(tiny_dataset, tiny_standardized):
    """Anomalous samples should on average score higher (= more anomalous)."""
    normal = tiny_standardized["normal_train"]
    X = tiny_standardized["X"]
    det = IForestDetector(n_estimators=100, random_state=0).fit(X[normal])
    scores = det.score(X)
    assert scores[tiny_dataset.is_anomaly].mean() > scores[~tiny_dataset.is_anomaly].mean()
