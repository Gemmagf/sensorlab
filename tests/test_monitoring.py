import numpy as np
import pytest

from sensorlab.monitoring import DriftMonitor, population_stability_index


def test_psi_zero_for_identical_samples(rng):
    x = rng.standard_normal(2000)
    edges = np.quantile(x, np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    assert population_stability_index(x, x, edges) == pytest.approx(0.0, abs=1e-9)


def test_drift_monitor_stable_then_alert(rng):
    names = ["a", "b", "c"]
    ref = rng.standard_normal((5000, 3))
    mon = DriftMonitor().fit(ref, names)

    same = rng.standard_normal((2000, 3))
    rep = mon.check(same)
    assert rep.status == "stable"
    assert rep.alert == [] and rep.n_current == 2000

    shifted = same.copy()
    shifted[:, 1] += 2.0  # sensor b recalibrated
    rep = mon.check(shifted)
    assert rep.status == "alert"
    assert rep.alert == ["b"]
    assert rep.psi["b"] > rep.psi["a"]
    assert rep.to_dict()["status"] == "alert"


def test_drift_monitor_validates_shape(rng):
    mon = DriftMonitor().fit(rng.standard_normal((100, 2)), ["a", "b"])
    with pytest.raises(ValueError):
        mon.check(rng.standard_normal((10, 3)))
    with pytest.raises(RuntimeError):
        DriftMonitor().check(rng.standard_normal((10, 2)))


def test_noise_floor_raises_thresholds_for_autocorrelated_runs(rng):
    # 6 reference "runs" whose mean wanders run to run (AR-like plant behaviour)
    names = ["a", "b"]
    parts, run_id = [], []
    for r in range(6):
        parts.append(rng.standard_normal((200, 2)) + np.array([rng.normal(0, 0.6), 0.0]))
        run_id.append(np.full(200, r))
    ref = np.vstack(parts)
    plain = DriftMonitor().fit(ref, names)
    calibrated = DriftMonitor().fit(ref, names, run_id=np.concatenate(run_id))
    # sensor a has real run-to-run variation -> its floor and thresholds go up; b stays textbook
    assert calibrated.noise_floor_["a"] > calibrated.noise_floor_["b"]
    assert calibrated.alert_thresholds_["a"] > plain.alert_thresholds_["a"] == 0.25
    new_run = rng.standard_normal((200, 2)) + np.array([0.7, 0.0])
    assert "a" in plain.check(new_run).alert  # textbook thresholds cry wolf
    assert calibrated.check(new_run).status == "stable"
    # a real recalibration is still caught
    recal = new_run.copy()
    recal[:, 0] += 3.0
    assert "a" in calibrated.check(recal).alert
