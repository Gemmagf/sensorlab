import numpy as np
import pytest

from sensorlab.config import TEP
from sensorlab.data import SyntheticTEPConfig, generate_synthetic_dataset
from sensorlab.data.synthetic import FAULT_CATALOGUE


def test_fault_catalogue_has_21_entries():
    assert len(FAULT_CATALOGUE) == 21


def test_generator_returns_expected_shape(tiny_cfg):
    sim = generate_synthetic_dataset(tiny_cfg)
    n_total_runs = tiny_cfg.n_normal_runs + TEP.n_fault_types * tiny_cfg.n_runs_per_fault
    samples_per_run = int(tiny_cfg.fault_run_minutes / tiny_cfg.sample_minutes)
    assert sim.X.shape == (n_total_runs * samples_per_run, TEP.n_total)
    assert sim.fault_id.shape == (sim.X.shape[0],)
    assert sim.run_id.shape == (sim.X.shape[0],)


def test_generator_is_deterministic(tiny_cfg):
    sim_a = generate_synthetic_dataset(tiny_cfg)
    sim_b = generate_synthetic_dataset(tiny_cfg)
    np.testing.assert_array_equal(sim_a.X, sim_b.X)
    np.testing.assert_array_equal(sim_a.fault_id, sim_b.fault_id)


def test_normal_runs_have_no_anomalies(tiny_dataset):
    normal_mask = tiny_dataset.fault_id == 0
    assert not tiny_dataset.is_anomaly[normal_mask].any()


def test_fault_runs_have_anomalies_after_onset(tiny_dataset):
    for rid in range(tiny_dataset.n_runs):
        fault = int(tiny_dataset.run_fault_id[rid])
        if fault == 0:
            continue
        onset = int(tiny_dataset.run_onsets[rid])
        mask = tiny_dataset.run_id == rid
        per_run_anom = tiny_dataset.is_anomaly[mask]
        assert per_run_anom[onset:].all()
        assert not per_run_anom[:onset].any()


def test_all_fault_ids_are_present(tiny_dataset):
    present = {int(x) for x in np.unique(tiny_dataset.run_fault_id)}
    assert present == set(range(TEP.n_fault_types + 1))


def test_synthetic_sensors_are_finite(tiny_dataset):
    assert np.isfinite(tiny_dataset.X).all()


@pytest.mark.parametrize("seed", [0, 7, 99])
def test_different_seeds_yield_different_data(seed):
    a = generate_synthetic_dataset(
        SyntheticTEPConfig(n_normal_runs=2, n_runs_per_fault=1, fault_run_minutes=60, seed=seed)
    )
    b = generate_synthetic_dataset(
        SyntheticTEPConfig(n_normal_runs=2, n_runs_per_fault=1, fault_run_minutes=60, seed=seed + 1)
    )
    assert not np.array_equal(a.X, b.X)
