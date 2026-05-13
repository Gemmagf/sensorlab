import numpy as np

from sensorlab.decision import (
    CostModel,
    cost_curve,
    evaluate_threshold,
    optimal_threshold,
)
from sensorlab.detection import PCAMonitor


def _fit_spc_and_score(tiny_dataset, tiny_standardized):
    normal = tiny_standardized["normal_train"]
    spc = PCAMonitor(var_explained=0.9).fit(tiny_standardized["X"][normal])
    return spc.score(tiny_standardized["X"])


def test_cost_model_defaults():
    cm = CostModel()
    assert cm.false_alarm_cost > 0
    assert cm.missed_fault_cost > 0
    assert cm.delay_cost_per_min >= 0


def test_evaluate_threshold_counts(tiny_dataset, tiny_standardized):
    scores = _fit_spc_and_score(tiny_dataset, tiny_standardized)
    cm = CostModel(false_alarm_cost=10.0, missed_fault_cost=1000.0, delay_cost_per_min=1.0)
    res = evaluate_threshold(
        scores,
        tiny_dataset.run_id,
        tiny_dataset.run_onsets,
        tiny_dataset.run_fault_id,
        threshold=float(np.median(scores)),
        cost=cm,
        samples_to_minutes=tiny_dataset.sample_minutes,
    )
    assert res.false_alarms >= 0
    assert res.missed_faults >= 0
    assert res.n_faulty_runs > 0


def test_cost_curve_shape(tiny_dataset, tiny_standardized):
    scores = _fit_spc_and_score(tiny_dataset, tiny_standardized)
    grid, results = cost_curve(
        scores,
        tiny_dataset.run_id,
        tiny_dataset.run_onsets,
        tiny_dataset.run_fault_id,
        cost=CostModel(),
        n_grid=20,
        samples_to_minutes=tiny_dataset.sample_minutes,
    )
    assert len(grid) == 20
    assert len(results) == 20
    assert all(r.expected_cost >= 0 for r in results)


def test_optimal_threshold_minimises_cost(tiny_dataset, tiny_standardized):
    scores = _fit_spc_and_score(tiny_dataset, tiny_standardized)
    grid, results = cost_curve(
        scores,
        tiny_dataset.run_id,
        tiny_dataset.run_onsets,
        tiny_dataset.run_fault_id,
        cost=CostModel(),
        n_grid=20,
        samples_to_minutes=tiny_dataset.sample_minutes,
    )
    best = optimal_threshold(
        scores,
        tiny_dataset.run_id,
        tiny_dataset.run_onsets,
        tiny_dataset.run_fault_id,
        cost=CostModel(),
        n_grid=20,
        samples_to_minutes=tiny_dataset.sample_minutes,
    )
    min_cost = min(r.expected_cost for r in results)
    assert best.expected_cost == min_cost


def test_extreme_thresholds_have_known_behaviour(tiny_dataset, tiny_standardized):
    scores = _fit_spc_and_score(tiny_dataset, tiny_standardized)
    cm = CostModel(false_alarm_cost=1.0, missed_fault_cost=1.0, delay_cost_per_min=0.0)
    huge = evaluate_threshold(
        scores,
        tiny_dataset.run_id,
        tiny_dataset.run_onsets,
        tiny_dataset.run_fault_id,
        threshold=float(scores.max() + 1.0),
        cost=cm,
        samples_to_minutes=tiny_dataset.sample_minutes,
    )
    # Nothing flagged: zero false alarms, all faults missed
    assert huge.false_alarms == 0
    assert huge.missed_faults == huge.n_faulty_runs
