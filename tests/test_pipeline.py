import numpy as np
import pytest

from sensorlab.pipeline import (
    ACTION_INTERVENE,
    ACTION_INVESTIGATE,
    ACTION_SCHEDULE,
    ACTION_WAIT,
    ALL_DETECTORS,
    MonitoringPipeline,
    PipelineConfig,
    confirm_alarms,
)


def test_confirm_alarms_requires_consecutive_hits_within_run():
    above = np.array([1, 1, 1, 0, 1, 1, 1, 1, 0, 1], dtype=bool)
    run_id = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    out = confirm_alarms(above, run_id, consecutive=3)
    # run 0: third consecutive hit at index 2; run 1: hits at 5,6,7 → confirmed at 7 only
    np.testing.assert_array_equal(out, [0, 0, 1, 0, 0, 0, 0, 1, 0, 0])
    np.testing.assert_array_equal(confirm_alarms(above, run_id, consecutive=1), above)


def test_config_roundtrip_and_validation():
    cfg = PipelineConfig(detectors=("IForest", "PCA-T2Q"), primary_detector="IForest")
    assert PipelineConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ValueError):
        PipelineConfig(detectors=("nope",))
    with pytest.raises(ValueError):
        PipelineConfig(detectors=("IForest",), primary_detector="LSTM-AE")


def test_pipeline_fit_state(fitted_pipeline, tiny_dataset):
    p = fitted_pipeline
    assert p.fitted_
    assert set(p.detectors_) == set(ALL_DETECTORS)
    assert p.primary_detector_ in ALL_DETECTORS
    assert set(p.far_thresholds_) == set(p.decision_thresholds_) == set(ALL_DETECTORS)
    assert p.sensor_names_ == tiny_dataset.sensor_names
    assert p.fit_report_["n_train_runs"] > 0 and p.fit_report_["n_val_runs"] > 0


def test_pipeline_predict_schema_and_action_logic(fitted_pipeline, tiny_dataset, tiny_splits):
    te = tiny_splits["test"]
    out = fitted_pipeline.predict(tiny_dataset.X[te], tiny_dataset.run_id[te])
    assert len(out) == te.sum()
    for col in (
        "run_id",
        "t_minutes",
        "primary_score",
        "threshold",
        "alarm",
        "fault_id_pred",
        "fault_name_pred",
        "rul_p50_min",
        "action",
    ):
        assert col in out.columns
    for name in ALL_DETECTORS:
        assert f"score_{name}" in out.columns

    # decision rules are internally consistent
    assert (out.loc[~out["alarm"], "action"] == ACTION_WAIT).all()
    fired = out[out["alarm"]]
    assert (fired.loc[fired["fault_id_pred"] == 0, "action"] == ACTION_INVESTIGATE).all()
    with_fault = fired[fired["fault_id_pred"] > 0]
    horizon = fitted_pipeline.config.intervene_horizon_min
    assert (with_fault.loc[with_fault["rul_p50_min"] > horizon, "action"] == ACTION_SCHEDULE).all()
    assert (
        with_fault.loc[with_fault["rul_p50_min"] <= horizon, "action"] == ACTION_INTERVENE
    ).all()
    # RUL intervals are ordered and non-negative on the large majority of rows
    ok = (out["rul_p10_min"] <= out["rul_p50_min"] + 1e-6) & (
        out["rul_p50_min"] <= out["rul_p90_min"] + 1e-6
    )
    assert ok.mean() > 0.9
    assert (out["rul_p10_min"] >= 0).all()


def test_pipeline_predict_single_stream_without_run_id(fitted_pipeline, tiny_dataset):
    X = tiny_dataset.X[:50]
    out = fitted_pipeline.predict(X)
    assert (out["run_id"] == 0).all()
    assert out["t_minutes"].iloc[-1] == 49 * tiny_dataset.sample_minutes


def test_pipeline_predict_validates_input(fitted_pipeline, tiny_dataset):
    with pytest.raises(ValueError, match="expected X of shape"):
        fitted_pipeline.predict(tiny_dataset.X[:5, :3])
    bad = tiny_dataset.X[:5].copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        fitted_pipeline.predict(bad)
    with pytest.raises(ValueError, match="run_id"):
        fitted_pipeline.predict(tiny_dataset.X[:5], run_id=np.zeros(4))
    with pytest.raises(RuntimeError):
        MonitoringPipeline().predict(tiny_dataset.X[:5])


def test_pipeline_evaluate_on_test_runs(fitted_pipeline, tiny_dataset, tiny_splits):
    res = fitted_pipeline.evaluate(tiny_dataset, tiny_splits["test"], shap_samples=30)
    n_test_runs = np.unique(tiny_dataset.run_id[tiny_splits["test"]]).size
    assert res["n_eval_runs"] == n_test_runs
    for name in ALL_DETECTORS:
        assert 0 <= res["detection"][name]["auroc"] <= 1
        assert res["decision"][name]["n_faulty_runs"] > 0
    assert 0 <= res["diagnosis"]["accuracy"] <= 1
    assert res["diagnosis"]["top_sensors_per_fault"]
    assert res["rul"]["mae_minutes"] >= 0
    assert set(res["actions"]["action_counts"]) <= {
        ACTION_WAIT,
        ACTION_INVESTIGATE,
        ACTION_SCHEDULE,
        ACTION_INTERVENE,
    }


def test_pipeline_evaluate_rejects_partial_runs(fitted_pipeline, tiny_dataset, tiny_splits):
    bad = tiny_splits["test"].copy()
    bad[np.where(bad)[0][0]] = False
    with pytest.raises(ValueError, match="splits run"):
        fitted_pipeline.evaluate(tiny_dataset, bad, shap_samples=0)


def test_pipeline_save_load_roundtrip(fitted_pipeline, tiny_dataset, tiny_splits, tmp_path):
    path = fitted_pipeline.save(tmp_path / "pipe.joblib")
    assert path.exists() and (tmp_path / "pipe.manifest.json").exists()
    loaded = MonitoringPipeline.load(path)
    te = tiny_splits["test"]
    a = fitted_pipeline.predict(tiny_dataset.X[te], tiny_dataset.run_id[te])
    b = loaded.predict(tiny_dataset.X[te], tiny_dataset.run_id[te])
    assert a.equals(b)
    man = loaded.manifest()
    assert man["primary_detector"] == fitted_pipeline.primary_detector_
    assert man["n_sensors"] == tiny_dataset.n_sensors


def test_pipeline_drift_check(fitted_pipeline, tiny_dataset, tiny_splits):
    normal_train = tiny_splits["train"] & (tiny_dataset.fault_id == 0)
    assert fitted_pipeline.check_drift(tiny_dataset.X[normal_train]).status == "stable"
    shifted = tiny_dataset.X[normal_train].copy()
    shifted[:, 0] += 10 * tiny_dataset.X[:, 0].std()
    rep = fitted_pipeline.check_drift(shifted)
    assert tiny_dataset.sensor_names[0] in rep.alert


def test_pipeline_fixed_primary_detector(tiny_dataset, tiny_splits, fast_pipeline_config):
    cfg = PipelineConfig(
        **{
            **fast_pipeline_config.to_dict(),
            "detectors": ("PCA-T2Q", "IForest"),
            "primary_detector": "PCA-T2Q",
        }
    )
    p = MonitoringPipeline(cfg).fit(tiny_dataset, tiny_splits["train"], tiny_splits["val"])
    assert p.primary_detector_ == "PCA-T2Q"
    out = p.predict(tiny_dataset.X[:30])
    assert "score_LSTM-AE" not in out.columns
