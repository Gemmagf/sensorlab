import numpy as np

from sensorlab.diagnosis import (
    FaultClassifier,
    explain_classifier,
    top_sensors_per_fault,
    window_features,
)


def test_window_features_shape(tiny_dataset, tiny_windows):
    W = tiny_windows["windows"]
    feats, names = window_features(W, tiny_dataset.sensor_names)
    assert feats.ndim == 2
    assert feats.shape[0] == W.shape[0]
    assert feats.shape[1] == 4 * tiny_dataset.n_sensors  # mean+std+last+slope per sensor
    assert len(names) == feats.shape[1]


def test_classifier_fit_predict_proba(tiny_dataset, tiny_windows, tiny_splits):
    W = tiny_windows["windows"]
    end_idx = tiny_windows["end_idx"]
    feats, fnames = window_features(W, tiny_dataset.sensor_names)
    labels = tiny_dataset.fault_id[end_idx]
    mask_train = tiny_splits["train"][end_idx]

    clf = FaultClassifier(n_estimators=30, max_depth=3).fit(
        feats[mask_train], labels[mask_train], feature_names=fnames
    )
    proba = clf.predict_proba(feats[:10])
    assert proba.shape == (10, len(clf.classes_))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_classifier_feature_importances_nonneg(tiny_dataset, tiny_windows, tiny_splits):
    W = tiny_windows["windows"]
    feats, fnames = window_features(W, tiny_dataset.sensor_names)
    labels = tiny_dataset.fault_id[tiny_windows["end_idx"]]
    mask_train = tiny_splits["train"][tiny_windows["end_idx"]]
    clf = FaultClassifier(n_estimators=30, max_depth=3).fit(
        feats[mask_train], labels[mask_train], feature_names=fnames
    )
    imp = clf.feature_importances_
    assert imp.shape == (feats.shape[1],)
    assert (imp >= 0).all()


def test_shap_report_shape(tiny_dataset, tiny_windows, tiny_splits):
    W = tiny_windows["windows"]
    end_idx = tiny_windows["end_idx"]
    feats, fnames = window_features(W, tiny_dataset.sensor_names)
    labels = tiny_dataset.fault_id[end_idx]
    mask_train = tiny_splits["train"][end_idx]
    clf = FaultClassifier(n_estimators=30, max_depth=3).fit(
        feats[mask_train], labels[mask_train], feature_names=fnames
    )
    rep = explain_classifier(clf, feats[:60], fnames, tiny_dataset.sensor_names, max_background=60)
    assert rep.shap_values.shape[0] == len(clf.classes_)
    assert rep.shap_values.shape[1] == 60
    assert rep.shap_values.shape[2] == feats.shape[1]
    assert rep.per_sensor_class_importance.shape == (len(clf.classes_), tiny_dataset.n_sensors)


def test_top_sensors_returns_k(tiny_dataset, tiny_windows, tiny_splits):
    W = tiny_windows["windows"]
    end_idx = tiny_windows["end_idx"]
    feats, fnames = window_features(W, tiny_dataset.sensor_names)
    labels = tiny_dataset.fault_id[end_idx]
    mask_train = tiny_splits["train"][end_idx]
    clf = FaultClassifier(n_estimators=30, max_depth=3).fit(
        feats[mask_train], labels[mask_train], feature_names=fnames
    )
    rep = explain_classifier(clf, feats[:40], fnames, tiny_dataset.sensor_names, max_background=40)
    per_fault = top_sensors_per_fault(rep, k=4)
    # Every non-zero class has 4 top sensors
    for fid, sensors in per_fault.items():
        assert fid != 0
        assert len(sensors) == 4
