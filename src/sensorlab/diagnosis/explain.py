"""SHAP attributions on the fault classifier.

TreeExplainer is exact and fast for XGBoost. We aggregate the per-feature
SHAP values back to the *sensor* level (summing across the four summary
features for each sensor) so the report says *"the dominant driver of fault 4
is the reactor temperature sensor"* rather than *"mean[XMEAS(9)]"*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shap

from sensorlab.diagnosis.classifier import FaultClassifier


@dataclass
class ShapReport:
    """SHAP attributions, kept per-feature and aggregated per-sensor."""

    shap_values: np.ndarray  # (n_classes, n_samples, n_features) — class-conditional
    feature_names: list[str]
    sensor_names: list[str]
    class_ids: np.ndarray
    per_sensor_class_importance: np.ndarray  # (n_classes, n_sensors) — mean|SHAP|

    def top_sensors(self, fault_id: int, k: int = 5) -> list[tuple[str, float]]:
        """Top-k sensors driving the prediction for the given fault id."""
        if fault_id not in self.class_ids:
            raise KeyError(f"fault_id {fault_id} not in classifier classes")
        cls_idx = int(np.where(self.class_ids == fault_id)[0][0])
        imps = self.per_sensor_class_importance[cls_idx]
        order = np.argsort(-imps)[:k]
        return [(self.sensor_names[i], float(imps[i])) for i in order]


def explain_classifier(
    clf: FaultClassifier,
    X: np.ndarray,
    feature_names: list[str],
    sensor_names: list[str],
    max_background: int = 200,
) -> ShapReport:
    """Compute SHAP values and aggregate them back to sensor-level importance."""
    explainer = shap.TreeExplainer(clf.booster)
    if X.shape[0] > max_background:
        idx = np.random.default_rng(0).choice(X.shape[0], max_background, replace=False)
        X_bg = X[idx]
    else:
        X_bg = X

    raw = explainer.shap_values(X_bg)
    # raw is either (n_samples, n_features) (binary) or list[len=n_classes] of those,
    # or a 3D array (n_samples, n_features, n_classes) on newer shap versions.
    shap_arr = _to_class_first(raw)

    abs_mean = np.abs(shap_arr).mean(axis=1)  # (n_classes, n_features)

    # Aggregate by sensor — features are named "mean[s]", "std[s]", "last[s]", "slope[s]"
    sensor_to_idx = {s: i for i, s in enumerate(sensor_names)}
    per_sensor = np.zeros((abs_mean.shape[0], len(sensor_names)), dtype=np.float64)
    for j, name in enumerate(feature_names):
        s_name = name[name.index("[") + 1 : -1]
        s_idx = sensor_to_idx.get(s_name)
        if s_idx is not None:
            per_sensor[:, s_idx] += abs_mean[:, j]

    return ShapReport(
        shap_values=shap_arr,
        feature_names=feature_names,
        sensor_names=sensor_names,
        class_ids=np.asarray(clf.classes_),
        per_sensor_class_importance=per_sensor,
    )


def top_sensors_per_fault(report: ShapReport, k: int = 5) -> dict[int, list[tuple[str, float]]]:
    return {
        int(fid): report.top_sensors(int(fid), k=k) for fid in report.class_ids if int(fid) != 0
    }


def _to_class_first(raw) -> np.ndarray:
    """Normalise SHAP output to shape ``(n_classes, n_samples, n_features)``."""
    if isinstance(raw, list):
        return np.stack(raw, axis=0)
    arr = np.asarray(raw)
    if arr.ndim == 3:
        # Newer shap: (n_samples, n_features, n_classes) → (n_classes, n_samples, n_features)
        return np.transpose(arr, (2, 0, 1))
    if arr.ndim == 2:
        # Binary: single class — add leading axis
        return arr[np.newaxis, ...]
    raise ValueError(f"unexpected SHAP output shape: {arr.shape}")
