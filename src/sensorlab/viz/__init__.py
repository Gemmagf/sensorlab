"""Plotting helpers (matplotlib only — keeps the dashboard import surface small)."""

from sensorlab.viz.plots import (
    plot_confusion_matrix,
    plot_cost_curve,
    plot_detection_scores,
    plot_pca_projection,
    plot_roc,
    plot_sensor_traces,
    plot_shap_summary,
)

__all__ = [
    "plot_confusion_matrix",
    "plot_cost_curve",
    "plot_detection_scores",
    "plot_pca_projection",
    "plot_roc",
    "plot_sensor_traces",
    "plot_shap_summary",
]
