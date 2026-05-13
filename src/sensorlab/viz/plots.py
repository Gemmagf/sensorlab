"""Consistent matplotlib plotting helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

PALETTE = {
    "normal": "#3b82f6",
    "fault": "#ef4444",
    "score": "#1f2937",
    "threshold": "#f59e0b",
    "secondary": "#10b981",
}


def plot_sensor_traces(
    X: np.ndarray,
    sensor_names: list[str],
    sensor_idx: list[int] | None = None,
    is_anomaly: np.ndarray | None = None,
    onset: int | None = None,
    ax=None,
):
    """Time-series plot of selected sensor channels with fault region shaded."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    idx = sensor_idx or list(range(min(6, X.shape[1])))
    for i in idx:
        ax.plot(X[:, i], label=sensor_names[i], linewidth=1.0)
    if is_anomaly is not None and is_anomaly.any():
        ax.axvspan(
            int(np.where(is_anomaly)[0][0]),
            len(X) - 1,
            alpha=0.10,
            color=PALETTE["fault"],
            label="fault active",
        )
    elif onset is not None and onset > 0:
        ax.axvspan(onset, len(X) - 1, alpha=0.10, color=PALETTE["fault"])
    ax.set_xlabel("sample")
    ax.set_ylabel("sensor value")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    return ax


def plot_detection_scores(
    scores: np.ndarray,
    is_anomaly: np.ndarray | None = None,
    threshold: float | None = None,
    title: str | None = None,
    ax=None,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(scores, color=PALETTE["score"], linewidth=1.0, label="score")
    if is_anomaly is not None and is_anomaly.any():
        onset = int(np.where(is_anomaly)[0][0])
        ax.axvspan(onset, len(scores) - 1, alpha=0.10, color=PALETTE["fault"], label="fault active")
    if threshold is not None:
        ax.axhline(
            threshold, color=PALETTE["threshold"], linewidth=1.2, linestyle="--", label="threshold"
        )
    ax.set_xlabel("sample")
    ax.set_ylabel("anomaly score")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    return ax


def plot_roc(curves: dict[str, tuple[np.ndarray, np.ndarray]], ax=None):
    """``curves`` maps detector name → (fpr, tpr)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    for name, (fpr, tpr) in curves.items():
        ax.plot(fpr, tpr, label=name, linewidth=1.5)
    ax.plot([0, 1], [0, 1], "k:", alpha=0.5)
    ax.set_xlabel("false alarm rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("Detector ROC")
    ax.legend()
    return ax


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], ax=None, normalize: bool = True):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        cm = cm / np.maximum(row_sum, 1)
    im = ax.imshow(cm, cmap="Blues", aspect="auto", vmin=0, vmax=1 if normalize else cm.max())
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_shap_summary(
    per_sensor_importance: np.ndarray, sensor_names: list[str], class_ids, top_k: int = 10, ax=None
):
    """Heatmap of |SHAP|-by-class for the top-k most informative sensors overall."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    total = per_sensor_importance.sum(axis=0)
    top = np.argsort(-total)[:top_k]
    data = per_sensor_importance[:, top]
    im = ax.imshow(data, cmap="magma", aspect="auto")
    ax.set_xticks(range(top_k))
    ax.set_yticks(range(len(class_ids)))
    ax.set_xticklabels([sensor_names[i] for i in top], rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(
        [f"F{int(c):02d}" if int(c) > 0 else "Normal" for c in class_ids], fontsize=8
    )
    ax.set_title("Mean |SHAP| per (fault, sensor)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_pca_projection(
    X: np.ndarray,
    labels: np.ndarray,
    label_names: list[str] | None = None,
    ax=None,
    max_classes: int = 8,
):
    """2D PCA scatter coloured by fault id — quick sanity-check of separability."""
    from sklearn.decomposition import PCA

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    Z = PCA(n_components=2).fit_transform(X)
    classes = np.unique(labels)[:max_classes]
    cmap = plt.colormaps["tab10"]
    for i, c in enumerate(classes):
        mask = labels == c
        name = label_names[int(c)] if label_names else str(c)
        ax.scatter(Z[mask, 0], Z[mask, 1], s=6, alpha=0.6, color=cmap(i % 10), label=name)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=7, loc="best")
    return ax


def plot_cost_curve(thresholds: np.ndarray, results, optimum_idx: int | None = None, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    costs = np.array([r.expected_cost for r in results])
    ax.plot(thresholds, costs, color=PALETTE["score"], linewidth=1.5)
    if optimum_idx is None:
        optimum_idx = int(costs.argmin())
    ax.axvline(
        thresholds[optimum_idx],
        color=PALETTE["threshold"],
        linestyle="--",
        label=f"optimum @ t={thresholds[optimum_idx]:.3f}",
    )
    ax.set_xlabel("threshold")
    ax.set_ylabel("expected total cost")
    ax.set_title("Cost vs detection threshold")
    ax.legend()
    return ax
