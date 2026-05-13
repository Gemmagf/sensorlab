"""Anomaly detection on the TEP sensor space.

Three complementary detectors are exposed with a common interface
(:class:`BaseDetector`): a classical multivariate SPC method
(:class:`PCAMonitor`), a tree-based unsupervised model
(:class:`IForestDetector`), and a deep recurrent autoencoder
(:class:`LSTMAutoencoder`). All three return higher scores for
more-anomalous windows so they can be evaluated and combined uniformly.
"""

from sensorlab.detection.autoencoder import LSTMAutoencoder
from sensorlab.detection.base import BaseDetector
from sensorlab.detection.iforest import IForestDetector
from sensorlab.detection.metrics import (
    auroc,
    detection_delay,
    false_alarm_rate,
    threshold_at_far,
    true_positive_rate,
)
from sensorlab.detection.spc import PCAMonitor

__all__ = [
    "BaseDetector",
    "IForestDetector",
    "LSTMAutoencoder",
    "PCAMonitor",
    "auroc",
    "detection_delay",
    "false_alarm_rate",
    "threshold_at_far",
    "true_positive_rate",
]
