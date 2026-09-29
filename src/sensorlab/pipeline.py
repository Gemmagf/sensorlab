"""The deployable artefact: one object that goes from raw sensor rows to a recommendation.

``MonitoringPipeline`` bundles the scaler, the three detectors, the fault
classifier, the RUL head, the calibrated thresholds, the cost model and a
drift monitor. It is what gets trained in the lab, serialised, shipped to the
plant, and called on every new batch of telemetry::

    pipe = MonitoringPipeline(PipelineConfig(ae_epochs=25)).fit(ds, train_mask, val_mask)
    pipe.save("models/pipeline.joblib")

    pipe = MonitoringPipeline.load("models/pipeline.joblib")
    out = pipe.predict(X_new, run_id=batch_ids)   # DataFrame with alarm / fault / RUL / action
    drift = pipe.check_drift(X_new)               # PSI per sensor

Design rules baked in (so a hand-over to the plant team cannot break them):

* detectors and the scaler see **normal training runs only**;
* FAR thresholds are calibrated on **validation-normal** samples, the
  cost-optimal operating threshold on **validation runs** — the test split
  is never touched during ``fit``;
* window-level outputs (LSTM-AE, classifier, RUL) are mapped to the sample
  axis through one helper, so every consumer is aligned on the same onsets
  and the same time unit;
* the classifier learns :attr:`TEPDataset.active_fault_id` (0 before the
  onset), never the run-level scenario id.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

import joblib
import numpy as np
import pandas as pd

from sensorlab import __version__
from sensorlab.data.loader import TEPDataset
from sensorlab.data.preprocess import (
    Standardizer,
    sliding_windows,
    train_val_test_split_by_run,
    window_index_per_sample,
    windows_to_per_sample,
)
from sensorlab.decision.cost import CostModel, DecisionResult, evaluate_threshold
from sensorlab.detection.autoencoder import LSTMAutoencoder
from sensorlab.detection.iforest import IForestDetector
from sensorlab.detection.metrics import (
    auroc,
    detection_delay,
    false_alarm_rate,
    threshold_at_far,
    true_positive_rate,
)
from sensorlab.detection.spc import PCAMonitor
from sensorlab.diagnosis.classifier import FaultClassifier, window_features
from sensorlab.diagnosis.explain import explain_classifier, top_sensors_per_fault
from sensorlab.evaluation import (
    _check_whole_runs,
    classification_report,
    evaluate_decision,
)
from sensorlab.monitoring import DriftMonitor, DriftReport
from sensorlab.rul.quantile import QuantileRUL, build_rul_targets

log = logging.getLogger("sensorlab.pipeline")

DETECTOR_PCA = "PCA-T2Q"
DETECTOR_IFOREST = "IForest"
DETECTOR_LSTM = "LSTM-AE"
ALL_DETECTORS: tuple[str, ...] = (DETECTOR_PCA, DETECTOR_IFOREST, DETECTOR_LSTM)

ACTION_WAIT = "wait"
ACTION_INVESTIGATE = "investigate"
ACTION_SCHEDULE = "schedule_maintenance"
ACTION_INTERVENE = "intervene_now"


@dataclass
class PipelineConfig:
    """Every knob of the pipeline in one serialisable place."""

    window: int = 20
    stride: int = 2
    far_target: float = 0.01
    consecutive: int = 3
    detectors: tuple[str, ...] = ALL_DETECTORS
    primary_detector: str = "auto"  # or one of ALL_DETECTORS
    pca_var_explained: float = 0.90
    iforest_estimators: int = 300
    ae_epochs: int = 25
    ae_hidden: int = 32
    ae_latent: int = 8
    xgb_estimators: int = 300
    xgb_max_depth: int = 6
    rul_estimators: int = 200
    rul_max_depth: int = 4
    rul_cap_minutes: float = 600.0
    intervene_horizon_min: float = 120.0
    cost: CostModel = field(default_factory=CostModel)
    seed: int = 0

    def __post_init__(self) -> None:
        unknown = set(self.detectors) - set(ALL_DETECTORS)
        if unknown:
            raise ValueError(f"unknown detectors {sorted(unknown)}; choose from {ALL_DETECTORS}")
        if self.primary_detector != "auto" and self.primary_detector not in self.detectors:
            raise ValueError("primary_detector must be 'auto' or one of config.detectors")
        if isinstance(self.cost, dict):
            self.cost = CostModel(**self.cost)
        self.detectors = tuple(self.detectors)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["detectors"] = list(self.detectors)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PipelineConfig:
        d = dict(d)
        if "detectors" in d:
            d["detectors"] = tuple(d["detectors"])
        return cls(**d)


def confirm_alarms(above: np.ndarray, run_id: np.ndarray, consecutive: int) -> np.ndarray:
    """True where the score has been above threshold for ``consecutive`` samples of the same run.

    Same rule as :func:`sensorlab.detection.metrics.detection_delay` and the
    decision layer, so what the pipeline alarms on is what was evaluated.
    """
    above = np.asarray(above, dtype=bool)
    out = np.zeros_like(above)
    if consecutive <= 1:
        return above.copy()
    for r in np.unique(run_id):
        pos = np.where(run_id == r)[0]
        a = above[pos]
        idx = np.arange(a.size)
        last_false = np.maximum.accumulate(np.where(~a, idx, -1))
        run_len = idx - last_false
        out[pos] = a & (run_len >= consecutive)
    return out


class MonitoringPipeline:
    """Fit once, serialise, score forever. See module docstring."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.fitted_: bool = False
        self.sensor_names_: list[str] = []
        self.fault_names_: list[str] = []
        self.sample_minutes_: float = 3.0
        self.scaler_: Standardizer | None = None
        self.detectors_: dict[str, Any] = {}
        self.far_thresholds_: dict[str, float] = {}
        self.decision_thresholds_: dict[str, float] = {}
        self.primary_detector_: str = ""
        self.classifier_: FaultClassifier | None = None
        self.feature_names_: list[str] = []
        self.rul_: QuantileRUL | None = None
        self.drift_: DriftMonitor | None = None
        self.fit_report_: dict[str, Any] = {}
        self.metadata_: dict[str, Any] = {}  # free-form provenance (dataset, git sha, …)
        self.fitted_at_: str = ""
        self.version_: str = __version__

    # ------------------------------------------------------------------ fit

    def fit(
        self,
        ds: TEPDataset,
        train_mask: np.ndarray | None = None,
        val_mask: np.ndarray | None = None,
    ) -> Self:
        cfg = self.config
        if train_mask is None or val_mask is None:
            train_mask, val_mask, _ = train_val_test_split_by_run(
                ds, frac_train=0.75, frac_val=0.25, seed=cfg.seed
            )
        normal_train = train_mask & (ds.fault_id == 0)
        normal_val = val_mask & (ds.fault_id == 0)
        if normal_train.sum() == 0 or normal_val.sum() == 0:
            raise ValueError("fit needs normal runs in both the train and the validation split")

        self.sensor_names_ = list(ds.sensor_names)
        self.fault_names_ = list(ds.fault_names)
        self.sample_minutes_ = float(ds.sample_minutes)
        timings: dict[str, float] = {}

        log.info("fit: %d train samples, %d val samples", train_mask.sum(), val_mask.sum())
        self.scaler_ = Standardizer.fit(ds.X[normal_train])
        Xz = self.scaler_.transform(ds.X)

        # --- detectors (normal-only) ------------------------------------------------
        self.detectors_ = {}
        for name in cfg.detectors:
            t0 = time.time()
            self.detectors_[name] = self._fit_detector(name, Xz, ds.run_id, normal_train)
            timings[f"fit_{name}"] = time.time() - t0
            log.info("fitted %s in %.1fs", name, timings[f"fit_{name}"])
        scores = self._detector_scores(Xz, ds.run_id)

        # --- thresholds: FAR on validation-normal, cost-optimal on validation runs ---
        self.far_thresholds_ = {
            n: threshold_at_far(s[normal_val], far=cfg.far_target) for n, s in scores.items()
        }
        val_decisions: dict[str, DecisionResult] = {
            n: evaluate_decision(s, ds, val_mask, cfg.cost, consecutive=cfg.consecutive)
            for n, s in scores.items()
        }
        self.decision_thresholds_ = {n: r.threshold for n, r in val_decisions.items()}
        if cfg.primary_detector == "auto":
            self.primary_detector_ = min(
                val_decisions, key=lambda n: val_decisions[n].expected_cost
            )
        else:
            self.primary_detector_ = cfg.primary_detector
        log.info("primary detector: %s", self.primary_detector_)

        # --- diagnosis + RUL on window features -------------------------------------
        windows, _, end_idx = sliding_windows(Xz, ds.run_id, window=cfg.window, stride=cfg.stride)
        feats, self.feature_names_ = window_features(windows, self.sensor_names_)
        labels = ds.active_fault_id[end_idx]
        train_w = train_mask[end_idx]

        t0 = time.time()
        self.classifier_ = FaultClassifier(
            n_estimators=cfg.xgb_estimators, max_depth=cfg.xgb_max_depth, random_state=cfg.seed
        ).fit(feats[train_w], labels[train_w], feature_names=self.feature_names_)
        timings["fit_classifier"] = time.time() - t0

        rul, rul_mask = build_rul_targets(
            ds.run_id,
            ds.is_anomaly,
            ds.run_onsets,
            ds.run_fault_id,
            samples_to_minutes=ds.sample_minutes,
            cap_minutes=cfg.rul_cap_minutes,
        )
        rul_train = train_w & rul_mask[end_idx]
        rul_cal = val_mask[end_idx] & rul_mask[end_idx]
        t0 = time.time()
        self.rul_ = QuantileRUL(
            n_estimators=cfg.rul_estimators, max_depth=cfg.rul_max_depth, random_state=cfg.seed
        ).fit(feats[rul_train], rul[end_idx][rul_train])
        # conformal margin on validation faulty windows → honest 80 % interval
        self.rul_.calibrate(feats[rul_cal], rul[end_idx][rul_cal])
        timings["fit_rul"] = time.time() - t0

        # --- drift reference on raw training-normal data ----------------------------
        self.drift_ = DriftMonitor().fit(
            ds.X[normal_train], self.sensor_names_, run_id=ds.run_id[normal_train]
        )

        self.fit_report_ = {
            "n_train_runs": int(np.unique(ds.run_id[train_mask]).size),
            "n_val_runs": int(np.unique(ds.run_id[val_mask]).size),
            "far_thresholds": dict(self.far_thresholds_),
            "decision_thresholds": dict(self.decision_thresholds_),
            "validation_expected_cost": {n: r.expected_cost for n, r in val_decisions.items()},
            "primary_detector": self.primary_detector_,
            "timings_seconds": timings,
        }
        self.fitted_at_ = datetime.now(UTC).isoformat(timespec="seconds")
        self.fitted_ = True
        return self

    def _fit_detector(self, name: str, Xz: np.ndarray, run_id: np.ndarray, normal_train):
        cfg = self.config
        if name == DETECTOR_PCA:
            return PCAMonitor(var_explained=cfg.pca_var_explained).fit(Xz[normal_train])
        if name == DETECTOR_IFOREST:
            return IForestDetector(n_estimators=cfg.iforest_estimators, random_state=cfg.seed).fit(
                Xz[normal_train]
            )
        if name == DETECTOR_LSTM:
            windows, _, end_idx = sliding_windows(Xz, run_id, window=cfg.window, stride=cfg.stride)
            ae = LSTMAutoencoder(
                window=cfg.window,
                epochs=cfg.ae_epochs,
                hidden=cfg.ae_hidden,
                latent=cfg.ae_latent,
                seed=cfg.seed,
            )
            return ae.fit(windows[normal_train[end_idx]])
        raise ValueError(name)

    # -------------------------------------------------------------- scoring

    def _check_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError("MonitoringPipeline must be fit (or loaded) before use")

    def _detector_scores(self, Xz: np.ndarray, run_id: np.ndarray) -> dict[str, np.ndarray]:
        """Per-sample anomaly score for every detector, aligned on the sample axis."""
        cfg = self.config
        out: dict[str, np.ndarray] = {}
        for name, det in self.detectors_.items():
            if name == DETECTOR_LSTM:
                windows, _, end_idx = sliding_windows(
                    Xz, run_id, window=cfg.window, stride=cfg.stride
                )
                win_scores = det.score(windows) if windows.shape[0] else np.empty(0, np.float32)
                out[name] = windows_to_per_sample(win_scores, end_idx, run_id, Xz.shape[0])
            else:
                out[name] = np.asarray(det.score(Xz), dtype=np.float32)
        return out

    def score_detectors(self, X: np.ndarray, run_id: np.ndarray | None = None):
        """Per-sample scores of every detector on raw (unscaled) sensor rows."""
        self._check_fitted()
        X, run_id = self._coerce_input(X, run_id)
        return self._detector_scores(self.scaler_.transform(X), run_id)

    def predict(self, X: np.ndarray, run_id: np.ndarray | None = None) -> pd.DataFrame:
        """Full inference: scores, confirmed alarm, fault diagnosis, RUL interval, action.

        Columns
        -------
        run_id, t_index, t_minutes, score_<detector>..., primary_score, threshold,
        above_threshold, alarm, fault_id_pred, fault_name_pred, fault_confidence,
        rul_p10_min, rul_p50_min, rul_p90_min, action
        """
        self._check_fitted()
        cfg = self.config
        X, run_id = self._coerce_input(X, run_id)
        n = X.shape[0]
        Xz = self.scaler_.transform(X)
        scores = self._detector_scores(Xz, run_id)

        primary = scores[self.primary_detector_]
        thr = self.decision_thresholds_[self.primary_detector_]
        above = primary >= thr
        alarm = confirm_alarms(above, run_id, cfg.consecutive)

        windows, _, end_idx = sliding_windows(Xz, run_id, window=cfg.window, stride=cfg.stride)
        n_classes = len(self.classifier_.classes_)
        if windows.shape[0]:
            feats, _ = window_features(windows, self.sensor_names_)
            proba_w = self.classifier_.predict_proba(feats)
            lo_w, med_w, hi_w = self.rul_.predict_interval(feats)
        else:
            feats = np.empty((0, len(self.feature_names_)), np.float32)
            proba_w = np.empty((0, n_classes))
            lo_w = med_w = hi_w = np.empty(0)
        idx = window_index_per_sample(end_idx, run_id, n)
        has_win = idx >= 0
        proba = np.full((n, n_classes), np.nan)
        proba[has_win] = proba_w[idx[has_win]]
        fault_pred = np.zeros(n, dtype=int)
        conf = np.full(n, np.nan)
        if has_win.any():
            best = np.argmax(proba[has_win], axis=1)
            fault_pred[has_win] = self.classifier_.classes_[best]
            conf[has_win] = proba[has_win][np.arange(best.size), best]
        rul_lo = np.full(n, np.nan)
        rul_med = np.full(n, np.nan)
        rul_hi = np.full(n, np.nan)
        rul_lo[has_win] = np.maximum(lo_w[idx[has_win]], 0)
        rul_med[has_win] = np.maximum(med_w[idx[has_win]], 0)
        rul_hi[has_win] = np.maximum(hi_w[idx[has_win]], 0)

        action = np.full(n, ACTION_WAIT, dtype=object)
        fired = alarm
        action[fired & (fault_pred == 0)] = ACTION_INVESTIGATE
        fault_fired = fired & (fault_pred > 0)
        action[fault_fired & (rul_med > cfg.intervene_horizon_min)] = ACTION_SCHEDULE
        action[fault_fired & ~(rul_med > cfg.intervene_horizon_min)] = ACTION_INTERVENE

        t_index = np.zeros(n, dtype=int)
        for r in np.unique(run_id):
            pos = np.where(run_id == r)[0]
            t_index[pos] = np.arange(pos.size)

        df = pd.DataFrame({"run_id": run_id, "t_index": t_index})
        df["t_minutes"] = t_index * self.sample_minutes_
        for name, s in scores.items():
            df[f"score_{name}"] = s
        df["primary_score"] = primary
        df["threshold"] = thr
        df["above_threshold"] = above
        df["alarm"] = alarm
        df["fault_id_pred"] = fault_pred
        df["fault_name_pred"] = [self.fault_names_[int(f)] for f in fault_pred]
        df["fault_confidence"] = conf
        df["rul_p10_min"] = rul_lo
        df["rul_p50_min"] = rul_med
        df["rul_p90_min"] = rul_hi
        df["action"] = action
        return df

    def check_drift(self, X: np.ndarray) -> DriftReport:
        self._check_fitted()
        return self.drift_.check(np.asarray(X))

    def _coerce_input(self, X, run_id):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != len(self.sensor_names_):
            raise ValueError(
                f"expected X of shape (n, {len(self.sensor_names_)}) in the training sensor "
                f"order {self.sensor_names_[:3]}…, got {X.shape}"
            )
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN/inf — impute before scoring")
        if run_id is None:
            run_id = np.zeros(X.shape[0], dtype=np.int64)
        run_id = np.asarray(run_id)
        if run_id.shape[0] != X.shape[0]:
            raise ValueError("run_id must have one entry per row of X")
        return X, run_id

    # ----------------------------------------------------------- evaluation

    def evaluate(self, ds: TEPDataset, eval_mask: np.ndarray, shap_samples: int = 300) -> dict:
        """Held-out evaluation of every layer on whole runs selected by ``eval_mask``."""
        self._check_fitted()
        cfg = self.config
        _check_whole_runs(eval_mask, ds.run_id)
        Xz = self.scaler_.transform(ds.X)
        scores = self._detector_scores(Xz, ds.run_id)

        detection: dict[str, dict] = {}
        decision: dict[str, dict] = {}
        for name, s in scores.items():
            # FAR threshold fixed at fit time (validation-normal); metrics on eval runs only
            thr = self.far_thresholds_[name]
            s_eval, y_eval = s[eval_mask], ds.is_anomaly[eval_mask]
            delay = detection_delay(
                s_eval,
                ds.run_id[eval_mask],
                ds.run_onsets,
                ds.run_fault_id,
                thr,
                samples_to_minutes=ds.sample_minutes,
                consecutive=cfg.consecutive,
            )
            detection[name] = {
                "threshold": float(thr),
                "far_target": cfg.far_target,
                "auroc": auroc(s_eval, y_eval),
                "tpr_at_far": true_positive_rate(s_eval, y_eval, thr),
                "far_observed": false_alarm_rate(s_eval, y_eval, thr),
                "fraction_detected": float(delay["fraction_detected"]),
                "median_delay_min": float(delay["median_min"]),
                "mean_delay_min": float(delay["mean_min"]),
                "p90_delay_min": float(delay["p90_min"]),
                "n_eval_faulty_runs": int(delay["n_faulty_runs"]),
                "fit_seconds": self.fit_report_["timings_seconds"].get(f"fit_{name}"),
            }
            # decision layer: operating threshold fixed at fit time (validation runs),
            # its cost measured on the eval runs
            r = evaluate_threshold(
                s_eval,
                ds.run_id[eval_mask],
                ds.run_onsets,
                ds.run_fault_id,
                self.decision_thresholds_[name],
                cfg.cost,
                samples_to_minutes=ds.sample_minutes,
                consecutive=cfg.consecutive,
            )
            decision[name] = {
                "threshold": r.threshold,
                "expected_cost": r.expected_cost,
                "false_alarms": r.false_alarms,
                "missed_faults": r.missed_faults,
                "mean_delay_min": r.mean_delay_min,
                "n_normal_samples": r.n_normal_samples,
                "n_faulty_runs": r.n_faulty_runs,
            }

        windows, _, end_idx = sliding_windows(Xz, ds.run_id, window=cfg.window, stride=cfg.stride)
        feats, _ = window_features(windows, self.sensor_names_)
        labels = ds.active_fault_id[end_idx]
        eval_w = eval_mask[end_idx]
        preds = self.classifier_.predict(feats[eval_w])
        diagnosis = classification_report(labels[eval_w], preds)
        diagnosis["fit_seconds"] = self.fit_report_["timings_seconds"].get("fit_classifier")
        if shap_samples > 0 and eval_w.sum() > 0:
            rep = explain_classifier(
                self.classifier_,
                feats[eval_w][:shap_samples],
                self.feature_names_,
                self.sensor_names_,
                max_background=shap_samples,
            )
            diagnosis["top_sensors_per_fault"] = {
                int(fid): [(n, round(v, 3)) for n, v in tops]
                for fid, tops in top_sensors_per_fault(rep, k=3).items()
            }

        rul, rul_mask = build_rul_targets(
            ds.run_id,
            ds.is_anomaly,
            ds.run_onsets,
            ds.run_fault_id,
            samples_to_minutes=ds.sample_minutes,
            cap_minutes=cfg.rul_cap_minutes,
        )
        rul_eval = eval_w & rul_mask[end_idx]
        lo, med, hi = self.rul_.predict_interval(feats[rul_eval])
        lo_raw, _, hi_raw = self.rul_._raw_interval(feats[rul_eval])
        y = rul[end_idx][rul_eval]
        rul_res = {
            "mae_minutes": self.rul_.mae(y, med),
            "coverage_80": self.rul_.coverage(y, lo, hi),
            "coverage_80_uncalibrated": self.rul_.coverage(y, lo_raw, hi_raw),
            "conformal_margin_min": self.rul_.conformal_margin_,
            "n_calibration_samples": self.rul_.n_calibration_,
            "pinball_50": self.rul_.pinball_loss(y, med, 0.5),
            "mean_interval_width_min": float(np.mean(hi - lo)),
            "n_eval_samples": int(rul_eval.sum()),
            "fit_seconds": self.fit_report_["timings_seconds"].get("fit_rul"),
        }

        # end-to-end action quality on eval runs: did we recommend the right thing?
        out = self.predict(ds.X[eval_mask], ds.run_id[eval_mask])
        y_anom = ds.is_anomaly[eval_mask]
        alarm = out["alarm"].to_numpy()
        actions = {
            "alarm_precision": float(y_anom[alarm].mean()) if alarm.any() else float("nan"),
            "alarm_recall": float(alarm[y_anom].mean()) if y_anom.any() else float("nan"),
            "action_counts": out["action"].value_counts().to_dict(),
        }

        return {
            "n_eval_runs": int(np.unique(ds.run_id[eval_mask]).size),
            "primary_detector": self.primary_detector_,
            "detection": detection,
            "diagnosis": diagnosis,
            "rul": rul_res,
            "decision": decision,
            "actions": actions,
        }

    # ---------------------------------------------------------- persistence

    def manifest(self) -> dict[str, Any]:
        """Human-readable summary written next to the model (model card lite)."""
        self._check_fitted()
        return {
            "sensorlab_version": self.version_,
            "fitted_at": self.fitted_at_,
            "config": self.config.to_dict(),
            "sensor_names": self.sensor_names_,
            "n_sensors": len(self.sensor_names_),
            "fault_names": self.fault_names_,
            "sample_minutes": self.sample_minutes_,
            "primary_detector": self.primary_detector_,
            "far_thresholds": self.far_thresholds_,
            "decision_thresholds": self.decision_thresholds_,
            "fit_report": self.fit_report_,
            "metadata": self.metadata_,
        }

    def save(self, path: str | Path) -> Path:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        path.with_suffix(".manifest.json").write_text(
            json.dumps(self.manifest(), indent=2, default=str)
        )
        log.info("saved pipeline to %s", path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> MonitoringPipeline:
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a MonitoringPipeline")
        if obj.version_ != __version__:
            log.warning(
                "pipeline was saved with sensorlab %s, running %s", obj.version_, __version__
            )
        return obj
