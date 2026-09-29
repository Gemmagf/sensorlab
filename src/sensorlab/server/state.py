"""Everything the API needs, computed once at start-up and kept in memory."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sensorlab.data import (
    SyntheticTEPConfig,
    TEPDataset,
    load_dataset,
    sliding_windows,
    train_val_test_split_by_run,
)
from sensorlab.diagnosis import explain_classifier, top_sensors_per_fault, window_features
from sensorlab.pipeline import MonitoringPipeline

log = logging.getLogger("sensorlab.server")

# Who owns which knob — the governance table. Mirrors docs/runbook.md §3.
OWNERSHIP: tuple[dict[str, str], ...] = (
    {"knob": "False-alarm cost", "owner": "Operations + finance", "key": "cost.false_alarm_cost"},
    {"knob": "Missed-fault cost", "owner": "Operations + finance", "key": "cost.missed_fault_cost"},
    {
        "knob": "Delay cost per minute",
        "owner": "Operations + finance",
        "key": "cost.delay_cost_per_min",
    },
    {
        "knob": "Intervention horizon (min)",
        "owner": "Maintenance planning",
        "key": "intervene_horizon_min",
    },
    {"knob": "FAR target", "owner": "Process engineering", "key": "far_target"},
    {"knob": "Primary detector", "owner": "Model owner", "key": "primary_detector"},
    {"knob": "Consecutive samples to confirm", "owner": "Model owner", "key": "consecutive"},
    {"knob": "Window / stride", "owner": "Model owner", "key": "window"},
)


@dataclass
class ServerState:
    pipeline: MonitoringPipeline
    ds: TEPDataset
    masks: dict[str, np.ndarray]
    scores: dict[str, np.ndarray]
    evaluation: dict[str, Any]
    predictions: pd.DataFrame
    results: dict[str, Any] | None
    model_path: str
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    audit: list[dict[str, Any]] = field(default_factory=list)
    _diagnosis: dict[str, Any] | None = None

    # ------------------------------------------------------------- derived

    @property
    def test_runs(self) -> np.ndarray:
        return np.unique(self.ds.run_id[self.masks["test"]])

    def normal_test_mask(self) -> np.ndarray:
        runs = [int(r) for r in self.test_runs if self.ds.run_fault_id[r] == 0]
        return self.ds.run_mask(runs)

    def reference_drift(self):
        """Drift status of the normal test runs — the health check's 'is the data still ours'."""
        m = self.normal_test_mask()
        if not m.any():
            return None
        return self.pipeline.check_drift(self.ds.X[m])

    def diagnosis(self, shap_samples: int = 300) -> dict[str, Any]:
        if self._diagnosis is None:
            pipe = self.pipeline
            Xz = pipe.scaler_.transform(self.ds.X)
            windows, _, end_idx = sliding_windows(
                Xz, self.ds.run_id, window=pipe.config.window, stride=pipe.config.stride
            )
            feats, fnames = window_features(windows, pipe.sensor_names_)
            in_test = np.where(self.masks["test"][end_idx])[0]
            idx = np.random.default_rng(0).choice(
                in_test, min(shap_samples, in_test.size), replace=False
            )
            rep = explain_classifier(
                pipe.classifier_,
                feats[idx],
                fnames,
                pipe.sensor_names_,
                max_background=shap_samples,
            )
            tops = top_sensors_per_fault(rep, k=3)
            self._diagnosis = {
                "metrics": {
                    k: v
                    for k, v in self.evaluation["diagnosis"].items()
                    if isinstance(v, int | float)
                },
                "top_sensors_per_fault": [
                    {
                        "fault_id": int(fid),
                        "fault_name": self.ds.fault_names[int(fid)],
                        "sensors": [{"sensor": s, "importance": round(v, 3)} for s, v in tops[fid]],
                    }
                    for fid in sorted(tops)
                ],
                "per_sensor_class_importance": {
                    "class_ids": [int(c) for c in rep.class_ids],
                    "sensors": rep.sensor_names,
                    "values": np.round(rep.per_sensor_class_importance, 4).tolist(),
                },
            }
        return self._diagnosis

    def run_summaries(self) -> list[dict[str, Any]]:
        rows = []
        for rid, g in self.predictions.groupby("run_id"):
            g = g.reset_index(drop=True)
            onset = int(self.ds.run_onsets[rid])
            true_fault = int(self.ds.run_fault_id[rid])
            alarms = g.index[g["alarm"]]
            pre = alarms[alarms < onset] if onset >= 0 else alarms
            post = alarms[alarms >= onset] if onset >= 0 else alarms[:0]
            first = int(post.min()) if len(post) else None
            if true_fault == 0:
                outcome = "false alarm" if len(alarms) else "clean"
            else:
                outcome = "caught" if first is not None else "missed"
            sm = self.ds.sample_minutes
            rows.append(
                {
                    "run_id": int(rid),
                    "true_fault_id": true_fault,
                    "true_fault": self.ds.fault_names[true_fault],
                    "onset_min": None if onset < 0 else onset * sm,
                    "first_alarm_min": None if first is None else first * sm,
                    "delay_min": None if first is None else (first - onset) * sm,
                    "pre_onset_alarms": len(pre) if true_fault > 0 else len(alarms),
                    "diagnosed_at_alarm": None
                    if first is None
                    else g.loc[first, "fault_name_pred"],
                    "rul_p50_at_alarm": None
                    if first is None
                    else round(float(g.loc[first, "rul_p50_min"])),
                    "final_action": g["action"].iloc[-1],
                    "outcome": outcome,
                }
            )
        return rows

    def acceptance_checks(self) -> list[dict[str, Any]]:
        """The runbook's release criteria, evaluated on the held-out runs of this process."""
        pipe = self.pipeline
        primary = pipe.primary_detector_
        det = self.evaluation["detection"][primary]
        dec = self.evaluation["decision"][primary]
        rul = self.evaluation["rul"]
        drift = self.reference_drift()
        far_limit = 2 * pipe.config.far_target
        checks = [
            {
                "id": "far",
                "name": "Observed FAR within 2x target (primary detector)",
                "value": det["far_observed"],
                "limit": far_limit,
                "passed": det["far_observed"] <= far_limit,
                "format": "pct",
            },
            {
                "id": "detected",
                "name": "Faulty test runs caught at the FAR threshold",
                "value": det["fraction_detected"],
                "limit": 0.9,
                "passed": det["fraction_detected"] >= 0.9,
                "format": "pct",
            },
            {
                "id": "missed",
                "name": "Missed faults at the operating threshold",
                "value": dec["missed_faults"],
                "limit": 0,
                "passed": dec["missed_faults"] == 0,
                "format": "int",
            },
            {
                "id": "coverage",
                "name": "RUL 80 % interval coverage within +/-5 pp of nominal",
                "value": rul["coverage_80"],
                "limit": 0.75,
                "passed": abs(rul["coverage_80"] - 0.8) <= 0.05,
                "format": "pct",
            },
            {
                "id": "drift",
                "name": "No drift alert on normal held-out runs",
                "value": drift.status if drift else "n/a",
                "limit": "stable",
                "passed": drift is not None and drift.status != "alert",
                "format": "text",
            },
        ]
        return checks

    def ownership(self) -> list[dict[str, Any]]:
        cfg = self.pipeline.config.to_dict()
        out = []
        for row in OWNERSHIP:
            key = row["key"]
            if key == "primary_detector":
                value = f"{self.pipeline.primary_detector_} ({cfg['primary_detector']})"
            elif key == "window":
                value = f"{cfg['window']} / {cfg['stride']}"
            elif key.startswith("cost."):
                value = cfg["cost"][key.split(".", 1)[1]]
            else:
                value = cfg[key]
            out.append({**row, "value": value})
        return out


# ----------------------------------------------------------------------- build


def _dataset_from_metadata(meta: dict[str, Any]) -> TEPDataset:
    if meta.get("data") == "real":
        root = meta.get("real_root")
        return load_dataset("real", real_root=Path(root) if root else None)
    return load_dataset(
        "synthetic",
        cfg=SyntheticTEPConfig(
            n_runs_per_fault=int(meta.get("n_runs_per_fault", 4)),
            n_normal_runs=int(meta.get("n_normal_runs", 12)),
            fault_run_minutes=int(meta.get("fault_run_minutes", 480)),
            seed=int(meta.get("seed", 0)),
        ),
    )


def build_state(
    model_path: str | Path,
    results_path: str | Path | None = None,
    pipeline: MonitoringPipeline | None = None,
    ds: TEPDataset | None = None,
    masks: dict[str, np.ndarray] | None = None,
) -> ServerState:
    """Load (or accept) a pipeline and precompute everything the dashboard shows.

    In production the evaluation set is whatever the manifest recorded at
    training time; tests pass ``pipeline``/``ds``/``masks`` directly.
    """
    pipe = pipeline or MonitoringPipeline.load(model_path)
    if ds is None:
        meta = pipe.metadata_.get("dataset") or {}
        if not meta:
            log.warning("manifest has no dataset provenance — using the default synthetic set")
        ds = _dataset_from_metadata(meta)
    if masks is None:
        seed = int((pipe.metadata_.get("dataset") or {}).get("seed", pipe.config.seed))
        tr, va, te = train_val_test_split_by_run(ds, seed=seed)
        masks = {"train": tr, "val": va, "test": te}

    log.info(
        "evaluating %s on %d held-out runs", model_path, np.unique(ds.run_id[masks["test"]]).size
    )
    scores = pipe.score_detectors(ds.X, ds.run_id)
    evaluation = pipe.evaluate(ds, masks["test"], shap_samples=0)
    predictions = pipe.predict(ds.X[masks["test"]], ds.run_id[masks["test"]])

    results = None
    if results_path and Path(results_path).exists():
        try:
            results = json.loads(Path(results_path).read_text())
        except json.JSONDecodeError:
            log.warning("could not parse %s", results_path)

    return ServerState(
        pipeline=pipe,
        ds=ds,
        masks=masks,
        scores=scores,
        evaluation=evaluation,
        predictions=predictions,
        results=results,
        model_path=str(model_path),
    )
