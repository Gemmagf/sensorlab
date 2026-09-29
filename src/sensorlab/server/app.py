"""FastAPI application: JSON API + static governance dashboard."""

from __future__ import annotations

import io
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from sensorlab import __version__
from sensorlab.data.loader import _canonical_sensor_name
from sensorlab.decision import CostModel, cost_curve, evaluate_threshold
from sensorlab.server.state import ServerState

log = logging.getLogger("sensorlab.server")
STATIC_DIR = Path(__file__).parent / "static"


def _clean(obj: Any) -> Any:
    """Make numpy / NaN safe for JSON."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    return obj


def create_app(state: ServerState) -> FastAPI:
    app = FastAPI(title="sensorlab governance", version=__version__, docs_url="/api/docs")
    pipe = state.pipeline
    ds = state.ds

    @app.get("/api/health")
    def health():
        drift = state.reference_drift()
        checks = state.acceptance_checks()
        return _clean(
            {
                "status": "ok" if all(c["passed"] for c in checks) else "degraded",
                "sensorlab_version": __version__,
                "model_version": pipe.version_,
                "model_path": state.model_path,
                "fitted_at": pipe.fitted_at_,
                "started_at": state.started_at,
                "primary_detector": pipe.primary_detector_,
                "drift_status": drift.status if drift else "n/a",
                "checks_passed": sum(c["passed"] for c in checks),
                "checks_total": len(checks),
            }
        )

    @app.get("/api/manifest")
    def manifest():
        return _clean(pipe.manifest())

    @app.get("/api/results")
    def results():
        return _clean(state.results) if state.results else JSONResponse(None)

    @app.get("/api/overview")
    def overview():
        ev = state.evaluation
        primary = pipe.primary_detector_
        drift = state.reference_drift()
        n_test = int(state.test_runs.size)
        n_faulty = int(sum(ds.run_fault_id[r] > 0 for r in state.test_runs))
        summaries = state.run_summaries()
        outcomes = pd.Series([r["outcome"] for r in summaries]).value_counts().to_dict()
        return _clean(
            {
                "model": {
                    "version": pipe.version_,
                    "fitted_at": pipe.fitted_at_,
                    "primary_detector": primary,
                    "detectors": list(pipe.detectors_),
                    "n_sensors": len(pipe.sensor_names_),
                    "dataset": pipe.metadata_.get("dataset", {}),
                    "sample_minutes": pipe.sample_minutes_,
                },
                "split": {
                    "train_runs": int(np.unique(ds.run_id[state.masks["train"]]).size),
                    "val_runs": int(np.unique(ds.run_id[state.masks["val"]]).size),
                    "test_runs": n_test,
                    "test_faulty_runs": n_faulty,
                },
                "kpis": {
                    "auroc": ev["detection"][primary]["auroc"],
                    "fraction_detected": ev["detection"][primary]["fraction_detected"],
                    "median_delay_min": ev["detection"][primary]["median_delay_min"],
                    "far_observed": ev["detection"][primary]["far_observed"],
                    "far_target": pipe.config.far_target,
                    "expected_cost": ev["decision"][primary]["expected_cost"],
                    "false_alarms": ev["decision"][primary]["false_alarms"],
                    "missed_faults": ev["decision"][primary]["missed_faults"],
                    "diagnosis_accuracy": ev["diagnosis"]["accuracy"],
                    "diagnosis_macro_f1": ev["diagnosis"]["macro_f1"],
                    "rul_mae_min": ev["rul"]["mae_minutes"],
                    "rul_coverage": ev["rul"]["coverage_80"],
                    "rul_coverage_raw": ev["rul"]["coverage_80_uncalibrated"],
                    "alarm_precision": ev["actions"]["alarm_precision"],
                    "alarm_recall": ev["actions"]["alarm_recall"],
                },
                "outcomes": outcomes,
                "detection": ev["detection"],
                "decision": ev["decision"],
                "thresholds": {
                    "far": pipe.far_thresholds_,
                    "operating": pipe.decision_thresholds_,
                },
                "cost_model": pipe.config.cost.__dict__,
                "acceptance": state.acceptance_checks(),
                "ownership": state.ownership(),
                "drift": drift.to_dict() if drift else None,
            }
        )

    @app.get("/api/runs")
    def runs():
        return _clean(state.run_summaries())

    @app.get("/api/runs/{run_id}")
    def run_detail(
        run_id: int, sensors: str = Query("XMEAS(1),XMEAS(5),XMEAS(10),XMEAS(15),XMV(1)")
    ):
        if run_id not in {int(r) for r in state.test_runs}:
            raise HTTPException(404, f"run {run_id} is not a held-out test run")
        m = ds.run_id == run_id
        pred = state.predictions[state.predictions["run_id"] == run_id].reset_index(drop=True)
        wanted = [s.strip() for s in sensors.split(",") if s.strip()]
        idx = [ds.sensor_names.index(s) for s in wanted if s in ds.sensor_names]
        onset = int(ds.run_onsets[run_id])
        return _clean(
            {
                "run_id": run_id,
                "true_fault_id": int(ds.run_fault_id[run_id]),
                "true_fault": ds.fault_names[int(ds.run_fault_id[run_id])],
                "onset_index": onset,
                "onset_min": None if onset < 0 else onset * ds.sample_minutes,
                "t_minutes": pred["t_minutes"].tolist(),
                "sensors": {ds.sensor_names[i]: ds.X[m, i].round(4).tolist() for i in idx},
                "scores": {n: np.round(s[m], 5).tolist() for n, s in state.scores.items()},
                "far_thresholds": pipe.far_thresholds_,
                "operating_threshold": pred["threshold"].iloc[0],
                "primary_detector": pipe.primary_detector_,
                "alarm": pred["alarm"].tolist(),
                "action": pred["action"].tolist(),
                "fault_pred": pred["fault_name_pred"].tolist(),
                "fault_confidence": pred["fault_confidence"].round(3).tolist(),
                "rul_p10": pred["rul_p10_min"].round(1).tolist(),
                "rul_p50": pred["rul_p50_min"].round(1).tolist(),
                "rul_p90": pred["rul_p90_min"].round(1).tolist(),
                "sensor_names": ds.sensor_names,
            }
        )

    @app.get("/api/decision")
    def decision(
        detector: str | None = None,
        fa: float = Query(None, ge=0),
        mf: float = Query(None, ge=0),
        ld: float = Query(None, ge=0),
        n_grid: int = Query(60, ge=10, le=200),
    ):
        det = detector or pipe.primary_detector_
        if det not in state.scores:
            raise HTTPException(404, f"unknown detector {det}")
        base = pipe.config.cost
        cost = CostModel(
            false_alarm_cost=base.false_alarm_cost if fa is None else fa,
            missed_fault_cost=base.missed_fault_cost if mf is None else mf,
            delay_cost_per_min=base.delay_cost_per_min if ld is None else ld,
        )
        te = state.masks["test"]
        s = state.scores[det]
        grid, res = cost_curve(
            s[te],
            ds.run_id[te],
            ds.run_onsets,
            ds.run_fault_id,
            cost=cost,
            n_grid=n_grid,
            samples_to_minutes=ds.sample_minutes,
            consecutive=pipe.config.consecutive,
        )
        costs = [r.expected_cost for r in res]
        best = int(np.argmin(costs))
        deployed = evaluate_threshold(
            s[te],
            ds.run_id[te],
            ds.run_onsets,
            ds.run_fault_id,
            pipe.decision_thresholds_[det],
            cost,
            samples_to_minutes=ds.sample_minutes,
            consecutive=pipe.config.consecutive,
        )
        return _clean(
            {
                "detector": det,
                "cost_model": cost.__dict__,
                "grid": grid.tolist(),
                "costs": costs,
                "false_alarms": [r.false_alarms for r in res],
                "missed_faults": [r.missed_faults for r in res],
                "oracle": res[best].__dict__,
                "deployed": deployed.__dict__,
            }
        )

    @app.get("/api/diagnosis")
    def diagnosis():
        return _clean(state.diagnosis())

    @app.get("/api/drift")
    def drift(sensor: str | None = None, shift: float = Query(0.0, ge=-5, le=5)):
        m = state.normal_test_mask()
        if not m.any():
            raise HTTPException(409, "no normal runs in the held-out split")
        X = ds.X[m].copy()
        if sensor:
            if sensor not in ds.sensor_names:
                raise HTTPException(404, f"unknown sensor {sensor}")
            j = ds.sensor_names.index(sensor)
            X[:, j] += shift * float(pipe.scaler_.std[j])
        rep = pipe.check_drift(X)
        mon = pipe.drift_
        return _clean(
            {
                **rep.to_dict(),
                "watch_thresholds": mon.watch_thresholds_,
                "noise_floor": mon.noise_floor_,
                "simulated": {"sensor": sensor, "shift_std": shift},
            }
        )

    @app.post("/api/score")
    async def score(
        file: UploadFile = File(...),
        run_col: str = "run_id",
        drift: bool = True,
        format: str = Query("json", pattern="^(json|csv)$"),
        preview_rows: int = Query(50, ge=0, le=1000),
    ):
        raw = await file.read()
        name = (file.filename or "").lower()
        try:
            df = (
                pd.read_parquet(io.BytesIO(raw))
                if name.endswith((".parquet", ".pq"))
                else pd.read_csv(io.BytesIO(raw))
            )
        except Exception as e:
            raise HTTPException(400, f"could not parse upload: {e}") from e
        rename = {
            c: _canonical_sensor_name(c)
            for c in df.columns
            if str(c).lower().startswith(("xmeas_", "xmv_")) and str(c).split("_")[-1].isdigit()
        }
        df = df.rename(columns=rename)
        missing = [s for s in pipe.sensor_names_ if s not in df.columns]
        if missing:
            raise HTTPException(422, {"error": "missing sensor columns", "missing": missing[:10]})
        X = df[pipe.sensor_names_].to_numpy(dtype=np.float32)
        if not np.isfinite(X).all():
            raise HTTPException(422, {"error": "input contains NaN/inf — impute before scoring"})
        run_id = df[run_col].to_numpy() if run_col in df.columns else None
        out = pipe.predict(X, run_id=run_id)
        drift_rep = pipe.check_drift(X).to_dict() if drift else None
        counts = out["action"].value_counts().to_dict()
        entry = {
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "file": file.filename,
            "rows": len(out),
            "runs": int(out["run_id"].nunique()),
            "alarms": int(out["alarm"].sum()),
            "actions": counts,
            "drift_status": drift_rep["status"] if drift_rep else None,
            "drift_alert": drift_rep["alert"] if drift_rep else [],
            "model_fitted_at": pipe.fitted_at_,
        }
        state.audit.append(entry)
        if format == "csv":
            return Response(
                out.to_csv(index=False),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f'attachment; filename="scored_{file.filename or "batch"}.csv"'
                },
            )
        return _clean(
            {
                "summary": entry,
                "drift": drift_rep,
                "preview": out.head(preview_rows).to_dict(orient="records"),
            }
        )

    @app.get("/api/audit")
    def audit():
        return _clean(list(reversed(state.audit)))

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
