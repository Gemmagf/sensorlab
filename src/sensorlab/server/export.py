"""Static export of the governance dashboard for GitHub Pages (or any file host).

``sensorlab export-site`` builds the same :class:`ServerState` the API uses and
writes every response the dashboard needs as JSON under ``<out>/data/``, next
to a copy of the static assets. No server, no cloud: the page is plain files.

What changes in static mode:

* the cost curve is computed **in the browser** from precomputed per-threshold
  counts (false alarms, missed faults, summed delay) — any cost mix, no API;
* the drift simulator reads a precomputed sensor x offset grid;
* upload-and-score and the audit log are replaced by CLI instructions.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from sensorlab import __version__
from sensorlab.decision import CostModel, evaluate_threshold
from sensorlab.server.app import STATIC_DIR, create_app
from sensorlab.server.state import ServerState

log = logging.getLogger("sensorlab.server")

DRIFT_SHIFTS: tuple[float, ...] = tuple(round(0.25 * i, 2) for i in range(13))  # 0 … 3 std


def _call(app, path: str) -> Any:
    """Invoke a GET route in-process (no network) and return its JSON."""
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        r = c.get(path)
        r.raise_for_status()
        return r.json()


def _decision_counts(state: ServerState, n_grid: int = 80) -> dict[str, Any]:
    """Per detector: threshold grid with cost-independent counts, so the browser can price any mix."""
    pipe, ds = state.pipeline, state.ds
    te = state.masks["test"]
    unit = CostModel(false_alarm_cost=1.0, missed_fault_cost=0.0, delay_cost_per_min=0.0)
    out: dict[str, Any] = {"detectors": {}, "cost_model": pipe.config.cost.__dict__}
    for det, s in state.scores.items():
        s_te = s[te]
        lo, hi = float(np.quantile(s_te, 0.01)), float(np.quantile(s_te, 0.999))
        grid = np.linspace(lo, hi, n_grid)
        rows = []
        for thr in [*grid.tolist(), pipe.decision_thresholds_[det]]:
            r = evaluate_threshold(
                s_te,
                ds.run_id[te],
                ds.run_onsets,
                ds.run_fault_id,
                thr,
                unit,
                samples_to_minutes=ds.sample_minutes,
                consecutive=pipe.config.consecutive,
            )
            n_detected = r.n_faulty_runs - r.missed_faults
            rows.append(
                {
                    "threshold": r.threshold,
                    "false_alarms": r.false_alarms,
                    "missed_faults": r.missed_faults,
                    "delay_sum_min": r.mean_delay_min * n_detected,
                    "mean_delay_min": r.mean_delay_min,
                    "n_faulty_runs": r.n_faulty_runs,
                    "n_normal_samples": r.n_normal_samples,
                }
            )
        deployed = rows.pop()
        out["detectors"][det] = {"grid": rows, "deployed": deployed}
    return out


def _drift_grid(state: ServerState) -> dict[str, Any]:
    pipe, ds = state.pipeline, state.ds
    m = state.normal_test_mask()
    base = pipe.check_drift(ds.X[m])
    mon = pipe.drift_
    grid: dict[str, dict[str, Any]] = {}
    for j, name in enumerate(ds.sensor_names):
        per_shift = {}
        for k in DRIFT_SHIFTS:
            X = ds.X[m].copy()
            X[:, j] += k * float(pipe.scaler_.std[j])
            rep = pipe.check_drift(X)
            per_shift[str(k)] = {
                "psi": {name: rep.psi[name]},
                "status": rep.status,
                "alert": rep.alert,
                "watch": rep.watch,
            }
        grid[name] = per_shift
    return {
        "base": base.to_dict(),
        "watch_thresholds": mon.watch_thresholds_,
        "alert_thresholds": mon.alert_thresholds_,
        "noise_floor": mon.noise_floor_,
        "shifts": list(DRIFT_SHIFTS),
        "grid": grid,
    }


def export_site(state: ServerState, out: str | Path, base_href: str | None = None) -> Path:
    out = Path(out)
    data = out / "data"
    if out.exists():
        shutil.rmtree(out)
    data.mkdir(parents=True)
    (data / "runs").mkdir()
    shutil.copytree(STATIC_DIR, out / "static")

    app = create_app(state)
    files: dict[str, Any] = {
        "health.json": _call(app, "/api/health"),
        "overview.json": _call(app, "/api/overview"),
        "manifest.json": _call(app, "/api/manifest"),
        "runs.json": _call(app, "/api/runs"),
        "diagnosis.json": _call(app, "/api/diagnosis"),
        "decision.json": _decision_counts(state),
        "drift.json": _drift_grid(state),
    }
    if state.results:
        files["results.json"] = state.results
    for name, payload in files.items():
        (data / name).write_text(json.dumps(payload, separators=(",", ":"), default=str))
    for rid in state.test_runs:
        detail = _call(app, f"/api/runs/{int(rid)}?sensors=" + ",".join(state.ds.sensor_names))
        (data / "runs" / f"{int(rid)}.json").write_text(
            json.dumps(detail, separators=(",", ":"), default=str)
        )

    html = (STATIC_DIR / "index.html").read_text()
    marker = '<script src="static/app.js"></script>'
    html = html.replace(
        marker,
        f'<script>window.SENSORLAB_STATIC = {{ version: "{__version__}", exported_at: "{state.started_at}" }};</script>\n  {marker}',
    )
    if base_href:
        html = html.replace("<head>", f'<head>\n  <base href="{base_href}" />', 1)
    (out / "index.html").write_text(html)
    (out / ".nojekyll").write_text("")
    log.info("static site written to %s (%d run files)", out, state.test_runs.size)
    return out
