"""Governance dashboard and scoring API for a saved :class:`MonitoringPipeline`.

``sensorlab serve`` loads the model, rebuilds the evaluation set recorded in
its manifest, and exposes a small JSON API plus a static dashboard::

    GET  /api/health        liveness + model identity + drift status on the reference batch
    GET  /api/overview      KPIs, acceptance checks, ownership table
    GET  /api/manifest      the model card
    GET  /api/results       published multi-seed results (artifacts/results.json)
    GET  /api/runs          per-run outcomes on the held-out test runs
    GET  /api/runs/{id}     traces, scores, alarms, RUL and actions along one run
    GET  /api/decision      cost curve for a detector under a cost mix
    GET  /api/diagnosis     SHAP driver sensors per fault
    GET  /api/drift         PSI per sensor on normal test runs, with an optional simulated offset
    POST /api/score         batch inference on an uploaded CSV/parquet (+ drift), audited
    GET  /api/audit         scoring calls made against this process
"""

from sensorlab.server.app import create_app
from sensorlab.server.state import ServerState, build_state

__all__ = ["ServerState", "build_state", "create_app"]
