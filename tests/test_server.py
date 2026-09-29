import io

import numpy as np
import pandas as pd
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from sensorlab.server import build_state, create_app  # noqa: E402


@pytest.fixture(scope="module")
def client(fitted_pipeline, tiny_dataset, tiny_splits, tmp_path_factory):
    d = tmp_path_factory.mktemp("srv")
    results = d / "results.json"
    results.write_text('{"aggregate": {"detection": {}}, "seeds": {"0": {}}, "generated_at": "x"}')
    state = build_state(
        model_path=d / "pipe.joblib",
        results_path=results,
        pipeline=fitted_pipeline,
        ds=tiny_dataset,
        masks=tiny_splits,
    )
    return TestClient(create_app(state))


def test_index_and_static(client):
    r = client.get("/")
    assert r.status_code == 200 and "Model governance" in r.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200


def test_health_and_manifest(client):
    h = client.get("/api/health").json()
    assert h["status"] in {"ok", "degraded"}
    assert h["checks_total"] == 5
    m = client.get("/api/manifest").json()
    assert m["n_sensors"] == 33 and "config" in m


def test_overview_structure(client):
    o = client.get("/api/overview").json()
    assert set(o["kpis"]) >= {"auroc", "expected_cost", "diagnosis_accuracy", "rul_coverage"}
    assert len(o["acceptance"]) == 5 and all("passed" in c for c in o["acceptance"])
    assert any(r["knob"].startswith("Missed-fault") for r in o["ownership"])
    assert set(o["thresholds"]["far"]) == set(o["model"]["detectors"])


def test_runs_and_run_detail(client, tiny_dataset, tiny_splits):
    runs = client.get("/api/runs").json()
    test_runs = np.unique(tiny_dataset.run_id[tiny_splits["test"]])
    assert len(runs) == test_runs.size
    assert {r["outcome"] for r in runs} <= {"caught", "missed", "clean", "false alarm"}
    rid = runs[0]["run_id"]
    d = client.get(f"/api/runs/{rid}?sensors=XMEAS(1),XMV(2)").json()
    assert set(d["sensors"]) == {"XMEAS(1)", "XMV(2)"}
    assert len(d["t_minutes"]) == len(d["action"]) == len(d["rul_p50"])
    assert client.get("/api/runs/999999").status_code == 404


def test_decision_curve(client):
    d = client.get("/api/decision?fa=50&mf=2000&ld=10&n_grid=20").json()
    assert len(d["grid"]) == len(d["costs"]) == 20
    assert d["cost_model"]["missed_fault_cost"] == 2000
    assert d["oracle"]["expected_cost"] <= d["deployed"]["expected_cost"] + 1e-6
    assert client.get("/api/decision?detector=nope").status_code == 404


def test_diagnosis_and_drift(client):
    d = client.get("/api/diagnosis").json()
    assert d["top_sensors_per_fault"] and len(d["top_sensors_per_fault"][0]["sensors"]) == 3
    r = client.get("/api/drift").json()
    assert r["status"] in {"stable", "watch", "alert"} and "XMEAS(1)" in r["psi"]
    r2 = client.get("/api/drift?sensor=XMEAS(1)&shift=4").json()
    assert "XMEAS(1)" in r2["alert"]
    assert client.get("/api/drift?sensor=nope").status_code == 404


def test_score_upload_json_csv_and_audit(client, tiny_dataset):
    m = tiny_dataset.run_id == 0
    df = pd.DataFrame(tiny_dataset.X[m], columns=tiny_dataset.sensor_names)
    df["run_id"] = 0
    buf = io.BytesIO(df.to_csv(index=False).encode())
    r = client.post("/api/score", files={"file": ("batch.csv", buf, "text/csv")})
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["rows"] == int(m.sum()) and body["drift"]["status"]
    assert {"alarm", "action"} <= set(body["preview"][0])

    buf = io.BytesIO(df.to_csv(index=False).encode())
    r = client.post("/api/score?format=csv", files={"file": ("batch.csv", buf, "text/csv")})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    assert "action" in r.text.splitlines()[0]

    audit = client.get("/api/audit").json()
    assert len(audit) == 2 and audit[0]["file"] == "batch.csv"

    bad = io.BytesIO(b"XMEAS(1),run_id\n1,0\n")
    assert (
        client.post("/api/score", files={"file": ("bad.csv", bad, "text/csv")}).status_code == 422
    )


def test_export_site_writes_static_bundle(fitted_pipeline, tiny_dataset, tiny_splits, tmp_path):
    import json

    from sensorlab.server.export import export_site

    state = build_state(
        model_path=tmp_path / "p.joblib",
        pipeline=fitted_pipeline,
        ds=tiny_dataset,
        masks=tiny_splits,
    )
    out = export_site(state, tmp_path / "site", base_href="/sensorlab/")
    html = (out / "index.html").read_text()
    assert "window.SENSORLAB_STATIC" in html and '<base href="/sensorlab/" />' in html
    assert (out / ".nojekyll").exists() and (out / "static" / "app.js").exists()
    for name in ("health", "overview", "manifest", "runs", "diagnosis", "decision", "drift"):
        assert (out / "data" / f"{name}.json").exists(), name
    runs = json.loads((out / "data" / "runs.json").read_text())
    assert all((out / "data" / "runs" / f"{r['run_id']}.json").exists() for r in runs)
    dec = json.loads((out / "data" / "decision.json").read_text())
    det = next(iter(dec["detectors"].values()))
    assert {"threshold", "false_alarms", "missed_faults", "delay_sum_min"} <= set(det["grid"][0])
    drift = json.loads((out / "data" / "drift.json").read_text())
    assert drift["shifts"][0] == 0.0 and "XMEAS(1)" in drift["grid"]
    assert drift["grid"]["XMEAS(1)"]["3.0"]["psi"]["XMEAS(1)"] > drift["base"]["psi"]["XMEAS(1)"]
