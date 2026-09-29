import json

import numpy as np
import pandas as pd
import pytest

from sensorlab.cli import build_parser, main
from sensorlab.data import SyntheticTEPConfig, load_dataset

FAST = [
    "--fast",
    "--n-normal-runs",
    "4",
    "--n-runs-per-fault",
    "2",
    "--fault-run-minutes",
    "180",
    "--window",
    "12",
]


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("cli")
    model = d / "pipe.joblib"
    results = d / "results.json"
    rc = main(["train", *FAST, "--model-out", str(model), "--results-out", str(results)])
    assert rc == 0
    return {"dir": d, "model": model, "results": results}


def test_train_writes_model_and_results(trained):
    assert trained["model"].exists()
    assert trained["model"].with_suffix(".manifest.json").exists()
    res = json.loads(trained["results"].read_text())
    assert res["sensorlab_version"]
    assert "0" in res["seeds"]
    seed0 = res["seeds"]["0"]
    assert set(seed0["detection"]) == {"PCA-T2Q", "IForest", "LSTM-AE"}
    assert seed0["split"]["n_test_runs"] > 0
    assert res["aggregate"]["detection"]["IForest"]["auroc"]["n"] == 1


def test_score_accepts_rieth_column_names_and_writes_output(trained):
    ds = load_dataset(
        "synthetic",
        cfg=SyntheticTEPConfig(n_normal_runs=1, n_runs_per_fault=0, fault_run_minutes=90, seed=42),
    )
    df = pd.DataFrame(
        ds.X,
        columns=[
            s.replace("XMEAS(", "xmeas_").replace("XMV(", "xmv_").rstrip(")")
            for s in ds.sensor_names
        ],
    )
    df["run_id"] = ds.run_id
    inp = trained["dir"] / "batch.csv"
    out = trained["dir"] / "scored.csv"
    df.to_csv(inp, index=False)
    rc = main(
        [
            "score",
            "--model",
            str(trained["model"]),
            "--input",
            str(inp),
            "--output",
            str(out),
            "--drift",
        ]
    )
    assert rc == 0
    scored = pd.read_csv(out)
    assert len(scored) == len(df)
    assert {"alarm", "action", "fault_name_pred", "rul_p50_min"} <= set(scored.columns)


def test_score_reports_missing_sensors(trained, tmp_path):
    inp = tmp_path / "bad.csv"
    pd.DataFrame({"XMEAS(1)": np.zeros(5)}).to_csv(inp, index=False)
    assert main(["score", "--model", str(trained["model"]), "--input", str(inp)]) == 2


def test_info_prints_manifest(trained, capsys):
    assert main(["info", "--model", str(trained["model"])]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["config"]["window"] == 12


def test_evaluate_multi_seed_aggregates(tmp_path):
    results = tmp_path / "r.json"
    rc = main(["evaluate", *FAST, "--seeds", "0", "1", "--results-out", str(results)])
    assert rc == 0
    res = json.loads(results.read_text())
    assert set(res["seeds"]) == {"0", "1"}
    assert res["aggregate"]["diagnosis"]["accuracy"]["n"] == 2


def test_parser_rejects_unknown_detector():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train", "--detectors", "nope"])
