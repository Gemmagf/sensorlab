"""Command-line interface: ``sensorlab train | evaluate | score | info``.

* ``train``     fit one pipeline, evaluate it on held-out runs, save model + results
* ``evaluate``  the same experiment over several seeds, reporting mean ± std
* ``score``     run a saved pipeline on a CSV/parquet of sensor rows (batch inference)
* ``info``      print the manifest of a saved pipeline

Every command is deterministic given its arguments and writes machine-readable
JSON, so it can sit in a scheduled job or a CI step without wrapping.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sensorlab  # noqa: F401 — platform shims first
from sensorlab import __version__
from sensorlab.config import ARTIFACTS_DIR, MODELS_DIR, PROJECT_ROOT
from sensorlab.data import SyntheticTEPConfig, load_dataset, train_val_test_split_by_run
from sensorlab.data.loader import TEPDataset, _canonical_sensor_name
from sensorlab.decision import CostModel
from sensorlab.evaluation import aggregate_seeds
from sensorlab.pipeline import ALL_DETECTORS, MonitoringPipeline, PipelineConfig

log = logging.getLogger("sensorlab.cli")


# ----------------------------------------------------------------------------- helpers


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _add_data_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("data")
    g.add_argument("--data", choices=["synthetic", "real"], default="synthetic")
    g.add_argument("--real-root", type=Path, default=None, help="parquet dir for --data real")
    g.add_argument("--n-runs-per-fault", type=int, default=4)
    g.add_argument("--n-normal-runs", type=int, default=12)
    g.add_argument("--fault-run-minutes", type=int, default=480)


def _add_model_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("model")
    g.add_argument("--window", type=int, default=20)
    g.add_argument("--stride", type=int, default=2)
    g.add_argument("--ae-epochs", type=int, default=25)
    g.add_argument("--far-target", type=float, default=0.01)
    g.add_argument("--consecutive", type=int, default=3)
    g.add_argument(
        "--detectors", nargs="+", default=list(ALL_DETECTORS), choices=list(ALL_DETECTORS)
    )
    g.add_argument("--primary-detector", default="auto", choices=["auto", *ALL_DETECTORS])
    g.add_argument("--false-alarm-cost", type=float, default=100.0)
    g.add_argument("--missed-fault-cost", type=float, default=5000.0)
    g.add_argument("--delay-cost-per-min", type=float, default=50.0)
    g.add_argument("--intervene-horizon-min", type=float, default=120.0)
    g.add_argument("--fast", action="store_true", help="small models for smoke tests")


def _load_data(args: argparse.Namespace, seed: int) -> TEPDataset:
    if args.data == "synthetic":
        cfg = SyntheticTEPConfig(
            n_runs_per_fault=args.n_runs_per_fault,
            n_normal_runs=args.n_normal_runs,
            fault_run_minutes=args.fault_run_minutes,
            seed=seed,
        )
        return load_dataset("synthetic", cfg=cfg)
    return load_dataset("real", real_root=args.real_root)


def _pipeline_config(args: argparse.Namespace, seed: int) -> PipelineConfig:
    cfg = PipelineConfig(
        window=args.window,
        stride=args.stride,
        far_target=args.far_target,
        consecutive=args.consecutive,
        detectors=tuple(args.detectors),
        primary_detector=args.primary_detector,
        ae_epochs=args.ae_epochs,
        intervene_horizon_min=args.intervene_horizon_min,
        cost=CostModel(
            false_alarm_cost=args.false_alarm_cost,
            missed_fault_cost=args.missed_fault_cost,
            delay_cost_per_min=args.delay_cost_per_min,
        ),
        seed=seed,
    )
    if args.fast:
        cfg.ae_epochs = min(cfg.ae_epochs, 3)
        cfg.iforest_estimators = 50
        cfg.xgb_estimators = 40
        cfg.rul_estimators = 40
    return cfg


def run_experiment(
    args: argparse.Namespace, seed: int
) -> tuple[MonitoringPipeline, dict[str, Any]]:
    """Load → split by run → fit on train/val → evaluate on test. One seed."""
    t0 = time.time()
    ds = _load_data(args, seed)
    train_m, val_m, test_m = train_val_test_split_by_run(ds, seed=seed)
    log.info(
        "seed %d: %d samples, %d runs, %d sensors | runs train/val/test = %d/%d/%d",
        seed,
        ds.n_samples,
        ds.n_runs,
        ds.n_sensors,
        np.unique(ds.run_id[train_m]).size,
        np.unique(ds.run_id[val_m]).size,
        np.unique(ds.run_id[test_m]).size,
    )
    pipe = MonitoringPipeline(_pipeline_config(args, seed)).fit(ds, train_m, val_m)
    results = pipe.evaluate(ds, test_m, shap_samples=0 if args.fast else 300)
    results["seed"] = seed
    results["split"] = {
        "n_train_runs": int(np.unique(ds.run_id[train_m]).size),
        "n_val_runs": int(np.unique(ds.run_id[val_m]).size),
        "n_test_runs": int(np.unique(ds.run_id[test_m]).size),
    }
    results["dataset"] = {
        "source": args.data,
        "n_samples": ds.n_samples,
        "n_runs": ds.n_runs,
        "n_sensors": ds.n_sensors,
        "fault_types": int(ds.run_fault_id.max()),
    }
    results["wall_seconds"] = time.time() - t0
    return pipe, results


def _summary_lines(res: dict[str, Any]) -> list[str]:
    lines = [f"=== seed {res['seed']} — test runs: {res['n_eval_runs']} ==="]
    for name, r in res["detection"].items():
        lines.append(
            f"  {name:8s} AUROC={r['auroc']:.3f}  TPR@FAR={r['tpr_at_far']:.3f}  "
            f"FAR_obs={r['far_observed']:.3f}  detected={r['fraction_detected']:.0%}  "
            f"median_delay={r['median_delay_min']:.0f}min"
        )
    for name, r in res["decision"].items():
        lines.append(
            f"  {name:8s} cost={r['expected_cost']:>10,.0f}  FA={r['false_alarms']:<4d} "
            f"missed={r['missed_faults']}/{r['n_faulty_runs']}  delay={r['mean_delay_min']:.1f}min"
        )
    d, u = res["diagnosis"], res["rul"]
    lines.append(f"  diagnosis acc={d['accuracy']:.3f} macro-F1={d['macro_f1']:.3f}")
    lines.append(f"  RUL MAE={u['mae_minutes']:.1f}min  80%-coverage={u['coverage_80']:.0%}")
    lines.append(f"  primary detector: {res['primary_detector']}")
    return lines


def _write_results(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    log.info("results written to %s", _relative(path))


def _base_payload(args: argparse.Namespace) -> dict[str, Any]:
    config = {k: v for k, v in vars(args).items() if k not in {"func", "command"}}
    for k, v in list(config.items()):
        if isinstance(v, Path):
            config[k] = _relative(v)
    return {
        "sensorlab_version": __version__,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": args.command,
        "config": config,
    }


# ---------------------------------------------------------------------------- commands


def cmd_train(args: argparse.Namespace) -> int:
    pipe, res = run_experiment(args, args.seed)
    payload = _base_payload(args) | {
        "seeds": {str(args.seed): res},
        "aggregate": aggregate_seeds([res]),
    }
    _write_results(args.results_out, payload)
    if not args.no_save:
        pipe.save(args.model_out)
    print("\n".join(_summary_lines(res)))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    per_seed = []
    for seed in args.seeds:
        _, res = run_experiment(args, seed)
        per_seed.append(res)
        print("\n".join(_summary_lines(res)))
    payload = _base_payload(args) | {
        "seeds": {str(r["seed"]): r for r in per_seed},
        "aggregate": aggregate_seeds(per_seed),
    }
    _write_results(args.results_out, payload)
    agg = payload["aggregate"]
    print("\n=== aggregate over seeds", list(args.seeds), "===")
    for name, r in agg["detection"].items():
        print(
            f"  {name:8s} AUROC={r['auroc']['mean']:.3f}±{r['auroc']['std']:.3f}  "
            f"detected={r['fraction_detected']['mean']:.0%}  "
            f"median_delay={r['median_delay_min']['mean']:.0f}±{r['median_delay_min']['std']:.0f}min"
        )
    return 0


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def cmd_score(args: argparse.Namespace) -> int:
    pipe = MonitoringPipeline.load(args.model)
    df = _read_table(args.input)
    # Accept Rieth-style names (xmeas_1) as well as canonical XMEAS(1)
    rename = {
        c: _canonical_sensor_name(c)
        for c in df.columns
        if c.lower().startswith(("xmeas_", "xmv_")) and c.split("_")[-1].isdigit()
    }
    df = df.rename(columns=rename)
    missing = [s for s in pipe.sensor_names_ if s not in df.columns]
    if missing:
        log.error("input is missing %d sensor columns, e.g. %s", len(missing), missing[:5])
        return 2
    run_id = df[args.run_col].to_numpy() if args.run_col in df.columns else None
    X = df[pipe.sensor_names_].to_numpy(dtype=np.float32)
    out = pipe.predict(X, run_id=run_id)
    if args.drift:
        rep = pipe.check_drift(X)
        log.info("drift status: %s (alert=%s watch=%s)", rep.status, rep.alert, rep.watch)
        out.attrs["drift"] = rep.to_dict()
    if args.output is None:
        print(out.to_string(max_rows=50))
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix.lower() in {".parquet", ".pq"}:
            out.to_parquet(args.output, index=False)
        else:
            out.to_csv(args.output, index=False)
        log.info("scored %d rows -> %s", len(out), args.output)
    counts = out["action"].value_counts().to_dict()
    print("actions:", json.dumps(counts))
    if args.drift:
        print("drift:", json.dumps(out.attrs["drift"] | {"psi": "…"}))
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    pipe = MonitoringPipeline.load(args.model)
    print(json.dumps(pipe.manifest(), indent=2, default=str))
    return 0


# ------------------------------------------------------------------------------ parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sensorlab", description=__doc__)
    p.add_argument("--version", action="version", version=f"sensorlab {__version__}")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="fit, evaluate on held-out runs, save model + results")
    _add_data_args(t)
    _add_model_args(t)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--model-out", type=Path, default=MODELS_DIR / "pipeline.joblib")
    t.add_argument("--results-out", type=Path, default=ARTIFACTS_DIR / "results.json")
    t.add_argument("--no-save", action="store_true")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate", help="multi-seed experiment with mean ± std")
    _add_data_args(e)
    _add_model_args(e)
    e.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    e.add_argument("--results-out", type=Path, default=ARTIFACTS_DIR / "results.json")
    e.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("score", help="batch inference with a saved pipeline")
    s.add_argument("--model", type=Path, default=MODELS_DIR / "pipeline.joblib")
    s.add_argument("--input", type=Path, required=True, help="CSV or parquet of sensor rows")
    s.add_argument("--output", type=Path, default=None, help="CSV/parquet; stdout if omitted")
    s.add_argument("--run-col", default="run_id")
    s.add_argument("--drift", action="store_true", help="also report per-sensor PSI drift")
    s.set_defaults(func=cmd_score)

    i = sub.add_parser("info", help="print the manifest of a saved pipeline")
    i.add_argument("--model", type=Path, default=MODELS_DIR / "pipeline.joblib")
    i.set_defaults(func=cmd_info)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
