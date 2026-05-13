#!/usr/bin/env python3
"""End-to-end training pipeline.

Trains every detector, the fault classifier, the RUL head, and computes the
cost-optimal decision threshold. Writes a summary JSON to ``artifacts/`` that
the README and the Streamlit dashboard read.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# Ensure libomp env vars take effect first
import sensorlab  # noqa: F401
from sensorlab.config import ARTIFACTS_DIR, ensure_dirs
from sensorlab.data import (
    Standardizer,
    SyntheticTEPConfig,
    load_dataset,
    sliding_windows,
    train_val_test_split_by_run,
)
from sensorlab.decision import CostModel, optimal_threshold
from sensorlab.detection import (
    IForestDetector,
    LSTMAutoencoder,
    PCAMonitor,
    auroc,
    detection_delay,
    threshold_at_far,
    true_positive_rate,
)
from sensorlab.diagnosis import (
    FaultClassifier,
    explain_classifier,
    top_sensors_per_fault,
    window_features,
)
from sensorlab.rul import QuantileRUL, build_rul_targets


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", choices=["synthetic", "real"], default="synthetic")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-runs-per-fault", type=int, default=4)
    p.add_argument("--n-normal-runs", type=int, default=12)
    p.add_argument("--fault-run-minutes", type=int, default=480)
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--ae-epochs", type=int, default=25)
    p.add_argument("--far-target", type=float, default=0.01)
    p.add_argument("--out", type=Path, default=ARTIFACTS_DIR / "results.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    print(f"[1/7] Loading {args.data} dataset…")
    if args.data == "synthetic":
        cfg = SyntheticTEPConfig(
            n_runs_per_fault=args.n_runs_per_fault,
            n_normal_runs=args.n_normal_runs,
            fault_run_minutes=args.fault_run_minutes,
            seed=args.seed,
        )
        ds = load_dataset("synthetic", cfg=cfg)
    else:
        ds = load_dataset("real")
    print(f"     n_samples={ds.n_samples}  n_runs={ds.n_runs}  n_sensors={ds.n_sensors}")

    print("[2/7] Splitting by run and standardising…")
    train_m, val_m, test_m = train_val_test_split_by_run(ds, seed=args.seed)
    normal_train = train_m & (ds.fault_id == 0)
    sc = Standardizer.fit(ds.X[normal_train])
    Xz = sc.transform(ds.X)

    norm_val = val_m & (ds.fault_id == 0)

    print("[3/7] Detection — PCA T²/Q, Isolation Forest, LSTM-AE…")
    detection_results: dict[str, dict] = {}
    detector_scores: dict[str, np.ndarray] = {}

    # PCA T²/Q
    t0 = time.time()
    spc = PCAMonitor(var_explained=0.9).fit(Xz[normal_train])
    s_spc = spc.score(Xz)
    detection_results["PCA-T2Q"] = _eval_detector(
        s_spc,
        ds,
        train_m,
        val_m,
        test_m,
        norm_val,
        far_target=args.far_target,
    ) | {"fit_seconds": time.time() - t0, "n_components": spc.n_components_}
    detector_scores["PCA-T2Q"] = s_spc

    # IForest
    t0 = time.time()
    ifo = IForestDetector(n_estimators=300, random_state=args.seed).fit(Xz[normal_train])
    s_ifo = ifo.score(Xz)
    detection_results["IForest"] = _eval_detector(
        s_ifo, ds, train_m, val_m, test_m, norm_val, far_target=args.far_target
    ) | {"fit_seconds": time.time() - t0}
    detector_scores["IForest"] = s_ifo

    # LSTM-AE
    t0 = time.time()
    windows, _, end_idx = sliding_windows(Xz, ds.run_id, window=args.window, stride=args.stride)
    ae = LSTMAutoencoder(
        window=args.window,
        epochs=args.ae_epochs,
        hidden=32,
        latent=8,
        batch_size=64,
        seed=args.seed,
    )
    in_normal_train = normal_train[end_idx]
    ae.fit(windows[in_normal_train])
    s_ae_win = ae.score(windows)
    # Lift window scores back to a per-sample array, forward-filling within each
    # run so detection_delay's consecutive-above-threshold logic still works.
    s_ae_per_sample = _windows_to_per_sample(s_ae_win, end_idx, ds.run_id, ds.n_samples)
    detection_results["LSTM-AE"] = _eval_detector(
        s_ae_per_sample,
        ds,
        train_m,
        val_m,
        test_m,
        norm_val,
        far_target=args.far_target,
        label_mask=np.isin(np.arange(ds.n_samples), end_idx),
    ) | {"fit_seconds": time.time() - t0, "final_loss": ae.history[-1]}
    detector_scores["LSTM-AE"] = s_ae_per_sample

    print("[4/7] Diagnosis — XGBoost multi-class + SHAP root cause…")
    t0 = time.time()
    feats, fnames = window_features(windows, ds.sensor_names)
    labels = ds.fault_id[end_idx]
    train_w = train_m[end_idx]
    test_w = test_m[end_idx]
    clf = FaultClassifier(n_estimators=300, max_depth=6, random_state=args.seed).fit(
        feats[train_w], labels[train_w], feature_names=fnames
    )
    preds = clf.predict(feats[test_w])
    acc = float((preds == labels[test_w]).mean())
    from sklearn.metrics import f1_score

    macro_f1 = float(f1_score(labels[test_w], preds, average="macro"))
    diagnosis_results = {"accuracy": acc, "macro_f1": macro_f1, "fit_seconds": time.time() - t0}

    # SHAP root-cause: aggregate per-fault top-3 sensors
    rep = explain_classifier(clf, feats[test_w][:500], fnames, ds.sensor_names, max_background=300)
    diagnosis_results["top_sensors_per_fault"] = {
        int(fid): [(name, round(v, 3)) for name, v in tops]
        for fid, tops in top_sensors_per_fault(rep, k=3).items()
    }

    print("[5/7] RUL — quantile regression…")
    t0 = time.time()
    rul, rul_mask = build_rul_targets(
        ds.run_id,
        ds.is_anomaly,
        ds.run_onsets,
        ds.run_fault_id,
        samples_to_minutes=ds.sample_minutes,
        cap_minutes=600,
    )
    rul_w = rul[end_idx]
    rul_mask_w = rul_mask[end_idx]
    qr_train = train_w & rul_mask_w
    qr_test = test_w & rul_mask_w
    qr = QuantileRUL(n_estimators=200, max_depth=4, random_state=args.seed).fit(
        feats[qr_train], rul_w[qr_train]
    )
    lo, med, hi = qr.predict_interval(feats[qr_test])
    y_true = rul_w[qr_test]
    rul_results = {
        "mae_minutes": qr.mae(y_true, med),
        "coverage_80": qr.coverage(y_true, lo, hi),
        "pinball_50": qr.pinball_loss(y_true, med, 0.5),
        "fit_seconds": time.time() - t0,
        "n_test_samples": int(qr_test.sum()),
    }

    print("[6/7] Decision layer — cost-optimal threshold for each detector…")
    cost = CostModel(false_alarm_cost=100.0, missed_fault_cost=5000.0, delay_cost_per_min=50.0)
    decision_results: dict[str, dict] = {}
    for name, scores in detector_scores.items():
        # restrict to samples that have a score (LSTM-AE has zeros outside end_idx)
        valid = np.ones(ds.n_samples, dtype=bool)
        if name == "LSTM-AE":
            valid = np.isin(np.arange(ds.n_samples), end_idx)
        res = optimal_threshold(
            scores[valid],
            ds.run_id[valid],
            ds.run_onsets,
            ds.run_fault_id,
            cost=cost,
            samples_to_minutes=ds.sample_minutes,
            n_grid=80,
        )
        decision_results[name] = {
            "threshold": res.threshold,
            "expected_cost_chf": res.expected_cost,
            "false_alarms": res.false_alarms,
            "missed_faults": res.missed_faults,
            "mean_delay_min": res.mean_delay_min,
        }

    print("[7/7] Writing summary…")
    summary = {
        "config": vars(args),
        "dataset": {
            "source": args.data,
            "n_samples": ds.n_samples,
            "n_runs": ds.n_runs,
            "n_sensors": ds.n_sensors,
            "fault_types": int(ds.run_fault_id.max()),
        },
        "detection": detection_results,
        "diagnosis": diagnosis_results,
        "rul": rul_results,
        "decision": decision_results,
        "cost_model": vars(cost),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nResults written to {args.out}")
    print("\n=== Headline numbers ===")
    for name, res in detection_results.items():
        print(
            f"  {name:10s}  AUROC={res['auroc_test']:.3f}  "
            f"TPR@FAR={args.far_target:.0%}={res['tpr_at_far']:.3f}  "
            f"frac_detected={res['fraction_detected']:.0%}  "
            f"median_delay={res['median_delay_min']:.0f}min"
        )
    print(
        f"  Diagnosis  macro-F1={diagnosis_results['macro_f1']:.3f}  "
        f"acc={diagnosis_results['accuracy']:.3f}"
    )
    print(
        f"  RUL        MAE={rul_results['mae_minutes']:.1f}min  "
        f"80%-coverage={rul_results['coverage_80']:.0%}"
    )


def _windows_to_per_sample(
    win_scores: np.ndarray, end_idx: np.ndarray, run_id: np.ndarray, n_samples: int
) -> np.ndarray:
    """Map per-window scores to a per-sample array, forward-filling within each run.

    Samples before the first window-end of a run inherit that run's first
    window-end score; samples between window-ends inherit the most recent one.
    This keeps ``detection_delay``'s consecutive-above-threshold logic well-defined.
    """
    out = np.full(n_samples, np.nan, dtype=np.float32)
    out[end_idx] = win_scores
    for r in np.unique(run_id):
        mask = np.where(run_id == r)[0]
        sub = out[mask]
        # Forward-fill within the run, then back-fill the leading NaNs
        last = np.nan
        for i in range(len(sub)):
            if np.isnan(sub[i]):
                sub[i] = last
            else:
                last = sub[i]
        # Back-fill leading NaNs (positions before first valid window end)
        first_valid = np.argmax(~np.isnan(sub))
        if np.isnan(sub[0]) and not np.isnan(sub[first_valid]):
            sub[:first_valid] = sub[first_valid]
        out[mask] = sub
    # Any remaining NaN means a run had no valid window — set to 0 as neutral baseline
    out[np.isnan(out)] = 0.0
    return out


def _eval_detector(
    scores: np.ndarray,
    ds,
    train_m: np.ndarray,
    val_m: np.ndarray,
    test_m: np.ndarray,
    norm_val: np.ndarray,
    far_target: float,
    label_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute AUROC, TPR at the chosen FAR, and detection delay on test runs."""
    if label_mask is None:
        label_mask = np.ones(len(scores), dtype=bool)
    thr = threshold_at_far(scores[norm_val & label_mask], far=far_target)
    test_mask = test_m & label_mask
    test_y = ds.is_anomaly[test_mask]
    test_s = scores[test_mask]
    a = auroc(test_s, test_y)
    tpr = true_positive_rate(test_s, test_y, thr)
    delay = detection_delay(
        scores,
        ds.run_id,
        ds.run_onsets,
        ds.run_fault_id,
        thr,
        samples_to_minutes=ds.sample_minutes,
    )
    return {
        "auroc_test": a,
        "threshold": thr,
        "tpr_at_far": tpr,
        "fraction_detected": delay["fraction_detected"],
        "median_delay_min": delay["median_min"],
        "mean_delay_min": delay["mean_min"],
        "p90_delay_min": delay["p90_min"],
    }


if __name__ == "__main__":
    main()
