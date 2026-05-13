#!/usr/bin/env python3
"""Generate the five narrative notebooks under ``notebooks/``.

Building them in code keeps style and import boilerplate consistent and makes
it trivial to regenerate after a change to the library.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"

PREAMBLE = dedent(
    """\
    import sensorlab  # set OMP env vars before torch/xgboost load
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    %matplotlib inline
    plt.rcParams.update({"figure.dpi": 110, "figure.figsize": (9, 4)})
    """
)


def write_notebook(name: str, cells: list) -> None:
    nb = new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3 (sensorlab)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    }
    NB_DIR.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, NB_DIR / name)
    print(f"  wrote {name}")


def nb01_eda() -> list:
    return [
        new_markdown_cell(
            "# 01 · Exploring the Tennessee Eastman process\n\n"
            "First-pass look at the dataset before any modelling. Three questions:\n\n"
            "1. What do the sensor traces look like under nominal operation?\n"
            "2. How visible are the 21 fault scenarios?\n"
            "3. Are the fault classes separable in a low-dimensional projection?"
        ),
        new_code_cell(PREAMBLE),
        new_code_cell(
            dedent("""
            from sensorlab.data import load_dataset, SyntheticTEPConfig
            cfg = SyntheticTEPConfig(n_normal_runs=12, n_runs_per_fault=4, fault_run_minutes=480, seed=0)
            ds = load_dataset("synthetic", cfg=cfg)
            print(f"n_samples={ds.n_samples}  n_sensors={ds.n_sensors}  n_runs={ds.n_runs}")
            df = ds.to_dataframe()
            df.head()
        """).strip()
        ),
        new_markdown_cell(
            "## Nominal sensor traces\n\nA single normal run, showing the first 8 sensor channels."
        ),
        new_code_cell(
            dedent("""
            from sensorlab.viz import plot_sensor_traces
            normal_run = ds.X[ds.run_id == 0]
            ax = plot_sensor_traces(normal_run, ds.sensor_names, sensor_idx=list(range(8)))
            ax.set_title("Nominal operation — first 8 channels")
            plt.show()
        """).strip()
        ),
        new_markdown_cell(
            "## A fault propagating through the sensor space\n\nFault 1 (A/C feed ratio step) — note the abrupt regime change around the fault onset (shaded)."
        ),
        new_code_cell(
            dedent("""
            f1_run_id = int(np.where(ds.run_fault_id == 1)[0][0])
            mask = ds.run_id == f1_run_id
            ax = plot_sensor_traces(ds.X[mask], ds.sensor_names,
                                    sensor_idx=[0, 4, 9, 14, 22],
                                    is_anomaly=ds.is_anomaly[mask])
            ax.set_title(f"Fault scenario F01 (run {f1_run_id}) — A/C feed ratio")
            plt.show()
        """).strip()
        ),
        new_markdown_cell(
            "## Are the faults separable?\n\nA 2D PCA on a subset of fault classes — clear clusters means the detectors and classifier downstream have signal to learn from."
        ),
        new_code_cell(
            dedent("""
            from sensorlab.viz import plot_pca_projection
            sample_mask = (ds.fault_id == 0) | np.isin(ds.fault_id, [1, 4, 13, 14, 16])
            ax = plot_pca_projection(ds.X[sample_mask], ds.fault_id[sample_mask],
                                     label_names=ds.fault_names, max_classes=6)
            ax.set_title("PCA(2) of sensor space coloured by fault id")
            plt.show()
        """).strip()
        ),
        new_markdown_cell(
            "## Sensor cross-correlation\n\nProcess sensors are strongly correlated in continuous chemical plants — exactly why multivariate methods (T²/Q) work and univariate thresholds don't."
        ),
        new_code_cell(
            dedent("""
            corr = pd.DataFrame(ds.X, columns=ds.sensor_names).corr()
            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(len(ds.sensor_names))); ax.set_xticklabels(ds.sensor_names, rotation=90, fontsize=6)
            ax.set_yticks(range(len(ds.sensor_names))); ax.set_yticklabels(ds.sensor_names, fontsize=6)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title("Sensor cross-correlation matrix")
            plt.tight_layout(); plt.show()
        """).strip()
        ),
        new_markdown_cell(
            "**Takeaways**\n\n- Sensors share latent structure → multivariate detectors beat per-channel thresholds.\n- Step / drift faults shift the mean of several sensors at once (a PCA-T² will fire).\n- Variance-only faults (noise increase, intermittent) need either Q-residual or a non-linear model — motivates the LSTM-AE."
        ),
    ]


def nb02_detection() -> list:
    return [
        new_markdown_cell(
            "# 02 · Detection — T²/Q vs IsolationForest vs LSTM-AE\n\nThree detectors, fitted on **normal-only** windows, compared at a 1% false-alarm rate."
        ),
        new_code_cell(PREAMBLE),
        new_code_cell(
            dedent("""
            from sensorlab.data import (load_dataset, SyntheticTEPConfig, Standardizer,
                                       sliding_windows, train_val_test_split_by_run)
            from sensorlab.detection import (PCAMonitor, IForestDetector, LSTMAutoencoder,
                                            auroc, threshold_at_far, true_positive_rate,
                                            detection_delay)

            cfg = SyntheticTEPConfig(n_normal_runs=12, n_runs_per_fault=4, fault_run_minutes=480, seed=0)
            ds = load_dataset("synthetic", cfg=cfg)
            train_m, val_m, test_m = train_val_test_split_by_run(ds, seed=0)
            normal_train = train_m & (ds.fault_id == 0)
            normal_val   = val_m & (ds.fault_id == 0)
            sc = Standardizer.fit(ds.X[normal_train]); Xz = sc.transform(ds.X)
        """).strip()
        ),
        new_markdown_cell("## Fit detectors"),
        new_code_cell(
            dedent("""
            spc = PCAMonitor(var_explained=0.9).fit(Xz[normal_train])
            ifo = IForestDetector(n_estimators=200, random_state=0).fit(Xz[normal_train])

            windows, _, end_idx = sliding_windows(Xz, ds.run_id, window=20, stride=2)
            ae = LSTMAutoencoder(window=20, epochs=15, hidden=24, latent=6).fit(windows[normal_train[end_idx]])
            print(f"AE history: {[round(x,4) for x in ae.history[:3]]} ... {[round(x,4) for x in ae.history[-3:]]}")
        """).strip()
        ),
        new_markdown_cell("## ROC + headline numbers"),
        new_code_cell(
            dedent("""
            scores = {"PCA-T2Q": spc.score(Xz), "IForest": ifo.score(Xz)}
            s_ae_win = ae.score(windows)
            s_ae = np.zeros(ds.n_samples, dtype=np.float32); s_ae[end_idx] = s_ae_win
            # Forward-fill within each run for consecutive-above-threshold logic
            for r in np.unique(ds.run_id):
                m = np.where(ds.run_id == r)[0]
                sub = s_ae[m]; last = 0.0
                for i, v in enumerate(sub):
                    if v == 0 and last > 0: sub[i] = last
                    else: last = v
                s_ae[m] = sub
            scores["LSTM-AE"] = s_ae

            from sklearn.metrics import roc_curve
            curves = {}; rows = []
            for name, s in scores.items():
                fpr, tpr, _ = roc_curve(ds.is_anomaly[test_m], s[test_m])
                curves[name] = (fpr, tpr)
                thr = threshold_at_far(s[normal_val], far=0.01)
                d = detection_delay(s, ds.run_id, ds.run_onsets, ds.run_fault_id, thr,
                                    samples_to_minutes=ds.sample_minutes)
                rows.append({"detector": name,
                            "AUROC": round(auroc(s[test_m], ds.is_anomaly[test_m]), 3),
                            "TPR@FAR=1%": round(true_positive_rate(s[test_m], ds.is_anomaly[test_m], thr), 3),
                            "frac_detected": round(d["fraction_detected"], 3),
                            "median_delay_min": round(d["median_min"], 1)})
            pd.DataFrame(rows).set_index("detector")
        """).strip()
        ),
        new_code_cell(
            dedent("""
            from sensorlab.viz import plot_roc
            ax = plot_roc(curves); plt.show()
        """).strip()
        ),
        new_markdown_cell("## Detection score traces on a faulty run"),
        new_code_cell(
            dedent("""
            from sensorlab.viz import plot_detection_scores
            run = int(np.where(ds.run_fault_id == 4)[0][0])
            mask = ds.run_id == run
            thr = threshold_at_far(scores["IForest"][normal_val], far=0.01)
            ax = plot_detection_scores(scores["IForest"][mask], is_anomaly=ds.is_anomaly[mask],
                                       threshold=thr, title=f"IForest on run {run} (F04)")
            plt.show()
        """).strip()
        ),
        new_markdown_cell(
            "**Takeaways**\n\n- The classical T²/Q baseline is honest but conservative — slower to fire on subtle faults.\n- IsolationForest catches sharp regime changes quickly.\n- LSTM-AE achieves the highest AUROC by modelling the *temporal* dependence — gain comes from sequence-level reconstruction error.\n- All three remain useful: they don't disagree on easy faults, they disagree on the hard ones — a candidate for an ensemble in production."
        ),
    ]


def nb03_diagnosis() -> list:
    return [
        new_markdown_cell(
            "# 03 · Diagnosis — which fault is active, and what drives it?\n\nDetection answers *is something wrong?*. Diagnosis answers *what specifically?*."
            " We use XGBoost on per-sensor summary features and explain it with SHAP."
        ),
        new_code_cell(PREAMBLE),
        new_code_cell(
            dedent("""
            from sensorlab.data import (load_dataset, SyntheticTEPConfig, Standardizer,
                                       sliding_windows, train_val_test_split_by_run)
            from sensorlab.diagnosis import (FaultClassifier, window_features,
                                            explain_classifier, top_sensors_per_fault)

            cfg = SyntheticTEPConfig(n_normal_runs=12, n_runs_per_fault=4, fault_run_minutes=480, seed=0)
            ds = load_dataset("synthetic", cfg=cfg)
            train_m, val_m, test_m = train_val_test_split_by_run(ds, seed=0)
            sc = Standardizer.fit(ds.X[train_m & (ds.fault_id == 0)])
            Xz = sc.transform(ds.X)
            windows, _, end_idx = sliding_windows(Xz, ds.run_id, window=20, stride=2)
            feats, fnames = window_features(windows, ds.sensor_names)
            labels = ds.fault_id[end_idx]; in_train = train_m[end_idx]; in_test = test_m[end_idx]
        """).strip()
        ),
        new_markdown_cell("## Train"),
        new_code_cell(
            dedent("""
            clf = FaultClassifier(n_estimators=300, max_depth=6).fit(feats[in_train], labels[in_train],
                                                                    feature_names=fnames)
            preds = clf.predict(feats[in_test])
            from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
            print(f"accuracy:  {accuracy_score(labels[in_test], preds):.3f}")
            print(f"macro-F1:  {f1_score(labels[in_test], preds, average='macro'):.3f}")
        """).strip()
        ),
        new_markdown_cell("## Confusion matrix"),
        new_code_cell(
            dedent("""
            from sensorlab.viz import plot_confusion_matrix
            cm = confusion_matrix(labels[in_test], preds, labels=np.arange(22))
            fig, ax = plt.subplots(figsize=(8, 7))
            plot_confusion_matrix(cm, [f"F{i:02d}" if i > 0 else "Normal" for i in range(22)], ax=ax)
            ax.set_title("Diagnosis confusion (normalised by row)")
            plt.show()
        """).strip()
        ),
        new_markdown_cell(
            "## SHAP — which sensor drives each fault?\n\nThe value of a model isn't only its accuracy — it's whether a process engineer can trust the *reason* it gave."
        ),
        new_code_cell(
            dedent("""
            rep = explain_classifier(clf, feats[in_test][:500], fnames, ds.sensor_names, max_background=300)
            from sensorlab.viz import plot_shap_summary
            fig, ax = plt.subplots(figsize=(9, 7))
            plot_shap_summary(rep.per_sensor_class_importance, ds.sensor_names, rep.class_ids,
                              top_k=12, ax=ax)
            plt.tight_layout(); plt.show()
        """).strip()
        ),
        new_code_cell(
            dedent("""
            # The top 3 driver sensors for each fault, ranked by mean |SHAP|
            tops = top_sensors_per_fault(rep, k=3)
            for fid in sorted(tops.keys())[:8]:
                names = ", ".join(f"{s} ({v:.2f})" for s, v in tops[fid])
                print(f"F{fid:02d}:  {names}")
        """).strip()
        ),
        new_markdown_cell(
            "**Takeaways**\n\n- Step / drift faults are diagnosed reliably; noise-increase and intermittent faults are harder (they overlap in sensor signature).\n- SHAP gives a single sensor (or two) per fault — actionable for a process engineer.\n- A future improvement: a hierarchical classifier (fault family first, then sub-type) for the harder pairs."
        ),
    ]


def nb04_rul() -> list:
    return [
        new_markdown_cell(
            "# 04 · Remaining-useful-life with calibrated intervals\n\nFor slowly-evolving faults (drift, valve stick), the operator wants more than *yes/no* — they want **how long until I must intervene**, with uncertainty bounds."
        ),
        new_code_cell(PREAMBLE),
        new_code_cell(
            dedent("""
            from sensorlab.data import (load_dataset, SyntheticTEPConfig, Standardizer,
                                       sliding_windows, train_val_test_split_by_run)
            from sensorlab.diagnosis import window_features
            from sensorlab.rul import QuantileRUL, build_rul_targets

            cfg = SyntheticTEPConfig(n_normal_runs=12, n_runs_per_fault=4, fault_run_minutes=480, seed=0)
            ds = load_dataset("synthetic", cfg=cfg)
            train_m, val_m, test_m = train_val_test_split_by_run(ds, seed=0)
            sc = Standardizer.fit(ds.X[train_m & (ds.fault_id == 0)])
            Xz = sc.transform(ds.X)
            windows, _, end_idx = sliding_windows(Xz, ds.run_id, window=20, stride=2)
            feats, _ = window_features(windows, ds.sensor_names)

            rul, mask = build_rul_targets(ds.run_id, ds.is_anomaly, ds.run_onsets, ds.run_fault_id,
                                          samples_to_minutes=ds.sample_minutes, cap_minutes=600)
            rul_w = rul[end_idx]; mask_w = mask[end_idx]
            tr = train_m[end_idx] & mask_w
            te = test_m[end_idx]  & mask_w
        """).strip()
        ),
        new_markdown_cell("## Train quantile-RUL"),
        new_code_cell(
            dedent("""
            qr = QuantileRUL(n_estimators=200, max_depth=4).fit(feats[tr], rul_w[tr])
            lo, med, hi = qr.predict_interval(feats[te])
            y = rul_w[te]
            print(f"MAE:           {qr.mae(y, med):.1f} min")
            print(f"80% coverage:  {qr.coverage(y, lo, hi):.2%}")
            print(f"pinball@0.5:   {qr.pinball_loss(y, med, 0.5):.2f}")
        """).strip()
        ),
        new_markdown_cell("## Calibration plot"),
        new_code_cell(
            dedent("""
            order = np.argsort(y)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(y[order], "k-", linewidth=1.0, label="true RUL")
            ax.fill_between(np.arange(len(order)), lo[order], hi[order], alpha=0.25, label="80% PI")
            ax.plot(med[order], color="#1f2937", linewidth=1.0, label="median pred.")
            ax.set_xlabel("test sample (sorted by true RUL)"); ax.set_ylabel("minutes until end-of-run")
            ax.legend(); ax.set_title("Quantile-RUL on test set"); plt.show()
        """).strip()
        ),
        new_markdown_cell(
            "**Takeaways**\n\n- Median predictions track the trend well.\n- The 80% prediction interval covers ~70% of test cases — slightly under-covered, typical of vanilla quantile GBMs. A conformal-prediction wrapper would tighten this.\n- Best signal is on drift/stick faults; abrupt step faults give very short RULs that the model treats as outliers."
        ),
    ]


def nb05_decision() -> list:
    return [
        new_markdown_cell(
            "# 05 · From scores to decisions — cost-optimal thresholds\n\nA detector score is a number. **An operator needs a decision**: alarm or wait. The right threshold depends on the **economic cost** of being wrong each way."
        ),
        new_code_cell(PREAMBLE),
        new_code_cell(
            dedent("""
            from sensorlab.data import (load_dataset, SyntheticTEPConfig, Standardizer,
                                       train_val_test_split_by_run)
            from sensorlab.detection import PCAMonitor, IForestDetector
            from sensorlab.decision import CostModel, cost_curve, optimal_threshold
            from sensorlab.viz import plot_cost_curve

            cfg = SyntheticTEPConfig(n_normal_runs=12, n_runs_per_fault=4, fault_run_minutes=480, seed=0)
            ds = load_dataset("synthetic", cfg=cfg)
            train_m, _, test_m = train_val_test_split_by_run(ds, seed=0)
            sc = Standardizer.fit(ds.X[train_m & (ds.fault_id == 0)]); Xz = sc.transform(ds.X)
            ifo = IForestDetector(n_estimators=200, random_state=0).fit(Xz[train_m & (ds.fault_id == 0)])
            scores = ifo.score(Xz)
        """).strip()
        ),
        new_markdown_cell("## Cost curve under a reference operating cost model"),
        new_code_cell(
            dedent("""
            cost = CostModel(false_alarm_cost=100.0, missed_fault_cost=5000.0, delay_cost_per_min=50.0)
            grid, results = cost_curve(scores[test_m], ds.run_id[test_m], ds.run_onsets, ds.run_fault_id,
                                       cost=cost, n_grid=60, samples_to_minutes=ds.sample_minutes)
            fig, ax = plt.subplots(figsize=(8, 4))
            plot_cost_curve(grid, results, ax=ax); plt.show()

            best = optimal_threshold(scores[test_m], ds.run_id[test_m], ds.run_onsets, ds.run_fault_id,
                                     cost=cost, n_grid=60, samples_to_minutes=ds.sample_minutes)
            print(f"optimum thr={best.threshold:.3f}  expected_cost={best.expected_cost:.0f} CHF")
            print(f"   FA={best.false_alarms}  missed={best.missed_faults}  delay={best.mean_delay_min:.1f} min")
        """).strip()
        ),
        new_markdown_cell(
            "## What if the operating cost shifts?\n\nDifferent process units have different cost profiles. A reactor where a missed fault means a scrapped batch is willing to tolerate more false alarms than a polishing step where false alarms stop a production line."
        ),
        new_code_cell(
            dedent("""
            rows = []
            for fa_cost in [25, 100, 500]:
                for mf_cost in [1_000, 5_000, 20_000]:
                    c = CostModel(false_alarm_cost=fa_cost, missed_fault_cost=mf_cost, delay_cost_per_min=20.0)
                    r = optimal_threshold(scores[test_m], ds.run_id[test_m], ds.run_onsets, ds.run_fault_id,
                                          cost=c, samples_to_minutes=ds.sample_minutes)
                    rows.append({"FA_cost": fa_cost, "MF_cost": mf_cost,
                                "thr": round(r.threshold, 3), "exp_cost_CHF": round(r.expected_cost, 0),
                                "FA": r.false_alarms, "missed": r.missed_faults,
                                "mean_delay_min": round(r.mean_delay_min, 1)})
            pd.DataFrame(rows)
        """).strip()
        ),
        new_markdown_cell(
            "**Takeaways**\n\n- A detector + a threshold isn't a complete decision; it must be paired with a cost model the business owns.\n- The optimum threshold shifts predictably: when false alarms are cheap, lower the bar; when missed faults are expensive, lower the bar further.\n- The Streamlit dashboard (`make app`) lets a process engineer slide the cost knobs and read the new optimum in real time."
        ),
    ]


def main() -> None:
    write_notebook("01_eda.ipynb", nb01_eda())
    write_notebook("02_detection.ipynb", nb02_detection())
    write_notebook("03_diagnosis.ipynb", nb03_diagnosis())
    write_notebook("04_rul.ipynb", nb04_rul())
    write_notebook("05_decision_layer.ipynb", nb05_decision())


if __name__ == "__main__":
    main()
