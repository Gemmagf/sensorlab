"""Interactive dashboard for the sensorlab pipeline.

Run with::

    make app          # or
    streamlit run app/streamlit_app.py

Everything shown here is produced by the same :class:`MonitoringPipeline`
that ``sensorlab train`` serialises and ``sensorlab score`` runs in batch —
the dashboard is a window onto the deployable artefact, not a parallel
implementation.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import sensorlab  # noqa: F401 — platform shims first
from sensorlab.config import ARTIFACTS_DIR
from sensorlab.data import (
    SyntheticTEPConfig,
    load_dataset,
    sliding_windows,
    train_val_test_split_by_run,
)
from sensorlab.decision import CostModel, cost_curve, optimal_threshold
from sensorlab.diagnosis import explain_classifier, window_features
from sensorlab.pipeline import ALL_DETECTORS, MonitoringPipeline, PipelineConfig
from sensorlab.viz import (
    plot_cost_curve,
    plot_detection_scores,
    plot_pca_projection,
    plot_sensor_traces,
    plot_shap_summary,
)

st.set_page_config(
    page_title="sensorlab — TEP fault detection",
    layout="wide",
    page_icon="🏭",
)

# ---------------------------------------------------------------------------
# Cached resources — every function is keyed on *all* the parameters that
# change its output, so moving any sidebar control rebuilds exactly what it
# should and nothing else.
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Generating data, fitting the pipeline (one-time)…")
def build_lab(
    seed: int,
    n_normal: int,
    n_per_fault: int,
    run_minutes: int,
    window: int,
    stride: int,
    ae_epochs: int,
):
    ds = load_dataset(
        "synthetic",
        cfg=SyntheticTEPConfig(
            n_normal_runs=n_normal,
            n_runs_per_fault=n_per_fault,
            fault_run_minutes=run_minutes,
            seed=seed,
        ),
    )
    train, val, test = train_val_test_split_by_run(ds, seed=seed)
    cfg = PipelineConfig(
        window=window,
        stride=stride,
        ae_epochs=ae_epochs,
        ae_hidden=24,
        ae_latent=6,
        iforest_estimators=150,
        xgb_estimators=120,
        xgb_max_depth=5,
        rul_estimators=60,
        seed=seed,
    )
    pipe = MonitoringPipeline(cfg).fit(ds, train, val)
    scores = pipe.score_detectors(ds.X, ds.run_id)
    evaluation = pipe.evaluate(ds, test, shap_samples=0)
    predictions = pipe.predict(ds.X[test], ds.run_id[test])
    return ds, {"train": train, "val": val, "test": test}, pipe, scores, evaluation, predictions


@st.cache_resource(show_spinner="Computing SHAP attributions…")
def compute_shap(
    lab_key: tuple, _pipe: MonitoringPipeline, _ds, _test_mask, sample_size: int = 400
):
    Xz = _pipe.scaler_.transform(_ds.X)
    windows, _, end_idx = sliding_windows(
        Xz, _ds.run_id, window=_pipe.config.window, stride=_pipe.config.stride
    )
    feats, fnames = window_features(windows, _pipe.sensor_names_)
    in_test = np.where(_test_mask[end_idx])[0]
    idx = np.random.default_rng(0).choice(in_test, min(sample_size, in_test.size), replace=False)
    return explain_classifier(
        _pipe.classifier_, feats[idx], fnames, _pipe.sensor_names_, max_background=sample_size
    )


@st.cache_data
def load_published_results() -> dict | None:
    path = Path(ARTIFACTS_DIR) / "results.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _fmt_delay(x: float) -> str:
    return "—" if x is None or np.isnan(x) else f"{x:.0f}"


# ---------------------------------------------------------------------------
# Sidebar — global configuration
# ---------------------------------------------------------------------------
st.sidebar.title("🏭 sensorlab")
st.sidebar.caption("Tennessee Eastman fault detection, diagnosis & decision lab")

with st.sidebar.expander("Dataset", expanded=False):
    seed = st.number_input("seed", 0, 9999, 0, 1)
    n_normal = st.slider("normal runs", 4, 20, 12)
    n_per_fault = st.slider("runs per fault", 2, 8, 4)
    run_minutes = st.slider("minutes per run", 120, 720, 480, 60)

with st.sidebar.expander("Models", expanded=False):
    window = st.slider("window size", 10, 40, 20, 2)
    stride = st.slider("stride", 1, 5, 2)
    ae_epochs = st.slider("LSTM-AE epochs", 5, 50, 15, 5)

lab_key = (
    int(seed),
    int(n_normal),
    int(n_per_fault),
    int(run_minutes),
    int(window),
    int(stride),
    int(ae_epochs),
)
ds, masks, pipe, scores, evaluation, predictions = build_lab(*lab_key)
train, val, test = masks["train"], masks["val"], masks["test"]
test_runs = np.unique(ds.run_id[test])

st.sidebar.markdown("---")
st.sidebar.metric("Samples", f"{ds.n_samples:,}")
st.sidebar.metric(
    "Runs (train / val / test)",
    f"{np.unique(ds.run_id[train]).size} / {np.unique(ds.run_id[val]).size} / {test_runs.size}",
)
st.sidebar.metric("Primary detector", pipe.primary_detector_)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
st.title("Industrial sensor anomaly lab")
st.markdown(
    "Synthetic Tennessee-Eastman telemetry → three detectors → SHAP diagnosis → "
    "RUL interval → cost-aware operating decision. Everything below is computed on "
    "**held-out test runs** by the same pipeline object that ships to the plant."
)

tab_approach, tab_overview, tab_detect, tab_diag, tab_decide, tab_operate = st.tabs(
    [
        "🧭 Approach",
        "📈 Live run",
        "🚨 Detector comparison",
        "🔍 Diagnosis (SHAP)",
        "💰 Decision layer",
        "🛠 Operate",
    ]
)

# ----- Tab 0: approach -------------------------------------------------------
with tab_approach:
    st.subheader("From a plant question to a deployable decision")
    st.markdown(
        """
A continuous chemical plant emits tens of correlated sensor streams. The team
running it has three operational questions, each a different ML problem:

1. **Detection** — *is something wrong, right now?* (unsupervised, normal-only training)
2. **Diagnosis** — *which fault, and which sensor is driving it?* (supervised + SHAP)
3. **Remaining useful life** — *how long until I must intervene?* (quantile regression)

A **decision layer** then turns model outputs into one of four actions —
`wait`, `investigate`, `schedule_maintenance`, `intervene_now` — using a cost
model that the business owns. That chain is one serialisable object
(`MonitoringPipeline`) with a CLI (`sensorlab train | evaluate | score`) and a
drift monitor, so hand-over to a plant team is a file and a command, not a
notebook.
"""
    )

    st.markdown("### Modelling decisions")
    st.markdown(
        """
| Decision | Rationale |
|---|---|
| Detectors trained on **normal data only** | Faults are rare and heterogeneous; supervised detection overfits to the faults seen in training and misses novel ones. |
| Three complementary detectors | **T²/Q** is the process-monitoring gold standard; **IsolationForest** a robust tabular default; **LSTM-AE** reads the temporal signature. They disagree on the hard faults. |
| **Split by run**, never by sample | Windows from one run in both train and test is leakage. Whole runs are held out. |
| Scaler fitted on **training-normal only** | Otherwise it absorbs fault variance and erases the signal. |
| Thresholds from **validation** only | FAR threshold on validation-normal, cost-optimal operating point on validation runs. The test split is untouched until reporting. |
| Diagnosis target = fault **active at time t** | Pre-onset samples of a faulty run are nominal; labelling them with the fault id asks the model to separate identical distributions. Fixing this alone moved accuracy from 0.57 to ~0.70. |
| Window-level features (mean, std, last, slope per sensor) | Tabular XGBoost beats an end-to-end CNN at this data size and yields sensor-level SHAP attributions. |
| Quantile GBM for RUL | A point estimate without uncertainty is dangerous; three quantiles give an 80 % interval whose *coverage* is reported next to its MAE. |
| Cost-aware threshold | "Best threshold" has no answer without a cost model. The Decision tab makes the trade-off explicit and negotiable. |
| Every window-level output aligned through one helper | LSTM-AE, classifier and RUL predictions are lifted to the sample axis in one place, so metrics, decisions and the dashboard agree on onsets and time units. |
"""
    )

    st.markdown("### Evaluation discipline")
    st.markdown(
        """
- **Detection**: AUROC (threshold-free), TPR and observed FAR at the calibrated
  threshold, fraction of faulty runs caught and median detection delay in minutes.
- **Diagnosis**: accuracy, balanced accuracy, macro-F1 on 22 classes.
- **RUL**: MAE of the median plus **80 % interval coverage** and mean width.
- **Decision**: expected cost, false alarms, missed faults on test runs at the
  threshold chosen on validation.
- **Actions**: precision/recall of the confirmed alarm on test samples.
- Published numbers are **mean ± std over three seeds**, stored in
  `artifacts/results.json` and rendered below — the README reads the same file.
"""
    )

    published = load_published_results()
    if published and "aggregate" in published:
        agg = published["aggregate"]
        seeds = list(published.get("seeds", {}).keys())
        st.markdown(f"### Published results (seeds {', '.join(seeds)}, test runs only)")
        rows = []
        for name, r in agg["detection"].items():
            rows.append(
                {
                    "detector": name,
                    "AUROC": f"{r['auroc']['mean']:.3f} ± {r['auroc']['std']:.3f}",
                    "TPR @ FAR 1 %": f"{r['tpr_at_far']['mean']:.2f} ± {r['tpr_at_far']['std']:.2f}",
                    "runs detected": f"{r['fraction_detected']['mean']:.0%}",
                    "median delay (min)": f"{r['median_delay_min']['mean']:.0f} ± {r['median_delay_min']['std']:.0f}",
                    "test cost (CHF)": f"{agg['decision'][name]['expected_cost']['mean']:,.0f}",
                }
            )
        st.dataframe(pd.DataFrame(rows).set_index("detector"))
        d, u = agg["diagnosis"], agg["rul"]
        st.caption(
            f"Diagnosis accuracy {d['accuracy']['mean']:.3f} ± {d['accuracy']['std']:.3f}, "
            f"macro-F1 {d['macro_f1']['mean']:.3f} · RUL MAE {u['mae_minutes']['mean']:.0f} min, "
            f"80 % coverage {u['coverage_80']['mean']:.0%}."
        )
    else:
        st.info("Run `sensorlab evaluate --seeds 0 1 2` to publish multi-seed results here.")

    st.markdown("### Honest limitations")
    st.markdown(
        """
- Numbers come from the **synthetic generator**. The real Rieth 2017 release
  (`make download-tep`, `sensorlab train --data real`) will shift them.
- The RUL target is *time until the end of the simulated run*; with a fixed
  run length it is a deterministic function of time since onset. Treat it as
  a **ranking of urgency**, not as a calibrated time-to-failure.
- The synthetic fault catalogue maps 21 faults onto 7 mechanics, so several
  fault ids share a signature and are legitimately confusable.
- The drift monitor uses PSI on marginals; it will not see a change in the
  correlation structure with unchanged marginals.
"""
    )

# ----- Tab 1: live run --------------------------------------------------------
with tab_overview:
    st.subheader("Pick a test run and watch the pipeline")
    default_idx = (
        int(np.where(ds.run_fault_id[test_runs] == 4)[0][0])
        if (ds.run_fault_id[test_runs] == 4).any()
        else 0
    )
    run_choice = st.selectbox(
        "test run",
        options=test_runs.tolist(),
        format_func=lambda r: f"#{r:02d}  ·  {ds.fault_names[int(ds.run_fault_id[r])]}",
        index=default_idx,
    )
    mask = ds.run_id == run_choice
    pred_run = predictions[predictions["run_id"] == run_choice].reset_index(drop=True)

    col_a, col_b = st.columns([1.6, 1])
    with col_a:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        plot_sensor_traces(
            ds.X[mask],
            ds.sensor_names,
            sensor_idx=[0, 4, 9, 14, 22],
            is_anomaly=ds.is_anomaly[mask],
            ax=ax,
        )
        ax.set_title(f"Run #{run_choice}: sensor traces")
        st.pyplot(fig)

        fig2, axes = plt.subplots(len(ALL_DETECTORS), 1, figsize=(10, 7), sharex=True)
        for ax_, name in zip(axes, ALL_DETECTORS, strict=True):
            plot_detection_scores(
                scores[name][mask],
                is_anomaly=ds.is_anomaly[mask],
                threshold=pipe.far_thresholds_[name],
                title=f"{name} (threshold @ FAR {pipe.config.far_target:.0%} on validation-normal)",
                ax=ax_,
            )
        st.pyplot(fig2)

    with col_b:
        st.markdown("**What the pipeline recommends over this run**")
        onset = int(ds.run_onsets[run_choice])
        _alarms = pred_run.index[pred_run["alarm"]]
        _alarms = _alarms[_alarms >= onset] if onset >= 0 else _alarms
        first_alarm = int(_alarms.min()) if len(_alarms) else None
        c1, c2 = st.columns(2)
        c1.metric("fault onset", "—" if onset < 0 else f"{onset * ds.sample_minutes:.0f} min")
        c2.metric(
            "first confirmed alarm after onset" if onset >= 0 else "first confirmed alarm",
            "none" if first_alarm is None else f"{first_alarm * ds.sample_minutes:.0f} min",
            delta=None
            if first_alarm is None or onset < 0
            else f"{(first_alarm - onset) * ds.sample_minutes:+.0f} min vs onset",
            delta_color="inverse",
        )
        if first_alarm is not None:
            row = pred_run.iloc[first_alarm]
            st.markdown(
                f"At the first alarm the classifier says **{row['fault_name_pred']}** "
                f"(confidence {row['fault_confidence']:.2f}), RUL median "
                f"**{row['rul_p50_min']:.0f} min** "
                f"[{row['rul_p10_min']:.0f}, {row['rul_p90_min']:.0f}] → action **`{row['action']}`**."
            )
        st.dataframe(
            pred_run[
                ["t_minutes", "primary_score", "alarm", "fault_name_pred", "rul_p50_min", "action"]
            ]
            .rename(columns={"t_minutes": "t (min)", "rul_p50_min": "RUL p50 (min)"})
            .iloc[:: max(1, len(pred_run) // 25)],
            height=420,
        )

    st.markdown("#### How to read this view")
    _is_normal = int(ds.run_fault_id[run_choice]) == 0
    if _is_normal:
        st.markdown(
            "This is a **normal** run. A well-calibrated pipeline should stay in `wait` for the "
            "entire run; any confirmed alarm here is a false alarm at the chosen operating point."
        )
    else:
        st.markdown(
            """
1. **Lead time** — minutes between the fault onset (shaded) and the first *confirmed* alarm
   (three consecutive samples above the operating threshold, the same rule the metrics use).
2. **Cleanliness** — does the score stay above threshold once the fault is active, or flicker?
3. **Action** — `investigate` means the detector fired but the classifier still reads normal;
   `schedule_maintenance` vs `intervene_now` is decided by the RUL median against the
   configured horizon.
"""
        )

# ----- Tab 2: detector comparison ---------------------------------------------
with tab_detect:
    st.subheader("Detector benchmark on the test split")
    rows = []
    for name, r in evaluation["detection"].items():
        rows.append(
            {
                "detector": name,
                "AUROC": round(r["auroc"], 3),
                "TPR@FAR=1%": round(r["tpr_at_far"], 3),
                "FAR observed": round(r["far_observed"], 3),
                "runs detected": f"{r['fraction_detected']:.0%}",
                "median delay (min)": _fmt_delay(r["median_delay_min"]),
                "fit (s)": round(r["fit_seconds"] or 0.0, 1),
            }
        )
    df = pd.DataFrame(rows).set_index("detector")
    st.dataframe(df)
    st.caption(
        f"Thresholds calibrated on validation-normal at FAR = {pipe.config.far_target:.0%}; "
        f"metrics on {test_runs.size} held-out test runs."
    )

    _best_auc = df["AUROC"].idxmax()
    _delays = df["median delay (min)"].replace("—", np.nan).astype(float)
    _best_delay = _delays.idxmin() if _delays.notna().any() else _best_auc
    _spread = float(df["AUROC"].max() - df["AUROC"].min())
    st.markdown("#### What this tells us")
    st.markdown(
        f"""
- **{_best_auc}** wins on AUROC ({df.loc[_best_auc, "AUROC"]:.2f}); **{_best_delay}** has the
  shortest median delay on the runs it catches.
- The AUROC spread is **{_spread:.2f}** — a real operational difference at the same false-alarm
  budget. The *observed* FAR column is the honesty check: a threshold set on validation
  should reproduce roughly the target FAR on test.
- The pipeline picked **{pipe.primary_detector_}** as primary detector by expected cost on the
  validation runs. Whether it is also the cheapest on test is shown in the Decision tab.
"""
    )

    st.markdown("### PCA-2 projection coloured by fault")
    fid_set = st.multiselect("fault ids to show", list(range(22)), default=[0, 1, 4, 13, 14, 16])
    sample_mask = np.isin(ds.active_fault_id, fid_set)
    if fid_set:
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_pca_projection(
            pipe.scaler_.transform(ds.X[sample_mask]),
            ds.active_fault_id[sample_mask],
            label_names=ds.fault_names,
            max_classes=len(fid_set),
            ax=ax,
        )
        st.pyplot(fig)
    st.caption(
        "Coloured by the fault *active at each sample* — pre-onset samples of faulty runs are "
        "drawn as Normal, which is what the classifier is asked to learn."
    )

# ----- Tab 3: diagnosis ---------------------------------------------------------
with tab_diag:
    st.subheader("Which fault is active? Which sensor drives it?")
    d = evaluation["diagnosis"]
    c1, c2, c3 = st.columns(3)
    c1.metric("accuracy (test)", f"{d['accuracy']:.3f}")
    c2.metric("balanced accuracy", f"{d['balanced_accuracy']:.3f}")
    c3.metric("macro-F1", f"{d['macro_f1']:.3f}")

    rep = compute_shap(lab_key, pipe, ds, test, sample_size=400)
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_shap_summary(
        rep.per_sensor_class_importance, ds.sensor_names, rep.class_ids, top_k=14, ax=ax
    )
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### Top driver sensors per fault")
    rows = []
    for fid in rep.class_ids:
        if int(fid) == 0:
            continue
        tops = rep.top_sensors(int(fid), k=3)
        rows.append(
            {
                "fault": ds.fault_names[int(fid)],
                "top_1": tops[0][0],
                "top_2": tops[1][0],
                "top_3": tops[2][0],
            }
        )
    st.dataframe(pd.DataFrame(rows).set_index("fault"))

    st.markdown("#### What this tells us")
    st.markdown(
        """
- Every fault type ends up with **one or two dominant driver sensors** — the lever for
  mean-time-to-root-cause on a real plant: *"sensor X is the dominant signal, consistent with
  fault family Y"* instead of *"something is wrong"*.
- Faults that share a driver sensor are exactly the pairs the confusion matrix mixes up
  (notebook 03). A **hierarchical classifier** (fault family → sub-type) is the natural fix.
- In a regulated environment a black box without attributions is not deployable; the SHAP
  table is what the process engineer signs off on.
"""
    )

# ----- Tab 4: decision layer ---------------------------------------------------
with tab_decide:
    st.subheader("Pick the cost model → see the optimal threshold on test runs")
    det_choice = st.selectbox(
        "detector", list(ALL_DETECTORS), index=list(ALL_DETECTORS).index(pipe.primary_detector_)
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        fa = st.number_input("false alarm cost (CHF)", 0.0, 10_000.0, 100.0, 10.0)
    with col2:
        mf = st.number_input("missed fault cost (CHF)", 100.0, 100_000.0, 5_000.0, 100.0)
    with col3:
        ld = st.number_input("delay cost (CHF / min late)", 0.0, 1_000.0, 50.0, 5.0)

    cost = CostModel(
        false_alarm_cost=float(fa), missed_fault_cost=float(mf), delay_cost_per_min=float(ld)
    )
    s = scores[det_choice]
    grid, results = cost_curve(
        s[test],
        ds.run_id[test],
        ds.run_onsets,
        ds.run_fault_id,
        cost=cost,
        n_grid=50,
        samples_to_minutes=ds.sample_minutes,
        consecutive=pipe.config.consecutive,
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    plot_cost_curve(grid, results, ax=ax)
    ax.axvline(
        pipe.decision_thresholds_[det_choice],
        color="#10b981",
        linestyle=":",
        label="threshold chosen on validation",
    )
    ax.legend()
    st.pyplot(fig)

    best = optimal_threshold(
        s[test],
        ds.run_id[test],
        ds.run_onsets,
        ds.run_fault_id,
        cost=cost,
        samples_to_minutes=ds.sample_minutes,
        consecutive=pipe.config.consecutive,
    )
    deployed = evaluation["decision"][det_choice]
    st.markdown(
        "**Oracle optimum on test** (what you would get if you could tune on test) vs **deployed threshold** (chosen on validation at the reference cost mix):"
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "threshold",
        f"{best.threshold:.3f}",
        delta=f"deployed {deployed['threshold']:.3f}",
        delta_color="off",
    )
    c2.metric(
        "expected cost",
        f"{best.expected_cost:,.0f} CHF",
        delta=f"deployed {deployed['expected_cost']:,.0f}",
        delta_color="off",
    )
    c3.metric(
        "false alarms",
        f"{best.false_alarms}",
        delta=f"deployed {deployed['false_alarms']}",
        delta_color="off",
    )
    c4.metric(
        "missed faults",
        f"{best.missed_faults}",
        delta=f"deployed {deployed['missed_faults']}",
        delta_color="off",
    )

    _ratio = mf / max(fa, 1e-6)
    st.markdown("#### What this tells us")
    st.markdown(
        f"""
- The optimum is **not** the highest-AUROC operating point — it is the threshold that minimises
  total expected cost under *your* cost mix (currently missed : false-alarm = **{_ratio:.0f} : 1**).
- The gap between the oracle and the deployed threshold is the price of choosing the operating
  point on validation instead of peeking at test — that is the number to report to the business.
- Lowering the false-alarm cost or raising the missed-fault cost shifts the optimum **down**: the
  detector becomes more eager. Drag *missed fault cost* to 20 000 CHF and watch the false-alarm
  count rise.
- *They* set the cost mix; *we* deliver the calibrated curve. That separation of concerns is what
  makes the model negotiable with finance and operations.
"""
    )

# ----- Tab 5: operate -----------------------------------------------------------
with tab_operate:
    st.subheader("What the plant team actually receives")
    st.markdown(
        """
`sensorlab train` saves the fitted pipeline plus a manifest; `sensorlab score --input batch.csv`
returns one row per sample with scores, confirmed alarm, diagnosed fault, RUL interval and the
recommended action. The tables below are that output on the held-out test runs.
"""
    )

    summary_rows = []
    for rid, g in predictions.groupby("run_id"):
        g = g.reset_index(drop=True)
        onset = int(ds.run_onsets[rid])
        true_fault = int(ds.run_fault_id[rid])
        alarms = g.index[g["alarm"]]
        pre = alarms[alarms < onset] if onset >= 0 else alarms
        post = alarms[alarms >= onset] if onset >= 0 else alarms[:0]
        first = int(post.min()) if len(post) else None
        if true_fault == 0:
            outcome = "false alarm" if len(alarms) else "clean"
        else:
            outcome = "caught" if first is not None else "missed"
        summary_rows.append(
            {
                "run": int(rid),
                "true fault": ds.fault_names[true_fault],
                "onset (min)": None if onset < 0 else onset * ds.sample_minutes,
                "first alarm after onset (min)": None
                if first is None
                else first * ds.sample_minutes,
                "delay (min)": None if first is None else (first - onset) * ds.sample_minutes,
                "pre-onset false alarms": len(pre) if true_fault > 0 else len(alarms),
                "diagnosed at alarm": None if first is None else g.loc[first, "fault_name_pred"],
                "RUL p50 at alarm": None
                if first is None
                else round(float(g.loc[first, "rul_p50_min"])),
                "final action": g["action"].iloc[-1],
                "outcome": outcome,
            }
        )
    summary = pd.DataFrame(summary_rows).set_index("run")
    outcome_counts = summary["outcome"].value_counts().to_dict()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "faulty runs caught",
        f"{outcome_counts.get('caught', 0)} / {outcome_counts.get('caught', 0) + outcome_counts.get('missed', 0)}",
    )
    c2.metric(
        "normal runs with a false alarm",
        f"{outcome_counts.get('false alarm', 0)} / {outcome_counts.get('false alarm', 0) + outcome_counts.get('clean', 0)}",
    )
    c3.metric("alarm precision (samples)", f"{evaluation['actions']['alarm_precision']:.2f}")
    c4.metric("alarm recall (samples)", f"{evaluation['actions']['alarm_recall']:.2f}")
    st.dataframe(summary, height=380)

    st.download_button(
        "Download scored test runs (CSV)",
        predictions.to_csv(index=False).encode(),
        file_name="sensorlab_scored_test_runs.csv",
        mime="text/csv",
    )

    st.markdown("### Drift check: has the plant changed under the model?")
    st.markdown(
        "The pipeline stores a compact reference of the training-normal distribution and reports "
        "a per-sensor **Population Stability Index** on every batch. Simulate a sensor "
        "recalibration and watch the monitor react before the detectors start lying."
    )
    normal_test_runs = [int(r) for r in test_runs if ds.run_fault_id[r] == 0]
    if normal_test_runs:
        c1, c2 = st.columns(2)
        drift_sensor = c1.selectbox("sensor to recalibrate", ds.sensor_names, index=0)
        drift_shift = c2.slider("offset (in training std units)", 0.0, 3.0, 0.0, 0.25)
        batch = ds.X[ds.run_mask(normal_test_runs)].copy()
        j = ds.sensor_names.index(drift_sensor)
        batch[:, j] += drift_shift * float(pipe.scaler_.std[j])
        rep = pipe.check_drift(batch)
        status_colour = {"stable": "🟢", "watch": "🟡", "alert": "🔴"}[rep.status]
        st.markdown(
            f"**Status: {status_colour} {rep.status}** on {rep.n_current} samples of normal test runs"
        )
        psi = pd.Series(rep.psi).sort_values(ascending=False)
        st.bar_chart(psi.head(12))
        if rep.alert:
            st.warning(
                f"PSI ≥ {pipe.drift_.alert_threshold}: {', '.join(rep.alert)} — do not trust alarms on these sensors until retrained."
            )
        elif rep.watch:
            st.info(f"PSI in the watch band: {', '.join(rep.watch)}.")
    else:
        st.info("No normal runs in the test split for this configuration — increase *normal runs*.")

    with st.expander("Pipeline manifest (what ships with the model)"):
        st.json(pipe.manifest(), expanded=False)

st.markdown("---")
st.caption(
    "Built with [sensorlab](https://github.com/Gemmagf/sensorlab). "
    "All data here is from the reproducible synthetic generator — swap to the real Tennessee "
    "Eastman release with `make download-tep`."
)
