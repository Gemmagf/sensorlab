"""Interactive dashboard for the sensorlab pipeline.

Run with::

    make app          # or
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import sensorlab  # noqa: F401 — env-var setup first
from sensorlab.data import (
    Standardizer,
    SyntheticTEPConfig,
    load_dataset,
    sliding_windows,
    train_val_test_split_by_run,
)
from sensorlab.decision import CostModel, cost_curve, optimal_threshold
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
    window_features,
)
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
# Cached resources
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Generating synthetic TEP data…")
def get_dataset(seed: int, n_normal: int, n_per_fault: int, run_minutes: int):
    cfg = SyntheticTEPConfig(
        n_normal_runs=n_normal,
        n_runs_per_fault=n_per_fault,
        fault_run_minutes=run_minutes,
        seed=seed,
    )
    return load_dataset("synthetic", cfg=cfg)


@st.cache_resource(show_spinner="Splitting & standardising…")
def get_splits(_ds, seed: int):
    train, val, test = train_val_test_split_by_run(_ds, seed=seed)
    sc = Standardizer.fit(_ds.X[train & (_ds.fault_id == 0)])
    return train, val, test, sc, sc.transform(_ds.X)


@st.cache_resource(show_spinner="Fitting detectors (one-time)…")
def fit_detectors(_ds, train, val, _Xz, window: int, stride: int, ae_epochs: int):
    normal_train = train & (_ds.fault_id == 0)
    spc = PCAMonitor(var_explained=0.9).fit(_Xz[normal_train])
    ifo = IForestDetector(n_estimators=200, random_state=0).fit(_Xz[normal_train])
    windows, _, end_idx = sliding_windows(_Xz, _ds.run_id, window=window, stride=stride)
    ae = LSTMAutoencoder(window=window, epochs=ae_epochs, hidden=24, latent=6, seed=0).fit(
        windows[normal_train[end_idx]]
    )
    s_ae_win = ae.score(windows)
    s_ae = np.zeros(_ds.n_samples, dtype=np.float32)
    s_ae[end_idx] = s_ae_win
    for r in np.unique(_ds.run_id):
        m = np.where(_ds.run_id == r)[0]
        sub = s_ae[m]
        last = 0.0
        for i, v in enumerate(sub):
            if v == 0 and last > 0:
                sub[i] = last
            else:
                last = v
        s_ae[m] = sub
    scores = {
        "PCA-T2Q": spc.score(_Xz),
        "IForest": ifo.score(_Xz),
        "LSTM-AE": s_ae,
    }
    return scores, windows, end_idx


@st.cache_resource(show_spinner="Training fault classifier (one-time)…")
def fit_classifier(_ds, train, _windows, end_idx):
    feats, fnames = window_features(_windows, _ds.sensor_names)
    in_train = train[end_idx]
    clf = FaultClassifier(n_estimators=200, max_depth=6).fit(
        feats[in_train], _ds.fault_id[end_idx][in_train], feature_names=fnames
    )
    return clf, feats, fnames


@st.cache_data(show_spinner="Computing SHAP report…")
def compute_shap(_clf, feats, fnames, sensor_names, sample_size: int = 400):
    idx = np.random.default_rng(0).choice(
        feats.shape[0], min(sample_size, feats.shape[0]), replace=False
    )
    return explain_classifier(_clf, feats[idx], fnames, sensor_names, max_background=sample_size)


# ---------------------------------------------------------------------------
# Sidebar — global configuration
# ---------------------------------------------------------------------------
st.sidebar.title("🏭 sensorlab")
st.sidebar.caption("Tennessee Eastman fault detection & decision lab")

with st.sidebar.expander("Dataset", expanded=False):
    seed = st.number_input("seed", 0, 9999, 0, 1)
    n_normal = st.slider("normal runs", 4, 20, 12)
    n_per_fault = st.slider("runs per fault", 2, 8, 4)
    run_minutes = st.slider("minutes per run", 120, 720, 480, 60)

with st.sidebar.expander("Models", expanded=False):
    window = st.slider("window size", 10, 40, 20, 2)
    stride = st.slider("stride", 1, 5, 2)
    ae_epochs = st.slider("LSTM-AE epochs", 5, 50, 15, 5)

ds = get_dataset(int(seed), int(n_normal), int(n_per_fault), int(run_minutes))
train, val, test, scaler, Xz = get_splits(ds, int(seed))
scores, windows, end_idx = fit_detectors(
    ds, train, val, Xz, int(window), int(stride), int(ae_epochs)
)
normal_val = val & (ds.fault_id == 0)

st.sidebar.markdown("---")
st.sidebar.metric("Total samples", f"{ds.n_samples:,}")
st.sidebar.metric("Runs", ds.n_runs)
st.sidebar.metric("Sensors", ds.n_sensors)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
st.title("Industrial sensor anomaly lab")
st.markdown(
    "An end-to-end demo: synthetic Tennessee-Eastman traces → three detectors → "
    "fault diagnosis with SHAP → cost-aware decision threshold."
)

tab_overview, tab_detect, tab_diag, tab_decide = st.tabs(
    ["📈 Live run", "🚨 Detector comparison", "🔍 Diagnosis (SHAP)", "💰 Decision layer"]
)

# ----- Tab 1: live run ------------------------------------------------------
with tab_overview:
    st.subheader("Pick a run and watch the detectors")
    run_choice = st.selectbox(
        "run",
        options=list(range(ds.n_runs)),
        format_func=lambda r: f"#{r:02d}  ·  {ds.fault_names[int(ds.run_fault_id[r])]}",
        index=int(np.where(ds.run_fault_id == 4)[0][0]),
    )
    mask = ds.run_id == run_choice
    far = st.slider("False-alarm target on validation normal", 0.001, 0.05, 0.01, 0.001)

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

        fig2, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        for ax_, (name, s) in zip(axes, scores.items(), strict=False):
            thr = threshold_at_far(s[normal_val], far=far)
            plot_detection_scores(
                s[mask], is_anomaly=ds.is_anomaly[mask], threshold=thr, title=name, ax=ax_
            )
        st.pyplot(fig2)

    with col_b:
        st.markdown("**Score summary on this run**")
        rows = []
        for name, s in scores.items():
            thr = threshold_at_far(s[normal_val], far=far)
            d = detection_delay(
                s,
                ds.run_id,
                ds.run_onsets,
                ds.run_fault_id,
                thr,
                samples_to_minutes=ds.sample_minutes,
            )
            rows.append(
                {
                    "detector": name,
                    "threshold": round(thr, 3),
                    "frac_detected_all_runs": round(d["fraction_detected"], 2),
                    "median_delay_min": (
                        round(d["median_min"], 0) if not np.isnan(d["median_min"]) else "—"
                    ),
                }
            )
        st.dataframe(pd.DataFrame(rows).set_index("detector"))

# ----- Tab 2: detector comparison ------------------------------------------
with tab_detect:
    st.subheader("Detector benchmark on the test split")
    rows = []
    for name, s in scores.items():
        thr = threshold_at_far(s[normal_val], far=0.01)
        a = auroc(s[test], ds.is_anomaly[test])
        tpr = true_positive_rate(s[test], ds.is_anomaly[test], thr)
        d = detection_delay(
            s, ds.run_id, ds.run_onsets, ds.run_fault_id, thr, samples_to_minutes=ds.sample_minutes
        )
        rows.append(
            {
                "detector": name,
                "AUROC": round(a, 3),
                "TPR@FAR=1%": round(tpr, 3),
                "frac_detected": round(d["fraction_detected"], 2),
                "median_delay_min": (
                    round(d["median_min"], 0) if not np.isnan(d["median_min"]) else "—"
                ),
            }
        )
    df = pd.DataFrame(rows).set_index("detector")
    st.dataframe(df)
    st.caption("Detectors fitted on normal-only training runs; metrics on held-out test runs.")

    st.markdown("### PCA-2 projection coloured by fault")
    fid_set = st.multiselect("fault ids to show", list(range(22)), default=[0, 1, 4, 13, 14, 16])
    sample_mask = np.isin(ds.fault_id, fid_set)
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_pca_projection(
        Xz[sample_mask],
        ds.fault_id[sample_mask],
        label_names=ds.fault_names,
        max_classes=len(fid_set),
        ax=ax,
    )
    st.pyplot(fig)

# ----- Tab 3: diagnosis -----------------------------------------------------
with tab_diag:
    st.subheader("Which fault is active? Which sensor drives it?")
    clf, feats, fnames = fit_classifier(ds, train, windows, end_idx)
    rep = compute_shap(clf, feats, fnames, ds.sensor_names, sample_size=400)
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

# ----- Tab 4: decision layer ------------------------------------------------
with tab_decide:
    st.subheader("Pick the cost model → see the optimal threshold")
    det_choice = st.selectbox("detector", list(scores.keys()), index=1)
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
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    plot_cost_curve(grid, results, ax=ax)
    st.pyplot(fig)

    best = optimal_threshold(
        s[test],
        ds.run_id[test],
        ds.run_onsets,
        ds.run_fault_id,
        cost=cost,
        samples_to_minutes=ds.sample_minutes,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("optimal threshold", f"{best.threshold:.3f}")
    c2.metric("expected cost", f"{best.expected_cost:,.0f} CHF")
    c3.metric("false alarms", f"{best.false_alarms}")
    c4.metric("missed faults", f"{best.missed_faults}")
    st.caption(
        f"Detector **{det_choice}** at this cost mix: "
        f"mean detection delay **{best.mean_delay_min:.1f} min** on {best.n_faulty_runs} faulty test runs."
    )

st.markdown("---")
st.caption(
    "Built with [sensorlab](https://github.com/Gemmagf/sensorlab). "
    "All data here is from the reproducible synthetic generator — swap to the real Tennessee "
    "Eastman release with `make download-tep`."
)
