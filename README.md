# sensorlab

[![CI](https://github.com/Gemmagf/sensorlab/actions/workflows/ci.yml/badge.svg)](https://github.com/Gemmagf/sensorlab/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)

> **Fault detection, diagnosis and remaining-useful-life on the Tennessee Eastman chemical process benchmark — from sensor telemetry to operational decisions.**

`sensorlab` is an end-to-end pipeline for **continuous-process monitoring**.
Three complementary detectors (classical multivariate SPC, Isolation Forest,
LSTM autoencoder) are compared head-to-head on detection delay and false alarm
rate. An XGBoost classifier with SHAP attributions pinpoints the root cause
sensor of each fault. A quantile-regression RUL head estimates time-to-failure
with calibrated uncertainty. A **cost-aware decision layer** maps detector
scores onto operational thresholds that minimise expected loss — turning a raw
anomaly score into an actionable "intervene now / schedule maintenance / wait"
recommendation.

---

## ✨ Why this project

The [Tennessee Eastman process](https://en.wikipedia.org/wiki/Tennessee_Eastman_Process)
(Downs & Vogel, 1993) is the canonical benchmark for fault detection in
continuous chemical processes — used in hundreds of papers
([Bathelt et al. 2015](https://doi.org/10.1016/j.ifacol.2015.08.199),
Yin et al., Lyman & Georgakis). It has **41 process measurements** + **12
manipulated variables**, **21 documented fault scenarios** ranging from step
changes (catalyst poisoning, feed loss) to slow drifts (sticking valves,
kinetics degradation), and is simulated from first principles — making it the
closest public stand-in for the kind of plant data a chemical R&D team
actually sees.

This repo treats TEP exactly the way an industrial team would: build a
detection layer, build a diagnosis layer on top, layer RUL on slow faults,
then translate everything into operating thresholds that minimise the
business cost.

## 🏗️ Architecture

```
                            ┌─────────────────────────┐
   TEP simulator            │   sensorlab.data        │
   or Rieth 2017 release ──►│   loaders & preprocess  │
                            └────────────┬────────────┘
                                         │ (X, y, fault_type)
              ┌──────────────────────────┼──────────────────────────┐
              ▼                          ▼                          ▼
  ┌────────────────────┐  ┌─────────────────────────┐  ┌────────────────────┐
  │  detection         │  │  diagnosis              │  │  rul               │
  │  • T² / Q SPC      │  │  • XGBoost multi-class  │  │  • quantile reg.   │
  │  • IsolationForest │  │  • SHAP per fault       │  │  • Cox survival    │
  │  • LSTM-AE         │  │                         │  │                    │
  └─────────┬──────────┘  └───────────┬─────────────┘  └─────────┬──────────┘
            │   scores                │  fault label             │ time-to-fail
            └──────────────┬──────────┴────────────┬─────────────┘
                           ▼                       ▼
                ┌──────────────────────┐  ┌────────────────────┐
                │  decision layer      │  │  Streamlit app     │
                │  cost-weighted       │──►   what-if explorer │
                │  threshold tuning    │  └────────────────────┘
                └──────────────────────┘
```

## 🚀 Quick start

```bash
# Set up
make install            # creates .venv (Python 3.11) and installs everything
make test               # 30+ unit tests should pass

# End-to-end on synthetic TEP-like data (no download needed)
make train

# Or on the real benchmark (Rieth et al. 2017, ~5 GB)
make download-tep
python scripts/train_all.py --data real

# Interactive
make notebook           # walk through 01_eda → 05_decision_layer
make app                # launch the Streamlit dashboard
```

## 📦 Package layout

```
src/sensorlab/
├── config.py              # paths, TEP spec, fault catalogue
├── data/
│   ├── synthetic.py       # reproducible TEP-like generator
│   ├── loader.py          # unified loader: synthetic or real
│   └── preprocess.py      # windowing, scaling, train/val/test splits
├── detection/
│   ├── spc.py             # Hotelling T² + SPE/Q residual statistics
│   ├── iforest.py         # IsolationForest wrapper
│   ├── autoencoder.py     # LSTM autoencoder (PyTorch)
│   └── metrics.py         # detection delay, FAR, TPR, AUC
├── diagnosis/
│   ├── classifier.py      # XGBoost multi-class
│   └── explain.py         # SHAP per-fault & global
├── rul/
│   ├── quantile.py        # quantile gradient boosting (q=0.1/0.5/0.9)
│   └── survival.py        # CoxPH baseline
├── decision/
│   └── cost.py            # expected-cost threshold optimisation
└── viz/
    └── plots.py           # consistent matplotlib helpers
```

## 🔬 Results

Numbers below come from `make train` on the synthetic generator (48 test runs
out of 96 total; 33 sensors; 21 fault scenarios). The same pipeline run with
`--data real` on the Rieth et al. 2017 release reproduces results in line with
the published Bathelt et al. 2015 baselines. Re-run reproduces from the JSON
in [artifacts/results.json](artifacts/).

### Detection — TPR at 1% FAR, median delay, AUROC

| Detector            | AUROC | TPR @ FAR=1% | Fraction detected | Median delay |
|---------------------|------:|-------------:|------------------:|-------------:|
| Hotelling T² + Q    | 0.76  | 0.15         | 50 %              | 106 min      |
| Isolation Forest    | 0.84  | 0.38         | 71 %              |  44 min      |
| **LSTM Autoencoder**| **0.93**  | **0.59**     | **90 %**          |  **42 min**  |

The progression matches the published TEP story: classical multivariate SPC
is conservative; tree ensembles catch sharp regime changes; the recurrent
deep model wins by reading the temporal signature.

### Diagnosis — 22-way classification (Normal + F01–F21)

| Model            | Accuracy | macro-F1 |
|------------------|---------:|---------:|
| XGBoost + SHAP   | 0.57     | 0.58     |

Chance accuracy on this 22-class task is ~5 %. SHAP attributions identify a
single dominant driver sensor for most fault types — see
[notebooks/03_diagnosis.ipynb](notebooks/03_diagnosis.ipynb).

### Remaining useful life

| Metric                           | Value      |
|----------------------------------|------------|
| MAE (test, faulty samples only)  | 94 min     |
| 80 % prediction-interval coverage| 69 %       |

Slight under-coverage is typical of vanilla quantile gradient boosting; a
conformal-prediction wrapper would tighten the bounds.

## 🧪 Tests & CI

```bash
make test    # pytest -v
make lint    # ruff check + format check
```

GitHub Actions runs the test suite on Python 3.11 and 3.12 on every push.

## 📚 References

- Downs, J.J. & Vogel, E.F. (1993). *A plant-wide industrial process control problem*. Computers & Chemical Engineering.
- Bathelt, A., Ricker, N.L. & Jelali, M. (2015). *Revision of the Tennessee Eastman process model*. IFAC-PapersOnLine.
- Rieth, C.A. et al. (2017). *Issues and Advances in Anomaly Detection Evaluation for Joint Human-Automated Systems*. Harvard Dataverse.

## 📄 License

MIT — see [LICENSE](LICENSE).
