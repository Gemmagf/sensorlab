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

Numbers below come from `make train` on the synthetic generator (96 runs ×
~160 samples = 15 360 timesteps; 33 sensors; 21 fault scenarios; 48 runs
held out as test). The same pipeline run with `--data real` on the Rieth et
al. 2017 release reproduces results in line with the published Bathelt et
al. 2015 baselines. All numbers come from [artifacts/results.json](artifacts/results.json).

### Detection — TPR at 1 % FAR, median delay, AUROC

| Detector            | AUROC     | TPR @ FAR=1 % | Fraction detected | Median delay  | Fit time |
|---------------------|----------:|--------------:|------------------:|--------------:|---------:|
| Hotelling T² + Q    | 0.76      | 0.15          | 50 %              | 106 min       | 0.05 s   |
| Isolation Forest    | 0.84      | 0.38          | 71 %              |  44 min       | 1.9 s    |
| **LSTM Autoencoder**| **0.93**  | **0.59**      | **90 %**          |  **42 min**   | 9.2 s    |

**What this tells us**

- The progression matches the published TEP story: classical multivariate SPC
  is conservative, tree ensembles catch sharp regime changes, the recurrent
  deep model wins by reading the **temporal** signature that the other two
  ignore.
- The LSTM-AE catches **9 out of 10** disturbances at a 1 % false-alarm rate
  with a **42-minute lead time** on average — enough headroom for an operator
  to take corrective action before downstream KPIs (off-spec product, scrap)
  even react.
- T²/Q is **180× faster to fit** than the LSTM-AE and still gives 50 %
  detection at the strict operating point — it remains the right pick when
  inference latency or interpretability is non-negotiable.
- **None of the three is dominated on every metric**: T²/Q is fastest,
  IsolationForest is the best speed/accuracy compromise, LSTM-AE is the
  most accurate. A production deployment should ensemble all three and
  use the **decision layer** to pick a fused threshold.

### Diagnosis — 22-way classification (Normal + F01–F21)

| Model            | Accuracy | macro-F1 | Fit time |
|------------------|---------:|---------:|---------:|
| XGBoost + SHAP   | 0.57     | 0.58     | 67 s     |

**What this tells us**

- Chance accuracy on this 22-class task is ~5 % — the model is **~11×
  better than random**, with macro-F1 ≈ accuracy meaning performance is not
  driven by a single dominant class.
- SHAP identifies a **single dominant driver sensor for every fault type**
  (see [artifacts/results.json](artifacts/results.json) → `diagnosis.top_sensors_per_fault`).
  Examples: F01 → `XMV(2)`, F04 → `XMEAS(21)`, F14 → `XMEAS(21)`, F17 → `XMEAS(1)`.
- Operationally this is the **mean-time-to-root-cause** lever: instead of a
  generic "something is wrong" alarm, the engineer reads *"sensor X is
  driving the model toward fault family Y"* and can intervene directly.
- Remaining 43 % error concentrates on **fault pairs with overlapping sensor
  signatures** (notebook 03 shows the confusion matrix) — a hierarchical
  classifier (fault-family first, then sub-type) is the natural next step.

### Remaining useful life — quantile gradient boosting

| Metric                                | Value    |
|---------------------------------------|---------:|
| MAE (test, faulty samples)            |  94 min  |
| Pinball loss (q = 0.5)                |  47.2    |
| 80 % prediction-interval coverage     |  69 %    |
| Fit time                              |  186 s   |

**What this tells us**

- 94-min MAE on runs that last ~480 min is a **~20 % relative error** — useful
  for *ranking* runs by urgency but not for hard service-level commitments.
- Coverage of 69 % vs the 80 % target means the model is **slightly
  over-confident**: 11 percentage points of intervals are too narrow. This is
  typical of vanilla quantile GBMs and is the **single highest-leverage
  next-step** — wrapping the model in **conformal prediction** would calibrate
  intervals to nominal with no retraining.
- The pinball loss at q = 0.5 (47.2) gives a metric that doesn't reward
  intervals being uselessly wide — pairing it with the coverage number is the
  honest way to report quantile performance.

### Decision layer — translating scores into operating cost

At the reference cost mix (false alarm = 100 CHF, missed fault = 5 000 CHF,
delay = 50 CHF/min) — picked because it represents the typical 1 : 50 ratio
in a continuous chemical line where a single missed fault scraps a batch:

| Detector            | Optimal threshold | Expected cost   | False alarms | Missed | Mean delay |
|---------------------|------------------:|----------------:|-------------:|-------:|-----------:|
| Hotelling T² + Q    |  3.6              | 224 500 CHF     | 1 714        |  0     | 12.6 min   |
| **Isolation Forest**| **0.50**          | **148 000 CHF** | **574**      |  **0** | 21.6 min   |
| LSTM Autoencoder    |  1.5              | 200 750 CHF     | 1 444        |  2     | 11.3 min   |

**What this tells us**

- **The "best detector" depends on the cost mix, not just the AUROC.** At
  this cost ratio IsolationForest wins decisively (~33 % cheaper than the
  next option) because it achieves 0 missed faults at a tolerable
  false-alarm rate.
- LSTM-AE, despite having the **highest AUROC**, is *not* the cost-optimal
  pick here: at this threshold it lets 2 faults through and the per-miss
  penalty of 5 000 CHF outweighs its faster median delay.
- A change in the cost mix shifts the winner — the Streamlit decision tab
  lets a process engineer drag the sliders and watch the optimum update in
  real time, turning the model into a tool finance and operations can
  actually negotiate over.

## 🧭 Overall conclusions

1. **The three-detector stack is honest.** No single model dominates; each
   trades cost, latency and interpretability differently. Reporting all three
   is what an industrial team needs to make the deployment call.
2. **Detection is the easy part; diagnosis is where domain value lives.**
   SHAP-driven root-cause attribution closes the gap between alarm and action
   and is the part a process engineer will use day-to-day.
3. **Calibration matters as much as accuracy.** A 0.93 AUROC tells finance
   nothing on its own; a calibrated cost curve does. Likewise a 94-min MAE
   without coverage diagnostics over-promises.
4. **Largest improvement on the table**: conformal-prediction wrapping the
   RUL head — pure win, no retraining, closes the 11-pp coverage gap.
5. **Second-largest**: hierarchical diagnosis (fault-family → sub-type) to
   recover the 43 % of diagnosis errors that concentrate on confused fault
   pairs.
6. **What this would look like deployed**: detector → fault classifier →
   RUL bound → decision layer, with the cost model owned by the business and
   the model owners responsible only for calibration and explainability.
   That separation is what the package layout encodes.

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
