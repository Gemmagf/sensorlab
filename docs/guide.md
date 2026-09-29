# sensorlab — user guide

How to install, run, read and extend the tool. The [README](../README.md) says *what* it is and
shows the results; the [runbook](runbook.md) is for the team that operates a deployed model.
This guide is for the person sitting in front of it for the first time.

---

## 1. What you are looking at

A continuous chemical plant emits dozens of correlated sensor streams. `sensorlab` turns those
streams into an operational decision in four layers:

| Layer | Question it answers | How |
|---|---|---|
| **Detection** | Is something wrong right now? | Three detectors trained on normal operation only: PCA T²/Q, Isolation Forest, LSTM autoencoder. Each returns an anomaly score per sample. |
| **Diagnosis** | Which fault, and which sensor is driving it? | XGBoost on window features (mean, std, last, slope per sensor) + SHAP attributions aggregated per sensor. |
| **Remaining useful life (RUL)** | How long until I must intervene? | Quantile gradient boosting (p10 / p50 / p90) with a conformal margin so the 80 % interval really covers 80 %. |
| **Decision** | What should the operator do? | A cost model (false alarm, missed fault, delay per minute) picks the operating threshold; alarm × diagnosis × RUL → `wait` · `investigate` · `schedule_maintenance` · `intervene_now`. |

All four live in one object, `MonitoringPipeline`, which is trained once, saved with a manifest,
and then used by the CLI, the API and the dashboard.

Data comes from the **Tennessee Eastman process** benchmark: either a built-in synthetic generator
(default, instant, reproducible) or the real Rieth et al. 2017 release (~5 GB download).

---

## 2. Install

```bash
git clone https://github.com/Gemmagf/sensorlab && cd sensorlab
make install          # creates .venv (Python 3.11), CPU-only torch, all extras
source .venv/bin/activate
sensorlab --version   # sensorlab 0.2.0
make test             # 110 tests, ~20 s — if this passes, everything works
```

Without `make`:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -e ".[serve,dev]"
```

Extras: `serve` (API + dashboard export), `dev` (tests, lint, notebooks), `real` (read the Rieth
RData files), `survival` (optional Cox baseline).

---

## 3. The five-minute tour

```bash
sensorlab train                # 1. fit on synthetic data, evaluate on held-out runs (~15 s)
sensorlab info                 # 2. print the manifest: thresholds, config, fit report
sensorlab export-site          # 3. write the dashboard to site/
python -m http.server -d site  # 4. open http://localhost:8000
```

What `train` produced:

| File | What it is |
|---|---|
| `models/pipeline.joblib` | the fitted pipeline — the thing you deploy |
| `models/pipeline.manifest.json` | model card: version, config, sensor layout, thresholds, dataset provenance |
| `artifacts/train_results.json` | held-out metrics of this single run |

The terminal summary at the end of `train` is the short version of the dashboard's Overview.

---

## 4. Command-line reference

Every command has `--help`. Data flags (`--data`, `--n-runs-per-fault`, `--n-normal-runs`,
`--fault-run-minutes`, `--real-root`) and model flags (`--window`, `--stride`, `--ae-epochs`,
`--far-target`, `--consecutive`, `--detectors`, `--primary-detector`, the three `--*-cost`
flags, `--intervene-horizon-min`, `--fast`) are shared by `train` and `evaluate`.

### `sensorlab train`
Fit one pipeline, evaluate it on the test runs, save model + manifest + results.

```bash
sensorlab train --seed 3 --ae-epochs 40 --missed-fault-cost 20000
sensorlab train --fast --no-save            # smoke run, small models, nothing written
sensorlab train --data real                 # after make download-tep + scripts/prepare_tep.py
```

### `sensorlab evaluate`
The same experiment over several seeds; writes per-seed results and mean ± std to
`artifacts/results.json`. This is the file the README tables and the dashboard's "published
results" line read from.

```bash
sensorlab evaluate --seeds 0 1 2            # ~1 min
```

### `sensorlab score`
Batch inference with a saved pipeline on a CSV or parquet file.

```bash
sensorlab score --input batch.csv --output scored.csv --drift
sensorlab score --input batch.parquet --output scored.parquet --run-col batch_id
```

Input contract:
- one row per sample, in time order within each run;
- the training sensor columns, named `XMEAS(1)` … `XMV(11)` (Rieth-style `xmeas_1` is accepted);
- optional `run_id` column (or `--run-col`) — rows of one run must be contiguous;
- no missing values (impute upstream; the tool refuses NaN rather than guessing).

Output: the input's rows with these columns appended (see §6 for how to read them):
`score_<detector>`, `primary_score`, `threshold`, `above_threshold`, `alarm`, `fault_id_pred`,
`fault_name_pred`, `fault_confidence`, `rul_p10_min`, `rul_p50_min`, `rul_p90_min`, `action`.
With `--drift` the per-sensor PSI report is logged and attached.

Exit codes: `0` ok · `2` missing sensor columns (the names are listed).

### `sensorlab serve`
The governance dashboard with a live scoring API.

```bash
sensorlab serve                              # http://127.0.0.1:8000, OpenAPI at /api/docs
sensorlab serve --host 0.0.0.0 --port 8080 --train-if-missing
curl -F file=@batch.csv "http://127.0.0.1:8000/api/score?format=csv" > scored.csv
```

### `sensorlab export-site`
The same dashboard as static files — no backend. This is what GitHub Pages serves.

```bash
sensorlab export-site --out site --base-href "/sensorlab/"   # sub-path hosting
```

### `sensorlab info`
Print the manifest of a saved pipeline.

---

## 5. Python API

```python
from sensorlab.data import load_dataset, SyntheticTEPConfig, train_val_test_split_by_run
from sensorlab.pipeline import MonitoringPipeline, PipelineConfig
from sensorlab.decision import CostModel

ds = load_dataset("synthetic", cfg=SyntheticTEPConfig(seed=0))
train, val, test = train_val_test_split_by_run(ds, seed=0)     # whole runs, never split

pipe = MonitoringPipeline(PipelineConfig(
    ae_epochs=25,
    cost=CostModel(false_alarm_cost=100, missed_fault_cost=5000, delay_cost_per_min=50),
    intervene_horizon_min=120,
)).fit(ds, train, val)                                          # test is never touched here

results = pipe.evaluate(ds, test)          # dict: detection / diagnosis / rul / decision / actions
out = pipe.predict(ds.X[test], ds.run_id[test])   # DataFrame, one row per sample
drift = pipe.check_drift(ds.X[test])       # DriftReport: status, psi, alert, watch

pipe.save("models/pipeline.joblib")
pipe = MonitoringPipeline.load("models/pipeline.joblib")
```

Lower-level pieces are importable on their own: `sensorlab.detection` (the three detectors and
`detection_delay`, `auroc`, `threshold_at_far`), `sensorlab.diagnosis` (`window_features`,
`FaultClassifier`, `explain_classifier`), `sensorlab.rul` (`QuantileRUL`, `build_rul_targets`),
`sensorlab.decision` (`cost_curve`, `optimal_threshold`), `sensorlab.monitoring`
(`DriftMonitor`). Notebooks `01` – `06` walk through each layer and then the pipeline.

---

## 6. Reading the outputs

### The `action` column (what the operator sees)

| Action | Meaning | Rule |
|---|---|---|
| `wait` | nothing to do | no confirmed alarm |
| `investigate` | the detector fired but the classifier reads *Normal* — a novel or unknown pattern | alarm ∧ `fault_id_pred == 0` |
| `schedule_maintenance` | fault identified, time to plan | alarm ∧ fault ∧ `rul_p50_min > intervene_horizon_min` |
| `intervene_now` | fault identified, horizon exceeded | alarm ∧ fault ∧ `rul_p50_min ≤ intervene_horizon_min` |

An **alarm is confirmed** only after `consecutive` samples (default 3, i.e. 9 minutes) in a row
above the operating threshold within the same run. `above_threshold` is the raw exceedance.

### Two thresholds per detector

- **FAR threshold** — the score above which 1 % (`far_target`) of validation-normal samples
  fall. Used for the diagnostic benchmark (AUROC, TPR, observed FAR, delay).
- **Operating threshold** — the one that minimised expected cost on the validation runs under
  the configured cost model. Used for `alarm` and `action`. With the reference cost mix
  (missed fault 50× a false alarm) it sits well below the FAR threshold, on purpose.

### RUL

`rul_p50_min` is the median estimate of minutes until the end of the fault episode;
`rul_p10_min` / `rul_p90_min` bound an 80 % interval that has been conformally calibrated on
validation. Treat it as a ranking of urgency; on the synthetic benchmark the target is time until
the end of the run, which is a function of time since onset, not a physical time-to-failure.

### Drift

`check_drift` compares each sensor's distribution in the batch with the training-normal
reference (PSI). Thresholds are set per sensor above the plant's own run-to-run noise floor
(measured leave-one-run-out at fit time). `stable` → fine; `watch` → investigate;
`alert` → do not trust alarms driven by those sensors until retrained. A batch that contains a
real fault will also show drift, so check on known-normal periods.

---

## 7. Reading the dashboard

Open the published page or `sensorlab serve`. Sections, in order:

| # | Section | Use it to |
|---|---|---|
| — | Masthead | see model version, fit date, primary detector, health (`ok` = all gates pass) and drift on the reference batch at a glance |
| 01 | Overview | the six numbers that matter on the held-out runs; the grey line quotes the published multi-seed results for comparison |
| 02 | Acceptance criteria | the runbook's release gates, PASS/FAIL, with the observed value and the limit |
| 03 | Ownership | who decides each knob and the value in force; thresholds per detector |
| 04 | Detection benchmark | compare detectors; *FAR observed* vs the 1 % target is the calibration check |
| 05 | Decision layer | type your own cost mix; the curve re-prices instantly (client-side in the static build). Solid line = oracle optimum on test, dotted = the threshold the model shipped with. The gap is the honest cost of choosing on validation |
| 06 | Held-out runs | one row per test run with outcome; **click a row** to open the inspector: raw traces, primary score vs operating threshold, other detectors normalised to their FAR threshold, RUL band, and the action timeline. Edit the sensor list to plot other channels |
| 07 | Diagnosis | accuracy / balanced accuracy / macro-F1 and the three driver sensors per fault (SHAP) |
| 08 | Drift | PSI per sensor with watch (dashed) and alert (solid) marks; pick a sensor and drag the offset to simulate a recalibration |
| 09 | Score a batch | live build: upload a CSV/parquet and download the scored file; static build: the CLI commands to do the same |
| 10 | Audit log | live build: every scoring call made against this process |

The page is light by default; the footer button switches to a dark theme and remembers the
choice in your browser.

---

## 8. Typical workflows

**Change the economics and re-ship.** Costs are inputs owned by the business, not
hyper-parameters. Edit them at train time and the operating threshold is re-derived on
validation:

```bash
sensorlab train --false-alarm-cost 250 --missed-fault-cost 20000 --delay-cost-per-min 30
sensorlab export-site      # or serve
```

**Compare seeds before believing a number.** `sensorlab evaluate --seeds 0 1 2 3 4` and read
the `std` next to each `mean` in `artifacts/results.json`.

**Try a different primary detector.** `--primary-detector IForest` (default `auto` picks the
cheapest on validation).

**Run on the real benchmark.**

```bash
pip install -e ".[real]"
make download-tep              # ~5 GB from Harvard Dataverse into data/raw/tep/
python scripts/prepare_tep.py  # RData -> data/processed/tep/*.parquet
sensorlab train --data real
```

**Regenerate the notebooks** after changing the library: `make notebooks` (builds from
`scripts/build_notebooks.py` and executes them).

**Deploy the live API.** `make docker && docker run -p 8000:8000 sensorlab`, or
`sensorlab serve` behind your reverse proxy. `/api/health` returns `degraded` when an
acceptance gate fails — point your monitoring at it.

---

## 9. Bringing your own plant

Everything downstream is source-agnostic; only the loader knows about the data. To onboard a
new source, return a `TEPDataset`:

```python
from sensorlab.data import TEPDataset

ds = TEPDataset(
    X=sensor_matrix,             # (n_samples, n_sensors) float, no NaN
    fault_id=run_scenario_per_sample,   # 0 = normal, 1..K = fault scenario of the run
    is_anomaly=fault_active_per_sample, # False before the onset, True after
    run_id=run_index_per_sample,        # contiguous 0..n_runs-1, samples of a run contiguous
    run_onsets=onset_index_per_run,     # -1 for normal runs
    run_fault_id=scenario_per_run,
    sensor_names=[...],
    fault_names=["Normal", "F01_...", ...],
    sample_minutes=3.0,
)
ds.validate()   # raises with a precise message if the contract is broken
```

Then `train_val_test_split_by_run(ds)` and `MonitoringPipeline(...).fit(ds, train, val)` as in §5.
`dataset_from_rieth_frame` in `sensorlab/data/loader.py` is a worked example that handles the
real benchmark's quirks (case of column names, non-unique run ids, different onsets in train and
test files).

Things to decide up front: the sample interval, how many normal runs you have (the detectors
and the drift reference train on those only), and the cost mix.

---

## 10. FAQ

**The primary detector on test is not the cheapest one.** It was chosen on validation. With the
reference cost mix the three detectors are within seed noise of each other; that is the honest
answer, and section 05 lets you see it.

**Every normal test run has a false alarm.** At a 50 : 1 missed-to-false-alarm cost ratio the
operating threshold is deliberately low. Raise the false-alarm cost (or lower the missed-fault
cost) and retrain.

**`investigate` on a run that is clearly faulty.** The detector fired before the classifier could
recognise the pattern (common right after the onset, or on novel faults). It is the honest
state: "something is off, we do not know what yet".

**Drift says `alert` on a batch I know is normal.** Check the batch is long enough (a few runs)
and from the same sensor layout. Single short runs of autocorrelated process data are noisy;
the thresholds are calibrated for the run length used at training.

**Where do the README numbers come from?** `artifacts/results.json`, written by
`sensorlab evaluate --seeds 0 1 2`. The dashboard reads the same file.

**Can I skip the LSTM?** `--detectors PCA-T2Q IForest`. Training then takes a couple of seconds.

**Something is slow on macOS.** Thread caps are applied there to avoid a libomp clash between
PyTorch and XGBoost. Linux uses all cores.
