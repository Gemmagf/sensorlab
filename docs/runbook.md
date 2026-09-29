# Operations runbook — sensorlab in a plant

This is the document the plant team gets with the model. It answers the questions that
come up *after* the demo: what exactly runs, who owns which number, what to do when the
model looks wrong, and when to retrain.

## 1. What ships

| Artefact | Produced by | Contains |
|---|---|---|
| `models/pipeline.joblib` | `sensorlab train` | scaler, 3 detectors, fault classifier, RUL head (+ conformal margin), thresholds, cost model, drift reference |
| `models/pipeline.manifest.json` | `sensorlab train` | version, config, sensor layout, thresholds, fit report — the model card |
| `artifacts/results.json` | `sensorlab evaluate --seeds …` | held-out metrics per seed and mean ± std — the evidence behind the README |

Loading a pipeline saved by a different `sensorlab` version logs a warning; treat a
major/minor mismatch as "retrain".

## 2. Scoring a batch

```bash
sensorlab score --model models/pipeline.joblib --input batch.csv --output scored.csv --drift
```

Input: one row per sample, the training sensor columns (`XMEAS(i)` / `XMV(j)`; Rieth-style
`xmeas_i` is accepted), optional `run_id` (a stream/batch identifier — samples of one run must
be contiguous and in time order). Missing sensors are a hard error (exit code 2): the model
was trained on a fixed layout and will not guess.

Output columns and how to read them:

| Column | Meaning |
|---|---|
| `score_<detector>` | per-sample anomaly score of each detector (higher = more anomalous) |
| `primary_score`, `threshold` | the score and operating threshold of the primary detector |
| `above_threshold` | raw exceedance |
| `alarm` | **confirmed** alarm: `consecutive` (default 3) samples in a row above threshold within the run — the same rule the reported metrics use |
| `fault_id_pred`, `fault_name_pred`, `fault_confidence` | diagnosed fault (0 = Normal) and class probability |
| `rul_p10_min`, `rul_p50_min`, `rul_p90_min` | 80 % remaining-useful-life interval, conformally calibrated |
| `action` | `wait` · `investigate` · `schedule_maintenance` · `intervene_now` |

Action rules (all deterministic, all in `MonitoringPipeline.predict`):

```
no confirmed alarm                              -> wait
alarm and diagnosed fault == Normal             -> investigate      (detector fired, classifier disagrees)
alarm, fault, RUL p50 >  intervene_horizon_min  -> schedule_maintenance
alarm, fault, RUL p50 <= intervene_horizon_min  -> intervene_now
```

## 3. Who owns which number

| Knob | Owner | Where |
|---|---|---|
| False-alarm / missed-fault / delay cost | operations + finance | `--false-alarm-cost`, `--missed-fault-cost`, `--delay-cost-per-min` at train time |
| Intervention horizon (minutes) | maintenance planning | `--intervene-horizon-min` |
| FAR target for the diagnostic thresholds | process engineering | `--far-target` |
| Detector choice | model owner (default `auto` = cheapest on validation) | `--primary-detector` |
| Everything else (window, epochs, trees) | model owner | `PipelineConfig` |

Changing a cost is a **retrain** (`sensorlab train`), not a config edit on the saved model:
the operating threshold is re-derived on the validation runs and the new manifest records it.

## 4. When the model looks wrong

Work through these in order.

1. **Drift first.** `sensorlab score --drift` reports a per-sensor Population Stability Index
   against the training-normal distribution. Process data is autocorrelated, so a batch of a
   few runs sits well above the textbook PSI cut-offs (0.10 / 0.25) while being normal; the
   monitor measures that noise floor at training time (leave-one-run-out) and sets per-sensor
   thresholds above it. The thresholds in force are in the report (`alert_thresholds`) and in
   the manifest.
   * `stable`: every sensor is within its run-to-run noise floor.
   * `watch`: a sensor is above its floor but below the alert line — investigate
     (recalibration, a new operating point, a supplier change). Alarms remain usable; document it.
   * `alert`: do not act on alarms driven by those sensors. Retrain on data that includes the
     new regime, or exclude the sensor (that is a retrain too). On the reference configuration
     a one-standard-deviation offset on a single sensor triggers `alert`; a half-standard-
     deviation offset does not.
   Note: PSI compares marginals; a change in *correlation* with unchanged marginals is
   invisible to it, and a faulty batch will also show as drift — check on known-normal periods.
2. **Alarm without fault (`investigate`).** Expected on rare/novel faults: the detectors are
   unsupervised, the classifier only knows the 21 scenarios. Log the episode; if it recurs, it
   is a new class for the next training set.
3. **Too many false alarms.** Check the observed FAR on a known-normal period against the
   manifest's `far_target`. If it is far above target and drift is `stable`, the cost mix is
   pushing the threshold down (missed faults are expensive by design). Revisit the costs with
   operations rather than editing the threshold.
4. **Late detection.** Compare the median delay on recent confirmed episodes with
   `results.json`. Slow faults (drift, valve stick) legitimately take longer; the LSTM-AE is the
   detector that reads them. `--primary-detector LSTM-AE` is a retrain-time choice.

## 5. Retraining

* **Cadence:** on drift `alert`, on any change to the cost mix, after every planned change to
  the sensor layout, and otherwise on a fixed schedule (quarterly is a reasonable default).
* **Procedure:** `sensorlab evaluate --seeds 0 1 2` on the new data first — compare to the
  previous `results.json`; then `sensorlab train` to produce the artefact; keep the old
  `pipeline.joblib` + manifest until the new one has run in shadow for a while.
* **Acceptance:** held-out AUROC and fraction detected not worse than the previous release
  beyond seed noise (the std in `results.json`), observed FAR within 2× target, RUL coverage
  within ±5 pp of nominal, no `alert` drift on the validation runs.

## 6. Known limitations to state up front

* Trained and evaluated on the synthetic generator unless `--data real` was used; numbers move
  on the Rieth 2017 release.
* RUL target is *time to end of run*: an urgency ranking, not a physical time-to-failure.
* No handling of missing values: impute upstream (the loader and `predict` reject NaNs).
* Single-plant model: no transfer across units without retraining.
