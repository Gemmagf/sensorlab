# Changelog

## 0.2.0 — deployable pipeline, honest evaluation

### Fixed
- **Diagnosis labels.** The classifier was trained on the run-level scenario id, which labels
  the nominal pre-onset quarter of every faulty run as a fault. It now learns
  `TEPDataset.active_fault_id` (0 until the onset). Test accuracy on the reference
  configuration moved from 0.57 to ~0.70 with no other change.
- **LSTM-AE in the decision layer.** Window scores were sub-sampled before the cost sweep
  while onsets stayed in original sample units, shifting every onset by ~180 min and
  halving the delay unit. All window-level outputs now go through one alignment helper
  (`windows_to_per_sample` / `window_index_per_sample`).
- **Real-data loader.** Rieth column names are lower-case (`xmeas_1`), `simulationRun`
  restarts for every fault and file, and faults start at sample 20 (train) or 160 (test).
  The loader now matches columns case-insensitively, builds unique run ids from
  `(split, faultNumber, simulationRun)` and applies the right onset. Covered by tests.
- **Streamlit cache.** Changing any dataset slider other than the seed crashed the app
  (splits cached on the seed only). Every cached step is keyed on all of its inputs.
- **Evaluation leakage / inconsistency.** Detection delay and decision costs were computed on
  train+val+test; AUROC on test only. Every reported number is now on held-out test runs, with
  thresholds fixed on validation. The README's "48 test runs" was 24.
- `TEPSpec` now describes the real benchmark (41 XMEAS + 11 XMV); the synthetic 33-channel
  layout lives in `SyntheticTEPConfig`.
- OpenMP thread caps are applied on macOS only; Linux/CI use all cores.

### Added
- `sensorlab.pipeline.MonitoringPipeline` — scaler + detectors + classifier + RUL + thresholds +
  cost model + drift monitor in one object with `fit / predict / evaluate / save / load`.
  `predict` returns per-sample scores, a *confirmed* alarm, the diagnosed fault, an RUL interval
  and an action (`wait` / `investigate` / `schedule_maintenance` / `intervene_now`).
- `sensorlab` CLI: `train`, `evaluate --seeds`, `score --input batch.csv --drift`, `info`.
- `sensorlab.evaluation` — leakage-guarded detector/decision evaluation and multi-seed aggregation.
- `sensorlab.monitoring.DriftMonitor` — per-sensor PSI against training-normal marginals, with
  thresholds self-calibrated above the leave-one-run-out noise floor.
- Conformal (CQR) calibration of the RUL interval on validation data; raw and calibrated coverage
  are both reported. RUL head switched to `HistGradientBoostingRegressor` (40× faster fit).
- `TEPDataset.validate()`, `active_fault_id`, `run_mask()`; `dataset_from_rieth_frame()`.
- Notebook 06 (end-to-end pipeline), Streamlit **Operate** tab (per-run outcomes, scored CSV
  download, drift simulation, manifest), CI smoke test of the CLI, `docs/runbook.md`.
- 102 tests (was 65).

## 0.1.0

Initial release.
