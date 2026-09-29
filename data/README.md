# Data

This directory holds raw and processed datasets. The contents of `raw/`,
`processed/`, `interim/` and `external/` are **gitignored**.

## Tennessee Eastman Process (TEP)

The pipeline supports two data sources, controlled by `sensorlab.data.loader`:

### 1. Synthetic TEP-like generator (default, reproducible)

A deterministic multivariate process simulator that produces sensor traces with
the same broad statistical structure as the TEP benchmark (33 variables,
multi-modal noise, drifts, step changes, oscillations). No download required.

```python
from sensorlab.data import generate_synthetic_dataset
ds = generate_synthetic_dataset(seed=0)
```

### 2. Real TEP dataset (Rieth et al. 2017 / Bathelt et al. 2015)

Run

```bash
make download-tep
```

This invokes [scripts/download_tep.py](../scripts/download_tep.py), which fetches
the [Rieth et al. 2017](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1)
release (~5 GB) into `data/raw/tep/`. Convert with

```bash
pip install -e ".[real]"      # pyreadr + pyarrow
python scripts/prepare_tep.py  # -> data/processed/tep/*.parquet (+ a `split` column)
sensorlab train --data real
```

Conventions the loader applies (see `sensorlab.data.loader.dataset_from_rieth_frame`):

| Rieth column      | Meaning in sensorlab                                                      |
|-------------------|---------------------------------------------------------------------------|
| `faultNumber`     | run scenario 0..21 → `fault_id` / `run_fault_id`                          |
| `simulationRun`   | restarts per fault and per file → combined into a unique `run_id`         |
| `sample`          | 1-based sample index; faults start at 20 (train files) or 160 (test files)|
| `xmeas_i`, `xmv_j`| 41 + 11 channels, matched case-insensitively, renamed `XMEAS(i)`/`XMV(j)` |

### 3. Your own plant

Onboarding a new data source means returning a `TEPDataset` (see its docstring for the
contract: a 2-D sensor matrix, per-sample run ids, per-run onsets and fault labels).
`TEPDataset.validate()` checks the invariants; everything downstream — detectors,
diagnosis, RUL, decision layer, CLI and dashboard — is source-agnostic.
