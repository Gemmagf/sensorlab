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
python scripts/prepare_tep.py
```

The trained models and the Streamlit dashboard work identically on either source.
