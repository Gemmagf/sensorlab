"""Unified dataset interface (synthetic or real TEP)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from sensorlab.config import TEP
from sensorlab.data.synthetic import (
    SyntheticTEPConfig,
    generate_synthetic_dataset,
)

DataSource = Literal["synthetic", "real"]


@dataclass
class TEPDataset:
    """Public, source-agnostic container for TEP-like data.

    Attributes
    ----------
    X : (n_samples, n_sensors) float array
    fault_id : (n_samples,) int array — 0=normal, 1..21=fault scenario
    is_anomaly : (n_samples,) bool — True at and after the fault onset
    run_id : (n_samples,) int — used to keep windows and splits run-local
    run_onsets : (n_runs,) int — per-run onset index (-1 for nominal runs)
    run_fault_id : (n_runs,) int — per-run fault label
    sensor_names : list of str
    fault_names : list of str (22: Normal + F01..F21)
    sample_minutes : float — sample interval in minutes
    """

    X: np.ndarray
    fault_id: np.ndarray
    is_anomaly: np.ndarray
    run_id: np.ndarray
    run_onsets: np.ndarray
    run_fault_id: np.ndarray
    sensor_names: list[str]
    fault_names: list[str]
    sample_minutes: float

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_sensors(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_runs(self) -> int:
        return int(self.run_fault_id.shape[0])

    def to_dataframe(self) -> pd.DataFrame:
        """Long-form DataFrame for plotting and EDA."""
        df = pd.DataFrame(self.X, columns=self.sensor_names)
        df["fault_id"] = self.fault_id
        df["fault_name"] = [self.fault_names[i] for i in self.fault_id]
        df["is_anomaly"] = self.is_anomaly
        df["run_id"] = self.run_id
        df["t_minutes"] = self._time_axis()
        return df

    def _time_axis(self) -> np.ndarray:
        """Per-sample time in minutes, restarting at each new run."""
        t = np.zeros(self.n_samples, dtype=np.float64)
        for r in np.unique(self.run_id):
            mask = self.run_id == r
            t[mask] = np.arange(mask.sum()) * self.sample_minutes
        return t

    def runs(self):
        """Yield (run_id, fault_id, X_run, mask_run, onset)."""
        for rid in np.unique(self.run_id):
            mask = self.run_id == rid
            yield (
                int(rid),
                int(self.run_fault_id[rid]),
                self.X[mask],
                self.is_anomaly[mask],
                int(self.run_onsets[rid]),
            )


def load_dataset(
    source: DataSource = "synthetic",
    *,
    cfg: SyntheticTEPConfig | None = None,
    real_root: Path | None = None,
) -> TEPDataset:
    """Load a TEPDataset from either the synthetic generator or the Rieth release."""
    if source == "synthetic":
        sim = generate_synthetic_dataset(cfg)
        return TEPDataset(
            X=sim.X,
            fault_id=sim.fault_id,
            is_anomaly=sim.is_anomaly,
            run_id=sim.run_id,
            run_onsets=sim.run_onsets,
            run_fault_id=sim.run_fault_id,
            sensor_names=sim.sensor_names,
            fault_names=list(TEP.fault_names),
            sample_minutes=sim.sample_minutes,
        )
    if source == "real":
        return _load_real(real_root)
    raise ValueError(f"unknown source: {source!r}")


def _load_real(root: Path | None) -> TEPDataset:
    """Load the Rieth et al. 2017 release if it has been downloaded.

    Run ``make download-tep`` first. We support either the original RData
    archives processed via :mod:`scripts.prepare_tep` (yielding parquet files
    in ``data/processed/tep/``) or any compatible columnar layout.
    """
    from sensorlab.config import PROCESSED_DIR

    root = root or PROCESSED_DIR / "tep"
    if not root.exists():
        raise FileNotFoundError(
            f"Real TEP data not found at {root}. Run `make download-tep` first."
        )

    parts: list[pd.DataFrame] = []
    for fp in sorted(root.glob("*.parquet")):
        parts.append(pd.read_parquet(fp))
    if not parts:
        raise FileNotFoundError(f"No parquet files in {root}")
    df = pd.concat(parts, ignore_index=True)

    sensor_cols = [c for c in df.columns if c.startswith(("XMEAS", "XMV"))]
    X = df[sensor_cols].to_numpy(dtype=np.float32)
    fault_id = df["faultNumber"].to_numpy(dtype=np.int16)
    run_id = df["simulationRun"].to_numpy(dtype=np.int32)
    is_anom = (fault_id > 0) & (df["sample"].to_numpy() >= 20)

    run_onsets = np.full(int(run_id.max() + 1), -1, dtype=np.int32)
    run_fault = np.zeros_like(run_onsets, dtype=np.int16)
    for rid in np.unique(run_id):
        mask = run_id == rid
        run_fault[rid] = fault_id[mask][0]
        if run_fault[rid] > 0:
            run_onsets[rid] = 20  # Rieth convention

    return TEPDataset(
        X=X,
        fault_id=fault_id,
        is_anomaly=is_anom,
        run_id=run_id,
        run_onsets=run_onsets,
        run_fault_id=run_fault,
        sensor_names=sensor_cols,
        fault_names=list(TEP.fault_names),
        sample_minutes=TEP.sample_minutes,
    )
