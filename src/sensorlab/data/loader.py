"""Unified dataset interface (synthetic or real TEP).

The :class:`TEPDataset` container is the *contract* every data source has to
satisfy. Onboarding a new plant means writing one function that returns a
``TEPDataset`` — everything downstream (detectors, diagnosis, RUL, decision
layer, CLI, dashboard) is source-agnostic.
"""

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
    fault_id : (n_samples,) int array — the *scenario* of the run each sample
        belongs to (0=normal, 1..21=fault). Note that samples **before** the
        onset of a faulty run still carry the run's fault id; use
        :attr:`active_fault_id` for the label that is actually true at time t.
    is_anomaly : (n_samples,) bool — True at and after the fault onset
    run_id : (n_samples,) int — contiguous 0..n_runs-1; keeps windows and
        splits run-local
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

    def __post_init__(self) -> None:
        self.validate()

    # ----- invariants --------------------------------------------------------

    def validate(self) -> None:
        """Raise ``ValueError`` if the container violates its contract."""
        n = self.X.shape[0]
        for name in ("fault_id", "is_anomaly", "run_id"):
            arr = getattr(self, name)
            if arr.shape[0] != n:
                raise ValueError(f"{name} has {arr.shape[0]} rows, X has {n}")
        if self.X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {self.X.shape}")
        if len(self.sensor_names) != self.X.shape[1]:
            raise ValueError(
                f"{len(self.sensor_names)} sensor names for {self.X.shape[1]} columns"
            )
        n_runs = self.run_fault_id.shape[0]
        if self.run_onsets.shape[0] != n_runs:
            raise ValueError("run_onsets and run_fault_id must have the same length")
        if n and (self.run_id.min() < 0 or self.run_id.max() >= n_runs):
            raise ValueError("run_id values must index into run_fault_id / run_onsets")
        if not np.isfinite(self.X).all():
            raise ValueError("X contains NaN or inf — impute or drop before building a dataset")

    # ----- derived views ------------------------------------------------------

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_sensors(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_runs(self) -> int:
        return int(self.run_fault_id.shape[0])

    @property
    def active_fault_id(self) -> np.ndarray:
        """Fault label that is true *at each sample*: 0 until the onset, then the fault id.

        This is the target for diagnosis. Training a classifier on
        :attr:`fault_id` instead would ask it to separate nominal pre-onset
        samples of a faulty run from nominal samples of a normal run — an
        impossible task that silently caps accuracy.
        """
        return np.where(self.is_anomaly, self.fault_id, 0).astype(self.fault_id.dtype)

    def run_mask(self, run_ids) -> np.ndarray:
        """Boolean sample mask selecting whole runs."""
        return np.isin(self.run_id, np.asarray(list(run_ids)))

    def to_dataframe(self) -> pd.DataFrame:
        """Long-form DataFrame for plotting and EDA."""
        df = pd.DataFrame(self.X, columns=self.sensor_names)
        df["fault_id"] = self.fault_id
        df["fault_name"] = [self.fault_names[i] for i in self.fault_id]
        df["active_fault_id"] = self.active_fault_id
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


# ---------------------------------------------------------------------------
# Real Rieth et al. 2017 release
# ---------------------------------------------------------------------------


def _infer_split(name: str) -> str:
    low = name.lower()
    if "test" in low:
        return "test"
    if "train" in low:
        return "train"
    raise ValueError(
        f"cannot infer train/test split from file name {name!r}; "
        "add a 'split' column or name the file *Training* / *Testing*"
    )


def _load_real(root: Path | None) -> TEPDataset:
    """Load the Rieth et al. 2017 release if it has been downloaded.

    Run ``make download-tep`` then ``python scripts/prepare_tep.py`` first.
    Expected layout: parquet files in ``data/processed/tep/`` with the Rieth
    columns ``faultNumber``, ``simulationRun``, ``sample``, ``xmeas_1..41``,
    ``xmv_1..11`` (column names are matched case-insensitively) and,
    optionally, ``split`` ∈ {train, test} — inferred from the file name when
    absent.

    Rieth conventions handled here:

    * ``simulationRun`` restarts at 1 for every ``faultNumber`` and for each
      of the train/test files, so a globally unique run id is built from the
      triple ``(split, faultNumber, simulationRun)``.
    * Faults are injected after sample 20 in the training files and after
      sample 160 in the testing files.
    """
    from sensorlab.config import PROCESSED_DIR

    root = root or PROCESSED_DIR / "tep"
    if not root.exists():
        raise FileNotFoundError(
            f"Real TEP data not found at {root}. Run `make download-tep` first."
        )

    parts: list[pd.DataFrame] = []
    for fp in sorted(root.glob("*.parquet")):
        df = pd.read_parquet(fp)
        df.columns = [str(c) for c in df.columns]
        if "split" not in df.columns:
            df["split"] = _infer_split(fp.name)
        parts.append(df)
    if not parts:
        raise FileNotFoundError(f"No parquet files in {root}")
    df = pd.concat(parts, ignore_index=True)
    return dataset_from_rieth_frame(df)


def dataset_from_rieth_frame(df: pd.DataFrame) -> TEPDataset:
    """Build a :class:`TEPDataset` from a Rieth-layout DataFrame (any case of column names)."""
    lower = {c.lower(): c for c in df.columns}
    required = ("faultnumber", "simulationrun", "sample", "split")
    missing = [c for c in required if c not in lower]
    if missing:
        raise ValueError(f"Rieth frame is missing columns: {missing}")

    sensor_cols = [
        lower[c] for c in lower if c.startswith(("xmeas", "xmv")) and c.split("_")[-1].isdigit()
    ]
    sensor_cols = sorted(sensor_cols, key=_sensor_sort_key)
    if not sensor_cols:
        raise ValueError("no xmeas_*/xmv_* sensor columns found")

    # Sort so that samples inside each run are in temporal order.
    df = df.sort_values(
        [lower["split"], lower["faultnumber"], lower["simulationrun"], lower["sample"]]
    ).reset_index(drop=True)

    fault_id = df[lower["faultnumber"]].to_numpy(dtype=np.int16)
    split = df[lower["split"]].astype(str).str.lower().to_numpy()
    sim_run = df[lower["simulationrun"]].to_numpy(dtype=np.int64)
    sample = df[lower["sample"]].to_numpy(dtype=np.int64)

    # Globally unique, contiguous run ids from (split, fault, simulationRun).
    key = pd.MultiIndex.from_arrays([split, fault_id, sim_run])
    run_id, uniques = pd.factorize(key, sort=True)
    run_id = run_id.astype(np.int32)
    n_runs = len(uniques)

    run_fault = np.zeros(n_runs, dtype=np.int16)
    run_onsets = np.full(n_runs, -1, dtype=np.int32)
    for rid, (spl, fid, _) in enumerate(uniques):
        run_fault[rid] = fid
        if fid > 0:
            run_onsets[rid] = (
                TEP.fault_onset_sample_test if spl == "test" else TEP.fault_onset_sample_train
            )

    # Rieth 'sample' is 1-based; onset index is 0-based within the run.
    onset_per_sample = run_onsets[run_id]
    is_anom = (fault_id > 0) & ((sample - 1) >= onset_per_sample)

    X = df[sensor_cols].to_numpy(dtype=np.float32)
    return TEPDataset(
        X=X,
        fault_id=fault_id,
        is_anomaly=is_anom,
        run_id=run_id,
        run_onsets=run_onsets,
        run_fault_id=run_fault,
        sensor_names=[_canonical_sensor_name(c) for c in sensor_cols],
        fault_names=list(TEP.fault_names),
        sample_minutes=TEP.sample_minutes,
    )


def _sensor_sort_key(col: str) -> tuple[int, int]:
    low = col.lower()
    group = 0 if low.startswith("xmeas") else 1
    return group, int(low.split("_")[-1])


def _canonical_sensor_name(col: str) -> str:
    """``xmeas_7`` → ``XMEAS(7)`` so real and synthetic names share one convention."""
    low = col.lower()
    prefix = "XMEAS" if low.startswith("xmeas") else "XMV"
    return f"{prefix}({int(low.split('_')[-1])})"
