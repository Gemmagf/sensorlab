"""Reproducible TEP-like multivariate process simulator.

Approach: a small number of *latent* process factors evolve as AR(1) processes
and are projected into the sensor space through a sparse mixing matrix. Faults
perturb specific latent factors with characteristic dynamics (step, ramp, drift,
oscillation, valve-stick, noise burst, intermittent). The resulting traces
share the cross-sensor correlation structure and the per-fault signatures of
the real Tennessee Eastman benchmark while remaining fully deterministic given
a seed — so the test suite and CI can run end-to-end without large downloads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from sensorlab.config import TEP

FaultKind = Literal[
    "step", "ramp", "drift", "sine", "stick", "noise_up", "intermittent"
]


@dataclass(frozen=True)
class FaultRecipe:
    """How a fault perturbs the latent process."""

    kind: FaultKind
    latent_index: int
    magnitude: float
    onset_fraction: float = 0.25


# 21 fault scenarios mapped onto 7 mechanics (mirrors the real TEP catalogue,
# where multiple fault numbers share the same underlying mechanism applied to
# different streams). Magnitudes are tuned so detectors have a wide spread of
# difficulties — clear step changes are near-perfect, noise/intermittent faults
# remain genuinely hard, matching the published TEP detection profile.
FAULT_CATALOGUE: tuple[FaultRecipe, ...] = (
    FaultRecipe("step", 0, 5.0),
    FaultRecipe("step", 1, 4.5),
    FaultRecipe("step", 2, 4.0),
    FaultRecipe("step", 3, 4.8),
    FaultRecipe("step", 4, 4.2),
    FaultRecipe("step", 0, -5.5),
    FaultRecipe("step", 1, -5.0),
    FaultRecipe("noise_up", 0, 2.5),
    FaultRecipe("noise_up", 2, 2.8),
    FaultRecipe("noise_up", 1, 2.4),
    FaultRecipe("noise_up", 3, 2.2),
    FaultRecipe("noise_up", 4, 2.6),
    FaultRecipe("drift", 0, 4.5),
    FaultRecipe("stick", 3, 5.0),
    FaultRecipe("stick", 4, 4.5),
    FaultRecipe("sine", 0, 3.5),
    FaultRecipe("sine", 1, 3.8),
    FaultRecipe("intermittent", 2, 3.2),
    FaultRecipe("intermittent", 3, 3.0),
    FaultRecipe("ramp", 4, 3.5),
    FaultRecipe("stick", 2, 4.2),
)


@dataclass
class SyntheticTEPConfig:
    """Parameters controlling the synthetic generator."""

    n_sensors: int = TEP.n_total
    n_latent: int = 6
    fault_run_minutes: int = 480
    sample_minutes: float = TEP.sample_minutes
    ar_coef: float = 0.92
    sensor_noise: float = 0.08
    seed: int = 0
    n_normal_runs: int = 10
    n_runs_per_fault: int = 3
    sensor_names: tuple[str, ...] = field(default_factory=tuple)


def _build_mixing_matrix(
    n_sensors: int, n_latent: int, rng: np.random.Generator
) -> np.ndarray:
    """Realistic sparse loading matrix: each latent affects ~30% of sensors strongly."""
    W = rng.standard_normal((n_sensors, n_latent)) * 0.6
    for j in range(n_latent):
        strong = rng.random(n_sensors) > 0.7
        W[strong, j] += rng.standard_normal(int(strong.sum())) * 1.4
    return W


def _ar1(n: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    """Stationary AR(1) sample of length n."""
    eps = rng.standard_normal(n) * np.sqrt(1.0 - rho**2)
    z = np.empty(n)
    z[0] = rng.standard_normal()
    for t in range(1, n):
        z[t] = rho * z[t - 1] + eps[t]
    return z


def _apply_fault(
    Z: np.ndarray, recipe: FaultRecipe, onset: int, rng: np.random.Generator
) -> np.ndarray:
    """Add the fault signature to the latent series in-place (returns new array)."""
    Z = Z.copy()
    idx = recipe.latent_index
    n = Z.shape[0] - onset
    t = np.arange(n)

    if recipe.kind == "step":
        Z[onset:, idx] += recipe.magnitude
    elif recipe.kind == "ramp":
        Z[onset:, idx] += recipe.magnitude * (t / max(n - 1, 1))
    elif recipe.kind == "drift":
        # Saturating log-drift
        scale = np.log1p(n / 50.0)
        Z[onset:, idx] += recipe.magnitude * np.log1p(t / 50.0) / max(scale, 1e-9)
    elif recipe.kind == "sine":
        Z[onset:, idx] += recipe.magnitude * np.sin(2 * np.pi * t / 40.0)
    elif recipe.kind == "stick":
        stuck = Z[onset, idx] + recipe.magnitude
        Z[onset:, idx] = stuck
    elif recipe.kind == "noise_up":
        Z[onset:, idx] += recipe.magnitude * rng.standard_normal(n)
    elif recipe.kind == "intermittent":
        bursts = ((np.arange(n) // 20) % 2 == 0).astype(float)
        Z[onset:, idx] += recipe.magnitude * bursts
    else:  # pragma: no cover — exhaustive
        raise ValueError(f"unknown fault kind: {recipe.kind}")
    return Z


def _simulate_run(
    fault_id: int,
    cfg: SyntheticTEPConfig,
    W: np.ndarray,
    bias: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Simulate one run. Returns (X, anomaly_mask, onset_index).

    onset_index is -1 for nominal runs.
    """
    n_samples = int(cfg.fault_run_minutes / cfg.sample_minutes)

    Z = np.stack([_ar1(n_samples, cfg.ar_coef, rng) for _ in range(cfg.n_latent)], axis=1)

    anomaly_mask = np.zeros(n_samples, dtype=bool)
    onset = -1
    if fault_id > 0:
        recipe = FAULT_CATALOGUE[fault_id - 1]
        onset = int(n_samples * recipe.onset_fraction)
        anomaly_mask[onset:] = True
        Z = _apply_fault(Z, recipe, onset, rng)

    noise = rng.standard_normal((n_samples, cfg.n_sensors)) * cfg.sensor_noise
    X = Z @ W.T + bias + noise
    return X.astype(np.float32), anomaly_mask, onset


def generate_synthetic_dataset(
    cfg: SyntheticTEPConfig | None = None,
) -> "RawSimulation":
    """Generate a full synthetic dataset of nominal + faulty runs.

    Each fault id (1..21) gets ``cfg.n_runs_per_fault`` runs; nominal operation
    gets ``cfg.n_normal_runs`` runs. Runs are concatenated; ``run_id`` keeps
    them separable for windowing and splitting.
    """
    cfg = cfg or SyntheticTEPConfig()
    rng = np.random.default_rng(cfg.seed)

    W = _build_mixing_matrix(cfg.n_sensors, cfg.n_latent, rng)
    bias = rng.standard_normal(cfg.n_sensors) * 0.3

    X_chunks: list[np.ndarray] = []
    fault_id_chunks: list[np.ndarray] = []
    anomaly_chunks: list[np.ndarray] = []
    run_id_chunks: list[np.ndarray] = []
    onsets: list[int] = []
    run_fault_id: list[int] = []

    run_counter = 0

    def _emit(fault_id: int, n_runs: int) -> None:
        nonlocal run_counter
        for _ in range(n_runs):
            X, mask, onset = _simulate_run(fault_id, cfg, W, bias, rng)
            n = X.shape[0]
            X_chunks.append(X)
            fault_id_chunks.append(np.full(n, fault_id, dtype=np.int16))
            anomaly_chunks.append(mask)
            run_id_chunks.append(np.full(n, run_counter, dtype=np.int32))
            onsets.append(onset)
            run_fault_id.append(fault_id)
            run_counter += 1

    _emit(0, cfg.n_normal_runs)
    for fid in range(1, TEP.n_fault_types + 1):
        _emit(fid, cfg.n_runs_per_fault)

    X = np.concatenate(X_chunks, axis=0)
    fault_id_arr = np.concatenate(fault_id_chunks, axis=0)
    is_anom = np.concatenate(anomaly_chunks, axis=0)
    run_id_arr = np.concatenate(run_id_chunks, axis=0)

    sensor_names = cfg.sensor_names or tuple(_default_sensor_names(cfg.n_sensors))

    return RawSimulation(
        X=X,
        fault_id=fault_id_arr,
        is_anomaly=is_anom,
        run_id=run_id_arr,
        run_onsets=np.array(onsets, dtype=np.int32),
        run_fault_id=np.array(run_fault_id, dtype=np.int16),
        sensor_names=list(sensor_names),
        sample_minutes=cfg.sample_minutes,
    )


def _default_sensor_names(n: int) -> list[str]:
    n_xmeas = TEP.n_xmeas
    n_xmv = TEP.n_xmv
    base = [f"XMEAS({i + 1})" for i in range(n_xmeas)] + [
        f"XMV({i + 1})" for i in range(n_xmv)
    ]
    if n <= len(base):
        return base[:n]
    return base + [f"X({i + 1})" for i in range(len(base), n)]


@dataclass
class RawSimulation:
    """Output of the synthetic generator (intermediate form before TEPDataset)."""

    X: np.ndarray
    fault_id: np.ndarray
    is_anomaly: np.ndarray
    run_id: np.ndarray
    run_onsets: np.ndarray
    run_fault_id: np.ndarray
    sensor_names: list[str]
    sample_minutes: float
