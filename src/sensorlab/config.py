"""Project paths and the Tennessee Eastman process specification."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


@dataclass(frozen=True)
class TEPSpec:
    """Tennessee Eastman process specification.

    Channel count and fault catalogue follow the Downs & Vogel (1993) /
    Bathelt et al. (2015) conventions used throughout the literature.
    """

    n_xmeas: int = 22
    n_xmv: int = 11
    n_total: int = 33
    n_fault_types: int = 21
    sample_minutes: float = 3.0
    nominal_steady_minutes: int = 60
    fault_names: tuple[str, ...] = field(
        default_factory=lambda: (
            "Normal",
            "F01_A/C_feed_ratio",
            "F02_B_composition",
            "F03_D_feed_temp",
            "F04_reactor_cooling_inlet",
            "F05_condenser_cooling_inlet",
            "F06_A_feed_loss",
            "F07_C_header_pressure_loss",
            "F08_A_B_C_feed_random",
            "F09_D_feed_temp_random",
            "F10_C_feed_temp_random",
            "F11_reactor_cooling_random",
            "F12_condenser_cooling_random",
            "F13_reaction_kinetics_slow_drift",
            "F14_reactor_cooling_valve_stick",
            "F15_condenser_cooling_valve_stick",
            "F16_unknown",
            "F17_unknown",
            "F18_unknown",
            "F19_unknown",
            "F20_unknown",
            "F21_valve_position_const",
        )
    )


TEP = TEPSpec()


def ensure_dirs() -> None:
    """Create all standard project directories if missing."""
    for d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, ARTIFACTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
