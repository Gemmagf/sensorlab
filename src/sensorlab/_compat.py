"""Platform compatibility shims — imported first thing from ``sensorlab/__init__``.

The single concern this module solves: on macOS arm64 PyTorch and XGBoost
both ship their own libomp; loading them in the wrong order causes XGBoost
to segfault when it builds a DMatrix. Forcing xgboost to load first (so its
libomp "wins"), combined with the duplicate-tolerance env var and a single
OpenMP thread, makes the two coexist reliably.

The thread caps are applied **only on macOS**: on Linux (CI, servers,
containers) they would silently throttle XGBoost, PyTorch and scikit-learn
to one core. Set ``SENSORLAB_SINGLE_THREAD=1`` to force the caps elsewhere.
"""

from __future__ import annotations

import os
import platform

IS_DARWIN = platform.system() == "Darwin"
SINGLE_THREAD = IS_DARWIN or os.environ.get("SENSORLAB_SINGLE_THREAD", "0") == "1"

if SINGLE_THREAD:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

if IS_DARWIN:
    # Order matters: xgboost MUST load before torch on macOS so its libomp wins.
    # Importing via importlib keeps the import sorter from reordering this block.
    import importlib

    importlib.import_module("xgboost")  # must precede torch
    torch = importlib.import_module("torch")

    # PyTorch picks up its own thread count via the C++ API; lock it down too.
    torch.set_num_threads(1)


def default_n_jobs() -> int:
    """Worker count for tree ensembles: 1 where OpenMP is fragile, all cores elsewhere."""
    return 1 if SINGLE_THREAD else -1
