"""Platform compatibility shims — imported first thing from ``sensorlab/__init__``.

The single concern this module solves: on macOS arm64 PyTorch and XGBoost
both ship their own libomp; loading them in the wrong order causes XGBoost
to segfault when it builds a DMatrix. Forcing xgboost to load first (so its
libomp "wins"), combined with the duplicate-tolerance env var, makes the two
coexist reliably.

We import xgboost and torch eagerly here purely for their side-effects — the
imported symbols are not used.
"""

from __future__ import annotations

import os
import platform

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Single OpenMP thread eliminates the cross-library threading races that
# cause both PyTorch LSTM and XGBoost to crash unpredictably on macOS arm64.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

if platform.system() == "Darwin":
    # Order matters: xgboost MUST load before torch on macOS so its libomp wins.
    # Importing via importlib keeps the import sorter from reordering this block.
    import importlib

    importlib.import_module("xgboost")  # must precede torch
    torch = importlib.import_module("torch")

    # PyTorch picks up its own thread count via the C++ API; lock it down too.
    torch.set_num_threads(1)
