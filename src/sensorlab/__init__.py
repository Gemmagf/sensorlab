"""sensorlab — fault detection, diagnosis & RUL on the Tennessee Eastman benchmark."""

# Resolves the macOS arm64 libomp clash between PyTorch and XGBoost. Must
# happen before either is imported transitively.
from sensorlab import _compat  # noqa: F401, I001

from sensorlab.config import TEP, ARTIFACTS_DIR, DATA_DIR, MODELS_DIR, PROJECT_ROOT

__version__ = "0.1.0"
__all__ = ["ARTIFACTS_DIR", "DATA_DIR", "MODELS_DIR", "PROJECT_ROOT", "TEP", "__version__"]
