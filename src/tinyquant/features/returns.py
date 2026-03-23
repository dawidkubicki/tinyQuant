"""Return series utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns_from_close(close: pd.Series | np.ndarray) -> np.ndarray:
    """Log returns; first element is NaN (dropped caller-side if needed)."""
    s = pd.Series(close, dtype="float64")
    return np.log(s / s.shift(1)).to_numpy(dtype=np.float64)
