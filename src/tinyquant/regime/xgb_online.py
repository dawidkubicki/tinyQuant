"""XGBoost regime classifier trained on GMM pseudo-labels (lazy xgboost import)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tinyquant.config.schema import XGBoostRegimeConfig


def _xgb() -> Any:
    try:
        import xgboost as xgb  # noqa: WPS433
    except Exception as e:  # pragma: no cover - platform-specific binary deps
        raise ImportError(
            "xgboost is required for this operation. On macOS install OpenMP: brew install libomp"
        ) from e
    return xgb


def train_and_save_xgb(
    X: np.ndarray,
    y: np.ndarray,
    cfg: XGBoostRegimeConfig,
    *,
    num_class: int,
) -> Any:
    xgb = _xgb()
    path = Path(cfg.model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clf = xgb.XGBClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        random_state=cfg.random_state,
        objective="multi:softprob",
        num_class=num_class,
    )
    clf.fit(X, y)
    clf.save_model(str(path))
    return clf


def load_xgb(cfg: XGBoostRegimeConfig) -> Any | None:
    path = Path(cfg.model_path)
    if not path.is_file():
        return None
    xgb = _xgb()
    clf = xgb.XGBClassifier()
    clf.load_model(str(path))
    return clf


def predict_regime_proba(
    clf: Any | None,
    x_live: np.ndarray,
    *,
    n_regimes: int,
) -> tuple[int, np.ndarray, float]:
    """
    Returns (argmax_regime, proba_vector, max_proba).
    If clf is None, uniform uncertainty.
    """
    x = np.asarray(x_live, dtype=np.float64).reshape(1, -1)
    if clf is None:
        p = np.ones(n_regimes) / n_regimes
        return int(np.argmax(p)), p, float(p.max())

    proba = clf.predict_proba(x)[0]
    k = int(np.argmax(proba))
    return k, proba.astype(np.float64), float(proba[k])
