"""Offline GMM for unsupervised regime clustering on historical feature rows."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.mixture import GaussianMixture

from tinyquant.config.schema import GMMRegimeConfig


def fit_and_save_gmm(X: np.ndarray, cfg: GMMRegimeConfig) -> GaussianMixture:
    path = Path(cfg.model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gmm = GaussianMixture(
        n_components=cfg.n_components,
        covariance_type=cfg.covariance_type,
        random_state=cfg.random_state,
        max_iter=500,
    )
    gmm.fit(X)
    joblib.dump(gmm, path)
    return gmm


def load_gmm(cfg: GMMRegimeConfig) -> GaussianMixture | None:
    path = Path(cfg.model_path)
    if not path.is_file():
        return None
    return joblib.load(path)
