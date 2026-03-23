"""Topological features: correlation distance -> Vietoris-Rips (ripser) -> landscape vector."""

from __future__ import annotations

import numpy as np
from ripser import ripser


def correlation_distance_matrix(corr: np.ndarray) -> np.ndarray:
    """d_ij = sqrt(max(0, 2*(1 - rho_ij)))."""
    c = np.asarray(corr, dtype=np.float64)
    np.fill_diagonal(c, 1.0)
    d = 2.0 * (1.0 - np.clip(c, -1.0, 1.0))
    d = np.sqrt(np.maximum(d, 0.0))
    np.fill_diagonal(d, 0.0)
    return d


def persistence_landscape_vector(
    distance_matrix: np.ndarray,
    *,
    max_dimension: int = 1,
    max_edge_length: float = 1.5,
    resolution: int = 100,
    t_min: float = 0.0,
    t_max: float = 1.0,
) -> np.ndarray:
    """
    Binned persistence magnitude vector (approximation to persistence landscape for fixed dim).
    Concatenates H0..Hk each into resolution/(k+1) bins.
    """
    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    if n < 2:
        return np.zeros(resolution)

    res = ripser(d, distance_matrix=True, maxdim=max_dimension, thresh=max_edge_length)
    dgms = res["dgms"]
    per_dim = max(1, resolution // (max_dimension + 1))
    parts: list[np.ndarray] = []

    for dim, dgm in enumerate(dgms[: max_dimension + 1]):
        if dgm is None or len(dgm) == 0:
            parts.append(np.zeros(per_dim))
            continue
        finite = dgm[np.isfinite(dgm[:, 1])]
        if len(finite) == 0:
            parts.append(np.zeros(per_dim))
            continue
        persist = np.sort(finite[:, 1] - finite[:, 0])[-per_dim:]
        v = np.zeros(per_dim)
        v[-len(persist) :] = persist
        parts.append(v)

    vec = np.concatenate(parts)
    if len(vec) < resolution:
        vec = np.pad(vec, (0, resolution - len(vec)))
    return vec[:resolution].astype(np.float64)


def landscape_grid_features(
    distance_matrix: np.ndarray,
    **kwargs: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (vector, correlation_matrix) for downstream XGBoost."""
    c = 1.0 - 0.5 * (distance_matrix**2)
    np.fill_diagonal(c, 1.0)
    vec = persistence_landscape_vector(distance_matrix, **kwargs)
    return vec, c
