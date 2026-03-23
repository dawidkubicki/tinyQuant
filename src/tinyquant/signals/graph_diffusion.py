"""Graph from epsilon correlations + Laplacian diffusion mispricing."""

from __future__ import annotations

import numpy as np


def _normalized_laplacian(adj: np.ndarray) -> np.ndarray:
    """Symmetric normalized L = I - D^{-1/2} A D^{-1/2}."""
    a = np.asarray(adj, dtype=np.float64)
    np.fill_diagonal(a, 0.0)
    d = a.sum(axis=1)
    d_inv_sqrt = np.zeros_like(d)
    mask = d > 1e-12
    d_inv_sqrt[mask] = 1.0 / np.sqrt(d[mask])
    d_mat = np.diag(d_inv_sqrt)
    return np.eye(a.shape[0]) - d_mat @ a @ d_mat


def diffusion_mispricing_scores(
    epsilon_window: np.ndarray,
    *,
    edge_threshold: float = 0.25,
    diffusion_steps: int = 3,
    diffusion_alpha: float = 0.3,
    score_clip: float = 3.0,
) -> np.ndarray:
    """
    epsilon_window: (T, N) idiosyncratic returns.
    Build |corr| graph, diffuse last-row return vector u.
    mispricing = u_actual - u_diffused (positive => underperformed vs neighbors => long tilt).
    """
    z = np.asarray(epsilon_window, dtype=np.float64)
    if z.size == 0:
        return np.array([])
    t, n = z.shape
    if t < 5 or n < 2:
        return np.zeros(n)

    c = np.corrcoef(z.T)
    np.fill_diagonal(c, 0.0)
    adj = np.abs(c)
    adj[adj < edge_threshold] = 0.0
    np.fill_diagonal(adj, 0.0)

    if adj.sum() < 1e-12:
        return np.zeros(n)

    l = _normalized_laplacian(adj)
    u = z[-1].copy()
    u[~np.isfinite(u)] = 0.0
    for _ in range(diffusion_steps):
        u = u - diffusion_alpha * (l @ u)
    actual = z[-1].copy()
    actual[~np.isfinite(actual)] = 0.0
    mis = actual - u
    mis = np.clip(mis, -score_clip, score_clip)
    return mis
