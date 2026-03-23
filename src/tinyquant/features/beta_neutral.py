"""Beta-neutralization: R_i = alpha + beta * R_bench + epsilon — use epsilon."""

from __future__ import annotations

import numpy as np
from scipy import stats


def _winsorize(x: np.ndarray, n_std: float) -> np.ndarray:
    if n_std <= 0 or len(x) < 2:
        return x
    m, sd = np.nanmean(x), np.nanstd(x)
    if sd == 0 or np.isnan(sd):
        return x
    lo, hi = m - n_std * sd, m + n_std * sd
    return np.clip(x, lo, hi)


def beta_neutral_residuals(
    alt_returns: np.ndarray,
    bench_returns: np.ndarray,
    *,
    winsorize_std: float = 4.0,
) -> tuple[float, float, np.ndarray]:
    """
    OLS of alt on benchmark on overlapping valid rows.
    Returns (alpha, beta, epsilon) same length as inputs (NaN where invalid).
    """
    a = np.asarray(alt_returns, dtype=np.float64).ravel()
    b = np.asarray(bench_returns, dtype=np.float64).ravel()
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    eps = np.full(n, np.nan, dtype=np.float64)

    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 10:
        return float("nan"), float("nan"), eps

    y = _winsorize(a[mask], winsorize_std)
    x = _winsorize(b[mask], winsorize_std)
    slope, intercept, _, _, _ = stats.linregress(x, y)
    pred = intercept + slope * b
    eps = a - pred
    return float(intercept), float(slope), eps


def epsilon_matrix(
    returns_df: np.ndarray,
    bench_col: int = 0,
    *,
    winsorize_std: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    returns_df: (T, N) columns are assets; bench_col indexes benchmark in same matrix.
    Returns eps (T,N), betas (N,), alphas (N,).
    """
    r = np.asarray(returns_df, dtype=np.float64)
    t, n_assets = r.shape
    bench = r[:, bench_col]
    eps = np.full_like(r, np.nan)
    betas = np.full(n_assets, np.nan)
    alphas = np.full(n_assets, np.nan)
    for j in range(n_assets):
        if j == bench_col:
            eps[:, j] = 0.0
            betas[j] = 1.0
            alphas[j] = 0.0
            continue
        alpha, beta, e = beta_neutral_residuals(r[:, j], bench, winsorize_std=winsorize_std)
        alphas[j], betas[j] = alpha, beta
        eps[:, j] = e
    return eps, betas, alphas
