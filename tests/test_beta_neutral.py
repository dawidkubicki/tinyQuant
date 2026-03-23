from __future__ import annotations

import numpy as np

from tinyquant.features.beta_neutral import beta_neutral_residuals, epsilon_matrix


def test_beta_neutral_residuals_orthogonal_noise() -> None:
    rng = np.random.default_rng(0)
    b = rng.normal(size=500)
    a = 0.3 * b + rng.normal(scale=0.1, size=500)
    alpha, beta, eps = beta_neutral_residuals(a, b)
    assert abs(beta - 0.3) < 0.05
    m = np.isfinite(eps) & np.isfinite(b)
    assert np.corrcoef(eps[m], b[m])[0, 1] ** 2 < 0.02


def test_epsilon_matrix_benchmark_column() -> None:
    rng = np.random.default_rng(1)
    t = 200
    bench = rng.normal(0, 0.02, size=t)
    a1 = 0.5 * bench + rng.normal(0, 0.01, size=t)
    a2 = rng.normal(0, 0.02, size=t)
    r = np.column_stack([bench, a1, a2])
    eps, betas, _ = epsilon_matrix(r, bench_col=0)
    assert np.allclose(eps[:, 0], 0.0)
    assert betas[1] > 0.3
