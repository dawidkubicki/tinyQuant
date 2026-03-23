from __future__ import annotations

import numpy as np

from tinyquant.signals.graph_diffusion import diffusion_mispricing_scores
from tinyquant.signals.tda_global import correlation_distance_matrix, persistence_landscape_vector


def test_diffusion_nonzero_on_structured_residuals() -> None:
    rng = np.random.default_rng(2)
    t, n = 80, 8
    z = rng.normal(0, 0.02, size=(t, n))
    z[-1, 0] += 0.05
    s = diffusion_mispricing_scores(z, edge_threshold=0.1, diffusion_steps=2, diffusion_alpha=0.25)
    assert s.shape == (n,)
    assert np.any(np.abs(s) > 1e-6)


def test_correlation_distance_and_landscape() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(40, 5))
    c = np.corrcoef(x.T)
    d = correlation_distance_matrix(c)
    assert d.shape == (5, 5)
    v = persistence_landscape_vector(d, max_dimension=1, max_edge_length=2.0, resolution=20)
    assert v.shape == (20,)
    assert np.all(np.isfinite(v))
