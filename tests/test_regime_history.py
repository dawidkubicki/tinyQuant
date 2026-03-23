from __future__ import annotations

import numpy as np

from tinyquant.orchestration.h4_cycle import regime_feature_row_from_panel
from tinyquant.regime.train_history import build_regime_feature_matrix_history


def test_regime_feature_row_from_panel_synthetic_shape(strategy_config) -> None:
    rng = np.random.default_rng(1)
    t, n = 180, 5
    rets = rng.normal(0, 0.012, size=(t, n))
    close = 100.0 * np.cumprod(1.0 + rets, axis=0)
    vol = rng.uniform(100.0, 10_000.0, size=(t, n))
    symbols = ["BTC/USD:USD"] + [f"ALT{i}/USD:USD" for i in range(1, n)]
    row = regime_feature_row_from_panel(
        close,
        vol_full=vol,
        symbols=symbols,
        bench_sym="BTC/USD:USD",
        cfg=strategy_config,
    )
    d = strategy_config.tda.landscape_resolution + 3
    assert row.shape == (d,)
    assert np.isfinite(row).all()


def test_build_regime_feature_matrix_history_non_empty(strategy_config) -> None:
    rng = np.random.default_rng(2)
    t, n = 250, 6
    rets = rng.normal(0, 0.01, size=(t, n))
    close = 100.0 * np.cumprod(1.0 + rets, axis=0)
    vol = rng.uniform(500.0, 5_000.0, size=(t, n))
    symbols = ["BTC/USD:USD"] + [f"ALT{i}/USD:USD" for i in range(1, n)]
    X = build_regime_feature_matrix_history(
        close,
        vol,
        symbols,
        "BTC/USD:USD",
        strategy_config,
        step=20,
        max_samples=50,
    )
    assert X.ndim == 2
    assert X.shape[1] == strategy_config.tda.landscape_resolution + 3
    assert X.shape[0] >= 1
    assert X.shape[0] <= 50
