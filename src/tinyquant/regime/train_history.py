"""Build historical regime feature matrix (same logic as live) and fit GMM + XGBoost."""

from __future__ import annotations

import logging

import numpy as np

from tinyquant.config.schema import StrategyConfig
from tinyquant.data.ccxt_client import CCXTDataClient
from tinyquant.data.universe import select_universe_symbols
from tinyquant.orchestration.h4_cycle import _fetch_ohlcv_panel, regime_feature_row_from_panel
from tinyquant.regime.gmm_offline import fit_and_save_gmm
from tinyquant.regime.xgb_online import train_and_save_xgb

logger = logging.getLogger(__name__)


def build_regime_feature_matrix_history(
    close: np.ndarray,
    vol_full: np.ndarray,
    symbols: list[str],
    bench_sym: str,
    cfg: StrategyConfig,
    *,
    step: int = 4,
    max_samples: int | None = None,
) -> np.ndarray:
    """
    Rolling regime feature rows; each row uses data up to and including bar index `end`.
    step: subsample bar indices (every `step` bars) to reduce TDA cost.
    """
    if step < 1:
        raise ValueError("step must be >= 1")
    if close.shape != vol_full.shape:
        raise ValueError("close and vol_full must match")
    min_bars = cfg.beta_neutralization.lookback_bars + 10
    t = close.shape[0]
    rows: list[np.ndarray] = []
    # end = inclusive last bar index; panel length = end + 1
    end = min_bars - 1
    while end < t:
        try:
            row = regime_feature_row_from_panel(
                close[: end + 1],
                vol_full=vol_full[: end + 1],
                symbols=symbols,
                bench_sym=bench_sym,
                cfg=cfg,
            )
            rows.append(row)
        except (RuntimeError, ValueError) as e:
            logger.debug("Skip end=%s: %s", end, e)
        end += step
    if not rows:
        return np.zeros((0, cfg.tda.landscape_resolution + 3), dtype=np.float64)
    X = np.stack(rows, axis=0)
    if max_samples is not None and X.shape[0] > max_samples:
        X = X[-max_samples:]
    return X


def fit_regime_models_from_exchange(
    cfg: StrategyConfig,
    *,
    api_key: str | None,
    secret: str | None,
    lookback_bars: int,
    step: int = 4,
    max_samples: int | None = None,
) -> tuple[int, int]:
    """
    Fetch OHLCV, build X_hist, fit GMM and XGBoost to paths in cfg.
    Returns (n_rows_fitted, feature_dim).
    """
    bench_sym = cfg.universe.benchmark_symbol
    client = CCXTDataClient(cfg.exchange, api_key=api_key, secret=secret)
    symbols = select_universe_symbols(client.exchange, cfg.universe)
    if bench_sym not in symbols:
        symbols = [bench_sym] + [s for s in symbols if s != bench_sym][: cfg.universe.top_n]
    dfs, valid = _fetch_ohlcv_panel(client, symbols, cfg, lookback_bars=lookback_bars)
    symbols = [s for s in symbols if s in valid]
    if bench_sym not in symbols:
        raise RuntimeError(f"Benchmark {bench_sym} missing after fetch")
    close, _ts = CCXTDataClient.close_prices_matrix(dfs, symbols)
    vol_full, _ = CCXTDataClient.aligned_column_matrix(dfs, symbols, "volume")
    if close.shape != vol_full.shape:
        raise RuntimeError("Aligned close/volume shape mismatch")
    X = build_regime_feature_matrix_history(
        close,
        vol_full,
        symbols,
        bench_sym,
        cfg,
        step=step,
        max_samples=max_samples,
    )
    if X.shape[0] < cfg.regime.gmm.n_components * 5:
        raise RuntimeError(
            f"Too few history rows ({X.shape[0]}); increase --lookback or decrease --step"
        )
    n_regimes = cfg.regime.gmm.n_components
    gmm = fit_and_save_gmm(X, cfg.regime.gmm)
    y = gmm.predict(X)
    train_and_save_xgb(X, y, cfg.regime.xgboost, num_class=n_regimes)
    logger.info("Saved GMM + XGBoost from %s historical feature rows (dim=%s)", X.shape[0], X.shape[1])
    return X.shape[0], X.shape[1]
