"""Single H4 pipeline: data → features → signals → regime → scores → portfolio → risk → execution."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tinyquant.config.schema import StrategyConfig
from tinyquant.data.ccxt_client import CCXTDataClient
from tinyquant.data.universe import select_universe_symbols
from tinyquant.execution.perp_executor import PaperPerpExecutor
from tinyquant.features.beta_neutral import epsilon_matrix
from tinyquant.features.returns import log_returns_from_close
from tinyquant.model.rd_gat import infer_token_scores
from tinyquant.portfolio.constraints import enforce_gross_net_limits
from tinyquant.portfolio.rebalance import diff_orders
from tinyquant.portfolio.sizing import volatility_parity_targets
from tinyquant.regime.gmm_offline import load_gmm
from tinyquant.regime.xgb_online import load_xgb, predict_regime_proba
from tinyquant.risk.portfolio_guards import daily_loss_breached, in_cooldown, load_risk_state
from tinyquant.signals.graph_diffusion import diffusion_mispricing_scores
from tinyquant.signals.sentiment_interface import sentiment_vector
from tinyquant.signals.tda_global import correlation_distance_matrix, persistence_landscape_vector

logger = logging.getLogger(__name__)


def _tradable_mask(symbols: list[str], benchmark: str) -> list[int]:
    return [i for i, s in enumerate(symbols) if s != benchmark]


def _synthetic_universe(n: int = 16, t: int = 320, seed: int = 42) -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.015, size=(t, n))
    close = 100.0 * np.cumprod(1.0 + rets, axis=0)
    symbols = ["BTC/USDT:USDT"] + [f"ALT{i}/USDT:USDT" for i in range(1, n)]
    return symbols, close.astype(np.float64)


def _fetch_ohlcv_panel(
    client: CCXTDataClient,
    symbols: list[str],
    cfg: StrategyConfig,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    tf = cfg.data.ohlcv_timeframe
    limit = cfg.data.ohlcv_lookback_bars
    dfs: dict[str, pd.DataFrame] = {}
    valid: list[str] = []
    for sym in symbols:
        try:
            df = client.fetch_ohlcv_df(
                sym,
                tf,
                limit,
                max_retries=cfg.runtime.max_retries,
                backoff=cfg.runtime.retry_backoff_seconds,
            )
            if len(df) < cfg.beta_neutralization.lookback_bars + 10:
                continue
            dfs[sym] = df
            valid.append(sym)
        except Exception as e:
            logger.warning("Skip %s: %s", sym, e)
    return dfs, valid


def build_regime_feature_row(
    landscape: np.ndarray,
    bench_returns_window: np.ndarray,
    volume_window: np.ndarray | None = None,
) -> np.ndarray:
    btc_vol = float(np.std(bench_returns_window[np.isfinite(bench_returns_window)]))
    parts = [landscape, np.array([btc_vol], dtype=np.float64)]
    if volume_window is not None and volume_window.size:
        vm = volume_window.astype(np.float64)
        vm = vm[np.isfinite(vm)]
        parts.append(np.array([float(np.mean(vm)), float(np.std(vm))], dtype=np.float64))
    else:
        parts.append(np.array([0.0, 0.0], dtype=np.float64))
    return np.concatenate(parts)


def run_h4_cycle(
    cfg: StrategyConfig,
    *,
    equity_usd: float = 10_000.0,
    synthetic: bool = False,
    api_key: str | None = None,
    secret: str | None = None,
    executor: PaperPerpExecutor | None = None,
) -> dict[str, Any]:
    paths = cfg.resolved_paths()
    risk_path = paths["risk_state"]
    audit_dir = paths["audit"]
    audit_dir.mkdir(parents=True, exist_ok=True)

    rs = load_risk_state(risk_path)
    if in_cooldown(rs):
        logger.warning("Risk cooldown active — skipping cycle")
        return {"skipped": True, "reason": "cooldown"}

    bench_sym = cfg.universe.benchmark_symbol

    client: CCXTDataClient | None = None
    if synthetic:
        symbols, close = _synthetic_universe()
        bench_idx = 0
    else:
        client = CCXTDataClient(cfg.exchange, api_key=api_key, secret=secret)
        symbols = select_universe_symbols(client.exchange, cfg.universe)
        if bench_sym not in symbols:
            symbols = [bench_sym] + [s for s in symbols if s != bench_sym][: cfg.universe.top_n]
        dfs, valid = _fetch_ohlcv_panel(client, symbols, cfg)
        symbols = [s for s in symbols if s in valid]
        if bench_sym not in symbols:
            raise RuntimeError(f"Benchmark {bench_sym} missing after fetch")
        close, _ts = CCXTDataClient.close_prices_matrix(dfs, symbols)
        if close.shape[0] < cfg.beta_neutralization.lookback_bars + 10:
            raise RuntimeError("Insufficient aligned history")
        bench_idx = symbols.index(bench_sym)

    t, n = close.shape
    rets = np.diff(np.log(np.clip(close, 1e-12, None)), axis=0)
    eps, betas, alphas = epsilon_matrix(
        rets,
        bench_col=bench_idx,
        winsorize_std=cfg.beta_neutralization.winsorize_std,
    )

    trad_idx = _tradable_mask(symbols, bench_sym)
    eps_t = eps[:, trad_idx]
    sym_t = [symbols[i] for i in trad_idx]

    gl = min(cfg.graph_signal.correlation_lookback_bars, eps_t.shape[0])
    diff_raw = diffusion_mispricing_scores(
        eps_t[-gl:],
        edge_threshold=cfg.graph_signal.edge_threshold,
        diffusion_steps=cfg.graph_signal.diffusion_steps,
        diffusion_alpha=cfg.graph_signal.diffusion_alpha,
        score_clip=cfg.graph_signal.score_clip,
    )

    tl = min(cfg.tda.correlation_lookback_bars, eps_t.shape[0])
    sub = eps_t[-tl:]
    corr = np.corrcoef(sub.T)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    dist = correlation_distance_matrix(corr)
    land = persistence_landscape_vector(
        dist,
        max_dimension=cfg.tda.max_dimension,
        max_edge_length=cfg.tda.max_edge_length,
        resolution=cfg.tda.landscape_resolution,
        t_min=cfg.tda.landscape_min,
        t_max=cfg.tda.landscape_max,
    )

    bench_r = rets[-tl:, bench_idx]
    feat_row = build_regime_feature_row(land, bench_r, volume_window=None)

    n_regimes = cfg.regime.gmm.n_components
    xgb_model = load_xgb(cfg.regime.xgboost)
    gmm = load_gmm(cfg.regime.gmm)

    if xgb_model is not None:
        regime_id, proba, conf = predict_regime_proba(xgb_model, feat_row, n_regimes=n_regimes)
    elif gmm is not None:
        try:
            proba = gmm.predict_proba(feat_row.reshape(1, -1))[0]
            regime_id = int(np.argmax(proba))
            conf = float(proba.max())
        except Exception:
            proba = np.ones(n_regimes, dtype=np.float64) / n_regimes
            regime_id = 0
            conf = float(proba.max())
    else:
        proba = np.ones(n_regimes, dtype=np.float64) / n_regimes
        regime_id = 0
        conf = float(proba.max())

    if conf < cfg.regime.min_regime_confidence:
        logger.info("Low regime confidence %.4f — using blended uniform prior", conf)

    funding_vec = np.zeros(len(sym_t))
    if not synthetic and client is not None:
        try:
            for j, sym in enumerate(sym_t):
                fr = client.fetch_funding_rate_series(sym, limit=50)
                if len(fr):
                    funding_vec[j] = float(fr["fundingRate"].iloc[-1])
        except Exception as e:
            logger.warning("Funding fetch degraded: %s", e)

    sent = sentiment_vector(len(sym_t), cfg.sentiment)

    n_t = len(sym_t)
    c = np.corrcoef(eps_t[-gl:].T) if gl >= 5 else np.eye(n_t)
    c = np.nan_to_num(c, nan=0.0)
    adj = (np.abs(c) >= cfg.graph_signal.edge_threshold).astype(np.float64)
    np.fill_diagonal(adj, 0.0)

    scores_t = infer_token_scores(
        diffusion_scores=diff_raw,
        funding_features=funding_vec,
        sentiment_per_token=sent,
        adjacency=adj,
        regime_id=regime_id,
        cfg=cfg.rd_gat,
        sentiment_cfg=cfg.sentiment,
    )

    scores_full = np.zeros(n)
    for j, ti in enumerate(trad_idx):
        scores_full[ti] = scores_t[j]

    targets = volatility_parity_targets(
        sym_t,
        scores_t,
        close[-cfg.portfolio.volatility_lookback_bars - 1 :, :][:, trad_idx],
        num_long=cfg.portfolio.num_long,
        num_short=cfg.portfolio.num_short,
        equity_usd=equity_usd,
        target_gross_exposure=cfg.portfolio.target_gross_exposure,
        vol_lookback=min(cfg.portfolio.volatility_lookback_bars, close.shape[0] - 1),
    )
    targets = enforce_gross_net_limits(
        targets,
        equity_usd=equity_usd,
        max_gross=cfg.portfolio.target_gross_exposure,
        max_net_frac=cfg.portfolio.max_net_exposure,
        max_leverage=cfg.portfolio.max_leverage,
    )

    execu = executor or PaperPerpExecutor(paper=cfg.execution.paper, fee_bps_taker=cfg.execution.fee_bps_taker)
    current: dict[str, float] = dict(execu.positions)
    deltas = diff_orders(targets, current, min_notional_usd=cfg.portfolio.min_notional_usd)
    fills = execu.apply_deltas(
        [
            {"symbol": d["symbol"], "delta_usd": d["delta_usd"], "side": d.get("side", "rebalance")}
            for d in deltas
        ]
    )

    breached = False
    close_fills: list[dict[str, Any]] = []
    if cfg.risk.close_all_on_breach:
        breached, _st = daily_loss_breached(
            risk_path,
            equity_usd,
            loss_limit_pct=cfg.risk.daily_portfolio_loss_limit_pct,
            cooldown_hours=float(cfg.risk.cooldown_hours_after_breach),
        )
        if breached:
            close_fills = execu.close_all()

    ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    audit = {
        "ts": ts,
        "symbols": symbols,
        "tradable": sym_t,
        "regime_id": regime_id,
        "regime_confidence": conf,
        "regime_proba": proba.tolist() if hasattr(proba, "tolist") else list(proba),
        "betas": {symbols[i]: float(betas[i]) for i in range(n) if np.isfinite(betas[i])},
        "diffusion_scores": {sym_t[i]: float(diff_raw[i]) for i in range(len(sym_t))},
        "gat_scores": {sym_t[i]: float(scores_t[i]) for i in range(len(sym_t))},
        "targets": targets,
        "deltas": deltas,
        "fills": fills + close_fills,
        "risk_daily_breach": breached,
        "synthetic": synthetic,
    }
    audit_path = audit_dir / f"h4_cycle_{ts}.json"
    audit_path.write_text(json.dumps(audit, default=str, indent=2), encoding="utf-8")

    return audit
