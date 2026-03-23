"""Universe selection: top-N linear perpetual swaps by 24h quote volume (Kraken Futures / CCXT)."""

from __future__ import annotations

import logging
from typing import Any

from tinyquant.config.schema import UniverseConfig

logger = logging.getLogger(__name__)


def _is_linear_swap_for_quote(market: dict[str, Any], quote: str) -> bool:
    """Linear US*-margined perpetual swap matching ``quote`` (Kraken: USD in CCXT symbols)."""
    if not market.get("swap") or not market.get("linear"):
        return False
    q = quote.upper()
    if (market.get("quote") or "").upper() != q and (market.get("quoteId") or "").upper() != q:
        return False
    settle = (market.get("settle") or "").upper()
    if settle and settle != q:
        return False
    if not market.get("active", True):
        return False
    return True


def select_universe_symbols(exchange: Any, cfg: UniverseConfig) -> list[str]:
    markets = exchange.load_markets()
    candidates: list[str] = []
    for sym, m in markets.items():
        if not _is_linear_swap_for_quote(m, cfg.quote_asset):
            continue
        if sym in cfg.blacklist or m.get("id") in cfg.blacklist:
            continue
        candidates.append(sym)

    tickers = exchange.fetch_tickers(candidates)
    scored: list[tuple[str, float]] = []
    for sym in candidates:
        t = tickers.get(sym) or {}
        qv = float(t.get("quoteVolume") or 0.0)
        if qv < cfg.min_quote_volume_usd_24h:
            continue
        scored.append((sym, qv))

    scored.sort(key=lambda x: -x[1])
    out = [s for s, _ in scored[: cfg.top_n]]

    if cfg.benchmark_symbol not in out and cfg.benchmark_symbol in tickers:
        if cfg.benchmark_symbol in candidates:
            out = [cfg.benchmark_symbol] + [x for x in out if x != cfg.benchmark_symbol]
            out = out[: cfg.top_n]

    logger.info("Universe size %s (top by quote volume)", len(out))
    return out
