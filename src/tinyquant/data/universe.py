"""Universe selection: top-N USDT perpetuals by 24h quote volume."""

from __future__ import annotations

import logging
from typing import Any

from tinyquant.config.schema import UniverseConfig

logger = logging.getLogger(__name__)


def _is_usdt_swap_perp(market: dict[str, Any], quote: str) -> bool:
    if not market.get("swap") and not market.get("linear"):
        return False
    if market.get("quote") != quote and market.get("quoteId") != quote.lower():
        return False
    sym = market.get("symbol", "") or market.get("id", "")
    if "USDT" not in sym.upper():
        return False
    if not market.get("active", True):
        return False
    return True


def select_universe_symbols(exchange: Any, cfg: UniverseConfig) -> list[str]:
    markets = exchange.load_markets()
    candidates: list[str] = []
    for sym, m in markets.items():
        if not _is_usdt_swap_perp(m, cfg.quote_asset):
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
