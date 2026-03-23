"""Time-decayed aggregation of stored news into per-ticker and macro signals."""

from __future__ import annotations

import time
from typing import Any

from tinyquant.news.store_sqlite import NewsSQLiteStore


def hours_since_event(now: float, event_unix: float | None, fallback_unix: float) -> float:
    t = event_unix if event_unix is not None else fallback_unix
    return max(0.0, (now - t) / 3600.0)


def decay_factor(hours: float, decay_base: float) -> float:
    if decay_base <= 0.0:
        return 0.0
    if decay_base >= 1.0:
        return 1.0
    return float(decay_base**hours)


def effective_sentiment(score: float, hours: float, decay_base: float) -> float:
    return float(score) * decay_factor(hours, decay_base)


def symbol_base_ticker(symbol: str) -> str:
    """CCXT perpetual like SOL/USDT:USDT -> SOL."""
    base = symbol.split("/")[0].strip()
    return base.upper()


def aggregate_news_for_universe(
    store: NewsSQLiteStore,
    tradable_symbols: list[str],
    *,
    now: float | None = None,
    window_hours: float = 24.0,
    decay_base: float = 0.9,
    macro_blend_into_tokens: float = 0.35,
) -> tuple[dict[str, float], float, dict[str, Any]]:
    """
    Returns:
      per_token_sentiment: map symbol -> score in [-1, 1]
      macro_aggregate: decay-weighted mean of MACRO items (for regime / audit)
      debug: counts and raw intermediates
    """
    now = time.time() if now is None else now
    since = now - window_hours * 3600.0
    store.init_schema()
    rows = store.fetch_since(since)

    micro_by_ticker: dict[str, list[float]] = {}
    macro_effective: list[float] = []

    for row in rows:
        ev_t = row.published_at if row.published_at is not None else row.analyzed_at
        if ev_t < since:
            continue
        h = hours_since_event(now, row.published_at, row.analyzed_at)
        eff = effective_sentiment(row.sentiment_score, h, decay_base)

        if row.classification == "MACRO":
            macro_effective.append(eff)
        elif row.classification == "MICRO":
            for ent in row.entities:
                micro_by_ticker.setdefault(ent, []).append(eff)

    macro_val = float(sum(macro_effective) / len(macro_effective)) if macro_effective else 0.0
    macro_val = max(-1.0, min(1.0, macro_val))

    ticker_avg: dict[str, float] = {}
    for t, vals in micro_by_ticker.items():
        if not vals:
            continue
        ticker_avg[t] = max(-1.0, min(1.0, sum(vals) / len(vals)))

    per_token: dict[str, float] = {}
    macro_component = macro_val * macro_blend_into_tokens
    for sym in tradable_symbols:
        base = symbol_base_ticker(sym)
        micro = ticker_avg.get(base, 0.0)
        combined = micro + macro_component
        per_token[sym] = max(-1.0, min(1.0, combined))

    debug: dict[str, Any] = {
        "n_rows_window": len(rows),
        "n_macro": len(macro_effective),
        "n_micro_tickers": len(micro_by_ticker),
        "macro_aggregate": macro_val,
        "macro_blend_into_tokens": macro_blend_into_tokens,
        "ticker_micro_avg": dict(sorted(ticker_avg.items())),
    }
    return per_token, macro_val, debug
