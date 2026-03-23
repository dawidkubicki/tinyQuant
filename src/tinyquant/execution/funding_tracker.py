"""Rough funding PnL estimate from rates and signed positions (USD notional)."""

from __future__ import annotations


def estimate_funding_pnl_usd(
    positions_signed_usd: dict[str, float],
    funding_rate_by_symbol: dict[str, float],
) -> float:
    """
    positions_signed_usd: positive long, negative short.
    funding paid by longs when rate > 0 on many venues -> long pays, short receives.
    PnL += -position * rate for long (+ for short).
    """
    total = 0.0
    for sym, pos in positions_signed_usd.items():
        r = float(funding_rate_by_symbol.get(sym, 0.0))
        total += -pos * r
    return total
