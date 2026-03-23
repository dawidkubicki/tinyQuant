"""Incremental delta between current notionals and targets."""

from __future__ import annotations


def diff_orders(
    targets: dict[str, dict[str, float]],
    current: dict[str, float],
    *,
    min_notional_usd: float,
) -> list[dict[str, object]]:
    """
    current: symbol -> signed notional (positive long, negative short).
    Returns list of {symbol, side, delta_usd} where |delta| >= min_notional or close.
    """
    out: list[dict[str, object]] = []
    all_syms = set(targets) | set(current)

    for sym in all_syms:
        tgt = targets.get(sym)
        cur = float(current.get(sym, 0.0))
        if tgt is None:
            if abs(cur) >= min_notional_usd:
                out.append(
                    {
                        "symbol": sym,
                        "side": "close",
                        "delta_usd": -cur,
                        "reason": "not_in_target",
                    }
                )
            continue

        side = tgt["side"]
        want = tgt["target_notional_usd"]
        signed_want = want if side == "long" else -want
        delta = signed_want - cur
        if abs(delta) >= min_notional_usd:
            out.append(
                {
                    "symbol": sym,
                    "side": "rebalance",
                    "delta_usd": delta,
                    "target_side": side,
                }
            )
    return out
