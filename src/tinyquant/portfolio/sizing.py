"""Volatility parity long/short book from scores."""

from __future__ import annotations

import numpy as np


def _inv_vol_weights(vol: np.ndarray) -> np.ndarray:
    v = np.asarray(vol, dtype=np.float64)
    v = np.where(np.isfinite(v) & (v > 1e-12), v, np.nanmedian(v))
    inv = 1.0 / v
    inv[~np.isfinite(inv)] = 0.0
    s = inv.sum()
    if s < 1e-12:
        w = np.ones_like(inv) / len(inv)
    else:
        w = inv / s
    return w


def volatility_parity_targets(
    symbols: list[str],
    scores: np.ndarray,
    close_history: np.ndarray,
    *,
    num_long: int,
    num_short: int,
    equity_usd: float,
    target_gross_exposure: float,
    vol_lookback: int,
) -> dict[str, dict[str, float]]:
    """
    scores: higher -> long candidates; lower -> short.
    close_history: (T, N) aligned with symbols.
    Returns symbol -> {side, target_notional_usd, weight_in_leg}.
    """
    n = len(symbols)
    if n == 0 or equity_usd <= 0:
        return {}

    r = np.diff(np.log(np.clip(close_history[-vol_lookback - 1 :], 1e-12, None)), axis=0)
    vol = np.std(r, axis=0)
    vol = np.where(np.isfinite(vol), vol, np.nanmedian(vol))

    order = np.argsort(-scores)
    long_idx = order[:num_long]
    long_set = set(long_idx.tolist())
    short_candidates = order[::-1]
    short_idx_list: list[int] = []
    for idx in short_candidates:
        if idx in long_set:
            continue
        short_idx_list.append(int(idx))
        if len(short_idx_list) >= num_short:
            break
    short_idx = np.array(short_idx_list, dtype=int)

    half_budget = equity_usd * target_gross_exposure * 0.5
    w_long = _inv_vol_weights(vol[long_idx])
    w_short = _inv_vol_weights(vol[short_idx])

    out: dict[str, dict[str, float]] = {}
    for i, idx in enumerate(long_idx):
        sym = symbols[idx]
        out[sym] = {
            "side": "long",
            "target_notional_usd": float(half_budget * w_long[i]),
            "weight_in_leg": float(w_long[i]),
            "score": float(scores[idx]),
        }
    for i, idx in enumerate(short_idx):
        sym = symbols[idx]
        out[sym] = {
            "side": "short",
            "target_notional_usd": float(half_budget * w_short[i]),
            "weight_in_leg": float(w_short[i]),
            "score": float(scores[idx]),
        }
    return out


def gross_net_exposure(targets: dict[str, dict[str, float]]) -> tuple[float, float]:
    long_sum = sum(t["target_notional_usd"] for t in targets.values() if t["side"] == "long")
    short_sum = sum(t["target_notional_usd"] for t in targets.values() if t["side"] == "short")
    gross = long_sum + short_sum
    net = long_sum - short_sum
    return gross, net
