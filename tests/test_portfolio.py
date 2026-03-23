from __future__ import annotations

import numpy as np

from tinyquant.portfolio.constraints import enforce_gross_net_limits
from tinyquant.portfolio.rebalance import diff_orders
from tinyquant.portfolio.sizing import volatility_parity_targets


def test_vol_parity_targets_shape() -> None:
    rng = np.random.default_rng(4)
    n = 10
    symbols = [f"S{i}" for i in range(n)]
    scores = rng.normal(size=n)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, size=(120, n)), axis=0)
    t = volatility_parity_targets(
        symbols,
        scores,
        close,
        num_long=2,
        num_short=2,
        equity_usd=10_000.0,
        target_gross_exposure=1.0,
        vol_lookback=60,
    )
    assert len(t) == 4
    sides = [x["side"] for x in t.values()]
    assert sides.count("long") == 2
    assert sides.count("short") == 2


def test_diff_orders_min_notional() -> None:
    targets = {
        "A": {"side": "long", "target_notional_usd": 1000, "weight_in_leg": 1.0, "score": 1.0},
        "B": {"side": "short", "target_notional_usd": 1000, "weight_in_leg": 1.0, "score": -1.0},
    }
    current = {"A": 0.0, "B": 0.0}
    d = diff_orders(targets, current, min_notional_usd=100)
    assert len(d) == 2


def test_enforce_gross_net_limits_scales() -> None:
    targets = {
        "A": {"side": "long", "target_notional_usd": 8000, "weight_in_leg": 1.0, "score": 1.0},
        "B": {"side": "short", "target_notional_usd": 8000, "weight_in_leg": 1.0, "score": -1.0},
    }
    out = enforce_gross_net_limits(targets, equity_usd=10_000.0, max_leverage=1.0)
    gross = sum(x["target_notional_usd"] for x in out.values())
    assert gross <= 10_000.0 + 1e-6
