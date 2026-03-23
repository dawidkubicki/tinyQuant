"""Scale portfolio targets to satisfy gross / net / leverage caps."""

from __future__ import annotations

from tinyquant.portfolio.sizing import gross_net_exposure


def enforce_gross_net_limits(
    targets: dict[str, dict[str, float]],
    *,
    equity_usd: float,
    max_gross: float | None = None,
    max_net_frac: float | None = None,
    max_leverage: float | None = None,
) -> dict[str, dict[str, float]]:
    """Mutates copy: scales all notionals uniformly if constraints violated."""
    if not targets or equity_usd <= 0:
        return targets

    out = {k: dict(v) for k, v in targets.items()}
    gross, net = gross_net_exposure(out)
    scale = 1.0

    if max_leverage is not None and max_leverage > 0:
        cap_g = max_leverage * equity_usd
        if gross > cap_g:
            scale = min(scale, cap_g / gross)

    if max_gross is not None and max_gross > 0:
        cap_g = max_gross * equity_usd
        if gross * scale > cap_g:
            scale = min(scale, cap_g / gross)

    if scale < 1.0:
        for t in out.values():
            t["target_notional_usd"] *= scale

    gross2, net2 = gross_net_exposure(out)
    if max_net_frac is not None and equity_usd > 0:
        max_net = max_net_frac * equity_usd
        if abs(net2) > max_net > 0:
            # shrink long or short leg differentially
            longs = {k: v for k, v in out.items() if v["side"] == "long"}
            shorts = {k: v for k, v in out.items() if v["side"] == "short"}
            ls, ss = sum(v["target_notional_usd"] for v in longs.values()), sum(
                v["target_notional_usd"] for v in shorts.values()
            )
            if net2 > max_net and ls > 0:
                excess = net2 - max_net
                f = max(0.0, (ls - excess) / ls)
                for k in longs:
                    out[k]["target_notional_usd"] *= f
            elif net2 < -max_net and ss > 0:
                excess = -net2 - max_net
                f = max(0.0, (ss - excess) / ss)
                for k in shorts:
                    out[k]["target_notional_usd"] *= f

    return out
