"""Wall-clock helpers for periodic H4 runs (UTC)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta


def seconds_until_next_bar_close(timeframe: str = "4h") -> float:
    """Approximate seconds until next 4h UTC boundary (00,04,08,12,16,20)."""
    if timeframe != "4h":
        raise NotImplementedError("Only 4h supported in scheduler helper")
    now = datetime.now(tz=UTC)
    h = now.hour
    next_h = ((h // 4) + 1) * 4
    if next_h >= 24:
        nxt = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    else:
        nxt = now.replace(hour=next_h, minute=0, second=0, microsecond=0)
    return max(0.0, (nxt - now).total_seconds())
