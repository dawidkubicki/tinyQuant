"""Portfolio-level daily loss kill-switch and cooldown."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path


@dataclass
class RiskState:
    day_key: str
    peak_equity_usd: float
    last_equity_usd: float
    cooldown_until_ts: float | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(s: str) -> RiskState:
        d = json.loads(s)
        return RiskState(
            day_key=d["day_key"],
            peak_equity_usd=float(d["peak_equity_usd"]),
            last_equity_usd=float(d["last_equity_usd"]),
            cooldown_until_ts=d.get("cooldown_until_ts"),
        )


def _today_key() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%d")


def load_risk_state(path: Path) -> RiskState | None:
    if not path.is_file():
        return None
    return RiskState.from_json(path.read_text(encoding="utf-8"))


def save_risk_state(path: Path, state: RiskState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state.to_json(), encoding="utf-8")


def update_risk_state(
    path: Path,
    equity_usd: float,
    *,
    loss_limit_pct: float,
    cooldown_hours: float,
) -> tuple[RiskState, bool]:
    """
    Returns (new_state, breached).
    breach: drawdown from intraday peak exceeds loss_limit_pct.
    """
    day = _today_key()
    prev = load_risk_state(path)
    now_ts = datetime.now(tz=UTC).timestamp()

    if prev is None or prev.day_key != day:
        state = RiskState(day_key=day, peak_equity_usd=equity_usd, last_equity_usd=equity_usd)
    else:
        state = RiskState(
            day_key=day,
            peak_equity_usd=max(prev.peak_equity_usd, equity_usd),
            last_equity_usd=equity_usd,
            cooldown_until_ts=prev.cooldown_until_ts,
        )

    breached = False
    if state.peak_equity_usd > 0:
        dd_pct = 100.0 * (state.peak_equity_usd - equity_usd) / state.peak_equity_usd
        if dd_pct >= loss_limit_pct:
            breached = True
            until = (datetime.now(tz=UTC) + timedelta(hours=cooldown_hours)).timestamp()
            prev_until = state.cooldown_until_ts or 0.0
            state.cooldown_until_ts = max(prev_until, until)

    save_risk_state(path, state)
    return state, breached


def in_cooldown(state: RiskState | None) -> bool:
    if state is None or state.cooldown_until_ts is None:
        return False
    return datetime.now(tz=UTC).timestamp() < state.cooldown_until_ts


def daily_loss_breached(
    path: Path,
    equity_usd: float,
    *,
    loss_limit_pct: float,
    cooldown_hours: float,
) -> tuple[bool, RiskState]:
    state, breached = update_risk_state(
        path,
        equity_usd,
        loss_limit_pct=loss_limit_pct,
        cooldown_hours=cooldown_hours,
    )
    return breached, state
