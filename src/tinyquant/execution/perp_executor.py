"""Perpetual execution — paper mode records intended orders; live uses CCXT (optional)."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PaperPerpExecutor:
    paper: bool = True
    fee_bps_taker: float = 5.0
    positions: dict[str, float] = field(default_factory=dict)
    order_log: list[dict[str, Any]] = field(default_factory=list)

    def apply_deltas(
        self,
        deltas: list[dict[str, object]],
        *,
        mark_prices: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        """Apply signed USD notional deltas to internal position map (paper)."""
        mark_prices = mark_prices or {}
        filled: list[dict[str, Any]] = []
        for d in deltas:
            sym = str(d["symbol"])
            delta = float(d["delta_usd"])
            if abs(delta) < 1e-9:
                continue
            old = self.positions.get(sym, 0.0)
            new = old + delta
            fee = abs(delta) * (self.fee_bps_taker / 10_000.0)
            rec = {
                "id": str(uuid.uuid4()),
                "symbol": sym,
                "delta_usd": delta,
                "position_before": old,
                "position_after": new,
                "fee_usd": fee,
                "paper": self.paper,
                "ts": time.time(),
            }
            self.order_log.append(rec)
            self.positions[sym] = new
            filled.append(rec)
            logger.info("PAPER exec %s delta=%.4f fee=%.6f", sym, delta, fee)
        return filled

    def close_all(self) -> list[dict[str, Any]]:
        deltas = [
            {"symbol": sym, "delta_usd": -pos, "side": "close_all"}
            for sym, pos in list(self.positions.items())
            if abs(pos) >= 1e-9
        ]
        return self.apply_deltas(deltas)
