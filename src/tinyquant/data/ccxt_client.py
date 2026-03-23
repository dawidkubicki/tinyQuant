"""CCXT wrapper: OHLCV (H4) and funding rates for perpetual swaps."""

from __future__ import annotations

import logging
import time
from typing import Any

import ccxt
import numpy as np
import pandas as pd

from tinyquant.config.schema import ExchangeConfig

logger = logging.getLogger(__name__)


class CCXTDataClient:
    def __init__(self, cfg: ExchangeConfig, api_key: str | None = None, secret: str | None = None):
        self.cfg = cfg
        try:
            klass = getattr(ccxt, cfg.id)
        except AttributeError as e:
            raise ValueError(
                f"Unknown CCXT exchange id {cfg.id!r}. tinyQuant is configured for Kraken Futures only "
                "(use exchange id 'krakenfutures')."
            ) from e
        opts: dict[str, Any] = {
            "enableRateLimit": cfg.enable_rate_limit,
            "options": {"defaultType": cfg.default_type},
        }
        if api_key and secret:
            opts["apiKey"] = api_key
            opts["secret"] = secret
        self.exchange = klass(opts)
        if cfg.sandbox and hasattr(self.exchange, "set_sandbox_mode"):
            self.exchange.set_sandbox_mode(True)

    def load_markets_safe(self) -> dict[str, Any]:
        return self.exchange.load_markets()

    def fetch_ohlcv_df(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
        *,
        max_retries: int = 5,
        backoff: float = 2.0,
    ) -> pd.DataFrame:
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                raw = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                break
            except Exception as e:
                last_exc = e
                logger.warning("fetch_ohlcv %s attempt %s: %s", symbol, attempt + 1, e)
                time.sleep(backoff * (attempt + 1))
        else:
            raise last_exc or RuntimeError("fetch_ohlcv failed")

        df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df

    def fetch_funding_rate_series(
        self,
        symbol: str,
        *,
        since_ms: int | None = None,
        limit: int = 1000,
        max_retries: int = 5,
        backoff: float = 2.0,
    ) -> pd.DataFrame:
        """Uses fetchFundingRateHistory when available; else empty DataFrame."""
        if not self.exchange.has.get("fetchFundingRateHistory"):
            logger.warning("Exchange %s has no fetchFundingRateHistory", self.cfg.id)
            return pd.DataFrame(columns=["ts", "fundingRate"])

        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                hist = self.exchange.fetch_funding_rate_history(symbol, since=since_ms, limit=limit)
                break
            except Exception as e:
                last_exc = e
                logger.warning("funding %s attempt %s: %s", symbol, attempt + 1, e)
                time.sleep(backoff * (attempt + 1))
        else:
            raise last_exc or RuntimeError("fetch_funding_rate_history failed")

        if not hist:
            return pd.DataFrame(columns=["ts", "fundingRate"])
        rows = []
        for x in hist:
            ts = x.get("timestamp")
            fr = x.get("fundingRate", x.get("info", {}).get("fundingRate"))
            if ts is not None and fr is not None:
                rows.append((pd.to_datetime(ts, unit="ms", utc=True), float(fr)))
        return pd.DataFrame(rows, columns=["ts", "fundingRate"])

    @staticmethod
    def aligned_column_matrix(
        dfs: dict[str, pd.DataFrame],
        symbols: list[str],
        column: str,
    ) -> tuple[np.ndarray, list[pd.Timestamp]]:
        """Align one OHLCV column per symbol on intersection of timestamps; (T, N) and index."""
        if not symbols:
            return np.zeros((0, 0)), []
        series_list = []
        for s in symbols:
            df = dfs[s].set_index("ts")[column].sort_index()
            series_list.append(df.rename(s))
        joined = pd.concat(series_list, axis=1, join="inner").sort_index()
        joined = joined.dropna(how="any")
        if joined.empty:
            return np.zeros((0, len(symbols))), []
        return joined.to_numpy(dtype=np.float64), list(joined.index)

    @staticmethod
    def close_prices_matrix(
        dfs: dict[str, pd.DataFrame],
        symbols: list[str],
    ) -> tuple[np.ndarray, list[pd.Timestamp]]:
        """Align close prices on intersection of timestamps; returns (T, N) and index."""
        return CCXTDataClient.aligned_column_matrix(dfs, symbols, "close")
