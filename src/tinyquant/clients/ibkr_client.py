from __future__ import annotations

import time
from datetime import datetime
from typing import Iterable

import pandas as pd
from ib_insync import IB, Stock, util
from tenacity import retry, stop_after_attempt, wait_exponential


class IBKRClient:
    def __init__(
        self,
        host: str,
        port: int,
        client_id: int,
        request_sleep_seconds: float = 0.35,
    ) -> None:
        self._host = host
        self._port = port
        self._client_id = client_id
        self._request_sleep_seconds = request_sleep_seconds
        self._ib = IB()

    @retry(wait=wait_exponential(multiplier=0.5, min=1, max=8), stop=stop_after_attempt(4), reraise=True)
    def connect(self) -> None:
        if not self._ib.isConnected():
            self._ib.connect(self._host, self._port, clientId=self._client_id, timeout=10)

    def disconnect(self) -> None:
        if self._ib.isConnected():
            self._ib.disconnect()

    def __enter__(self) -> "IBKRClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.disconnect()

    def fetch_daily_ohlcv(
        self,
        ticker: str,
        years_back: int = 5,
        end_datetime: str = "",
    ) -> pd.DataFrame:
        contract = Stock(ticker, "SMART", "USD")
        duration_str = f"{years_back} Y"
        bars = self._ib.reqHistoricalData(
            contract,
            endDateTime=end_datetime,
            durationStr=duration_str,
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )
        time.sleep(self._request_sleep_seconds)
        if not bars:
            return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])

        df = util.df(bars)
        if df.empty:
            return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])
        df = df.rename(columns={"date": "date", "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"})
        df["ticker"] = ticker.upper()
        return df[["date", "ticker", "open", "high", "low", "close", "volume"]]

    def fetch_universe_daily_ohlcv(
        self,
        tickers: Iterable[str],
        years_back: int = 5,
        start_date: datetime | None = None,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for ticker in tickers:
            frame = self.fetch_daily_ohlcv(ticker=ticker, years_back=years_back)
            if start_date is not None and not frame.empty:
                frame = frame[frame["date"] >= pd.Timestamp(start_date)]
            frames.append(frame)
        if not frames:
            return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])
        return pd.concat(frames, ignore_index=True)
