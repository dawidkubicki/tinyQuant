from __future__ import annotations

from datetime import datetime
from typing import Iterable

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential
from twelvedata import TDClient


METRIC_FIELD_MAP = {
    "earnings_per_share": "eps",
    "price_to_earnings_ratio": "pe_ratio",
    "debt_to_equity": "debt_to_equity",
    "revenue_growth": "revenue_growth",
}


class TwelveDataFundamentalsClient:
    def __init__(self, api_key: str) -> None:
        self._client = TDClient(apikey=api_key)

    @retry(wait=wait_exponential(multiplier=0.5, min=1, max=8), stop=stop_after_attempt(4), reraise=True)
    def _request_statistics(self, ticker: str) -> dict:
        response = self._client.statistics(symbol=ticker).as_json()
        if not isinstance(response, dict):
            raise RuntimeError(f"Unexpected Twelve Data response for {ticker}: {response}")
        return response

    def fetch_fundamentals(self, ticker: str) -> pd.DataFrame:
        payload = self._request_statistics(ticker)
        values: dict[str, float | None] = {column: None for column in METRIC_FIELD_MAP.values()}
        for source_key, target_key in METRIC_FIELD_MAP.items():
            raw_val = payload.get(source_key)
            values[target_key] = _coerce_float(raw_val)

        publish_date = payload.get("datetime") or payload.get("updated_at")
        if publish_date is None:
            publish_date = datetime.utcnow().strftime("%Y-%m-%d")

        frame = pd.DataFrame(
            [
                {
                    "publish_date": publish_date,
                    "ticker": ticker.upper(),
                    **values,
                }
            ]
        )
        return frame

    def fetch_universe_fundamentals(self, tickers: Iterable[str]) -> pd.DataFrame:
        frames = [self.fetch_fundamentals(ticker) for ticker in tickers]
        if not frames:
            return pd.DataFrame(columns=["publish_date", "ticker", "eps", "pe_ratio", "debt_to_equity", "revenue_growth"])
        return pd.concat(frames, ignore_index=True)


def _coerce_float(value: object) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
