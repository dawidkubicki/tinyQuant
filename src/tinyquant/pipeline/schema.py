from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


PRICE_COLUMNS = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

FUNDAMENTAL_COLUMNS = [
    "publish_date",
    "ticker",
    "eps",
    "pe_ratio",
    "debt_to_equity",
    "revenue_growth",
]

FACT_COLUMNS = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "eps",
    "pe_ratio",
    "debt_to_equity",
    "revenue_growth",
]


@dataclass(frozen=True)
class DataContracts:
    prices: list[str]
    fundamentals: list[str]
    fact: list[str]


def contracts() -> DataContracts:
    return DataContracts(
        prices=PRICE_COLUMNS.copy(),
        fundamentals=FUNDAMENTAL_COLUMNS.copy(),
        fact=FACT_COLUMNS.copy(),
    )


def _ensure_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = set(required).difference(frame.columns)
    if missing:
        missing_sorted = ", ".join(sorted(missing))
        raise ValueError(f"{name} is missing required columns: {missing_sorted}")


def standardize_prices(prices: pd.DataFrame) -> pd.DataFrame:
    _ensure_columns(prices, PRICE_COLUMNS, "prices")
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
    df["ticker"] = df["ticker"].astype(str).str.upper()
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("Int64")
    return df[PRICE_COLUMNS].sort_values(["ticker", "date"]).reset_index(drop=True)


def standardize_fundamentals(fundamentals: pd.DataFrame) -> pd.DataFrame:
    _ensure_columns(fundamentals, FUNDAMENTAL_COLUMNS, "fundamentals")
    df = fundamentals.copy()
    df["publish_date"] = pd.to_datetime(df["publish_date"], utc=True).dt.tz_convert(None)
    df["ticker"] = df["ticker"].astype(str).str.upper()
    for col in ["eps", "pe_ratio", "debt_to_equity", "revenue_growth"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[FUNDAMENTAL_COLUMNS].sort_values(["ticker", "publish_date"]).reset_index(drop=True)
