from __future__ import annotations

import pandas as pd

from tinyquant.pipeline.schema import standardize_fundamentals, standardize_prices


FUNDAMENTAL_VALUE_COLUMNS = ["eps", "pe_ratio", "debt_to_equity", "revenue_growth"]


def merge_prices_with_fundamentals(prices: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """
    Merge daily prices with sparse fundamentals and forward-fill only from publish_date.
    """
    prices_std = standardize_prices(prices)
    fund_std = standardize_fundamentals(fundamentals)

    prices_std = prices_std.sort_values(["ticker", "date"]).reset_index(drop=True)
    fund_std = fund_std.sort_values(["ticker", "publish_date"]).reset_index(drop=True)

    merged = pd.merge_asof(
        prices_std,
        fund_std,
        by="ticker",
        left_on="date",
        right_on="publish_date",
        direction="backward",
        allow_exact_matches=True,
    )
    merged = merged.drop(columns=["publish_date"])
    return _validate_fact_table(merged)


def _validate_fact_table(frame: pd.DataFrame) -> pd.DataFrame:
    dup_count = frame.duplicated(subset=["date", "ticker"]).sum()
    if dup_count > 0:
        raise ValueError(f"Detected {dup_count} duplicate (date, ticker) rows")

    required = {"date", "ticker", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required fact columns: {sorted(missing)}")

    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)
    return frame


def data_quality_report(frame: pd.DataFrame) -> dict:
    rows = len(frame)
    tickers = frame["ticker"].nunique() if "ticker" in frame.columns else 0
    min_date = frame["date"].min() if "date" in frame.columns and rows else None
    max_date = frame["date"].max() if "date" in frame.columns and rows else None
    missing_ratio = {}
    for col in ["open", "high", "low", "close", "volume", *FUNDAMENTAL_VALUE_COLUMNS]:
        if col in frame.columns:
            missing_ratio[col] = float(frame[col].isna().mean())
    return {
        "rows": rows,
        "tickers": int(tickers),
        "min_date": str(min_date) if min_date is not None else None,
        "max_date": str(max_date) if max_date is not None else None,
        "missing_ratio": missing_ratio,
    }
