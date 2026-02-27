from __future__ import annotations

import pandas as pd

from tinyquant.pipeline.merge_and_fill import merge_prices_with_fundamentals


def test_forward_fill_starts_at_publish_date_without_leakage() -> None:
    prices = pd.DataFrame(
        [
            {"date": "2024-01-01", "ticker": "AAA", "open": 10, "high": 11, "low": 9, "close": 10, "volume": 100},
            {"date": "2024-01-02", "ticker": "AAA", "open": 10, "high": 11, "low": 9, "close": 10, "volume": 110},
            {"date": "2024-01-03", "ticker": "AAA", "open": 10, "high": 11, "low": 9, "close": 10, "volume": 120},
        ]
    )
    fundamentals = pd.DataFrame(
        [
            {
                "publish_date": "2024-01-02",
                "ticker": "AAA",
                "eps": 1.2,
                "pe_ratio": 20.0,
                "debt_to_equity": 0.5,
                "revenue_growth": 0.1,
            }
        ]
    )

    merged = merge_prices_with_fundamentals(prices, fundamentals)
    jan_1 = merged.loc[merged["date"] == pd.Timestamp("2024-01-01"), "eps"].iloc[0]
    jan_2 = merged.loc[merged["date"] == pd.Timestamp("2024-01-02"), "eps"].iloc[0]
    jan_3 = merged.loc[merged["date"] == pd.Timestamp("2024-01-03"), "eps"].iloc[0]

    assert pd.isna(jan_1)
    assert jan_2 == 1.2
    assert jan_3 == 1.2


def test_deduplicates_enforced_by_validator() -> None:
    prices = pd.DataFrame(
        [
            {"date": "2024-01-01", "ticker": "AAA", "open": 10, "high": 11, "low": 9, "close": 10, "volume": 100},
            {"date": "2024-01-01", "ticker": "AAA", "open": 10, "high": 11, "low": 9, "close": 10, "volume": 100},
        ]
    )
    fundamentals = pd.DataFrame(
        [
            {
                "publish_date": "2023-12-20",
                "ticker": "AAA",
                "eps": 1.0,
                "pe_ratio": 18.0,
                "debt_to_equity": 0.6,
                "revenue_growth": 0.08,
            }
        ]
    )

    try:
        merge_prices_with_fundamentals(prices, fundamentals)
    except ValueError as exc:
        assert "duplicate" in str(exc).lower()
    else:
        raise AssertionError("Expected duplicate validation to fail")
