from __future__ import annotations

import logging
from datetime import timedelta

import pandas as pd

from tinyquant.clients.ibkr_client import IBKRClient
from tinyquant.clients.twelve_data_client import TwelveDataFundamentalsClient
from tinyquant.config import get_settings
from tinyquant.pipeline.merge_and_fill import data_quality_report, merge_prices_with_fundamentals
from tinyquant.storage.parquet_store import ParquetStore


def run() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    logger = logging.getLogger(__name__)

    store = ParquetStore(settings.data_dir)
    tickers = settings.universe_list
    last_dt = store.last_fact_date()
    start_date = (last_dt - timedelta(days=10)).to_pydatetime() if last_dt is not None else None

    with IBKRClient(
        host=settings.ibkr_host,
        port=settings.ibkr_port,
        client_id=settings.ibkr_client_id,
        request_sleep_seconds=settings.ibkr_request_sleep_seconds,
    ) as ibkr:
        prices = ibkr.fetch_universe_daily_ohlcv(tickers=tickers, years_back=5, start_date=start_date)

    fundamentals_client = TwelveDataFundamentalsClient(settings.twelve_data_api_key)
    fresh_fundamentals = fundamentals_client.fetch_universe_fundamentals(tickers=tickers)

    existing_fundamentals = store.read_raw_fundamentals()
    all_fundamentals = (
        pd.concat([existing_fundamentals, fresh_fundamentals], ignore_index=True)
        .sort_values(["ticker", "publish_date"])
        .drop_duplicates(subset=["ticker", "publish_date"], keep="last")
    )
    store.write_raw_fundamentals(all_fundamentals.reset_index(drop=True))

    existing_prices = store.read_raw_ibkr()
    all_prices = (
        pd.concat([existing_prices, prices], ignore_index=True)
        .sort_values(["ticker", "date"])
        .drop_duplicates(subset=["ticker", "date"], keep="last")
    )
    store.write_raw_ibkr(all_prices.reset_index(drop=True))

    fact = merge_prices_with_fundamentals(prices=all_prices, fundamentals=all_fundamentals)
    path = store.upsert_fact(fact)

    dq = data_quality_report(fact)
    logger.info("Daily update completed. Fact upserted to %s", path)
    logger.info("DQ report: %s", dq)


if __name__ == "__main__":
    run()
