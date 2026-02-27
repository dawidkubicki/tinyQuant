from __future__ import annotations

import logging

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
    logger.info("Bootstrap started for %d tickers", len(tickers))

    with IBKRClient(
        host=settings.ibkr_host,
        port=settings.ibkr_port,
        client_id=settings.ibkr_client_id,
        request_sleep_seconds=settings.ibkr_request_sleep_seconds,
    ) as ibkr:
        prices = ibkr.fetch_universe_daily_ohlcv(tickers=tickers, years_back=5)

    fundamentals_client = TwelveDataFundamentalsClient(settings.twelve_data_api_key)
    fundamentals = fundamentals_client.fetch_universe_fundamentals(tickers=tickers)

    fact = merge_prices_with_fundamentals(prices=prices, fundamentals=fundamentals)

    store.write_raw_ibkr(prices)
    store.write_raw_fundamentals(fundamentals)
    path = store.write_fact(fact)

    dq = data_quality_report(fact)
    logger.info("Bootstrap completed. Fact saved to %s", path)
    logger.info("DQ report: %s", dq)


if __name__ == "__main__":
    run()
