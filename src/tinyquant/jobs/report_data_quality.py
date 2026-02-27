from __future__ import annotations

import json
import logging

from tinyquant.config import get_settings
from tinyquant.pipeline.merge_and_fill import data_quality_report
from tinyquant.storage.parquet_store import ParquetStore


def run() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    logger = logging.getLogger(__name__)

    store = ParquetStore(settings.data_dir)
    fact = store.read_fact()
    report = data_quality_report(fact)
    logger.info("Data Quality Report:\n%s", json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
