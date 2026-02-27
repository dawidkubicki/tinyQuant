# tinyQuant

Data foundation for an algorithmic trading stack (Phase 1 only).

## What this contains

- Historical daily OHLCV ingestion from IBKR.
- Fundamental metrics ingestion from Twelve Data.
- Merge + forward fill that avoids look-ahead bias.
- Parquet storage for raw and processed datasets.
- Two jobs:
  - `bootstrap_history`: initial 5-year backfill.
  - `update_daily`: incremental daily update.

## Setup

1. Create and activate a virtual environment:
   - `python -m venv .venv`
   - `source .venv/bin/activate`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Configure environment:
   - `cp .env.example .env`
   - fill in `TWELVE_DATA_API_KEY`
4. Make sure TWS / IB Gateway is running with API enabled on `7497`.

## Run

From repository root:

- Initial backfill:
  - `PYTHONPATH=src python -m tinyquant.jobs.bootstrap_history`
- Daily delta update:
  - `PYTHONPATH=src python -m tinyquant.jobs.update_daily`
- Data quality report:
  - `PYTHONPATH=src python -m tinyquant.jobs.report_data_quality`

## Output layout

- `data/raw/ibkr_daily/` - raw OHLCV parquet files.
- `data/raw/twelve_data_fundamentals/` - raw fundamentals parquet files.
- `data/processed/fact_market_daily/` - merged model-ready dataset.

## Daily schedule (example)

- 16:05 NY time: run `update_daily`.
- Optional: run `report_data_quality` and alert on failures.
