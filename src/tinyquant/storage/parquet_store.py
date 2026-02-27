from __future__ import annotations

from pathlib import Path

import pandas as pd


class ParquetStore:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.raw_ibkr_dir = self.root_dir / "raw" / "ibkr_daily"
        self.raw_fund_dir = self.root_dir / "raw" / "twelve_data_fundamentals"
        self.fact_dir = self.root_dir / "processed" / "fact_market_daily"

    def ensure_layout(self) -> None:
        self.raw_ibkr_dir.mkdir(parents=True, exist_ok=True)
        self.raw_fund_dir.mkdir(parents=True, exist_ok=True)
        self.fact_dir.mkdir(parents=True, exist_ok=True)

    def write_raw_ibkr(self, frame: pd.DataFrame) -> Path:
        self.ensure_layout()
        path = self.raw_ibkr_dir / "ibkr_daily.parquet"
        frame.to_parquet(path, index=False)
        return path

    def write_raw_fundamentals(self, frame: pd.DataFrame) -> Path:
        self.ensure_layout()
        path = self.raw_fund_dir / "fundamentals.parquet"
        frame.to_parquet(path, index=False)
        return path

    def write_fact(self, frame: pd.DataFrame) -> Path:
        self.ensure_layout()
        path = self.fact_dir / "fact_market_daily.parquet"
        frame.to_parquet(path, index=False)
        return path

    def read_raw_ibkr(self) -> pd.DataFrame:
        path = self.raw_ibkr_dir / "ibkr_daily.parquet"
        if not path.exists():
            return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])
        return pd.read_parquet(path)

    def read_raw_fundamentals(self) -> pd.DataFrame:
        path = self.raw_fund_dir / "fundamentals.parquet"
        if not path.exists():
            return pd.DataFrame(columns=["publish_date", "ticker", "eps", "pe_ratio", "debt_to_equity", "revenue_growth"])
        return pd.read_parquet(path)

    def read_fact(self) -> pd.DataFrame:
        path = self.fact_dir / "fact_market_daily.parquet"
        if not path.exists():
            return pd.DataFrame(
                columns=[
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
            )
        return pd.read_parquet(path)

    def upsert_fact(self, fresh_rows: pd.DataFrame) -> Path:
        existing = self.read_fact()
        if existing.empty:
            return self.write_fact(fresh_rows.sort_values(["ticker", "date"]).reset_index(drop=True))

        combined = pd.concat([existing, fresh_rows], ignore_index=True)
        combined = combined.sort_values(["ticker", "date"]).drop_duplicates(subset=["date", "ticker"], keep="last")
        return self.write_fact(combined.reset_index(drop=True))

    def last_fact_date(self) -> pd.Timestamp | None:
        fact = self.read_fact()
        if fact.empty:
            return None
        return pd.to_datetime(fact["date"]).max()
