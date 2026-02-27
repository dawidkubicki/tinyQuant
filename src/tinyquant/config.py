from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    ibkr_host: str = Field(default="127.0.0.1", alias="IBKR_HOST")
    ibkr_port: int = Field(default=7497, alias="IBKR_PORT")
    ibkr_client_id: int = Field(default=1, alias="IBKR_CLIENT_ID")
    ibkr_request_sleep_seconds: float = Field(default=0.35, alias="IBKR_REQUEST_SLEEP_SECONDS")

    twelve_data_api_key: str = Field(alias="TWELVE_DATA_API_KEY")

    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")
    universe: str = Field(default="SPY,AAPL,MSFT", alias="UNIVERSE")

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @property
    def universe_list(self) -> list[str]:
        return [ticker.strip().upper() for ticker in self.universe.split(",") if ticker.strip()]


def get_settings() -> Settings:
    return Settings()
