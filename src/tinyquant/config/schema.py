"""Pydantic schema for strategy YAML — validates the full architecture contract."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

class RuntimeConfig(BaseModel):
    timezone: str = "UTC"
    cycle_interval: Literal["4h", "1h", "1d"] = "4h"
    max_retries: int = Field(ge=1, le=50, default=5)
    retry_backoff_seconds: float = Field(gt=0, default=2.0)


class ExchangeConfig(BaseModel):
    id: Literal["krakenfutures"] = "krakenfutures"
    default_type: Literal["swap", "future"] = "swap"
    sandbox: bool = False
    enable_rate_limit: bool = True


class UniverseConfig(BaseModel):
    """Kraken Futures linear perpetuals use CCXT symbols like ``BASE/USD:USD`` (USD-margined).

    Fund the account with **USDC** (or other allowed collateral) on Kraken; USDT is not used by this project.
    """

    top_n: int = Field(ge=5, le=200, default=50)
    quote_asset: str = "USD"
    min_quote_volume_usd_24h: float = Field(ge=0, default=0.0)
    blacklist: list[str] = Field(default_factory=list)
    benchmark_symbol: str = "BTC/USD:USD"

    @field_validator("quote_asset")
    @classmethod
    def quote_asset_upper_no_usdt(cls, v: str) -> str:
        u = v.strip().upper()
        if u == "USDT":
            raise ValueError(
                "quote_asset USDT is not supported; use USD (Kraken Futures CCXT convention). "
                "Deposit/margin on Kraken with USDC in the UI."
            )
        if not u.isalnum() or len(u) < 2:
            raise ValueError("quote_asset must be a short alphanumeric code (e.g. USD).")
        return u

    @model_validator(mode="after")
    def benchmark_matches_quote(self) -> UniverseConfig:
        sym = self.benchmark_symbol.strip()
        if "/" not in sym or ":" not in sym:
            raise ValueError(
                "benchmark_symbol must be CCXT swap form BASE/QUOTE:SETTLE, e.g. BTC/USD:USD"
            )
        base_rest, settle_part = sym.split("/", 1)
        if ":" not in settle_part or not base_rest:
            raise ValueError("benchmark_symbol must be CCXT swap form BASE/QUOTE:SETTLE")
        quote, settle = settle_part.split(":", 1)
        if quote.upper() != self.quote_asset or settle.upper() != self.quote_asset:
            raise ValueError(
                f"benchmark_symbol quote/settle must match quote_asset={self.quote_asset!r}, got {sym!r}"
            )
        return self


class DataConfig(BaseModel):
    ohlcv_timeframe: str = "4h"
    ohlcv_lookback_bars: int = Field(ge=50, le=5000, default=500)
    funding_lookback_hours: int = Field(ge=24, le=8760, default=720)
    max_missing_bars_ratio: float = Field(ge=0.0, le=1.0, default=0.05)


class BetaNeutralizationConfig(BaseModel):
    lookback_bars: int = Field(ge=20, le=2000, default=120)
    method: Literal["ols"] = "ols"
    winsorize_std: float = Field(ge=0, default=4.0)


class GraphSignalConfig(BaseModel):
    correlation_lookback_bars: int = Field(ge=20, le=2000, default=120)
    edge_threshold: float = Field(ge=0.0, le=1.0, default=0.25)
    diffusion_steps: int = Field(ge=1, le=50, default=3)
    diffusion_alpha: float = Field(gt=0, le=1.0, default=0.3)
    score_clip: float = Field(gt=0, default=3.0)


class TDAConfig(BaseModel):
    correlation_lookback_bars: int = Field(ge=20, le=2000, default=120)
    max_dimension: int = Field(ge=0, le=2, default=1)
    max_edge_length: float = Field(gt=0, default=1.5)
    landscape_resolution: int = Field(ge=10, le=500, default=100)
    landscape_min: float = Field(ge=0.0, default=0.0)
    landscape_max: float = Field(gt=0, default=1.0)

    @field_validator("landscape_max")
    @classmethod
    def landscape_range(cls, v: float, info) -> float:
        lo = info.data.get("landscape_min", 0.0)
        if v <= lo:
            raise ValueError("landscape_max must be > landscape_min")
        return v


class GMMRegimeConfig(BaseModel):
    n_components: int = Field(ge=2, le=20, default=3)
    covariance_type: Literal["full", "tied", "diag", "spherical"] = "full"
    random_state: int = 42
    model_path: str = "./data/models/gmm_regime.joblib"


class XGBoostRegimeConfig(BaseModel):
    model_path: str = "./data/models/xgb_regime.json"
    n_estimators: int = Field(ge=10, default=100)
    max_depth: int = Field(ge=1, default=4)
    learning_rate: float = Field(gt=0, le=1.0, default=0.1)
    subsample: float = Field(gt=0, le=1.0, default=0.9)
    colsample_bytree: float = Field(gt=0, le=1.0, default=0.9)
    random_state: int = 42


class RegimeConfig(BaseModel):
    gmm: GMMRegimeConfig = Field(default_factory=GMMRegimeConfig)
    xgboost: XGBoostRegimeConfig = Field(default_factory=XGBoostRegimeConfig)
    min_regime_confidence: float = Field(ge=0.0, le=1.0, default=0.55)


class RDGATFallbackBlend(BaseModel):
    diffusion_weight: float = Field(ge=0.0, le=1.0, default=0.45)
    funding_weight: float = Field(ge=0.0, le=1.0, default=0.35)
    sentiment_weight: float = Field(ge=0.0, le=1.0, default=0.20)

    @model_validator(mode="after")
    def weights_sum_one(self) -> RDGATFallbackBlend:
        s = self.diffusion_weight + self.funding_weight + self.sentiment_weight
        if abs(s - 1.0) > 1e-6:
            raise ValueError("fallback_blend weights must sum to 1.0")
        return self


class RDGATConfig(BaseModel):
    hidden_dim: int = Field(ge=8, le=512, default=32)
    num_heads: int = Field(ge=1, le=16, default=4)
    dropout: float = Field(ge=0.0, le=0.9, default=0.1)
    checkpoint_dir: str = "./data/models/rd_gat"
    score_temperature: float = Field(gt=0, default=1.0)
    fallback_blend: RDGATFallbackBlend = Field(default_factory=RDGATFallbackBlend)


class PortfolioConfig(BaseModel):
    num_long: int = Field(ge=1, le=50, default=5)
    num_short: int = Field(ge=1, le=50, default=5)
    target_gross_exposure: float = Field(gt=0, le=10.0, default=1.0)
    max_net_exposure: float = Field(ge=0.0, le=1.0, default=0.05)
    max_leverage: float = Field(gt=0, le=125.0, default=3.0)
    volatility_lookback_bars: int = Field(ge=10, le=2000, default=60)
    min_notional_usd: float = Field(ge=0, default=20.0)


class ExecutionConfig(BaseModel):
    paper: bool = True
    order_type: Literal["market", "limit"] = "market"
    slippage_bps_cap: float = Field(ge=0, default=50)
    reduce_only_on_close: bool = True
    partial_fill_timeout_seconds: int = Field(ge=1, default=30)
    fee_bps_taker: float = Field(ge=0, default=5.0)


class RiskConfig(BaseModel):
    daily_portfolio_loss_limit_pct: float = Field(gt=0, le=100.0, default=3.0)
    close_all_on_breach: bool = True
    cooldown_hours_after_breach: int = Field(ge=0, le=168, default=24)
    state_path: str = "./data/risk_state.json"


class ObservabilityConfig(BaseModel):
    log_level: str = "INFO"
    audit_dir: str = "./data/audit"
    metrics_enabled: bool = True
    alert_webhook_url: str | None = None


class SentimentNewsFeedItem(BaseModel):
    name: str = Field(min_length=1)
    url: str = Field(min_length=8)


def _default_sentiment_news_feeds() -> list[SentimentNewsFeedItem]:
    return [
        SentimentNewsFeedItem(
            name="coindesk",
            url="https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml",
        ),
        SentimentNewsFeedItem(name="cryptoslate", url="https://cryptoslate.com/feed/"),
    ]


class SentimentNewsConfig(BaseModel):
    """RSS + Ollama + SQLite pipeline (see `tinyquant.news.worker`)."""

    sqlite_path: str = "./data/news_sentiment.db"
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3"
    ollama_timeout_seconds: float = Field(gt=0, default=120.0)
    ollama_max_retries: int = Field(ge=0, default=2)

    feeds: list[SentimentNewsFeedItem] = Field(default_factory=_default_sentiment_news_feeds)
    limit_per_feed: int = Field(ge=1, le=100, default=5)
    max_body_chars: int = Field(ge=0, le=100_000, default=1200)
    fetch_ld_json: bool = False
    rss_timeout_seconds: float = Field(gt=0, default=20.0)
    poll_interval_seconds: float = Field(ge=30.0, default=900.0)

    decay_window_hours: float = Field(gt=0, default=24.0)
    decay_base: float = Field(gt=0.0, le=1.0, default=0.9)
    macro_blend_into_tokens: float = Field(ge=0.0, le=1.0, default=0.35)

    append_macro_to_regime_features: bool = False
    user_agent: str = "tinyQuant-news/0.1 (+https://github.com/) research"


class SentimentConfig(BaseModel):
    enabled: bool = False
    placeholder_value: float = Field(ge=-1.0, le=1.0, default=0.0)
    news: SentimentNewsConfig = Field(default_factory=SentimentNewsConfig)


class StrategyConfig(BaseModel):
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    beta_neutralization: BetaNeutralizationConfig = Field(default_factory=BetaNeutralizationConfig)
    graph_signal: GraphSignalConfig = Field(default_factory=GraphSignalConfig)
    tda: TDAConfig = Field(default_factory=TDAConfig)
    regime: RegimeConfig = Field(default_factory=RegimeConfig)
    rd_gat: RDGATConfig = Field(default_factory=RDGATConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    sentiment: SentimentConfig = Field(default_factory=SentimentConfig)

    def resolved_paths(self, base_dir: Path | None = None) -> dict[str, Path]:
        root = base_dir or Path.cwd()
        return {
            "gmm": root / self.regime.gmm.model_path,
            "xgb": root / self.regime.xgboost.model_path,
            "rd_gat": root / self.rd_gat.checkpoint_dir,
            "risk_state": root / self.risk.state_path,
            "audit": root / self.observability.audit_dir,
        }
