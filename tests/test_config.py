from __future__ import annotations

import pytest
from pydantic import ValidationError

from tinyquant.config.loaders import load_strategy_config
from tinyquant.config.schema import ExchangeConfig, StrategyConfig, UniverseConfig


def test_load_default_yaml() -> None:
    cfg = load_strategy_config()
    assert isinstance(cfg, StrategyConfig)
    assert cfg.runtime.cycle_interval == "4h"
    assert cfg.universe.top_n == 50
    assert cfg.exchange.id == "krakenfutures"
    assert cfg.universe.quote_asset == "USD"
    assert cfg.universe.benchmark_symbol == "BTC/USD:USD"


def test_exchange_only_krakenfutures() -> None:
    ExchangeConfig(id="krakenfutures")
    with pytest.raises(ValidationError):
        ExchangeConfig.model_validate({"id": "binance"})


def test_universe_rejects_usdt_quote() -> None:
    with pytest.raises(ValidationError):
        UniverseConfig(quote_asset="USDT", benchmark_symbol="BTC/USDT:USDT")


def test_universe_benchmark_must_match_quote() -> None:
    with pytest.raises(ValidationError):
        UniverseConfig(quote_asset="USD", benchmark_symbol="BTC/USDT:USDT")


def test_fallback_blend_sums_to_one() -> None:
    cfg = load_strategy_config()
    b = cfg.rd_gat.fallback_blend
    assert abs(b.diffusion_weight + b.funding_weight + b.sentiment_weight - 1.0) < 1e-9
