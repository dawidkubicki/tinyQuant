from __future__ import annotations

from tinyquant.config.loaders import load_strategy_config
from tinyquant.config.schema import StrategyConfig


def test_load_default_yaml() -> None:
    cfg = load_strategy_config()
    assert isinstance(cfg, StrategyConfig)
    assert cfg.runtime.cycle_interval == "4h"
    assert cfg.universe.top_n == 50


def test_fallback_blend_sums_to_one() -> None:
    cfg = load_strategy_config()
    b = cfg.rd_gat.fallback_blend
    assert abs(b.diffusion_weight + b.funding_weight + b.sentiment_weight - 1.0) < 1e-9
