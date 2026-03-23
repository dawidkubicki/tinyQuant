from __future__ import annotations

from pathlib import Path

import pytest

from tinyquant.config.loaders import load_strategy_config
from tinyquant.config.schema import StrategyConfig


@pytest.fixture
def strategy_config(tmp_path: Path) -> StrategyConfig:
    cfg = load_strategy_config()
    risk = cfg.risk.model_copy(update={"state_path": str(tmp_path / "risk_state.json")})
    obs = cfg.observability.model_copy(update={"audit_dir": str(tmp_path / "audit")})
    gmm = cfg.regime.gmm.model_copy(update={"model_path": str(tmp_path / "gmm.joblib")})
    xgb = cfg.regime.xgboost.model_copy(update={"model_path": str(tmp_path / "xgb.json")})
    regime = cfg.regime.model_copy(update={"gmm": gmm, "xgboost": xgb})
    return cfg.model_copy(update={"risk": risk, "observability": obs, "regime": regime})
