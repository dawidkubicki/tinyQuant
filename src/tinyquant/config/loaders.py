"""Load strategy YAML and merge with optional .env overrides."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import ValidationError

from tinyquant.config.schema import StrategyConfig

DEFAULT_STRATEGY_REL = Path("config/strategy.market_neutral.h4.yaml")
ENV_STRATEGY_PATH = "TINYQUANT_STRATEGY_YAML"


def load_strategy_config(
    yaml_path: Path | str | None = None,
    *,
    load_dotenv_file: bool = True,
) -> StrategyConfig:
    if load_dotenv_file:
        load_dotenv()

    path = yaml_path or os.environ.get(ENV_STRATEGY_PATH)
    if path:
        p = Path(path).expanduser().resolve()
    else:
        p = (Path.cwd() / DEFAULT_STRATEGY_REL).resolve()
        if not p.is_file():
            pkg_root = Path(__file__).resolve().parents[3]
            p = (pkg_root / DEFAULT_STRATEGY_REL).resolve()

    if not p.is_file():
        raise FileNotFoundError(f"Strategy YAML not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValidationError.from_exception_data(
            "StrategyConfig", [{"type": "dict_type", "loc": (), "input": raw}]
        )

    return StrategyConfig.model_validate(raw)
