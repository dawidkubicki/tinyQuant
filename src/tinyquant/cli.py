"""CLI: run-once H4 cycle, train regime models (synthetic bootstrap)."""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

from tinyquant.config.loaders import load_strategy_config
from tinyquant.orchestration.h4_cycle import run_h4_cycle
from tinyquant.regime.gmm_offline import fit_and_save_gmm
from tinyquant.regime.xgb_online import train_and_save_xgb


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def cmd_run_once(args: argparse.Namespace) -> int:
    cfg = load_strategy_config(args.config)
    _setup_logging(cfg.observability.log_level)
    api_key = args.api_key or os.environ.get("BINANCE_API_KEY")
    secret = args.secret or os.environ.get("BINANCE_API_SECRET")
    run_h4_cycle(
        cfg,
        equity_usd=args.equity,
        synthetic=args.synthetic,
        api_key=api_key,
        secret=secret,
    )
    return 0


def cmd_train_regime(args: argparse.Namespace) -> int:
    cfg = load_strategy_config(args.config)
    _setup_logging(cfg.observability.log_level)
    rng = np.random.default_rng(args.seed)
    n_regimes = cfg.regime.gmm.n_components
    dim = cfg.tda.landscape_resolution + 3
    n = args.samples
    X = rng.normal(size=(n, dim))
    gmm = fit_and_save_gmm(X, cfg.regime.gmm)
    y = gmm.predict(X)
    train_and_save_xgb(X, y, cfg.regime.xgboost, num_class=n_regimes)
    logging.getLogger(__name__).info("Saved GMM and XGBoost to data/models")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(prog="tinyquant-h4")
    p.add_argument("--config", type=str, default=None, help="Path to strategy YAML")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run-once", help="Execute a single H4 decision cycle")
    r.add_argument("--synthetic", action="store_true", help="Use synthetic prices (no exchange)")
    r.add_argument("--equity", type=float, default=10_000.0)
    r.add_argument("--api-key", type=str, default=None)
    r.add_argument("--secret", type=str, default=None)
    r.set_defaults(func=cmd_run_once)

    t = sub.add_parser("train-regime", help="Fit GMM + XGBoost on synthetic features (bootstrap)")
    t.add_argument("--samples", type=int, default=800)
    t.add_argument("--seed", type=int, default=42)
    t.set_defaults(func=cmd_train_regime)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
