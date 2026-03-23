"""CLI: run-once H4 cycle, run-loop, train regime models, train RD-GAT."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

from tinyquant.config.loaders import load_strategy_config
from tinyquant.model.rd_gat_train import train_rd_gat_export_npz
from tinyquant.news.worker import run_news_loop, sync_news_once
from tinyquant.orchestration.h4_cycle import run_h4_cycle
from tinyquant.orchestration.scheduler import seconds_until_next_bar_close
from tinyquant.regime.gmm_offline import fit_and_save_gmm
from tinyquant.regime.train_history import fit_regime_models_from_exchange
from tinyquant.regime.xgb_online import train_and_save_xgb


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _kraken_api_credentials(
    args_key: str | None,
    args_secret: str | None,
) -> tuple[str | None, str | None]:
    """Read Kraken Futures API key/secret from CLI args or env (CCXT ``apiKey`` / ``secret``)."""
    key = args_key or os.environ.get("KRAKEN_API_KEY") or os.environ.get("KRAKEN_FUTURES_API_KEY")
    secret = args_secret or os.environ.get("KRAKEN_API_SECRET") or os.environ.get(
        "KRAKEN_FUTURES_API_SECRET"
    )
    return key, secret


def cmd_run_once(args: argparse.Namespace) -> int:
    cfg = load_strategy_config(args.config)
    _setup_logging(cfg.observability.log_level)
    api_key, secret = _kraken_api_credentials(args.api_key, args.secret)
    run_h4_cycle(
        cfg,
        equity_usd=args.equity,
        synthetic=args.synthetic,
        api_key=api_key,
        secret=secret,
    )
    return 0


def cmd_run_loop(args: argparse.Namespace) -> int:
    cfg = load_strategy_config(args.config)
    _setup_logging(cfg.observability.log_level)
    log = logging.getLogger(__name__)
    if cfg.data.ohlcv_timeframe != "4h":
        log.error(
            "run-loop only supports alignment to 4h UTC bars; got ohlcv_timeframe=%s",
            cfg.data.ohlcv_timeframe,
        )
        return 2
    api_key, secret = _kraken_api_credentials(args.api_key, args.secret)
    margin = float(args.post_boundary_seconds)
    max_err = int(args.max_consecutive_errors)
    n_fail = 0
    first = True
    while True:
        try:
            if not (first and args.run_immediately):
                delay = seconds_until_next_bar_close("4h") + margin
                log.info("Sleeping %.0f s until next 4h UTC close (+margin)", delay)
                time.sleep(delay)
            first = False
            run_h4_cycle(
                cfg,
                equity_usd=args.equity,
                synthetic=args.synthetic,
                api_key=api_key,
                secret=secret,
            )
            n_fail = 0
        except KeyboardInterrupt:
            log.info("Stopping run-loop")
            return 0
        except Exception:
            log.exception("H4 cycle failed")
            n_fail += 1
            if max_err > 0 and n_fail >= max_err:
                log.error("Exiting after %s consecutive failures", n_fail)
                return 1


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


def cmd_train_regime_history(args: argparse.Namespace) -> int:
    cfg = load_strategy_config(args.config)
    _setup_logging(cfg.observability.log_level)
    api_key, secret = _kraken_api_credentials(args.api_key, args.secret)
    lookback = args.lookback if args.lookback is not None else cfg.data.ohlcv_lookback_bars
    fit_regime_models_from_exchange(
        cfg,
        api_key=api_key,
        secret=secret,
        lookback_bars=lookback,
        step=args.step,
        max_samples=args.max_samples,
    )
    return 0


def cmd_news_sync(args: argparse.Namespace) -> int:
    cfg = load_strategy_config(args.config)
    _setup_logging(cfg.observability.log_level)
    log = logging.getLogger(__name__)
    stats = sync_news_once(cfg.sentiment.news)
    log.info(
        "News sync: feeds=%s seen=%s classified=%s skipped_existing=%s failed=%s",
        stats.feeds_processed,
        stats.items_seen,
        stats.items_classified,
        stats.items_skipped_existing,
        stats.items_failed,
    )
    return 0


def cmd_news_loop(args: argparse.Namespace) -> int:
    cfg = load_strategy_config(args.config)
    _setup_logging(cfg.observability.log_level)
    log = logging.getLogger(__name__)
    log.info("Starting news loop (poll_interval=%ss)", cfg.sentiment.news.poll_interval_seconds)
    try:
        run_news_loop(cfg.sentiment.news)
    except KeyboardInterrupt:
        log.info("News loop stopped")
    return 0


def cmd_train_rd_gat(args: argparse.Namespace) -> int:
    cfg = load_strategy_config(args.config)
    _setup_logging(cfg.observability.log_level)
    out = Path(cfg.rd_gat.checkpoint_dir) / f"regime_{args.regime}.npz"
    try:
        train_rd_gat_export_npz(
            Path(args.data),
            out,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            lr=args.lr,
            seed=args.seed,
        )
    except ImportError as e:
        logging.getLogger(__name__).error("%s", e)
        return 2
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

    loop = sub.add_parser("run-loop", help="Sleep until each 4h UTC close, then run H4 cycle")
    loop.add_argument("--synthetic", action="store_true", help="Use synthetic prices (no exchange)")
    loop.add_argument("--equity", type=float, default=10_000.0)
    loop.add_argument("--api-key", type=str, default=None)
    loop.add_argument("--secret", type=str, default=None)
    loop.add_argument(
        "--post-boundary-seconds",
        type=float,
        default=12.0,
        help="Extra sleep after 4h boundary so the exchange candle is closed",
    )
    loop.add_argument(
        "--max-consecutive-errors",
        type=int,
        default=0,
        help="Exit after N consecutive cycle failures (0 = never exit)",
    )
    loop.add_argument(
        "--run-immediately",
        action="store_true",
        help="Run first cycle without waiting (then follow the 4h schedule)",
    )
    loop.set_defaults(func=cmd_run_loop)

    t = sub.add_parser("train-regime", help="Fit GMM + XGBoost on synthetic features (bootstrap)")
    t.add_argument("--samples", type=int, default=800)
    t.add_argument("--seed", type=int, default=42)
    t.set_defaults(func=cmd_train_regime)

    th = sub.add_parser(
        "train-regime-history",
        help="Fetch OHLCV, build historical regime features, fit GMM + XGB to YAML paths",
    )
    th.add_argument(
        "--lookback",
        type=int,
        default=None,
        help="OHLCV bars per symbol (default: strategy ohlcv_lookback_bars)",
    )
    th.add_argument("--step", type=int, default=4, help="Bar step between feature rows (TDA cost)")
    th.add_argument("--max-samples", type=int, default=None, help="Keep only the last N rows")
    th.add_argument("--api-key", type=str, default=None)
    th.add_argument("--secret", type=str, default=None)
    th.set_defaults(func=cmd_train_regime_history)

    tg = sub.add_parser(
        "train-rd-gat",
        help="Train RD-GAT from training .npz (x, adj, y) and write regime_{k}.npz",
    )
    tg.add_argument("--data", type=str, required=True, help="Training pack .npz with x, adj, y")
    tg.add_argument("--regime", type=int, required=True, help="Regime id for output filename")
    tg.add_argument("--hidden-dim", type=int, default=8)
    tg.add_argument("--epochs", type=int, default=300)
    tg.add_argument("--lr", type=float, default=0.05)
    tg.add_argument("--seed", type=int, default=42)
    tg.set_defaults(func=cmd_train_rd_gat)

    ns = sub.add_parser(
        "news-sync",
        help="Fetch RSS headlines, classify with Ollama, store JSON in SQLite (one shot)",
    )
    ns.set_defaults(func=cmd_news_sync)

    nl = sub.add_parser(
        "news-loop",
        help="Run news-sync on poll_interval_seconds (default 900s) until interrupted",
    )
    nl.set_defaults(func=cmd_news_loop)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
