"""Sentiment vector: placeholder or aggregated news scores from SQLite + decay."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from tinyquant.config.schema import SentimentConfig
from tinyquant.news.aggregation import aggregate_news_for_universe
from tinyquant.news.store_sqlite import NewsSQLiteStore

logger = logging.getLogger(__name__)


def sentiment_vector(
    tradable_symbols: list[str],
    cfg: SentimentConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Per-tradable-symbol sentiment in [-1, 1], aligned with `tradable_symbols` order.

    Returns (vector, debug_dict) for audit / regime hooks.
    """
    n = len(tradable_symbols)
    if not cfg.enabled:
        v = np.full(n, cfg.placeholder_value, dtype=np.float64)
        return v, {"mode": "disabled"}

    try:
        store = NewsSQLiteStore(cfg.news.sqlite_path)
        per_map, macro_val, agg_debug = aggregate_news_for_universe(
            store,
            tradable_symbols,
            window_hours=cfg.news.decay_window_hours,
            decay_base=cfg.news.decay_base,
            macro_blend_into_tokens=cfg.news.macro_blend_into_tokens,
        )
        vec = np.array([float(per_map.get(sym, cfg.placeholder_value)) for sym in tradable_symbols], dtype=np.float64)
        debug = {**agg_debug, "macro_aggregate": macro_val, "mode": "news_db"}
        return vec, debug
    except Exception as e:
        logger.warning("Sentiment from news DB failed: %s", e)
        v = np.full(n, cfg.placeholder_value, dtype=np.float64)
        return v, {"mode": "error", "error": str(e)}
