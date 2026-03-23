"""Placeholder sentiment — returns constant vector until real NLP feed is wired."""

from __future__ import annotations

import numpy as np

from tinyquant.config.schema import SentimentConfig


def sentiment_vector(n_tokens: int, cfg: SentimentConfig) -> np.ndarray:
    v = np.full(n_tokens, cfg.placeholder_value, dtype=np.float64)
    if not cfg.enabled:
        return v
    return v
