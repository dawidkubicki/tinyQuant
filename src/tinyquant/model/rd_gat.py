"""
Regime-dependent scoring: optional numpy checkpoint per regime; else deterministic blend.

Full GAT training is out of scope for bootstrap; checkpoints can be added as .npz:
  W: (F, H), u: (H,), v: (H,) for attention logits a_ij = (x_i W u)(x_j W v) / sqrt(H)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from tinyquant.config.schema import RDGATConfig, SentimentConfig

logger = logging.getLogger(__name__)


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    m, s = np.nanmean(x), np.nanstd(x)
    if not np.isfinite(s) or s < 1e-12:
        return np.zeros_like(x)
    return (x - m) / s


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(np.clip(z, -30, 30))
    row_sums = e.sum(axis=1, keepdims=True)
    degenerate = row_sums.squeeze() < 1e-12
    out = e / (row_sums + 1e-12)
    if np.any(degenerate):
        n = out.shape[1]
        out[degenerate] = 1.0 / n
    return out


def _gat_forward(
    x: np.ndarray,
    adj: np.ndarray,
    bundle: dict[str, np.ndarray],
) -> np.ndarray:
    """Single-layer GAT-style aggregation; x (N,F), adj (N,N) binary or weighted."""
    w = bundle["W"]  # (F, H)
    a_src = bundle["a_src"]  # (H,)
    a_dst = bundle["a_dst"]  # (H,)
    n, fdim = x.shape
    h = x @ w  # (N,H)
    # e_ij = LeakyReLU((h_i + h_j) dot a) simplified: dot(h_i, a_src) + dot(h_j, a_dst)
    qi = h @ a_src  # (N,)
    kj = h @ a_dst  # (N,)
    logits = qi[:, None] + kj[None, :]
    logits = np.tanh(logits) * 2.0
    masked = np.where(adj > 0, logits, -1e9)
    att = _softmax_rows(masked)
    out = att @ h  # (N,H)
    if "W_out" in bundle:
        out = out @ bundle["W_out"]  # (H,1) or (H,)
    return out.ravel()


def _load_regime_bundle(checkpoint_dir: Path, regime: int) -> dict[str, np.ndarray] | None:
    p = checkpoint_dir / f"regime_{regime}.npz"
    if not p.is_file():
        return None
    try:
        return dict(np.load(p, allow_pickle=False))
    except Exception as e:
        logger.warning("Failed loading RD-GAT checkpoint %s: %s", p, e)
        return None


def infer_token_scores(
    diffusion_scores: np.ndarray,
    funding_features: np.ndarray,
    sentiment_per_token: np.ndarray,
    adjacency: np.ndarray,
    regime_id: int,
    cfg: RDGATConfig,
    sentiment_cfg: SentimentConfig,
) -> np.ndarray:
    """
    Per-token score in [-1, 1]. Uses checkpoint if present; else weighted z-score blend.
    """
    n = len(diffusion_scores)
    d = _zscore(diffusion_scores)
    f = _zscore(funding_features)
    s = np.asarray(sentiment_per_token, dtype=np.float64).ravel()
    if len(s) != n:
        s = np.resize(s, (n,))

    cp_dir = Path(cfg.checkpoint_dir)
    bundle = _load_regime_bundle(cp_dir, regime_id)
    if bundle is not None and all(k in bundle for k in ("W", "a_src", "a_dst")):
        x = np.stack([d, f, s], axis=1)
        raw = _gat_forward(x, adjacency, bundle)
        out = np.tanh(raw / cfg.score_temperature)
        return np.clip(out, -1.0, 1.0)

    b = cfg.fallback_blend
    if sentiment_cfg.enabled:
        raw = b.diffusion_weight * d + b.funding_weight * f + b.sentiment_weight * s
    else:
        wsum = b.diffusion_weight + b.funding_weight
        raw = (b.diffusion_weight / wsum) * d + (b.funding_weight / wsum) * f
    out = np.tanh(raw / cfg.score_temperature)
    return np.clip(out, -1.0, 1.0)
