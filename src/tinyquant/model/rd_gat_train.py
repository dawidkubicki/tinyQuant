"""
Train RD-GAT weights and export .npz compatible with infer_token_scores / _gat_forward.

Requires optional dependency: pip install '.[torch]'.
Training pack: .npz with arrays x (N, 3), adj (N, N), y (N,) — targets are proxy scores (e.g. forward returns).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as e:
        raise ImportError(
            "PyTorch is required for train-rd-gat. Install with: pip install 'tinyquant[torch]'"
        ) from e
    return torch, nn, F


def train_rd_gat_export_npz(
    training_npz: Path,
    output_npz: Path,
    *,
    hidden_dim: int = 8,
    epochs: int = 300,
    lr: float = 0.05,
    seed: int = 42,
) -> None:
    torch, nn, F = _torch()
    torch.manual_seed(seed)
    data = dict(np.load(training_npz, allow_pickle=False))
    for k in ("x", "adj", "y"):
        if k not in data:
            raise KeyError(f"training pack must contain '{k}'")
    x = np.asarray(data["x"], dtype=np.float64)
    adj = np.asarray(data["adj"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.float64).ravel()
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("x must have shape (N, 3) for diffusion, funding, sentiment z-scores")
    n = x.shape[0]
    if adj.shape != (n, n) or y.shape[0] != n:
        raise ValueError("adj must be (N, N) and y length N")

    x_t = torch.tensor(x, dtype=torch.float64)
    adj_t = torch.tensor(adj, dtype=torch.float64)
    y_t = torch.tensor(y, dtype=torch.float64)

    f_dim, h_dim = 3, int(hidden_dim)

    class RDGATTrain(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.W = nn.Parameter(torch.randn(f_dim, h_dim, dtype=torch.float64) * 0.05)
            self.a_src = nn.Parameter(torch.randn(h_dim, dtype=torch.float64) * 0.05)
            self.a_dst = nn.Parameter(torch.randn(h_dim, dtype=torch.float64) * 0.05)
            self.W_out = nn.Parameter(torch.randn(h_dim, 1, dtype=torch.float64) * 0.05)

        def forward(self, x_: torch.Tensor, a_: torch.Tensor) -> torch.Tensor:
            h = x_ @ self.W
            qi = h @ self.a_src
            kj = h @ self.a_dst
            logits = qi[:, None] + kj[None, :]
            logits = torch.tanh(logits) * 2.0
            masked = torch.where(a_ > 0, logits, torch.tensor(-1e9, dtype=logits.dtype, device=logits.device))
            att = F.softmax(masked, dim=1)
            agg = att @ h
            return (agg @ self.W_out).squeeze(-1)

    model = RDGATTrain()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        pred = model(x_t, adj_t)
        loss = F.mse_loss(pred, y_t)
        loss.backward()
        opt.step()
        if ep % 100 == 0:
            logger.info("epoch %s loss=%.6f", ep, float(loss.item()))

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        W = model.W.cpu().numpy()
        a_src = model.a_src.cpu().numpy()
        a_dst = model.a_dst.cpu().numpy()
        bundle: dict[str, Any] = {
            "W": W,
            "a_src": a_src,
            "a_dst": a_dst,
            "W_out": model.W_out.cpu().numpy(),
        }
    np.savez(output_npz, **bundle)
    logger.info("Wrote checkpoint %s", output_npz)
