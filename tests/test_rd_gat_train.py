from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from tinyquant.model.rd_gat_train import train_rd_gat_export_npz


def test_train_rd_gat_export_npz_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 8
    x = rng.normal(size=(n, 3))
    adj = (rng.random(size=(n, n)) > 0.6).astype(np.float64)
    np.fill_diagonal(adj, 0.0)
    y = rng.normal(size=(n,))
    pack = tmp_path / "train.npz"
    out = tmp_path / "regime_0.npz"
    np.savez(pack, x=x, adj=adj, y=y)
    train_rd_gat_export_npz(pack, out, hidden_dim=4, epochs=50, lr=0.1, seed=0)
    b = dict(np.load(out, allow_pickle=False))
    assert b["W"].shape == (3, 4)
    assert b["a_src"].shape == (4,)
    assert b["a_dst"].shape == (4,)
    assert b["W_out"].shape == (4, 1)
