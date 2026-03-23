from __future__ import annotations

import numpy as np
import pandas as pd

from tinyquant.data.ccxt_client import CCXTDataClient


def test_aligned_column_matrix_inner_join_and_order() -> None:
    ts = pd.date_range("2024-01-01", periods=5, freq="4h", tz="UTC")
    a = pd.DataFrame(
        {
            "ts": ts,
            "close": [1.0, 2.0, 3.0, 4.0, 5.0],
            "volume": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    b = pd.DataFrame(
        {
            "ts": ts[1:4],
            "close": [1.1, 2.1, 3.1],
            "volume": [11.0, 21.0, 31.0],
        }
    )
    dfs = {"S1": a, "S2": b}
    close_m, idx = CCXTDataClient.aligned_column_matrix(dfs, ["S1", "S2"], "close")
    vol_m, idx2 = CCXTDataClient.aligned_column_matrix(dfs, ["S1", "S2"], "volume")
    assert idx == idx2
    assert close_m.shape == (3, 2)
    assert np.allclose(close_m[:, 0], [2.0, 3.0, 4.0])
    assert np.allclose(close_m[:, 1], [1.1, 2.1, 3.1])
    assert np.allclose(vol_m[:, 0], [20.0, 30.0, 40.0])
    assert np.allclose(vol_m[:, 1], [11.0, 21.0, 31.0])


def test_close_prices_matrix_delegates_to_aligned() -> None:
    ts = pd.date_range("2024-01-01", periods=2, freq="4h", tz="UTC")
    dfs = {
        "A": pd.DataFrame({"ts": ts, "close": [1.0, 2.0], "volume": [1.0, 2.0]}),
        "B": pd.DataFrame({"ts": ts, "close": [3.0, 4.0], "volume": [3.0, 4.0]}),
    }
    c1, _ = CCXTDataClient.close_prices_matrix(dfs, ["A", "B"])
    c2, _ = CCXTDataClient.aligned_column_matrix(dfs, ["A", "B"], "close")
    assert np.allclose(c1, c2)
