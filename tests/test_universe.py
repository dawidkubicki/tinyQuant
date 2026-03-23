from __future__ import annotations

from typing import Any

from tinyquant.config.schema import UniverseConfig
from tinyquant.data.universe import select_universe_symbols


def _market(
    *,
    swap: bool,
    linear: bool,
    quote: str,
    settle: str | None,
    active: bool = True,
    mid: str | None = None,
) -> dict[str, Any]:
    return {
        "swap": swap,
        "linear": linear,
        "quote": quote,
        "quoteId": quote.lower(),
        "settle": settle,
        "active": active,
        "id": mid or "x",
    }


class _FakeEx:
    def __init__(self, markets: dict[str, dict[str, Any]], tickers: dict[str, dict[str, Any]]):
        self._markets = markets
        self._tickers = tickers

    def load_markets(self) -> dict[str, dict[str, Any]]:
        return self._markets

    def fetch_tickers(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        return {s: self._tickers.get(s, {}) for s in symbols}


def test_select_universe_linear_usd_sorted_and_benchmark_pinned() -> None:
    cfg = UniverseConfig(
        top_n=5,
        quote_asset="USD",
        min_quote_volume_usd_24h=100.0,
        benchmark_symbol="BTC/USD:USD",
        blacklist=[],
    )
    markets = {
        "BTC/USD:USD": _market(swap=True, linear=True, quote="USD", settle="USD", mid="pf_btc"),
        "ETH/USD:USD": _market(swap=True, linear=True, quote="USD", settle="USD"),
        "SOL/USD:USD": _market(swap=True, linear=True, quote="USD", settle="USD"),
        "BTC/USD:BTC": _market(swap=True, linear=False, quote="USD", settle="BTC"),
    }
    # BTC below volume threshold → dropped from top-by-volume list, then pinned as benchmark
    tickers = {
        "BTC/USD:USD": {"quoteVolume": 50.0},
        "ETH/USD:USD": {"quoteVolume": 900.0},
        "SOL/USD:USD": {"quoteVolume": 800.0},
    }
    ex = _FakeEx(markets, tickers)
    out = select_universe_symbols(ex, cfg)
    assert out[0] == "BTC/USD:USD"
    assert set(out) == {"BTC/USD:USD", "ETH/USD:USD", "SOL/USD:USD"}


def test_select_universe_skips_non_linear_and_volume_threshold() -> None:
    cfg = UniverseConfig(
        top_n=10,
        quote_asset="USD",
        min_quote_volume_usd_24h=1_000.0,
        benchmark_symbol="BTC/USD:USD",
    )
    markets = {
        "BTC/USD:USD": _market(swap=True, linear=True, quote="USD", settle="USD"),
        "ETH/USD:USD": _market(swap=True, linear=True, quote="USD", settle="USD"),
    }
    tickers = {
        "BTC/USD:USD": {"quoteVolume": 2_000.0},
        "ETH/USD:USD": {"quoteVolume": 100.0},
    }
    ex = _FakeEx(markets, tickers)
    out = select_universe_symbols(ex, cfg)
    assert out == ["BTC/USD:USD"]


def test_select_universe_settle_mismatch_excluded() -> None:
    cfg = UniverseConfig(top_n=5, quote_asset="USD", benchmark_symbol="BTC/USD:USD")
    markets = {
        "WEIRD/USD:EUR": _market(swap=True, linear=True, quote="USD", settle="EUR"),
    }
    tickers = {"WEIRD/USD:EUR": {"quoteVolume": 1e9}}
    ex = _FakeEx(markets, tickers)
    out = select_universe_symbols(ex, cfg)
    assert out == []
