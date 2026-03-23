# tinyQuant

Modular **H4 market-neutral** engine for **crypto perpetual futures** on **Kraken Futures** (CCXT `krakenfutures`, USD-margined linear perps in symbols; fund margin with **USDC** on Kraken — no USDT in config): data ingestion (CCXT), beta-neutral residuals, graph diffusion mispricing, TDA landscape features, GMM/XGBoost regime detection, regime-dependent scoring (RD-GAT checkpoint or deterministic fallback), volatility-parity long/short book, paper execution, and portfolio-level risk cooldown.

**Pełna dokumentacja (PL):** zobacz katalog [`docs/`](docs/README.md) — architektura, szkolenie modeli, uruchomienie, konfiguracja YAML.

## Quick start

```bash
cd /path/to/tinyQuant
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

### Configuration

- **Strategy parameters**: [`config/strategy.market_neutral.h4.yaml`](config/strategy.market_neutral.h4.yaml) (validated by Pydantic — see `src/tinyquant/config/schema.py`).
- **Secrets**: copy `.env.example` → `.env` and set exchange API keys if you run live data.

### CLI

```bash
# Paper / synthetic single cycle (no exchange I/O)
tinyquant-h4 --config config/strategy.market_neutral.h4.yaml run-once --synthetic

# Live cycle (requires Kraken Futures API keys + network)
export KRAKEN_API_KEY=...
export KRAKEN_API_SECRET=...
tinyquant-h4 --config config/strategy.market_neutral.h4.yaml run-once

# Bootstrap regime models (synthetic features — replace with your historical pipeline)
tinyquant-h4 --config config/strategy.market_neutral.h4.yaml train-regime

# News: RSS → Ollama (JSON) → SQLite (requires local Ollama + model, e.g. llama3)
tinyquant-h4 --config config/strategy.market_neutral.h4.yaml news-sync
```

On **macOS**, if `train-regime` fails to load XGBoost (`libomp.dylib`), install OpenMP: `brew install libomp`.

## Layout

| Path | Role |
|------|------|
| `src/tinyquant/data/` | CCXT OHLCV + funding, universe selection |
| `src/tinyquant/features/` | Returns, beta-neutral epsilon |
| `src/tinyquant/signals/` | Graph diffusion, TDA landscape, sentiment from news DB |
| `src/tinyquant/news/` | RSS ingest, Ollama classifier, SQLite store, decay aggregation |
| `src/tinyquant/regime/` | GMM (offline) + XGBoost (online) |
| `src/tinyquant/model/` | RD-GAT checkpoint inference + fallback blend |
| `src/tinyquant/portfolio/` | Vol parity sizing, constraints, rebalance deltas |
| `src/tinyquant/execution/` | Paper executor + funding PnL helper |
| `src/tinyquant/risk/` | Daily loss tracking + cooldown |
| `src/tinyquant/orchestration/` | `run_h4_cycle` wiring |

## Disclaimer

This is research/engineering scaffolding. It is **not** investment advice. Live trading requires your own due diligence, exchange compliance, and risk controls.
