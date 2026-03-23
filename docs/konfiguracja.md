# Konfiguracja: YAML i miejsca do uzupełnienia

## Główny plik strategii

**Ścieżka:** [`config/strategy.market_neutral.h4.yaml`](../config/strategy.market_neutral.h4.yaml)

Jest to **jedyny kontrakt parametrów** ładowany przez `load_strategy_config()`. Struktura jest walidowana przez modele Pydantic w `src/tinyquant/config/schema.py` — błędny typ lub niedozwolona wartość zakończy się wyjątkiem przy starcie.

## Jak wskazać inny plik YAML

1. Flaga CLI: `--config /pełna/ścieżka/do/pliku.yaml`
2. Zmienna środowiska: `TINYQUANT_STRATEGY_YAML`
3. Domyślnie: `./config/strategy.market_neutral.h4.yaml` względem **bieżącego katalogu roboczego**, a przy braku pliku — próba względem korzenia pakietu (szczegóły w `loaders.py`).

## Sekcje YAML — co uzupełnić

### `runtime`

| Pole | Przykład | Uwagi |
|------|----------|--------|
| `timezone` | `UTC` | Informacyjne dla operatora; logika schedulera używa UTC. |
| `cycle_interval` | `4h` | Dozwolone wartości ograniczone schematem (np. `4h`). |
| `max_retries`, `retry_backoff_seconds` | `5`, `2.0` | Ponawianie zapytań CCXT przy błędach przejściowych. |

### `exchange`

| Pole | Uzupełnij |
|------|-----------|
| `id` | Id CCXT, np. `binance` |
| `default_type` | `swap` dla perpetual USDT-M |
| `sandbox` | `true` tylko jeśli giełda i klucze to sandbox |
| `enable_rate_limit` | Zwykle `true` |

**Kluczy API nie umieszczaj w YAML** — tylko w `.env` lub menedżerze sekretów.

### `universe`

| Pole | Uzupełnij |
|------|-----------|
| `top_n` | Liczba instrumentów (np. 50) |
| `min_quote_volume_usd_24h` | Próg odfiltrowania illikwidów |
| `blacklist` | Lista symboli CCXT do wykluczenia |
| `benchmark_symbol` | Musi być zgodny z konwencją CCXT, np. `BTC/USDT:USDT` |

### `data`

Okna na świece i funding; `max_missing_bars_ratio` — próg jakości panelu (obecnie głównie przy ręcznej rozbudowie walidacji).

### `beta_neutralization`

Długość okna regresji względem BTC, metoda (`ols`), winsoryzacja outlierów.

### `graph_signal`

Próg krawędzi korelacji, liczba kroków dyfuzji, \(\alpha\) dyfuzji, obcięcie score.

### `tda`

Parametry **ripser**: maksymalny wymiar homologii, próg `thresh`, rozdzielczość wektora wyjściowego, zakres siatki landscape w schemacie (wpływ na dokumentację cech — patrz kod `tda_global.py`).

### `regime`

| Podsekcja | Plik / pole | Uzupełnij |
|-----------|-------------|-----------|
| `gmm.model_path` | `.joblib` | Ścieżka zapisu/odczytu GMM |
| `gmm.n_components` | `3` | Liczba reżimów — **musi się zgadzać** z `num_class` XGBoost |
| `xgboost.model_path` | `.json` | Model XGBoost |
| `xgboost.*` | hiperparametry | Jak w sklearn/XGB API |
| `min_regime_confidence` | `0.55` | Logika informacyjna przy niskiej pewności |

### `rd_gat`

| Pole | Uzupełnij |
|------|-----------|
| `checkpoint_dir` | Katalog z `regime_0.npz`, … |
| `fallback_blend` | Trzy wagi sumujące się do **1.0** (walidacja Pydantic) |
| `score_temperature` | Skalowanie `tanh` |

### `portfolio`

Liczba long/short, `target_gross_exposure`, `max_net_exposure`, `max_leverage`, okno zmienności, minimalne delty notional.

### `execution`

`paper`, typ zlecenia, szacowane opłaty (`fee_bps_taker`), limity czasowe — pod przyszłą egzekucję live.

### `risk`

Dzienny limit straty w %, `close_all_on_breach`, czas cooldownu, ścieżka `state_path` (persistencja).

### `observability`

`log_level`, `audit_dir`, webhook (placeholder).

### `sentiment`

`enabled`, `placeholder_value` — do czasu podłączenia prawdziwego feedu.

## Schema Pydantic

Jeśli dodajesz nowe pole w YAML:

1. Dodaj je do odpowiedniego modelu w `src/tinyquant/config/schema.py`.
2. Uzupełnij domyślny plik [`config/strategy.market_neutral.h4.yaml`](../config/strategy.market_neutral.h4.yaml).
3. Użyj wartości w kodzie (np. w `h4_cycle.py`).

## Powiązanie z kodem

- Ładowanie: `tinyquant.config.loaders.load_strategy_config`
- Typ: `tinyquant.config.schema.StrategyConfig`
- Ścieżki względem CWD: metoda `resolved_paths()` na `StrategyConfig` (konsolidacja pod kątem narzędzi zewnętrznych).

## Checklist przed pierwszym live

- [ ] `.env` z kluczami tylko na środowisko testowe / ograniczone uprawnienia API (np. brak wypłat).
- [ ] `execution.paper: true` dopóki nie masz warstwy live orderów.
- [ ] `universe.blacklist` — tokeny wykluczone z polityki funduszu.
- [ ] `risk.daily_portfolio_loss_limit_pct` i `cooldown_hours_after_breach` zgodne z mandatem.
- [ ] Ścieżki `data/models/*` wskazują na właściwe artefakty (lub świadomie korzystasz z fallbacku).
