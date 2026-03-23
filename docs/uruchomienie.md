# Uruchomienie

## Wymagania

- Python **≥ 3.11**
- Repozytorium sklonowane lokalnie

## Instalacja

```bash
cd /ścieżka/do/tinyQuant
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Opcjonalnie tylko runtime (bez pytest/ruff):

```bash
pip install -e .
```

## Zmienne środowiskowe i `.env`

1. Skopiuj [`.env.example`](../.env.example) → `.env`.
2. Uzupełnij **tylko** jeśli uruchamiasz **live** (CCXT → giełda):

| Zmienna | Znaczenie |
|---------|-----------|
| `BINANCE_API_KEY` | Klucz API (przykład dla Binance) |
| `BINANCE_API_SECRET` | Sekret |
| `TINYQUANT_STRATEGY_YAML` | Opcjonalnie: pełna ścieżka do YAML strategii |
| `LOG_LEVEL` | np. `INFO`, `DEBUG` |

Klucze można też podać w CLI (`--api-key`, `--secret`) — mniej wygodne, nie zalecane w skryptach produkcyjnych (wyciek w historii poleceń).

## CLI: `tinyquant-h4`

Po instalacji edytowalnej dostępny jest entrypoint zdefiniowany w `pyproject.toml`:

### Jeden cykl (zalecany pierwszy test)

**Bez sieci i bez kluczy** — syntetyczne ceny:

```bash
tinyquant-h4 --config config/strategy.market_neutral.h4.yaml run-once --synthetic
```

**Z giełdą** (wymaga sieci + kluczy w `.env`):

```bash
tinyquant-h4 --config config/strategy.market_neutral.h4.yaml run-once
```

Parametry przydatne przy testach:

| Parametr | Opis |
|----------|------|
| `--equity` | Equity konta w USD do sizingu i ryzyka (domyślnie `10000`) |
| `--synthetic` | Pomija CCXT, losuje panel cen |

### Bootstrap modeli reżimu (losowe dane)

```bash
tinyquant-h4 --config config/strategy.market_neutral.h4.yaml train-regime --samples 800 --seed 42
```

Zobacz [Szkolenie i artefakty](szkolenie-i-artyfakty.md) — to **nie** jest trening na prawdziwym rynku.

## Testy automatyczne

```bash
pytest -q
```

## Newsy i Ollama (opcjonalnie)

1. Uruchom lokalnie [Ollama](https://ollama.com) i pobierz model zgodny z `sentiment.news.ollama_model` w YAML (np. `ollama pull llama3`).
2. W strategii ustaw `sentiment.enabled: true` oraz ścieżkę `sentiment.news.sqlite_path` (domyślnie `./data/news_sentiment.db`).
3. Jednorazowy zrzut RSS + klasyfikacja:

```bash
tinyquant-h4 --config config/strategy.market_neutral.h4.yaml news-sync
```

4. Pętla w tle (np. co 15 min — `poll_interval_seconds: 900`):

```bash
tinyquant-h4 --config config/strategy.market_neutral.h4.yaml news-loop
```

Cykl `run-once` **nie** woła Ollamy — tylko czyta zagregowany sentyment z SQLite. Host API możesz nadpisać zmienną `OLLAMA_HOST` (standard biblioteki `ollama`); adres w YAML (`ollama_host`) jest przekazywany explicite z konfiguracji.

## Harmonogram co 4 godziny

Obecnie repo nie uruchamia demona — cykl wywołujesz **ręcznie** lub z **cron** / **systemd timer** / schedulera w chmurze, np.:

```bash
0 */4 * * * cd /path/to/tinyQuant && .venv/bin/tinyquant-h4 --config config/strategy.market_neutral.h4.yaml run-once
```

Dostosuj czas do **zamknięcia świecy H4** na wybranej strefie (domyślnie logika w kodzie zakłada pracę w UTC).

Helper: `src/tinyquant/orchestration/scheduler.py` — `seconds_until_next_bar_close` dla przybliżonego odliczenia do kolejnej granicy 4h UTC.

## Tryb paper vs live

- W [`config/strategy.market_neutral.h4.yaml`](../config/strategy.market_neutral.h4.yaml) pole `execution.paper: true` sprawia, że **egzekucja** jest tylko zapisana w executorze paper (brak `create_order` przez CCXT w tej warstwie).
- Włączenie prawdziwych zleceń wymaga **rozbudowy** `execution` o wywołania CCXT i ścisłych kontroli ryzyka — poza minimalnym zakresem obecnej implementacji.

## Typowe problemy

| Problem | Działanie |
|---------|-----------|
| `Strategy YAML not found` | Uruchom z **katalogu głównym repo** lub ustaw `TINYQUANT_STRATEGY_YAML` / `--config` na absolutną ścieżkę. |
| XGBoost nie ładuje się na macOS | `brew install libomp` (patrz README i [Szkolenie](szkolenie-i-artyfakty.md)). |
| Brak `fetchFundingRateHistory` na giełdzie | Funding w sygnale będzie zerowy; rozważ zmianę `exchange.id` w YAML. |
