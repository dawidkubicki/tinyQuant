# Dokumentacja tinyQuant

Materiały opisujące **fundusz krypto H4 (market-neutral, perpetual)** zaimplementowany w tym repozytorium.

| Dokument | Opis |
|----------|------|
| [Architektura i przepływ danych](architektura.md) | Jak działa „mózg” systemu krok po kroku (od CCXT do zleceń). |
| [Szkolenie modeli i artefakty](szkolenie-i-artyfakty.md) | Co trzeba wytrenować najpierw, kolejność, pliki wyjściowe, RD-GAT. |
| [Uruchomienie](uruchomienie.md) | Instalacja, zmienne środowiskowe, CLI, tryb paper / synthetic / live. |
| [Konfiguracja](konfiguracja.md) | Plik YAML, `.env`, gdzie uzupełniać parametry i ścieżki. |

## Szybka ścieżka

1. Zainstaluj zależności (patrz [Uruchomienie](uruchomienie.md)).
2. Skopiuj `.env.example` → `.env` (klucze tylko dla trybu live).
3. Dostosuj [`config/strategy.market_neutral.h4.yaml`](../config/strategy.market_neutral.h4.yaml).
4. Opcjonalnie: `train-regime` (bootstrap) lub własny pipeline cech historycznych — szczegóły w [Szkolenie](szkolenie-i-artyfakty.md).
5. `tinyquant-h4 … run-once --synthetic` — pierwszy suchy przebieg bez giełdy.

## Zastrzeżenie

Kod ma charakter **inżynierski / badawczy**. Nie stanowi porady inwestycyjnej. Handel instrumentami pochodnymi wiąże się z ryzykiem utraty kapitału.
