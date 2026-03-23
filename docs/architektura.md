# Architektura i przepływ danych

System realizuje **jeden cykl decyzyjny co interwał H4** (konfigurowalny w YAML). Poniżej logiczna kolejść modułów — odpowiada ona funkcji `run_h4_cycle` w `src/tinyquant/orchestration/h4_cycle.py`.

## Schemat wysokiego poziomu

```
Harmonogram H4
    → pobranie OHLCV + funding (CCXT)
    → uniwersum (Top-N płynność)
    → zwroty logarytmiczne
    → beta-neutralizacja względem BTC → ε (idiosynkratyczne)
    → sygnał lokalny: graf korelacji ε + dyfuzja Laplace’a → mispricing
    → sygnał globalny: macierz odległości z korelacji → Vietoris–Rips (ripser) → wektor „landscape”
    → cechy reżimu (landscape + zmienność BTC + rezerwa na statystyki wolumenu)
    → detektor reżimu: XGBoost (jeśli plik modelu istnieje) albo GMM albo rozkład jednostajny
    → scoring tokenów: RD-GAT (checkpoint `.npz` per reżim) albo **fallback** (ważona kombinacja diffusion / funding / sentyment)
    → budowa portfela: volatility parity, long / short, limity gross / net / dźwigni
    → delty rebalansu vs aktualne pozycje (paper)
    → egzekucja paper + ewentualny kill-switch dzienny
    → zapis audytu JSON
```

## 1. Dane i uniwersum

- **Moduły:** `src/tinyquant/data/ccxt_client.py`, `universe.py`
- **Wejście:** konfiguracja giełdy (`exchange.*`), filtry uniwersum (`universe.*`).
- **Wyjście:** wyrównany panel cen zamknięcia (macierz czas × instrumenty), lista symboli (BTC jako benchmark jest traktowany osobno od „handlowalnych” altów w części sygnałów).

**Funding:** ostatnie stopy finansowania są pobierane per symbol (jeśli giełda wspiera `fetchFundingRateHistory` w CCXT). Brak danych → wektor fundingów wypełniany zerami.

## 2. Beta-neutralizacja

- **Moduł:** `src/tinyquant/features/beta_neutral.py`
- Dla każdego instrumentu (poza benchmarkiem BTC): regresja OLS  
  \(R_i = \alpha + \beta R_{BTC} + \epsilon\) na wspólnym oknie.
- **Wyjście:** macierz ε — na niej budowany jest graf korelacji i dyfuzja (nie na surowych cenach).

## 3. Sygnał lokalny (graf + dyfuzja)

- **Moduł:** `src/tinyquant/signals/graph_diffusion.py`
- Graf: węzły = alty (bez BTC w tej warstwie), krawędź jeśli \|korelacja(ε)\| ≥ próg z YAML.
- Dyfuzja: iteracje na **znormalizowanym Laplasjanie**; **mispricing** = obserwowany zwrot vs stan po dyfuzji (interpretacja mean-reversion w kodzie jest „tanio/drogo” względem sąsiadów).

## 4. Sygnał globalny (TDA)

- **Moduł:** `src/tinyquant/signals/tda_global.py`
- Z macierzy korelacji ε budowana jest macierz odległości (np. \(\sqrt{2(1-\rho)}\)).
- **ripser** liczy persystencję z limitem progu (`max_edge_length`) i wymiaru (`max_dimension`).
- Wynik jest **wektorem stałej długości** (`landscape_resolution` w YAML) — uproszczona reprezentacja diagramów (nie pełny persistence landscape w sensie teoretycznym, lecz wektor o stałym wymiarze pod XGBoost).

## 5. Detektor reżimu (GMM + XGBoost)

- **Moduły:** `src/tinyquant/regime/gmm_offline.py`, `xgb_online.py`
- **Offline (GMM):** segmentacja przestrzeni cech (historia) → etykiety klastrów.
- **Online (XGBoost):** klasyfikacja bieżącego wektora cech → prawdopodobieństwa reżimów.
- **W runtime:** jeśli istnieje wytrenowany XGBoost (plik z `regime.xgboost.model_path`), używany jest on; w przeciwnym razie próba GMM na bieżącej cechce; jeśli brak modeli — reżim traktowany jest jako niepewny (rozklad jednostajny).

Import **xgboost** jest **leniwy** (ładowany dopiero przy treningu/ładowaniu modelu), żeby środowiska bez poprawnie zlinkowanego OpenMP na macOS nadal mogły importować resztę pakietu.

## 6. RD-GAT / scoring końcowy

- **Moduł:** `src/tinyquant/model/rd_gat.py`
- Jeśli w `rd_gat.checkpoint_dir` istnieje plik `regime_{k}.npz` z wagami (`W`, `a_src`, `a_dst`, opcjonalnie `W_out`), używany jest uproszczony mechanizm attention na grafie.
- W przeciwnym razie działa **fallback**: standaryzacja cech (diffusion, funding, placeholder sentymentu) i blend wag z `rd_gat.fallback_blend`, następnie `tanh` → score w \([-1, 1]\).

**Sentyment:** na razie stały placeholder (`sentiment.*` w YAML); pole pod przyszłe newsy / social.

## 7. Portfel market-neutral

- **Moduły:** `src/tinyquant/portfolio/sizing.py`, `constraints.py`, `rebalance.py`
- Wybór **top long** i **top short** po score (z wyłączeniem BTC z listy stron).
- **Volatility parity** w obrębie nogi long i osobno short (odwrotność realizowanej zmienności na oknie).
- Skalowanie do limitów **target gross**, **max net**, **max leverage** (względem equity przekazanego do cyklu).

## 8. Egzekucja i ryzyko

- **Paper:** `src/tinyquant/execution/perp_executor.py` — aktualizacja mapy pozycji w notionalu USD i log „zleceń”.
- **Ryzyko dzienne:** `src/tinyquant/risk/portfolio_guards.py` — spadek equity względem **szczytu intraday** powyżej progu z YAML → zapis cooldownu, opcjonalnie `close_all` na executorze.

Uwaga: w obecnej wersji **equity** przekazywane do cyklu jest parametrem (`--equity`); w produkcji należy je podawać z rzeczywistego salda / equity konta futures, inaczej kill-switch nie odzwierciedla PnL.

## 9. Audyt

- Każdy przebieg zapisuje plik JSON w katalogu `observability.audit_dir` (domyślnie `./data/audit`), nazwa z timestampem — pełny zrzut decyzji pośrednich i końcowych.

## Katalog kodu (mapowanie)

| Ścieżka | Rola |
|---------|------|
| `src/tinyquant/config/` | Ładowanie YAML + walidacja Pydantic |
| `src/tinyquant/data/` | CCXT, uniwersum |
| `src/tinyquant/features/` | Zwroty, beta-neutral |
| `src/tinyquant/signals/` | Dyfuzja, TDA, sentyment (stub) |
| `src/tinyquant/regime/` | GMM, XGBoost |
| `src/tinyquant/model/` | RD-GAT / fallback |
| `src/tinyquant/portfolio/` | Sizing, constrainty, rebalans |
| `src/tinyquant/execution/` | Paper, funding helper |
| `src/tinyquant/risk/` | Kill-switch / cooldown |
| `src/tinyquant/orchestration/` | `run_h4_cycle`, helper schedulera |
| `src/tinyquant/cli.py` | `tinyquant-h4` |
