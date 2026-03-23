# Szkolenie modeli i artefakty

Ten dokument odpowiada na pytania: **co muszę wytrenować najpierw**, **gdzie leżą pliki** i **co działa bez treningu**.

## Co działa „od ręki” (bez własnych modeli)

| Element | Zachowanie bez plików modeli |
|---------|------------------------------|
| Beta-neutral, graf, dyfuzja, TDA | Zawsze liczone z danych rynkowych (lub synthetic). |
| Reżim | Brak XGB i GMM → **jednostajne** prawdopodobieństwa reżimów (niska pewność); log w konsoli przy niskiej `min_regime_confidence`. |
| Scoring tokenów | **Fallback** w `rd_gat.py` (blend diffusion + funding + placeholder sentymentu), wynik w \([-1,1]\). |
| Portfel + paper | Działa na score z fallbacku. |

**Wniosek:** możesz uruchomić pełny cykl (`run-once --synthetic` lub live) **bez wcześniejszego treningu** — jako pipeline deterministyczny + heurystyka końcowa.

## Kolejność treningu (zalecana produkcyjnie)

Docelowo modele powinny być uczone na **historii tego samego wektora cech**, który liczy runtime (`build_regime_feature_row` w `h4_cycle.py`):

1. **Zbuduj macierz cech historycznych** \(X \in \mathbb{R}^{T_{\text{hist}} \times D}\)  
   - \(D =\) `tda.landscape_resolution` **+ 3**  
   - te **+3** to: zmienność BTC na oknie + dwa sloty na statystyki wolumenu (obecnie przy braku wolumenu w cechach reżimu wstawiane są **zera** — warto to później uzupełnić w kodzie, patrz niżej).

2. **GMM (nienadzorowane)**  
   - Dopasuj `GaussianMixture` na \(X\).  
   - Zapis: ścieżka z `regime.gmm.model_path` (np. `./data/models/gmm_regime.joblib`).

3. **Etykiety pseudonadzorowane**  
   - \(y_t = \arg\max_k P(\text{reżim}=k \mid X_t)\) z GMM (lub `gmm.predict`).

4. **XGBoost (nadzorowane)**  
   - Trenuj klasyfikator wieloklasowy na \((X, y)\).  
   - Zapis: `regime.xgboost.model_path` (np. `./data/models/xgb_regime.json`).

5. **(Opcjonalnie) RD-GAT / checkpoints**  
   - Obecny kod oczekuje plików `regime_{k}.npz` w `rd_gat.checkpoint_dir` z kluczami opisanymi w `src/tinyquant/model/rd_gat.py` (`W`, `a_src`, `a_dst`, opcjonalnie `W_out`).  
   - **Nie ma** w repozytorium automatycznego treningu GAT — to osobny etap badawczy (PyTorch / własny skrypt eksportu wag do `.npz`).

## Komenda wbudowana: `train-regime`

```bash
tinyquant-h4 --config config/strategy.market_neutral.h4.yaml train-regime \
  --samples 800 \
  --seed 42
```

**Co robi:** generuje **losowe** \(X\) o wymiarze \(D\) (normalne), trenuje GMM, potem XGBoost na etykietach z GMM, zapisuje pliki pod ścieżkami z YAML.

**Po co:** szybki **smoke test** ścieżki zapisu/odczytu modeli, **nie** jako model rynkowy.

**Produkcja:** zastąp to własnym notebookiem / skryptem, który:
- ładuje historyczne świece,
- powtarza dokładnie logikę cech (TDA + vol BTC + wolumen),
- zapisuje GMM i XGBoost w **tych samych ścieżkach** co w YAML (lub zmień ścieżki w YAML).

## macOS a XGBoost

Jeśli `train-regime` kończy się błędem ładowania `libxgboost.dylib` / `libomp.dylib`:

```bash
brew install libomp
```

Import `xgboost` w projekcie jest **opóźniony** — reszta testów i `run-once --synthetic` działa bez działającego XGBoost.

## Gdzie fizycznie leżą artefakty

| Artefakt | Domyślna ścieżka (z YAML) |
|----------|---------------------------|
| GMM | `./data/models/gmm_regime.joblib` |
| XGBoost | `./data/models/xgb_regime.json` |
| RD-GAT (opcjonalnie) | `./data/models/rd_gat/regime_0.npz`, `regime_1.npz`, … |
| Stan ryzyka (cooldown) | `./data/risk_state.json` |
| Audyt cykli | `./data/audit/h4_cycle_*.json` |

Ścieżki są względem **katalogu roboczego** przy starcie procesu (chyba że podasz ścieżki absolutne w YAML).

## Co warto uzupełnić w kodzie (kolejne iteracje)

- **Wolumen w cechach reżimu:** `build_regime_feature_row` przyjmuje `volume_window`, ale `run_h4_cycle` obecnie przekazuje `None` — dwa ostatnie wymiary cechy to zera. Warto podać wolumen z OHLCV na tym samym oknie co TDA.
- **Prawdziwy trening RD-GAT:** osobny pipeline + eksport wag do `.npz` zgodny z `infer_token_scores`.
- **Scheduler produkcyjny:** `scheduler.py` zawiera helper czasu; pełna pętla (cron / systemd / cloud scheduler) jest poza minimalnym zakresem repo.
