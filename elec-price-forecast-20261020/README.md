# Electricity Price Forecasting (DE-LU)

Day-ahead / week-ahead electricity price forecasting with a rigorous
walk-forward evaluation protocol, plus a small FastAPI inference service.

Built as a portfolio project for **mid/long-term price & load forecasting roles
in power trading** (中长期电价及负荷预测、交易策略支撑方向).

---

## Results (rolling-origin out-of-sample, 90 daily origins, 2026-07-20 → 2026-10-18)

### Day-ahead (h = 1..24)

| model | MAE | RMSE | sMAPE | R² |
|---|---:|---:|---:|---:|
| **LightGBM (direct, change-model)** | **30.61** | **45.76** | **38.9%** | **0.62** |
| Naive 24h (same hour yesterday) | 36.38 | 54.71 | 49.9% | 0.45 |
| Weekly profile (4-week median) | 36.56 | 52.18 | 48.5% | 0.50 |
| Naive 168h (same hour last week) | 44.99 | 65.24 | 56.1% | 0.22 |

→ **−15.9% MAE and −16.4% RMSE vs the strongest naive baseline.**

### Week-ahead (h = 1..168)

| model | MAE | RMSE | sMAPE | R² |
|---|---:|---:|---:|---:|
| **LightGBM (direct, change-model)** | **36.88** | **53.27** | **41.1%** | **0.48** |
| Weekly profile (4-week median) | 37.07 | 52.70 | 47.9% | 0.49 |
| Naive 168h | 46.26 | 66.65 | 56.0% | 0.18 |

### Mid / long-term (walk-forward)

| horizon | champion | MAE | benchmark beaten |
|---|---|---:|---|
| next-week average | 4-week moving average | **16.66 EUR/MWh** | Ridge ML challenger: 21.40 |
| next-month average | last-month persistence | **14.72 EUR/MWh** | Ridge ML challenger: 18.02 |

The ML challenger **lost** to simple benchmarks at weekly/monthly granularity
in this window, so the benchmark stays champion. A model is only promoted if it
beats the incumbent out-of-sample — that is the whole point of the protocol.

### Ablation: do fundamentals help?

| feature set | day-ahead MAE | week-ahead MAE |
|---|---:|---:|
| price-only | 31.18 | 37.67 |
| price + residual load / load / renewables (lagged 24h) | **30.61** | **36.88** |

Fundamentals add ~1.8–2.1% MAE. `load_vs_7d` ranks in the top-10 features by
split gain. Real forecasts of residual load (not lagged actuals) would add more.

---

## Methodology (the part that matters in interviews)

1. **Gate-closure information set.** Forecast origins are aligned to **23:00
   local (D-1)** — the last price observable at the noon day-ahead auction gate
   closure — so h=1 maps to 00:00 of the delivery day. Nothing from the
   delivery day leaks into the features. A leakage unit test proves it.

2. **Change modelling, not level modelling.** Prices drifted from ~80 to
   ~180 EUR/MWh within the sample. Tree models cannot extrapolate a level they
   have never seen (first version of this model had bias ≈ −47 EUR/MWh for
   exactly that reason). The model therefore predicts the **change relative to
   the last observed price**, and all price features are gaps to that anchor.

3. **Direct multi-horizon.** One LightGBM model serves h = 1..168 with the
   horizon as a feature, instead of recursive one-step rolling. Direct
   forecasting avoids compounding one-step errors at 168h.

4. **Rolling-origin backtest, no random splits.** Random splits on hourly data
   leak through neighbouring hours. 84 daily test origins, each predicted from
   its own information set.

5. **sMAPE/WAPE instead of MAPE.** 5.8% of hours in this market clear at
   **negative prices** — MAPE is undefined/nonsense around zero.

6. **Champion / challenger for mid-term.** Simple benchmarks (moving average,
   persistence) are champions; ML challengers are promoted only if they win
   out-of-sample. They didn't at weekly/monthly granularity — reported as is.

7. **Error by price regime.** The model over-forecasts negative hours by
   ≈ +68 EUR and under-forecasts spikes by ≈ −157 EUR (regression to the mean).
   For trading use, that asymmetry calls for quantile models or a regime
   classifier on top — documented as the main known limitation.

## Known limitations (stated honestly)

- Spike hours are systematically under-predicted (L1 loss + rare regime).
- Mid/long-term ML did not beat simple benchmarks here; the benchmark is kept.
- Only price history + lagged fundamentals are used; adding day-ahead
  *forecasts* of residual load, cross-border flows and fuel prices is the
  clearest next lever.
- One market (DE-LU). The pipeline is market-agnostic by design
  (see `data/README.md` for the CSV schema to plug in 山西/广东/蒙西 spot data).

## Quick start

```bash
pip install -r requirements.txt

# 1. data (open API, no key, CC BY 4.0)
python scripts/fetch_data.py

# 2. backtest -> reports/ (metrics CSVs, figures, summary.json)
python scripts/run_backtest.py

# 3. train the serving model -> artifacts/model.joblib
python scripts/train_final.py

# 4. serve
uvicorn elecprice.service:app --app-dir src --port 8000
curl "http://localhost:8000/predict?horizon=24"
```

Run tests (feature leakage, alignment, metrics, API):

```bash
pytest tests -q
```

## Structure

```
src/elecprice/
  data.py       parsing, hourly normalisation, fundamentals join
  features.py   gap features, target calendar, direct dataset, leakage guard
  models.py     LightGBM direct multi-horizon + naive baselines
  backtest.py   rolling-origin evaluation, mid-term champion/challenger
  metrics.py    MAE/RMSE/sMAPE/WAPE/bias/R², by-horizon, by-price-regime
  plots.py      figures for reports/
  service.py    FastAPI /predict /model/info /health
scripts/        fetch_data / run_backtest / train_final
tests/          leakage, alignment, metrics, API smoke tests
```

## Data

Open data from [energy-charts.info](https://api.energy-charts.info)
(source: Bundesnetzagentur | SMARD.de), licence **CC BY 4.0**.
DE-LU day-ahead price 2024-01 → 2026-10, plus German generation/load
(residual load included). See `data/README.md`.
