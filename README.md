# Cross-Sectional Equity Return ML

Machine-learning pipeline for monthly US large-cap equity return prediction and
out-of-sample portfolio backtesting (Jan 2000 – Dec 2023).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Symlink data files — see data/README.md

# Run full pipeline (~1–1.5 hours)
python run_all.py
```

Use `python run_all.py --skip-eda --skip-linear --skip-gbm` to rerun only
backtest and reporting after predictions exist.

## Data

Place input CSVs in `data/` (see `data/README.md` for setup):

| File | Description |
|------|-------------|
| `mma_sample_v2.csv` | Panel: ~1,000 stocks/month, 147 characteristics, target `stock_exret` |
| `factor_char_list.csv` | List of 147 predictor column names |
| `mkt_ind.csv` | Risk-free rate and S&P 500 monthly returns |

Predictors are lagged at time *t*; target `stock_exret` is excess return at *t+1*.

## Individual scripts

```bash
python scripts/check_data.py
python scripts/run_eda.py
python scripts/validate_features.py
python scripts/run_linear_baselines.py
python scripts/run_gbm.py
python scripts/run_backtest.py
python scripts/run_performance_report.py
```

## Project layout

```
src/           Core library (data loading, features, models, backtest)
scripts/       Runnable entry points
run_all.py     End-to-end pipeline entry point
data/          Input CSVs (gitignored; symlink or copy locally)
outputs/       Reports, plots, predictions (gitignored)
```

## Pipeline overview

1. **Data validation** — load panel, market, and factor list; run integrity checks
2. **Feature engineering** — cross-sectional rank normalization, imputation, scaling
3. **Model training** — expanding-window OOS forecasts (2010–2023)
4. **Portfolio backtest** — long-short, long-only, and alternative weighting schemes
5. **Performance reporting** — return, risk, alpha, Sharpe, drawdown, turnover, OOS R²

## Methodology (summary)

- **Data:** US large-cap panel, 147 lagged characteristics, monthly excess returns (2000–2023).
- **Features:** Cross-sectional median imputation, rank normalization to [-1, 1], train-only scaling.
- **Models:** Expanding-window refits (8yr train / 2yr val / 1yr OOS); Ridge, LASSO, OLS, EN + LightGBM (`RANDOM_STATE=42`).
- **Portfolios:** Long-short decile (50+50), long-only top-50, rank-weighted and vol-scaled long-75 (50–100 names).
- **Evaluation:** OOS R² (Gu-Kelly-Xiu, no mean subtraction), CAPM alpha (Newey-West, maxlags=3) vs S&P 500 excess.

## Results (OOS 2010–2023)

### Portfolios vs S&P 500 (excess)

| Name | Ann. Return | Ann. Stdev | Sharpe | CAPM α (ann.) | Info Ratio | Max DD | Max 1M Loss | Turnover |
|------|------------|------------|--------|---------------|------------|--------|-------------|----------|
| S&P 500 (excess) | 10.7% | 14.8% | 0.72 | 0.0% | −0.23 | 0.29 | −12.6% | — |
| Long-only top-50 | 14.1% | 15.2% | 0.92 | 4.4% | 0.61 | 0.28 | −14.6% | 41% |
| Rank-weighted long-75 | 14.1% | 15.2% | 0.93 | 4.5% | 0.63 | 0.28 | −14.7% | 38% |
| Vol-scaled long-75 | 13.4% | 14.4% | 0.93 | 4.1% | 0.65 | 0.29 | −15.1% | 38% |
| Long-short decile (50+50) | 6.2% | 22.8% | 0.27 | 10.8% | 0.49 | 1.35 | −16.4% | 33% |

### Prediction models (OOS R²)

| Model | OOS R² |
|-------|--------|
| Ridge | 0.71% |
| LASSO | 0.67% |
| Elastic Net | 0.67% |
| LightGBM | ~0% |
| OLS | −0.20% |

Artifacts: `outputs/reports/performance_table.csv`, `performance_report.md`, `cumulative_returns.png`.

## Reproducibility

- Pinned dependencies in `requirements.txt`
- Global seed `42` in `src/config.py` (LightGBM, NumPy)
- `expected_metrics.json` + verification step in `run_all.py`
