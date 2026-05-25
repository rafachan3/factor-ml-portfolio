# Cross-Sectional Equity Return ML

Machine-learning pipeline for monthly US large-cap equity return prediction and
out-of-sample portfolio backtesting (Jan 2000 – Dec 2023).

## Data

Place input CSVs in `data/` (see `data/README.md` for setup):

| File | Description |
|------|-------------|
| `mma_sample_v2.csv` | Panel: ~1,000 stocks/month, 147 characteristics, target `stock_exret` |
| `factor_char_list.csv` | List of 147 predictor column names |
| `mkt_ind.csv` | Risk-free rate and S&P 500 monthly returns |

Predictors are lagged at time *t*; target `stock_exret` is excess return at *t+1*.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Validate input data integrity
python scripts/check_data.py

# Exploratory data analysis (plots + summary in outputs/eda/)
python scripts/run_eda.py

# Validate feature preprocessing pipeline
python scripts/validate_features.py

# Penalized linear baselines (OLS, LASSO, Ridge, EN) + OOS R²
python scripts/run_linear_baselines.py

# LightGBM advanced model + OOS R²
python scripts/run_gbm.py

# Portfolio backtests (ridge predictions by default)
python scripts/run_backtest.py
```

## Project layout

```
src/           Core library (data loading, features, models, backtest)
scripts/       Runnable entry points
data/          Input CSVs (gitignored; symlink or copy locally)
outputs/       Reports, plots, predictions (gitignored)
notebooks/     Optional exploratory notebooks
```

## Pipeline overview

1. **Data validation** — load panel, market, and factor list; run integrity checks
2. **Feature engineering** — cross-sectional rank normalization, imputation, scaling
3. **Model training** — expanding-window OOS forecasts (2010–2023)
4. **Portfolio backtest** — long-short, long-only, and alternative weighting schemes
5. **Performance reporting** — return, risk, alpha, Sharpe, drawdown, turnover, OOS R²

## Methodology (summary)

*To be completed after full pipeline run.*

## Results (OOS 2010–2023)

*To be completed after full pipeline run.*
