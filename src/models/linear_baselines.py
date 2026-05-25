"""OLS, LASSO, Ridge, and Elastic Net with expanding-window OOS training."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from src.features import cross_sectional_rank_normalize
from src.metrics import oos_r2_no_mean
from src.paths import TARGET_COLUMN
from src.splits import iter_expanding_windows, split_panel_by_window

MODEL_NAMES = ("ols", "lasso", "ridge", "en")

# Hyperparameter grids from the reference penalized-linear workflow.
LASSO_LAMBDAS = np.arange(-4, 4.1, 0.1)
RIDGE_LAMBDAS = np.arange(-1, 8.1, 0.1)
EN_LAMBDAS = np.arange(-4, 4.1, 0.1)

ID_COLS = ("year", "month", "date", "permno", TARGET_COLUMN)


def _scale_splits(
    train: pd.DataFrame,
    validate: pd.DataFrame,
    test: pd.DataFrame,
    factors: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Fit StandardScaler on train and transform all splits."""
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[list(factors)])
    x_val = scaler.transform(validate[list(factors)])
    x_test = scaler.transform(test[list(factors)])
    return x_train, x_val, x_test, scaler


def _tune_lasso(
    x_train: np.ndarray,
    y_train_dm: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    y_mean: float,
) -> Lasso:
    """Select LASSO penalty via validation MSE (warm-started grid)."""
    model = Lasso(max_iter=1_000_000, fit_intercept=False, warm_start=True)
    best_mse = np.inf
    best_alpha = float(10 ** LASSO_LAMBDAS[0])
    for log_lambda in LASSO_LAMBDAS:
        alpha = float(10 ** log_lambda)
        model.set_params(alpha=alpha)
        model.fit(x_train, y_train_dm)
        preds = model.predict(x_val) + y_mean
        mse = mean_squared_error(y_val, preds)
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
    model.set_params(alpha=best_alpha)
    model.fit(x_train, y_train_dm)
    return model


def _tune_ridge(
    x_train: np.ndarray,
    y_train_dm: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    y_mean: float,
) -> Ridge:
    """Select Ridge penalty via validation MSE."""
    best_mse = np.inf
    best_model: Ridge | None = None
    for log_lambda in RIDGE_LAMBDAS:
        model = Ridge(alpha=(10**log_lambda) * 0.5, fit_intercept=False)
        model.fit(x_train, y_train_dm)
        preds = model.predict(x_val) + y_mean
        mse = mean_squared_error(y_val, preds)
        if mse < best_mse:
            best_mse = mse
            best_model = model
    assert best_model is not None
    return best_model


def _tune_elastic_net(
    x_train: np.ndarray,
    y_train_dm: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    y_mean: float,
) -> ElasticNet:
    """Select Elastic Net penalty via validation MSE (warm-started grid)."""
    model = ElasticNet(max_iter=1_000_000, fit_intercept=False, warm_start=True)
    best_mse = np.inf
    best_alpha = float(10 ** EN_LAMBDAS[0])
    for log_lambda in EN_LAMBDAS:
        alpha = float(10 ** log_lambda)
        model.set_params(alpha=alpha)
        model.fit(x_train, y_train_dm)
        preds = model.predict(x_val) + y_mean
        mse = mean_squared_error(y_val, preds)
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
    model.set_params(alpha=best_alpha)
    model.fit(x_train, y_train_dm)
    return model


def _fit_window_models(
    train: pd.DataFrame,
    validate: pd.DataFrame,
    test: pd.DataFrame,
    factors: Sequence[str],
    target: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Train all linear baselines for one expanding window."""
    x_train, x_val, x_test, _ = _scale_splits(train, validate, test, factors)

    y_train = train[target].values
    y_val = validate[target].values
    y_mean = float(np.mean(y_train))
    y_train_dm = y_train - y_mean

    predictions = test[list(ID_COLS)].copy()

    ols = LinearRegression(fit_intercept=False)
    ols.fit(x_train, y_train_dm)
    predictions["ols"] = ols.predict(x_test) + y_mean

    lasso = _tune_lasso(x_train, y_train_dm, x_val, y_val, y_mean)
    predictions["lasso"] = lasso.predict(x_test) + y_mean

    ridge = _tune_ridge(x_train, y_train_dm, x_val, y_val, y_mean)
    predictions["ridge"] = ridge.predict(x_test) + y_mean

    en = _tune_elastic_net(x_train, y_train_dm, x_val, y_val, y_mean)
    predictions["en"] = en.predict(x_test) + y_mean

    return predictions


def run_linear_baselines(
    panel: pd.DataFrame,
    factors: Sequence[str],
    target: str = TARGET_COLUMN,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run expanding-window OLS / LASSO / Ridge / EN baselines.

    Preprocessing:
      1. Cross-sectional rank normalization on the full panel (once).
      2. Per window: train-only StandardScaler, Y demeaning on train.

    Args:
        panel: Raw stock-month panel with non-null target.
        factors: Predictor column names.
        target: Target return column.
        verbose: Print progress per OOS year.

    Returns:
        Concatenated OOS predictions and pooled OOS R² by model.
    """
    if panel[target].isna().any():
        raise ValueError(f"Panel contains missing values in '{target}'.")

    if verbose:
        print("Rank-normalizing full panel (one-time)...", flush=True)
    ranked = cross_sectional_rank_normalize(panel, factors)

    fold_preds: list[pd.DataFrame] = []
    windows = list(iter_expanding_windows())

    for window in windows:
        if verbose:
            print(
                f"  OOS year {window.test_year} (fold {window.counter + 1}/"
                f"{len(windows)})...",
                flush=True,
            )
        train, validate, test = split_panel_by_window(ranked, window)
        if train.empty or validate.empty or test.empty:
            raise ValueError(f"Empty split for OOS year {window.test_year}.")
        fold_pred = _fit_window_models(train, validate, test, factors, target)
        fold_preds.append(fold_pred)

    predictions = pd.concat(fold_preds, ignore_index=True)
    y_true = predictions[target].values
    oos_r2 = {
        model: oos_r2_no_mean(y_true, predictions[model].values)
        for model in MODEL_NAMES
    }
    return predictions, oos_r2


def format_oos_r2_report(oos_r2: dict[str, float]) -> str:
    """Format OOS R² results as plain text."""
    lines = ["Out-of-Sample R² (2010–2023, no-mean-subtraction)", ""]
    for model in MODEL_NAMES:
        lines.append(f"  {model:6s}: {oos_r2[model]:.6f}")
    return "\n".join(lines)
