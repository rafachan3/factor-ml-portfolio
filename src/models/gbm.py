"""LightGBM gradient boosted trees with expanding-window OOS training."""

from __future__ import annotations

import itertools
from typing import Any, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE
from src.features import cross_sectional_rank_normalize
from src.metrics import oos_r2_no_mean
from src.paths import TARGET_COLUMN
from src.splits import iter_expanding_windows, split_panel_by_window
MODEL_NAME = "lgbm"
EARLY_STOPPING_ROUNDS = 50
MAX_BOOST_ROUNDS = 1000

ID_COLS = ("year", "month", "date", "permno", TARGET_COLUMN)

# Compact validation-tuning grid (keeps runtime manageable).
PARAM_GRID: dict[str, list[Any]] = {
    "num_leaves": [15, 31, 63],
    "learning_rate": [0.05, 0.1],
    "min_data_in_leaf": [100, 500],
    "lambda_l1": [0.0, 1.0],
    "lambda_l2": [0.1, 1.0],
}


def _scale_splits(
    train: pd.DataFrame,
    validate: pd.DataFrame,
    test: pd.DataFrame,
    factors: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit StandardScaler on train and transform all splits."""
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[list(factors)])
    x_val = scaler.transform(validate[list(factors)])
    x_test = scaler.transform(test[list(factors)])
    return x_train, x_val, x_test


def _iter_param_combinations() -> list[dict[str, Any]]:
    """Enumerate all hyperparameter combinations in PARAM_GRID."""
    keys = list(PARAM_GRID.keys())
    values = [PARAM_GRID[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _base_lgb_params(trial_params: dict[str, Any]) -> dict[str, Any]:
    """Build LightGBM parameter dict with fixed settings."""
    return {
        "objective": "regression",
        "metric": "mse",
        "verbosity": -1,
        "seed": RANDOM_STATE,
        "feature_fraction_seed": RANDOM_STATE,
        "bagging_seed": RANDOM_STATE,
        "data_random_seed": RANDOM_STATE,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "num_leaves": trial_params["num_leaves"],
        "learning_rate": trial_params["learning_rate"],
        "min_data_in_leaf": trial_params["min_data_in_leaf"],
        "lambda_l1": trial_params["lambda_l1"],
        "lambda_l2": trial_params["lambda_l2"],
    }


def _train_lgbm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    params: dict[str, Any],
) -> lgb.Booster:
    """Train one LightGBM model with early stopping on validation data."""
    train_set = lgb.Dataset(x_train, label=y_train)
    val_set = lgb.Dataset(x_val, label=y_val, reference=train_set)
    return lgb.train(
        params,
        train_set,
        num_boost_round=MAX_BOOST_ROUNDS,
        valid_sets=[val_set],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )


def _tune_lgbm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> lgb.Booster:
    """Select LightGBM hyperparameters by validation MSE."""
    best_mse = np.inf
    best_booster: lgb.Booster | None = None

    for trial in _iter_param_combinations():
        params = _base_lgb_params(trial)
        booster = _train_lgbm(x_train, y_train, x_val, y_val, params)
        preds = booster.predict(x_val, num_iteration=booster.best_iteration)
        mse = mean_squared_error(y_val, preds)
        if mse < best_mse:
            best_mse = mse
            best_booster = booster

    if best_booster is None:
        raise RuntimeError("LightGBM tuning failed to produce a model.")
    return best_booster


def _fit_window_gbm(
    train: pd.DataFrame,
    validate: pd.DataFrame,
    test: pd.DataFrame,
    factors: Sequence[str],
    target: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Train LightGBM for one expanding window and predict the test set."""
    x_train, x_val, x_test = _scale_splits(train, validate, test, factors)
    y_train = train[target].values
    y_val = validate[target].values

    booster = _tune_lgbm(x_train, y_train, x_val, y_val)
    predictions = test[list(ID_COLS)].copy()
    predictions[MODEL_NAME] = booster.predict(
        x_test,
        num_iteration=booster.best_iteration,
    )
    return predictions


def run_gbm(
    panel: pd.DataFrame,
    factors: Sequence[str],
    target: str = TARGET_COLUMN,
    verbose: bool = True,
) -> tuple[pd.DataFrame, float]:
    """Run expanding-window LightGBM with annual refits (2010–2023).

    Preprocessing matches linear baselines:
      1. Cross-sectional rank normalization on the full panel (once).
      2. Per window: train-only StandardScaler.

    Args:
        panel: Raw stock-month panel with non-null target.
        factors: Predictor column names.
        target: Target return column.
        verbose: Print progress per OOS year.

    Returns:
        Concatenated OOS predictions and pooled OOS R².
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
        fold_preds.append(_fit_window_gbm(train, validate, test, factors, target))

    predictions = pd.concat(fold_preds, ignore_index=True)
    oos_r2 = oos_r2_no_mean(
        predictions[target].values,
        predictions[MODEL_NAME].values,
    )
    return predictions, oos_r2


def format_gbm_report(oos_r2: float, linear_r2: float | None = None) -> str:
    """Format LightGBM OOS R² report, optionally vs linear best."""
    lines = [
        "Out-of-Sample R² — LightGBM (2010–2023, no-mean-subtraction)",
        "",
        f"  {MODEL_NAME:6s}: {oos_r2:.6f}",
    ]
    if linear_r2 is not None:
        lines.append(f"  ridge (linear benchmark): {linear_r2:.6f}")
        delta = oos_r2 - linear_r2
        lines.append(f"  lift vs ridge: {delta:+.6f}")
    return "\n".join(lines)
