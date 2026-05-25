"""Evaluation metrics for return prediction models."""

import numpy as np


def oos_r2_no_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Out-of-sample R² without subtracting the historical mean.

    Formula from Gu, Kelly, and Xiu (2020):

        R²_OOS = 1 - Σ(r - r̂)² / Σ r²

    The benchmark is zero predictability (not the historical mean).

    Args:
        y_true: Realized returns.
        y_pred: Model predictions.

    Returns:
        Out-of-sample R².
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum(np.square(y_true - y_pred))
    ss_tot = np.sum(np.square(y_true))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot
