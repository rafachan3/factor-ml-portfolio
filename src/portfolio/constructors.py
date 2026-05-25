"""Portfolio construction strategies for predicted stock returns."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.paths import TARGET_COLUMN

MIN_HOLDINGS = 50
MAX_HOLDINGS = 100
OOS_START_YEAR = 2010

ID_COLS = ["year", "month", "date", "permno"]


def filter_oos_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    """Keep OOS prediction rows (2010–2023)."""
    return pred[pred["year"] >= OOS_START_YEAR].copy()


def assign_decile(
    pred: pd.DataFrame,
    score_col: str,
    group_cols: tuple[str, str] = ("year", "month"),
) -> pd.Series:
    """Assign decile ranks 0–9 within each month (0 = lowest predictions)."""
    grouped = pred.groupby(list(group_cols))[score_col]
    return np.floor(
        grouped.transform(lambda s: s.rank())
        * 10
        / grouped.transform(lambda s: len(s) + 1)
    ).astype(int)


def _monthly_portfolio_return(
    group: pd.DataFrame,
    weight_col: str,
) -> float:
    """Compute weighted average realized excess return for one month."""
    weights = group[weight_col].values
    weights = weights / weights.sum()
    return float(np.average(group[TARGET_COLUMN].values, weights=weights))


def _validate_holdings(count: int, strategy: str, period: tuple) -> None:
    """Raise if holdings count violates 50–100 constraint."""
    if count < MIN_HOLDINGS or count > MAX_HOLDINGS:
        raise ValueError(
            f"{strategy} at {period}: {count} names (require "
            f"{MIN_HOLDINGS}–{MAX_HOLDINGS})."
        )


def build_long_short_decile(
    pred: pd.DataFrame,
    score_col: str,
    n_long: int = 50,
    n_short: int = 50,
) -> pd.DataFrame:
    """Long-short decile strategy with capped 50+50 names.

    Assigns deciles like the reference workflow, then holds the top
    ``n_long`` stocks in decile 9 (long) and bottom ``n_short`` in decile 0
    (short), equal-weighted within each leg.

    Args:
        pred: Predictions with ``stock_exret`` and score column.
        score_col: Model prediction column.
        n_long: Long leg size.
        n_short: Short leg size.

    Returns:
        Monthly portfolio returns with columns year, month, strategy_ret.
    """
    total_names = n_long + n_short
    if total_names < MIN_HOLDINGS or total_names > MAX_HOLDINGS:
        raise ValueError(
            f"long_short_decile requires {MIN_HOLDINGS}–{MAX_HOLDINGS} "
            f"total names; got {total_names}."
        )

    data = pred.copy()
    data["decile"] = assign_decile(data, score_col)

    monthly_returns: list[dict] = []
    for (year, month), month_df in data.groupby(["year", "month"]):
        long_pool = month_df[month_df["decile"] == 9].nlargest(n_long, score_col)
        short_pool = month_df[month_df["decile"] == 0].nsmallest(n_short, score_col)

        _validate_holdings(len(long_pool) + len(short_pool), "long_short_decile",
                          (year, month))
        if len(long_pool) < n_long or len(short_pool) < n_short:
            raise ValueError(
                f"Insufficient stocks for long_short_decile at {(year, month)}."
            )

        long_ret = long_pool[TARGET_COLUMN].mean()
        short_ret = short_pool[TARGET_COLUMN].mean()
        monthly_returns.append({
            "year": year,
            "month": month,
            "strategy_ret": long_ret - short_ret,
            "n_long": len(long_pool),
            "n_short": len(short_pool),
            "n_total": len(long_pool) + len(short_pool),
        })

    return pd.DataFrame(monthly_returns)


def build_long_only_top_n(
    pred: pd.DataFrame,
    score_col: str,
    n_stocks: int = 50,
) -> pd.DataFrame:
    """Long-only equal-weight portfolio of top-N predicted stocks.

    Args:
        pred: Predictions with realized returns.
        score_col: Model prediction column.
        n_stocks: Number of long positions (50–100).

    Returns:
        Monthly portfolio returns.
    """
    if n_stocks < MIN_HOLDINGS or n_stocks > MAX_HOLDINGS:
        raise ValueError(
            f"long_only_top_n requires n in [{MIN_HOLDINGS}, {MAX_HOLDINGS}]."
        )

    monthly_returns: list[dict] = []
    for (year, month), month_df in pred.groupby(["year", "month"]):
        longs = month_df.nlargest(n_stocks, score_col)
        _validate_holdings(len(longs), "long_only_top_n", (year, month))
        if len(longs) < n_stocks:
            raise ValueError(
                f"Insufficient stocks for long_only_top_n at {(year, month)}."
            )
        monthly_returns.append({
            "year": year,
            "month": month,
            "strategy_ret": longs[TARGET_COLUMN].mean(),
            "n_long": len(longs),
            "n_short": 0,
            "n_total": len(longs),
        })

    return pd.DataFrame(monthly_returns)


def build_rank_weighted_long(
    pred: pd.DataFrame,
    score_col: str,
    n_stocks: int = 75,
) -> pd.DataFrame:
    """Long-only portfolio weighted by predicted-return ranks.

    Selects top ``n_stocks`` by prediction; weights are proportional to
    shifted scores (positive, sum to 1).

    Args:
        pred: Predictions with realized returns.
        score_col: Model prediction column.
        n_stocks: Number of holdings (50–100).

    Returns:
        Monthly portfolio returns.
    """
    if n_stocks < MIN_HOLDINGS or n_stocks > MAX_HOLDINGS:
        raise ValueError(
            f"rank_weighted_long requires n in [{MIN_HOLDINGS}, {MAX_HOLDINGS}]."
        )

    monthly_returns: list[dict] = []
    for (year, month), month_df in pred.groupby(["year", "month"]):
        longs = month_df.nlargest(n_stocks, score_col).copy()
        _validate_holdings(len(longs), "rank_weighted_long", (year, month))
        if len(longs) < n_stocks:
            raise ValueError(
                f"Insufficient stocks for rank_weighted_long at {(year, month)}."
            )

        scores = longs[score_col].values
        shifted = scores - scores.min() + 1e-8
        weights = shifted / shifted.sum()
        port_ret = float(np.average(longs[TARGET_COLUMN].values, weights=weights))

        monthly_returns.append({
            "year": year,
            "month": month,
            "strategy_ret": port_ret,
            "n_long": len(longs),
            "n_short": 0,
            "n_total": len(longs),
        })

    return pd.DataFrame(monthly_returns)


def build_vol_scaled_long(
    pred: pd.DataFrame,
    score_col: str,
    vol_col: str = "ivol_capm_21d",
    n_stocks: int = 75,
) -> pd.DataFrame:
    """Long-only inverse-volatility-weighted top-N portfolio.

    Args:
        pred: Predictions merged with volatility column.
        score_col: Model prediction column.
        vol_col: Realized volatility column for scaling.
        n_stocks: Number of holdings (50–100).

    Returns:
        Monthly portfolio returns.
    """
    if n_stocks < MIN_HOLDINGS or n_stocks > MAX_HOLDINGS:
        raise ValueError(
            f"vol_scaled_long requires n in [{MIN_HOLDINGS}, {MAX_HOLDINGS}]."
        )
    if vol_col not in pred.columns:
        raise ValueError(f"Missing volatility column '{vol_col}'.")

    monthly_returns: list[dict] = []
    for (year, month), month_df in pred.groupby(["year", "month"]):
        candidates = month_df.nlargest(n_stocks, score_col).copy()
        _validate_holdings(len(candidates), "vol_scaled_long", (year, month))

        vol = candidates[vol_col].replace(0, np.nan)
        if vol.isna().any():
            vol = vol.fillna(vol.median())
        inv_vol = 1.0 / vol
        weights = inv_vol / inv_vol.sum()
        port_ret = float(
            np.average(candidates[TARGET_COLUMN].values, weights=weights)
        )

        monthly_returns.append({
            "year": year,
            "month": month,
            "strategy_ret": port_ret,
            "n_long": len(candidates),
            "n_short": 0,
            "n_total": len(candidates),
        })

    return pd.DataFrame(monthly_returns)


STRATEGY_BUILDERS: dict[str, Callable[..., pd.DataFrame]] = {
    "long_short_decile": build_long_short_decile,
    "long_only_top50": lambda p, s: build_long_only_top_n(p, s, n_stocks=50),
    "rank_weighted_long75": lambda p, s: build_rank_weighted_long(p, s, n_stocks=75),
    "vol_scaled_long75": build_vol_scaled_long,
}
