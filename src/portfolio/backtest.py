"""Run multi-strategy portfolio backtests on model predictions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_loader import load_market_data, load_mma_panel
from src.portfolio.constructors import (
    STRATEGY_BUILDERS,
    assign_decile,
    filter_oos_predictions,
)
from src.portfolio.performance import compute_performance_stats, turnover_rate

VOL_COL = "ivol_capm_21d"


def _merge_volatility(
    pred: pd.DataFrame,
    panel: pd.DataFrame,
    vol_col: str = VOL_COL,
) -> pd.DataFrame:
    """Attach volatility column from the raw panel for vol-scaled strategies."""
    vol_data = panel[["permno", "date", vol_col]].drop_duplicates()
    return pred.merge(vol_data, on=["permno", "date"], how="left")


def _extract_holdings_long_short(
    pred: pd.DataFrame,
    score_col: str,
    n_long: int = 50,
    n_short: int = 50,
) -> pd.DataFrame:
    """Return long and short holdings for turnover calculation."""
    data = pred.copy()
    data["decile"] = assign_decile(data, score_col)
    rows: list[pd.DataFrame] = []
    for (year, month), month_df in data.groupby(["year", "month"]):
        longs = month_df[month_df["decile"] == 9].nlargest(n_long, score_col)
        shorts = month_df[month_df["decile"] == 0].nsmallest(n_short, score_col)
        longs = longs.assign(leg="long")
        shorts = shorts.assign(leg="short")
        rows.append(longs)
        rows.append(shorts)
    return pd.concat(rows, ignore_index=True)[["date", "permno", "leg"]]


def _extract_holdings_long_only(
    pred: pd.DataFrame,
    score_col: str,
    n_stocks: int,
) -> pd.DataFrame:
    """Return long-only holdings for turnover calculation."""
    rows: list[pd.DataFrame] = []
    for _, month_df in pred.groupby(["year", "month"]):
        longs = month_df.nlargest(n_stocks, score_col)
        rows.append(longs)
    return pd.concat(rows, ignore_index=True)[["date", "permno"]]


def run_backtest(
    predictions: pd.DataFrame,
    score_col: str,
    panel: pd.DataFrame | None = None,
    market: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all portfolio strategies and compute performance summaries.

    Args:
        predictions: OOS model predictions with stock_exret.
        score_col: Column with predicted returns (e.g. ``ridge``).
        panel: Raw panel for volatility merge (loaded if None).
        market: Market data (loaded if None).

    Returns:
        Monthly returns per strategy and summary statistics table.
    """
    pred = filter_oos_predictions(predictions)
    if panel is None:
        panel = load_mma_panel()
    if market is None:
        market = load_market_data()

    pred_vol = _merge_volatility(pred, panel)

    monthly_by_strategy: dict[str, pd.DataFrame] = {}
    stats_rows: list[dict] = []

    for strategy_name, builder in STRATEGY_BUILDERS.items():
        if strategy_name == "vol_scaled_long75":
            monthly = builder(pred_vol, score_col)
            holdings = _extract_holdings_long_only(pred_vol, score_col, 75)
        elif strategy_name == "long_short_decile":
            monthly = builder(pred, score_col)
            holdings = _extract_holdings_long_short(pred, score_col)
        elif strategy_name == "long_only_top50":
            monthly = builder(pred, score_col)
            holdings = _extract_holdings_long_only(pred, score_col, 50)
        else:
            monthly = builder(pred, score_col)
            holdings = _extract_holdings_long_only(pred, score_col, 75)

        monthly["strategy"] = strategy_name
        monthly_by_strategy[strategy_name] = monthly

        perf = compute_performance_stats(monthly, market)
        perf["strategy"] = strategy_name
        perf["turnover"] = turnover_rate(holdings)
        perf["avg_n_total"] = float(monthly["n_total"].mean())
        stats_rows.append(perf)

    all_monthly = pd.concat(monthly_by_strategy.values(), ignore_index=True)
    summary = pd.DataFrame(stats_rows).set_index("strategy")
    return all_monthly, summary


def save_backtest_outputs(
    monthly: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Persist monthly returns and summary statistics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(output_dir / "monthly_returns.csv", index=False)
    summary.to_csv(output_dir / "strategy_summary.csv")
    summary.round(6).to_string().strip()
