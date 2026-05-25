"""Portfolio performance metrics for backtests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as sm


def prepare_market_factor(market: pd.DataFrame) -> pd.DataFrame:
    """Add S&P 500 excess return column to market data."""
    mkt = market.copy()
    mkt["mkt_rf"] = mkt["sp_ret"] - mkt["rf"]
    return mkt


def compute_performance_stats(
    monthly_returns: pd.DataFrame,
    market: pd.DataFrame,
    return_col: str = "strategy_ret",
) -> dict[str, float]:
    """Compute standard portfolio performance statistics.

    Args:
        monthly_returns: DataFrame with year, month, and return column.
        market: Market data with rf, sp_ret, year, month.
        return_col: Portfolio return column (excess returns).

    Returns:
        Dictionary of performance metrics.
    """
    mkt = prepare_market_factor(market)
    data = monthly_returns.merge(mkt, on=["year", "month"], how="inner")
    rets = data[return_col]

    # CAPM alpha (monthly intercept); annualize by ×12.
    nw_ols = sm.ols(formula=f"{return_col} ~ mkt_rf", data=data).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": 3},
        use_t=True,
    )
    alpha_monthly = float(nw_ols.params["Intercept"])
    alpha_annual = alpha_monthly * 12
    beta = float(nw_ols.params["mkt_rf"])
    alpha_tstat = float(nw_ols.tvalues["Intercept"])
    resid_std = float(np.sqrt(nw_ols.mse_resid))
    info_ratio = (
        alpha_monthly / resid_std * np.sqrt(12) if resid_std > 0 else float("nan")
    )

    sharpe = rets.mean() / rets.std() * np.sqrt(12) if rets.std() > 0 else float("nan")

    log_rets = np.log(rets + 1)
    cum_log = log_rets.cumsum()
    max_drawdown = float((cum_log.cummax() - cum_log).max())
    max_1m_loss = float(rets.min())

    return {
        "annualized_return": float(rets.mean() * 12),
        "annualized_stdev": float(rets.std() * np.sqrt(12)),
        "sharpe": float(sharpe),
        "capm_alpha_annual": alpha_annual,
        "capm_alpha_monthly": alpha_monthly,
        "capm_beta": beta,
        "alpha_tstat": alpha_tstat,
        "information_ratio": float(info_ratio),
        "max_drawdown": max_drawdown,
        "max_1m_loss": max_1m_loss,
        "n_months": len(rets),
    }


def turnover_rate(holdings: pd.DataFrame) -> float:
    """Compute average monthly turnover from permno holdings.

    Turnover for month t is the fraction of names not held in month t-1.

    Args:
        holdings: DataFrame with date and permno for selected positions.

    Returns:
        Average monthly turnover rate.
    """
    data = holdings[["permno", "date"]].copy()
    data["period"] = data["date"].dt.to_period("M")
    turnovers: list[float] = []
    periods = sorted(data["period"].unique())

    for idx in range(1, len(periods)):
        prev_permnos = set(data.loc[data["period"] == periods[idx - 1], "permno"])
        curr_permnos = set(data.loc[data["period"] == periods[idx], "permno"])
        if not curr_permnos:
            continue
        retained = len(prev_permnos & curr_permnos)
        turnovers.append(1.0 - retained / len(curr_permnos))

    return float(np.mean(turnovers)) if turnovers else float("nan")
