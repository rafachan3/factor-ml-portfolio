"""Full OOS performance reporting (2010–2023) vs S&P 500."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import load_market_data
from src.paths import OUTPUTS_DIR
from src.portfolio.performance import compute_performance_stats, prepare_market_factor

REPORT_DIR = OUTPUTS_DIR / "reports"
OOS_START_YEAR = 2010

METRIC_COLUMNS = [
    "annualized_return",
    "annualized_stdev",
    "sharpe",
    "capm_alpha_annual",
    "information_ratio",
    "max_drawdown",
    "max_1m_loss",
    "turnover",
    "oos_r2",
]


def _load_oos_r2() -> dict[str, float]:
    """Load OOS R² for prediction models from saved reports."""
    r2: dict[str, float] = {}
    linear_path = OUTPUTS_DIR / "linear_oos_r2.txt"
    if linear_path.exists():
        for line in linear_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if ":" in line and not line.startswith("Out"):
                name, value = line.split(":", 1)
                r2[name.strip()] = float(value.strip())
    gbm_path = OUTPUTS_DIR / "gbm_oos_r2.txt"
    if gbm_path.exists():
        for line in gbm_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("lgbm"):
                r2["lgbm"] = float(line.split(":")[-1].strip())
    return r2


def _benchmark_monthly_returns(market: pd.DataFrame) -> pd.DataFrame:
    """Build monthly S&P 500 excess return series for OOS period."""
    mkt = prepare_market_factor(market)
    mkt = mkt[mkt["year"] >= OOS_START_YEAR].copy()
    return mkt.rename(columns={"mkt_rf": "strategy_ret"})[
        ["year", "month", "strategy_ret"]
    ]


def build_performance_table(
    monthly_returns: pd.DataFrame,
    strategy_summary: pd.DataFrame,
    market: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assemble full performance table: strategies, benchmark, and models.

    Args:
        monthly_returns: Long-format monthly returns per strategy.
        strategy_summary: Precomputed strategy statistics from backtest.
        market: Market data; loaded if None.

    Returns:
        Unified performance table indexed by name.
    """
    if market is None:
        market = load_market_data()

    rows: list[dict] = []

    # S&P 500 excess return benchmark.
    bench_monthly = _benchmark_monthly_returns(market)
    bench_stats = compute_performance_stats(bench_monthly, market)
    rows.append({
        "name": "SP500_excess",
        "type": "benchmark",
        **{k: bench_stats.get(k, np.nan) for k in METRIC_COLUMNS if k != "oos_r2"},
        "oos_r2": np.nan,
        "avg_n_total": np.nan,
    })

    # Portfolio strategies.
    for strategy in strategy_summary.index:
        stats = strategy_summary.loc[strategy].to_dict()
        rows.append({
            "name": strategy,
            "type": "portfolio",
            "annualized_return": stats["annualized_return"],
            "annualized_stdev": stats["annualized_stdev"],
            "sharpe": stats["sharpe"],
            "capm_alpha_annual": stats["capm_alpha_annual"],
            "information_ratio": stats["information_ratio"],
            "max_drawdown": stats["max_drawdown"],
            "max_1m_loss": stats["max_1m_loss"],
            "turnover": stats.get("turnover", np.nan),
            "oos_r2": np.nan,
            "avg_n_total": stats.get("avg_n_total", np.nan),
        })

    # Prediction models (OOS R² only; other metrics N/A).
    oos_r2 = _load_oos_r2()
    for model_name, r2 in sorted(oos_r2.items()):
        rows.append({
            "name": f"model_{model_name}",
            "type": "model",
            "annualized_return": np.nan,
            "annualized_stdev": np.nan,
            "sharpe": np.nan,
            "capm_alpha_annual": np.nan,
            "information_ratio": np.nan,
            "max_drawdown": np.nan,
            "max_1m_loss": np.nan,
            "turnover": np.nan,
            "oos_r2": r2,
            "avg_n_total": np.nan,
        })

    table = pd.DataFrame(rows).set_index("name")
    return table


def plot_cumulative_returns(
    monthly_returns: pd.DataFrame,
    market: pd.DataFrame,
    output_path: Path,
    strategies: list[str] | None = None,
) -> None:
    """Plot cumulative log returns for strategies vs S&P 500 excess.

    Args:
        monthly_returns: Long-format monthly strategy returns.
        market: Market data with sp_ret and rf.
        output_path: Path to save PNG figure.
        strategies: Strategies to plot; defaults to all in data.
    """
    mkt = prepare_market_factor(market)
    mkt = mkt[mkt["year"] >= OOS_START_YEAR].sort_values(["year", "month"])

    fig, ax = plt.subplots(figsize=(12, 6))

    bench_log = np.log(mkt["mkt_rf"] + 1).cumsum()
    ax.plot(
        range(len(bench_log)),
        bench_log.values,
        label="S&P 500 (excess)",
        color="black",
        linewidth=2,
        linestyle="--",
    )

    if strategies is None:
        strategies = sorted(monthly_returns["strategy"].unique())

    for strategy in strategies:
        strat = monthly_returns[monthly_returns["strategy"] == strategy].sort_values(
            ["year", "month"]
        )
        cum_log = np.log(strat["strategy_ret"] + 1).cumsum()
        ax.plot(range(len(cum_log)), cum_log.values, label=strategy, linewidth=1.5)

    ax.set_title("Cumulative Returns — OOS 2010–2023")
    ax.set_xlabel("Month index")
    ax.set_ylabel("Cumulative log return")
    ax.legend(loc="upper left", fontsize=9)
    ax.axhline(0, color="gray", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _format_cell(value: float) -> str:
    """Format one table cell for markdown output."""
    if pd.isna(value):
        return "—"
    return f"{value:.4f}"


def table_to_markdown(table: pd.DataFrame) -> str:
    """Render performance table as markdown without extra dependencies."""
    cols = ["type"] + [c for c in METRIC_COLUMNS if c in table.columns]
    if "avg_n_total" in table.columns:
        cols.append("avg_n_total")

    header = "| name | " + " | ".join(cols) + " |"
    sep = "|---|" + "|".join(["---"] * len(cols)) + "|"
    rows = [header, sep]
    for name, row in table.iterrows():
        cells = [str(name), str(row.get("type", ""))]
        for col in METRIC_COLUMNS:
            if col in table.columns:
                cells.append(_format_cell(row[col]))
        if "avg_n_total" in table.columns:
            cells.append(_format_cell(row["avg_n_total"]))
        rows.append("| " + " | ".join(cells) + " |")

    footer = [
        "",
        "CAPM alpha: annualized intercept from Newey-West regression "
        "(maxlags=3) of portfolio excess return on S&P 500 excess return.",
        "",
        "OOS R²: Gu-Kelly-Xiu formula without mean subtraction in denominator.",
    ]
    return "\n".join(["# OOS Performance Report (2010–2023)", ""] + rows + footer)


def run_performance_report(
    backtest_dir: Path | None = None,
    output_dir: Path | None = None,
    plot_strategies: list[str] | None = None,
) -> pd.DataFrame:
    """Generate performance table, markdown report, and cumulative return plot.

    Args:
        backtest_dir: Directory with backtest CSV outputs.
        output_dir: Report output directory.
        plot_strategies: Strategies to include in plot (default: top 3 + L/S).

    Returns:
        Unified performance table.
    """
    backtest_dir = backtest_dir or (OUTPUTS_DIR / "backtest")
    output_dir = output_dir or REPORT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    monthly = pd.read_csv(backtest_dir / "monthly_returns.csv")
    summary = pd.read_csv(backtest_dir / "strategy_summary.csv", index_col=0)
    market = load_market_data()

    table = build_performance_table(monthly, summary, market)

    table.to_csv(output_dir / "performance_table.csv")
    md = table_to_markdown(table)
    (output_dir / "performance_report.md").write_text(md, encoding="utf-8")

    if plot_strategies is None:
        plot_strategies = [
            "long_only_top50",
            "vol_scaled_long75",
            "long_short_decile",
        ]

    plot_cumulative_returns(
        monthly,
        market,
        output_dir / "cumulative_returns.png",
        strategies=plot_strategies,
    )

    return table
