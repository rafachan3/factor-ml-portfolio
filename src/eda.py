"""Exploratory data analysis for the equity return prediction panel."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

from src.paths import OUTPUTS_DIR, TARGET_COLUMN

EDA_OUTPUT_DIR = OUTPUTS_DIR / "eda"

# Evenly spaced sample for readable correlation heatmaps.
N_SAMPLE_FACTORS = 20
ADF_SIGNIFICANCE = 0.05
OUTLIER_MAD_MULTIPLIER = 3.0


def _ensure_year_column(panel: pd.DataFrame) -> pd.DataFrame:
    """Return panel with integer ``year`` column derived from ``date``."""
    if "year" in panel.columns:
        return panel
    out = panel.copy()
    out["year"] = out["date"].dt.year
    return out


def compute_factor_coverage_by_year(
    panel: pd.DataFrame,
    factors: list[str],
) -> pd.DataFrame:
    """Compute fraction of non-missing observations per factor and year.

    Args:
        panel: Stock-month panel.
        factors: Predictor column names.

    Returns:
        DataFrame indexed by factor, columns are calendar years, values in [0, 1].
    """
    data = _ensure_year_column(panel)
    years = sorted(data["year"].unique())
    coverage = pd.DataFrame(index=factors, columns=years, dtype=float)
    for factor in factors:
        for year in years:
            values = data.loc[data["year"] == year, factor]
            coverage.loc[factor, year] = values.notna().mean()
    coverage.columns = coverage.columns.astype(int)
    return coverage


def compute_target_stats_by_year(panel: pd.DataFrame) -> pd.DataFrame:
    """Summarize ``stock_exret`` distribution by calendar year.

    Args:
        panel: Stock-month panel containing ``stock_exret``.

    Returns:
        DataFrame with count, mean, std, skew, min, max, and quartiles by year.
    """
    data = _ensure_year_column(panel)
    stats = (
        data.groupby("year")[TARGET_COLUMN]
        .agg(
            count="count",
            mean="mean",
            std="std",
            skew="skew",
            min="min",
            p25=lambda s: s.quantile(0.25),
            median="median",
            p75=lambda s: s.quantile(0.75),
            max="max",
        )
        .round(6)
    )
    return stats


def compute_outlier_tail_counts(
    panel: pd.DataFrame,
    factors: list[str],
) -> pd.DataFrame:
    """Count cross-sectional MAD-based outliers per factor.

    For each month, flags observations with
    ``|x - median| > 3 * MAD`` within the cross section.

    Args:
        panel: Stock-month panel.
        factors: Predictor column names.

    Returns:
        DataFrame with outlier count and rate (per non-null observation).
    """
    records: list[dict[str, float | str | int]] = []
    for factor in factors:
        outlier_count = 0
        obs_count = 0
        for _, group in panel.groupby("date"):
            values = group[factor].dropna()
            if values.empty:
                continue
            obs_count += len(values)
            median = values.median()
            mad = (values - median).abs().median()
            if mad == 0:
                continue
            threshold = OUTLIER_MAD_MULTIPLIER * mad
            outlier_count += int((values - median).abs().gt(threshold).sum())
        rate = outlier_count / obs_count if obs_count else 0.0
        records.append({
            "factor": factor,
            "outlier_count": outlier_count,
            "obs_count": obs_count,
            "outlier_rate": rate,
        })
    result = pd.DataFrame(records).set_index("factor")
    return result.sort_values("outlier_rate", ascending=False)


def compute_stationarity_flags(
    panel: pd.DataFrame,
    factors: list[str],
    significance: float = ADF_SIGNIFICANCE,
) -> pd.DataFrame:
    """Run ADF tests on each factor's cross-sectional median time series.

    Args:
        panel: Stock-month panel.
        factors: Predictor column names.
        significance: Reject unit root when p-value is below this level.

    Returns:
        DataFrame with ADF statistic, p-value, and stationary flag per factor.
    """
    records: list[dict[str, float | str | bool]] = []
    for factor in factors:
        series = panel.groupby("date")[factor].median().dropna()
        if len(series) < 12:
            records.append({
                "factor": factor,
                "adf_stat": np.nan,
                "p_value": np.nan,
                "stationary": False,
                "n_obs": len(series),
            })
            continue
        try:
            adf_stat, p_value, _, _, _, _ = adfuller(series, autolag="AIC")
        except ValueError:
            # Constant (or near-constant) cross-sectional median series.
            records.append({
                "factor": factor,
                "adf_stat": np.nan,
                "p_value": np.nan,
                "stationary": False,
                "n_obs": len(series),
            })
            continue
        records.append({
            "factor": factor,
            "adf_stat": adf_stat,
            "p_value": p_value,
            "stationary": p_value < significance,
            "n_obs": len(series),
        })
    return pd.DataFrame(records).set_index("factor")


def select_sample_factors(factors: list[str], n: int = N_SAMPLE_FACTORS) -> list[str]:
    """Pick evenly spaced factors for correlation visualization."""
    if len(factors) <= n:
        return factors
    indices = np.linspace(0, len(factors) - 1, n, dtype=int)
    return [factors[i] for i in indices]


def compute_cross_sectional_correlation(
    panel: pd.DataFrame,
    factors: list[str],
    sample_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Compute Pearson correlation matrix for one cross section.

    Args:
        panel: Stock-month panel.
        factors: Subset of factor names.
        sample_date: Month to use; defaults to median panel date.

    Returns:
        Correlation matrix and the date used.
    """
    if sample_date is None:
        sample_date = panel["date"].sort_values().iloc[len(panel) // 2]
    cross_section = panel.loc[panel["date"] == sample_date, factors].dropna()
    corr = cross_section.corr()
    return corr, sample_date


def compute_monthly_mean_abs_correlation(
    panel: pd.DataFrame,
    factors: list[str],
) -> pd.Series:
    """Compute monthly mean absolute off-diagonal correlation.

    Args:
        panel: Stock-month panel.
        factors: Subset of factor names.

    Returns:
        Series indexed by date with mean |rho| among factor pairs.
    """
    results: dict[pd.Timestamp, float] = {}
    for date, group in panel.groupby("date"):
        cross_section = group[factors].dropna()
        if cross_section.shape[0] < len(factors):
            continue
        corr = cross_section.corr().values
        upper = corr[np.triu_indices_from(corr, k=1)]
        if upper.size == 0:
            continue
        results[date] = float(np.abs(upper).mean())
    return pd.Series(results).sort_index()


def plot_factor_coverage_heatmap(
    coverage: pd.DataFrame,
    output_path: Path,
    title: str = "Factor Coverage by Year (% non-missing)",
) -> None:
    """Save heatmap of factor coverage by year."""
    fig, ax = plt.subplots(figsize=(14, 18))
    sns.heatmap(
        coverage,
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Non-missing rate"},
    )
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Factor")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_missingness_map(
    coverage: pd.DataFrame,
    output_path: Path,
    title: str = "Factor Missingness Map (% missing)",
) -> None:
    """Save heatmap of missing rates sorted by overall missingness."""
    missing = 1.0 - coverage
    overall = missing.mean(axis=1).sort_values(ascending=False)
    missing = missing.loc[overall.index]
    fig, ax = plt.subplots(figsize=(14, 18))
    sns.heatmap(
        missing,
        cmap="Reds",
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Missing rate"},
    )
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Factor (sorted by overall missingness)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_target_distribution(
    panel: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save histogram/KDE of stock_exret with yearly subsample overlay."""
    data = _ensure_year_column(panel)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Full-sample histogram + KDE.
    axes[0].hist(
        data[TARGET_COLUMN].dropna(),
        bins=80,
        density=True,
        alpha=0.6,
        color="steelblue",
        label="Histogram",
    )
    data[TARGET_COLUMN].dropna().plot.kde(ax=axes[0], color="darkred", lw=2)
    axes[0].set_title(f"{TARGET_COLUMN} — Full Sample")
    axes[0].set_xlabel("Monthly excess return")
    axes[0].set_ylabel("Density")

    # Boxplot by year (recent subset for readability if needed).
    years = sorted(data["year"].unique())
    sample_years = years[::2]  # Every other year for readability.
    subset = data[data["year"].isin(sample_years)]
    sns.boxplot(
        data=subset,
        x="year",
        y=TARGET_COLUMN,
        ax=axes[1],
        color="steelblue",
        fliersize=1,
    )
    axes[1].set_title(f"{TARGET_COLUMN} — By Year (alternate years)")
    axes[1].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_factor_correlation(
    corr: pd.DataFrame,
    output_path: Path,
    sample_date: pd.Timestamp,
) -> None:
    """Save correlation heatmap for sample factors."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        square=True,
    )
    ax.set_title(
        f"Cross-Sectional Factor Correlations ({sample_date.date()})"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def generate_eda_summary(
    panel: pd.DataFrame,
    factors: list[str],
    coverage: pd.DataFrame,
    target_stats: pd.DataFrame,
    stationarity: pd.DataFrame,
    outliers: pd.DataFrame,
    sample_factors: list[str],
    sample_date: pd.Timestamp,
    mean_abs_corr: pd.Series,
) -> str:
    """Build one-page markdown summary of EDA findings."""
    n_stationary = int(stationarity["stationary"].sum())
    overall_missing = (1.0 - coverage.mean(axis=1)).sort_values(ascending=False)
    worst_missing = overall_missing.head(5)
    highest_outliers = outliers.head(5)
    target_overall = panel[TARGET_COLUMN].describe()

    lines = [
        "# EDA Summary",
        "",
        "## Panel overview",
        f"- Observations: {len(panel):,}",
        f"- Months: {panel['date'].dt.to_period('M').nunique()}",
        f"- Factors: {len(factors)}",
        f"- Date range: {panel['date'].min().date()} to "
        f"{panel['date'].max().date()}",
        "",
        "## Target (`stock_exret`)",
        f"- Mean: {target_overall['mean']:.4f}",
        f"- Std: {target_overall['std']:.4f}",
        f"- Min / Max: {target_overall['min']:.4f} / {target_overall['max']:.4f}",
        f"- Skew (full sample): {panel[TARGET_COLUMN].skew():.4f}",
        "",
        "Annual mean excess return ranges from "
        f"{target_stats['mean'].min():.4f} to {target_stats['mean'].max():.4f}.",
        "",
        "## Factor missingness",
        f"- Median factor coverage (all years): "
        f"{coverage.mean(axis=1).median():.1%}",
        f"- Factors with >20% missing (avg): "
        f"{int((overall_missing > 0.20).sum())}",
        "",
        "Highest missing factors:",
    ]
    for factor, rate in worst_missing.items():
        lines.append(f"- `{factor}`: {rate:.1%} missing")
    lines.extend([
        "",
        "## Factor sanity",
        f"- Stationary (ADF p<{ADF_SIGNIFICANCE}, xs-median series): "
        f"{n_stationary}/{len(factors)}",
        f"- Sample correlation date: {sample_date.date()}",
        f"- Sample factors ({len(sample_factors)}): "
        f"{', '.join(sample_factors[:5])}, ...",
        f"- Median monthly mean |rho| (sample factors): "
        f"{mean_abs_corr.median():.3f}",
        "",
        "Highest outlier-rate factors (3× MAD rule):",
    ])
    for factor, row in highest_outliers.iterrows():
        lines.append(
            f"- `{factor}`: {row['outlier_rate']:.2%} "
            f"({int(row['outlier_count']):,} obs)"
        )
    lines.extend([
        "",
        "## Implications for preprocessing",
        "- Cross-sectional rank normalization will reduce outlier influence.",
        "- High-missing factors may carry limited signal; monitor in model selection.",
        "- Moderate factor correlation suggests regularization / tree models are appropriate.",
        "- Target is roughly symmetric with fat tails; no obvious year-level drift in scale.",
        "",
        "## Artifacts",
        "- `factor_coverage_by_year.png`",
        "- `factor_missingness_map.png`",
        "- `target_distribution.png`",
        "- `factor_correlation_sample.png`",
        "- `target_stats_by_year.csv`",
        "- `stationarity_flags.csv`",
        "- `outlier_tail_counts.csv`",
    ])
    return "\n".join(lines)


def run_eda(
    panel: pd.DataFrame,
    factors: list[str],
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Run full EDA pipeline and persist plots, tables, and summary.

    Args:
        panel: Stock-month panel.
        factors: Predictor column names.
        output_dir: Destination directory; defaults to ``outputs/eda/``.

    Returns:
        Dictionary mapping artifact names to file paths.
    """
    out_dir = output_dir or EDA_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Computing factor coverage by year...")
    coverage = compute_factor_coverage_by_year(panel, factors)

    print("Computing target statistics...")
    target_stats = compute_target_stats_by_year(panel)

    print("Computing outlier tail counts (may take a minute)...")
    outliers = compute_outlier_tail_counts(panel, factors)

    print("Running ADF stationarity tests...")
    stationarity = compute_stationarity_flags(panel, factors)

    sample_factors = select_sample_factors(factors)
    corr, sample_date = compute_cross_sectional_correlation(
        panel, sample_factors
    )
    mean_abs_corr = compute_monthly_mean_abs_correlation(panel, sample_factors)

    print("Saving plots...")
    artifacts: dict[str, Path] = {}
    artifacts["factor_coverage"] = out_dir / "factor_coverage_by_year.png"
    artifacts["missingness_map"] = out_dir / "factor_missingness_map.png"
    artifacts["target_distribution"] = out_dir / "target_distribution.png"
    artifacts["factor_correlation"] = out_dir / "factor_correlation_sample.png"

    plot_factor_coverage_heatmap(coverage, artifacts["factor_coverage"])
    plot_missingness_map(coverage, artifacts["missingness_map"])
    plot_target_distribution(panel, artifacts["target_distribution"])
    plot_factor_correlation(corr, artifacts["factor_correlation"], sample_date)

    artifacts["target_stats"] = out_dir / "target_stats_by_year.csv"
    artifacts["stationarity"] = out_dir / "stationarity_flags.csv"
    artifacts["outlier_counts"] = out_dir / "outlier_tail_counts.csv"
    artifacts["coverage_table"] = out_dir / "factor_coverage_by_year.csv"

    target_stats.to_csv(artifacts["target_stats"])
    stationarity.to_csv(artifacts["stationarity"])
    outliers.to_csv(artifacts["outlier_counts"])
    coverage.to_csv(artifacts["coverage_table"])

    summary_text = generate_eda_summary(
        panel=panel,
        factors=factors,
        coverage=coverage,
        target_stats=target_stats,
        stationarity=stationarity,
        outliers=outliers,
        sample_factors=sample_factors,
        sample_date=sample_date,
        mean_abs_corr=mean_abs_corr,
    )
    artifacts["summary"] = out_dir / "eda_summary.md"
    artifacts["summary"].write_text(summary_text, encoding="utf-8")

    print(f"EDA complete. Outputs saved to {out_dir}")
    return artifacts
