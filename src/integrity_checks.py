"""Data integrity checks for the equity return prediction panel."""

from dataclasses import dataclass, field

import pandas as pd

from src.paths import EXPECTED_END, EXPECTED_START, N_EXPECTED_FACTORS, TARGET_COLUMN


@dataclass
class IntegrityReport:
    """Structured results from panel integrity validation."""

    passed: bool = True
    sections: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def add_section(self, title: str, lines: list[str]) -> None:
        """Append a titled section to the report."""
        self.sections.append(f"=== {title} ===")
        self.sections.extend(lines)
        self.sections.append("")

    def fail(self, message: str) -> None:
        """Record a validation failure."""
        self.passed = False
        self.failures.append(message)

    def render(self) -> str:
        """Return the full report as plain text."""
        status = "PASS" if self.passed else "FAIL"
        header = [f"Data Integrity Report — {status}", ""]
        if self.failures:
            header.append("Failures:")
            header.extend(f"  - {msg}" for msg in self.failures)
            header.append("")
        return "\n".join(header + self.sections)


def _year_month_str(series: pd.Series) -> pd.Series:
    """Convert datetime series to YYYY-MM strings."""
    return series.dt.to_period("M").astype(str)


def run_integrity_checks(
    panel: pd.DataFrame,
    factors: list[str],
    market: pd.DataFrame,
) -> IntegrityReport:
    """Run all M1 integrity checks on loaded data.

    Args:
        panel: Stock-month panel from mma_sample_v2.csv.
        factors: Factor names from factor_char_list.csv.
        market: Market data from mkt_ind.csv.

    Returns:
        IntegrityReport with pass/fail flag and detailed sections.
    """
    report = IntegrityReport()

    # --- Row counts ---
    report.add_section(
        "Row counts",
        [
            f"Panel rows: {len(panel):,}",
            f"Factor list entries: {len(factors)}",
            f"Market rows: {len(market):,}",
            f"Panel columns: {len(panel.columns)}",
        ],
    )

    # --- Date coverage ---
    if "date" not in panel.columns:
        report.fail("Panel missing 'date' column.")
        return report

    min_date = panel["date"].min()
    max_date = panel["date"].max()
    min_ym = _year_month_str(pd.Series([min_date])).iloc[0]
    max_ym = _year_month_str(pd.Series([max_date])).iloc[0]
    n_months = panel["date"].dt.to_period("M").nunique()

    date_lines = [
        f"Panel date range: {min_date.date()} to {max_date.date()} "
        f"({min_ym} to {max_ym})",
        f"Unique months in panel: {n_months}",
        f"Expected: {EXPECTED_START} to {EXPECTED_END}",
    ]
    if min_ym > EXPECTED_START or max_ym < EXPECTED_END:
        report.fail(
            f"Date coverage ({min_ym}–{max_ym}) does not span "
            f"{EXPECTED_START}–{EXPECTED_END}."
        )
    report.add_section("Date coverage", date_lines)

    # --- permno uniqueness within month ---
    if "permno" not in panel.columns:
        report.fail("Panel missing 'permno' column.")
    else:
        dup_mask = panel.duplicated(subset=["date", "permno"], keep=False)
        n_dup_rows = int(dup_mask.sum())
        n_dup_pairs = (
            panel.loc[dup_mask, ["date", "permno"]]
            .drop_duplicates()
            .shape[0]
        )
        permno_lines = [
            f"Duplicate (date, permno) rows: {n_dup_rows:,}",
            f"Duplicate (date, permno) pairs: {n_dup_pairs:,}",
        ]
        if n_dup_pairs > 0:
            report.fail(
                f"Found {n_dup_pairs} duplicate (date, permno) pairs."
            )
        # Cross-section size sanity check (~1,000 stocks/month).
        xs_counts = panel.groupby("date")["permno"].nunique()
        permno_lines.extend([
            f"Stocks per month — min: {xs_counts.min()}, "
            f"median: {int(xs_counts.median())}, max: {xs_counts.max()}",
        ])
        report.add_section("permno uniqueness (within month)", permno_lines)

    # --- Target non-null counts ---
    if TARGET_COLUMN not in panel.columns:
        report.fail(f"Panel missing target column '{TARGET_COLUMN}'.")
    else:
        n_total = len(panel)
        n_nonnull = int(panel[TARGET_COLUMN].notna().sum())
        n_null = n_total - n_nonnull
        pct_nonnull = 100.0 * n_nonnull / n_total if n_total else 0.0
        target_by_year = (
            panel.assign(year=panel["date"].dt.year)
            .groupby("year")[TARGET_COLUMN]
            .apply(lambda s: s.notna().sum())
        )
        target_lines = [
            f"Target column: {TARGET_COLUMN}",
            f"Non-null: {n_nonnull:,} / {n_total:,} ({pct_nonnull:.1f}%)",
            f"Null: {n_null:,}",
            "Non-null counts by year:",
        ]
        target_lines.extend(
            f"  {int(yr)}: {int(cnt):,}" for yr, cnt in target_by_year.items()
        )
        report.add_section("Target (stock_exret) non-null counts", target_lines)

    # --- Factor column presence ---
    missing_factors = [f for f in factors if f not in panel.columns]
    extra_in_list = len(factors) - N_EXPECTED_FACTORS
    factor_lines = [
        f"Factors in list: {len(factors)} (expected {N_EXPECTED_FACTORS})",
        f"Present in panel: {len(factors) - len(missing_factors)}",
        f"Missing from panel: {len(missing_factors)}",
    ]
    if missing_factors:
        factor_lines.append("Missing factor names (first 20):")
        factor_lines.extend(f"  - {name}" for name in missing_factors[:20])
        if len(missing_factors) > 20:
            factor_lines.append(f"  ... and {len(missing_factors) - 20} more")
        report.fail(
            f"{len(missing_factors)} factor(s) from factor_char_list.csv "
            "absent in panel."
        )
    if len(factors) != N_EXPECTED_FACTORS:
        report.fail(
            f"factor_char_list.csv has {len(factors)} entries; "
            f"expected {N_EXPECTED_FACTORS}."
        )
    report.add_section("Factor columns vs factor_char_list.csv", factor_lines)

    # --- Market data coverage ---
    mkt_months = market[["year", "month"]].drop_duplicates().shape[0]
    mkt_lines = [
        f"Market rows: {len(market):,}",
        f"Unique year-months: {mkt_months}",
        f"Year range: {market['year'].min()}–{market['year'].max()}",
        f"Columns: {list(market.columns)}",
    ]
    required_mkt_cols = {"rf", "year", "month", "sp_ret"}
    missing_mkt = required_mkt_cols - set(market.columns)
    if missing_mkt:
        report.fail(f"Market data missing columns: {sorted(missing_mkt)}")
    report.add_section("Market data (mkt_ind.csv)", mkt_lines)

    return report
