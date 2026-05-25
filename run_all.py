#!/usr/bin/env python3
"""Run the full Cross-Sectional Equity Return ML pipeline end-to-end."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Set global seeds before any stochastic operations.
from src.config import RANDOM_STATE  # noqa: E402

np.random.seed(RANDOM_STATE)

PROJECT_ROOT = Path(__file__).resolve().parent
EXPECTED_METRICS_PATH = PROJECT_ROOT / "expected_metrics.json"


def _step(name: str) -> None:
    print(f"\n{'=' * 60}\nSTEP: {name}\n{'=' * 60}", flush=True)


def verify_reproducibility() -> int:
    """Compare key outputs to expected_metrics.json within tolerance."""
    if not EXPECTED_METRICS_PATH.exists():
        print("Skipping reproducibility check (no expected_metrics.json).")
        return 0

    expected = json.loads(EXPECTED_METRICS_PATH.read_text(encoding="utf-8"))
    tol = expected.get("tolerance", 0.001)
    failures: list[str] = []

    from src.paths import OUTPUTS_DIR

    # OOS R² checks.
    linear_r2_path = OUTPUTS_DIR / "linear_oos_r2.txt"
    if linear_r2_path.exists():
        for line in linear_r2_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("ridge"):
                ridge_r2 = float(line.split(":")[-1].strip())
                exp = expected["ridge_oos_r2"]
                if abs(ridge_r2 - exp) > tol:
                    failures.append(f"ridge OOS R2: {ridge_r2} vs expected {exp}")

    gbm_r2_path = OUTPUTS_DIR / "gbm_oos_r2.txt"
    if gbm_r2_path.exists():
        for line in gbm_r2_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("lgbm"):
                lgbm_r2 = float(line.split(":")[-1].strip())
                exp = expected["lgbm_oos_r2"]
                if abs(lgbm_r2 - exp) > tol:
                    failures.append(f"lgbm OOS R2: {lgbm_r2} vs expected {exp}")

    # Portfolio metrics.
    summary_path = OUTPUTS_DIR / "backtest" / "strategy_summary.csv"
    if summary_path.exists():
        import pandas as pd

        summary = pd.read_csv(summary_path, index_col=0)
        checks = [
            ("long_only_top50", "sharpe", "long_only_top50_sharpe"),
            ("vol_scaled_long75", "sharpe", "vol_scaled_long75_sharpe"),
        ]
        for strategy, col, key in checks:
            if strategy in summary.index:
                val = float(summary.loc[strategy, col])
                exp = expected[key]
                if abs(val - exp) > tol:
                    failures.append(f"{strategy} {col}: {val} vs expected {exp}")

    table_path = OUTPUTS_DIR / "reports" / "performance_table.csv"
    if table_path.exists():
        import pandas as pd

        table = pd.read_csv(table_path, index_col=0)
        if "SP500_excess" in table.index:
            val = float(table.loc["SP500_excess", "annualized_return"])
            exp = expected["sp500_annualized_return"]
            if abs(val - exp) > tol:
                failures.append(
                    f"SP500 annualized_return: {val} vs expected {exp}"
                )

    if failures:
        print("\nReproducibility check FAILED:")
        for msg in failures:
            print(f"  - {msg}")
        return 1

    print("\nReproducibility check PASSED (within tolerance).")
    return 0


def main() -> int:
    """Execute pipeline steps in order."""
    parser = argparse.ArgumentParser(
        description="Run full equity return ML pipeline."
    )
    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Skip exploratory data analysis (~2 min).",
    )
    parser.add_argument(
        "--skip-linear",
        action="store_true",
        help="Skip linear baselines (~35 min).",
    )
    parser.add_argument(
        "--skip-gbm",
        action="store_true",
        help="Skip LightGBM training (~30 min).",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip reproducibility metric verification.",
    )
    args = parser.parse_args()

    t0 = time.time()
    print(f"Pipeline start (RANDOM_STATE={RANDOM_STATE})", flush=True)

    from src.data_loader import load_all
    from src.integrity_checks import run_integrity_checks
    from src.paths import OUTPUTS_DIR, TARGET_COLUMN

    # --- Data validation ---
    _step("Data integrity checks")
    data = load_all()
    report = run_integrity_checks(
        panel=data["panel"],
        factors=data["factors"],
        market=data["market"],
    )
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "m1_integrity_report.txt").write_text(
        report.render(), encoding="utf-8"
    )
    print(report.render())
    if not report.passed:
        return 1
    panel = data["panel"][data["panel"][TARGET_COLUMN].notna()].copy()
    factors = data["factors"]

    if not args.skip_eda:
        _step("Exploratory data analysis")
        from src.eda import run_eda

        run_eda(panel, factors)

    _step("Feature pipeline validation")
    from src.features import transform_panel_splits
    import pandas as pd

    panel_valid = panel.copy()
    starting = pd.Timestamp("2000-01-01")
    train_end = starting + pd.DateOffset(years=8)
    val_end = starting + pd.DateOffset(years=10)
    test_end = starting + pd.DateOffset(years=11)
    train = panel_valid[
        (panel_valid["date"] >= starting) & (panel_valid["date"] < train_end)
    ]
    validate = panel_valid[
        (panel_valid["date"] >= train_end) & (panel_valid["date"] < val_end)
    ]
    test = panel_valid[
        (panel_valid["date"] >= val_end) & (panel_valid["date"] < test_end)
    ]
    transform_panel_splits(train, validate, test, factors)
    print("Feature pipeline validation OK.")

    if not args.skip_linear:
        _step("Penalized linear baselines")
        from src.models.linear_baselines import (
            format_oos_r2_report,
            run_linear_baselines,
        )
        from src.paths import OUTPUTS_DIR

        pred, oos_r2 = run_linear_baselines(panel, factors)
        out_dir = OUTPUTS_DIR / "predictions"
        out_dir.mkdir(parents=True, exist_ok=True)
        pred.to_csv(out_dir / "linear_predictions.csv", index=False)
        report = format_oos_r2_report(oos_r2)
        print(report)
        (OUTPUTS_DIR / "linear_oos_r2.txt").write_text(report + "\n", encoding="utf-8")

    if not args.skip_gbm:
        _step("LightGBM model")
        from src.models.gbm import format_gbm_report, run_gbm
        from src.paths import OUTPUTS_DIR

        pred_gbm, gbm_r2 = run_gbm(panel, factors)
        out_dir = OUTPUTS_DIR / "predictions"
        out_dir.mkdir(parents=True, exist_ok=True)
        pred_gbm.to_csv(out_dir / "gbm_predictions.csv", index=False)
        linear_r2 = None
        linear_path = OUTPUTS_DIR / "linear_oos_r2.txt"
        if linear_path.exists():
            for line in linear_path.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("ridge"):
                    linear_r2 = float(line.split(":")[-1].strip())
        report = format_gbm_report(gbm_r2, linear_r2=linear_r2)
        print(report)
        (OUTPUTS_DIR / "gbm_oos_r2.txt").write_text(report + "\n", encoding="utf-8")

    _step("Portfolio backtests")
    import pandas as pd
    from src.paths import OUTPUTS_DIR
    from src.portfolio.backtest import run_backtest, save_backtest_outputs

    pred_path = OUTPUTS_DIR / "predictions" / "linear_predictions.csv"
    if not pred_path.exists():
        print(f"Missing {pred_path}. Run linear baselines first.")
        return 1
    predictions = pd.read_csv(pred_path, parse_dates=["date"])
    monthly, summary = run_backtest(predictions, score_col="ridge")
    save_backtest_outputs(monthly, summary, OUTPUTS_DIR / "backtest")
    print(summary[["annualized_return", "sharpe", "capm_alpha_annual"]].round(4))

    _step("Performance report")
    from src.reporting.performance_report import run_performance_report

    table = run_performance_report()
    print(table[["annualized_return", "sharpe", "oos_r2"]].round(4))

    if not args.skip_verify:
        _step("Reproducibility verification")
        if verify_reproducibility() != 0:
            return 1

    elapsed = (time.time() - t0) / 60
    print(f"\nPipeline complete in {elapsed:.1f} minutes.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
