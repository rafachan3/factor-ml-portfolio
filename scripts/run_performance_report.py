#!/usr/bin/env python3
"""Generate full OOS performance report vs S&P 500 (2010–2023)."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import OUTPUTS_DIR  # noqa: E402
from src.reporting.performance_report import run_performance_report  # noqa: E402


def main() -> int:
    """Build performance table and cumulative return plot."""
    backtest_dir = OUTPUTS_DIR / "backtest"
    if not (backtest_dir / "monthly_returns.csv").exists():
        print("Backtest outputs not found. Run scripts/run_backtest.py first.")
        return 1

    table = run_performance_report()
    out_dir = OUTPUTS_DIR / "reports"

    print(f"Report saved to {out_dir}/")
    print("\n=== Performance Table (selected columns) ===")
    cols = [
        "annualized_return",
        "sharpe",
        "capm_alpha_annual",
        "information_ratio",
        "max_drawdown",
        "oos_r2",
    ]
    print(table[cols].round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
