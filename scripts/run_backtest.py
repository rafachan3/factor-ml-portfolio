#!/usr/bin/env python3
"""Run portfolio backtests on OOS model predictions (2010–2023)."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import OUTPUTS_DIR  # noqa: E402
from src.portfolio.backtest import run_backtest, save_backtest_outputs  # noqa: E402

DEFAULT_MODEL = "ridge"
DEFAULT_PRED_PATH = OUTPUTS_DIR / "predictions" / "linear_predictions.csv"


def main() -> int:
    """Run three portfolio strategies and save performance summaries."""
    import argparse

    parser = argparse.ArgumentParser(description="Run portfolio backtests.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Prediction column to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PRED_PATH,
        help="Path to predictions CSV.",
    )
    args = parser.parse_args()

    if not args.predictions.exists():
        print(f"Predictions file not found: {args.predictions}")
        print("Run scripts/run_linear_baselines.py first.")
        return 1

    import pandas as pd

    pred = pd.read_csv(args.predictions, parse_dates=["date"])
    if args.model not in pred.columns:
        print(f"Column '{args.model}' not in predictions.")
        return 1

    print(f"Using model: {args.model}")
    print(f"Predictions: {len(pred):,} rows")

    monthly, summary = run_backtest(pred, score_col=args.model)
    out_dir = OUTPUTS_DIR / "backtest"
    save_backtest_outputs(monthly, summary, out_dir)

    print(f"\nOutputs saved to {out_dir}/")
    print("\n=== Strategy Summary ===")
    cols = [
        "annualized_return",
        "annualized_stdev",
        "sharpe",
        "capm_alpha_annual",
        "information_ratio",
        "max_drawdown",
        "max_1m_loss",
        "turnover",
        "avg_n_total",
    ]
    print(summary[cols].round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
