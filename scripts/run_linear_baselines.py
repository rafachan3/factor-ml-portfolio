#!/usr/bin/env python3
"""Train penalized linear baselines and report OOS R² (2010–2023)."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_all  # noqa: E402
from src.models.linear_baselines import (  # noqa: E402
    format_oos_r2_report,
    run_linear_baselines,
)
from src.paths import OUTPUTS_DIR, TARGET_COLUMN  # noqa: E402


def main() -> int:
    """Load data, run linear baselines, save predictions and OOS R²."""
    data = load_all()
    panel = data["panel"]
    factors = data["factors"]

    panel = panel[panel[TARGET_COLUMN].notna()].copy()
    print(f"Panel rows with target: {len(panel):,}")

    predictions, oos_r2 = run_linear_baselines(panel, factors)

    out_dir = OUTPUTS_DIR / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = out_dir / "linear_predictions.csv"
    predictions.to_csv(pred_path, index=False)
    print(f"\nPredictions saved to {pred_path}")

    report = format_oos_r2_report(oos_r2)
    print(f"\n{report}")

    report_path = OUTPUTS_DIR / "linear_oos_r2.txt"
    report_path.write_text(report + "\n", encoding="utf-8")
    print(f"Report saved to {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
