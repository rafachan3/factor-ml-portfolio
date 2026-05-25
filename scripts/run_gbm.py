#!/usr/bin/env python3
"""Train LightGBM model and report OOS R² (2010–2023)."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_all  # noqa: E402
from src.models.gbm import format_gbm_report, run_gbm  # noqa: E402
from src.paths import OUTPUTS_DIR, TARGET_COLUMN  # noqa: E402


def _load_linear_benchmark() -> float | None:
    """Load Ridge OOS R² from linear baseline report if available."""
    report_path = OUTPUTS_DIR / "linear_oos_r2.txt"
    if not report_path.exists():
        return None
    for line in report_path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("ridge"):
            return float(line.split(":")[-1].strip())
    return None


def main() -> int:
    """Load data, run LightGBM, save predictions and OOS R²."""
    data = load_all()
    panel = data["panel"][data["panel"][TARGET_COLUMN].notna()].copy()
    factors = data["factors"]

    print(f"Panel rows with target: {len(panel):,}")
    predictions, oos_r2 = run_gbm(panel, factors)

    out_dir = OUTPUTS_DIR / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "gbm_predictions.csv"
    predictions.to_csv(pred_path, index=False)
    print(f"\nPredictions saved to {pred_path}")

    linear_r2 = _load_linear_benchmark()
    report = format_gbm_report(oos_r2, linear_r2=linear_r2)
    print(f"\n{report}")

    report_path = OUTPUTS_DIR / "gbm_oos_r2.txt"
    report_path.write_text(report + "\n", encoding="utf-8")
    print(f"Report saved to {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
