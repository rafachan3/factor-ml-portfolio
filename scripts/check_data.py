#!/usr/bin/env python3
"""Run data integrity checks and save report to outputs/."""

import sys
from pathlib import Path

# Allow running as `python scripts/check_data.py` from repo root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_all  # noqa: E402
from src.integrity_checks import run_integrity_checks  # noqa: E402
from src.paths import OUTPUTS_DIR  # noqa: E402


def main() -> int:
    """Load data, run checks, print and persist report."""
    data = load_all()
    report = run_integrity_checks(
        panel=data["panel"],
        factors=data["factors"],
        market=data["market"],
    )
    text = report.render()
    print(text)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "m1_integrity_report.txt"
    out_path.write_text(text, encoding="utf-8")
    print(f"Report saved to {out_path}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
