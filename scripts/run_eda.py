#!/usr/bin/env python3
"""Run exploratory data analysis and save outputs to outputs/eda/."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_all  # noqa: E402
from src.eda import run_eda  # noqa: E402


def main() -> int:
    """Load panel data and run EDA pipeline."""
    data = load_all()
    run_eda(panel=data["panel"], factors=data["factors"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
