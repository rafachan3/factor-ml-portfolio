"""Expanding-window train / validation / test splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd

OOS_START = pd.Timestamp("2000-01-01")
OOS_END = pd.Timestamp("2024-01-01")


@dataclass(frozen=True)
class ExpandingWindow:
    """One expanding-window split for annual OOS refitting."""

    counter: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validate_end: pd.Timestamp
    test_end: pd.Timestamp

    @property
    def test_year(self) -> int:
        """Calendar year of the OOS test window start."""
        return self.train_end.year + 2


def iter_expanding_windows(
    panel_start: pd.Timestamp = OOS_START,
    final_limit: pd.Timestamp = OOS_END,
) -> Iterator[ExpandingWindow]:
    """Yield expanding 8yr-train / 2yr-val / 1yr-test window definitions.

    Matches the reference penalized-linear expanding-window protocol:
    first OOS year is 2010, last is 2023 (14 annual folds).

    Args:
        panel_start: Sample start date (Jan 2000).
        final_limit: Stop when test end would exceed this date.

    Yields:
        ExpandingWindow metadata for each annual refit.
    """
    counter = 0
    while panel_start + pd.DateOffset(years=11 + counter) <= final_limit:
        yield ExpandingWindow(
            counter=counter,
            train_start=panel_start,
            train_end=panel_start + pd.DateOffset(years=8 + counter),
            validate_end=panel_start + pd.DateOffset(years=10 + counter),
            test_end=panel_start + pd.DateOffset(years=11 + counter),
        )
        counter += 1


def split_panel_by_window(
    panel: pd.DataFrame,
    window: ExpandingWindow,
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cut a panel into train, validation, and test subsets for one window.

    Args:
        panel: Full stock-month panel.
        window: Expanding-window date boundaries.
        date_col: Date column for filtering.

    Returns:
        Train, validation, and test DataFrames.
    """
    dates = panel[date_col]
    train = panel[(dates >= window.train_start) & (dates < window.train_end)]
    validate = panel[(dates >= window.train_end) & (dates < window.validate_end)]
    test = panel[(dates >= window.validate_end) & (dates < window.test_end)]
    return train, validate, test
