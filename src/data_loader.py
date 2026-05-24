"""Load equity panel, factor list, and market CSV datasets."""

from pathlib import Path

import pandas as pd

from src.paths import (
    FACTOR_LIST_PATH,
    MMA_SAMPLE_PATH,
    MKT_IND_PATH,
    TARGET_COLUMN,
)


def _require_file(path: Path) -> None:
    """Raise FileNotFoundError if path does not exist."""
    if not path.exists():
        raise FileNotFoundError(
            f"Expected data file not found: {path}\n"
            "Symlink or copy input CSVs into the data/ directory."
        )


def load_factor_list(path: Path | None = None) -> list[str]:
    """Load the list of 147 stock characteristic column names.

    Args:
        path: Optional override for factor_char_list.csv location.

    Returns:
        Ordered list of factor column names.
    """
    file_path = path or FACTOR_LIST_PATH
    _require_file(file_path)
    factors = pd.read_csv(file_path)["variable"].tolist()
    return factors


def load_mma_panel(path: Path | None = None) -> pd.DataFrame:
    """Load the main stock-month panel (mma_sample_v2.csv).

    The ``date`` column is parsed as datetime and represents the first day
    of the return month (t+1 for ``stock_exret``).

    Args:
        path: Optional override for mma_sample_v2.csv location.

    Returns:
        DataFrame with one row per stock-month observation.
    """
    file_path = path or MMA_SAMPLE_PATH
    _require_file(file_path)
    panel = pd.read_csv(file_path, parse_dates=["date"], low_memory=False)
    return panel


def load_market_data(path: Path | None = None) -> pd.DataFrame:
    """Load risk-free rate and S&P 500 returns (mkt_ind.csv).

    Args:
        path: Optional override for mkt_ind.csv location.

    Returns:
        DataFrame with columns rf, year, month, sp_ret.
    """
    file_path = path or MKT_IND_PATH
    _require_file(file_path)
    market = pd.read_csv(file_path)
    return market


def load_all() -> dict[str, pd.DataFrame | list[str]]:
    """Load all input datasets into a single dictionary.

    Returns:
        Keys: ``panel``, ``factors``, ``market``, ``target``.
    """
    factors = load_factor_list()
    panel = load_mma_panel()
    market = load_market_data()
    return {
        "panel": panel,
        "factors": factors,
        "market": market,
        "target": TARGET_COLUMN,
    }
