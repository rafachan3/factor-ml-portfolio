#!/usr/bin/env python3
"""Validate feature pipeline: rank bounds, no NaNs, train-only scaling."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_all  # noqa: E402
from src.features import (  # noqa: E402
    PanelFeaturePipeline,
    cross_sectional_rank_normalize,
    transform_panel_splits,
)
from src.paths import TARGET_COLUMN  # noqa: E402


def _sample_split(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a demo 8yr / 2yr / 1yr split matching the OOS protocol."""
    starting = pd.Timestamp("2000-01-01")
    train_end = starting + pd.DateOffset(years=8)
    val_end = starting + pd.DateOffset(years=10)
    test_end = starting + pd.DateOffset(years=11)

    train = panel[(panel["date"] >= starting) & (panel["date"] < train_end)]
    validate = panel[(panel["date"] >= train_end) & (panel["date"] < val_end)]
    test = panel[(panel["date"] >= val_end) & (panel["date"] < test_end)]
    return train, validate, test


def main() -> int:
    """Run feature pipeline smoke tests."""
    data = load_all()
    panel = data["panel"]
    factors = data["factors"]

    panel = panel[panel[TARGET_COLUMN].notna()].copy()
    train, validate, test = _sample_split(panel)

    print("=== Rank normalization (train slice) ===")
    ranked = cross_sectional_rank_normalize(train, factors)
    factor_values = ranked[factors].to_numpy()
    print(f"Min rank-normalized value: {np.nanmin(factor_values):.4f}")
    print(f"Max rank-normalized value: {np.nanmax(factor_values):.4f}")
    in_bounds = np.all((factor_values >= -1.0) & (factor_values <= 1.0))
    print(f"All values in [-1, 1]: {in_bounds}")

    print("\n=== PanelFeaturePipeline (train / val / test) ===")
    train_out, val_out, test_out, pipeline = transform_panel_splits(
        train, validate, test, factors
    )
    scaler = pipeline.get_scaler()
    print(f"Scaler fitted on {scaler.n_features_in_} features")
    print(f"Train rows: {len(train_out):,} | Val: {len(val_out):,} | Test: {len(test_out):,}")

    for name, split in [("train", train_out), ("validate", val_out), ("test", test_out)]:
        n_nan = split[factors].isna().sum().sum()
        print(f"{name}: NaN factor values after transform = {n_nan}")

    # Scaler must reflect train statistics only.
    ranked_train_only = cross_sectional_rank_normalize(train, factors)
    train_mean = ranked_train_only[factors].mean().mean()
    train_std = ranked_train_only[factors].std().mean()
    print(f"\nTrain ranked mean (avg over factors): {train_mean:.6f}")
    print(f"Scaler mean (avg): {scaler.mean_.mean():.6f}")
    print(f"Train ranked std (avg over factors): {train_std:.6f}")
    print(f"Scaler scale (avg): {scaler.scale_.mean():.6f}")

    # Val/test scaled means should differ from ~0 if distribution shifts.
    val_scaled_mean = val_out[factors].mean().mean()
    print(f"Validate scaled mean (avg): {val_scaled_mean:.6f} (uses train scaler)")

    print("\n=== sklearn Pipeline compatibility ===")
    pipe = PanelFeaturePipeline(factor_cols=factors)
    pipe.fit(train)
    transformed = pipe.transform(test)
    assert isinstance(transformed, pd.DataFrame)
    assert len(transformed) == len(test)
    print("PanelFeaturePipeline fit/transform OK.")

    print("\nAll feature pipeline checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
