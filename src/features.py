"""Feature preprocessing for cross-sectional equity return models."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from src.paths import TARGET_COLUMN


def cross_sectional_rank_normalize(
    panel: pd.DataFrame,
    factors: Sequence[str],
    date_col: str = "date",
) -> pd.DataFrame:
    """Apply cross-sectional median imputation and rank normalization.

    For each month and each factor:
      1. Impute missing values with the cross-sectional median.
      2. Dense-rank stocks and scale ranks to [-1, 1].

    Matches the preprocessing in the reference penalized-linear workflow.
    No parameters are estimated across months (no look-ahead).

    Args:
        panel: Stock-month panel containing ``date_col`` and factor columns.
        factors: Predictor column names to transform.
        date_col: Column used to group cross sections.

    Returns:
        Copy of ``panel`` with factor columns rank-normalized.
    """
    if date_col not in panel.columns:
        raise ValueError(f"Panel missing date column '{date_col}'.")

    out = panel.copy()
    grouped = out.groupby(date_col, sort=False)

    for factor in factors:
        if factor not in out.columns:
            raise ValueError(f"Panel missing factor column '{factor}'.")

        transformed = pd.Series(index=out.index, dtype=float)
        for _, group in grouped:
            values = group[factor].copy()
            median = values.median(skipna=True)
            values = values.fillna(median)
            ranks = values.rank(method="dense") - 1
            rank_max = ranks.max()
            if rank_max > 0:
                values = (ranks / rank_max) * 2 - 1
            else:
                values = 0.0
            transformed.loc[group.index] = values.values

        out[factor] = transformed

    return out


class CrossSectionalRankNormalizer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible cross-sectional rank normalizer for panel data.

    ``fit`` is a no-op: each month's transform uses only that month's cross
    section. Accepts a pandas DataFrame with a date column and factor columns.
    """

    def __init__(
        self,
        factor_cols: Sequence[str],
        date_col: str = "date",
        copy: bool = True,
    ) -> None:
        self.factor_cols = list(factor_cols)
        self.date_col = date_col
        self.copy = copy

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> CrossSectionalRankNormalizer:
        """Validate input columns; no state is learned."""
        self._validate_input(X)
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Rank-normalize factor columns within each date cross section."""
        check_is_fitted(self, "feature_names_in_")
        self._validate_input(X)
        panel = X.copy() if self.copy else X
        ranked = cross_sectional_rank_normalize(
            panel=panel,
            factors=self.factor_cols,
            date_col=self.date_col,
        )
        return ranked

    def _validate_input(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CrossSectionalRankNormalizer requires a DataFrame.")
        if self.date_col not in X.columns:
            raise ValueError(f"Input missing date column '{self.date_col}'.")
        missing = [c for c in self.factor_cols if c not in X.columns]
        if missing:
            raise ValueError(f"Input missing factor columns: {missing[:5]}")

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return output feature names (unchanged column names)."""
        check_is_fitted(self, "feature_names_in_")
        return np.asarray(self.feature_names_in_, dtype=object)


class PanelFeaturePipeline(BaseEstimator, TransformerMixin):
    """Two-step panel feature pipeline: rank normalize, then train-only scale.

    ``fit`` rank-transforms the training panel and fits ``StandardScaler`` on
    factor columns. ``transform`` rank-transforms the input panel (each month
    using its own cross section) and applies the training scaler.

    Args:
        factor_cols: Predictor columns to preprocess.
        date_col: Cross-sectional grouping column.
        copy: Whether to copy input data before transforming.
    """

    def __init__(
        self,
        factor_cols: Sequence[str],
        date_col: str = "date",
        copy: bool = True,
    ) -> None:
        self.factor_cols = list(factor_cols)
        self.date_col = date_col
        self.copy = copy

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> PanelFeaturePipeline:
        """Fit StandardScaler on rank-normalized training factors."""
        self._validate_input(X)
        self.rank_normalizer_ = CrossSectionalRankNormalizer(
            factor_cols=self.factor_cols,
            date_col=self.date_col,
            copy=True,
        )
        self.rank_normalizer_.fit(X)

        ranked_train = self.rank_normalizer_.transform(X)
        self.scaler_ = StandardScaler()
        self.scaler_.fit(ranked_train[self.factor_cols])

        self.feature_names_in_ = list(X.columns)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Rank-normalize and scale factor columns."""
        check_is_fitted(self, "scaler_")
        self._validate_input(X)

        panel = X.copy() if self.copy else X
        ranked = self.rank_normalizer_.transform(panel)
        ranked[self.factor_cols] = self.scaler_.transform(ranked[self.factor_cols])
        return ranked

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Fit on training panel and return transformed training panel."""
        return self.fit(X, y).transform(X)

    def get_scaler(self) -> StandardScaler:
        """Return the fitted StandardScaler for inspection."""
        check_is_fitted(self, "scaler_")
        return self.scaler_

    def _validate_input(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("PanelFeaturePipeline requires a DataFrame.")
        if self.date_col not in X.columns:
            raise ValueError(f"Input missing date column '{self.date_col}'.")
        missing = [c for c in self.factor_cols if c not in X.columns]
        if missing:
            raise ValueError(f"Input missing factor columns: {missing[:5]}")


def transform_panel_splits(
    train: pd.DataFrame,
    validate: pd.DataFrame,
    test: pd.DataFrame,
    factors: Sequence[str],
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, PanelFeaturePipeline]:
    """Preprocess train/validate/test splits with no scaler leakage.

    Rank normalization is applied independently within each split (each month
    uses only stocks present in that split for that month). Because splits are
    by date, this matches ranking the full panel before date cuts.

    Args:
        train: Training panel.
        validate: Validation panel.
        test: Test panel.
        factors: Factor column names.
        date_col: Cross-sectional grouping column.

    Returns:
        Tuple of transformed train, validate, test, and fitted pipeline.
    """
    pipeline = PanelFeaturePipeline(
        factor_cols=factors,
        date_col=date_col,
    )
    train_out = pipeline.fit_transform(train)
    validate_out = pipeline.transform(validate)
    test_out = pipeline.transform(test)
    return train_out, validate_out, test_out, pipeline


def extract_xy(
    panel: pd.DataFrame,
    factors: Sequence[str],
    target: str = TARGET_COLUMN,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix X and target vector y from a panel."""
    X = panel[list(factors)].values
    y = panel[target].values
    return X, y
