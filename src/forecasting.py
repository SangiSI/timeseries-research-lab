"""
Forecasting models for time series prediction.

This module contains training and inference helpers for baseline and
machine learning models used to forecast future values of operational
time series data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor


@dataclass(frozen=True)
class ForecastResult:
    """Container for forecast outputs."""
    y_pred: np.ndarray
    model_name: str


def time_split(df: pd.DataFrame, frac_train: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-based split (no shuffling).

    Parameters
    ----------
    df : pd.DataFrame
        Data sorted by time.
    frac_train : float
        Fraction used for training.

    Returns
    -------
    (train_df, test_df)
    """
    if not (0.1 <= frac_train <= 0.95):
        raise ValueError("frac_train should be between 0.1 and 0.95")

    split_idx = int(len(df) * frac_train)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


def naive_lag1_forecast(test_df: pd.DataFrame, lag_col: str = "kpi_value_lag_1") -> ForecastResult:
    """
    Naive forecast using lag-1 feature (y[t] ≈ y[t-1]).

    Parameters
    ----------
    test_df : pd.DataFrame
        Test dataframe containing lag_col.
    lag_col : str
        Name of lag-1 feature.

    Returns
    -------
    ForecastResult
    """
    if lag_col not in test_df.columns:
        raise KeyError(f"Missing {lag_col} for naive forecast.")
    y_pred = test_df[lag_col].to_numpy()
    return ForecastResult(y_pred=y_pred, model_name="naive_lag1")


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 300,
    min_samples_leaf: int = 2,
    random_state: int = 42,
) -> RandomForestRegressor:
    """
    Train a RandomForest regressor (strong, robust baseline for tabular time series features).
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        min_samples_leaf=min_samples_leaf,
    )
    model.fit(X_train, y_train)
    return model


def train_hist_gb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_depth: int = 6,
    random_state: int = 42,
) -> HistGradientBoostingRegressor:
    """
    Train Histogram-based Gradient Boosting regressor (fast and often strong on tabular features).
    """
    model = HistGradientBoostingRegressor(
        random_state=random_state,
        max_depth=max_depth,
    )
    model.fit(X_train, y_train)
    return model


def predict(model, X: pd.DataFrame) -> np.ndarray:
    """Model-agnostic prediction helper."""
    return model.predict(X)


def make_forecast_pack(
    test_df: pd.DataFrame,
    y_true_col: str,
    preds: Dict[str, np.ndarray],
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Build a tidy prediction dataframe for downstream residual analysis.

    Parameters
    ----------
    test_df : pd.DataFrame
        Test dataframe containing date and ground truth.
    y_true_col : str
        Ground truth column name.
    preds : Dict[str, np.ndarray]
        Dict of model_name -> predictions.
    date_col : str
        Date column name.

    Returns
    -------
    pd.DataFrame
        Columns: date, y_true, <pred columns...>
    """
    if date_col not in test_df.columns:
        raise KeyError(f"Missing column: {date_col}")
    if y_true_col not in test_df.columns:
        raise KeyError(f"Missing column: {y_true_col}")

    out = pd.DataFrame({
        "date": test_df[date_col].values,
        "y_true": test_df[y_true_col].values,
    })
    for name, arr in preds.items():
        out[f"y_pred_{name}"] = arr
    return out
