"""
Feature engineering utilities for time series data.

This module provides reusable functions to generate calendar features,
lagged values, and rolling statistics commonly used in forecasting
and anomaly detection workflows.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd


def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Add common calendar features derived from a datetime column.

    Adds:
      - dow (0=Mon..6=Sun)
      - month (1..12)
      - day (1..31)
      - is_weekend (0/1)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing date_col.
    date_col : str
        Name of datetime column.

    Returns
    -------
    pd.DataFrame
        Copy of df with added features.
    """
    if date_col not in df.columns:
        raise KeyError(f"Missing required column: {date_col}")

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    if out[date_col].isna().any():
        raise ValueError(f"Column {date_col} contains non-parsable dates.")

    out["dow"] = out[date_col].dt.dayofweek
    out["month"] = out[date_col].dt.month
    out["day"] = out[date_col].dt.day
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    return out


def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lags: Sequence[int] = (1, 7, 14),
) -> pd.DataFrame:
    """
    Add lagged features for a target column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target_col : str
        Name of target column.
    lags : Sequence[int]
        Lags to compute.

    Returns
    -------
    pd.DataFrame
        Copy of df with lag features appended.
    """
    if target_col not in df.columns:
        raise KeyError(f"Missing required column: {target_col}")
    if not lags:
        return df.copy()

    out = df.copy()
    for lag in lags:
        if lag <= 0:
            raise ValueError("All lags must be positive integers.")
        out[f"{target_col}_lag_{lag}"] = out[target_col].shift(lag)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    windows: Sequence[int] = (7, 14, 28),
    shift: int = 1,
) -> pd.DataFrame:
    """
    Add rolling mean/std features using historical values only.

    Rolling features are computed on (target shifted by `shift`) to avoid leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target_col : str
        Target column.
    windows : Sequence[int]
        Window sizes.
    shift : int
        Shift applied before rolling aggregation (default=1).

    Returns
    -------
    pd.DataFrame
        Copy of df with rolling features appended.
    """
    if target_col not in df.columns:
        raise KeyError(f"Missing required column: {target_col}")
    if shift < 0:
        raise ValueError("shift must be >= 0")

    out = df.copy()
    series = out[target_col].shift(shift)

    for w in windows:
        if w <= 1:
            raise ValueError("windows must be > 1")
        out[f"{target_col}_roll_mean_{w}"] = series.rolling(w).mean()
        out[f"{target_col}_roll_std_{w}"] = series.rolling(w).std()
    return out


def build_feature_table(
    df: pd.DataFrame,
    date_col: str = "date",
    target_col: str = "kpi_value",
    lags: Sequence[int] = (1, 7, 14),
    windows: Sequence[int] = (7, 14, 28),
    include_promo_flag: bool = True,
    promo_col: str = "promo_index",
) -> pd.DataFrame:
    """
    Convenience function to build a feature table for forecasting/anomaly workflows.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataframe with at least date_col and target_col.
    date_col : str
        Date column name.
    target_col : str
        Target KPI column name.
    lags : Sequence[int]
        Lag feature configuration.
    windows : Sequence[int]
        Rolling feature configuration.
    include_promo_flag : bool
        Whether to add a binary promo flag when promo_col exists.
    promo_col : str
        External signal column.

    Returns
    -------
    pd.DataFrame
        Feature-enriched dataframe (may contain NaNs due to lags/rolls).
    """
    out = add_calendar_features(df, date_col=date_col)
    out = add_lag_features(out, target_col=target_col, lags=lags)
    out = add_rolling_features(out, target_col=target_col, windows=windows, shift=1)

    if include_promo_flag and promo_col in out.columns:
        out["promo_flag"] = (out[promo_col] > 0).astype(int)

    return out
