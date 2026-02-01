"""
Anomaly detection methods for time series data.

This module implements statistical and unsupervised anomaly detection
techniques, with a focus on residual-based detection following
forecasting models.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def compute_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute residual and absolute residual arrays.

    Returns
    -------
    (residual, abs_residual)
    """
    resid = np.asarray(y_true) - np.asarray(y_pred)
    abs_resid = np.abs(resid)
    return resid, abs_resid


def rolling_zscore_flags(
    residuals: pd.Series,
    window: int = 28,
    threshold: float = 3.0,
) -> pd.Series:
    """
    Detect anomalies using rolling z-score over residuals.

    Parameters
    ----------
    residuals : pd.Series
        Residual series (y_true - y_pred).
    window : int
        Rolling window length.
    threshold : float
        Z-score magnitude threshold.

    Returns
    -------
    pd.Series
        Binary anomaly flags (1 anomaly, 0 normal).
    """
    if window <= 3:
        raise ValueError("window should be > 3")
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    mu = residuals.rolling(window).mean()
    sigma = residuals.rolling(window).std().replace(0, 1e-6)
    z = (residuals - mu) / sigma
    flags = (z.abs() > threshold).astype(int)
    return flags


def isolation_forest_flags(
    features: np.ndarray,
    contamination: float = 0.02,
    random_state: int = 42,
    n_estimators: int = 300,
) -> np.ndarray:
    """
    Detect anomalies using Isolation Forest.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix (e.g., residual, abs_residual, etc.).
    contamination : float
        Expected fraction of anomalies.
    random_state : int
        Random seed.
    n_estimators : int
        Number of trees.

    Returns
    -------
    np.ndarray
        Binary anomaly flags (1 anomaly, 0 normal).
    """
    if not (0.001 <= contamination <= 0.2):
        raise ValueError("contamination should be in [0.001, 0.2] for practical use")

    X = np.asarray(features, dtype=float)
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    preds = model.fit_predict(X)  # -1 is anomaly
    return (preds == -1).astype(int)


def attach_residual_columns(
    df_pred: pd.DataFrame,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred_rf",
    prefix: str = "rf",
) -> pd.DataFrame:
    """
    Add residual and abs residual columns to a prediction dataframe.

    Parameters
    ----------
    df_pred : pd.DataFrame
        Prediction dataframe containing y_true_col and y_pred_col.
    y_true_col : str
        Ground truth column name.
    y_pred_col : str
        Prediction column name.
    prefix : str
        Prefix for new columns.

    Returns
    -------
    pd.DataFrame
        Copy of df_pred with residual columns.
    """
    if y_true_col not in df_pred.columns or y_pred_col not in df_pred.columns:
        raise KeyError(f"Missing {y_true_col} or {y_pred_col}")

    out = df_pred.copy()
    resid = out[y_true_col] - out[y_pred_col]
    out[f"resid_{prefix}"] = resid
    out[f"abs_resid_{prefix}"] = resid.abs()
    return out
