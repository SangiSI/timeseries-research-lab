"""
Evaluation metrics for forecasting and anomaly detection.

This module provides helper functions to compute regression metrics
for forecasting models and classification-style metrics for anomaly
detection performance.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import precision_recall_fscore_support


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute basic regression metrics for forecasting.

    Returns
    -------
    dict: rmse, mae
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "rmse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def anomaly_metrics(y_true: np.ndarray, y_pred_flag: np.ndarray) -> Dict[str, float]:
    """
    Compute classification-style metrics for anomaly flags.

    Returns
    -------
    dict: precision, recall, f1
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred_flag = np.asarray(y_pred_flag).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_flag, average="binary", zero_division=0
    )
    return {"precision": float(p), "recall": float(r), "f1": float(f1)}
