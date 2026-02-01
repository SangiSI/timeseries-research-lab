"""
End-to-end pipeline for time series forecasting and anomaly detection.

This script runs a reproducible, notebook-free workflow that:
1. Loads raw time series data
2. Builds feature tables
3. Trains forecasting models
4. Generates forecasts and residuals
5. Detects anomalies using statistical and ML methods
6. Writes artifacts to disk for evaluation and analysis

The pipeline is designed to demonstrate how applied research code
can be executed in a production-style setting without relying on
interactive notebooks.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features import build_feature_table
from src.forecasting import (
    time_split,
    naive_lag1_forecast,
    train_random_forest,
    train_hist_gb,
    predict,
    make_forecast_pack,
)
from src.anomaly import (
    attach_residual_columns,
    rolling_zscore_flags,
    isolation_forest_flags,
)
from src.metrics import regression_metrics, anomaly_metrics


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EXP_DIR = ROOT / "experiments"


def main() -> None:
    """
    Execute the full forecasting and anomaly detection pipeline.

    This function orchestrates the complete workflow:
    - feature engineering
    - time-based train/test split
    - model training and forecasting
    - residual-based anomaly detection
    - metric computation and persistence

    All outputs are written to the data/ and experiments/ directories.
    """
    DATA_DIR.mkdir(exist_ok=True)
    EXP_DIR.mkdir(exist_ok=True)

    raw_path = DATA_DIR / "synthetic_kpi.csv"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Missing dataset at {raw_path}. "
            "Generate the dataset before running the pipeline."
        )

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    df = (
        pd.read_csv(raw_path, parse_dates=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    feat = build_feature_table(
        df,
        date_col="date",
        target_col="kpi_value",
    )
    feat = feat.dropna().reset_index(drop=True)

    feat_path = DATA_DIR / "feature_table.parquet"
    feat.to_parquet(feat_path, index=False)

    # ------------------------------------------------------------------
    # Train / test split
    # ------------------------------------------------------------------
    train_df, test_df = time_split(feat, frac_train=0.8)

    target = "kpi_value"
    drop_cols = ["date", target]
    if "is_anomaly" in feat.columns:
        drop_cols.append("is_anomaly")

    X_cols = [c for c in feat.columns if c not in drop_cols]
    X_train, y_train = train_df[X_cols], train_df[target]
    X_test, y_test = test_df[X_cols], test_df[target]

    # ------------------------------------------------------------------
    # Forecasting models
    # ------------------------------------------------------------------
    preds = {}

    naive = naive_lag1_forecast(test_df, lag_col="kpi_value_lag_1")
    preds["naive"] = naive.y_pred

    rf = train_random_forest(X_train, y_train)
    preds["rf"] = predict(rf, X_test)

    gbr = train_hist_gb(X_train, y_train)
    preds["gbr"] = predict(gbr, X_test)

    pred_pack = make_forecast_pack(
        test_df,
        y_true_col=target,
        preds=preds,
        date_col="date",
    )

    pred_path = DATA_DIR / "forecast_predictions.parquet"
    pred_pack.to_parquet(pred_path, index=False)

    # ------------------------------------------------------------------
    # Forecasting metrics
    # ------------------------------------------------------------------
    fm = []
    for name in preds:
        metrics = regression_metrics(
            pred_pack["y_true"].values,
            pred_pack[f"y_pred_{name}"].values,
        )
        metrics["model"] = name
        fm.append(metrics)

    forecast_metrics = (
        pd.DataFrame(fm)
        .sort_values("rmse")
        .reset_index(drop=True)
    )
    forecast_metrics.to_csv(
        EXP_DIR / "forecast_metrics.csv", index=False
    )

    # ------------------------------------------------------------------
    # Residuals and anomaly detection
    # ------------------------------------------------------------------
    pred_with_resid = attach_residual_columns(
        pred_pack,
        y_true_col="y_true",
        y_pred_col="y_pred_rf",
        prefix="rf",
    )

    pred_with_resid["anom_z"] = (
        rolling_zscore_flags(
            pred_with_resid["resid_rf"],
            window=28,
            threshold=3.0,
        )
        .fillna(0)
        .astype(int)
    )

    X_iso = (
        pred_with_resid[["resid_rf", "abs_resid_rf"]]
        .fillna(0.0)
        .to_numpy()
    )
    pred_with_resid["anom_iso"] = isolation_forest_flags(
        X_iso,
        contamination=0.02,
    )

    if "is_anomaly" in df.columns:
        pred_with_resid = pred_with_resid.merge(
            df[["date", "is_anomaly"]],
            on="date",
            how="left",
        )

    out_path = DATA_DIR / "anomaly_outputs.parquet"
    pred_with_resid.to_parquet(out_path, index=False)

    # ------------------------------------------------------------------
    # Anomaly metrics (if labels exist)
    # ------------------------------------------------------------------
    if (
        "is_anomaly" in pred_with_resid.columns
        and pred_with_resid["is_anomaly"].notna().any()
    ):
        y_true_flag = (
            pred_with_resid["is_anomaly"]
            .fillna(0)
            .astype(int)
            .values
        )

        am = []
        for name, col in [
            ("zscore", "anom_z"),
            ("isoforest", "anom_iso"),
        ]:
            m = anomaly_metrics(y_true_flag, pred_with_resid[col].values)
            m["method"] = name
            am.append(m)

        anomaly_metrics_df = (
            pd.DataFrame(am)
            .sort_values("f1", ascending=False)
            .reset_index(drop=True)
        )
        anomaly_metrics_df.to_csv(
            EXP_DIR / "anomaly_metrics.csv", index=False
        )

    print("Pipeline completed successfully.")
    print("Artifacts written to:")
    print(f"  - {feat_path}")
    print(f"  - {pred_path}")
    print(f"  - {out_path}")
    print(f"  - {EXP_DIR / 'forecast_metrics.csv'}")
    if (EXP_DIR / "anomaly_metrics.csv").exists():
        print(f"  - {EXP_DIR / 'anomaly_metrics.csv'}")


if __name__ == "__main__":
    main()
