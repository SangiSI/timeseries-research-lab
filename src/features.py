def add_lag_features(df, lags):
    for lag in lags:
        df[f"lag_{lag}"] = df["kpi_value"].shift(lag)
    return df
