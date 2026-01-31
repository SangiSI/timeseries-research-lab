# Dataset Description

This dataset contains synthetic but realistic time series data designed to simulate
operational or consumer-facing KPIs commonly observed in enterprise analytics systems.

The data is intended for experimentation with:
- Time series forecasting
- Anomaly detection
- Residual-based analysis
- Method comparison under noisy, partially labeled conditions

## Structure

Each row represents a daily aggregated metric.

### Columns

- `date`  
  Daily timestamp.

- `kpi_value`  
  Primary operational KPI (e.g. sales index, demand signal, engagement metric).

- `promo_index`  
  Proxy for external effects such as promotions, campaigns, or market events.

- `baseline_trend`  
  Smoothed long-term trend component.

- `seasonality`  
  Weekly seasonal factor.

- `is_anomaly`  
  Synthetic anomaly indicator (used only for evaluation and benchmarking).

## Notes

- Anomaly labels are artificially injected and should not be treated as ground truth.
- The dataset reflects realistic challenges such as noise, seasonality, and drift.
- No real or proprietary data is used.
