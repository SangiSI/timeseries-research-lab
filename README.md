# Time Series Research Lab

![Pylint](https://github.com/SangiSI/timeseries-research-lab/actions/workflows/pylint.yml/badge.svg)

Applied time series forecasting and anomaly detection using ML, statistical baselines, and residual analysis.

---

## Overview

This repository demonstrates an **applied research workflow** for **time series forecasting and anomaly detection** on operational / consumer-style data.

The focus is not on academic perfection, but on:
- data-centric modeling
- method comparison
- residual-based anomaly detection
- production-oriented execution

The project mirrors how forecasting and anomaly detection systems are typically built and evaluated in **real enterprise analytics environments**.

---

## Key Capabilities

- Feature-based time series forecasting (ML + baselines)
- Residual-driven anomaly detection
- Statistical and unsupervised ML methods
- Time-aware train/test splitting
- Reproducible, notebook-free pipeline execution
- Clear separation between research and reusable logic

---

## Repository Structure

```text
timeseries-research-lab/
├── data/
│   ├── synthetic_kpi.csv          # Synthetic time series dataset
│   ├── feature_table.parquet      # Generated feature table
│   └── README.md                  # Dataset documentation
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_forecasting_models.ipynb
│   ├── 04_anomaly_detection.ipynb
│   └── 05_model_comparison.ipynb
│
├── src/
│   ├── features.py                # Feature engineering utilities
│   ├── forecasting.py             # Forecasting models and helpers
│   ├── anomaly.py                 # Anomaly detection logic
│   └── metrics.py                 # Evaluation metrics
│
├── scripts/
│   └── run_pipeline.py             # End-to-end executable pipeline
│
├── experiments/
│   ├── forecast_metrics.csv
│   └── anomaly_metrics.csv
│
└── README.md
```

---

## Dataset

The dataset (`synthetic_kpi.csv`) simulates **aggregated operational / consumer KPIs**, similar to:

- sales or demand indices  
- market activity signals  
- engagement or performance metrics  

**Characteristics**
- Daily frequency
- Long-term trend + weekly seasonality
- External effects (promotion-like signal)
- Noise and drift
- Injected anomalies (for benchmarking only)

> ⚠️ All data is synthetic and anonymized.  
> Anomaly labels are artificial and used only for evaluation.

See `data/README.md` for details.

---

## Modeling Approach

### Forecasting
- Naive lag-based baseline
- Random Forest regression
- Histogram-based Gradient Boosting
- Feature-driven modeling using lags, rolling statistics, and calendar features

### Anomaly Detection
- Residual-based statistical detection (rolling z-score)
- Unsupervised ML (Isolation Forest)
- Emphasis on stability and false-positive control

Forecasting and anomaly detection are treated as **coupled problems**, where anomalies are identified as **deviations from expected behavior**.

---

## Notebooks (Research & Analysis)

The notebooks document the applied research process:

1. **Exploration** – trends, seasonality, external effects  
2. **Feature Engineering** – lags, rolling windows, calendar signals  
3. **Forecasting Models** – baselines vs ML models  
4. **Anomaly Detection** – statistical and ML approaches  
5. **Comparison** – accuracy, stability, and trade-offs  

Notebooks are intended for **analysis and explanation**, not production execution.

---

## Pipeline Execution (Production-Style)

In addition to notebooks, the repository includes a **fully runnable pipeline** that executes the complete workflow end-to-end without notebooks.

### Run the pipeline

From the repository root:

```bash
python -m scripts.run_pipeline
