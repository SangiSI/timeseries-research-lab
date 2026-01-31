import numpy as np

def zscore_anomaly(series, threshold=3):
    z = (series - series.mean()) / series.std()
    return np.abs(z) > threshold
