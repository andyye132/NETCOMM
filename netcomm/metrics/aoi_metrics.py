import numpy as np


def mean_aoi(latencies_ms) -> float:
    arr = np.asarray(latencies_ms)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))


def percentile_latency(latencies_ms, q: float = 0.99) -> float:
    arr = np.asarray(latencies_ms)
    if arr.size == 0:
        return 0.0
    return float(np.quantile(arr, q))
