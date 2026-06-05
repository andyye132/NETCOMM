from typing import Dict

import numpy as np


def brier_score(predicted, observed) -> float:
    p = np.asarray(predicted, dtype=np.float64)
    o = np.asarray(observed, dtype=np.float64)
    if p.size == 0:
        return 0.0
    return float(np.mean((p - o) ** 2))


def reliability_bins(predicted, observed, n_bins: int = 10) -> Dict:
    p = np.asarray(predicted, dtype=np.float64)
    o = np.asarray(observed, dtype=np.float64)
    if p.size == 0:
        return {"bin_centers": np.array([]), "bin_mean_pred": np.array([]),
                "bin_freq_obs": np.array([]), "bin_count": np.array([])}
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_mean_pred = np.zeros(n_bins)
    bin_freq_obs = np.zeros(n_bins)
    bin_count = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (p >= lo) & (p < hi if b < n_bins - 1 else p <= hi)
        if not np.any(mask):
            continue
        bin_mean_pred[b] = float(np.mean(p[mask]))
        bin_freq_obs[b] = float(np.mean(o[mask]))
        bin_count[b] = int(np.sum(mask))
    return {
        "bin_centers": centers,
        "bin_mean_pred": bin_mean_pred,
        "bin_freq_obs": bin_freq_obs,
        "bin_count": bin_count,
    }


def expected_calibration_error(predicted, observed, n_bins: int = 10) -> float:
    bins = reliability_bins(predicted, observed, n_bins)
    counts = bins["bin_count"]
    total = float(counts.sum())
    if total <= 0.0:
        return 0.0
    weights = counts / total
    gaps = np.abs(bins["bin_mean_pred"] - bins["bin_freq_obs"])
    return float(np.sum(weights * gaps))
