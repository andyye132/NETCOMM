import numpy as np


class AoITracker:
    def __init__(self):
        self.latencies_ms = []

    def update(self, latency_ms: float):
        self.latencies_ms.append(float(latency_ms))

    def mean(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return float(np.mean(self.latencies_ms))

    def percentile(self, q: float = 0.99) -> float:
        if not self.latencies_ms:
            return 0.0
        return float(np.quantile(self.latencies_ms, q))
