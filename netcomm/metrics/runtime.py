import time
from collections import defaultdict


class RuntimeRecorder:
    def __init__(self):
        self.totals_ms = defaultdict(float)
        self.counts = defaultdict(int)
        self._last = None

    def lap(self, label: str):
        now = time.perf_counter()
        if self._last is not None:
            dt_ms = (now - self._last) * 1e3
            self.totals_ms[label] += dt_ms
            self.counts[label] += 1
        self._last = now

    def summary(self):
        return {k: self.totals_ms[k] / max(1, self.counts[k]) for k in self.totals_ms}
