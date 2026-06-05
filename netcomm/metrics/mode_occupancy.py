from collections import Counter
from typing import Dict, List


def mode_fractions(action_log: List[str]) -> Dict[str, float]:
    if not action_log:
        return {k: 0.0 for k in ("react", "predict", "diversify", "drop")}
    c = Counter(action_log)
    total = float(len(action_log))
    return {k: c.get(k, 0) / total
            for k in ("react", "predict", "diversify", "drop")}
