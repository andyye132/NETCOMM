import numpy as np


def delivery_probability(outcomes) -> float:
    if hasattr(outcomes, "delivered_by_deadline"):
        arr = np.asarray(outcomes.delivered_by_deadline)
    else:
        arr = np.asarray(outcomes)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))
