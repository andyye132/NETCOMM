import numpy as np


def rebuilds_per_minute(rebuild_events, sim_minutes: float) -> float:
    arr = np.asarray(rebuild_events)
    if sim_minutes <= 0.0:
        return 0.0
    return float(np.sum(arr)) / float(sim_minutes)
