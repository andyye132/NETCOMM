
def adaptive_cadence(pi_up_local: float, base_cadence: float) -> float:
    cadence = base_cadence / (max(0.0, pi_up_local) + 0.1)
    return max(cadence, base_cadence / 10.0)
