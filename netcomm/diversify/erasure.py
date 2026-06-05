from itertools import combinations
from typing import Optional

import numpy as np


def k_of_n_decode_prob(survivals: np.ndarray, k: int, n: int,
                       correlations: Optional[np.ndarray] = None,
                       gamma: float = 0.5) -> float:
    # why: Poisson-binomial sum over subsets of size >= k. n <= 8 in practice
    # so the 2**n enumeration is fine; for larger n switch to recursion.
    s = np.clip(np.asarray(survivals, dtype=np.float64), 0.0, 1.0)
    n = int(n)
    k = int(k)
    if n == 0 or k > n:
        return 0.0
    total = 0.0
    indices = list(range(n))
    for size in range(k, n + 1):
        for combo in combinations(indices, size):
            prod = 1.0
            ci = 0
            for i in indices:
                if ci < len(combo) and combo[ci] == i:
                    prod *= s[i]
                    ci += 1
                else:
                    prod *= (1.0 - s[i])
            total += prod
    if correlations is not None:
        # Bahadur first-order correction: subtract shared-failure mass.
        corr = float(np.sum(np.asarray(correlations, dtype=np.float64)))
        total = total - gamma * corr
    return float(np.clip(total, 0.0, 1.0))
