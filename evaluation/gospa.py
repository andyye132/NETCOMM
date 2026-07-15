"""GOSPA — Generalized Optimal Sub-Pattern Assignment metric (Rahmathullah,
Garcia-Fernandez, Svensson, FUSION 2017).

Scores an estimated set of target positions against the true set. Solved exactly
as a min-cost assignment that may leave points unassigned, each unassigned point
paying the penalty c^p / alpha. At alpha = 2 the metric decomposes into

    GOSPA^p = (localization error)^p + (c^p/2)·(#missed) + (c^p/2)·(#false)

where missed = true targets with no matched estimate, false = estimates with no
matched truth. Preferred over OSPA because it cleanly separates localization,
missed, and false (and avoids OSPA's cardinality-normalization "spooky effect").
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

_BIG = 1e12


def gospa(estimates: Sequence, truths: Sequence, c: float = 10.0,
          p: float = 2.0, alpha: float = 2.0) -> Dict:
    """GOSPA distance between estimate positions and true positions.

    Returns dict: total, localization, missed (int), false (int), n_matched (int).
    estimates/truths are (·, 2) arrays of 2D positions (empty allowed).
    """
    # This implementation forbids matches beyond the cutoff (-> the pair becomes
    # 1 missed + 1 false), which equals the canonical capped-distance GOSPA only at
    # alpha = 2 (where 2*(c^p/alpha) == c^p). alpha != 2 needs the capped variant.
    if not np.isclose(alpha, 2.0):
        raise ValueError("gospa() supports alpha=2 only (the decomposing case); "
                         f"got alpha={alpha}. Use the capped-distance variant for alpha!=2.")
    X = np.asarray(estimates, dtype=float).reshape(-1, 2)   # estimates
    Y = np.asarray(truths, dtype=float).reshape(-1, 2)      # truths
    m, n = X.shape[0], Y.shape[0]
    pen = (c ** p) / alpha

    if m == 0 and n == 0:
        return {"total": 0.0, "localization": 0.0, "missed": 0, "false": 0, "n_matched": 0}
    if m == 0:
        return {"total": (n * pen) ** (1.0 / p), "localization": 0.0,
                "missed": n, "false": 0, "n_matched": 0}
    if n == 0:
        return {"total": (m * pen) ** (1.0 / p), "localization": 0.0,
                "missed": 0, "false": m, "n_matched": 0}

    # square (m+n) cost matrix: rows = [estimates | truth-dummies], cols = [truths | estimate-dummies]
    size = m + n
    C = np.full((size, size), _BIG)
    D = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=2) ** p   # (m, n)
    D[D > c ** p] = _BIG                                             # forbid matches beyond cutoff
    C[:m, :n] = D
    np.fill_diagonal(C[:m, n:n + m], pen)        # estimate i -> its dummy (false)
    np.fill_diagonal(C[m:m + n, :n], pen)        # truth j -> its dummy (missed)
    C[m:m + n, n:n + m] = 0.0                    # dummy-dummy (free)

    rows, cols = linear_sum_assignment(C)
    loc, matched, missed, false = 0.0, 0, 0, 0
    for r, cc in zip(rows, cols):
        if r < m and cc < n:                     # estimate-truth pairing
            if C[r, cc] < _BIG:
                loc += C[r, cc]
                matched += 1
            else:                                # forbidden match -> both unassigned
                false += 1
                missed += 1
        elif r < m and cc >= n:                  # estimate -> dummy
            false += 1
        elif r >= m and cc < n:                  # truth -> dummy
            missed += 1
        # r>=m and cc>=n: dummy-dummy, no cost

    total = (loc + pen * (missed + false)) ** (1.0 / p)
    return {"total": total, "localization": loc ** (1.0 / p),
            "missed": missed, "false": false, "n_matched": matched}
