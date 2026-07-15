"""Generic submodular maximizers: sequential greedy (Eq 11) and RSP (Algorithm 1).

Target-agnostic: they operate on a list of per-drone candidate actions, where each
action carries its per-target / per-horizon-step position information, plus a
Gaussian-MI set objective. Each drone selects ONE action (partition matroid, Eq 7).

  sequential_greedy : drones decide one at a time (in a given order), each maximizing
                      its marginal MI gain given all previously committed drones
                      -> g(X^g) >= 1/2 g(X*).
  rsp(n_d)          : Randomized Sequential Partitions (Alg 1): shuffle the drones and
                      split into n_d ordered rounds; within a round all drones plan in
                      PARALLEL against the committed union of EARLIER rounds (ignoring
                      same-round peers), then commit the round. VERIFIED semantics:
                      n_d = n_r -> sequential greedy (one drone per round); n_d = 1 ->
                      fully parallel (no coordination). More rounds -> closer to greedy.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .objective import set_objective, batched_objective


def _best_action(cand_i, base_info, priors, weights, F, Q, base_obj):
    """Argmax over drone i's candidate actions of the marginal gain given base_info.

    Evaluates all of drone i's candidates on-device (vmapped objective) and pulls the
    whole (n_actions,) vector back in a single host transfer, instead of one float()
    device->host sync per candidate action.
    """
    objs = np.asarray(batched_objective(base_info, cand_i, priors, weights, F, Q))
    best_a = int(np.argmax(objs))                    # first-max tie-break matches the old loop
    best_obj = float(objs[best_a])
    return best_a, best_obj - base_obj, best_obj


def sequential_greedy(candidate_infos: List[np.ndarray], priors, weights, F, Q,
                      order: Optional[Sequence[int]] = None) -> Dict:
    """Sequential greedy (Eq 11). candidate_infos[i] is (n_actions, M, L, 2, 2)."""
    n_r = len(candidate_infos)
    M, L = priors.shape[0], candidate_infos[0].shape[2]
    order = range(n_r) if order is None else order
    A = np.zeros((M, L, 2, 2))
    base = 0.0
    chosen = np.zeros(n_r, dtype=int)
    gains = np.zeros(n_r)
    for i in order:
        a, g, obj = _best_action(candidate_infos[i], A, priors, weights, F, Q, base)
        A = A + candidate_infos[i][a]
        base = obj
        chosen[i] = a
        gains[i] = g
    return {"chosen": chosen, "gains": gains, "redundancy": np.zeros(n_r),
            "total_mi": base, "perm": np.array(list(order))}


def rsp(candidate_infos: List[np.ndarray], priors, weights, F, Q,
        n_d: int, seed: int = 0) -> Dict:
    """Randomized Sequential Partitions (Algorithm 1). n_d rounds, constant in n_r."""
    n_r = len(candidate_infos)
    M, L = priors.shape[0], candidate_infos[0].shape[2]
    n_d = int(max(1, min(n_d, n_r)))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_r)
    rounds = np.array_split(perm, n_d)               # n_d ordered, balanced rounds

    A = np.zeros((M, L, 2, 2))                        # committed union of earlier rounds
    empty = np.zeros((M, L, 2, 2))
    chosen = np.zeros(n_r, dtype=int)
    gains = np.zeros(n_r)
    redundancy = np.zeros(n_r)
    for members in rounds:
        base = set_objective(priors, weights, A, F, Q)   # prior-rounds objective
        picks = {}
        for i in members:                            # plan in PARALLEL vs prior rounds only
            a, g, _ = _best_action(candidate_infos[i], A, priors, weights, F, Q, base)
            ge = set_objective(priors, weights, empty + candidate_infos[i][a], F, Q)  # gain if alone
            picks[int(i)] = (a, g, ge)
        for i in members:                            # commit the whole round together
            a, g, ge = picks[int(i)]
            A = A + candidate_infos[i][a]
            chosen[i] = a
            gains[i] = g
            redundancy[i] = max(0.0, ge - g)         # MI earlier rounds already covered
    total = set_objective(priors, weights, A, F, Q)
    return {"chosen": chosen, "gains": gains, "redundancy": redundancy,
            "total_mi": total, "perm": perm}


def plan(candidate_infos, priors, weights, F, Q, method="rsp", n_d=2, seed=0) -> Dict:
    """Dispatch: 'greedy' -> sequential_greedy; 'rsp' -> rsp(n_d)."""
    if method == "greedy":
        return sequential_greedy(candidate_infos, priors, weights, F, Q)
    if method == "rsp":
        return rsp(candidate_infos, priors, weights, F, Q, n_d, seed)
    raise ValueError(f"unknown method {method!r}; expected 'greedy' or 'rsp'")
