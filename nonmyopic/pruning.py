"""Pruned minimax planner: same optimum as tree.minimax_value, fewer visited nodes.

Two optimality-preserving prunings from Zhang & Tokekar (2016), built on the exact
full-enumeration tree in tree.py:

  * **Alpha-beta pruning (Thm 3).** The tree alternates control levels (MIN: the robot
    picks an action) and measurement levels (MAX: "nature" picks the worst candidate
    measurement). We carry a `beta` = best (lowest) worst-case value found so far at the
    enclosing MIN node. A control (MIN) subtree whose running max already reaches
    `beta` (within an `eps1` slack) cannot win and is cut. We NEVER prune measurement
    (MAX) nodes: the robot does not control the measurement, so every candidate must be
    expanded to certify the worst case. eps1 = 0 is exact.

  * **Algebraic-redundancy pruning (Thm 2, via the Riccati monotonicity Thm 5).** Sibling
    measurement-result nodes at one measurement level share the SAME robot pose and the SAME
    remaining control set; in the Kalman update the POSTERIOR COVARIANCE is independent of
    the measurement VALUE (Sigma_upd = (I - K H) Sigma_pred), so siblings differ only in
    their updated MEAN. The subtree below depends on the mean only through the future,
    estimate-dependent measurement noise R. At the DEEPEST measurement level (the next node
    is a covariance-trace leaf) the value is mean-independent, so by Thm 5 a sibling whose
    posterior covariance is PSD-dominated, Sigma^A <= Sigma^B, has subtree value <= the
    dominator's and is redundant for the MAX -- it is collapsed away exactly. (At shallower
    measurement levels the value depends on the mean through the deeper R, so we do NOT drop
    there: covariance domination alone is not sufficient. The full Thm-2 convex-combination
    bound with the worst-case-noise term -- which WOULD justify pruning at every level -- is
    left as a noted extension.) eps2 loosens the PSD test.
    This special case never removes the true argmax of the MAX.

`minimax_value_pruned` returns (value, best_first_action, n_nodes_visited) and must equal
tree.minimax_value's (value, action) on every instance; the node count is the payoff.
"""
from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np

from .riccati import cv_matrices, predict, kalman_update, riccati_step, H
from .sampler import candidate_measurements
from .tree import action_offsets


def _psd_dominated(Sa: np.ndarray, Sb: np.ndarray, eps: float) -> bool:
    """True if Sa <= Sb (+ eps I) in the PSD order on the FULL covariance. By the Riccati
    monotonicity (Thm 5) the worst-case-trace subtree value is monotone in the WHOLE
    posterior covariance (the velocity block drives future predicted covariance, not just
    the position block), so a sibling A with Sa <= Sb is PSD-dominated: every future trace
    under A is <= the one under B, hence A is redundant for the MAX and can be dropped."""
    diff = Sb - Sa + eps * np.eye(Sa.shape[0])
    diff = 0.5 * (diff + diff.T)
    return bool(np.linalg.eigvalsh(diff).min() >= -1e-12)


def _surviving_measurements(covs: List[np.ndarray], eps2: float) -> List[int]:
    """Indices of measurement-result siblings NOT PSD-dominated by another sibling.

    Dropping a dominated node never changes the MAX at a mean-independent (deepest)
    measurement level: its subtree value is <= the dominator's (Thm 2 one-hot case / Thm
    5). Ties are broken by index so the kept set is deterministic and at least one node
    always survives. (Here, where all siblings share the value-independent posterior
    covariance, the dominance is mutual and this collapses them to a single keeper.)"""
    n = len(covs)
    keep = list(range(n))
    dropped = [False] * n
    for i in range(n):
        if dropped[i]:
            continue
        for j in range(n):
            if i == j or dropped[j]:
                continue
            # drop i if it is dominated by j; tie -> keep the lower index (drop the higher)
            if _psd_dominated(covs[i], covs[j], eps2):
                if not _psd_dominated(covs[j], covs[i], eps2) or i > j:
                    dropped[i] = True
                    break
    survivors = [k for k in keep if not dropped[k]]
    return survivors if survivors else [0]


def _min_node(robot_xy, x_hat, Sigma, depth, R_fn, offs, F, Q, n_meas,
              eps1: float, eps2: float, use_alpha: bool, use_redundancy: bool,
              counter: List[int]) -> Tuple[float, int]:
    """MIN (control) node. The robot picks the action minimizing the worst-case (MAX over
    candidate measurements) trace. The covariance is advanced by the Eq-7 Riccati map rho
    (riccati_step, predicted-to-predicted) -- the SAME for every candidate measurement at a
    node (value-independent); each candidate advances only the MEAN that sets the deeper R.
    The depth-0 leaf value is the trace of the Eq-7 PREDICTED covariance position block.

    Alpha pruning (Thm 3): each action's value is a MAX, so once an action's running max
    reaches the incumbent best at THIS node it can never beat it and the rest of that action's
    measurement subtree is skipped. We never drop a measurement node for being unable to
    improve the MAX -- we only stop expanding an ALREADY-LOSING control action. Returns
    (value, best_action_index)."""
    if depth == 0:
        return float(np.trace(Sigma[:2, :2])), -1

    best_val, best_a = np.inf, 0
    for ai in range(offs.shape[0]):
        new_robot = np.asarray(robot_xy, float) + offs[ai]
        x_pred, Sigma_pred = predict(x_hat, Sigma, F, Q)
        R = R_fn(new_robot, x_pred[:2])

        # Eq-7 covariance for the children (value-independent: identical across candidates).
        Sigma_next = riccati_step(Sigma, R, F, Q)
        cands = candidate_measurements(x_pred, Sigma_pred, R, n_meas)
        # advance the MEAN per candidate measurement (covariance is the shared Sigma_next)
        means = [kalman_update(x_pred, Sigma_pred, z, R)[0] for z in cands]
        covs = [Sigma_next for _ in cands]

        # Redundancy pruning only at the deepest measurement level (next node is a
        # mean-independent trace leaf), where covariance domination is sufficient and exact.
        # Under Eq 7 the candidate covariances are identical, so this collapses them to one.
        if use_redundancy and depth == 1:
            survivors = _surviving_measurements(covs, eps2)
        else:
            survivors = list(range(len(cands)))

        # MAX over surviving candidate measurements; abandon THIS control action as soon as
        # its running worst case reaches the incumbent best (a larger max can't win the MIN).
        worst = -np.inf
        for k in survivors:
            counter[0] += 1
            val, _ = _min_node(new_robot, means[k], Sigma_next, depth - 1, R_fn, offs, F, Q,
                               n_meas, eps1, eps2, use_alpha, use_redundancy, counter)
            if val > worst:
                worst = val
            if use_alpha and worst >= best_val - eps1:
                break                       # this action can't beat the incumbent; stop its MAX
        if worst < best_val:
            best_val, best_a = worst, ai
    return best_val, best_a


def minimax_value_pruned(robot_xy, x_hat, Sigma, depth: int, R_fn: Callable,
                         offs: np.ndarray, F, Q, n_meas: int,
                         eps1: float = 0.0, eps2: float = 0.0,
                         use_alpha: bool = True, use_redundancy: bool = True
                         ) -> Tuple[float, int, int]:
    """Pruned minimax value + best first action + #measurement-nodes visited.

    With eps1 = eps2 = 0 this returns the SAME value and best action as
    tree.minimax_value (full enumeration) while visiting strictly fewer nodes whenever a
    cut fires. Signature mirrors tree.minimax_value with extra pruning controls."""
    counter = [0]
    val, ai = _min_node(np.asarray(robot_xy, float), np.asarray(x_hat, float),
                        np.asarray(Sigma, float), int(depth), R_fn, offs, F, Q, int(n_meas),
                        float(eps1), float(eps2), bool(use_alpha), bool(use_redundancy),
                        counter)
    return val, ai, counter[0]


def plan_single_target_pruned(robot_xy, x_hat, Sigma, R_fn: Callable, cfg, dt: float):
    """Pruned analogue of tree.plan_single_target. Returns
    (minimax_value, best_action_offset(2,), best_action_index, n_nodes_visited)."""
    offs = action_offsets(cfg, dt)
    F, Q = cv_matrices(dt, cfg.q)
    val, ai, n = minimax_value_pruned(
        np.asarray(robot_xy, float)[:2], np.asarray(x_hat, float), np.asarray(Sigma, float),
        int(cfg.horizon), R_fn, offs, F, Q, int(cfg.n_meas),
        eps1=float(cfg.eps1), eps2=float(cfg.eps2),
        use_alpha=bool(cfg.use_alpha_pruning), use_redundancy=bool(cfg.use_redundancy_pruning))
    return val, offs[ai], ai, n
