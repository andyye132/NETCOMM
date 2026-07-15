"""Sequential-greedy multi-robot assignment over the per-robot minimax tree.

ATTRIBUTION (verified against both PDFs, 2026-07-12): Zhang & Tokekar 2016 is a
SINGLE-robot, single-target method; multi-robot/multi-target is explicitly deferred to
future work (Sec. VII), pointing at Tokekar, Isler & Franchi, IROS 2014 (their ref [17]).
That paper's actual Algorithm 1 is a JOINT greedy: each iteration searches ALL unassigned
robots x ALL their candidate trajectories and commits the single best (robot, trajectory)
pair — robot order is an output of the marginal-gain ranking, and the (1/2 - eps) MGC
guarantee is stated for that joint rule. What follows is neither paper's algorithm: a
FIXED-ORDER sequential greedy (each robot in turn picks its best (target, first-action)
given prior commitments) wrapped around the 2016 paper's per-robot minimax tree. Fixed
order costs O(R) expensive tree plans per robot instead of the joint rule's O(R^2); since
the minimax-trace objective is not submodular, neither variant carries a guarantee here.

The joint multi-robot minimax (all robots' controls vs. all measurements) is intractable,
so robots are processed one at a time; each robot picks the (target, first-action) that
most reduces that target's worst-case trace GIVEN the commitments of the robots already
placed. A robot already assigned to a target conditions that target's covariance (one
measurement update at the assigning robot's planned pose), so the next robot to consider
that target plans against the tighter posterior. One target may receive several robots.

This is a HEURISTIC. The (1 - 1/e) / (1/2) submodular-greedy guarantees do NOT apply: the
minimax-trace objective is not submodular in the robot set, and the per-robot trees are
coupled only through the shared posterior we thread between them. We keep it simple and
correct; the guarantee is not claimed.

Each robot still plans with the full per-robot minimax tree (pruned or exact), so the
within-robot worst-case-measurement reasoning is preserved; only the across-robot coupling
is greedy.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .riccati import cv_matrices, predict, kalman_update, riccati_step, H
from .tree import action_offsets, minimax_value, minimax_value_vectorized
from .pruning import minimax_value_pruned


def _condition_on_robot(x_hat, Sigma, robot_xy, action_off, R_fn, F, Q):
    """Condition this target on one expected measurement from a robot committed to it at its
    planned first pose, tightening the prior the NEXT robot plans against. We apply one
    Kalman INFORMATION update at the robot's planned pose (no time advance): the covariance
    becomes (I - K H) Sigma -- exactly the update half of the Eq-7 map riccati_step, and PSD-
    tighter than Sigma -- and the mean is advanced through the nominal (predicted) measurement.
    Keeping the conditioning at the SAME time step makes the next robot's planned value no
    larger than the previous robot's (the shared-target sanity the assignment test pins).
    Returns the conditioned (x, Sigma) the next robot should plan against."""
    new_robot = np.asarray(robot_xy, float)[:2] + np.asarray(action_off, float)
    x_pred, _ = predict(x_hat, Sigma, F, Q)
    R = R_fn(new_robot, x_pred[:2])             # noise at the predicted target position
    # information update on the PRIOR Sigma (gain on Sigma): the update half of Eq-7.
    S = H @ Sigma @ H.T + R
    K = Sigma @ H.T @ np.linalg.inv(S)
    Sigma_upd = (np.eye(Sigma.shape[0]) - K @ H) @ Sigma
    z = x_pred[:2]                              # nominal (predicted) measurement
    x_upd = x_hat + K @ (z - H @ x_hat)         # mean update consistent with the prior-gain K
    return x_upd, Sigma_upd


def greedy_assignment(robots_xy, means, covs, R_fns, cfg, dt: float,
                      use_pruning: bool = True,
                      weights: Optional[Sequence[float]] = None,
                      r_params: Optional[Sequence[Sequence[float]]] = None
                      ) -> Tuple[np.ndarray, np.ndarray, List[int], List[float]]:
    """Assign R robots to M targets and plan each robot's first move (sequential greedy).

    robots_xy : (R, 2) robot xy positions.
    means     : (M, 4) per-target CV-state priors [px, py, vx, vy].
    covs      : (M, 4, 4) per-target priors.
    R_fns     : list of R callables R_fn(robot_xy(2,), target_xy(2,)) -> 2x2 noise; one per
                robot (each robot may have its own altitude/sensor, hence its own R model).
    cfg       : MinimaxConfig (horizon, action set, n_meas, pruning flags/eps).
    dt        : control step.
    weights   : OPTIONAL (M,) PHD weights (default None = current unweighted behaviour). When
                given AND cfg.weight_by_phd is True, the per-robot assignment maximizes the
                PHD-WEIGHTED marginal reward w_m * (baseline_m - value_m), i.e. the robot's
                contribution to the multi-target leaf objective sum_j w_j tr(Sigma_j); a robot
                prefers to tighten higher-weight (more-confident/important) targets. The
                returned `values` remain the unweighted per-robot minimax traces.
    r_params  : OPTIONAL list of R tuples (delta1, delta2, B, C). When provided, each robot's
                per-target minimax value is computed by the GENUINE-JAX jitted vectorized
                exact full-enumeration minimax (tree.minimax_value_vectorized) under the paper
                Eqs 4-5 isotropic state-dependent noise, instead of the numpy planner. The
                jitted kernel is the per-robot value; the greedy outer loop stays Python. The
                Eq-7 objective is identical either way (default None -> numpy planner via
                R_fns, used by the camera-noise adapter whose R_fn is not jit-traceable).

    Returns
    -------
    new_xy      : (R, 2) each robot's planned first position (robot_xy + best action offset).
    offsets     : (R, 2) the chosen first-action displacement per robot.
    assignment  : list of length R, the target index each robot was assigned to (-1 if none).
    values      : list of length R, the per-robot minimax value at its chosen (target, action).
    """
    robots_xy = np.asarray(robots_xy, float).reshape(-1, 2)
    means = np.asarray(means, float).reshape(-1, 4)
    covs = np.asarray(covs, float).reshape(-1, 4, 4)
    R = robots_xy.shape[0]
    M = means.shape[0]

    offs = action_offsets(cfg, dt)
    F, Q = cv_matrices(dt, cfg.q)

    new_xy = robots_xy.copy()
    offsets = np.zeros((R, 2))
    assignment: List[int] = [-1] * R
    values: List[float] = [float("nan")] * R

    if M == 0:
        return new_xy, offsets, assignment, values

    # PHD-weighting: honored only when weights are supplied AND the config flag is set.
    use_w = bool(cfg.weight_by_phd) and weights is not None
    w = np.asarray(weights, float).reshape(-1) if use_w else np.ones(M)
    if use_w and w.shape[0] != M:
        raise ValueError(f"weights length {w.shape[0]} != number of targets {M}")

    # live per-target posteriors, tightened as robots are committed to them
    post_x = [means[m].copy() for m in range(M)]
    post_S = [covs[m].copy() for m in range(M)]

    def _plan(robot_xy, x_hat, Sigma, R_fn, rp):
        """Per-robot minimax. Returns (value, action_offset, action_index).

        rp = (delta1,delta2,B,C) -> genuine-JAX jitted vectorized exact minimax; else the
        numpy planner (pruned or exact). All paths score the SAME Eq-7 objective."""
        if rp is not None:
            val, ai = minimax_value_vectorized(
                robot_xy, x_hat, Sigma, int(cfg.horizon), rp, offs, F, Q, int(cfg.n_meas))
        elif use_pruning:
            val, ai, _n = minimax_value_pruned(
                robot_xy, x_hat, Sigma, int(cfg.horizon), R_fn, offs, F, Q, int(cfg.n_meas),
                eps1=float(cfg.eps1), eps2=float(cfg.eps2),
                use_alpha=bool(cfg.use_alpha_pruning),
                use_redundancy=bool(cfg.use_redundancy_pruning))
        else:
            val, ai = minimax_value(robot_xy, x_hat, Sigma, int(cfg.horizon), R_fn, offs,
                                    F, Q, int(cfg.n_meas))
        return val, offs[ai], ai

    for r in range(R):
        R_fn = R_fns[r]
        rp = r_params[r] if r_params is not None else None
        # baseline worst-case trace per target (no further robot) -> reward = baseline - value;
        # PHD-weighted by w_m when enabled (the marginal gain to sum_j w_j tr(Sigma_j)).
        best_target, best_off, best_val, best_reward = -1, np.zeros(2), float("inf"), -np.inf
        for m in range(M):
            x_hat, Sigma = post_x[m], post_S[m]
            baseline = float(np.trace(Sigma[:2, :2]))
            val, off, _ai = _plan(robots_xy[r], x_hat, Sigma, R_fn, rp)
            reward = float(w[m]) * (baseline - val)     # PHD-weighted tightening of target m
            if reward > best_reward + 1e-12:
                best_target, best_off, best_val, best_reward = m, off, val, reward
        assignment[r] = best_target
        offsets[r] = best_off
        new_xy[r] = robots_xy[r] + best_off
        values[r] = best_val
        # commit: condition the chosen target's posterior so the next robot plans tighter
        if best_target >= 0:
            post_x[best_target], post_S[best_target] = _condition_on_robot(
                post_x[best_target], post_S[best_target], robots_xy[r], best_off, R_fn, F, Q)

    return new_xy, offsets, assignment, values
