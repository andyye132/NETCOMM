"""The single-robot minimax tree (Zhang & Tokekar, Algorithm 1).

min_u max_z tr(Sigma_T): the robot (min) chooses controls, "nature" (max) chooses the
worst-case measurement. Alternating control levels (one branch per action, MIN) and
measurement levels (one branch per candidate measurement, MAX). The covariance is advanced
by the paper's Eq-7 Riccati map rho (PREDICTED-to-PREDICTED: information-update the prior
THEN predict; see riccati.riccati_step), and the leaf value is the trace of that PREDICTED
covariance position block. The covariance map is measurement-VALUE independent (Eq 7 depends
on z only through the state-dependent noise R via the target-position estimate), so the MAX
over measurements bites only through the future R; the per-candidate Kalman update advances
the MEAN that determines that R.

Two equivalent implementations live here:

  * `minimax_value` -- the exact recursive full-enumeration tree (numpy). The
    algebraic-redundancy / alpha pruning in pruning.py builds on it and returns the same
    optimum (the pruning tests pin this).
  * `minimax_value_vectorized` -- a genuine-JAX (jnp + jit + vmap + lax.scan) EXACT
    full-enumeration minimax for the fixed small horizon and fixed action set. Pruning was
    only a speed optimization proven equal to full enumeration, so this vectorized exact
    enumeration is faithful AND jit-able; it is the LIVE path used by greedy_assignment.
"""
from __future__ import annotations

from functools import partial
from typing import Callable, Tuple

import numpy as np

import jax
import jax.numpy as jnp

from .riccati import (
    cv_matrices, predict, kalman_update, riccati_step,
    predict_j, kalman_update_j, riccati_step_j, H, _HJ,
)
from .sampler import candidate_measurements, candidate_measurements_j


def action_offsets(cfg, dt: float) -> np.ndarray:
    """Control set U: optional STAY + K compass directions at reach v_max*dt (xy moves)."""
    r = cfg.v_max * dt
    offs = [np.zeros(2)] if cfg.include_stay else []
    for k in range(cfg.n_directions):
        ang = 2.0 * np.pi * k / cfg.n_directions
        offs.append(r * np.array([np.cos(ang), np.sin(ang)]))
    return np.array(offs, dtype=float)


def minimax_value(robot_xy, x_hat, Sigma, depth: int, R_fn: Callable,
                  offs: np.ndarray, F, Q, n_meas: int) -> Tuple[float, int]:
    """Exact minimax value + best first action (recursive numpy reference).

    depth = remaining (control, measurement) pairs. R_fn(robot_xy(2,), target_xy(2,)) -> 2x2
    state-dependent measurement noise R.

    At a control (MIN) node the robot moves and the target is predicted; the noise R is
    formed at the predicted target position. The covariance is advanced by the Eq-7 Riccati
    map rho (riccati_step) -- the SAME for every candidate measurement (value-independent).
    Each candidate measurement advances only the MEAN (which sets the deeper R). The leaf
    value (next node after the deepest measurement) is tr(rho(Sigma)[:2,:2]).
    """
    robot_xy = np.asarray(robot_xy, float)
    x_hat = np.asarray(x_hat, float)
    Sigma = np.asarray(Sigma, float)
    best_val, best_a = np.inf, 0
    for ai in range(offs.shape[0]):                             # MIN over the FIRST action
        v = _mm_action_value(robot_xy, x_hat, Sigma, int(depth), R_fn, offs, F, Q,
                             int(n_meas), ai)
        if v < best_val:
            best_val, best_a = v, ai
    return best_val, best_a


def _mm_action_value(robot_xy, x_hat, Sigma, depth, R_fn, offs, F, Q, n_meas, ai):
    """Worst-case (MAX over measurements / future MIN over controls) value of taking action
    `ai` at this control node. The covariance advances by the Eq-7 map rho; the leaf (deepest
    level) value is tr(rho(Sigma)[:2,:2]); each candidate measurement advances only the MEAN."""
    new_robot = robot_xy + offs[ai]                             # MIN: robot picks action
    x_pred, Sigma_pred = predict(x_hat, Sigma, F, Q)
    R = R_fn(new_robot, x_pred[:2])                             # state-dependent noise
    Sigma_next = riccati_step(Sigma, R, F, Q)                   # Eq-7 predicted covariance
    if depth <= 1:
        return float(np.trace(Sigma_next[:2, :2]))             # leaf = tr of predicted cov
    worst = -np.inf                                            # MAX: nature picks measurement
    for z in candidate_measurements(x_pred, Sigma_pred, R, n_meas):
        x_upd, _ = kalman_update(x_pred, Sigma_pred, z, R)     # advance MEAN only
        best_child = np.inf                                    # MIN over next control level
        for aj in range(offs.shape[0]):
            v = _mm_action_value(new_robot, x_upd, Sigma_next, depth - 1, R_fn, offs, F, Q,
                                 n_meas, aj)
            if v < best_child:
                best_child = v
        if best_child > worst:
            worst = best_child
    return worst


def plan_single_target(robot_xy, x_hat, Sigma, R_fn: Callable, cfg, dt: float):
    """Plan one robot's best first move to minimize the worst-case trace of one target.

    Returns (minimax_value, best_action_offset(2,), best_action_index).
    """
    offs = action_offsets(cfg, dt)
    F, Q = cv_matrices(dt, cfg.q)
    val, ai = minimax_value(np.asarray(robot_xy, float)[:2], np.asarray(x_hat, float),
                            np.asarray(Sigma, float), int(cfg.horizon), R_fn, offs, F, Q,
                            int(cfg.n_meas))
    return val, offs[ai], ai


# --------------------------------------------------------------------------------------------
# Genuine-JAX vectorized exact full-enumeration minimax (the LIVE path).
#
# For a fixed horizon H, fixed action set (n_a actions) and fixed n_meas, the minimax value is
# computed by enumerating every (control, measurement) trajectory with lax.scan over the
# horizon and vmap over the branch factor, then reducing the resulting leaf-trace tensor by
# alternating jnp.min (MIN/control levels) and jnp.max (MAX/measurement levels). The covariance
# recursion is Eq-7 (value-independent), so the tensor of leaf traces is exact. The whole thing
# is jit-compiled (static horizon/n_a/n_meas); only the multi-robot greedy outer loop is Python.
#
# State-dependent R is supplied as a jit-traceable callable R_fn_j(robot_xy(2,), tgt_xy(2,))
# -> (2,2); the camera/paper noise models are elementwise jnp and trace fine.
# --------------------------------------------------------------------------------------------

def _expand_level(state, offs, R_fn_j, F, Q, n_meas, spread):
    """One control+measurement level. Given a batch of nodes (robot_xy, x_hat, Sigma) flattened
    over the current frontier, expand each by all actions then all candidate measurements.

    Returns:
      leaf_trace : (..., n_a) tr(rho(Sigma)[:2,:2]) for each action at THIS node (the value if
                   this were the deepest level), and
      children   : new (robot_xy, x_hat, Sigma) batched over (..., n_a, n_meas) for the next
                   level (covariance is the Eq-7 map, shared across the n_meas measurements;
                   the mean differs per measurement).
    """
    robot_xy, x_hat, Sigma = state                              # (...,2),(...,4),(...,4,4)

    def per_action(robot_xy, x_hat, Sigma, off):
        new_robot = robot_xy + off
        x_pred, Sigma_pred = predict_j(x_hat, Sigma, F, Q)
        R = R_fn_j(new_robot, x_pred[:2])
        Sigma_next = riccati_step_j(Sigma, R, F, Q)             # Eq-7 (value-independent)
        leaf_trace = jnp.trace(Sigma_next[:2, :2])
        cands = candidate_measurements_j(x_pred, Sigma_pred, R, n_meas, spread)   # (n_meas,2)
        # advance MEAN per candidate measurement (covariance is the shared Sigma_next)
        x_upd = jax.vmap(lambda z: kalman_update_j(x_pred, Sigma_pred, z, R)[0])(cands)  # (n_meas,4)
        robot_b = jnp.broadcast_to(new_robot, (n_meas, 2))
        Sigma_b = jnp.broadcast_to(Sigma_next, (n_meas,) + Sigma_next.shape)
        return leaf_trace, robot_b, x_upd, Sigma_b

    # vmap over actions (innermost), then over any leading frontier dims.
    f = per_action
    f = jax.vmap(f, in_axes=(None, None, None, 0))              # over actions
    # leading frontier dims:
    lead = robot_xy.ndim - 1
    for _ in range(lead):
        f = jax.vmap(f, in_axes=(0, 0, 0, None))
    leaf_trace, robot_b, x_upd, Sigma_b = f(robot_xy, x_hat, Sigma, offs)
    return leaf_trace, (robot_b, x_upd, Sigma_b)


@partial(jax.jit, static_argnums=(3, 4, 5))
def _minimax_value_core(robot_xy, x_hat, Sigma, depth, n_a, n_meas,
                        offs, Fm, Qm, R_params, spread):
    """Jitted exact vectorized minimax. depth (horizon), n_a, n_meas are static.

    R_params packs the paper/camera-style state-dependent noise so R_fn_j is jit-clean:
    R_params = (delta1, delta2, B, C) -> isotropic Eqs 4-5 noise. Returns scalar minimax value
    AND a per-first-action value vector (n_a,) for argmin selection in Python.
    """
    d1, d2, B, C = R_params

    def R_fn_j(robot_xy, tgt_xy):
        dist = jnp.linalg.norm(robot_xy - tgt_xy)
        dd = jnp.where(dist <= B, C * dist / B, C)
        var = d1 ** 2 + d2 ** 2 * dd
        return var * jnp.eye(2)

    # Build the leaf-trace tensor over the horizon by repeated expansion.
    state = (robot_xy, x_hat, Sigma)
    leaf_tensors = []                                           # leaf-trace at each depth's actions
    for lvl in range(depth):
        leaf_trace, children = _expand_level(state, offs, R_fn_j, Fm, Qm, n_meas, spread)
        leaf_tensors.append(leaf_trace)
        state = children                                        # frontier grows by (n_a, n_meas)

    # The value tensor is the deepest level's leaf traces, shaped
    #   (n_a, n_meas, n_a, n_meas, ..., n_a)   [depth control levels, depth-1 meas levels].
    value = leaf_tensors[-1]                                    # (..., n_a) at the deepest control
    # Reduce from the leaves up: alternating max (measurement) then min (control).
    # Current trailing axis is a control (MIN) level -> min. Then a measurement (MAX) -> max.
    # Repeat until only the first control level's n_a axis remains.
    reduce_min = True
    while value.ndim > 1:
        value = jnp.min(value, axis=-1) if reduce_min else jnp.max(value, axis=-1)
        reduce_min = not reduce_min
    # value now has shape (n_a,): the worst-case trace achievable from each FIRST action.
    first_action_values = value
    return jnp.min(first_action_values), first_action_values


def minimax_value_vectorized(robot_xy, x_hat, Sigma, depth, R_params, offs, F, Q,
                             n_meas, spread: float = 3.0):
    """Genuine-JAX exact minimax. Returns (value: float, best_action_index: int).

    robot_xy(2,), x_hat(4,), Sigma(4,4) host arrays; depth/n_meas ints; offs(n_a,2);
    F,Q(4,4); R_params=(delta1,delta2,B,C) for the Eqs 4-5 isotropic state-dependent noise.
    """
    offs_j = jnp.asarray(offs)
    n_a = int(offs.shape[0])
    val, fav = _minimax_value_core(
        jnp.asarray(robot_xy, float), jnp.asarray(x_hat, float), jnp.asarray(Sigma, float),
        int(depth), n_a, int(n_meas), offs_j, jnp.asarray(F), jnp.asarray(Q),
        tuple(float(p) for p in R_params), float(spread))
    return float(val), int(jnp.argmin(fav))
