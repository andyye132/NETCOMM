"""Gaussian mutual-information objective for the greedy/RSP planner (Corah & Michael).

The receding-horizon sim objective (paper Eq 28) is a sum over the horizon of the
mutual information between each target's state and the observations up to that step:

    g(X) = sum_j  sum_{k=1..l}  I( X^t_{j,t+k} ; Y_{j,t+1:t+k}(X) | history )

For linear-Gaussian targets (CV motion, GM-PHD covariance prior P0) this is closed
form: at step k it is the entropy drop between the OBSERVATION-FREE prediction of the
target and the FILTERED estimate given the chosen observations:

    g_j = sum_k  1/2 ( logdet P_pred0[k]  -  logdet P_filt[k] )

where P_pred0 is P0 predicted forward with no measurements and P_filt also folds in
the selected sensors' position information M = H^T R^{-1} H at each step. This is the
information-filter recursion (same structure as evaluation/pcrlb.py), reused here as
the planner's value function. g_j is normalized (g_j(empty)=0), monotone, and
submodular in the set of sensors, so the greedy 1/2 bound and RSP's Theorem 1 hold.

Self-contained (imports nothing from netcomm): the per-sensor 2x2 position information
is supplied by the caller (the adapter injects sensors.measurement_information).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

# measurement matrix H = [I2 | 0] (cameras observe ground position of a CV target)
_H = jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])


def cv_matrices(dt: float, q: float):
    """Constant-velocity F and discrete white-noise-acceleration Q over [px,py,vx,vy]."""
    F = jnp.array([[1.0, 0.0, dt, 0.0],
                   [0.0, 1.0, 0.0, dt],
                   [0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0]])
    a, b, c = dt ** 4 / 4.0, dt ** 3 / 2.0, dt ** 2
    Q = q * jnp.array([[a, 0.0, b, 0.0],
                       [0.0, a, 0.0, b],
                       [b, 0.0, c, 0.0],
                       [0.0, b, 0.0, c]])
    return F, Q


def _slogdet(A):
    return jnp.linalg.slogdet(A)[1]


def target_horizon_mi(P0, F, Q, pos_info_steps):
    """Horizon-summed MI for ONE target (Eq 28, Gaussian closed form).

    P0             : (4,4) GM-PHD prior covariance of the target.
    pos_info_steps : (L, 2, 2) summed sensor position information (sum_s R_s^{-1}) at each
                     of the L horizon steps (zero where no selected sensor observes it).
    Returns the scalar information gain sum_k 1/2 (logdet P_pred0[k] - logdet P_filt[k]) >= 0.
    Exactly 0 when no sensor is selected (P_filt tracks P_pred0); the covariances are PD
    (GM-PHD has a covariance floor) so the log-dets need no SPD jitter.
    """
    P_pred0 = P0
    P_filt = P0
    total = 0.0
    L = pos_info_steps.shape[0]
    for k in range(L):
        P_pred0 = F @ P_pred0 @ F.T + Q                       # no-measurement prediction
        P_pred_f = F @ P_filt @ F.T + Q                       # filtered prediction
        P_filt = jnp.linalg.inv(jnp.linalg.inv(P_pred_f) + _H.T @ pos_info_steps[k] @ _H)
        total = total + 0.5 * (_slogdet(P_pred0) - _slogdet(P_filt))
    return total


# vmap over targets: priors (M,4,4), info (M,L,2,2) -> (M,) per-target MI
_targets_mi = jax.jit(jax.vmap(lambda P0, info, F, Q: target_horizon_mi(P0, F, Q, info),
                               in_axes=(0, 0, None, None)))


def _team_objective(priors, weights, info_steps, F, Q):
    """On-device team objective (jnp scalar, no host sync). See set_objective."""
    per_target = _targets_mi(priors, info_steps, F, Q)
    return jnp.sum(jnp.asarray(weights) * per_target)


def set_objective(priors, weights, info_steps, F, Q):
    """Team objective g(X) = sum_j w_j * (horizon MI of target j) for a given selection.

    priors     : (M,4,4) target prior covariances.
    weights    : (M,) PHD weights (>=0); pass ones for unweighted.
    info_steps : (M, L, 2, 2) summed selected-sensor position info per target per step.
    """
    return float(_team_objective(priors, weights, info_steps, F, Q))


# Batched objective over ONE drone's candidate actions, evaluated entirely on-device:
# given the committed base_info (M,L,2,2) and that drone's candidates (n_actions,M,L,2,2),
# vmap the team objective over actions -> (n_actions,) jnp array. The maximizer argmaxes
# this in a single device->host transfer instead of one float() sync per candidate.
batched_objective = jax.jit(
    jax.vmap(lambda base_info, cand_a, priors, weights, F, Q:
             _team_objective(priors, weights, base_info + cand_a, F, Q),
             in_axes=(None, 0, None, None, None, None)))
