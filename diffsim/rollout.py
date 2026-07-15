"""The differentiable episode rollout and a gradient-descent optimizer over drone positions.

``episode_uncertainty`` is the jax.grad-able scalar objective: a smooth, PCRLB-style sum of
posterior position-covariance trace over a whole episode, as a function of where the drones are.
Lower = better tracking. ``jax.grad`` of it w.r.t. the drone positions is the direction to move
the drones to track better — demonstrating that the sim is differentiable end-to-end."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from .model import cv_matrices, target_info


def episode_uncertainty(drone_xy, target_traj, cfg):
    """Total tracking uncertainty for static drones observing known moving targets.

    drone_xy    : (N, 2) drone ground positions (altitude cfg.h). THIS is what we differentiate.
    target_traj : (T, M, 2) true target positions over the horizon.
    returns     : scalar loss = sum over time and targets of trace(posterior position cov).

    The belief is one differentiable Kalman/Gaussian per target (fixed M, no data association,
    no argmax), updated in information form with the soft drone measurement information. Because
    the covariance recursion depends only on geometry (not on stochastic measurements), the loss
    is deterministic and smooth in drone_xy."""
    F, Q, H = cv_matrices(cfg.dt, cfg.q)
    M = target_traj.shape[1]
    eye4 = jnp.eye(4)
    Sigma0 = jnp.broadcast_to(
        jnp.diag(jnp.array([cfg.p0_pos, cfg.p0_pos, cfg.p0_vel, cfg.p0_vel])), (M, 4, 4))

    def step(Sigma, t_xy):                                      # Sigma (M,4,4), t_xy (M,2)
        Sigma_pred = jax.vmap(lambda S: F @ S @ F.T + Q)(Sigma)
        info = jax.vmap(lambda tt: target_info(drone_xy, tt, cfg))(t_xy)        # (M,2,2)

        def update(Sp, Mi):
            Jp = jnp.linalg.inv(Sp + 1e-6 * eye4)              # prior information
            return jnp.linalg.inv(Jp + H.T @ Mi @ H + 1e-9 * eye4)              # + measurement info

        Sigma_post = jax.vmap(update)(Sigma_pred, info)
        loss_t = jnp.sum(jax.vmap(lambda S: jnp.trace(S[:2, :2]))(Sigma_post))
        return Sigma_post, loss_t

    _, losses = lax.scan(step, Sigma0, target_traj)
    return jnp.sum(losses)


def optimize_drone_positions(drone_xy0, target_traj, cfg, steps=80, step_m=4.0):
    """Gradient DESCENT on drone positions to minimize episode_uncertainty. Uses per-drone
    normalized steps of ``step_m`` metres (robust to gradient scale) and clips to the area.
    Returns (final_xy (N,2), loss_history list of length steps+1)."""
    xmn, xmx, ymn, ymx = cfg.area
    value_and_grad = jax.jit(jax.value_and_grad(lambda d: episode_uncertainty(d, target_traj, cfg)))
    d = jnp.asarray(drone_xy0, dtype=float)
    history = []
    for _ in range(steps):
        L, g = value_and_grad(d)
        history.append(float(L))
        g_unit = g / (jnp.linalg.norm(g, axis=1, keepdims=True) + 1e-9)
        d = d - step_m * g_unit
        d = jnp.stack([jnp.clip(d[:, 0], xmn, xmx), jnp.clip(d[:, 1], ymn, ymx)], axis=1)
    history.append(float(episode_uncertainty(d, target_traj, cfg)))
    return d, history
