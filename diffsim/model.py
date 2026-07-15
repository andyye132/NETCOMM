"""Differentiable motion + sensor model (pure JAX). The only relaxation vs the real sim is the
sensor's hard field-of-view gate, replaced here by a smooth sigmoid 'visibility' so that the
measurement information — and therefore the tracking uncertainty — is differentiable in the
drone position."""
from __future__ import annotations

import jax
import jax.numpy as jnp


def cv_matrices(dt, q):
    """Constant-velocity transition F, process noise Q (discrete white-noise accel), and the
    position measurement matrix H, for a 4D state [px, py, vx, vy]."""
    F = jnp.array([[1.0, 0.0, dt, 0.0],
                   [0.0, 1.0, 0.0, dt],
                   [0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0]])
    Q = q * jnp.array([[dt ** 3 / 3, 0.0, dt ** 2 / 2, 0.0],
                       [0.0, dt ** 3 / 3, 0.0, dt ** 2 / 2],
                       [dt ** 2 / 2, 0.0, dt, 0.0],
                       [0.0, dt ** 2 / 2, 0.0, dt]])
    H = jnp.array([[1.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 0.0]])
    return F, Q, H


def pair_info(drone_xy, target_xy, h, half_fov, focal_px, sigma_px, sigma_alt, p_detect, k_vis):
    """Soft per-(drone, target) 2x2 position Fisher information E[w * R^{-1}].

    Mirrors the real camera covariance (grows with slant range / off-nadir angle) but replaces
    the hard footprint gate with a smooth sigmoid visibility w = sigmoid(k_vis*(footprint - rho)),
    so info(drone_xy) is differentiable everywhere (moving a drone toward a target smoothly
    raises the information and lowers the posterior covariance)."""
    rx = target_xy[0] - drone_xy[0]
    ry = target_xy[1] - drone_xy[1]
    rho = jnp.hypot(rx, ry)
    foot = h * jnp.tan(half_fov)
    vis = jax.nn.sigmoid(k_vis * (foot - rho))                 # smooth footprint (no hard gate)
    s = jnp.hypot(rho, h)
    alpha = jnp.arctan2(rho, h)
    cos_a = jnp.maximum(jnp.cos(alpha), 1e-3)
    sigma_t = (s / focal_px) * sigma_px
    sigma_r = (s / focal_px) * sigma_px / cos_a + jnp.tan(alpha) * sigma_alt
    phi = jnp.arctan2(ry, rx)
    c, sn = jnp.cos(phi), jnp.sin(phi)
    Rot = jnp.array([[c, -sn], [sn, c]])
    Dinv = jnp.diag(jnp.array([1.0 / jnp.maximum(sigma_r ** 2, 1e-6),
                               1.0 / jnp.maximum(sigma_t ** 2, 1e-6)]))
    return vis * p_detect * (Rot @ Dinv @ Rot.T)               # (2, 2) position information


def target_info(drone_xy, target_xy, cfg):
    """Total soft position information about one target, summed over all drones (2x2)."""
    per = jax.vmap(lambda d: pair_info(d, target_xy, cfg.h, cfg.half_fov_rad, cfg.focal_px,
                                       cfg.sigma_px, cfg.sigma_alt, cfg.p_detect, cfg.k_vis))(drone_xy)
    return jnp.sum(per, axis=0)
