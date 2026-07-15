"""Discretize the candidate measurements for a measurement (max) tree level (paper Fig 1).

The robot cannot control the actual measurement, so the minimax tree branches over a finite
set of candidate measurements that discretize the Gaussian N(H x_pred, S), S = H Sigma H^T + R,
within ~3 sigma (99.7% of the mass). For a 2D measurement we return the predicted-measurement
mean plus (n_meas - 1) points spread on the `spread`-sigma covariance ellipse.
"""
from __future__ import annotations

from typing import List

import numpy as np

import jax.numpy as jnp

from .riccati import H, _HJ


def candidate_measurements(x_pred, Sigma_pred, R, n_meas: int, spread: float = 3.0) -> List[np.ndarray]:
    """n_meas candidate 2D measurements discretizing N(H x_pred, S) within `spread` sigma."""
    mean = H @ np.asarray(x_pred, float)
    S = H @ np.asarray(Sigma_pred, float) @ H.T + np.asarray(R, float)
    vals, vecs = np.linalg.eigh(S)
    axes = vecs * (spread * np.sqrt(np.maximum(vals, 0.0)))      # principal half-axes (2x2)
    cands = [mean.copy()]
    m = max(int(n_meas) - 1, 0)
    for i in range(m):
        ang = 2.0 * np.pi * i / m if m > 0 else 0.0
        cands.append(mean + axes @ np.array([np.cos(ang), np.sin(ang)]))
    return cands


def candidate_measurements_j(x_pred, Sigma_pred, R, n_meas: int, spread: float = 3.0):
    """JAX twin of candidate_measurements: (n_meas, 2) candidates discretizing N(H x_pred, S)
    within `spread` sigma. n_meas is a static int (the ring size is built with jnp.arange).
    Returns the predicted mean first, then (n_meas-1) points on the spread-sigma ellipse --
    matching candidate_measurements exactly so the JAX and numpy minimax agree under x64."""
    mean = _HJ @ x_pred
    S = _HJ @ Sigma_pred @ _HJ.T + R
    vals, vecs = jnp.linalg.eigh(S)
    axes = vecs * (spread * jnp.sqrt(jnp.maximum(vals, 0.0)))    # principal half-axes (2x2)
    m = max(int(n_meas) - 1, 0)
    if m == 0:
        return mean[None, :]
    angs = 2.0 * jnp.pi * jnp.arange(m) / m
    ring = (axes @ jnp.stack([jnp.cos(angs), jnp.sin(angs)], axis=0)).T   # (m, 2)
    return jnp.concatenate([mean[None, :], mean[None, :] + ring], axis=0)
