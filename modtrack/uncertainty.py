"""First-order (Jacobian) uncertainty propagation and BEV measurement primitives.

A sensor front-end turns a raw measurement (range/bearing, pixel+depth, ...) into
a calibrated BEV position-covariance pair ``(z, R)``. The covariance is obtained
by pushing the input covariance through the measurement map's Jacobian:

    R_out = J Sigma_in J^T,    J = d f / d x.

This is ModTrack's analytical "covariance chain" — every R is traceable to a
physical sensor characteristic rather than guessed. Dimension-general.

The ``polar_to_cartesian`` helpers are the range/bearing front-end primitive: the
drone-sensor analogue of the paper's radar polar->Cartesian Jacobian (Eq. 1).
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def propagate_covariance(jacobian, cov_in) -> np.ndarray:
    """R_out = J Sigma_in J^T."""
    J = np.asarray(jacobian, dtype=float)
    S = np.asarray(cov_in, dtype=float)
    return J @ S @ J.T


def finite_diff_jacobian(fn: Callable, x, eps: float = 1e-6) -> np.ndarray:
    """Central-difference Jacobian of ``fn`` at ``x``.

    Used to cross-check analytic Jacobians in tests and as a fallback when no
    closed form is supplied. Shape is (m, n) for fn: R^n -> R^m.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    f0 = np.asarray(fn(x), dtype=float).reshape(-1)
    m = f0.shape[0]
    J = np.zeros((m, n))
    for j in range(n):
        dx = np.zeros(n)
        dx[j] = eps
        f_plus = np.asarray(fn(x + dx), dtype=float).reshape(-1)
        f_minus = np.asarray(fn(x - dx), dtype=float).reshape(-1)
        J[:, j] = (f_plus - f_minus) / (2.0 * eps)
    return J


def propagate_covariance_fn(fn: Callable, x, cov_in, eps: float = 1e-6) -> np.ndarray:
    """Propagate ``cov_in`` through nonlinear ``fn`` via its finite-diff Jacobian."""
    return propagate_covariance(finite_diff_jacobian(fn, x, eps), cov_in)


# ---------------------------------------------------------------------------
# Range/bearing front-end primitive (the drone-sensor analogue, 2D)
# ---------------------------------------------------------------------------

def polar_to_cartesian(z_polar) -> np.ndarray:
    """[r, theta] -> [x, y] with x = r cos(theta), y = r sin(theta)."""
    r, th = float(z_polar[0]), float(z_polar[1])
    return np.array([r * np.cos(th), r * np.sin(th)])


def polar_to_cartesian_jacobian(z_polar) -> np.ndarray:
    """Analytic Jacobian d[x, y] / d[r, theta] of ``polar_to_cartesian``."""
    r, th = float(z_polar[0]), float(z_polar[1])
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -r * s],
                     [s,  r * c]])
