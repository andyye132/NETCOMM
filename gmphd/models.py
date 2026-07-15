"""Motion and measurement models for the 2D BEV GM-PHD filter.

State is m = [px, py, vx, vy]. Motion is constant-velocity with discrete
white-noise-acceleration (DWNA) process noise. Measurements are positions, so the
measurement matrix is H = [I2 | 0].

The public ``cv_model`` / ``measurement_matrix`` / ``mvn_pdf`` keep their original
NumPy signatures (and ``mvn_pdf`` still raises on a non-PSD covariance, as the
standalone density contract requires). The genuine-JAX numeric kernels used by
the filter (``mvn_pdf_jax`` and friends, defined in ``kernels.py``) instead apply
a symmetrize + eigenvalue-floor PSD safety so a numerically non-PSD innovation
covariance degrades gracefully rather than raising.

Self-contained (no netcomm / modtrack imports) by design.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def cv_model(dt: float, q: float) -> Tuple[np.ndarray, np.ndarray]:
    """Constant-velocity transition F and DWNA process-noise Q.

    F = [[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]]
    Q = q * [[dt^4/4 I, dt^3/2 I],[dt^3/2 I, dt^2 I]]   (block form over [pos, vel])
    """
    F = np.array([
        [1.0, 0.0, dt, 0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    a = dt ** 4 / 4.0
    b = dt ** 3 / 2.0
    c = dt ** 2
    Q = q * np.array([
        [a, 0.0, b, 0.0],
        [0.0, a, 0.0, b],
        [b, 0.0, c, 0.0],
        [0.0, b, 0.0, c],
    ])
    return F, Q


def measurement_matrix() -> np.ndarray:
    """H = [I2 | 0] mapping state [px,py,vx,vy] to observed position [px,py]."""
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ])


def mvn_pdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """Multivariate-normal density N(x; mean, cov), evaluated stably via slogdet+solve.

    Raises ``ValueError`` if ``cov`` is not positive definite — the standalone
    density contract. (The filter's internal JAX kernel uses a PSD-safe variant.)
    """
    d = np.asarray(x, dtype=float) - np.asarray(mean, dtype=float)
    k = d.shape[0]
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("mvn_pdf: covariance matrix is not positive definite")
    sol = np.linalg.solve(cov, d)
    return float(np.exp(-0.5 * (d @ sol) - 0.5 * (k * np.log(2.0 * np.pi) + logdet)))
