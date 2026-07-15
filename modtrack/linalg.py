"""Small linear-algebra helpers shared across ModTrack stages.

Kept numpy-only so the analytical core and its unit tests run fast and without
any JAX/CUDA dependency. Everything is dimension-general (d = 2 or 3).
"""
from __future__ import annotations

import numpy as np


def is_symmetric(A, atol: float = 1e-8) -> bool:
    A = np.asarray(A, dtype=float)
    return A.ndim == 2 and A.shape[0] == A.shape[1] and bool(np.allclose(A, A.T, atol=atol))


def is_psd(A, tol: float = 1e-10) -> bool:
    # why: symmetric with smallest eigenvalue >= -tol. eigvalsh uses the
    # symmetric solver; we symmetrize first to ignore tiny asymmetry.
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return False
    w = np.linalg.eigvalsh(0.5 * (A + A.T))
    return bool(w.min() >= -tol)


def symmetrize(A) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    return 0.5 * (A + A.T)


def inv(A, jitter: float = 0.0) -> np.ndarray:
    # why: optional diagonal jitter to regularize near-singular covariances.
    A = np.asarray(A, dtype=float)
    if jitter:
        A = A + jitter * np.eye(A.shape[0])
    return np.linalg.inv(A)


def mahalanobis_sq(dz, cov) -> float:
    # why: d^2 = dz^T cov^{-1} dz via a linear solve (no explicit inverse) for
    # numerical stability. This is the chi^2 statistic used in gating/clustering.
    dz = np.asarray(dz, dtype=float)
    cov = np.asarray(cov, dtype=float)
    return float(dz @ np.linalg.solve(cov, dz))
