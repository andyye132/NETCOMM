"""Linear-Gaussian motion models for the PCRLB recursion (CV and CA).

State ordering matches the GM-PHD filter: position block first, then velocity
(then acceleration for CA), each block spanning the spatial dimensions:
  CV:  x = [px, py, vx, vy]                      (4-dim in 2D)
  CA:  x = [px, py, vx, vy, ax, ay]             (6-dim in 2D)

F is the constant-velocity / constant-acceleration transition; Q is the standard
discrete white-noise-acceleration (CV) / white-noise-jerk (CA) process noise. The
measurement matrix H selects position (cameras observe ground position).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def cv_matrices(dt: float, q: float, dim: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Constant-velocity F and discrete white-noise-acceleration Q over [pos, vel]."""
    I, Z = np.eye(dim), np.zeros((dim, dim))
    F = np.block([[I, dt * I], [Z, I]])
    Q = q * np.block([[(dt ** 4 / 4) * I, (dt ** 3 / 2) * I],
                      [(dt ** 3 / 2) * I, (dt ** 2) * I]])
    return F, Q


def ca_matrices(dt: float, q: float, dim: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Constant-acceleration F and discrete white-noise-jerk Q over [pos, vel, acc]."""
    I, Z = np.eye(dim), np.zeros((dim, dim))
    F = np.block([[I, dt * I, 0.5 * dt ** 2 * I],
                  [Z, I,      dt * I],
                  [Z, Z,      I]])
    Q = q * np.block([
        [(dt ** 5 / 20) * I, (dt ** 4 / 8) * I, (dt ** 3 / 6) * I],
        [(dt ** 4 / 8) * I,  (dt ** 3 / 3) * I, (dt ** 2 / 2) * I],
        [(dt ** 3 / 6) * I,  (dt ** 2 / 2) * I, (dt) * I],
    ])
    return F, Q


def model_matrices(motion: str, dt: float, q: float, dim: int = 2):
    """Return (F, Q, H) for the chosen motion model; H = [I_dim | 0...] selects position."""
    if motion == "cv":
        F, Q = cv_matrices(dt, q, dim)
    elif motion == "ca":
        F, Q = ca_matrices(dt, q, dim)
    else:
        raise ValueError(f"unknown motion model {motion!r}; expected 'cv' or 'ca'")
    H = np.zeros((dim, F.shape[0]))
    H[:, :dim] = np.eye(dim)
    return F, Q, H
