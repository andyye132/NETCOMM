"""HMM motion models for the GM-PHD tracker (ModTrack Stage 5, paper Section 3.5.3).

Three motion modes share a linear state-transition  x_{k+1} = F_s x_k + w_k,
w_k ~ N(0, Q_s), over the BEV state  m = [pos, vel]  (4-dim in 2D, 6-dim in 3D,
ordered [pos(0..d-1), vel(0..d-1)]):

    STATIONARY        : constant-velocity structure with velocity damped toward 0
    CONSTANT_VELOCITY : the standard CV transition  F = [[I, dt I], [0, I]]
    MANEUVERING       : same CV transition with elevated process noise (Q3 >> Q2)

Process noise uses the discrete white-noise-acceleration (DWNA) form
    Q = q * [[dt^4/4 I, dt^3/2 I], [dt^3/2 I, dt^2 I]].

Mode switching follows a class-specific Markov chain Pi with per-mode
self-transition probabilities (paper Table 7: 0.75 / 0.94 / 0.10).

Modes are 0-indexed here (STATIONARY=0, CONSTANT_VELOCITY=1, MANEUVERING=2),
mapping to the paper's s in {1, 2, 3}. Everything is dimension-general.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

STATIONARY = 0
CONSTANT_VELOCITY = 1
MANEUVERING = 2
MODES = (STATIONARY, CONSTANT_VELOCITY, MANEUVERING)


# ---------------------------------------------------------------------------
# State-transition matrices  F_s
# ---------------------------------------------------------------------------

def cv_transition_matrix(dt: float, dim: int = 2) -> np.ndarray:
    """Constant-velocity transition F = [[I, dt I], [0, I]] over [pos, vel]."""
    I = np.eye(dim)
    Z = np.zeros((dim, dim))
    return np.block([[I, dt * I], [Z, I]])


def stationary_transition_matrix(dt: float, dim: int = 2, damping: float = 0.0) -> np.ndarray:
    """Stationary transition: CV structure with the velocity block scaled by
    ``damping`` in [0, 1] (0 = velocity fully damped to zero each step)."""
    I = np.eye(dim)
    Z = np.zeros((dim, dim))
    return np.block([[I, dt * I], [Z, damping * I]])


def transition_matrix(mode: int, dt: float, dim: int = 2,
                      stationary_damping: float = 0.0) -> np.ndarray:
    if mode in (CONSTANT_VELOCITY, MANEUVERING):
        return cv_transition_matrix(dt, dim)
    if mode == STATIONARY:
        return stationary_transition_matrix(dt, dim, stationary_damping)
    raise ValueError(f"unknown motion mode: {mode!r}")


# ---------------------------------------------------------------------------
# Process noise  Q_s
# ---------------------------------------------------------------------------

def cv_process_noise(dt: float, q: float, dim: int = 2) -> np.ndarray:
    """Discrete white-noise-acceleration covariance for a CV model."""
    I = np.eye(dim)
    Z = np.zeros((dim, dim))
    q_pp = (dt ** 4 / 4.0) * I
    q_pv = (dt ** 3 / 2.0) * I
    q_vv = (dt ** 2) * I
    return q * np.block([[q_pp, q_pv], [q_pv, q_vv]])


def process_noise(mode: int, dt: float, dim: int = 2, q_cv: float = 1.0,
                  maneuver_scale: float = 100.0, stationary_scale: float = 0.01) -> np.ndarray:
    """Mode-dependent process noise. MANEUVERING inflates CV noise by
    ``maneuver_scale`` (Q3 >> Q2); STATIONARY shrinks it by ``stationary_scale``."""
    base = cv_process_noise(dt, q_cv, dim)
    if mode == CONSTANT_VELOCITY:
        return base
    if mode == MANEUVERING:
        return maneuver_scale * base
    if mode == STATIONARY:
        return stationary_scale * base
    raise ValueError(f"unknown motion mode: {mode!r}")


# ---------------------------------------------------------------------------
# Mode-switching Markov chain  Pi
# ---------------------------------------------------------------------------

def mode_transition_matrix(pi_stay: Tuple[float, float, float] = (0.75, 0.94, 0.10)) -> np.ndarray:
    """3x3 Markov chain; row i = P(next mode | current mode i). Off-diagonal mass
    (1 - pi_stay[i]) is split uniformly across the other two modes."""
    P = np.zeros((3, 3))
    for i in range(3):
        P[i, i] = pi_stay[i]
        off = (1.0 - pi_stay[i]) / 2.0
        for j in range(3):
            if j != i:
                P[i, j] = off
    return P


# ---------------------------------------------------------------------------
# Kalman predict primitive
# ---------------------------------------------------------------------------

def predict_state(m: np.ndarray, P: np.ndarray, F: np.ndarray, Q: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """One linear predict step: m' = F m,  P' = F P F^T + Q."""
    m = np.asarray(m, dtype=float)
    P = np.asarray(P, dtype=float)
    m_pred = F @ m
    P_pred = F @ P @ F.T + Q
    return m_pred, P_pred
