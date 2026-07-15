"""Kalman steps + the state-dependent Riccati map rho (paper Eqs 2-7).

The minimax tree propagates a target's Gaussian (x_hat, Sigma) through control steps
(robot moves, covariance unchanged) and measurement steps. The measurement noise R is
STATE-DEPENDENT: it grows with the robot-target distance (paper Eqs 4-5; in the sim, the
camera R).

Eq 7 (the Kalman Riccati map rho the paper minimizes the trace of) is a PREDICTED-to-
PREDICTED covariance recursion:

    rho(Sigma) = F Sigma F^T
               - F Sigma H^T (H Sigma H^T + S_w)^{-1} H Sigma F^T
               + S_v

with F the target motion model C_t, S_w = R the (state-dependent) measurement noise, and
S_v = Q the process noise. Equivalently rho(Sigma) = F (I - K H) Sigma F^T + Q with the
Kalman gain K = Sigma H^T (H Sigma H^T + R)^{-1} formed on the PRIOR Sigma -- i.e. INFORMATION
UPDATE FIRST, then PREDICT. The leaf objective tr(Sigma_T) is the trace of this PREDICTED
covariance (position block). Note that rho is measurement-VALUE independent (it depends on z
only through R via the target-position estimate), which is exactly what makes the
algebraic-redundancy pruning and the vectorized enumeration valid.

This module provides BOTH a small numpy reference (used by the dynamic pruning tree and the
correctness tests) and genuine-JAX kernels (jnp + jit/vmap) used by the live vectorized
minimax in tree.py. The reusable JAX Riccati for the evaluation scorecard lives in
evaluation/pcrlb; this one is local to the minimax planner.
"""
from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

# H = [I2 | 0] selects position from the CV state [px, py, vx, vy]
H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
_HJ = jnp.asarray(H)


def cv_matrices(dt: float, q: float):
    """Constant-velocity F and DWNA process noise Q (the target motion C_t, Sigma_v)."""
    F = np.array([[1.0, 0.0, dt, 0.0],
                  [0.0, 1.0, 0.0, dt],
                  [0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0]])
    a, b, c = dt ** 4 / 4.0, dt ** 3 / 2.0, dt ** 2
    Q = q * np.array([[a, 0.0, b, 0.0],
                      [0.0, a, 0.0, b],
                      [b, 0.0, c, 0.0],
                      [0.0, b, 0.0, c]])
    return F, Q


def predict(x, Sigma, F, Q):
    return F @ x, F @ Sigma @ F.T + Q


def kalman_update(x_pred, Sigma_pred, z, R):
    """One Kalman measurement update with measurement z and (state-dependent) noise R.

    Used by the dynamic pruning tree to advance the (mean, covariance) of a node. Note the
    leaf-scoring covariance map the paper minimizes is riccati_step (Eq 7), not the bare
    posterior of this update -- see the module docstring.
    """
    S = H @ Sigma_pred @ H.T + R
    K = Sigma_pred @ H.T @ np.linalg.inv(S)
    x = x_pred + K @ (z - H @ x_pred)
    Sigma = (np.eye(Sigma_pred.shape[0]) - K @ H) @ Sigma_pred
    return x, Sigma


def riccati_step(Sigma, R, F, Q):
    """The state-dependent Riccati map rho of paper Eq 7 (predicted-to-predicted).

        rho(Sigma) = F Sigma F^T - F Sigma H^T (H Sigma H^T + R)^{-1} H Sigma F^T + Q

    i.e. INFORMATION-UPDATE the PRIOR Sigma (gain on Sigma, not on the predicted cov), THEN
    PREDICT. Returns the predicted covariance Sigma_{t+1}; the minimax leaf value is the trace
    of its position block. R = S_w (measurement noise), Q = S_v (process noise), F = C_t.
    """
    S = H @ Sigma @ H.T + R
    K = Sigma @ H.T @ np.linalg.inv(S)             # Kalman gain on the PRIOR covariance
    Sigma_upd = (np.eye(Sigma.shape[0]) - K @ H) @ Sigma
    return F @ Sigma_upd @ F.T + Q                  # then predict -> Eq-7 predicted covariance


def paper_noise(robot_xy, target_xy, delta1: float, delta2: float, B: float, C: float) -> np.ndarray:
    """Paper Eqs 4-5: isotropic measurement noise with variance delta1^2 + delta2^2 * d,
    where d = C*||X_r-X_o||/B for distance <= B, else d = C (saturation cap)."""
    dist = float(np.linalg.norm(np.asarray(robot_xy, float) - np.asarray(target_xy, float)))
    d = C * dist / B if dist <= B else C
    var = delta1 ** 2 + delta2 ** 2 * d
    return var * np.eye(2)


# --------------------------------------------------------------------------------------------
# Genuine-JAX kernels (jnp). These mirror the numpy reference above and are jit/vmap-composed
# by the vectorized minimax in tree.py. Use x64 in tight numeric tests for parity with numpy.
# --------------------------------------------------------------------------------------------

def riccati_step_j(Sigma, R, F, Q):
    """JAX Eq-7 Riccati map rho (predicted-to-predicted). See riccati_step."""
    S = _HJ @ Sigma @ _HJ.T + R
    K = Sigma @ _HJ.T @ jnp.linalg.inv(S)
    Sigma_upd = (jnp.eye(Sigma.shape[0]) - K @ _HJ) @ Sigma
    return F @ Sigma_upd @ F.T + Q


def predict_j(x, Sigma, F, Q):
    """JAX predict (mean + covariance). Covariance is measurement-independent."""
    return F @ x, F @ Sigma @ F.T + Q


def kalman_update_j(x_pred, Sigma_pred, z, R):
    """JAX Kalman update -- advances the MEAN (the covariance is value-independent)."""
    S = _HJ @ Sigma_pred @ _HJ.T + R
    K = Sigma_pred @ _HJ.T @ jnp.linalg.inv(S)
    x = x_pred + K @ (z - _HJ @ x_pred)
    Sigma = (jnp.eye(Sigma_pred.shape[0]) - K @ _HJ) @ Sigma_pred
    return x, Sigma
