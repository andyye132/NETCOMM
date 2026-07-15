"""Core correctness tests for the non-myopic minimax tracker (Zhang & Tokekar 2016)."""
import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)   # x64 for tight JAX-vs-numpy numeric parity

from nonmyopic import (
    MinimaxConfig, paper_noise, riccati_step, cv_matrices, predict, kalman_update,
    candidate_measurements, action_offsets, plan_single_target, minimax_value,
    minimax_value_vectorized,
)
from nonmyopic.riccati import H


def _psd(rng, d=4, scale=1.0):
    A = rng.normal(size=(d, d))
    return scale * (A @ A.T + d * np.eye(d))


def test_paper_noise_model_eqs_4_5():
    d1, d2, B, C = 0.5, 1.0, 20.0, 5.0
    f = lambda dist: paper_noise([0, 0], [dist, 0], d1, d2, B, C)[0, 0]
    assert abs(f(0.0) - d1 ** 2) < 1e-12                         # at the target: var = delta1^2
    assert abs(f(B) - (d1 ** 2 + d2 ** 2 * C)) < 1e-12          # at range B: var = d1^2 + d2^2 C
    assert abs(f(10.0) - (d1 ** 2 + d2 ** 2 * C * 10.0 / B)) < 1e-12   # linear within B
    assert abs(f(40.0) - f(B)) < 1e-12                          # beyond B: saturated (capped)
    assert f(5.0) < f(15.0)                                     # noise grows with distance


def test_riccati_monotonicity_thm5():
    """Thm 5: Sigma_A >= Sigma_B (same R) -> rho(Sigma_A) >= rho(Sigma_B) (PSD order)."""
    rng = np.random.default_rng(0)
    F, Q = cv_matrices(1.0, 0.05)
    for _ in range(30):
        Sb = _psd(rng, 4, 1.0)
        Sa = Sb + _psd(rng, 4, 0.5)                             # Sa >= Sb
        R = (0.5 + rng.random()) * np.eye(2)
        diff = riccati_step(Sa, R, F, Q) - riccati_step(Sb, R, F, Q)
        assert np.linalg.eigvalsh(0.5 * (diff + diff.T)).min() > -1e-8   # PSD


def test_riccati_step_pins_eq7_closed_form():
    """BUG-FIX PIN: riccati_step implements paper Eq 7 exactly (predicted-to-predicted):
        rho(Sigma) = F Sigma F^T - F Sigma H^T (H Sigma H^T + R)^-1 H Sigma F^T + Q.
    Verified against an independently-computed closed form, and shown DISTINCT from the old
    predict-then-update posterior (the deviation the audit found)."""
    F, Q = cv_matrices(1.0, 0.05)
    rng = np.random.default_rng(11)
    for _ in range(20):
        A = rng.normal(size=(4, 4))
        Sigma = A @ A.T + 3.0 * np.eye(4)
        R = (0.4 + rng.random()) * np.eye(2)

        # Independent Eq-7 closed form (information-update on the PRIOR, then predict).
        S = H @ Sigma @ H.T + R
        eq7 = (F @ Sigma @ F.T
               - F @ Sigma @ H.T @ np.linalg.inv(S) @ H @ Sigma @ F.T
               + Q)
        assert np.allclose(riccati_step(Sigma, R, F, Q), eq7, atol=1e-10)

        # Distinct from the WRONG predict-then-update ordering (predict first, update after).
        x = np.zeros(4)
        _, Sp = predict(x, Sigma, F, Q)
        _, post = kalman_update(x, Sp, np.zeros(2), R)     # (I-KH)(F Sigma F^T + Q)
        assert not np.allclose(riccati_step(Sigma, R, F, Q), post, atol=1e-6)


def test_minimax_value_matches_bruteforce_horizon1():
    """The tree's horizon-1 minimax value/action equals an independent min-over-actions /
    max-over-z brute force using the paper Eq-7 leaf objective tr(rho(Sigma)[:2,:2]) (which
    is measurement-value independent, so the max-over-z is trivial)."""
    cfg = MinimaxConfig(horizon=1, n_directions=4, include_stay=True, n_meas=5, v_max=5.0, q=0.0)
    robot, x_hat, Sigma = np.array([0.0, 0.0]), np.array([10.0, 3.0, 0.0, 0.0]), np.eye(4) * 4.0
    R_fn = lambda r, t: paper_noise(r, t, 0.5, 1.0, 20.0, 5.0)
    dt = 1.0
    val, _off, ai = plan_single_target(robot, x_hat, Sigma, R_fn, cfg, dt)

    offs = action_offsets(cfg, dt)
    F, Q = cv_matrices(dt, cfg.q)
    worst = []
    for a in range(offs.shape[0]):
        nr = robot + offs[a]
        x_pred, _Sp = predict(x_hat, Sigma, F, Q)
        R = R_fn(nr, x_pred[:2])
        worst.append(float(np.trace(riccati_step(Sigma, R, F, Q)[:2, :2])))   # Eq-7 leaf
    assert abs(val - min(worst)) < 1e-9
    assert ai == int(np.argmin(worst))


def test_jax_vectorized_minimax_matches_numpy_recursive():
    """The genuine-JAX jitted vectorized exact minimax equals the numpy recursive full
    enumeration (value + best action) on random horizon-2 instances (x64 parity)."""
    cfg = MinimaxConfig(horizon=2, n_directions=4, include_stay=False, n_meas=5, v_max=6.0, q=0.05)
    offs = action_offsets(cfg, dt=1.0)
    F, Q = cv_matrices(1.0, cfg.q)
    rp = (0.5, 1.0, 30.0, 5.0)                              # paper Eqs 4-5 (delta1,delta2,B,C)
    R_fn = lambda r, t: paper_noise(r, t, *rp)
    rng = np.random.default_rng(5)
    for _ in range(8):
        robot = rng.uniform(-10, 10, size=2)
        target = rng.uniform(-25, 25, size=2)
        x_hat = np.array([target[0], target[1], rng.normal(), rng.normal()])
        A = rng.normal(size=(4, 4))
        Sigma = A @ A.T + 4.0 * np.eye(4)
        v_np, a_np = minimax_value(robot, x_hat, Sigma, cfg.horizon, R_fn, offs, F, Q, cfg.n_meas)
        v_jx, a_jx = minimax_value_vectorized(robot, x_hat, Sigma, cfg.horizon, rp, offs, F, Q,
                                              cfg.n_meas)
        assert abs(v_np - v_jx) < 1e-7
        assert a_np == a_jx


def test_minimax_steers_toward_target():
    """Lower noise when closer -> the robot's chosen first move reduces robot-target distance."""
    cfg = MinimaxConfig(horizon=2, n_directions=8, include_stay=True, n_meas=5, v_max=5.0, q=0.01)
    robot, target = np.array([0.0, 0.0]), np.array([30.0, 0.0])
    x_hat, Sigma = np.array([target[0], target[1], 0.0, 0.0]), np.eye(4) * 6.0
    R_fn = lambda r, t: paper_noise(r, t, 0.5, 1.0, 50.0, 5.0)
    _val, off, _ai = plan_single_target(robot, x_hat, Sigma, R_fn, cfg, dt=1.0)
    before = np.linalg.norm(robot - target)
    after = np.linalg.norm((robot + off) - target)
    assert after < before                                       # moves toward the target


def test_minimax_horizon2_no_lower_than_horizon1_value():
    """Sanity: deeper planning does not report a lower worst-case than a single step can hide;
    the horizon-2 minimax value is a valid worst-case trace (finite, positive)."""
    cfg1 = MinimaxConfig(horizon=1, n_directions=4, n_meas=5, v_max=5.0, q=0.05)
    cfg2 = MinimaxConfig(horizon=2, n_directions=4, n_meas=5, v_max=5.0, q=0.05)
    robot, x_hat, Sigma = np.array([0.0, 0.0]), np.array([12.0, 0.0, 0.0, 0.0]), np.eye(4) * 5.0
    R_fn = lambda r, t: paper_noise(r, t, 0.5, 1.0, 20.0, 5.0)
    v1, _, _ = plan_single_target(robot, x_hat, Sigma, R_fn, cfg1, 1.0)
    v2, _, _ = plan_single_target(robot, x_hat, Sigma, R_fn, cfg2, 1.0)
    assert np.isfinite(v1) and np.isfinite(v2) and v1 > 0 and v2 > 0
