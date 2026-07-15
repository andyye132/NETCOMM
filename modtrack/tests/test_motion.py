"""Known-answer + invariant tests for the HMM motion models (Stage 5 foundation)."""
import numpy as np
import pytest

from modtrack import linalg
from modtrack import motion as M


# --- transition matrices ---------------------------------------------------

def test_cv_transition_matrix_known():
    # dt = 0.5, 2D -> 4x4 block [[I, 0.5 I], [0, I]]
    F = M.cv_transition_matrix(0.5, dim=2)
    expected = np.array([
        [1, 0, 0.5, 0],
        [0, 1, 0, 0.5],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=float)
    np.testing.assert_allclose(F, expected)


def test_cv_transition_advances_position_by_velocity():
    F = M.cv_transition_matrix(0.5, dim=2)
    m = np.array([1.0, 2.0, 3.0, 4.0])     # pos (1,2), vel (3,4)
    # x' = 1 + 0.5*3 = 2.5 ; y' = 2 + 0.5*4 = 4 ; velocity unchanged
    np.testing.assert_allclose(F @ m, [2.5, 4.0, 3.0, 4.0])


def test_stationary_damps_velocity_to_zero():
    F = M.transition_matrix(M.STATIONARY, dt=0.5, dim=2, stationary_damping=0.0)
    m = np.array([1.0, 2.0, 3.0, 4.0])
    # one residual drift step then velocity is zeroed
    np.testing.assert_allclose(F @ m, [2.5, 4.0, 0.0, 0.0])


def test_maneuvering_uses_cv_transition():
    np.testing.assert_allclose(
        M.transition_matrix(M.MANEUVERING, 0.7, dim=2),
        M.cv_transition_matrix(0.7, dim=2),
    )


def test_transition_matrix_3d_shape_and_action():
    F = M.cv_transition_matrix(1.0, dim=3)
    assert F.shape == (6, 6)
    m = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0])   # at origin, vel (1,2,3)
    np.testing.assert_allclose(F @ m, [1.0, 2.0, 3.0, 1.0, 2.0, 3.0])


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        M.transition_matrix(99, 0.5)


# --- process noise ---------------------------------------------------------

def test_cv_process_noise_known_blocks():
    # dt = 2, q = 1, dim = 1: dt^4/4 = 4, dt^3/2 = 4, dt^2 = 4
    Q = M.cv_process_noise(dt=2.0, q=1.0, dim=1)
    np.testing.assert_allclose(Q, [[4.0, 4.0], [4.0, 4.0]])


def test_maneuver_noise_is_scaled_cv_noise():
    base = M.process_noise(M.CONSTANT_VELOCITY, dt=0.5, dim=2)
    man = M.process_noise(M.MANEUVERING, dt=0.5, dim=2, maneuver_scale=100.0)
    np.testing.assert_allclose(man, 100.0 * base)


def test_stationary_noise_is_smaller_than_cv():
    cv = M.process_noise(M.CONSTANT_VELOCITY, dt=0.5, dim=2)
    st = M.process_noise(M.STATIONARY, dt=0.5, dim=2, stationary_scale=0.01)
    np.testing.assert_allclose(st, 0.01 * cv)


@pytest.mark.parametrize("mode", M.MODES)
def test_process_noise_is_psd(mode):
    assert linalg.is_psd(M.process_noise(mode, dt=0.5, dim=2))


# --- mode-switching Markov chain ------------------------------------------

def test_mode_transition_matrix_rows_sum_to_one():
    P = M.mode_transition_matrix()
    np.testing.assert_allclose(P.sum(axis=1), [1.0, 1.0, 1.0])


def test_mode_transition_diagonal_matches_pi_stay():
    P = M.mode_transition_matrix(pi_stay=(0.75, 0.94, 0.10))
    np.testing.assert_allclose(np.diag(P), [0.75, 0.94, 0.10])
    # off-diagonals split the remaining mass uniformly
    assert P[0, 1] == pytest.approx((1 - 0.75) / 2)
    assert P[2, 0] == pytest.approx((1 - 0.10) / 2)


# --- Kalman predict primitive ---------------------------------------------

def test_predict_state_cv_known():
    F = M.cv_transition_matrix(0.5, dim=2)
    m = np.array([0.0, 0.0, 2.0, -1.0])
    P = np.eye(4)
    m_pred, P_pred = M.predict_state(m, P, F, np.zeros((4, 4)))
    np.testing.assert_allclose(m_pred, [1.0, -0.5, 2.0, -1.0])   # moved by dt*vel
    np.testing.assert_allclose(P_pred, F @ F.T)                  # Q = 0


def test_predict_covariance_adds_process_noise_and_stays_psd():
    F = M.cv_transition_matrix(0.5, dim=2)
    P = np.eye(4)
    Q = M.process_noise(M.CONSTANT_VELOCITY, 0.5, dim=2)
    _, P_pred = M.predict_state(np.zeros(4), P, F, Q)
    np.testing.assert_allclose(P_pred, F @ P @ F.T + Q)
    assert linalg.is_psd(P_pred)
    # prediction never reduces uncertainty below the propagated prior
    assert linalg.is_psd(P_pred - F @ P @ F.T)
