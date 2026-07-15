"""Known-answer + cross-check tests for modtrack.uncertainty.

These pin the "covariance chain" to hand-computed answers: propagating a
covariance through a known Jacobian, and the range/bearing front-end primitive
whose geometry has an intuitive closed form.
"""
import numpy as np
import pytest

from modtrack import linalg
from modtrack import uncertainty as U


def test_propagate_covariance_linear_known():
    # J = diag(2, 3), Sigma_in = I  ->  R_out = J J^T = diag(4, 9)
    J = np.diag([2.0, 3.0])
    R_out = U.propagate_covariance(J, np.eye(2))
    np.testing.assert_allclose(R_out, np.diag([4.0, 9.0]))


def test_propagate_covariance_identity_passes_through():
    S = np.array([[2.0, 0.5], [0.5, 1.0]])
    np.testing.assert_allclose(U.propagate_covariance(np.eye(2), S), S)


def test_finite_diff_jacobian_matches_linear_map():
    A = np.array([[2.0, 0.0], [1.0, 3.0]])
    J = U.finite_diff_jacobian(lambda x: A @ x, np.array([0.7, -1.2]))
    np.testing.assert_allclose(J, A, atol=1e-6)


def test_polar_to_cartesian_known_points():
    np.testing.assert_allclose(U.polar_to_cartesian([2.0, 0.0]), [2.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(U.polar_to_cartesian([1.0, np.pi / 2]), [0.0, 1.0], atol=1e-12)


def test_polar_jacobian_matches_finite_diff():
    z = np.array([5.0, 0.7])
    J_analytic = U.polar_to_cartesian_jacobian(z)
    J_numeric = U.finite_diff_jacobian(U.polar_to_cartesian, z)
    np.testing.assert_allclose(J_analytic, J_numeric, atol=1e-6)


def test_range_bearing_covariance_known_geometry():
    # Target straight ahead (theta = 0) at range r = 5, with range std^2 = 0.04
    # and bearing std^2 = 0.01. Jacobian is diag(1, r), so:
    #   x-variance = sigma_r^2          = 0.04   (range noise -> radial / x)
    #   y-variance = r^2 * sigma_th^2   = 25*0.01 = 0.25  (bearing noise -> tangential / y)
    z = np.array([5.0, 0.0])
    cov_polar = np.diag([0.04, 0.01])
    J = U.polar_to_cartesian_jacobian(z)
    R_xy = U.propagate_covariance(J, cov_polar)
    np.testing.assert_allclose(R_xy, np.diag([0.04, 0.25]), atol=1e-12)
    assert linalg.is_psd(R_xy)


def test_propagate_covariance_3d_known():
    # dimension-general: J = diag(2,3,4), Sigma = I3 -> diag(4, 9, 16)
    J = np.diag([2.0, 3.0, 4.0])
    R_out = U.propagate_covariance(J, np.eye(3))
    np.testing.assert_allclose(R_out, np.diag([4.0, 9.0, 16.0]))
    assert linalg.is_psd(R_out)


def test_propagate_covariance_fn_matches_analytic():
    z = np.array([3.0, -0.4])
    cov_polar = np.diag([0.04, 0.02])
    R_fd = U.propagate_covariance_fn(U.polar_to_cartesian, z, cov_polar)
    R_an = U.propagate_covariance(U.polar_to_cartesian_jacobian(z), cov_polar)
    np.testing.assert_allclose(R_fd, R_an, atol=1e-6)
    assert linalg.is_psd(R_fd)
