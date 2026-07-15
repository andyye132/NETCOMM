"""Known-answer + invariant tests for modtrack.linalg."""
import numpy as np
import pytest

from modtrack import linalg


def test_is_symmetric_known():
    assert linalg.is_symmetric(np.eye(2))
    assert linalg.is_symmetric([[1.0, 2.0], [2.0, 4.0]])
    assert not linalg.is_symmetric([[1.0, 2.0], [3.0, 4.0]])


def test_is_psd_known():
    assert linalg.is_psd(np.eye(3))
    assert linalg.is_psd([[2.0, 0.0], [0.0, 3.0]])
    assert not linalg.is_psd([[1.0, 0.0], [0.0, -1.0]])     # indefinite
    assert linalg.is_psd([[1.0, 0.0], [0.0, 0.0]])          # PSD (rank-deficient)


def test_symmetrize_idempotent_and_averages():
    A = np.array([[1.0, 3.0], [1.0, 5.0]])
    S = linalg.symmetrize(A)
    # off-diagonals become the average: (3 + 1) / 2 = 2
    np.testing.assert_allclose(S, [[1.0, 2.0], [2.0, 5.0]])
    assert linalg.is_symmetric(S)


def test_mahalanobis_sq_known_values():
    # dz = [2, 0] under cov = 2 I  ->  d^2 = 4 / 2 = 2.0
    assert linalg.mahalanobis_sq([2.0, 0.0], 2.0 * np.eye(2)) == pytest.approx(2.0)
    # dz = [1, 1] under identity  ->  d^2 = 1 + 1 = 2.0
    assert linalg.mahalanobis_sq([1.0, 1.0], np.eye(2)) == pytest.approx(2.0)
    # zero displacement -> 0
    assert linalg.mahalanobis_sq([0.0, 0.0], np.eye(2)) == pytest.approx(0.0)


def test_mahalanobis_sq_matches_explicit_inverse():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(2, 2))
    cov = A @ A.T + np.eye(2)          # SPD
    dz = rng.normal(size=2)
    expected = float(dz @ np.linalg.inv(cov) @ dz)
    assert linalg.mahalanobis_sq(dz, cov) == pytest.approx(expected)
