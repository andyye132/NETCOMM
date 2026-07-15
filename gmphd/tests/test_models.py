"""Known-answer tests for the GM-PHD motion / measurement / density models."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from gmphd import kernels
from gmphd.models import cv_model, measurement_matrix, mvn_pdf


def test_cv_transition_known():
    F, _ = cv_model(dt=0.5, q=1.0)
    np.testing.assert_allclose(F, [
        [1, 0, 0.5, 0],
        [0, 1, 0, 0.5],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])


def test_cv_advances_position_by_velocity():
    F, _ = cv_model(0.5, 1.0)
    np.testing.assert_allclose(F @ np.array([1.0, 2.0, 3.0, 4.0]), [2.5, 4.0, 3.0, 4.0])


def test_cv_process_noise_known_entries():
    # dt = 2, q = 1: dt^4/4 = 4, dt^3/2 = 4, dt^2 = 4
    _, Q = cv_model(dt=2.0, q=1.0)
    np.testing.assert_allclose(Q, [
        [4, 0, 4, 0],
        [0, 4, 0, 4],
        [4, 0, 4, 0],
        [0, 4, 0, 4],
    ])


def test_process_noise_scales_linearly_with_q():
    _, Q1 = cv_model(0.5, 1.0)
    _, Q3 = cv_model(0.5, 3.0)
    np.testing.assert_allclose(Q3, 3.0 * Q1)


def test_measurement_matrix_extracts_position():
    H = measurement_matrix()
    np.testing.assert_allclose(H @ np.array([7.0, 8.0, 9.0, 10.0]), [7.0, 8.0])


def test_mvn_pdf_peak_value_2d():
    # N(x; x, S) at the mean = 1 / (2*pi*sqrt(|S|)); for S = 2 I -> 1/(4*pi)
    val = mvn_pdf(np.zeros(2), np.zeros(2), 2.0 * np.eye(2))
    assert val == pytest.approx(1.0 / (4.0 * np.pi))


def test_mvn_pdf_rejects_non_psd_covariance():
    # a non-PSD covariance must fail loudly, not silently return garbage
    with pytest.raises(ValueError):
        mvn_pdf(np.zeros(2), np.zeros(2), np.array([[1.0, 0.0], [0.0, -1.0]]))


def test_mvn_pdf_matches_scipy():
    from scipy.stats import multivariate_normal
    x, mean = np.array([0.3, -1.2]), np.array([0.0, 0.0])
    cov = np.array([[2.0, 0.5], [0.5, 1.0]])
    expected = multivariate_normal(mean=mean, cov=cov).pdf(x)
    assert mvn_pdf(x, mean, cov) == pytest.approx(expected)


# --- JAX kernels: PSD safety + agreement with the NumPy density -----------

def test_mvn_pdf_jax_matches_numpy_on_psd():
    # on a well-conditioned (PSD) covariance the JAX kernel matches mvn_pdf
    x, mean = np.array([0.3, -1.2]), np.array([0.0, 0.0])
    cov = np.array([[2.0, 0.5], [0.5, 1.0]])
    expected = mvn_pdf(x, mean, cov)
    got = float(kernels.mvn_pdf_jax(np.asarray(x), np.asarray(mean), np.asarray(cov)))
    assert got == pytest.approx(expected, rel=1e-9)


def test_mvn_pdf_jax_degrades_gracefully_on_non_psd():
    # the filter's internal density must NOT raise on a non-PSD covariance
    # (unlike the public mvn_pdf); it floors eigenvalues and returns a finite value
    cov = np.array([[1.0, 0.0], [0.0, -1.0]])   # one negative eigenvalue
    val = float(kernels.mvn_pdf_jax(np.zeros(2), np.zeros(2), cov))
    assert np.isfinite(val) and val >= 0.0


def test_psd_floor_makes_matrix_positive_semidefinite():
    # symmetrize + eigenvalue floor -> all eigenvalues >= floor, result symmetric
    A = np.array([[1.0, 2.0], [0.0, -3.0]])      # neither symmetric nor PSD
    Pf = np.asarray(kernels.psd_floor(np.asarray(A)))
    np.testing.assert_allclose(Pf, Pf.T, atol=1e-10)
    eigs = np.linalg.eigvalsh(Pf)
    assert (eigs > 0).all()


def test_update_does_not_raise_on_degenerate_innovation():
    # exercise the update path with a near-singular measurement covariance; the
    # PSD floor in the Kalman kernel must keep it finite rather than raising
    from gmphd.config import GMPHDConfig
    from gmphd.gmphd import GMPHDFilter
    from gmphd.types import Detection, GaussianComponent
    filt = GMPHDFilter(GMPHDConfig(p_detect=1.0, clutter_intensity=0.0))
    pred = [GaussianComponent(w=1.0, m=np.zeros(4), P=np.eye(4))]
    updated = filt.update(pred, [Detection(z=[0.0, 0.0], R=1e-14 * np.eye(2))])
    assert len(updated) == 2
    assert all(np.all(np.isfinite(c.m)) and np.all(np.isfinite(c.P)) for c in updated)
