"""Known-answer + invariant tests for precision-weighted fusion (Stage 4)."""
import numpy as np
import pytest

from modtrack import linalg
from modtrack.types import Detection
from modtrack.fusion import precision_weighted_fuse


def _det(z, R, sid=0, conf=1.0, feature=None):
    return Detection(z=np.asarray(z, float), R=np.asarray(R, float),
                     sensor_id=sid, conf=conf, feature=feature)


def test_two_equal_detections_halves_covariance():
    # Two identical (z, R)=( (0,0), I ).  P^{-1} = I + I = 2I  ->  R = 0.5 I.
    d1 = _det([0.0, 0.0], np.eye(2), sid=1)
    d2 = _det([0.0, 0.0], np.eye(2), sid=2)
    fused = precision_weighted_fuse([d1, d2])
    np.testing.assert_allclose(fused.z, [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(fused.R, 0.5 * np.eye(2), atol=1e-12)


def test_two_detections_mean_is_midpoint():
    # z1=(0,0), z2=(2,0), equal R=I  ->  fused mean = (1, 0), R = 0.5 I.
    d1 = _det([0.0, 0.0], np.eye(2), sid=1)
    d2 = _det([2.0, 0.0], np.eye(2), sid=2)
    fused = precision_weighted_fuse([d1, d2])
    np.testing.assert_allclose(fused.z, [1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(fused.R, 0.5 * np.eye(2), atol=1e-12)


def test_unequal_precision_weights_toward_confident_sensor():
    # z1=(0,0) R=I  (precise),  z2=(10,0) R=100 I  (imprecise).
    # info = I + 0.01 I = 1.01 I  -> R = (1/1.01) I.
    # info_z = [0,0] + 0.01*[10,0] = [0.1, 0]  -> z = [0.1/1.01, 0].
    d1 = _det([0.0, 0.0], np.eye(2), sid=1)
    d2 = _det([10.0, 0.0], 100.0 * np.eye(2), sid=2)
    fused = precision_weighted_fuse([d1, d2])
    np.testing.assert_allclose(fused.z, [0.1 / 1.01, 0.0], atol=1e-12)
    np.testing.assert_allclose(fused.R, (1.0 / 1.01) * np.eye(2), atol=1e-12)


def test_fusion_is_order_independent():
    d1 = _det([0.0, 1.0], [[2.0, 0.3], [0.3, 1.0]], sid=1)
    d2 = _det([3.0, -1.0], [[1.0, -0.2], [-0.2, 2.0]], sid=2)
    f_ab = precision_weighted_fuse([d1, d2])
    f_ba = precision_weighted_fuse([d2, d1])
    np.testing.assert_allclose(f_ab.z, f_ba.z, atol=1e-12)
    np.testing.assert_allclose(f_ab.R, f_ba.R, atol=1e-12)


def test_adding_a_sensor_never_increases_covariance():
    # Loewner order: R_fused <= each input R  (eigenvalues of R_in - R_fused >= 0).
    d1 = _det([0.0, 0.0], [[2.0, 0.0], [0.0, 1.0]], sid=1)
    d2 = _det([1.0, 1.0], [[1.0, 0.0], [0.0, 3.0]], sid=2)
    fused = precision_weighted_fuse([d1, d2])
    for det in (d1, d2):
        assert linalg.is_psd(det.R - fused.R)
    assert linalg.is_psd(fused.R)


def test_single_detection_returns_itself():
    d1 = _det([1.5, -2.0], [[2.0, 0.4], [0.4, 1.0]], sid=7)
    fused = precision_weighted_fuse([d1])
    np.testing.assert_allclose(fused.z, d1.z, atol=1e-12)
    np.testing.assert_allclose(fused.R, d1.R, atol=1e-12)
    assert tuple(fused.members) == (7,)


def test_common_mode_pose_term_does_not_vanish():
    # Each R = R_indep + R_pose with R_indep = I, R_pose = 0.25 I  ->  R = 1.25 I.
    # Fusing two: P_indep = 0.5 I, R_fused = 0.5 I + 0.25 I = 0.75 I.
    R_pose = 0.25 * np.eye(2)
    R = np.eye(2) + R_pose
    d1 = _det([0.0, 0.0], R, sid=1)
    d2 = _det([0.0, 0.0], R, sid=2)
    fused = precision_weighted_fuse([d1, d2], R_pose=R_pose)
    np.testing.assert_allclose(fused.R, 0.75 * np.eye(2), atol=1e-12)


def test_feature_pooling_is_normalized_average():
    # Orthonormal features [1,0] and [0,1], equal conf -> normalize([1,1]).
    d1 = _det([0.0, 0.0], np.eye(2), sid=1, feature=[1.0, 0.0])
    d2 = _det([0.0, 0.0], np.eye(2), sid=2, feature=[0.0, 1.0])
    fused = precision_weighted_fuse([d1, d2])
    expected = np.array([1.0, 1.0]) / np.sqrt(2.0)
    np.testing.assert_allclose(fused.feature, expected, atol=1e-12)
    assert np.linalg.norm(fused.feature) == pytest.approx(1.0)


def test_empty_set_raises():
    with pytest.raises(ValueError):
        precision_weighted_fuse([])


# --- robustness fixes from adversarial review ------------------------------

def test_rpose_larger_than_R_raises():
    # R_pose >= R makes R_indep = R - R_pose non-PSD -> must fail loudly, not
    # silently invert a negative-definite matrix.
    d1 = _det([0.0, 0.0], np.eye(2), sid=1)
    d2 = _det([0.0, 0.0], np.eye(2), sid=2)
    with pytest.raises(ValueError):
        precision_weighted_fuse([d1, d2], R_pose=2.0 * np.eye(2))


def test_cancelling_features_pool_to_none():
    # opposite unit features with equal confidence sum to ~0 -> no usable cue
    d1 = _det([0.0, 0.0], np.eye(2), sid=1, feature=[1.0, 0.0])
    d2 = _det([0.0, 0.0], np.eye(2), sid=2, feature=[-1.0, 0.0])
    assert precision_weighted_fuse([d1, d2]).feature is None


@pytest.mark.parametrize("d", [2, 3])
def test_two_equal_detections_any_dim(d):
    # "dimension-general": fusing two equal (0, I) detections halves cov in any d.
    z, R = np.zeros(d), np.eye(d)
    fused = precision_weighted_fuse([_det(z, R, sid=1), _det(z, R, sid=2)])
    np.testing.assert_allclose(fused.z, np.zeros(d), atol=1e-12)
    np.testing.assert_allclose(fused.R, 0.5 * np.eye(d), atol=1e-12)


def test_detection_copies_input_arrays():
    # external mutation of the source array must not corrupt the Detection
    z = np.array([1.0, 2.0])
    det = Detection(z=z, R=np.eye(2), sensor_id=0)
    z[0] = 999.0
    assert det.z[0] == pytest.approx(1.0)


def test_detection_equality_and_hash_do_not_raise():
    # array fields previously made dataclass __eq__/__hash__ blow up; now identity.
    d1 = _det([0.0, 0.0], np.eye(2), sid=1)
    d2 = _det([0.0, 0.0], np.eye(2), sid=1)
    assert d1 == d1
    assert d1 != d2
    _ = hash(d1)            # must not raise
