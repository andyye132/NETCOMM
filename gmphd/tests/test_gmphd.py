"""Known-answer + invariant + end-to-end tests for the GM-PHD filter."""
import jax
import numpy as np
import pytest

# tight numeric tolerances on hand-computed values need float64
jax.config.update("jax_enable_x64", True)

from gmphd.config import GMPHDConfig
from gmphd.gmphd import GMPHDFilter
from gmphd.models import cv_model
from gmphd.types import Detection, GaussianComponent


def _comp(w, m, P):
    return GaussianComponent(w=w, m=np.asarray(m, float), P=np.asarray(P, float))


# --- predict ---------------------------------------------------------------

def test_predict_scales_weight_and_moves_mean():
    filt = GMPHDFilter(GMPHDConfig(dt=0.5, q=0.0, p_survival=0.9))
    c = _comp(1.0, [0.0, 0.0, 2.0, -1.0], np.eye(4))
    (p,) = filt.predict([c])
    assert p.w == pytest.approx(0.9)                       # scaled by p_S
    np.testing.assert_allclose(p.m, [1.0, -0.5, 2.0, -1.0])  # moved by dt*vel
    np.testing.assert_allclose(p.P, filt.F @ np.eye(4) @ filt.F.T)  # Q = 0


# --- measurement-driven birth ---------------------------------------------

def test_birth_seeds_component_at_measurement():
    filt = GMPHDFilter(GMPHDConfig(birth_weight=0.2, birth_vel_var=25.0))
    (b,) = filt.birth([Detection(z=[3.0, 4.0], R=2.0 * np.eye(2))])
    assert b.w == pytest.approx(0.2)
    np.testing.assert_allclose(b.m, [3.0, 4.0, 0.0, 0.0])
    np.testing.assert_allclose(b.P[:2, :2], 2.0 * np.eye(2))
    assert b.P[2, 2] == pytest.approx(25.0) and b.P[3, 3] == pytest.approx(25.0)


def test_birth_mode_defaults_to_measurement_driven():
    # default mode is unchanged: one birth per detection, none without detections
    filt = GMPHDFilter(GMPHDConfig())
    assert filt.cfg.birth_mode == "measurement"
    assert filt.birth([]) == []
    assert len(filt.birth([Detection(z=[1.0, 2.0], R=np.eye(2))])) == 1


# --- paper-faithful (Vo & Ma Eq. 17) intensity birth ----------------------

def test_birth_intensity_mode_uses_fixed_gm_independent_of_detections():
    intensity = [
        (0.3, np.array([5.0, 5.0, 0.0, 0.0]), np.diag([4.0, 4.0, 9.0, 9.0])),
        (0.2, np.array([-5.0, -5.0, 0.0, 0.0]), np.diag([4.0, 4.0, 9.0, 9.0])),
    ]
    filt = GMPHDFilter(GMPHDConfig(birth_mode="intensity", birth_intensity=intensity))
    # detection-independent: same fixed births with and without detections
    no_det = filt.birth([])
    with_det = filt.birth([Detection(z=[100.0, 100.0], R=np.eye(2))])
    assert len(no_det) == 2 and len(with_det) == 2
    assert [c.w for c in no_det] == pytest.approx([0.3, 0.2])
    np.testing.assert_allclose(no_det[0].m, [5.0, 5.0, 0.0, 0.0])
    np.testing.assert_allclose(no_det[1].m, [-5.0, -5.0, 0.0, 0.0])
    np.testing.assert_allclose(no_det[0].P, np.diag([4.0, 4.0, 9.0, 9.0]))


def test_birth_intensity_mode_step_spawns_target_at_fixed_location():
    # a fixed birth Gaussian + repeated detections there should yield a track,
    # exercising the 'intensity' path through a full step.
    intensity = [(0.5, np.array([0.0, 0.0, 0.0, 0.0]), np.diag([1.0, 1.0, 9.0, 9.0]))]
    cfg = GMPHDConfig(birth_mode="intensity", birth_intensity=intensity,
                      p_detect=0.98, clutter_intensity=1e-5)
    filt = GMPHDFilter(cfg)
    R = 0.25 * np.eye(2)
    ests = []
    for _ in range(10):
        ests = filt.step([Detection(z=[0.0, 0.0], R=R)])
    assert len(ests) >= 1
    best = min(ests, key=lambda e: np.linalg.norm(e.position))
    assert np.linalg.norm(best.position) < 1.0


# --- update: the canonical known-answer case ------------------------------

def test_update_single_component_single_measurement_no_clutter():
    # predicted: w=1, m=0, P=I4 ; measurement z=0, R=I2 ; p_D=1, clutter=0.
    # S = H P H^T + R = 2 I2 ; the detection term gets full weight 1, mean stays 0,
    # and the position covariance halves: P -> diag(0.5, 0.5, 1, 1).
    filt = GMPHDFilter(GMPHDConfig(p_detect=1.0, clutter_intensity=0.0))
    pred = [_comp(1.0, np.zeros(4), np.eye(4))]
    updated = filt.update(pred, [Detection(z=[0.0, 0.0], R=np.eye(2))])
    # one missed term (w=0) + one detection term (w=1)
    detection = max(updated, key=lambda c: c.w)
    assert detection.w == pytest.approx(1.0)
    np.testing.assert_allclose(detection.m, np.zeros(4), atol=1e-12)
    np.testing.assert_allclose(detection.P, np.diag([0.5, 0.5, 1.0, 1.0]), atol=1e-12)
    missed = min(updated, key=lambda c: c.w)
    assert missed.w == pytest.approx(0.0)


def test_update_missed_detection_decays_weight():
    # no measurements -> only the missed term survives, scaled by (1 - p_D)
    filt = GMPHDFilter(GMPHDConfig(p_detect=0.9))
    (out,) = filt.update([_comp(1.0, np.zeros(4), np.eye(4))], [])
    assert out.w == pytest.approx(0.1)


def test_clutter_reduces_detection_weight():
    pred = [_comp(1.0, np.zeros(4), np.eye(4))]
    det = [Detection(z=[0.0, 0.0], R=np.eye(2))]
    w_no_clutter = max(GMPHDFilter(GMPHDConfig(p_detect=1.0, clutter_intensity=0.0))
                       .update(pred, det), key=lambda c: c.w).w
    w_clutter = max(GMPHDFilter(GMPHDConfig(p_detect=1.0, clutter_intensity=0.05))
                    .update(pred, det), key=lambda c: c.w).w
    assert w_no_clutter == pytest.approx(1.0)
    assert w_clutter < 1.0


# --- prune / merge / cap / extract ----------------------------------------

def test_prune_drops_low_weight():
    filt = GMPHDFilter(GMPHDConfig(prune_threshold=0.1))
    comps = [_comp(0.05, np.zeros(4), np.eye(4)), _comp(0.9, np.ones(4), np.eye(4))]
    pruned = filt.prune(comps)
    assert len(pruned) == 1 and pruned[0].w == pytest.approx(0.9)


def test_merge_combines_coincident_components():
    filt = GMPHDFilter(GMPHDConfig(merge_threshold=4.0))
    comps = [_comp(0.5, np.zeros(4), np.eye(4)), _comp(0.5, np.zeros(4), np.eye(4))]
    merged = filt.merge(comps)
    assert len(merged) == 1
    assert merged[0].w == pytest.approx(1.0)
    np.testing.assert_allclose(merged[0].m, np.zeros(4))


def test_merge_keeps_far_apart_components_separate():
    filt = GMPHDFilter(GMPHDConfig(merge_threshold=4.0))
    comps = [_comp(0.6, np.zeros(4), np.eye(4)),
             _comp(0.6, np.array([10.0, 0, 0, 0]), np.eye(4))]   # 100 >> U
    assert len(filt.merge(comps)) == 2


def test_merge_gate_uses_candidate_own_covariance_not_seed():
    """Vo & Ma 2006 Table II: the merge gate is
        d2 = (m_i - m_j)^T (P_i)^{-1} (m_i - m_j) <= U,
    where j = argmax weight (the seed) and i is the CANDIDATE component, so the
    distance uses the CANDIDATE's own covariance P_i, not the seed's P_j.

    Discriminating case (would fail if the seed's P were used, as the pre-fix
    code did): seed at the origin with a BROAD covariance (100 I) and the
    candidate at diff = [3,0,0,0] with a TIGHT covariance (I), U = 4.
      - correct (uses P_i = I): d2 = 9 > 4  -> NOT merged -> 2 components
      - buggy   (uses P_j = 100 I): d2 = 0.09 <= 4 -> merged -> 1 component
    """
    filt = GMPHDFilter(GMPHDConfig(merge_threshold=4.0))
    seed = _comp(0.9, np.zeros(4), 100.0 * np.eye(4))            # heaviest -> seed
    cand = _comp(0.1, np.array([3.0, 0.0, 0.0, 0.0]), np.eye(4))  # tight P_i = I
    merged = filt.merge([seed, cand])
    assert len(merged) == 2          # candidate's own (tight) cov keeps them apart


def test_merge_gate_candidate_cov_can_admit_when_seed_cov_would_reject():
    """Mirror of the above: candidate's own (broad) covariance pulls it INTO the
    seed's group even though the seed's (tight) covariance would have rejected it.
    seed P_j = I, candidate P_i = 100 I at diff = [3,0,0,0], U = 4.
      - correct (uses P_i = 100 I): d2 = 0.09 <= 4 -> merged -> 1 component
      - buggy   (uses P_j = I): d2 = 9 > 4 -> NOT merged -> 2 components
    """
    filt = GMPHDFilter(GMPHDConfig(merge_threshold=4.0))
    seed = _comp(0.9, np.zeros(4), np.eye(4))                     # heaviest -> seed
    cand = _comp(0.1, np.array([3.0, 0.0, 0.0, 0.0]), 100.0 * np.eye(4))
    merged = filt.merge([seed, cand])
    assert len(merged) == 1
    assert merged[0].w == pytest.approx(1.0)


def test_merge_moment_match_covariance_with_different_covs():
    """Hand value: coincident means, different covs -> P_bar is the weighted
    average of the component covs (the spread term vanishes)."""
    filt = GMPHDFilter(GMPHDConfig(merge_threshold=4.0))
    c1 = _comp(0.25, np.zeros(4), np.diag([1.0, 1.0, 1.0, 1.0]))
    c2 = _comp(0.75, np.zeros(4), np.diag([3.0, 3.0, 3.0, 3.0]))
    (m,) = filt.merge([c1, c2])
    assert m.w == pytest.approx(1.0)
    np.testing.assert_allclose(m.m, np.zeros(4), atol=1e-12)
    # 0.25*1 + 0.75*3 = 2.5 on every diagonal
    np.testing.assert_allclose(np.diag(m.P), [2.5, 2.5, 2.5, 2.5], atol=1e-12)


def test_merge_moment_match_with_mean_spread():
    """Hand value including the spread term P_bar = sum w_i (P_i + (m_i-mbar)(...)^T).
    Two equal-weight components at +/- d on px with P = I:
      mbar = 0, spread per component = d^2 on the px diagonal,
      P_bar[0,0] = 1 + d^2, all other diagonals = 1."""
    filt = GMPHDFilter(GMPHDConfig(merge_threshold=1e6))   # large U -> always merge
    d = 2.0
    c1 = _comp(0.5, np.array([+d, 0.0, 0.0, 0.0]), np.eye(4))
    c2 = _comp(0.5, np.array([-d, 0.0, 0.0, 0.0]), np.eye(4))
    (m,) = filt.merge([c1, c2])
    np.testing.assert_allclose(m.m, np.zeros(4), atol=1e-12)
    np.testing.assert_allclose(np.diag(m.P), [1.0 + d * d, 1.0, 1.0, 1.0], atol=1e-12)


def test_cap_keeps_heaviest():
    filt = GMPHDFilter(GMPHDConfig(max_components=2))
    comps = [_comp(0.1, np.zeros(4), np.eye(4)),
             _comp(0.9, np.zeros(4), np.eye(4)),
             _comp(0.5, np.zeros(4), np.eye(4))]
    capped = filt.cap(comps)
    assert len(capped) == 2
    assert sorted(c.w for c in capped) == [0.5, 0.9]


def test_extract_reports_above_threshold_only():
    filt = GMPHDFilter(GMPHDConfig(extract_threshold=0.5))
    filt.components = [_comp(0.6, np.zeros(4), np.eye(4)),
                       _comp(0.4, np.ones(4), np.eye(4))]
    ests = filt.extract()
    assert len(ests) == 1 and ests[0].w == pytest.approx(0.6)


def test_cardinality_is_total_mass():
    filt = GMPHDFilter()
    filt.components = [_comp(0.7, np.zeros(4), np.eye(4)), _comp(0.6, np.ones(4), np.eye(4))]
    assert filt.cardinality == pytest.approx(1.3)


# --- end-to-end ------------------------------------------------------------

def test_tracks_single_constant_velocity_target():
    rng = np.random.default_rng(0)
    cfg = GMPHDConfig(dt=1.0, q=0.01, p_detect=0.98, clutter_intensity=1e-5,
                      birth_weight=1e-3, prune_threshold=1e-4, merge_threshold=4.0)
    filt = GMPHDFilter(cfg)
    F, _ = cv_model(1.0, 0.0)
    truth = np.array([0.0, 0.0, 1.0, 0.5])
    R = 0.25 * np.eye(2)
    estimates = []
    for _ in range(25):
        truth = F @ truth
        z = truth[:2] + rng.multivariate_normal(np.zeros(2), R)
        estimates = filt.step([Detection(z=z, R=R)])
    assert len(estimates) >= 1
    best = min(estimates, key=lambda e: np.linalg.norm(e.position - truth[:2]))
    # position within ~3 sigma of measurement noise, velocity converged
    assert np.linalg.norm(best.position - truth[:2]) < 1.5
    assert np.linalg.norm(best.velocity - truth[2:]) < 0.5
    assert 0.7 < filt.cardinality < 1.6                        # ~one target


def test_tracks_two_separated_targets():
    rng = np.random.default_rng(1)
    cfg = GMPHDConfig(dt=1.0, q=0.01, p_detect=0.98, clutter_intensity=1e-5)
    filt = GMPHDFilter(cfg)
    F, _ = cv_model(1.0, 0.0)
    a = np.array([0.0, 0.0, 0.5, 0.0])
    b = np.array([20.0, 20.0, -0.5, 0.0])
    R = 0.25 * np.eye(2)
    estimates = []
    for _ in range(25):
        a, b = F @ a, F @ b
        dets = [Detection(z=t[:2] + rng.multivariate_normal(np.zeros(2), R), R=R)
                for t in (a, b)]
        estimates = filt.step(dets)
    assert len(estimates) == 2
    centers = sorted(float(e.position[0]) for e in estimates)
    assert centers[0] < 10.0 and centers[1] > 10.0            # one near each target


def test_empty_scene_stays_empty():
    filt = GMPHDFilter()
    for _ in range(5):
        assert filt.step([]) == []
    assert filt.cardinality == pytest.approx(0.0)
