"""Known-answer geometry tests for the downward-camera sensor model, plus a
controlled camera -> GM-PHD end-to-end chain."""
import numpy as np
import pytest

from netcomm.tracking.sensors import (
    CameraSensorConfig, camera_measurement, in_footprint, simulate_detections,
    los_probability, elevation_deg,
)

# clean config: ground-sampling = h/f, no altitude noise, no covariance floor
GEO = CameraSensorConfig(focal_px=10.0, sigma_px=1.0, sigma_alt=0.0,
                         half_fov_rad=np.deg2rad(60.0), p_detect=1.0,
                         min_cov=0.0, min_altitude=1.0)


def test_nadir_measurement_is_isotropic():
    # target directly below: z = target, R = ((h/f) sigma_px)^2 I = I for h=10,f=10
    z, R = camera_measurement([0.0, 0.0, 10.0], [0.0, 0.0], GEO)
    np.testing.assert_allclose(z, [0.0, 0.0])
    np.testing.assert_allclose(R, np.eye(2), atol=1e-12)


def test_uncertainty_grows_with_altitude():
    _, R10 = camera_measurement([0.0, 0.0, 10.0], [0.0, 0.0], GEO)
    _, R20 = camera_measurement([0.0, 0.0, 20.0], [0.0, 0.0], GEO)
    np.testing.assert_allclose(R20, 4.0 * R10, atol=1e-12)   # doubles altitude -> 4x cov


def test_off_nadir_covariance_is_radially_elongated():
    # drone at origin/alt 10, target at (10,0): alpha=45deg, slant=sqrt(200)
    #   sigma_t = sqrt(200)/10 = 1.4142 ; sigma_r = 1.4142 / cos45 = 2.0
    #   bearing phi = 0 -> R = diag(sigma_r^2, sigma_t^2) = diag(4, 2)
    z, R = camera_measurement([0.0, 0.0, 10.0], [10.0, 0.0], GEO)
    np.testing.assert_allclose(z, [10.0, 0.0])
    np.testing.assert_allclose(R, np.diag([4.0, 2.0]), atol=1e-9)
    # major axis (radial) points along the drone->target bearing (+x)
    evals, evecs = np.linalg.eigh(R)
    major = evecs[:, int(np.argmax(evals))]
    assert abs(abs(major[0]) - 1.0) < 1e-6


def test_altitude_noise_adds_radial_uncertainty():
    cfg = CameraSensorConfig(focal_px=10.0, sigma_px=1.0, sigma_alt=1.0,
                             half_fov_rad=np.deg2rad(60.0), min_cov=0.0)
    # sigma_r = 2.0 + tan45 * sigma_alt = 3.0 -> radial variance 9, tangential 2
    _, R = camera_measurement([0.0, 0.0, 10.0], [10.0, 0.0], cfg)
    np.testing.assert_allclose(R, np.diag([9.0, 2.0]), atol=1e-9)


def test_in_footprint_uses_altitude_and_fov():
    drone = [0.0, 0.0, 10.0]    # footprint radius = 10 * tan(60) ~= 17.32 m
    assert in_footprint(drone, [15.0, 0.0], GEO)
    assert not in_footprint(drone, [20.0, 0.0], GEO)


def test_simulate_detections_one_per_visible_target():
    rng = np.random.default_rng(0)
    drones = np.array([[0.0, 0.0, 10.0]])
    targets = np.array([[1.0, 0.0], [100.0, 100.0]])   # second is outside footprint
    dets = simulate_detections(drones, targets, GEO, rng)
    assert len(dets) == 1                               # only the visible target
    assert np.linalg.norm(dets[0].z - targets[0]) < 5.0


# --------------------------------------------------------------------------------------
# LoS/NLoS paper-fidelity: Alzenad & Yanikomeroglu 2018, "Coverage and Rate Analysis for
# UAV Base Stations with LoS/NLoS Propagation", Eq (1) (A2G LoS-probability sigmoid).
# --------------------------------------------------------------------------------------

def test_los_defaults_match_cited_paper_dense_urban():
    """The default sigmoid constants are the DENSE-URBAN values used in the cited
    paper's numerical results (Sec. VI: a=12.08, b=0.11)."""
    cfg = CameraSensorConfig()
    assert (cfg.los_a, cfg.los_b) == (12.08, 0.11)


def test_elevation_angle_is_paper_theta():
    """Eq (1)'s angle is theta = (180/pi)*atan(h/z), z = horizontal range. The code's
    elevation_deg must reproduce it (90 deg overhead, 45 deg when h == z, ->0 grazing)."""
    assert abs(elevation_deg([0.0, 0.0, 10.0], [0.0, 0.0]) - 90.0) < 1e-6   # nadir (rho floored to 1e-9)
    assert abs(elevation_deg([0.0, 0.0, 10.0], [10.0, 0.0]) - 45.0) < 1e-9  # h == z
    # general case vs the literal paper formula
    h, z = 30.0, 40.0
    theta_paper = np.degrees(np.arctan2(h, z))     # (180/pi)*atan(h/z)
    assert abs(elevation_deg([0.0, 0.0, h], [z, 0.0]) - theta_paper) < 1e-9


def test_los_probability_matches_eq1_reference_values():
    """Pin P_LoS to an INDEPENDENT transcription of Eq (1) with the paper's dense-urban
    (a, b), across a sweep, plus literal hand-computed spot values."""
    cfg = CameraSensorConfig()                      # dense-urban a=12.08, b=0.11

    def eq1(theta_deg, a=12.08, b=0.11):            # Eq (1), transcribed straight from the paper
        return 1.0 / (1.0 + a * np.exp(-b * (theta_deg - a)))

    for theta in [0.0, 5.0, 15.0, 30.0, 45.0, 60.0, 90.0]:
        assert abs(los_probability(theta, cfg) - eq1(theta)) < 1e-12

    # literal reference values (Eq (1) with a=12.08, b=0.11)
    assert abs(los_probability(90.0, cfg) - 0.9977162471) < 1e-9   # near-certain overhead
    assert abs(los_probability(45.0, cfg) - 0.7557740819) < 1e-9
    assert abs(los_probability(0.0, cfg) - 0.0214499170) < 1e-9    # grazing -> NLoS-dominated


def test_los_probability_monotonic_and_complementary():
    """P_LoS increases with elevation and P_NLoS = 1 - P_LoS (paper P_N(z) = 1 - P_L(z))."""
    cfg = CameraSensorConfig()
    thetas = np.linspace(0.0, 90.0, 19)
    pls = np.array([los_probability(t, cfg) for t in thetas])
    assert np.all(np.diff(pls) > 0)                              # strictly increasing
    assert np.all(pls >= 0.0) and np.all(pls <= 1.0)
    # complement identity holds by construction
    assert abs((1.0 - los_probability(37.0, cfg)) - (1.0 - los_probability(37.0, cfg))) < 1e-15


def test_other_environments_override_constants():
    """Non-default environments are supported by overriding (a, b) per the Al-Hourani table."""
    urban = CameraSensorConfig(los_a=9.61, los_b=0.16)
    dense = CameraSensorConfig()                                 # 12.08, 0.11
    # at a low/oblique angle dense-urban has a lower LoS probability than urban
    assert los_probability(30.0, dense) < los_probability(30.0, urban)


def test_camera_to_gmphd_chain_tracks_target():
    # three stationary drones overhead; a target crosses their shared footprint.
    from netcomm.tracking.tracker import GMPHDTracker
    from gmphd import GMPHDConfig
    rng = np.random.default_rng(0)
    cfg = CameraSensorConfig(focal_px=600.0, sigma_px=1.0, sigma_alt=0.2,
                             half_fov_rad=np.deg2rad(60.0), p_detect=1.0)
    trk = GMPHDTracker(GMPHDConfig(dt=1.0, q=0.01, p_detect=0.98, clutter_intensity=1e-5))
    drones = np.array([[45.0, 50.0, 20.0], [55.0, 53.0, 20.0], [50.0, 45.0, 20.0]])
    target = np.array([50.0, 50.0])
    vel = np.array([1.0, 0.5])
    estimates = []
    for _ in range(25):
        target = target + vel
        dets = simulate_detections(drones, target[None, :], cfg, rng)
        estimates = trk.step(dets)
    assert len(estimates) >= 1
    best = min(estimates, key=lambda e: np.linalg.norm(e.position - target))
    assert np.linalg.norm(best.position - target) < 2.0       # multi-view fused track
