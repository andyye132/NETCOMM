"""Tests for the coverage-certainty heat map."""
import numpy as np
import pytest

from netcomm.tracking.coverage import CoverageField
from netcomm.tracking.sensors import CameraSensorConfig


def test_starts_fully_uncertain():
    f = CoverageField((0, 100, 0, 100), nx=10, ny=10)
    assert f.mean_certainty() == pytest.approx(0.0)
    assert f.snapshot().shape == (10, 10)


def test_decay_is_exact_with_no_drones():
    f = CoverageField((0, 100, 0, 100), nx=8, ny=8, decay_rate=0.3)
    f.certainty[:] = 1.0
    f.step(np.zeros((0, 3)), CameraSensorConfig(), dt=1.0)
    np.testing.assert_allclose(f.certainty, np.exp(-0.3), atol=1e-9)


def test_observation_raises_certainty_only_in_footprint():
    f = CoverageField((0, 100, 0, 100), nx=20, ny=20, obs_gain=2.0)
    drone = np.array([[50.0, 50.0, 20.0]])           # overhead the centre
    cfg = CameraSensorConfig(half_fov_rad=np.deg2rad(35.0))
    f.step(drone, cfg, dt=0.5)
    # centre cell certainty rose; a far corner stayed uncertain (0)
    assert f.certainty.max() > 0.05
    assert f.certainty[0, 0] == pytest.approx(0.0)   # corner is outside the footprint
    # the hottest cell is near the drone's nadir point (50,50)
    iy, ix = np.unravel_index(np.argmax(f.certainty), f.certainty.shape)
    assert abs(f.gx[iy, ix] - 50.0) < 15.0 and abs(f.gy[iy, ix] - 50.0) < 15.0


def test_certainty_stays_in_unit_interval():
    f = CoverageField((0, 100, 0, 100), nx=12, ny=12, obs_gain=50.0)
    drone = np.array([[50.0, 50.0, 30.0]])
    for _ in range(50):
        f.step(drone, CameraSensorConfig(half_fov_rad=np.deg2rad(60.0)), dt=1.0)
    assert f.certainty.min() >= 0.0 and f.certainty.max() <= 1.0


def test_repeated_observation_approaches_one():
    f = CoverageField((0, 100, 0, 100), nx=12, ny=12, obs_gain=3.0)
    drone = np.array([[50.0, 50.0, 20.0]])
    cfg = CameraSensorConfig(half_fov_rad=np.deg2rad(60.0))
    before = f.mean_certainty()
    for _ in range(40):
        f.step(drone, cfg, dt=0.5)
    assert f.mean_certainty() > before
    assert f.certainty.max() > 0.8           # well-observed cells become certain
