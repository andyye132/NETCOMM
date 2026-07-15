"""Tests for the object path presets."""
import numpy as np
import pytest

from netcomm.tracking.paths import (
    preset_trajectories, pattern_trajectories, random_walk_trajectories, PATTERNS,
)
from netcomm.tracking import run_preset_tracking


def test_trajectory_shape():
    tr = preset_trajectories("circle", 4, 50, 0.1, (0, 100, 0, 100), 5.0)
    assert tr.shape == (50, 4, 2)


def test_circle_points_lie_on_a_circle():
    area = (0, 200, 0, 200)
    tr = pattern_trajectories(1, 120, 0.1, area, "circle", 5.0)
    R = 0.34 * 200
    d = np.hypot(tr[:, 0, 0] - 100, tr[:, 0, 1] - 100)
    np.testing.assert_allclose(d, R, atol=1e-6)


def test_figure8_passes_through_the_centre():
    area = (0, 200, 0, 200)
    tr = pattern_trajectories(1, 600, 0.1, area, "figure8", 10.0)   # >= one full loop
    assert np.hypot(tr[:, 0, 0] - 100, tr[:, 0, 1] - 100).min() < 5.0   # the 8's crossing


@pytest.mark.parametrize("pattern", PATTERNS)
def test_patterns_phase_offset_spreads_objects(pattern):
    tr = pattern_trajectories(4, 1, 0.1, (0, 300, 0, 300), pattern, 5.0)
    pts = tr[0]                                       # the 4 objects at t=0
    # phase-offset spreads them out (figure-8 self-intersects, so not all unique)
    assert len({(round(p[0], 1), round(p[1], 1)) for p in pts}) >= 2


def test_random_walk_stays_in_bounds_via_respawn():
    area = (0, 100, 0, 100)
    tr = random_walk_trajectories(6, 400, 0.1, area, 8.0, seed=3)
    assert tr[..., 0].min() >= -1e-6 and tr[..., 0].max() <= 100 + 1e-6
    assert tr[..., 1].min() >= -1e-6 and tr[..., 1].max() <= 100 + 1e-6


def test_unknown_preset_raises():
    with pytest.raises(ValueError):
        preset_trajectories("spiral", 3, 10, 0.1, (0, 10, 0, 10), 1.0)


def test_run_preset_tracking_runs():
    out = run_preset_tracking([(150.0, 150.0, 30.0)], "random_walk", n_objects=5,
                              n_steps=20, dt=0.1, area_xy=(0, 300, 0, 300),
                              object_speed=6.0, seed=0)
    assert len(out["frames"]) == 20
    assert out["n_targets"] == 5
    assert out["preset"] == "random_walk"
    assert all(f["targets"].shape == (5, 2) for f in out["frames"])
