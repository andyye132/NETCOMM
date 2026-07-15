"""Tests for the ground-target population."""
import numpy as np
import pytest

from netcomm.tracking.targets import TargetConfig, TargetPopulation


def test_spawn_count_and_bounds():
    rng = np.random.default_rng(0)
    cfg = TargetConfig(n_targets=5, area_xy=(0.0, 100.0, 0.0, 100.0), v_max=5.0)
    pop = TargetPopulation.spawn(cfg, rng)
    assert pop.n == 5
    assert pop.positions.shape == (5, 2)
    assert np.all(pop.positions[:, 0] >= 0) and np.all(pop.positions[:, 0] <= 100)
    assert np.all(pop.positions[:, 1] >= 0) and np.all(pop.positions[:, 1] <= 100)


def test_constant_velocity_advance():
    pop = TargetPopulation([[50.0, 50.0]], [[1.0, 2.0]], (0.0, 100.0, 0.0, 100.0))
    pop.advance(1.0)
    np.testing.assert_allclose(pop.positions, [[51.0, 52.0]])


def test_reflective_boundary_bounces():
    pop = TargetPopulation([[99.0, 50.0]], [[5.0, 0.0]], (0.0, 100.0, 0.0, 100.0))
    pop.advance(1.0)                  # would reach x=104, reflects to 96, vx flips
    np.testing.assert_allclose(pop.positions, [[96.0, 50.0]])
    np.testing.assert_allclose(pop.velocities, [[-5.0, 0.0]])
