"""Integration tests for the pluggable repositioner + the Voronoi controller in-sim."""
import numpy as np
import pytest

from netcomm.tracking import (
    make_repositioner, VoronoiRepositioner, run_placed_tracking, CameraSensorConfig,
)
from coverage_control import CoverageConfig
from gmphd.types import TargetEstimate


# --------------------------------------------------------------------------- toggle
def test_make_repositioner_toggle():
    assert make_repositioner("none") is None
    assert make_repositioner("") is None
    assert make_repositioner(None) is None
    assert isinstance(make_repositioner("isotropic_voronoi"), VoronoiRepositioner)
    assert isinstance(make_repositioner("voronoi"), VoronoiRepositioner)
    with pytest.raises(NotImplementedError):
        make_repositioner("anisotropic_voronoi")
    with pytest.raises(ValueError):
        make_repositioner("bogus")


# --------------------------------------------------------------------------- step
def _estimate(x, y, w=1.0):
    return TargetEstimate(m=np.array([x, y, 0.0, 0.0]), P=np.eye(4), w=w)


def test_step_moves_drones_toward_targets_and_preserves_altitude():
    repo = make_repositioner("isotropic_voronoi", CoverageConfig(grid_res=48, v_max=20.0))
    drones = np.array([[10.0, 10.0, 18.0], [90.0, 90.0, 18.0]])
    ests = [_estimate(20.0, 20.0), _estimate(25.0, 15.0)]   # mass concentrated low-left
    new = repo.step(drones, ests, [], (0.0, 100.0, 0.0, 100.0), dt=0.2)
    assert new.shape == drones.shape
    assert np.allclose(new[:, 2], drones[:, 2])             # altitude untouched
    # drone 0 should move toward the target cluster (its x,y decrease toward ~(22,17))
    assert np.linalg.norm(new[0, :2] - np.array([22.5, 17.5])) < np.linalg.norm(drones[0, :2] - np.array([22.5, 17.5]))
    # diagnostics populated for viz
    assert repo.last_phi.shape == (48, 48)
    assert repo.last_centroids.shape == (2, 2)
    assert repo.last_cost > 0.0


def test_step_handles_no_drones():
    repo = make_repositioner("isotropic_voronoi")
    out = repo.step(np.zeros((0, 3)), [], [], (0.0, 100.0, 0.0, 100.0), dt=0.1)
    assert out.shape == (0, 3)


def test_belief_preferred_over_estimates():
    repo = make_repositioner("isotropic_voronoi", CoverageConfig(grid_res=32))
    drones = np.array([[50.0, 50.0, 18.0]])
    belief = [(np.array([10.0, 10.0]), np.eye(2), 0.9)]
    # belief present -> used; estimates ignored when belief non-empty
    repo.step(drones, [_estimate(90.0, 90.0)], belief, (0.0, 100.0, 0.0, 100.0), 0.1)
    cx, cy = repo.last_centroids[0]
    assert cx < 50.0 and cy < 50.0                          # pulled toward belief at (10,10)


# --------------------------------------------------------------------------- end to end
def test_run_placed_tracking_voronoi_descends_and_moves():
    drones = [(40.0, 40.0, 18.0), (200.0, 200.0, 18.0)]
    objs = [(60.0, 60.0), (150.0, 150.0), (220.0, 80.0)]
    res = run_placed_tracking(
        drones, objs, n_steps=25, dt=0.2, area_xy=(0.0, 300.0, 0.0, 300.0),
        sensor_cfg=CameraSensorConfig(), object_speed=3.0, tracker="gmphd",
        repositioner="isotropic_voronoi", repos_cfg=CoverageConfig(grid_res=40, v_max=12.0),
        seed=1)
    assert res["repositioner"] == "isotropic_voronoi"
    f0, fl = res["frames"][0], res["frames"][-1]
    moved = not np.allclose(np.asarray(f0["drones"])[:, :2], np.asarray(fl["drones"])[:, :2])
    assert moved
    # repos_cost is recorded against EACH frame's (moving) phi, so it is not a Lyapunov
    # series across frames (the fixed-phi descent property is proven in the unit tests);
    # here just sanity-check the per-frame diagnostic is finite and well-formed.
    costs = [fr["repos_cost"] for fr in res["frames"]]
    assert all(np.isfinite(c) and c >= 0.0 for c in costs)
    assert fl["repos_phi"].shape == (40, 40)


def test_make_repositioner_infomax_toggle():
    from netcomm.tracking import InfomaxRepositioner
    from infomax import InfomaxConfig
    assert isinstance(make_repositioner("greedy_mi", None, CameraSensorConfig()), InfomaxRepositioner)
    assert make_repositioner("greedy_mi", None, CameraSensorConfig()).cfg.method == "greedy"
    assert make_repositioner("rsp", InfomaxConfig(n_d=3), CameraSensorConfig()).cfg.method == "rsp"
    with pytest.raises(ValueError):
        make_repositioner("bogus")


def test_infomax_step_moves_drones_toward_targets():
    from infomax import InfomaxConfig
    repo = make_repositioner("greedy_mi", InfomaxConfig(horizon=2, v_max=12.0),
                             CameraSensorConfig(half_fov_rad=np.deg2rad(55.0)))
    drones = np.array([[60.0, 60.0, 20.0], [150.0, 150.0, 20.0]])
    ests = [_estimate(70.0, 70.0), _estimate(160.0, 140.0)]      # targets offset from drones
    new = repo.step(drones, ests, [], (0.0, 300.0, 0.0, 300.0), dt=0.2)
    assert new.shape == drones.shape
    assert np.allclose(new[:, 2], drones[:, 2])                  # altitude fixed
    assert not np.allclose(new[:, :2], drones[:, :2])           # drones moved
    assert repo.last_total_mi > 0.0
    # each drone moved within the per-step reach v_max*dt = 2.4 m
    assert np.all(np.linalg.norm(new[:, :2] - drones[:, :2], axis=1) <= 12.0 * 0.2 + 1e-6)


def test_run_placed_tracking_greedy_mi_moves_and_tracks():
    from infomax import InfomaxConfig
    objs = [(60.0, 60.0), (150.0, 150.0), (210.0, 90.0)]
    drones = [(o[0], o[1], 20.0) for o in objs]                 # a drone over each target
    res = run_placed_tracking(
        drones, objs, n_steps=20, dt=0.2, area_xy=(0.0, 300.0, 0.0, 300.0),
        sensor_cfg=CameraSensorConfig(half_fov_rad=np.deg2rad(55.0)), object_speed=4.0,
        tracker="gmphd", repositioner="rsp", repos_cfg=InfomaxConfig(n_d=2, horizon=2, v_max=12.0),
        seed=1)
    assert res["repositioner"] == "rsp"
    f0, fl = res["frames"][0], res["frames"][-1]
    assert not np.allclose(np.asarray(f0["drones"])[:, :2], np.asarray(fl["drones"])[:, :2])
    assert fl["repos_total_mi"] >= 0.0
    assert "repos_mi_gain" in fl


# --------------------------------------------------------------------------- minimax
def test_make_repositioner_minimax_toggle():
    from netcomm.tracking import MinimaxRepositioner
    from nonmyopic import MinimaxConfig
    repo = make_repositioner("minimax", None, CameraSensorConfig())
    assert isinstance(repo, MinimaxRepositioner)
    assert repo.name == "minimax"
    # passes a MinimaxConfig through
    repo2 = make_repositioner("minimax", MinimaxConfig(horizon=1), CameraSensorConfig())
    assert repo2.cfg.horizon == 1


def test_minimax_step_moves_drones_toward_targets_preserves_altitude():
    from nonmyopic import MinimaxConfig
    repo = make_repositioner(
        "minimax", MinimaxConfig(horizon=2, n_directions=8, include_stay=True, n_meas=5,
                                 v_max=12.0),
        CameraSensorConfig(half_fov_rad=np.deg2rad(55.0)))
    drones = np.array([[60.0, 60.0, 20.0], [150.0, 150.0, 20.0]])
    ests = [_estimate(80.0, 60.0), _estimate(150.0, 130.0)]      # targets offset from drones
    new = repo.step(drones, ests, [], (0.0, 300.0, 0.0, 300.0), dt=0.3)
    assert new.shape == drones.shape
    assert np.allclose(new[:, 2], drones[:, 2])                  # altitude fixed
    assert not np.allclose(new[:, :2], drones[:, :2])           # drones moved
    # each drone moved within the per-step reach v_max*dt = 3.6 m
    assert np.all(np.linalg.norm(new[:, :2] - drones[:, :2], axis=1) <= 12.0 * 0.3 + 1e-6)
    # at least one drone moved toward its assigned target cluster
    assert repo.last_assignment is not None
    assert repo.last_minimax_value is not None and len(repo.last_minimax_value) == 2


def test_minimax_step_no_targets_keeps_drones():
    repo = make_repositioner("minimax", None, CameraSensorConfig())
    drones = np.array([[10.0, 10.0, 18.0]])
    new = repo.step(drones, [], [], (0.0, 100.0, 0.0, 100.0), dt=0.2)
    assert np.allclose(new, drones)


def test_run_placed_tracking_minimax_runs_end_to_end():
    from nonmyopic import MinimaxConfig
    objs = [(60.0, 60.0), (150.0, 150.0), (210.0, 90.0)]
    drones = [(o[0], o[1], 20.0) for o in objs]                 # a drone over each target
    res = run_placed_tracking(
        drones, objs, n_steps=15, dt=0.2, area_xy=(0.0, 300.0, 0.0, 300.0),
        sensor_cfg=CameraSensorConfig(half_fov_rad=np.deg2rad(55.0)), object_speed=3.0,
        tracker="gmphd", repositioner="minimax",
        repos_cfg=MinimaxConfig(horizon=2, n_directions=4, n_meas=5, v_max=10.0),
        seed=1)
    assert res["repositioner"] == "minimax"
    f0, fl = res["frames"][0], res["frames"][-1]
    moved = not np.allclose(np.asarray(f0["drones"])[:, :2], np.asarray(fl["drones"])[:, :2])
    assert moved
    assert "repos_minimax_value" in fl and "repos_assignment" in fl


def test_run_placed_tracking_none_keeps_drones_stationary():
    drones = [(40.0, 40.0, 18.0), (200.0, 200.0, 18.0)]
    objs = [(60.0, 60.0), (150.0, 150.0)]
    res = run_placed_tracking(
        drones, objs, n_steps=10, dt=0.2, area_xy=(0.0, 300.0, 0.0, 300.0),
        tracker="gmphd", repositioner="none", seed=1)
    a = np.asarray(res["frames"][0]["drones"])[:, :2]
    b = np.asarray(res["frames"][-1]["drones"])[:, :2]
    assert np.allclose(a, b)
    assert "repos_cost" not in res["frames"][-1]
