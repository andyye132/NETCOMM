"""End-to-end phi-COUPLING tests for the wired tracking pipeline.

The deep audit flagged that the *coupling* between the three stages was never
tested directly:

    GM-PHD belief  --(_phi_sources in repositioner.py)-->  importance density phi
                   --(CoverageController / MI / minimax)-->  drone reposition action

Each stage is unit-tested in isolation (gmphd/, coverage_control/, infomax/,
nonmyopic/, and netcomm/tracking/tests/test_repositioner.py), but nothing asserts
that the *wired* loop in runner.run_placed_tracking actually steers the drones
using the filter's belief. These tests close that gap on a hand-checked
ground-truth scenario:

  * objects are clustered tight in ONE region (lower-left, ~ (40, 40));
  * one drone starts over the cluster so the GM-PHD immediately accrues belief
    mass there (cold-start would otherwise leave belief — and thus phi — flat);
  * the other drones start far away (upper-right / lower-right corners).

We then assert the COUPLING holds for repositioner='isotropic_voronoi' (and also
for 'greedy_mi' and 'minimax'):

  1. the GM-PHD belief mass — i.e. the rasterized phi — concentrates AT the true
     cluster (phi argmax + >80% of the above-floor mass within a small radius);
  2. the repositioner MOVES the drones measurably toward that region over the
     episode (mean drone->cluster distance decreases substantially);
  3. tracked targets' estimate-to-truth distance stays BOUNDED (small).

The FAIL-IF-PHI-DROPPED guard (``test_voronoi_flat_phi_does_not_reach_cluster``)
runs the SAME Voronoi controller with an empty belief/estimates (phi collapses to
the uniform floor). With flat phi the drones converge to the geometric Voronoi
centroids of the *uniform* region, NOT to the cluster — so the belief-driven run
lands the far drones an order of magnitude closer to the cluster than the flat
run. That contrast is exactly what breaks if phi were dropped or flattened.
"""
import warnings

import numpy as np
import pytest

from netcomm.tracking import (
    run_placed_tracking, VoronoiRepositioner, CameraSensorConfig,
)
from netcomm.tracking.repositioner import _phi_sources
from coverage_control import CoverageConfig
from infomax import InfomaxConfig
from nonmyopic import MinimaxConfig


AREA = (0.0, 200.0, 0.0, 200.0)
CLUSTER = np.array([40.0, 40.0])          # the hand-placed ground-truth target cluster


def _clustered_scenario():
    """4 objects clustered tight at CLUSTER; one drone over the cluster (so the
    GM-PHD sees it and builds belief), two drones in far corners."""
    rng = np.random.default_rng(0)
    objs = [(float(CLUSTER[0] + rng.uniform(-6, 6)),
             float(CLUSTER[1] + rng.uniform(-6, 6))) for _ in range(4)]
    drones = [(40.0, 40.0, 30.0),         # over the cluster -> feeds belief
              (170.0, 170.0, 30.0),       # far upper-right
              (180.0, 30.0, 30.0)]        # far lower-right
    return drones, objs


def _mean_drone_to_cluster(frame) -> float:
    dr = np.asarray(frame["drones"])[:, :2]
    return float(np.mean(np.linalg.norm(dr - CLUSTER, axis=1)))


def _max_estimate_to_truth(frame) -> float:
    """Worst (over tracks) distance from a track estimate to its nearest truth."""
    tg = np.asarray(frame["targets"]).reshape(-1, 2)
    ests = frame["estimates"]
    if not ests or tg.shape[0] == 0:
        return float("inf")
    return max(float(np.min(np.linalg.norm(tg - np.asarray(pos)[:2], axis=1)))
               for (pos, _cov, _w) in ests)


def _phi_mass_fraction_near_cluster(phi, radius=40.0) -> tuple:
    """(fraction of above-floor phi mass within `radius` of CLUSTER, argmax-xy).

    phi is built by build_phi_grid on the (ny, nx) cell-center grid of AREA, so we
    reconstruct that grid here to locate the mass. The uniform floor is subtracted
    so we measure the BELIEF-DRIVEN bumps, not the cold-start baseline."""
    phi = np.asarray(phi)
    ny, nx = phi.shape
    xmn, xmx, ymn, ymx = AREA
    xs = xmn + (np.arange(nx) + 0.5) / nx * (xmx - xmn)
    ys = ymn + (np.arange(ny) + 0.5) / ny * (ymx - ymn)
    gx, gy = np.meshgrid(xs, ys)
    above = phi - phi.min()                              # strip the uniform floor
    near = np.hypot(gx - CLUSTER[0], gy - CLUSTER[1]) <= radius
    total = above.sum()
    frac = float(above[near].sum() / total) if total > 1e-9 else 0.0
    gi = np.unravel_index(int(np.argmax(phi)), phi.shape)
    return frac, np.array([xs[gi[1]], ys[gi[0]]])


# --------------------------------------------------------------------------- phi build
def test_phi_sources_concentrates_on_belief_cluster():
    """Stage-1 coupling, isolated: a GM-PHD belief clustered at CLUSTER must yield
    phi centers/weights at the cluster, and build_phi_grid must put its peak there."""
    belief = [(CLUSTER + np.array([dx, dy]), np.eye(2) * 4.0, 0.9)
              for dx, dy in [(-4, 4), (4, -4), (0, 0)]]
    centers, weights = _phi_sources([], belief)
    assert centers.shape == (3, 2)
    assert np.all(weights > 0.0)
    # every phi center sits at the cluster (the belief means), not scattered/flat
    assert np.all(np.linalg.norm(centers - CLUSTER, axis=1) <= 6.0)

    repo = VoronoiRepositioner(CoverageConfig(grid_res=48, bump_sigma=18.0))
    repo.step(np.array([[170.0, 170.0, 30.0]]), [], belief, AREA, 0.5)
    frac, argmax_xy = _phi_mass_fraction_near_cluster(repo.last_phi)
    assert frac > 0.8                                    # phi mass concentrated at cluster
    assert np.linalg.norm(argmax_xy - CLUSTER) < 25.0    # phi peak ~ at the cluster


# ------------------------------------------------------------------- isotropic_voronoi
def test_voronoi_coupling_belief_drives_drones_to_cluster():
    """Full wired pipeline: belief -> phi -> Voronoi action steers drones to the
    cluster, phi concentrates there, and tracks stay locked to truth."""
    drones, objs = _clustered_scenario()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                  # benign non-PSD sampling warns
        res = run_placed_tracking(
            drones, objs, n_steps=30, dt=0.5, area_xy=AREA,
            sensor_cfg=CameraSensorConfig(half_fov_rad=np.deg2rad(45.0)),
            object_speed=1.0, tracker="gmphd",
            repositioner="isotropic_voronoi",
            repos_cfg=CoverageConfig(grid_res=48, v_max=20.0, bump_sigma=18.0),
            seed=3)

    assert res["repositioner"] == "isotropic_voronoi"
    f0, fl = res["frames"][0], res["frames"][-1]

    # (1) phi (the rasterized GM-PHD belief mass) concentrates AT the true cluster.
    frac, argmax_xy = _phi_mass_fraction_near_cluster(fl["repos_phi"])
    assert frac > 0.8, f"phi mass not concentrated at cluster (frac={frac:.3f})"
    assert np.linalg.norm(argmax_xy - CLUSTER) < 25.0, \
        f"phi peak at {argmax_xy} not near cluster {CLUSTER}"

    # (2) the repositioner MOVES drones measurably toward the cluster over the episode.
    d0, dl = _mean_drone_to_cluster(f0), _mean_drone_to_cluster(fl)
    assert dl < d0 - 25.0, f"drones did not approach cluster (first={d0:.1f}, last={dl:.1f})"

    # (3) tracked targets' estimate-to-truth distance stays bounded.
    assert fl["n_estimates"] >= 1, "tracker produced no estimates"
    assert _max_estimate_to_truth(fl) < 8.0, "track estimate drifted from truth"


def test_voronoi_flat_phi_does_not_reach_cluster():
    """FAIL-IF-PHI-DROPPED guard. Same Voronoi controller / same start, but phi is
    FLAT (empty belief+estimates -> uniform floor). With flat phi the far drones go
    to the geometric Voronoi centroids of the uniform region, NOT to the cluster, so
    they land far away. The belief-driven run (above) lands them an order of magnitude
    closer — proving the drones move because of phi, not regardless of it."""
    drones, _ = _clustered_scenario()
    drones = np.asarray([list(d) for d in drones], dtype=float)
    cfg = CoverageConfig(grid_res=48, v_max=20.0, bump_sigma=18.0)

    # Belief CLUSTERED at the true cluster -> phi peaked there.
    repo_b = VoronoiRepositioner(cfg)
    belief = [(CLUSTER + np.array([dx, dy]), np.eye(2) * 4.0, 0.9)
              for dx, dy in [(-4, 4), (4, -4), (0, 0)]]
    pos_b = drones.copy()
    for _ in range(20):
        pos_b[:, :2] = repo_b.step(pos_b, [], belief, AREA, 0.5)[:, :2]

    # FLAT phi -> no belief/estimates at all.
    repo_f = VoronoiRepositioner(cfg)
    pos_f = drones.copy()
    for _ in range(20):
        pos_f[:, :2] = repo_f.step(pos_f, [], [], AREA, 0.5)[:, :2]

    d_belief = np.linalg.norm(pos_b[:, :2] - CLUSTER, axis=1)
    d_flat = np.linalg.norm(pos_f[:, :2] - CLUSTER, axis=1)

    # With belief, a drone LOCKS ONTO the cluster (the phi peak owns a Voronoi cell);
    # the remaining drones spread to cover the floor area. With FLAT phi every cell is
    # geometric, so even the nearest drone is stranded far from the cluster.
    assert np.min(d_belief) < 20.0, f"belief-driven drone not at cluster: {d_belief}"
    assert np.min(d_flat) > 40.0, f"flat phi unexpectedly reached cluster: {d_flat}"
    # The coupling effect must be large: belief lands the drones much closer on average.
    assert np.mean(d_flat) > np.mean(d_belief) + 40.0, \
        f"flat({np.mean(d_flat):.1f}) not clearly worse than belief({np.mean(d_belief):.1f})"


# --------------------------------------------------------------- greedy_mi / minimax
@pytest.mark.parametrize("name, cfg", [
    ("greedy_mi", InfomaxConfig(horizon=2, v_max=20.0)),
    ("minimax", MinimaxConfig(horizon=2, n_directions=8, n_meas=5, v_max=20.0,
                              include_stay=True)),
])
def test_information_repositioners_couple_belief_to_motion(name, cfg):
    """The MI and minimax planners build their target priors from the SAME GM-PHD
    output (_target_priors). On the clustered scenario they too must steer the far
    drones toward the cluster while the tracks stay locked to truth — coupling that
    would vanish if the belief/estimates were dropped (no targets -> drones hold)."""
    drones, objs = _clustered_scenario()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = run_placed_tracking(
            drones, objs, n_steps=30, dt=0.5, area_xy=AREA,
            sensor_cfg=CameraSensorConfig(half_fov_rad=np.deg2rad(50.0)),
            object_speed=1.0, tracker="gmphd",
            repositioner=name, repos_cfg=cfg, seed=3)

    assert res["repositioner"] == name
    f0, fl = res["frames"][0], res["frames"][-1]

    # drones approach the cluster substantially (information gradient pulls them in)
    d0, dl = _mean_drone_to_cluster(f0), _mean_drone_to_cluster(fl)
    assert dl < d0 - 25.0, f"{name}: drones did not approach cluster ({d0:.1f}->{dl:.1f})"

    # tracks stay bounded to truth
    assert fl["n_estimates"] >= 1, f"{name}: no estimates produced"
    assert _max_estimate_to_truth(fl) < 8.0, f"{name}: track drifted from truth"
