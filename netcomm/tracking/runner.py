"""In-sim tracking loop: drive the NETCOMM drones as cameras and track ground targets.

Reuses the existing NetCommWorld for drone kinematics (so the sensors ARE the
network drones), spawns a target population, and runs a toggleable tracker each
step. Records ground truth, detections, and estimates for metrics / visualization.

The tracker toggle ('none' | 'gmphd' | 'modtrack'-later) keeps this decoupled from
the packet/routing loop for now; the network loop can be fused in later for the
joint repositioning objective.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from netcomm.runner import NetCommWorld
from netcomm.types import NetCommConfig
from gmphd import GMPHDConfig

from .sensors import CameraSensorConfig, simulate_detections
from .targets import TargetConfig, TargetPopulation
from .tracker import make_tracker
from .repositioner import make_repositioner
from .coverage import CoverageField


def _safe_tracker_step(trk, detections, prev_estimates, prev_belief) -> Tuple[list, list, bool]:
    """Advance the tracker one step, guarding against a degenerate/non-PSD covariance.

    A near-singular or non-positive-definite measurement/posterior covariance can make
    the GM-PHD update raise (np.linalg.LinAlgError on solve, or ValueError from the
    mvn_pdf positive-definite check). Rather than crash the whole episode, the step
    DEGRADES: we keep the prior step's belief/estimates and flag that it happened.

    Returns (estimates, belief, degraded).
    """
    if trk is None:
        return [], [], False
    try:
        estimates = trk.step(detections)
        belief = trk.belief() if hasattr(trk, "belief") else []
        return estimates, belief, False
    except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
        warnings.warn(f"tracker step degraded (non-PSD/degenerate covariance): {exc!r}; "
                      "keeping prior belief for this step", RuntimeWarning, stacklevel=2)
        return list(prev_estimates), list(prev_belief), True


def _target_seed(seed: int) -> int:
    """A target-spawn seed decorrelated from the DRONE-placement seed.

    Callers (the GUI random-drone preview, the test harness) sample drone positions with
    ``np.random.default_rng(seed)`` drawing uniform x,y in the same order the target
    spawner uses. With the same raw ``seed`` the i-th target lands exactly on the i-th
    drone (so every object is always inside a drone footprint). Spawning a child
    ``SeedSequence`` makes the target stream independent of any ``default_rng(seed)`` used
    for drones, while staying deterministic in ``seed`` (matched-seed fairness across
    methods is preserved)."""
    return int(np.random.SeedSequence(int(seed), spawn_key=(0x7A06E7,)).generate_state(1)[0])


def _attach_repos_frame(frame: Dict, repo) -> None:
    """Record the repositioner's per-step diagnostics for visualization (getattr-guarded
    so it works for any repositioner: Voronoi exposes labels/centroids/phi/cost, the
    MI planner exposes mi_gain/redundancy/total_mi)."""
    if repo is None:
        return
    for key, attr in (("repos_labels", "last_labels"), ("repos_centroids", "last_centroids"),
                      ("repos_phi", "last_phi"), ("repos_mi_gain", "last_mi_gain"),
                      ("repos_redundancy", "last_redundancy"),
                      ("repos_minimax_value", "last_minimax_value"),
                      ("repos_assignment", "last_assignment")):
        v = getattr(repo, attr, None)
        if v is not None:
            frame[key] = v.copy() if isinstance(v, np.ndarray) else v
    if hasattr(repo, "last_cost"):
        frame["repos_cost"] = float(repo.last_cost)
    if hasattr(repo, "last_total_mi"):
        frame["repos_total_mi"] = float(repo.last_total_mi)


def run_tracking_episode(
    cfg: NetCommConfig,
    n_steps: int,
    key,
    tracker: str = "gmphd",
    repositioner: str = "none",
    repos_cfg=None,
    target_cfg: Optional[TargetConfig] = None,
    sensor_cfg: Optional[CameraSensorConfig] = None,
    gmphd_cfg: Optional[GMPHDConfig] = None,
    v_max: float = 10.0,
    record: bool = True,
    coverage_decay: float = 0.3,
    coverage_grid: tuple = (28, 28),
    progress=None,
) -> Dict:
    """Run the drones-as-cameras tracking loop on the sim world.

    Returns a dict with per-step frames (drone poses, target ground truth,
    detections, estimates) and summary fields — the same frame schema as
    run_placed_tracking / run_preset_tracking (belief, coverage, target_ids).
    """
    target_cfg = target_cfg or TargetConfig(area_xy=cfg.area_xy)
    sensor_cfg = sensor_cfg or CameraSensorConfig()
    gmphd_cfg = gmphd_cfg or GMPHDConfig(dt=float(cfg.dt))

    key, sk_world, sk_rng = jax.random.split(key, 3)
    world = NetCommWorld(cfg, sk_world, v_max=v_max)
    rng = np.random.default_rng(int(jax.random.randint(sk_rng, (), 0, 2 ** 31 - 1)))
    targets = TargetPopulation.spawn(target_cfg, rng)
    trk = make_tracker(tracker, gmphd_cfg)
    repo = make_repositioner(repositioner, repos_cfg, sensor_cfg)
    field = CoverageField(cfg.area_xy, nx=coverage_grid[0], ny=coverage_grid[1],
                          decay_rate=coverage_decay)
    target_ids = np.arange(targets.n)         # TargetPopulation reflects; identities stable

    frames: List[Dict] = []
    prev_estimates, prev_belief = [], []          # repositioner uses last step's belief (causal)
    n_degraded = 0
    for t in range(n_steps):
        world.advance(cfg.dt)
        targets.advance(cfg.dt)

        pos = np.asarray(world.state.pos)
        valid = np.asarray(world.valid, dtype=bool)
        airborne = valid & (pos[:, 2] >= sensor_cfg.min_altitude)
        # the drones ARE the network nodes: override their (x, y) with the repositioner
        # and zero those nodes' xy velocity so world.advance no longer drifts them.
        if repo is not None and airborne.any():
            idx = np.where(airborne)[0]
            new = repo.step(pos[idx], prev_estimates, prev_belief, cfg.area_xy, cfg.dt)
            new_pos = world.state.pos.at[idx, 0].set(jnp.asarray(new[:, 0])) \
                                     .at[idx, 1].set(jnp.asarray(new[:, 1]))
            new_vel = world.state.vel.at[idx, 0].set(0.0).at[idx, 1].set(0.0)
            world.state = world.state._replace(pos=new_pos, vel=new_vel)
            pos = np.asarray(world.state.pos)
        sensor_xyz = pos[airborne]

        detections = simulate_detections(sensor_xyz, targets.positions, sensor_cfg,
                                          rng, area=cfg.area_xy)
        estimates, belief, degraded = _safe_tracker_step(
            trk, detections, prev_estimates, prev_belief)
        n_degraded += int(degraded)
        field.step(sensor_xyz, sensor_cfg, cfg.dt)

        if record:
            frame = {
                "t": t,
                "drones": sensor_xyz.copy(),
                "targets": targets.positions.copy(),
                "target_ids": target_ids.copy(),
                "detections": [(d.z.copy(), d.R.copy()) for d in detections],
                "estimates": [(e.position.copy(), e.position_covariance.copy(), float(e.w))
                              for e in estimates],
                "n_estimates": len(estimates),
                "belief": list(belief),
                "coverage": field.snapshot(),
                "mean_certainty": field.mean_certainty(),
                "tracker_degraded": degraded,
            }
            _attach_repos_frame(frame, repo)
            frames.append(frame)
        prev_estimates, prev_belief = estimates, belief
        if progress is not None:
            progress(t + 1, n_steps)

    result = {
        "frames": frames,
        "tracker": tracker,
        "repositioner": repositioner,
        "n_targets": targets.n,
        "n_steps": n_steps,
        "dt": float(cfg.dt),
        "area_xy": tuple(float(x) for x in cfg.area_xy),
        "sensor_cfg": sensor_cfg,
        "gmphd_cfg": gmphd_cfg,
        "sensor_half_fov_rad": float(sensor_cfg.half_fov_rad),
        "n_degraded_steps": n_degraded,
    }
    if trk is not None and hasattr(trk, "cardinality"):
        result["final_cardinality"] = float(trk.cardinality)
    return result


def run_placed_tracking(
    drone_specs: Sequence[Sequence[float]],
    object_specs: Sequence[Sequence[float]],
    *,
    n_steps: int,
    dt: float,
    area_xy,
    sensor_cfg: Optional[CameraSensorConfig] = None,
    gmphd_cfg: Optional[GMPHDConfig] = None,
    object_speed: float = 2.0,
    tracker: str = "gmphd",
    repositioner: str = "none",
    repos_cfg=None,
    seed: int = 0,
    coverage_decay: float = 0.3,
    coverage_grid: tuple = (28, 28),
    progress=None,
) -> Dict:
    """Run the tracking sim from explicitly PLACED drones and objects (for the app).

    drone_specs : list of (x, y, radius) — stationary downward cameras; the ground
        footprint radius is the user-chosen ``radius`` (internally converted to an
        altitude ``radius / tan(half_fov)`` so the camera covariance model applies).
    object_specs : list of (x, y) — ground targets; each is given a random
        constant velocity (magnitude up to ``object_speed``) so there is motion to
        track. Returns the same per-frame structure as ``run_tracking_episode``.
    """
    sensor_cfg = sensor_cfg or CameraSensorConfig()
    gmphd_cfg = gmphd_cfg or GMPHDConfig(dt=float(dt))
    rng = np.random.default_rng(seed)
    tan_fov = float(np.tan(sensor_cfg.half_fov_rad))

    if len(drone_specs) > 0:
        drones_xyz = np.array(
            [[float(x), float(y), max(float(r), 1e-3) / max(tan_fov, 1e-6)]
             for (x, y, r) in drone_specs], dtype=float)
    else:
        drones_xyz = np.zeros((0, 3))

    if len(object_specs) > 0:
        pos = np.array([[float(x), float(y)] for (x, y) in object_specs], dtype=float)
        vel = rng.uniform(-object_speed, object_speed, size=pos.shape)
        targets = TargetPopulation(pos, vel, area_xy)
    else:
        targets = TargetPopulation(np.zeros((0, 2)), np.zeros((0, 2)), area_xy)

    trk = make_tracker(tracker, gmphd_cfg)
    repo = make_repositioner(repositioner, repos_cfg, sensor_cfg)
    field = CoverageField(area_xy, nx=coverage_grid[0], ny=coverage_grid[1],
                          decay_rate=coverage_decay)

    frames: List[Dict] = []
    prev_estimates, prev_belief = [], []          # repositioner uses last step's belief (causal)
    n_degraded = 0
    for t in range(n_steps):
        if targets.n > 0:
            targets.advance(dt)
        # reposition drones BEFORE sensing, from the previous step's tracker belief
        if repo is not None and drones_xyz.shape[0] > 0:
            drones_xyz[:, :2] = repo.step(drones_xyz, prev_estimates, prev_belief,
                                          area_xy, dt)[:, :2]
        if drones_xyz.shape[0] > 0 and targets.n > 0:
            detections = simulate_detections(drones_xyz, targets.positions, sensor_cfg,
                                             rng, area=area_xy)
        else:
            detections = []
        estimates, belief, degraded = _safe_tracker_step(
            trk, detections, prev_estimates, prev_belief)
        n_degraded += int(degraded)
        field.step(drones_xyz, sensor_cfg, dt)
        frame = {
            "t": t,
            "drones": drones_xyz.copy(),
            "targets": targets.positions.copy(),
            "target_ids": np.arange(targets.n),   # TargetPopulation reflects; stable ids
            "detections": [(d.z.copy(), d.R.copy()) for d in detections],
            "estimates": [(e.position.copy(), e.position_covariance.copy(), float(e.w))
                          for e in estimates],
            "n_estimates": len(estimates),
            "belief": list(belief),
            "coverage": field.snapshot(),
            "mean_certainty": field.mean_certainty(),
            "tracker_degraded": degraded,
        }
        _attach_repos_frame(frame, repo)
        frames.append(frame)
        prev_estimates, prev_belief = estimates, belief
        if progress is not None:
            progress(t + 1, n_steps)

    result = {
        "frames": frames,
        "tracker": tracker,
        "repositioner": repositioner,
        "n_targets": targets.n,
        "n_steps": n_steps,
        "dt": float(dt),
        "area_xy": tuple(float(x) for x in area_xy),
        "sensor_cfg": sensor_cfg,
        "gmphd_cfg": gmphd_cfg,
        "sensor_half_fov_rad": float(sensor_cfg.half_fov_rad),
        "n_degraded_steps": n_degraded,
    }
    if trk is not None and hasattr(trk, "cardinality"):
        result["final_cardinality"] = float(trk.cardinality)
    return result


def run_preset_tracking(
    drone_specs: Sequence[Sequence[float]],
    preset: str,
    *,
    n_objects: int,
    n_steps: int,
    dt: float,
    area_xy,
    sensor_cfg: Optional[CameraSensorConfig] = None,
    gmphd_cfg: Optional[GMPHDConfig] = None,
    object_speed: float = 4.0,
    tracker: str = "gmphd",
    repositioner: str = "none",
    repos_cfg=None,
    seed: int = 0,
    coverage_decay: float = 0.3,
    coverage_grid: tuple = (64, 64),
    progress=None,
) -> Dict:
    """Run tracking on a generated PRESET scenario (a pedestrian path family).

    ``preset`` is one of: random_walk, random_walk_varied, figure8, circle,
    square, triangle (see netcomm.tracking.paths). The object positions come from
    the preset's trajectory; drones are placed as in ``run_placed_tracking``.
    """
    from .paths import preset_trajectories

    sensor_cfg = sensor_cfg or CameraSensorConfig()
    gmphd_cfg = gmphd_cfg or GMPHDConfig(dt=float(dt))
    tan_fov = float(np.tan(sensor_cfg.half_fov_rad))
    if len(drone_specs) > 0:
        drones_xyz = np.array(
            [[float(x), float(y), max(float(r), 1e-3) / max(tan_fov, 1e-6)]
             for (x, y, r) in drone_specs], dtype=float)
    else:
        drones_xyz = np.zeros((0, 3))

    # NB: targets use a seed decorrelated from the drone-placement seed (see _target_seed)
    # so objects are NOT spawned on top of the drones that the GUI/harness sampled from
    # the same raw seed. return_ids: identity labels split at respawn teleports so the
    # evaluation can treat the replacement as a NEW track (see paths.py).
    object_traj, object_ids = preset_trajectories(
        preset, int(n_objects), int(n_steps), float(dt), area_xy, float(object_speed),
        seed=_target_seed(seed), return_ids=True)
    rng = np.random.default_rng(seed + 777)
    trk = make_tracker(tracker, gmphd_cfg)
    repo = make_repositioner(repositioner, repos_cfg, sensor_cfg)
    field = CoverageField(area_xy, nx=coverage_grid[0], ny=coverage_grid[1],
                          decay_rate=coverage_decay)

    frames: List[Dict] = []
    prev_estimates, prev_belief = [], []          # repositioner uses last step's belief (causal)
    n_degraded = 0
    for t in range(n_steps):
        tg = object_traj[t]
        if repo is not None and drones_xyz.shape[0] > 0:
            drones_xyz[:, :2] = repo.step(drones_xyz, prev_estimates, prev_belief,
                                          area_xy, dt)[:, :2]
        if drones_xyz.shape[0] > 0 and tg.shape[0] > 0:
            detections = simulate_detections(drones_xyz, tg, sensor_cfg, rng, area=area_xy)
        else:
            detections = []
        estimates, belief, degraded = _safe_tracker_step(
            trk, detections, prev_estimates, prev_belief)
        n_degraded += int(degraded)
        field.step(drones_xyz, sensor_cfg, dt)
        frame = {
            "t": t,
            "drones": drones_xyz.copy(),
            "targets": tg.copy(),
            "target_ids": object_ids[t].copy(),
            "detections": [(d.z.copy(), d.R.copy()) for d in detections],
            "estimates": [(e.position.copy(), e.position_covariance.copy(), float(e.w))
                          for e in estimates],
            "n_estimates": len(estimates),
            "belief": list(belief),
            "coverage": field.snapshot(),
            "mean_certainty": field.mean_certainty(),
            "tracker_degraded": degraded,
        }
        _attach_repos_frame(frame, repo)
        frames.append(frame)
        prev_estimates, prev_belief = estimates, belief
        if progress is not None:
            progress(t + 1, n_steps)

    result = {
        "frames": frames,
        "tracker": tracker,
        "repositioner": repositioner,
        "n_targets": int(n_objects),
        "n_steps": n_steps,
        "dt": float(dt),
        "area_xy": tuple(float(x) for x in area_xy),
        "sensor_cfg": sensor_cfg,
        "gmphd_cfg": gmphd_cfg,
        "sensor_half_fov_rad": float(sensor_cfg.half_fov_rad),
        "preset": preset,
        "n_degraded_steps": n_degraded,
    }
    if trk is not None and hasattr(trk, "cardinality"):
        result["final_cardinality"] = float(trk.cardinality)
    return result
