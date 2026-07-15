"""Pluggable repositioning controllers + the toggle (mirrors tracker.py).

A ``Repositioner`` takes the current drone poses plus the tracker's target
estimates and returns updated drone poses — the "where should the drones move"
policy, separate from the "where are the targets" estimator. ``make_repositioner``
is the toggle: 'none' (drones don't reposition) and 'isotropic_voronoi' (the
standalone coverage_control Voronoi/Lloyd controller). Future methods register
here exactly as new trackers register in tracker.py.

The importance density phi for the Voronoi controller is built from the tracker
output (so the drones move over where the FILTER thinks targets are): each GM-PHD
belief component / target estimate contributes a Gaussian bump weighted by its
PHD weight. This adapter is the only place that knows about both the tracker and
the coverage_control package; coverage_control itself stays sim-agnostic.
"""
from __future__ import annotations

from dataclasses import replace
from typing import List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

from coverage_control import CoverageConfig, CoverageController
from infomax import InfomaxConfig, InfomaxController
from nonmyopic import MinimaxConfig, greedy_assignment
from .sensors import CameraSensorConfig, measurement_information, camera_measurement


@runtime_checkable
class Repositioner(Protocol):
    name: str

    def step(self, drones_xyz, estimates, belief, area_xy, dt) -> np.ndarray:
        ...


def _phi_sources(estimates, belief) -> Tuple[np.ndarray, np.ndarray]:
    """Centers (M, 2) + weights (M,) seeding phi, preferring the full belief mixture.

    ``belief`` is the GM-PHD mixture as [(pos(2,), cov(2,2), w), ...] (richer,
    continuous). ``estimates`` is the thresholded [TargetEstimate, ...] fallback.
    Either may be None/empty -> no centers -> phi is the uniform floor (cold start).
    """
    if belief:
        centers = np.array([np.asarray(pos, float)[:2] for (pos, _P, _w) in belief], float)
        weights = np.array([float(w) for (_pos, _P, w) in belief], float)
        return centers.reshape(-1, 2), weights.reshape(-1)
    if estimates:
        centers = np.array([np.asarray(e.position, float)[:2] for e in estimates], float)
        weights = np.array([float(e.w) for e in estimates], float)
        return centers.reshape(-1, 2), weights.reshape(-1)
    return np.zeros((0, 2)), np.zeros((0,))


def _target_priors(estimates, belief):
    """Per-target priors for the MI planner: means (M,4) [px,py,vx,vy], covs (M,4,4),
    weights (M,). Prefers ESTIMATES (full GM-PHD state incl. velocity); falls back to the
    2x2 belief (embedded with a diffuse velocity prior)."""
    if estimates:
        means = np.array([np.asarray(e.m, float).reshape(4) for e in estimates])
        covs = np.array([np.asarray(e.P, float).reshape(4, 4) for e in estimates])
        weights = np.array([float(e.w) for e in estimates])
        return means, covs, weights
    if belief:
        means, covs, weights = [], [], []
        for (pos, P, w) in belief:
            pos = np.asarray(pos, float).reshape(-1)
            means.append([pos[0], pos[1], 0.0, 0.0])
            C = np.eye(4) * 1e3                              # diffuse velocity prior
            C[:2, :2] = np.asarray(P, float).reshape(2, 2)
            covs.append(C)
            weights.append(float(w))
        return np.array(means), np.array(covs), np.array(weights)
    return np.zeros((0, 4)), np.zeros((0, 4, 4)), np.zeros((0,))


class InfomaxRepositioner:
    """Greedy / RSP mutual-information planner (Corah & Michael 2021) over GM-PHD targets."""
    name = "greedy_mi"

    def __init__(self, config: Optional[InfomaxConfig] = None,
                 sensor_cfg: Optional[CameraSensorConfig] = None):
        self.cfg = config or InfomaxConfig()
        self.sensor_cfg = sensor_cfg or CameraSensorConfig()
        self.controller = InfomaxController(self.cfg)
        self.last_actions = None
        self.last_mi_gain = None
        self.last_redundancy = None
        self.last_total_mi = 0.0
        self.last_candidates = None

    def step(self, drones_xyz, estimates, belief, area_xy, dt) -> np.ndarray:
        drones = np.asarray(drones_xyz, dtype=float).reshape(-1, 3)
        if drones.shape[0] == 0:
            return drones
        means, covs, weights = _target_priors(estimates, belief)
        # ungated (smooth) information so the greedy-MI planner has a gradient toward
        # targets even before they enter a footprint; actual sensing still respects the FOV.
        info_fn = lambda dxyz, txy: measurement_information(dxyz, txy, self.sensor_cfg, gate=False)
        out = self.controller.step(drones, means, covs, weights, area_xy, float(dt), info_fn)
        new = drones.copy()
        new[:, :2] = out["new_xy"]                          # z fixed (xy-only reposition)
        self.last_actions = out["chosen"]
        self.last_mi_gain = out["gains"]
        self.last_redundancy = out["redundancy"]
        self.last_total_mi = out["total_mi"]
        self.last_candidates = out["candidates"]
        return new


class MinimaxRepositioner:
    """Non-myopic minimax target-tracking repositioner (Zhang & Tokekar 2016).

    Each drone plans a closed-loop policy with a per-robot minimax tree (min over its
    controls, max over the worst-case measurement) that minimizes the worst-case trace of
    a target's Kalman posterior under state-dependent (camera) measurement noise. The joint
    multi-robot minimax is intractable, so drones are assigned to targets by SEQUENTIAL
    GREEDY assignment — OUR extension in the direction the paper's future-work section
    points (their prior IROS-2014 one-step greedy assignment), not an algorithm from the
    2016 paper itself. Heuristic (no submodular guarantee for minimax-trace)."""
    name = "minimax"

    def __init__(self, config: Optional[MinimaxConfig] = None,
                 sensor_cfg: Optional[CameraSensorConfig] = None):
        self.cfg = config or MinimaxConfig()
        self.sensor_cfg = sensor_cfg or CameraSensorConfig()
        self.last_minimax_value = None
        self.last_assignment = None
        self.last_offsets = None

    def step(self, drones_xyz, estimates, belief, area_xy, dt) -> np.ndarray:
        drones = np.asarray(drones_xyz, dtype=float).reshape(-1, 3)
        if drones.shape[0] == 0:
            return drones
        means, covs, weights = _target_priors(estimates, belief)
        if means.shape[0] == 0:
            self.last_minimax_value = np.zeros(drones.shape[0])
            self.last_assignment = [-1] * drones.shape[0]
            self.last_offsets = np.zeros((drones.shape[0], 2))
            return drones.copy()
        # State-dependent noise R_fn per drone from the camera model, UNGATED so it grows
        # smoothly with slant range and gives the minimax tree a gradient toward targets at
        # all distances (sensing in the sim still respects the FOV). Each drone keeps its own
        # altitude, hence its own R model.
        R_fns = [self._make_R_fn(float(drones[r, 2])) for r in range(drones.shape[0])]
        use_pruning = bool(self.cfg.use_alpha_pruning or self.cfg.use_redundancy_pruning)
        # PHD weights let the assignment prefer higher-mass targets (honored only when
        # cfg.weight_by_phd is set).
        pass_weights = bool(self.cfg.weight_by_phd)
        new_xy, offsets, assignment, values = greedy_assignment(
            drones[:, :2], means, covs, R_fns, self.cfg, float(dt),
            use_pruning=use_pruning,
            weights=(weights if pass_weights else None))
        new = drones.copy()
        new[:, :2] = new_xy                                 # z fixed (xy-only reposition)
        self.last_minimax_value = np.asarray(values, float)
        self.last_assignment = assignment
        self.last_offsets = offsets
        return new

    def _make_R_fn(self, drone_z: float):
        """R_fn(robot_xy(2,), target_xy(2,)) -> 2x2 camera measurement noise at this drone's
        altitude, UNGATED (smooth in slant range; differentiable everywhere)."""
        cfg = self.sensor_cfg
        return lambda robot_xy, target_xy: camera_measurement(
            [robot_xy[0], robot_xy[1], drone_z], target_xy, cfg)[1]


class VoronoiRepositioner:
    """Isotropic Voronoi based coverage (Cortes et al. 2004) over a tracker-driven phi."""
    name = "isotropic_voronoi"

    def __init__(self, config: Optional[CoverageConfig] = None):
        self.cfg = config or CoverageConfig()
        self.controller = CoverageController(self.cfg)
        # last-step diagnostics for visualization
        self.last_labels: Optional[np.ndarray] = None
        self.last_centroids: Optional[np.ndarray] = None
        self.last_phi: Optional[np.ndarray] = None
        self.last_cost: float = 0.0

    def step(self, drones_xyz, estimates, belief, area_xy, dt) -> np.ndarray:
        drones = np.asarray(drones_xyz, dtype=float).reshape(-1, 3)
        if drones.shape[0] == 0:
            return drones
        centers, weights = _phi_sources(estimates, belief)
        out = self.controller.step(drones[:, :2], centers, weights, area_xy, float(dt))
        new = drones.copy()
        new[:, :2] = out["new_xy"]
        # altitude optimization is off by default and has no Cortes-2D basis; z is
        # left fixed even when the flag is set until a defensible rule is added.
        self.last_labels = out["labels"]
        self.last_centroids = out["centroids"]
        self.last_phi = out["phi"]
        self.last_cost = out["cost"]
        return new


def _require_cfg_type(cfg, expected, method: str):
    """None -> method defaults; wrong TYPE -> loud TypeError. A silently-swallowed
    mismatched config would make a sweep report DEFAULT numbers under the caller's
    intended tuned config — the worst kind of benchmark bug."""
    if cfg is None:
        return None
    if not isinstance(cfg, expected):
        raise TypeError(
            f"repositioner {method!r} needs a {expected.__name__} config, got "
            f"{type(cfg).__name__}; pass the method's own config type (or None for defaults)")
    return cfg


def make_repositioner(name: Optional[str], cfg=None, sensor_cfg=None):
    """Toggle a repositioner by name. Returns None when repositioning is off.

    cfg is the method's config (CoverageConfig for voronoi, InfomaxConfig for greedy_mi/rsp,
    MinimaxConfig for minimax) or None for that method's defaults — a config object of the
    WRONG type raises TypeError rather than being silently replaced by defaults.
    sensor_cfg (CameraSensorConfig) is needed by the MI/minimax planners to build their
    (information / measurement-noise) sensor model.
    """
    key = (name or "none").lower()
    if key in ("none", "off", ""):
        return None
    if key in ("isotropic_voronoi", "voronoi", "lloyd"):
        return VoronoiRepositioner(_require_cfg_type(cfg, CoverageConfig, key))
    if key in ("greedy_mi", "rsp", "infomax"):
        icfg = _require_cfg_type(cfg, InfomaxConfig, key) or InfomaxConfig()
        # 'greedy_mi' forces sequential greedy; 'rsp' uses RSP with the config's n_d.
        icfg = replace(icfg, method="greedy" if key == "greedy_mi" else "rsp")
        return InfomaxRepositioner(icfg, sensor_cfg)
    if key == "minimax":
        mcfg = _require_cfg_type(cfg, MinimaxConfig, key) or MinimaxConfig()
        return MinimaxRepositioner(mcfg, sensor_cfg)
    if key == "anisotropic_voronoi":
        raise NotImplementedError("anisotropic Voronoi repositioner is not implemented yet")
    raise ValueError(
        f"unknown repositioner {name!r}; expected none|isotropic_voronoi|greedy_mi|rsp|minimax")
