"""Candidate action sets for continuous drones (the discretization of U_i, Eq 7).

A single-step move is STAY plus K compass directions at reach r = v_max*dt (optionally
several radial speed rings). An "action" for the planner is a full l-step trajectory
(one move per horizon step), so the candidate set per drone is the set of reachable
l-step paths. The drone executes the FIRST move of the chosen trajectory (receding
horizon). All moves keep altitude fixed (xy-only repositioning).
"""
from __future__ import annotations

import itertools

import numpy as np


def single_step_offsets(cfg, dt: float) -> np.ndarray:
    """(n_single, 2) xy displacements: STAY + K directions x speed rings, magnitude <= v_max*dt."""
    r = cfg.v_max * dt
    offs = [np.zeros(2)] if cfg.include_stay else []
    rings = np.linspace(1.0 / cfg.n_speed_rings, 1.0, cfg.n_speed_rings)
    for ring in rings:
        for k in range(cfg.n_directions):
            ang = 2.0 * np.pi * k / cfg.n_directions
            offs.append(ring * r * np.array([np.cos(ang), np.sin(ang)]))
    return np.array(offs, dtype=float)


def drone_trajectories(drone_xy, cfg, dt: float, area_xy) -> np.ndarray:
    """All l-step trajectories from drone_xy: (n_traj, L, 2) absolute positions, clipped to area.

    n_traj = n_single ** horizon. Trajectory t's step k is the drone's position after k moves.
    """
    offs = single_step_offsets(cfg, dt)
    L = int(cfg.horizon)
    xmn, xmx, ymn, ymx = (float(a) for a in area_xy)
    trajs = []
    for combo in itertools.product(range(len(offs)), repeat=L):
        pos = np.array(drone_xy, dtype=float)
        path = []
        for step in combo:
            pos = pos + offs[step]
            pos = np.array([min(max(pos[0], xmn), xmx), min(max(pos[1], ymn), ymx)])
            path.append(pos.copy())
        trajs.append(path)
    return np.array(trajs, dtype=float)
