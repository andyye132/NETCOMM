"""Object path / motion presets for pedestrian-tracking scenarios.

Each generator returns an object-trajectory array of shape (n_steps, N, 2).

Two families:
  - random_walk : semi-random correlated walk. Objects may leave the area and are
    immediately replaced by a new object entering from a random edge, so a roughly
    constant N stays on screen. Optionally each object has a different speed.
  - pattern     : N objects follow the SAME parametric path (figure-8 / circle /
    square / triangle), phase-offset so they are spread around it.
"""
from __future__ import annotations

import numpy as np

PATTERNS = ("figure8", "circle", "square", "triangle")


def _spawn_on_edge(area, rng):
    """A fresh entry position on a random edge, heading inward."""
    xmn, xmx, ymn, ymx = area
    edge = int(rng.integers(0, 4))
    spread = np.pi / 3
    if edge == 0:                                  # left edge, heading right
        p = [xmn, rng.uniform(ymn, ymx)]; h = rng.uniform(-spread, spread)
    elif edge == 1:                                # right edge, heading left
        p = [xmx, rng.uniform(ymn, ymx)]; h = np.pi + rng.uniform(-spread, spread)
    elif edge == 2:                                # bottom edge, heading up
        p = [rng.uniform(xmn, xmx), ymn]; h = np.pi / 2 + rng.uniform(-spread, spread)
    else:                                          # top edge, heading down
        p = [rng.uniform(xmn, xmx), ymx]; h = -np.pi / 2 + rng.uniform(-spread, spread)
    return np.array(p, dtype=float), float(h)


def random_walk_trajectories(n, n_steps, dt, area, speed, speed_varies=False,
                             turn=0.3, seed=0, return_ids=False):
    """With ``return_ids`` also returns an (n_steps, n) int array of IDENTITY labels:
    a respawned slot gets a fresh globally-unique id, because the respawn is a new
    physical object entering (the old one left) — the position teleports, so any
    consumer that treats an array row as one continuous track (PCRLB, OSPA^2, MOT)
    must split tracks where the id changes."""
    rng = np.random.default_rng(seed)
    xmn, xmx, ymn, ymx = area
    pos = rng.uniform([xmn, ymn], [xmx, ymx], size=(n, 2))
    heading = rng.uniform(0.0, 2 * np.pi, size=n)
    spd = rng.uniform(0.4 * speed, speed, size=n) if speed_varies else np.full(n, float(speed))
    traj = np.zeros((n_steps, n, 2))
    ids = np.zeros((n_steps, n), dtype=int)
    cur_ids = np.arange(n)
    next_id = n
    for t in range(n_steps):
        heading = heading + rng.normal(0.0, turn, size=n)     # correlated drift
        vel = np.stack([np.cos(heading), np.sin(heading)], axis=1) * spd[:, None]
        pos = pos + vel * dt
        for i in range(n):                                    # respawn anyone who left
            if not (xmn <= pos[i, 0] <= xmx and ymn <= pos[i, 1] <= ymx):
                pos[i], heading[i] = _spawn_on_edge(area, rng)
                cur_ids[i] = next_id                          # new object, new identity
                next_id += 1
        traj[t] = pos
        ids[t] = cur_ids
    return (traj, ids) if return_ids else traj


def _pattern_point(pattern, cx, cy, R, u):
    """Point at normalized phase u in [0, 1) on the given closed path."""
    if pattern == "circle":
        a = 2 * np.pi * u
        return cx + R * np.cos(a), cy + R * np.sin(a)
    if pattern == "figure8":                                  # Gerono lemniscate
        a = 2 * np.pi * u
        return cx + R * np.cos(a), cy + 0.5 * R * np.sin(2 * a)
    if pattern == "square":
        corners = [(-R, -R), (R, -R), (R, R), (-R, R)]
        seg = int(u * 4) % 4
        loc = u * 4 - int(u * 4)
        x0, y0 = corners[seg]; x1, y1 = corners[(seg + 1) % 4]
        return cx + x0 + (x1 - x0) * loc, cy + y0 + (y1 - y0) * loc
    if pattern == "triangle":
        angs = [np.pi / 2, np.pi / 2 + 2 * np.pi / 3, np.pi / 2 + 4 * np.pi / 3]
        corners = [(R * np.cos(a), R * np.sin(a)) for a in angs]
        seg = int(u * 3) % 3
        loc = u * 3 - int(u * 3)
        x0, y0 = corners[seg]; x1, y1 = corners[(seg + 1) % 3]
        return cx + x0 + (x1 - x0) * loc, cy + y0 + (y1 - y0) * loc
    raise ValueError(f"unknown pattern {pattern!r}")


def _pattern_length(pattern, R):
    return {"circle": 2 * np.pi * R, "figure8": 6.1 * R,
            "square": 8.0 * R, "triangle": 5.2 * R}[pattern]


def pattern_trajectories(n, n_steps, dt, area, pattern, speed, seed=0, return_ids=False):
    xmn, xmx, ymn, ymx = area
    cx, cy = (xmn + xmx) / 2.0, (ymn + ymx) / 2.0
    R = 0.34 * min(xmx - xmn, ymx - ymn)
    period = max(_pattern_length(pattern, R) / max(speed, 1e-3), 1e-3)   # seconds/loop
    traj = np.zeros((n_steps, n, 2))
    for i in range(n):
        phase0 = i / max(n, 1)                                # spread objects around the loop
        for t in range(n_steps):
            u = ((t * dt) / period + phase0) % 1.0
            traj[t, i] = _pattern_point(pattern, cx, cy, R, u)
    if return_ids:                                            # patterns never respawn
        return traj, np.tile(np.arange(n), (n_steps, 1))
    return traj


def preset_trajectories(preset, n, n_steps, dt, area, speed, seed=0, return_ids=False):
    """Dispatch a named preset to its trajectory generator.

    ``return_ids`` additionally returns the (n_steps, n) identity-label array (see
    random_walk_trajectories): identities are stable for pattern presets and split
    at respawn teleports for the random-walk presets."""
    if preset == "random_walk":
        return random_walk_trajectories(n, n_steps, dt, area, speed, speed_varies=False,
                                        seed=seed, return_ids=return_ids)
    if preset == "random_walk_varied":
        return random_walk_trajectories(n, n_steps, dt, area, speed, speed_varies=True,
                                        seed=seed, return_ids=return_ids)
    if preset in PATTERNS:
        return pattern_trajectories(n, n_steps, dt, area, preset, speed, seed=seed,
                                    return_ids=return_ids)
    raise ValueError(f"unknown preset {preset!r}")
