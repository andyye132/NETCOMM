"""Coverage-certainty heat map (the information field the search optimizes).

A grid over the area holds a per-cell certainty in [0, 1]: 1 = certain (red),
0 = uncertain (blue). Each step every cell DECAYS toward 0 (uncertainty grows
where we are not looking) and cells under a drone's camera footprint are RESTORED
toward 1, with restore strength proportional to the camera's sensing precision at
that cell (directly overhead / low covariance -> restores fast; the oblique
footprint edge restores slowly). This ties the "heat" to the GM-PHD-style
measurement uncertainty.

The scalar objective is the mean certainty (maximize): keep the whole map red by
covering it well. With stationary drones only the footprints stay red and the
rest fades to blue — which is exactly the signal that motivates moving the drones.

The per-step update is a genuine-JAX kernel: the per-drone footprint precision is
computed with ``jnp`` and ``vmap`` over drones and ``jit``-compiled; only the
host-side bookkeeping stays in Python.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .sensors import CameraSensorConfig


@jax.jit
def _coverage_step_kernel(certainty, gx, gy, drones_xyz, decay_rate, obs_gain,
                          fov_falloff, tan_fov, min_altitude, dt):
    """One coverage update as pure jnp (vmap over drones, jit-compiled).

    certainty/gx/gy : (ny, nx) float arrays.
    drones_xyz      : (N, 3) float array with N >= 1.
    Returns the new (ny, nx) certainty, clipped to [0, 1].
    """
    certainty = certainty * jnp.exp(-decay_rate * dt)

    def _drone_precision(d):
        # an airborne drone contributes an exponential nadir-to-edge falloff inside its
        # circular ground footprint; a below-min-altitude drone contributes nothing.
        rho = jnp.hypot(gx - d[0], gy - d[1])
        r_foot = d[2] * tan_fov
        in_fp = rho <= r_foot
        q = jnp.where(in_fp,
                      jnp.exp(-fov_falloff * rho / jnp.maximum(r_foot, 1e-6)),
                      0.0)
        airborne = d[2] >= min_altitude
        return jnp.where(airborne, q, 0.0)

    precision = jnp.sum(jax.vmap(_drone_precision)(drones_xyz), axis=0)
    certainty = 1.0 - (1.0 - certainty) * jnp.exp(-obs_gain * precision * dt)
    return jnp.clip(certainty, 0.0, 1.0)


class CoverageField:
    def __init__(self, area_xy, nx: int = 64, ny: int = 64,
                 decay_rate: float = 0.3, obs_gain: float = 2.0,
                 fov_falloff: float = 3.0):
        self.area = tuple(float(a) for a in area_xy)
        self.nx, self.ny = int(nx), int(ny)
        xmn, xmx, ymn, ymx = self.area
        xs = xmn + (np.arange(self.nx) + 0.5) / self.nx * (xmx - xmn)
        ys = ymn + (np.arange(self.ny) + 0.5) / self.ny * (ymx - ymn)
        # gx/gy/certainty stay NumPy host arrays (callers index and mutate them
        # directly); the per-step update is the jitted jnp kernel below.
        self.gx, self.gy = np.meshgrid(xs, ys)          # (ny, nx)
        self.certainty = np.zeros((self.ny, self.nx))   # start fully uncertain
        self.decay_rate = float(decay_rate)
        self.obs_gain = float(obs_gain)
        self.fov_falloff = float(fov_falloff)

    def step(self, drones_xyz, sensor_cfg: CameraSensorConfig, dt: float) -> None:
        drones = np.asarray(drones_xyz, dtype=float).reshape(-1, 3)
        # decay-only (no airborne sensors) is the closed form; skip the vmap kernel
        # (vmap needs >= 1 row) so the empty-drone case stays exact and cheap.
        if drones.shape[0] == 0:
            self.certainty = self.certainty * float(np.exp(-self.decay_rate * dt))
            return
        tan_fov = float(np.tan(sensor_cfg.half_fov_rad))
        out = _coverage_step_kernel(
            jnp.asarray(self.certainty), jnp.asarray(self.gx), jnp.asarray(self.gy),
            jnp.asarray(drones), self.decay_rate, self.obs_gain, self.fov_falloff,
            tan_fov, float(sensor_cfg.min_altitude), float(dt))
        self.certainty = np.asarray(out)

    def mean_certainty(self) -> float:
        return float(self.certainty.mean())

    def total_heat(self) -> float:
        return float(self.certainty.sum())

    def snapshot(self) -> np.ndarray:
        return self.certainty.copy()
