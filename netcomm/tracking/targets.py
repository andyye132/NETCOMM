"""Ground targets to be tracked — a flexible, configurable 2D population.

Targets are distinct from the network drones: 2D ground objects (z=0) that move
with constant velocity and reflect off the area bounds. This is the "create
objects to be tracked" scenario knob.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def _advance_kernel(positions, velocities, dt, lo, hi):
    """One constant-velocity step with reflective bounds, vectorized in jnp.

    positions/velocities : (M, 2). lo/hi : (2,) per-axis bounds. Returns the
    updated (positions, velocities). Reflection mirrors the overshoot and flips the
    corresponding velocity component (folded once, matching the scalar update)."""
    new = positions + dt * velocities
    below = new < lo
    above = new > hi
    new = jnp.where(below, lo + (lo - new), new)
    new = jnp.where(above, hi - (new - hi), new)
    velocities = jnp.where(jnp.logical_or(below, above), -velocities, velocities)
    return new, velocities


@dataclass
class TargetConfig:
    n_targets: int = 3
    area_xy: Tuple[float, float, float, float] = (0.0, 100.0, 0.0, 100.0)
    v_max: float = 5.0                      # max per-axis speed (m/s)


class TargetPopulation:
    """M ground targets with constant-velocity motion and reflective bounds."""

    def __init__(self, positions, velocities, area_xy):
        self.positions = np.array(positions, dtype=float).reshape(-1, 2)
        self.velocities = np.array(velocities, dtype=float).reshape(-1, 2)
        self.area_xy = tuple(float(a) for a in area_xy)

    @classmethod
    def spawn(cls, cfg: TargetConfig, rng: np.random.Generator) -> "TargetPopulation":
        xmn, xmx, ymn, ymx = cfg.area_xy
        pos = np.stack([rng.uniform(xmn, xmx, cfg.n_targets),
                        rng.uniform(ymn, ymx, cfg.n_targets)], axis=1)
        vel = rng.uniform(-cfg.v_max, cfg.v_max, size=(cfg.n_targets, 2))
        return cls(pos, vel, cfg.area_xy)

    @property
    def n(self) -> int:
        return int(self.positions.shape[0])

    def advance(self, dt: float) -> None:
        """One constant-velocity step with reflective boundaries (jitted jnp kernel)."""
        if self.positions.shape[0] == 0:
            return
        lo = jnp.array([self.area_xy[0], self.area_xy[2]])
        hi = jnp.array([self.area_xy[1], self.area_xy[3]])
        new, vel = _advance_kernel(jnp.asarray(self.positions), jnp.asarray(self.velocities),
                                   float(dt), lo, hi)
        self.positions = np.asarray(new)
        self.velocities = np.asarray(vel)
