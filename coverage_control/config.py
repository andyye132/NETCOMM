"""Configuration for Voronoi-based coverage control (Cortes et al. 2004).

One flat dataclass holds every knob (CleanRL style): the quadrature grid
resolution, the gradient-descent gain and integration cap, the rasterization of
the importance density phi from target estimates, and the (optional, off by
default) altitude-optimization switch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class CoverageConfig:
    # --- quadrature grid (the phi / Voronoi discretization of Q) ---
    grid_res: int = 64              # cells per axis; the quadrature knob (-> exact as it grows)

    # --- control law  u_i = -k (p_i - C_{V_i})  (Cortes eq. 10) ---
    gain: float = 1.0               # k: descent gain toward the cell centroid
    v_max: float = 10.0             # per-step speed cap (m/s); direction preserved when clipped
    step_mode: str = "control"      # "control" (speed-capped gradient descent) | "lloyd" (jump to centroid)

    # --- importance density phi rasterization (phi = floor + sum_m w_m N(.; c_m, sigma^2 I)) ---
    bump_sigma: float = 8.0         # m: spatial extent of each target's importance bump
    bump_floor: float = 1e-3        # uniform density floor so unexplained area is still covered

    # --- optional altitude optimization (NOT from Cortes 2D; off by default) ---
    optimize_altitude: bool = False
    alt_bounds: Tuple[float, float] = (2.0, 60.0)
