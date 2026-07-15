"""Configuration for the differentiable core. Plain Python floats so the values bake in as
constants when the rollout is jitted/differentiated (only drone positions are traced)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DiffSimConfig:
    dt: float = 1.0                          # step time (s)
    q: float = 0.05                          # CV process-noise intensity
    h: float = 30.0                          # drone altitude (m); footprint = h*tan(half_fov)
    half_fov_rad: float = float(np.deg2rad(35.0))
    focal_px: float = 600.0                  # camera focal length (px)
    sigma_px: float = 1.5                    # pixel-localization std
    sigma_alt: float = 0.5                   # altitude/pose std (m)
    p_detect: float = 0.95                   # nominal detection prob
    k_vis: float = 0.2                       # softness of the footprint sigmoid (1/m). The hard
                                             # FOV gate is replaced by sigmoid(k_vis*(foot-rho))
                                             # so visibility — and the loss — is differentiable.
    p0_pos: float = 100.0                    # initial position variance (m^2)
    p0_vel: float = 25.0                     # initial velocity variance ((m/s)^2)
    area: tuple = (0.0, 300.0, 0.0, 300.0)   # xmn, xmx, ymn, ymx (for clipping in optimization)
