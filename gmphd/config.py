"""Configuration for the GM-PHD filter (Vo & Ma 2006 parameters)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class GMPHDConfig:
    # --- motion model ---
    dt: float = 1.0                 # time step between frames
    q: float = 0.01                 # CV process-noise intensity

    # --- detection / survival ---
    p_survival: float = 0.99        # p_S: target survival probability per step
    p_detect: float = 0.95          # p_D: probability a present target is detected

    # --- clutter ---
    clutter_intensity: float = 1e-5  # kappa(z): uniform clutter intensity (lambda_c / area)

    # --- birth model ---------------------------------------------------------
    # birth_mode selects how new tracks are spawned each predict step:
    #   'measurement' (DEFAULT) -- measurement-driven birth: one birth Gaussian
    #       per detection, centered at the measurement with its covariance and a
    #       broad velocity prior. This is the live-sim behavior and is unchanged.
    #   'intensity'             -- Vo & Ma 2006 fixed spontaneous-birth GM intensity
    #       gamma_k (paper Eq. 17): a configurable, detection-independent list of
    #       birth Gaussians (weight, mean, cov) added every predict step.
    birth_mode: str = "measurement"

    # measurement-driven ('measurement') birth params.
    # why: birth weight is kept small so a birth seeded at an ALREADY-explained
    # measurement contributes negligibly to an established track (no velocity
    # corruption on merge). When a measurement is unexplained, the birth is its
    # only explainer and still gains near-full posterior weight, so genuinely new
    # targets still appear within one frame.
    birth_weight: float = 1e-3      # weight of a birth component per detection
    birth_vel_var: float = 25.0     # initial velocity variance for born targets

    # spontaneous-birth GM intensity ('intensity' mode): list of (weight, mean,
    # cov) where mean is length-4 [px,py,vx,vy] and cov is 4x4 (paper Eq. 17).
    # Empty by default; populate to enable fixed birth locations.
    birth_intensity: List[Tuple[float, "np.ndarray", "np.ndarray"]] = field(
        default_factory=list)

    # --- component management ---
    prune_threshold: float = 1e-4   # T: drop components below this weight
    merge_threshold: float = 4.0    # U: Mahalanobis^2 distance for merging components
    max_components: int = 100       # J_max: cap on retained components

    # --- state extraction ---
    extract_threshold: float = 0.5  # report components with weight above this
