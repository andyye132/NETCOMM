"""Data types for the GM-PHD filter.

All array fields are coerced to float ndarrays and copied on construction.
``eq=False`` keeps the auto-generated dataclass __eq__/__hash__ from blowing up
on ndarray fields (object identity is used instead).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(eq=False)
class Detection:
    """A position measurement with its own (calibrated) covariance.

    z : (2,) measured BEV position
    R : (2, 2) measurement covariance — per-measurement (uncertainty-aware)
    """
    z: np.ndarray
    R: np.ndarray

    def __post_init__(self):
        self.z = np.array(self.z, dtype=float).reshape(-1)
        self.R = np.array(self.R, dtype=float)
        if self.z.shape[0] != 2 or self.R.shape != (2, 2):
            raise ValueError("Detection expects z of shape (2,) and R of shape (2, 2)")


@dataclass(eq=False)
class GaussianComponent:
    """One Gaussian term of the PHD intensity: weight w, mean m, covariance P."""
    w: float
    m: np.ndarray          # (4,) state [px, py, vx, vy]
    P: np.ndarray          # (4, 4) state covariance

    def __post_init__(self):
        self.w = float(self.w)
        self.m = np.array(self.m, dtype=float).reshape(-1)
        self.P = np.array(self.P, dtype=float)


@dataclass(eq=False)
class TargetEstimate:
    """An extracted target: state mean, covariance, and PHD weight."""
    m: np.ndarray          # (4,)
    P: np.ndarray          # (4, 4)
    w: float

    def __post_init__(self):
        self.m = np.array(self.m, dtype=float).reshape(-1)
        self.P = np.array(self.P, dtype=float)
        self.w = float(self.w)

    @property
    def position(self) -> np.ndarray:
        return self.m[:2]

    @property
    def velocity(self) -> np.ndarray:
        return self.m[2:]

    @property
    def position_covariance(self) -> np.ndarray:
        return self.P[:2, :2]
