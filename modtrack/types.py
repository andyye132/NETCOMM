"""Core data types for the ModTrack pipeline.

A ``Detection`` is the sensor-agnostic unit of input: a calibrated BEV
position ``z`` with covariance ``R`` (the (z, R) interface), optionally carrying
an appearance feature and detection confidence. A ``FusedDetection`` is the
output of multi-view fusion (Stage 4).

Both are dimension-general: ``z`` may be 2D (BEV now) or 3D (later); ``R`` is the
matching ``(d, d)`` covariance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from . import linalg


@dataclass(frozen=True, eq=False)
class Detection:
    """A calibrated BEV position-covariance detection from one sensor."""

    z: np.ndarray                       # (d,) BEV position
    R: np.ndarray                       # (d, d) covariance, symmetric PSD
    sensor_id: int = -1                 # which sensor produced it
    conf: float = 1.0                   # detection confidence in [0, 1]
    class_id: int = -1                  # semantic class (-1 = unknown)
    feature: Optional[np.ndarray] = None  # (F,) appearance feature, optional

    def __post_init__(self):
        z = np.array(self.z, dtype=float).reshape(-1)        # copy: Detection is a value type
        R = linalg.symmetrize(np.asarray(self.R, dtype=float))
        d = z.shape[0]
        if R.shape != (d, d):
            raise ValueError(f"R must be ({d}, {d}) to match z of dim {d}; got {R.shape}")
        # frozen dataclass: bypass the immutability guard to store coerced arrays
        object.__setattr__(self, "z", z)
        object.__setattr__(self, "R", R)
        if self.feature is not None:
            object.__setattr__(self, "feature", np.array(self.feature, dtype=float).reshape(-1))

    @property
    def dim(self) -> int:
        return int(self.z.shape[0])


@dataclass(frozen=True, eq=False)
class FusedDetection:
    """Fused multi-view estimate (z_hat, R_hat) with pooled appearance feature."""

    z: np.ndarray
    R: np.ndarray
    feature: Optional[np.ndarray] = None
    conf: float = 1.0
    members: Sequence[int] = ()         # sensor_ids that contributed

    def __post_init__(self):
        object.__setattr__(self, "z", np.array(self.z, dtype=float).reshape(-1))
        object.__setattr__(self, "R", linalg.symmetrize(np.asarray(self.R, dtype=float)))
        object.__setattr__(self, "members", tuple(self.members))
        if self.feature is not None:
            object.__setattr__(self, "feature", np.array(self.feature, dtype=float).reshape(-1))

    @property
    def dim(self) -> int:
        return int(self.z.shape[0])
