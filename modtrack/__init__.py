"""ModTrack: sensor-agnostic multi-view tracking core.

Analytical implementation of the ModTrack pipeline (Iyer, Roberts, Ayanian).
The package is intentionally decoupled from the NETCOMM sim: it imports nothing
from ``netcomm`` and operates purely on calibrated BEV position-covariance pairs
``(z, R)``. This is the sensor-agnostic interface that lets the same tracker core
run behind any front-end (cameras, radar, or — here — drone sensors).

Stages (paper Section 3):
  - uncertainty.py : Jacobian covariance propagation  (front-end primitive, Stage 2)
  - clustering.py  : chi^2 graph clustering            (Stage 3)   [TODO]
  - fusion.py      : precision-weighted fusion         (Stage 4)
  - gmphd.py       : identity-informed GM-PHD filter   (Stage 5)   [TODO]

Everything is dimension-general (2D BEV now, 3D later) and numpy-first so the
unit-test suite runs in milliseconds without JAX/CUDA.
"""
from . import linalg
from .types import Detection, FusedDetection
from .uncertainty import (
    propagate_covariance,
    propagate_covariance_fn,
    finite_diff_jacobian,
    polar_to_cartesian,
    polar_to_cartesian_jacobian,
)
from .fusion import precision_weighted_fuse
from .clustering import (
    chi2_distance,
    same_target_probability,
    chi2_gate_value,
    cluster_detections,
)

__all__ = [
    "linalg",
    "Detection",
    "FusedDetection",
    "propagate_covariance",
    "propagate_covariance_fn",
    "finite_diff_jacobian",
    "polar_to_cartesian",
    "polar_to_cartesian_jacobian",
    "precision_weighted_fuse",
    "chi2_distance",
    "same_target_probability",
    "chi2_gate_value",
    "cluster_detections",
]
